'''
copied from https://github.com/huggingface/transformers/blob/v4.39-release/src/transformers/models/llama/modeling_llama.py
4.39.0 version
'''

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ..modules.rms_layernorm import create_norm, FusedRMSNorm
from ..modules.rope_embedding import fast_rope_embedding
from ..modules.fused_cross_entropy import FusedProjectionPlusCrossEntropyLoss, fused_cross_entropy

logger = logging.get_logger(__name__)

import sys
import packaging
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

from packaging.version import parse

TORCH_MINIMUM_VERSION = parse("2.2.0")
XFORMERS_MINIMUM_VERSION = parse("0.0.24")
torch_version = parse(importlib_metadata.version("torch"))
xformers_version = parse(importlib_metadata.version("xformers"))

_is_torch_greather_than_2_2_0 = (
    torch_version.major,
    torch_version.minor,
    torch_version.release[-1],
) >= (
    TORCH_MINIMUM_VERSION.major,
    TORCH_MINIMUM_VERSION.minor,
    TORCH_MINIMUM_VERSION.release[-1],
)

_is_xformers_greather_than_0_0_24 = (
    xformers_version.major,
    xformers_version.minor,
    xformers_version.release[-1],
) >= (
    XFORMERS_MINIMUM_VERSION.major,
    XFORMERS_MINIMUM_VERSION.minor,
    XFORMERS_MINIMUM_VERSION.release[-1],
)

# assert _is_torch_greather_than_2_2_0 and _is_xformers_greather_than_0_0_24

_torch_sdpa_OPS = {
    'base': {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    'math': {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    'flash': {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    'mem_efficient': {"enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}

import xformers.ops as xops
_xformers_OPS = {
    'base': None, 
    'cutlass': (xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp),
    'flash': (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
}
_xformers_OP = _xformers_OPS['base']

# USE_XFORMERS = True
USE_XFORMERS = False

import gc
def release_memory():
    torch.cuda.empty_cache()
    gc.collect()

USE_BASELINE = {
    'rope' : False,
    'layernorm' : False,
    'attn' : False,
    'ce_loss' : False,
}

UPCASTING_LAST_HIDDEN = False
# UPCASTING_LAST_HIDDEN = True
N_LOOP_ITERS = 8
IGNORE_INDEX = -100
REDUCTION = 'mean'

from pdb import set_trace as Tra


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.45"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        ## 4.44~ patch
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) #, B, T, C
        key_states = self.k_proj(hidden_states) # B, T, C
        value_states = self.v_proj(hidden_states) # B, T, C

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # B, num_heads, T, head_dim 
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # B, num_kv_heads, T, head_dim 
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # B, num_kv_heads, T, head_dim 

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        if USE_BASELINE['rope']:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, key_states = fast_rope_embedding(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if USE_BASELINE['attn']:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
            else:
                attn_output = self.o_proj(attn_output)
        else:
            causal_mask = attention_mask
            # if attention_mask is not None and cache_position is not None:
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

            if USE_XFORMERS:
                query_states = query_states.transpose(1,2).contiguous()
                key_states = key_states.transpose(1,2).contiguous()
                value_states = value_states.transpose(1,2).contiguous()

                attn_output = xops.memory_efficient_attention(
                    query_states, # B, T, H, C
                    key_states, # B, T, H, C
                    value_states, # B, T, H, C
                    attn_bias = attention_mask if attention_mask is not None else xops.LowerTriangularMask(), # should be B, H, T, T
                    p=self.attention_dropout if self.training else 0.0,
                    op=_xformers_OP,
                ) # B, T, H, C
                attn_output = attn_output.contiguous() # B, T, H, C

            else:
                # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
                # Reference: https://github.com/pytorch/pytorch/issues/112577.
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

                # https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, # B, H, T, C
                    key_states, # B, H, T, C
                    value_states, # B, H, T, C
                    attn_mask=causal_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=True if causal_mask is None and q_len > 1 else False,
                ) # B, H, T, C
                attn_output = attn_output.transpose(1, 2).contiguous() # B, T, H, C

            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

        return attn_output, None, None


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        if USE_BASELINE['layernorm']:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = create_norm("fused_rmsnorm", config.hidden_size, config.rms_norm_eps)
            self.post_attention_layernorm = create_norm("fused_rmsnorm", config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):

        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if USE_BASELINE['layernorm']:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = create_norm("fused_rmsnorm", config.hidden_size, config.rms_norm_eps)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.use_deepspeed_cpu_gradient_checkpoint = False

        # Initialize weights and apply final processing
        self.post_init()

    def set_deepspeed_cpu_gradient_checkpoint(self, num_checkpoints):
        from deepspeed.runtime.activation_checkpointing import checkpointing
        def checkpoint(function, *args):
            "custom deepspeed checkpoint method because every HF tfm block should return tuple"
            all_outputs = []
            checkpointing.CheckpointFunction.apply(function, all_outputs, *args)
            return tuple(all_outputs)

        # https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/activation_checkpointing/checkpointing.py#L1070
        self.num_checkpoints = num_checkpoints
        checkpointing.configure(
            mpu_=None,
            deepspeed_config=None,
            partition_activations=False,
            contiguous_checkpointing=False,
            num_checkpoints=num_checkpoints, # only use these for now
            checkpoint_in_cpu=True, # only use these for now
            synchronize=False,
            profile=False,
        )
        self._gradient_checkpointing_func = checkpoint
        self.use_deepspeed_cpu_gradient_checkpoint = True
        self.gradient_checkpointing = True

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        assert (
            not output_attentions
            and not use_cache
            and not cache_position
            and not past_key_values
        ), "This class automatically use efficient SDPA not supporting producing attention wegiths"

        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        logger.warning_once("because this model class is only for training, we remove attention_mask and use 'is_causal' option, don't use this for model.generate()")
        # causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        del attention_mask
        causal_mask = None 
        release_memory()

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        ###################################################################
        ####### customized activation checkpointing with deepspeed ########
        ###################################################################
        if self.gradient_checkpointing and self.training and self.use_deepspeed_cpu_gradient_checkpoint:
            assert not (past_key_values or output_attentions or use_cache)

            def create_custom_forward(start, end):
                def custom_forward(*inputs):
                    output = []
                    layers_ = self.layers[start:end]
                    x_ = inputs[0] # hidden_states
                    for i, layer in enumerate(layers_):
                        x_ = layer(
                            x_, 
                            causal_mask,
                            position_ids,
                            position_embeddings,
                        )[0] # tuple to tensor
                        output.append(x_)
                    return output
                return custom_forward

            l = 0
            total_num_layers = len(self.layers)
            chunk_length = self.num_checkpoints ## how many checkpoint do you want?

            # appending first inp
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            while l < total_num_layers:
                hidden_states_list = self._gradient_checkpointing_func(
                    create_custom_forward(l, l+chunk_length),
                    hidden_states,
                )
                l += chunk_length

                if output_hidden_states:
                    # appending layerwise outputs
                    for hidden_states in hidden_states_list:
                        all_hidden_states += (hidden_states,)
                hidden_states = hidden_states_list[-1]

            if output_hidden_states:
                # input + layerwise outputs (execpt last hidden) should be the number of layers
                all_hidden_states = all_hidden_states[:-1]
                assert len(all_hidden_states) == len(self.layers), print(f"len(all_hidden_states) ({len(all_hidden_states)}) != len(self.layers) ({len(self.layers)})")
                
        else:
            for idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        raise NotImplementedError("This model class supports only autoregressive training")


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        debug_info = "====="*15 + "\n< Use Original Impl >\n"
        for k, v in USE_BASELINE.items():
            debug_info += f'{k}: {v}\n'
        debug_info += "====="*15
        print(debug_info)


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # print(f'hidden_states.size(): {hidden_states.size()}')
        if self.config.pretraining_tp > 1:
            raise NotImplementedError
        
        assert labels is not None, "This model class does not support producing logits alone"

        if USE_BASELINE['ce_loss']:
            logits = self.lm_head(hidden_states)
            logits = logits.float() # upcasting

            logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            labels = labels[:, 1:].contiguous().view(-1).to(logits.device)
            loss = CrossEntropyLoss()(logits, labels)
            # release_memory()
        else:
            B, T, C = hidden_states.size()
            if UPCASTING_LAST_HIDDEN:
                hidden_states = hidden_states.float() # upcasting

            hidden_states = hidden_states[:, :-1, :].contiguous().view(-1, C)
            labels = labels[:, 1:].contiguous().view(-1).to(hidden_states.device)
            assert hidden_states.size(0) == labels.size(0)
            assert hidden_states.ndim == 2 and labels.ndim == 1

            loss = fused_cross_entropy(
                hidden_states, 
                self.lm_head.weight, 
                labels, 
                n_loop_iters=N_LOOP_ITERS,
                ignore_index=IGNORE_INDEX,
                reduction=REDUCTION,
            )
            logits = None
            # release_memory()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        raise NotImplementedError("This model class is for training")

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        raise NotImplementedError("This model class is for training")