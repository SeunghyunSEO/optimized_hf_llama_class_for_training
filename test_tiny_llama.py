import random
import numpy as np
import torch

from transformers import LlamaConfig, AutoTokenizer
from src.models.modeling_llama import LlamaForCausalLM
from src.models.optimized_modeling_llama import OptimizedLlamaForCausalLM

from pdb import set_trace as Tra


def print_model_stat(config, model):
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'''
    config     : {config}
    {'====='*15}
    model      : {model}
    {'====='*15}
    num_params : {num_params}
    num_trainable_params : {num_trainable_params}
    {'====='*15}
    getattr(config, "_attn_implementation", None) : {getattr(config, "_attn_implementation", None)}
    {'====='*15}
    ''')

def _reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dummy_input(
    vocab_size,
    device,
    seed=1234,
    batch_size=1,
    seq_len=4096,
):
    _reset_seeds(seed)
    pad_token_id = -100
    input_ids = torch.randint(vocab_size, (batch_size, seq_len))
    labels = torch.cat((input_ids[:, 1:], torch.full((batch_size, 1), pad_token_id)),1)
    attention_mask = torch.full((batch_size, seq_len), 1)
    return {
        'input_ids': input_ids.to(device),
        'labels': labels.to(device),
        'attention_mask': attention_mask.to(device),
    }

def get_output(
    model,
    inputs,
    seed,
):
    _reset_seeds(seed)
    outputs = model(**inputs)
    loss = outputs.loss
    loss.mean().backward()
    weights_grads = {n : p.grad for n, p in model.named_parameters()}
    return loss, weights_grads

device = "cuda:0"

# dtype = torch.float16
# dtype = torch.bfloat16
dtype = torch.float32

seed = 1234
batch_size = 2
seq_len = 155

vocab_size = 128
hidden_size = 256
intermediate_size = 128
num_hidden_layers = 2
num_attention_heads = 4
num_key_value_heads = 2
hidden_act = "silu"

small_llama_config = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_key_value_heads,
    hidden_act=hidden_act,
)
small_llama_config._attn_implementation = 'sdpa'

_reset_seeds(seed)
optimized_model = OptimizedLlamaForCausalLM._from_config(small_llama_config).to(dtype=dtype, device=device).train()
print_model_stat(optimized_model.config, optimized_model)
_reset_seeds(seed)
vanilla_model = LlamaForCausalLM._from_config(small_llama_config).to(dtype=dtype, device=device).train()
print_model_stat(vanilla_model.config, vanilla_model)
for (n, p), (n_, p_) in zip(vanilla_model.named_parameters(), optimized_model.named_parameters()):
    assert (n == n_) and ((p==p_).sum() == p.numel())

inputs = get_dummy_input(vocab_size, device, seed, batch_size, seq_len)
optimized_model_outputs = get_output(optimized_model, inputs, seed)
inputs = get_dummy_input(vocab_size, device, seed, batch_size, seq_len)
vanilla_model_outputs = get_output(vanilla_model, inputs, seed)
assert (len(vanilla_model_outputs[1].keys()) == len(optimized_model_outputs[1].keys()))

print(f'''
< loss >
vanilla_model_outputs[0]   : {vanilla_model_outputs[0]:.60f}
optimized_model_outputs[0] : {optimized_model_outputs[0]:.60f}
allclose : {torch.allclose(vanilla_model_outputs[0], optimized_model_outputs[0])}
''')

for key, grad, grad_ in zip(vanilla_model_outputs[1].keys(), vanilla_model_outputs[1].values(), optimized_model_outputs[1].values()):
    print(f'{key}: {torch.allclose(grad, grad_)}')