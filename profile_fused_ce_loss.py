import os
import argparse
from datetime import timedelta

import random
import numpy as np
from tqdm import tqdm

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from transformers import LlamaConfig, AutoTokenizer
from src.models.modeling_llama import LlamaForCausalLM
from src.models.optimized_modeling_llama import OptimizedLlamaForCausalLM
from src.modules.unsloth.models import FastLanguageModel

from src.utils import ContextManagers, get_torch_profiler

from pdb import set_trace as Tra


CLASS_SET = {
    'auto': {
        'config': LlamaConfig,
        'model': LlamaForCausalLM,
        'tokenizer': AutoTokenizer,
    },
    'optimized': {
        'config': LlamaConfig,
        'model': OptimizedLlamaForCausalLM,
        'tokenizer': AutoTokenizer,
    },
    'unsloth': {
        'config': LlamaConfig,
        'model': FastLanguageModel,
        'tokenizer': AutoTokenizer,
    },
}

DTYPE_SET = {
    "fp32" : torch.float32,
    "fp16" : torch.float16,
    "bf16" : torch.bfloat16,
}

def get_accelerator():
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=int(os.getenv("DDP_TIMEOUT", 1800))))
    accelerator = Accelerator(
        log_with=["mlflow"],
        project_dir="./",
        kwargs_handlers=[kwargs]
    )
    return accelerator

def get_model(
    model_path: str,
    class_type: str,
    dtype: str,
    use_deepspeed_activation_checkpointing: bool = False,
    num_checkpoints: int = 1,
):
    torch_dtype = DTYPE_SET[dtype]
    class_set = CLASS_SET[class_type]

    config_class = class_set['config']
    model_class = class_set['model']
    tokenizer_class = class_set['tokenizer']

    ## configuration
    config = config_class.from_pretrained(model_path)

    ## model
    if class_type != 'unsloth':
        args = {
            'pretrained_model_name_or_path': model_path,
            'config': config,
            'torch_dtype': torch_dtype,
        }
        model = model_class.from_pretrained(**args)
        if use_deepspeed_activation_checkpointing:
            model.model.set_deepspeed_cpu_gradient_checkpoint(num_checkpoints)
            print(f'''
            {model.model.use_deepspeed_cpu_gradient_checkpoint}
            {model.model._gradient_checkpointing_func}
            {model.model.num_checkpoints}
            ''')
    else:
        args = {
            'model_name': model_path,
            'max_seq_length': config.max_position_embeddings,
            'load_in_4bit': False,
            'dtype': torch_dtype,
            'device_map': None,
            'low_cpu_mem_usage': False,
        }
        model = model_class.from_pretrained(**args)

    ## tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_path)

    print(f'''
    config: {config}
    {'====='*15}
    model : {model}
    {'====='*15}
    ''')
    return model, tokenizer, config

def _get_deepspeed_wrapped_model(model, accelerator):
    ## this is dummy dataloader and scheduler
    from accelerate.test_utils.training import RegressionDataset
    from torch.utils.data import DataLoader
    from transformers import get_scheduler
    train_set = RegressionDataset(length=80)
    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=0.001)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=1000,
    )
    model, optimizer, _, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    return model, optimizer, lr_scheduler

def _reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dummy_input(
    tokenizer,
    device,
    seed=1234,
    batch_size=1,
    seq_len=4096,
):
    _reset_seeds(seed)
    input_ids = torch.randint(len(tokenizer), (batch_size, seq_len))
    labels = torch.cat((input_ids[:, 1:], torch.full((batch_size, 1), tokenizer.eos_token_id)),1)
    attention_mask = torch.full((batch_size, seq_len), 1)
    return {
        'input_ids': input_ids.to(device),
        'labels': labels.to(device),
        'attention_mask': attention_mask.to(device),
    }

def main(args):

    accelerator = get_accelerator()
    rank = accelerator.process_index
    device = accelerator.device
    world_size = torch.distributed.get_world_size()

    # hard coding for 
    ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
    # ds_config["train_batch_size"] = 1 * world_size
    # ds_config["train_micro_batch_size_per_gpu"] = 1
    # ds_config["gradient_accumulation_steps"] = 1
    ds_config["wall_clock_breakdown"] = True # Enable timing of the latency of forward/backward/update training phases
    ds_config["dump_state"] = False # Print out state information of DeepSpeed object after initialization
    if args.dtype == "bf16":
        ds_config["bf16"] = {'enabled': True}
    elif args.dtype == "fp16":
        ds_config["bf16"] = {'enabled': True}
    else:
        raise NotImplementedError
    accelerator.state.deepspeed_plugin.deepspeed_config = ds_config

    model, tokenizer, _ = get_model(
        args.model_path, 
        args.class_type,
        args.dtype,
        args.use_deepspeed_activation_checkpointing,
        args.num_checkpoints,
    )
    model, optimizer, lr_scheduler = _get_deepspeed_wrapped_model(model, accelerator)
    model.train()

    ## get torch profiler
    if args.use_torch_profiler:
        if args.recording_from_beginning:
            num_wait_steps, num_warmup_steps, num_active_steps, num_repeat = 0, 0, 3, 1
        else:
            num_wait_steps, num_warmup_steps, num_active_steps, num_repeat = 1, 2, 3, 1
        num_iter = int((num_wait_steps + num_warmup_steps + num_active_steps)*num_repeat)
        context = [
            get_torch_profiler(
                num_wait_steps=num_wait_steps,
                num_warmup_steps=num_warmup_steps,
                num_active_steps=num_active_steps,
                num_repeat=num_repeat,
            )
        ]
    else:
        context = []

    ## training (profiling) loop
    batch = get_dummy_input(tokenizer, device, args.seed, args.batch_size, args.seq_len)
    with ContextManagers(context) as p:
        for _ in tqdm(range(num_iter)):
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            # assert loss.dtype == DTYPE_SET[args.dtype], f"dtype error occurs ({loss.dtype} != {DTYPE_SET[args.dtype]})"

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            p.step()

            if rank == 0:
                print(f'''
                input.size() : {batch["input_ids"].size()}
                loss         : {loss}
                grad         : {model.model.embed_tokens.weight.grad}
                ''')

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str,
    )
    parser.add_argument(
        "--class_type", 
        type=str,
        default="optimized",
        choices=["auto", "optimized", "unsloth"]
    )
    parser.add_argument(
        "--dtype", 
        type=str,
        default="bf16",
        choices=["fp16", "bf16"]
    )
    parser.add_argument(
        "--recording_from_beginning", 
        action='store_true'
    )
    parser.add_argument(
        "--use_torch_profiler", 
        action='store_true'
    )
    parser.add_argument(
        "--use_deepspeed_activation_checkpointing", 
        action='store_true'
    )
    parser.add_argument(
        "--num_checkpoints", 
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seq_len", 
        type=int,
        default=32768,
    )
    args, _ = parser.parse_known_args()
    main(args)