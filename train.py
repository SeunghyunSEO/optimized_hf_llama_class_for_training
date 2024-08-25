import os
import random
import argparse
import numpy as np

import torch

from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from trl import SFTTrainer
from datasets import load_dataset

from src.modules.unsloth.chat_templates import get_chat_template # copy unsloth
from src.models.optimized_modeling_llama import OptimizedLlamaForCausalLM

from pdb import set_trace as Tra


CLASS_SET = {
    'auto': {
        'config': AutoConfig,
        'model': AutoModelForCausalLM,
        'tokenizer': AutoTokenizer,
    },
    'optimized': {
        'config': AutoConfig,
        'model': OptimizedLlamaForCausalLM,
        'tokenizer': AutoTokenizer,
    },
}
DTYPE_SET = {
    "fp32" : torch.float32,
    "fp16" : torch.float16,
    "bf16" : torch.bfloat16,
}

def _reset_seeds(seed_val: int = 42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def get_model_and_tokenizer(args: argparse.Namespace):
    class_set = CLASS_SET[args.class_type] if not args.class_type == 'liger' else CLASS_SET['auto']
    torch_dtype = DTYPE_SET[args.dtype]

    config_class = class_set['config']
    model_class = class_set['model']
    tokenizer_class = class_set['tokenizer']
    config = config_class.from_pretrained(args.model_path)
    config._attn_implementation = 'sdpa'
    args_ = {
        'pretrained_model_name_or_path': args.model_path,
        'config': config,
        'torch_dtype': torch_dtype,
    }
    model = model_class.from_pretrained(**args_)
    if args.use_grad_ckpt:
        model.gradient_checkpointing_enable()
    if args.class_type == 'liger':
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        apply_liger_kernel_to_llama() 
        
    tokenizer = tokenizer_class.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_sft_dataset(tokenizer):
    # source from https://huggingface.co/blog/mlabonne/sft-llama3
    # https://github.com/unslothai/unsloth/blob/12b437e12204532f82542c12ac1ab00d19e3ebbf/unsloth/chat_templates.py#L708
    tokenizer = get_chat_template(
        tokenizer,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        chat_template="chatml",
    )
    def apply_template(examples):
        messages = examples["conversations"]
        text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
        return {"text": text}
    dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    dataset = dataset.map(apply_template, batched=True)
    return dataset

def get_hf_train_arguments(args):
    if 'zero3' in args.ds_config: zero = 'zero3'
    elif 'zero2' in args.ds_config: zero = 'zero2'
    elif 'zero1' in args.ds_config: zero = 'zero1'
    else: raise NotImplementedError

    # with open(args.ds_config, encoding="utf-8") as f:
    #     ds_config = json.load(f)

    return TrainingArguments(
        ## optimizer setting
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,

        num_train_epochs=2,
        max_steps=-1,
        per_device_train_batch_size=args.per_device_train_batch_size, # https://github.com/huggingface/accelerate/blame/6af157ea93dfbace1db88b0fdc7dfb568dfdd5a5/src/accelerate/accelerator.py#L1537
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,

        ## distributed setting
        deepspeed=args.ds_config,
        gradient_checkpointing=False,
        bf16=True if args.dtype == "bf16" else False,
        fp16=True if args.dtype == "fp16" else False,
        ddp_timeout=int(os.getenv("DDP_TIMEOUT", 1800)),

        ## logging, eval and save logic
        output_dir=os.path.join(
            args.output_dir,
            "class_{}_dtype_{}_zero_{}_gradckpt_{}_bsz_{}_seqlen_{}_accum_{}".format(
                args.class_type,
                args.dtype,
                zero,
                args.use_grad_ckpt,
                args.per_device_train_batch_size,
                args.max_input_length,
                args.gradient_accumulation_steps,
            )
        ),
        logging_steps=args.logging_steps,
        report_to=['tensorboard'],
        do_eval=False,
        evaluation_strategy="no",
        save_strategy="no",
        save_safetensors=False,
    )

class Collator:
    def __init__(self, args: argparse.Namespace, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, batch):
        if isinstance(batch[0], Dict):
            batch = [b['text'] for b in batch]
        input_dict = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        labels = input_dict['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        input_dict['labels'] = labels
        return input_dict

def main(args: argparse.Namespace):
    _reset_seeds(args.seed)
    training_kwargs = get_hf_train_arguments(args) # for dist.init and zero-3, you should set TrainingArguments first
    model, tokenizer = get_model_and_tokenizer(args)
    train_dataset = get_sft_dataset(tokenizer)
    trainer_kwargs = {
        "args": training_kwargs,
        "tokenizer": tokenizer,
        "model": model,
        "train_dataset": train_dataset,
        "dataset_text_field": "text",
        "max_seq_length": args.max_input_length,
        "dataset_num_proc": 2,
        "packing": True,
    }
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--class_type", type=str, default='auto', choices=['auto', 'liger', 'optimized'])
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16'])
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--ds_config", type=str, default="ds_configs/ds_config_zero3_bf16.json")
    parser.add_argument("--use_grad_ckpt", action='store_true')

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    parser.add_argument("--output_dir", type=str, default='./tbd')
    parser.add_argument("--logging_steps", type=int, default=5)

    args, _ = parser.parse_known_args()
    main(args)