import os
import random
import argparse
import numpy as np

import torch

from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from src.models.optimized_modeling_llama import OptimizedLlamaForCausalLM

from transformers import Trainer
# from trl import SFTTrainer as Trainer
from datasets import load_dataset
from transformers import default_data_collator

from pdb import set_trace as Tra


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
    model_class = OptimizedLlamaForCausalLM if args.class_type == 'custom_optimized' else AutoModelForCausalLM
    assert args.dtype == 'bf16'
    torch_dtype = DTYPE_SET[args.dtype]

    if args.class_type == 'unsloth':
        config = AutoConfig.from_pretrained(args.model_path)
        config._attn_implementation = 'sdpa'
        args_ = {
            'model_name': args.model_path,
            'max_seq_length': config.max_position_embeddings,
            'load_in_4bit': False,
            'dtype': torch_dtype,
            'device_map': None,
            'low_cpu_mem_usage': False,
        }
        from unsloth import FastLanguageModel 
        model = FastLanguageModel.from_pretrained(**args_)
    else:
        model_class = model_class
        config = AutoConfig.from_pretrained(args.model_path)
        config._attn_implementation = 'sdpa'
        args_ = {
            'pretrained_model_name_or_path': args.model_path,
            'config': config,
            'torch_dtype': torch_dtype,
        }
        model = model_class.from_pretrained(**args_)
        if args.class_type == 'liger':
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            liger_args = {
                'rope': False,
                'cross_entropy': False,
                'rms_norm': False,
                'swiglu': False,
                'fused_linear_cross_entropy': True,
            }
            apply_liger_kernel_to_llama(**liger_args)

    if args.use_grad_ckpt:
        model.gradient_checkpointing_enable()
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_sft_dataset(tokenizer, num_train_samples=100):
    # source from https://huggingface.co/blog/mlabonne/sft-llama3

    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"
    def apply_template(examples):
        text = []
        seq_len = []
        for example in examples["conversations"]:
            message = [{"role": "system", "content": "You are a helpful assistant."}]
            for i, item in enumerate(example):
                # assert item["from"] in ["gpt", "human"], f"item['from']: {item['from']} is not supported"
                if i == 0 and item["from"] == "system":
                    message[0]["content"] = item["value"]
                message.append({
                    "role": "assistant" if item["from"] == "gpt" else "user",
                    "content": item["value"],
                })
            output = tokenizer.apply_chat_template(message, tokenize=False)
            text.append(output)
            seq_len.append(len(output))
        return {"text": text, "seq_len": seq_len}
    dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    dataset = dataset.map(apply_template, batched=True)
    dataset = dataset.sort('seq_len', reverse=True).select(range(num_train_samples)) # to test how good is fuesed kernel at long context inputs
    return dataset

class Collator:
    def __init__(self, args: argparse.Namespace, tokenizer):
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.max_input_length = args.max_input_length

    def __call__(self, batch):
        assistant_turn_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        output_dict_ = []
        for item in batch:
            x = self.tokenizer.encode("".join(item['text'].split(assistant_turn_prefix)[:-1]) + assistant_turn_prefix, truncation=False)
            y = self.tokenizer.encode(item['text'].split(assistant_turn_prefix)[-1], truncation=False)
            input_ids = x + y
            labels = [-100] * len(x) + y
            attention_mask = [1] * len(input_ids)
            output_dict_.append(
                {
                    "input_ids": input_ids[-self.max_input_length:],
                    "labels":  labels[-self.max_input_length:],
                    "attention_mask": attention_mask[-self.max_input_length:],
                }
            )

        max_len = max([len(_['input_ids']) for _ in output_dict_])
        output_dict = []
        for item in output_dict_:
            output_dict.append(
                {
                    "input_ids": item['input_ids'] + (max_len-len(item['input_ids'])) * [self.tokenizer.pad_token_id],
                    "labels": item['labels'] + (max_len-len(item['labels'])) * [-100],
                    "attention_mask": item['attention_mask'] + (max_len - len(item['attention_mask'])) * [0],
                }
            )
        return default_data_collator(output_dict)

def get_hf_train_arguments(args):
    if args.ds_config is not None:
        if 'zero3' in args.ds_config: dist_suffix = 'zero3'
        elif 'zero2' in args.ds_config: dist_suffix = 'zero2'
        elif 'zero1' in args.ds_config: dist_suffix = 'zero1'
        else: raise NotImplementedError
    elif args.fsdp_config is not None:
        dist_suffix = 'fsdp'
    else:
        raise NotImplementedError

    return TrainingArguments(
        ## optimizer setting
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        num_train_epochs=1,
        max_steps=-1,
        per_device_train_batch_size=args.per_device_train_batch_size, # https://github.com/huggingface/accelerate/blame/6af157ea93dfbace1db88b0fdc7dfb568dfdd5a5/src/accelerate/accelerator.py#L1537
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,

        ## distributed setting
        deepspeed=args.ds_config if args.ds_config is not None else None,
        fsdp_config=args.fsdp_config if args.fsdp_config is not None else None,
        gradient_checkpointing=False,
        bf16=True if args.dtype == "bf16" else False,
        fp16=True if args.dtype == "fp16" else False,
        ddp_timeout=int(os.getenv("DDP_TIMEOUT", 1800)),

        ## logging, eval and save logic
        output_dir=os.path.join(
            args.output_dir,
            "class_{}_dtype_{}_dist_{}_gradckpt_{}_bsz_{}_seqlen_{}_accum_{}".format(
                args.class_type,
                args.dtype,
                dist_suffix,
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
        remove_unused_columns=False,
    )

def main(args: argparse.Namespace):
    _reset_seeds(args.seed)
    training_kwargs = get_hf_train_arguments(args) # for dist.init and zero-3, you should set TrainingArguments first
    model, tokenizer = get_model_and_tokenizer(args)
    train_dataset = get_sft_dataset(tokenizer, args.num_train_samples)
    trainer_kwargs = {
        "args": training_kwargs,
        "tokenizer": tokenizer,
        "model": model,
        "train_dataset": train_dataset,
        "data_collator": Collator(args, tokenizer), # https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L348
    }
    trainer = Trainer(**trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--class_type", type=str, default='custom_optimized', choices=['auto', 'liger', 'custom_optimized', 'unsloth'])
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16'])
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--fsdp_config", type=str, default=None)
    parser.add_argument("--ds_config", type=str, default=None)
    parser.add_argument("--use_grad_ckpt", action='store_true')

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    parser.add_argument("--output_dir", type=str, default='./tbd')
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--num_train_samples", type=int, default=100)

    args, _ = parser.parse_known_args()
    main(args)