"""
train.py
"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from datasets import load_dataset, DatasetDict
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os
import wandb
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Meta-LLama model with LoRA on pre-tokenized JSONL"
    )
    parser.add_argument("--model_id",      type=str,
                        default="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    parser.add_argument("--train_file",    type=str, default="train.jsonl")
    parser.add_argument("--output_dir",    type=str, default="outputs")
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--seed",          type=int, default=3407)
    parser.add_argument("--wandb_project", type=str, default="MeGPT")
    parser.add_argument("--wandb_name",    type=str, default=f"MeGPT-{timestamp}")
    return parser.parse_args()

def get_lora_targets(model):
    unique = set()
    for name, module in model.named_modules():
        if any(key in type(module).__name__ for key in ("Linear4bit","Linear")):
            unique.add(name.split(".")[-1])
    return list(unique)

def prepare_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    specials = ["<|OTHER|>","<|ME|>","<|DT_LONG|>","<|DT_SHORT|>"]
    tok.add_special_tokens({"additional_special_tokens": specials})
    # ensure we have a pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_and_split(path, seed):
    raw = load_dataset("json", data_files=path, split="train")
    # get tokens from "input_ids" field
    ds = raw.map(lambda ex: {"labels": ex["input_ids"]}, batched=False)
    s = ds.train_test_split(test_size=0.1, seed=seed)
    splits = DatasetDict({
        "train": s["train"],
        "eval":   s["test"],
    })
    splits.save_to_disk(os.path.join("dataset_splits"))
    print("Dataset sizes:", {k: len(v) for k,v in splits.items()})
    return splits

def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) tokenizer
    tokenizer = prepare_tokenizer(args.model_id)

    # 2) model + 4-bit quantization
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    # 3) prepare LoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=get_lora_targets(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 4) data
    splits = load_and_split(args.train_file, args.seed)

    # 5) trainer
    train_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        packing=True,
        warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        fp16=False,  # bfloat16 is used
        dataset_text_field="text",
        dataloader_num_workers=2,
        weight_decay=0.01,
        seed=args.seed,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        torch_compile=True,
        greater_is_better=False,
        logging_dir="./logs",
        #wandb
        report_to="wandb",
        run_name=args.wandb_name,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=splits["train"],
        eval_dataset=splits["eval"],
        args=train_args,
    )
    
    wandb.watch(model, log="all", log_freq=100)

    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        trainer.train()
    trainer.save_pretrained(args.output_dir)
    
    wandb.finish()  # <--------------- explicitly end the run

if __name__ == "__main__":
    main()