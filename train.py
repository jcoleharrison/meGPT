"""
train.py - Fixed version
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
from transformers import Trainer, TrainingArguments
import os
import wandb
from datetime import datetime
import numpy as np
from torch.profiler import (
    profile as torch_profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)
from transformers import TrainerCallback, DataCollatorForLanguageModeling

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Meta-LLama model with LoRA on pre-tokenized JSONL"
    )
    parser.add_argument("--model_id",      type=str,
                        default="unsloth/Llama-3.2-3B-bnb-4bit")
    parser.add_argument("--train_file",    type=str, default="train.jsonl")
    parser.add_argument("--output_dir",    type=str, default="outputs")
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--epochs",     type=int, default=5)
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

def tokenize_and_label(batch, tokenizer):
    toks = tokenizer(
        batch["text"],
        truncation=True,
        padding=False,     # no dataset‐level padding
        max_length=2048
    )
    toks["labels"] = toks["input_ids"].copy()
    return toks


def resize_embeddings(model, tokenizer, target_weights: torch.Tensor = None):
    """
    Resize the input and output embeddings of the model to match the tokenizer size.
    This function also initializes the new embeddings with a mean and covariance
    based on the existing embeddings.
    """
    # 0) Ensure the model is in evaluation mode
    model.eval()

    # 1) Figure out sizes
    old_vocab = len(model.get_input_embeddings().weight)
    new_vocab = len(tokenizer)
    n_new     = new_vocab - old_vocab
    emb       = model.get_input_embeddings()
    device    = emb.weight.device
    dim       = emb.weight.size(1)
    generator = torch.Generator(device=device).manual_seed(0)

    # 2) Resize to grow the matrix (new rows init’d by HF default)
    model.resize_token_embeddings(new_vocab)

    if target_weights is not None:
        # If target_weights are provided, use them to initialize the new embeddings
        with torch.no_grad():
            model.get_input_embeddings().weight[old_vocab:] = target_weights.to(device=device, dtype=emb.weight.dtype)
            model.tie_weights()  # Re-tie input and output embeddings
        print(f"Resized embeddings with provided target weights: {target_weights.shape}")
        return
    else:
        # 3) Grab the “old” weights and compute mean + covariance
        with torch.no_grad():
            W_old = emb.weight[:old_vocab]                # shape [old_vocab, dim]
            mu     = W_old.mean(dim=0, keepdim=True)       # [1, dim]
            X     = W_old - mu                             # zero-centered
            # unbiased covariance matrix: [dim, dim]
            sigma     = (X.T @ X) / (old_vocab - 1)

            # regularize Σ for numerical stability
            eps = 1e-5
            sigma += torch.eye(dim, device=device) * eps

            # Cholesky factorization of Σ = L Lᵀ
            L32 = torch.linalg.cholesky(sigma.to(torch.float32))     # [dim, dim], float32
            L   = L32.to(device=device, dtype=torch.bfloat16)       # back to bfloat16 or float16

            # 4) Sample n_new vectors: z ~ N(0, I) →  μ + z @ Lᵀ
            z = torch.randn(n_new, dim, device=device, generator=generator).to(torch.bfloat16)    # [n_new, dim]
            new_embs = mu + (z @ L.T)                      # [n_new, dim]

            # 5) Overwrite the new rows
            model.get_output_embeddings().weight[old_vocab:] = new_embs

        # 6) Re-tie input and output embeddings
        model.tie_weights()

def preprocess_data(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def load_and_split(path, seed):
    raw = load_dataset("json", data_files=path, split="train")
    
    # Fix 1: Remove the input_ids mapping since SFTTrainer uses text field
    # The dataset already has "text" field from prep.py, so we don't need to map anything
    
    s = raw.train_test_split(test_size=0.05, seed=seed)
    splits = DatasetDict({
        "train": s["train"],
        "eval":   s["test"],
    })
    splits.save_to_disk(os.path.join("dataset_splits"))
    print("Dataset sizes:", {k: len(v) for k,v in splits.items()})
    return splits


# callback that advances the profiler on each step
class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof
    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
    def on_train_end(self, args, state, control, **kwargs):
        self.prof.__exit__(None, None, None)  # stop profiling


def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = prepare_tokenizer(args.model_id)

    
    splits = load_dataset("json", data_files=args.train_file, split="train").train_test_split(test_size=0.05, seed=args.seed)
    splits = DatasetDict(train=splits["train"], eval=splits["test"])
    tokenized_splits = splits.map(
        lambda x: tokenize_and_label(x, tokenizer),
        batched=True,
        num_proc=4,                # parallelize
        remove_columns=["text","conversation_id","num_tokens"],
    )
    tokenized_splits.set_format(
        type="torch",
        columns=["input_ids","attention_mask","labels"]
    )

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
    
    # Calculate the number of new tokens
    num_new_tokens = len(tokenizer) - len(model.get_input_embeddings().weight)

    # Resize embeddings to match tokenizer size
    resize_embeddings(model, tokenizer)
    # model.resize_token_embeddings(len(tokenizer))

    # 3) prepare LoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=get_lora_targets(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, 
        mlm=False, 
        pad_to_multiple_of=8
    )

    # 5) trainer
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        max_steps=5,
        # num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        fp16=False,  # bfloat16 is used
        dataloader_num_workers=4,
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
        greater_is_better=False,
        logging_dir="./logs",

        #wandb
        report_to="wandb",
        run_name=args.wandb_name,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["eval"],
        args=train_args,
        data_collator=data_collator,
    )
    wandb.watch(model, log="all", log_freq=100)
    
    trainer.train()

    # Save only resized new embedding weights
    resized_weights = model.get_input_embeddings().weight[len(model.get_input_embeddings().weight) - num_new_tokens:]
    torch.save(resized_weights, os.path.join(args.output_dir, "added_embeddings.pt"))

    wandb.finish()

if __name__ == "__main__":
    main()