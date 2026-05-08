#!/usr/bin/env python3
"""
train.py — Mythos v3 fine-tuning script
Target model : Qwen/Qwen2.5-7B-Instruct
Configuration: LoRA r=128 α=256 | ZeRO-2 | 4×GPU | bf16
Effective batch: 2 per GPU × 4 GPUs × 16 grad accum = 128
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Mythos v3 fine-tuning")
    p.add_argument("--model-id",       default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model ID or local path")
    p.add_argument("--train-file",     default="training_data_v5/train.jsonl")
    p.add_argument("--val-file",       default="training_data_v5/val.jsonl")
    p.add_argument("--output-dir",     default="./mythos-v3-lora")
    p.add_argument("--merged-dir",     default="./mythos-v3-merged",
                   help="Directory to save final merged model (set empty to skip)")

    # LoRA
    p.add_argument("--lora-r",         type=int,   default=128)
    p.add_argument("--lora-alpha",     type=int,   default=256)
    p.add_argument("--lora-dropout",   type=float, default=0.05)

    # Training
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--per-gpu-batch",  type=int,   default=2)
    p.add_argument("--grad-accum",     type=int,   default=16)
    p.add_argument("--max-seq-len",    type=int,   default=2048,
                   help="Max sequence length (maps to SFTConfig.max_length)")
    p.add_argument("--warmup-steps",   type=int,   default=100)
    p.add_argument("--save-steps",     type=int,   default=500)
    p.add_argument("--eval-steps",     type=int,   default=500)
    p.add_argument("--log-steps",      type=int,   default=10)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap training examples (debugging)")
    p.add_argument("--max-val-samples",   type=int, default=5000)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--deepspeed",      default="deepspeed_zero2.json",
                   help="Path to DeepSpeed config (pass empty string to disable)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LoRA target modules (covers all projection layers in Qwen2)
# ─────────────────────────────────────────────────────────────────────────────

QWEN_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str, max_samples: int | None = None) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    log.info(f"Loaded {len(records):,} examples from {path}")
    return Dataset.from_list(records)


def apply_chat_template(example: dict, tokenizer) -> dict:
    """Convert messages list → formatted string using the model's chat template."""
    messages = example["messages"]
    # apply_chat_template returns a string
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class ProgressCallback(TrainerCallback):
    """Logs step timing and ETA."""
    def __init__(self):
        self._last_log = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step  = state.global_step
        total = state.max_steps
        pct   = 100 * step / total if total else 0
        loss  = logs.get("loss", logs.get("train_loss", "?"))
        lr    = logs.get("learning_rate", "?")
        if isinstance(lr, float):
            lr = f"{lr:.2e}"
        log.info(f"Step {step}/{total} ({pct:.1f}%) | loss={loss} | lr={lr}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main    = local_rank == 0

    if is_main:
        log.info("=" * 60)
        log.info("  Mythos v3 — Security LLM Fine-tuning")
        log.info("=" * 60)
        log.info(f"  Model      : {args.model_id}")
        log.info(f"  LoRA       : r={args.lora_r}  alpha={args.lora_alpha}")
        log.info(f"  Precision  : BFloat16")
        log.info(f"  Per-GPU BS : {args.per_gpu_batch}")
        log.info(f"  Grad accum : {args.grad_accum}")
        log.info(f"  GPUs       : {torch.cuda.device_count()}")
        eff = args.per_gpu_batch * args.grad_accum * torch.cuda.device_count()
        log.info(f"  Eff. batch : {eff}")
        log.info(f"  LR         : {args.lr}")
        log.info(f"  Epochs     : {args.epochs}")
        log.info(f"  Max seqlen : {args.max_seq_len}")
        log.info("=" * 60)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Datasets ───────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    train_ds = load_jsonl(args.train_file, args.max_train_samples)
    val_ds   = load_jsonl(args.val_file,   args.max_val_samples)

    log.info("Applying chat template...")
    train_ds = train_ds.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        num_proc=8,
        desc="Format train",
    )
    val_ds = val_ds.map(
        lambda ex: apply_chat_template(ex, tokenizer),
        num_proc=4,
        desc="Format val",
    )

    # ── Model ──────────────────────────────────────────────────────────────
    log.info(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,          # must be False with gradient checkpointing
        attn_implementation="flash_attention_2",  # faster attention if available
    )

    # ── LoRA ───────────────────────────────────────────────────────────────
    log.info(f"Applying LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=QWEN_LORA_TARGETS,
        modules_to_save=[],       # keep base model frozen except LoRA
    )
    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()
        # Sanity check memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            log.info(f"GPU memory after model load: {allocated:.1f} GB")

    # ── Training arguments ─────────────────────────────────────────────────
    ds_config = args.deepspeed if (args.deepspeed and Path(args.deepspeed).exists()) else None
    if ds_config is None and args.deepspeed:
        log.warning(f"DeepSpeed config '{args.deepspeed}' not found — running without DeepSpeed")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_gpu_batch,
        per_device_eval_batch_size=args.per_gpu_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # ── Sequence & packing ──────────────────────────────────────────────
        max_length=args.max_seq_len,           # renamed from max_seq_length
        packing=True,                          # fill sequences to max_length
        packing_strategy="bfd",                # best-fit decreasing (default)
        dataset_text_field="text",
        dataset_num_proc=8,
        # ── Loss: train on assistant turns only ─────────────────────────────
        assistant_only_loss=True,              # ignore system+user tokens in loss
        # ── Logging & saving ────────────────────────────────────────────────
        logging_steps=args.log_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # ── Reproducibility ─────────────────────────────────────────────────
        seed=args.seed,
        data_seed=args.seed,
        # ── Optimizer ───────────────────────────────────────────────────────
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # ── DeepSpeed ───────────────────────────────────────────────────────
        deepspeed=ds_config,
        # ── Reporting ───────────────────────────────────────────────────────
        report_to="none",          # change to "wandb" if you want W&B tracking
        run_name="mythos-v3",
        # ── Misc ────────────────────────────────────────────────────────────
        remove_unused_columns=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,    # replaces deprecated `tokenizer=`
        callbacks=[ProgressCallback()],
    )

    # ── Resume from checkpoint if available ───────────────────────────────
    resume = None
    ckpt_dir = Path(args.output_dir)
    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("checkpoint-*"), key=os.path.getmtime)
        if checkpoints:
            resume = str(checkpoints[-1])
            log.info(f"Resuming from checkpoint: {resume}")

    # ── Train ──────────────────────────────────────────────────────────────
    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume)

    # ── Save final adapter ─────────────────────────────────────────────────
    if is_main:
        log.info(f"Saving LoRA adapter to {args.output_dir}/final")
        trainer.save_model(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")
        log.info("LoRA adapter saved.")

        # ── Merge & save full model (optional) ────────────────────────────
        if args.merged_dir:
            log.info(f"Merging LoRA weights into base model → {args.merged_dir}")
            from peft import PeftModel
            log.info("  Loading base model for merge...")
            base = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="cpu",
            )
            peft_model = PeftModel.from_pretrained(base, f"{args.output_dir}/final")
            merged = peft_model.merge_and_unload()
            merged.save_pretrained(args.merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.merged_dir)
            log.info(f"Merged model saved to {args.merged_dir}")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
