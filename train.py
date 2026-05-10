#!/usr/bin/env python3
"""
train.py — Mythos v3 fine-tuning script
Target model : Qwen/Qwen2.5-7B-Instruct
Configuration: LoRA r=128 α=256 | Unsloth | single GPU | bf16
Dataset split : 90 / 5 / 5  (train / val / test)
"""

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

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
    p.add_argument("--train-file",     default="training_data/train.jsonl")
    p.add_argument("--val-file",       default="training_data/val.jsonl")
    p.add_argument("--test-file",      default="training_data/test.jsonl",
                   help="Held-out test set evaluated once after training (90/5/5 split)")
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
    p.add_argument("--max-test-samples",  type=int, default=None,
                   help="Cap test examples for final evaluation (default: full test set)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--tokenized-dir",  default="./tokenized_data",
                   help="Directory of pre-formatted Arrow datasets (from pretokenize.py). "
                        "If present, skips apply_chat_template map at startup.")
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
    """Logs step timing, ETA, and VRAM usage."""
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
        if step % 10 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved  = torch.cuda.memory_reserved()  / 1024**3
            log.info(f"  VRAM: {allocated:.1f} GB allocated / {reserved:.1f} GB reserved")


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
        log.info(f"  Train file : {args.train_file}")
        log.info(f"  Val file   : {args.val_file}")
        log.info(f"  Test file  : {args.test_file}")
        log.info(f"  Split      : 90 / 5 / 5  (train / val / test)")
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

    # ── Model + Tokenizer (Unsloth loads directly to GPU) ──────────────────
    log.info(f"Loading model via Unsloth: {args.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Datasets ───────────────────────────────────────────────────────────
    tok_dir = Path(args.tokenized_dir)
    use_pretokenized = (
        tok_dir.exists()
        and (tok_dir / "train").exists()
        and (tok_dir / "val").exists()
    )

    if use_pretokenized:
        log.info(f"Loading pre-formatted datasets from {tok_dir} ...")
        train_ds = load_from_disk(str(tok_dir / "train"))
        val_ds   = load_from_disk(str(tok_dir / "val"))

        if args.max_train_samples:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        if args.max_val_samples:
            val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

        test_ds = None
        test_tok = tok_dir / "test"
        if test_tok.exists():
            test_ds = load_from_disk(str(test_tok))
            if args.max_test_samples:
                test_ds = test_ds.select(range(min(args.max_test_samples, len(test_ds))))
        else:
            log.warning("Pre-formatted test split not found — skipping final test eval")

        log.info(f"  train : {len(train_ds):,} examples")
        log.info(f"  val   : {len(val_ds):,} examples")
        if test_ds is not None:
            log.info(f"  test  : {len(test_ds):,} examples")

    else:
        log.info(f"Pre-formatted data not found at {tok_dir} — applying chat template on the fly")
        log.info("  (Run pretokenize.py once to speed up future starts)")

        log.info("Loading datasets...")
        train_ds = load_jsonl(args.train_file, args.max_train_samples)
        val_ds   = load_jsonl(args.val_file,   args.max_val_samples)

        test_ds = None
        if args.test_file and Path(args.test_file).exists():
            test_ds = load_jsonl(args.test_file, args.max_test_samples)
        elif args.test_file:
            log.warning(f"Test file not found: {args.test_file} — skipping final test eval")

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
        if test_ds is not None:
            test_ds = test_ds.map(
                lambda ex: apply_chat_template(ex, tokenizer),
                num_proc=4,
                desc="Format test",
            )

    # ── LoRA (Unsloth-optimised) ───────────────────────────────────────────
    log.info(f"Applying LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=QWEN_LORA_TARGETS,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    if is_main:
        model.print_trainable_parameters()
        # Sanity check memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            log.info(f"GPU memory after model load: {allocated:.1f} GB")

    # ── Training arguments ─────────────────────────────────────────────────
    sft_sig = inspect.signature(SFTConfig.__init__).parameters
    sft_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_gpu_batch,
        "per_device_eval_batch_size": args.per_gpu_batch,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": args.warmup_steps,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "logging_steps": args.log_steps,
        "logging_first_step": True,
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "eval_steps": args.eval_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": args.seed,
        "data_seed": args.seed,
        "optim": "adamw_torch_fused",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "report_to": "none",
        "run_name": "mythos-v3",
        "remove_unused_columns": True,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
    }

    # Version-specific compatibility across TRL/Transformers releases.
    if "max_length" in sft_sig:
        sft_kwargs["max_length"] = args.max_seq_len
    elif "max_seq_length" in sft_sig:
        sft_kwargs["max_seq_length"] = args.max_seq_len
    if "dataset_text_field" in sft_sig:
        sft_kwargs["dataset_text_field"] = "text"
    if "dataset_num_proc" in sft_sig:
        sft_kwargs["dataset_num_proc"] = 8
    if "packing" in sft_sig:
        sft_kwargs["packing"] = True
    if "packing_strategy" in sft_sig:
        sft_kwargs["packing_strategy"] = "bfd"
    if "assistant_only_loss" in sft_sig:
        sft_kwargs["assistant_only_loss"] = True
    if "eval_strategy" in sft_sig:
        sft_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in sft_sig:
        sft_kwargs["evaluation_strategy"] = "steps"

    # Keep only kwargs accepted by this installed SFTConfig.
    sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in sft_sig}
    training_args = SFTConfig(**sft_kwargs)

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

    # ── Evaluate on held-out test set (run once, after best model selected) ─
    if test_ds is not None:
        log.info("=" * 60)
        log.info("Running final evaluation on held-out TEST set...")
        log.info("=" * 60)
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        if is_main:
            log.info("Test set results:")
            for k, v in test_metrics.items():
                log.info(f"  {k}: {v}")

            # Persist test metrics alongside the adapter
            metrics_path = Path(args.output_dir) / "test_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(test_metrics, f, indent=2)
            log.info(f"Test metrics saved to {metrics_path}")

    # ── Save final adapter ─────────────────────────────────────────────────
    if is_main:
        log.info(f"Saving LoRA adapter to {args.output_dir}/final")
        trainer.save_model(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")
        log.info("LoRA adapter saved.")

        # ── Merge & save full model (optional) ────────────────────────────
        if args.merged_dir:
            log.info(f"Merging LoRA weights into base model → {args.merged_dir}")
            model.save_pretrained_merged(
                args.merged_dir, tokenizer, save_method="merged_16bit"
            )
            log.info(f"Merged model saved to {args.merged_dir}")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
