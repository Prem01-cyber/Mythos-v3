#!/usr/bin/env python3
"""
pretokenize.py — Pre-format Mythos v3 training data
Run ONCE before training. Applies the Qwen chat template to every example
and saves the resulting text strings as HuggingFace Arrow datasets to disk.

SFTTrainer still handles tokenization + packing internally, which preserves
assistant_only_loss masking. This script eliminates the per-run apply_chat_template
map over 1.6M examples.

Usage:
    python3 pretokenize.py                          # all defaults
    python3 pretokenize.py --data-dir ./training_data --output-dir ./tokenized_data
    python3 pretokenize.py --num-proc 16            # more CPU cores
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_jsonl(path: Path, max_samples: int | None = None) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_args():
    p = argparse.ArgumentParser(description="Pre-format Mythos v3 data")
    p.add_argument("--model-id",    default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model ID (needed for tokenizer / chat template)")
    p.add_argument("--data-dir",    default="./training_data",
                   help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    p.add_argument("--output-dir",  default="./tokenized_data",
                   help="Output directory for Arrow datasets")
    p.add_argument("--num-proc",    type=int, default=8,
                   help="CPU processes for parallel map (default: 8)")
    p.add_argument("--overwrite",   action="store_true",
                   help="Re-process splits that already exist on disk")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Mythos v3 — Pre-tokenization")
    log.info("=" * 60)
    log.info(f"  Model      : {args.model_id}")
    log.info(f"  Data dir   : {data_dir.resolve()}")
    log.info(f"  Output dir : {output_dir.resolve()}")
    log.info(f"  Num proc   : {args.num_proc}")
    log.info("=" * 60)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    log.info(f"  Vocab size : {tokenizer.vocab_size:,}")

    # ── Splits ───────────────────────────────────────────────────────────────
    splits = [
        ("train", "train.jsonl", None),
        ("val",   "val.jsonl",   None),   # cap applied in train.py at load time
        ("test",  "test.jsonl",  None),
    ]

    for split_name, filename, max_samples in splits:
        src = data_dir / filename
        dst = output_dir / split_name

        if not src.exists():
            log.warning(f"  [{split_name}] Source not found: {src} — skipping")
            continue

        if dst.exists() and not args.overwrite:
            log.info(f"  [{split_name}] Already exists at {dst} — skipping (use --overwrite to redo)")
            continue

        log.info(f"\n[{split_name}] Loading {src} ...")
        records = load_jsonl(src, max_samples)
        log.info(f"  Loaded {len(records):,} examples")

        ds = Dataset.from_list(records)

        # Apply chat template: messages → formatted text string
        def apply_template(batch):
            texts = []
            for msgs in batch["messages"]:
                try:
                    text = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    text = ""
                texts.append(text)
            return {"text": texts}

        t1 = time.time()
        ds = ds.map(
            apply_template,
            batched=True,
            batch_size=500,
            num_proc=args.num_proc,
            remove_columns=ds.column_names,
            desc=f"Format {split_name}",
        )

        # Drop any examples that produced empty text
        before = len(ds)
        ds = ds.filter(lambda ex: len(ex["text"]) > 0, num_proc=args.num_proc)
        dropped = before - len(ds)
        if dropped:
            log.warning(f"  Dropped {dropped} empty examples")

        ds.save_to_disk(str(dst))
        elapsed = time.time() - t1
        log.info(f"  Saved {len(ds):,} examples → {dst}  ({elapsed:.1f}s)")

    total = time.time() - t0
    log.info("\n" + "=" * 60)
    log.info(f"  Pre-tokenization complete in {total/60:.1f} minutes")
    log.info(f"  Output: {output_dir.resolve()}")
    log.info("=" * 60)
    log.info("\nNext step: run bash launch.sh")


if __name__ == "__main__":
    main()
