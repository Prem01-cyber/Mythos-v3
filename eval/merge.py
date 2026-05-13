#!/usr/bin/env python3
"""
merge.py — Merge Mythos v3 LoRA adapter into a full BF16 model for vLLM.

Loads the base Qwen2.5-7B-Instruct model via Unsloth, applies the saved LoRA
adapter, merges weights, and saves a standalone HuggingFace model directory
that vLLM can load directly.

Usage:
    python3 eval/merge.py                                    # defaults
    python3 eval/merge.py --adapter ./mythos-v3-7b-lora/final
    python3 eval/merge.py --adapter ./mythos-v3-7b-lora/final \
                          --output  ./mythos-v3-merged
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Merge Mythos v3 LoRA adapter")
    p.add_argument("--adapter", default="./mythos-v3-7b-lora/final",
                   help="Path to saved LoRA adapter (default: ./mythos-v3-7b-lora/final)")
    p.add_argument("--base",    default="Qwen/Qwen2.5-7B-Instruct",
                   help="Base model ID or path")
    p.add_argument("--output",  default="./mythos-v3-merged",
                   help="Output directory for merged model")
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Max sequence length for Unsloth (default: 4096)")
    return p.parse_args()


def main():
    args = parse_args()
    adapter_path = Path(args.adapter)
    output_path  = Path(args.output)

    if not adapter_path.exists():
        log.error(f"Adapter not found: {adapter_path}")
        log.error("Check ls mythos-v3-7b-lora/ for available checkpoints.")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  Mythos v3 — LoRA Merge")
    log.info("=" * 60)
    log.info(f"  Adapter  : {adapter_path.resolve()}")
    log.info(f"  Base     : {args.base}")
    log.info(f"  Output   : {output_path.resolve()}")
    log.info("=" * 60)

    import torch
    t0 = time.time()

    # ── Load base model via Unsloth ───────────────────────────────────────────
    log.info("Loading base model (Unsloth)...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    # ── Apply LoRA adapter ────────────────────────────────────────────────────
    log.info(f"Applying LoRA adapter from {adapter_path} ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))
    log.info("Adapter applied.")

    # ── Merge and save ────────────────────────────────────────────────────────
    log.info("Merging weights (BF16) and saving — this takes ~5-10 min ...")
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )

    elapsed = (time.time() - t0) / 60
    log.info("=" * 60)
    log.info(f"  Merge complete in {elapsed:.1f} min")
    log.info(f"  Merged model → {output_path.resolve()}")
    log.info("=" * 60)
    log.info("")
    log.info("Next step — run evaluation:")
    log.info(f"  FINETUNED={output_path} bash eval/run_eval.sh --quick")


if __name__ == "__main__":
    main()
