#!/usr/bin/env python3
"""
pretokenize.py — Pre-format and pre-pack Mythos v3 training data
Run ONCE before training. Two-stage pipeline:

  Stage 1 — Chat template formatting:
    messages → text strings → saved as Arrow datasets under tokenized_data/{split}/

  Stage 2 — Tokenize + pack:
    text → input_ids → concatenated with EOS separators → chunked into
    max_seq_len-token blocks → saved under tokenized_data/{split}_packed/

With pre-packed data, SFTTrainer skips both its tokenization pass (~2 min)
and its packing pass (~15 min), reducing training startup from ~17 min to
under 30 seconds.

Usage:
    python3 pretokenize.py                                # all defaults
    python3 pretokenize.py --max-seq-len 1024            # match training seq len
    python3 pretokenize.py --num-proc 16                 # more CPU cores
    python3 pretokenize.py --no-pack                     # text format only
    python3 pretokenize.py --overwrite                   # force redo all splits
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
    p = argparse.ArgumentParser(description="Pre-format and pre-pack Mythos v3 data")
    p.add_argument("--model-id",    default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model ID (needed for tokenizer / chat template)")
    p.add_argument("--data-dir",    default="./training_data",
                   help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    p.add_argument("--output-dir",  default="./tokenized_data",
                   help="Output directory for Arrow datasets")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Sequence length for packing (must match training --max-seq-len)")
    p.add_argument("--num-proc",    type=int, default=8,
                   help="CPU processes for parallel map (default: 8)")
    p.add_argument("--overwrite",   action="store_true",
                   help="Re-process splits that already exist on disk")
    p.add_argument("--no-pack",     action="store_true",
                   help="Skip Stage 2 (text format only, no pre-packing)")
    return p.parse_args()


def pack_into_chunks(
    text_ds: Dataset,
    tokenizer,
    max_seq_len: int,
    num_proc: int,
    split_name: str,
) -> Dataset:
    """
    Tokenize a text dataset and stream-pack all tokens into fixed-length chunks.

    Each conversation is tokenized without padding/truncation, then appended
    to a rolling buffer with an EOS separator.  Every time the buffer reaches
    max_seq_len tokens a new training chunk is emitted.  The final partial
    buffer is discarded.

    Returns a Dataset with a single 'input_ids' column of shape (n_chunks, max_seq_len).
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        vocab = tokenizer.get_vocab()
        for candidate in ("<|im_end|>", "<|endoftext|>"):
            if candidate in vocab:
                eos_id = vocab[candidate]
                break
    if eos_id is None:
        raise ValueError("Cannot determine eos_token_id from tokenizer.")

    log.info(f"  [{split_name}] Stage 2a — tokenising {len(text_ds):,} examples (parallel, no truncation)...")
    t0 = time.time()

    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )

    ds_tok = text_ds.map(
        tok_fn,
        batched=True,
        batch_size=500,
        num_proc=num_proc,
        remove_columns=text_ds.column_names,
        desc=f"Tokenize {split_name}",
    )
    log.info(f"  [{split_name}] Tokenisation done in {time.time() - t0:.1f}s")

    log.info(f"  [{split_name}] Stage 2b — packing into {max_seq_len}-token chunks...")
    t1 = time.time()

    buf: list[int] = []
    chunk_arrays: list[np.ndarray] = []
    n_input_tokens = 0

    for batch in ds_tok.iter(batch_size=5_000):
        for ids in batch["input_ids"]:
            buf.extend(ids)
            buf.append(eos_id)
            n_input_tokens += len(ids) + 1

            # Emit complete chunks; keep the remainder in buf
            while len(buf) >= max_seq_len:
                chunk_arrays.append(np.array(buf[:max_seq_len], dtype=np.int32))
                del buf[:max_seq_len]

    if not chunk_arrays:
        raise ValueError(
            f"[{split_name}] No packed chunks produced — check data and --max-seq-len."
        )

    # Stack into (n_chunks, max_seq_len) numpy array — Arrow ingests this natively
    all_chunks = np.stack(chunk_arrays)
    n_chunks = len(all_chunks)
    util_pct = n_input_tokens / (n_chunks * max_seq_len) * 100
    log.info(
        f"  [{split_name}] {len(text_ds):,} examples → {n_input_tokens:,} tokens "
        f"→ {n_chunks:,} packed chunks  "
        f"(token utilisation: {util_pct:.1f}%,  pack time: {time.time() - t1:.1f}s)"
    )

    return Dataset.from_dict({"input_ids": all_chunks})


def main():
    args = parse_args()
    t_total = time.time()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Mythos v3 — Pre-tokenization + Packing")
    log.info("=" * 60)
    log.info(f"  Model      : {args.model_id}")
    log.info(f"  Data dir   : {data_dir.resolve()}")
    log.info(f"  Output dir : {output_dir.resolve()}")
    log.info(f"  Max seq len: {args.max_seq_len}  (must match training --max-seq-len)")
    log.info(f"  Num proc   : {args.num_proc}")
    log.info(f"  Pack       : {'no (--no-pack)' if args.no_pack else 'yes'}")
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
    log.info(f"  EOS token  : {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")

    # ── Splits ───────────────────────────────────────────────────────────────
    splits = [
        ("train", "train.jsonl"),
        ("val",   "val.jsonl"),
        ("test",  "test.jsonl"),
    ]

    packed_dirs_created = []

    for split_name, filename in splits:
        src      = data_dir / filename
        dst_text = output_dir / split_name
        dst_pack = output_dir / f"{split_name}_packed"

        if not src.exists():
            log.warning(f"  [{split_name}] Source not found: {src} — skipping")
            continue

        # ── Stage 1: chat-template formatting ────────────────────────────────
        if dst_text.exists() and not args.overwrite:
            log.info(f"  [{split_name}] Text format already exists — loading from {dst_text}")
            from datasets import load_from_disk
            text_ds = load_from_disk(str(dst_text))
        else:
            log.info(f"\n[{split_name}] Stage 1 — applying chat template to {src} ...")
            records = load_jsonl(src)
            log.info(f"  Loaded {len(records):,} examples")

            ds_raw = Dataset.from_list(records)

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
            text_ds = ds_raw.map(
                apply_template,
                batched=True,
                batch_size=500,
                num_proc=args.num_proc,
                remove_columns=ds_raw.column_names,
                desc=f"Format {split_name}",
            )

            before = len(text_ds)
            text_ds = text_ds.filter(
                lambda ex: len(ex["text"]) > 0, num_proc=args.num_proc
            )
            dropped = before - len(text_ds)
            if dropped:
                log.warning(f"  Dropped {dropped} empty examples")

            text_ds.save_to_disk(str(dst_text))
            log.info(f"  Saved {len(text_ds):,} text examples → {dst_text}  ({time.time()-t1:.1f}s)")

        # ── Stage 2: tokenise + pack ──────────────────────────────────────────
        if args.no_pack:
            continue

        if dst_pack.exists() and not args.overwrite:
            log.info(f"  [{split_name}] Packed format already exists — skipping (use --overwrite to redo)")
            packed_dirs_created.append(split_name)
            continue

        log.info(f"\n[{split_name}] Stage 2 — tokenise + pack (seq_len={args.max_seq_len}) ...")
        t2 = time.time()
        packed_ds = pack_into_chunks(text_ds, tokenizer, args.max_seq_len, args.num_proc, split_name)
        packed_ds.save_to_disk(str(dst_pack))
        log.info(f"  Saved {len(packed_ds):,} packed chunks → {dst_pack}  ({time.time()-t2:.1f}s)")
        packed_dirs_created.append(split_name)

    # ── Write metadata so train.py can validate seq_len ──────────────────────
    if not args.no_pack and packed_dirs_created:
        meta = {
            "max_seq_len": args.max_seq_len,
            "model_id":    args.model_id,
            "splits":      packed_dirs_created,
            "created_at":  datetime.now(timezone.utc).isoformat(),
        }
        meta_path = output_dir / "packed_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info(f"\n  Packed metadata → {meta_path}")

    total = time.time() - t_total
    log.info("\n" + "=" * 60)
    log.info(f"  Pre-tokenization complete in {total/60:.1f} minutes")
    log.info(f"  Output: {output_dir.resolve()}")
    if not args.no_pack:
        log.info(f"  Packed dirs: {', '.join(f'{s}_packed' for s in packed_dirs_created)}")
    log.info("=" * 60)
    log.info("\nNext step: run bash launch.sh")


if __name__ == "__main__":
    main()
