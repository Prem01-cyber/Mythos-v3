#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Mythos v3 — Training launch script
#  Config  : LoRA r=128 α=256 | Unsloth | single GPU | bf16 | effective batch=128
#  Dataset : training_data/  (90/5/5 train/val/test split)
#
#  Usage:
#    bash launch.sh                          # run with defaults
#    bash launch.sh --model-id Qwen/Qwen2.5-14B-Instruct   # larger model
#    bash launch.sh --max-train-samples 100000              # debug run
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
# Prevent Unsloth from auto-enabling its "smart offload" which moves embedding gradients
# to CPU RAM every backward pass, causing 9-16 GB/s PCIe spikes and ~4x slower steps.
export UNSLOTH_DISABLE_SMART_OFFLOAD=1

# Uncomment to enable WandB:
# export WANDB_PROJECT="mythos-v3"
# export WANDB_RUN_NAME="lora-r128-unsloth"

# ── GPU configuration ─────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-1}

# Verify GPU count
AVAILABLE_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Requested GPUs : $NUM_GPUS"
echo "Available GPUs : $AVAILABLE_GPUS"

if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "ERROR: Only $AVAILABLE_GPUS GPUs available, $NUM_GPUS requested."
    exit 1
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────
echo ""
echo "=== Pre-flight checks ==="

python3 - <<'PYCHECK'
import torch, sys

# GPU memory check
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1024**3
    print(f"  GPU {i}: {mem_gb:.1f} GB")
    if mem_gb < 20:
        print(f"    WARNING: GPU {i} has < 20GB. Consider smaller batch or ZeRO-3.")

# Check packages
required = ["unsloth", "trl"]
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} — MISSING")
        missing.append(pkg)

if missing:
    print(f"\nInstall missing packages:\n  pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll checks passed.")
PYCHECK

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_FILE="${SCRIPT_DIR}/training_data/train.jsonl"
VAL_FILE="${SCRIPT_DIR}/training_data/val.jsonl"
TEST_FILE="${SCRIPT_DIR}/training_data/test.jsonl"
TOKENIZED_DIR="${SCRIPT_DIR}/tokenized_data"
OUTPUT_DIR="${SCRIPT_DIR}/mythos-v3-lora"
MERGED_DIR="${SCRIPT_DIR}/mythos-v3-merged"
MODEL_ID=${MODEL_ID:-"Qwen/Qwen2.5-1.5B-Instruct"}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}   # must match --max-seq-len passed to train.py below
LORA_R=${LORA_R:-64}               # 64 = fast+quality, 128 = max capacity (marginal gain, +3% FLOPs)
LORA_ALPHA=${LORA_ALPHA:-128}      # keep alpha = 2 × r (standard scaling)

# ── Verify raw data exist ────────────────────────────────────────────────────
for f in "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

# ── Pre-tokenization + packing (run once; skipped automatically after first run) ─
echo ""
echo "=== Pre-tokenization + Packing ==="
PACKED_META="${TOKENIZED_DIR}/packed_metadata.json"

# Check whether a valid packed cache already exists for the current seq_len
NEED_PACK=true
if [ -f "$PACKED_META" ]; then
    CACHED_SEQLEN=$(python3 -c "import json; print(json.load(open('$PACKED_META'))['max_seq_len'])" 2>/dev/null || echo 0)
    if [ "$CACHED_SEQLEN" = "$MAX_SEQ_LEN" ] \
       && [ -d "${TOKENIZED_DIR}/train_packed" ] \
       && [ -d "${TOKENIZED_DIR}/val_packed" ]; then
        NEED_PACK=false
    fi
fi

if [ "$NEED_PACK" = "false" ]; then
    echo "  Pre-packed data found (seq_len=${MAX_SEQ_LEN}) — skipping"
else
    echo "  Running pretokenize.py  (text format + packing; ~10-20 min, only needed once per seq_len)..."
    python3 "${SCRIPT_DIR}/pretokenize.py" \
        --model-id   "$MODEL_ID" \
        --data-dir   "${SCRIPT_DIR}/training_data" \
        --output-dir "$TOKENIZED_DIR" \
        --num-proc   16 \
        --max-seq-len "$MAX_SEQ_LEN"
    echo "  Pre-tokenization + packing complete."
fi

TRAIN_LINES=$(wc -l < "$TRAIN_FILE")
VAL_LINES=$(wc -l < "$VAL_FILE")
TEST_LINES=$(wc -l < "$TEST_FILE")
echo ""
echo "=== Dataset (90/5/5 split) ==="
echo "  Train examples : $TRAIN_LINES  (90%)"
echo "  Val examples   : $VAL_LINES  (5%)"
echo "  Test examples  : $TEST_LINES  (5%)  ← held-out, evaluated after training"
echo "  Tokenized dir  : $TOKENIZED_DIR"
echo "  Output dir     : $OUTPUT_DIR"

# ── Training math ────────────────────────────────────────────────────────────
PER_GPU_BATCH=${PER_GPU_BATCH:-16}
PER_GPU_EVAL_BATCH=${PER_GPU_EVAL_BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
EPOCHS=3
EFF_BATCH=$(python3 -c "print($PER_GPU_BATCH * $NUM_GPUS * $GRAD_ACCUM)")

# Use packed chunk count for step estimate when available (raw TRAIN_LINES is misleading
# after packing: 1.6M raw examples pack into ~567K chunks at seq_len=1024)
if [ -d "${TOKENIZED_DIR}/train_packed" ]; then
    TRAIN_CHUNKS=$(python3 -c "
from datasets import load_from_disk
ds = load_from_disk('${TOKENIZED_DIR}/train_packed')
print(len(ds))
" 2>/dev/null || echo "$TRAIN_LINES")
else
    TRAIN_CHUNKS="$TRAIN_LINES"
fi
TOTAL_STEPS=$(python3 -c "import math; print(math.ceil($TRAIN_CHUNKS / $EFF_BATCH) * $EPOCHS)")
TOTAL_HOURS=$(python3 -c "print(f'{$TOTAL_STEPS * 42 / 3600:.1f}')")   # ~42s/it on A100 80GB

echo ""
echo "=== Training configuration ==="
echo "  Model          : $MODEL_ID"
echo "  LoRA           : r=${LORA_R}  alpha=${LORA_ALPHA}  dropout=0.0"
echo "  Per-GPU batch  : $PER_GPU_BATCH"
echo "  Eval batch     : $PER_GPU_EVAL_BATCH"
echo "  Grad accumul.  : $GRAD_ACCUM"
echo "  Effective batch: $EFF_BATCH"
echo "  GPUs           : $NUM_GPUS"
echo "  Epochs         : $EPOCHS"
echo "  Train chunks   : $TRAIN_CHUNKS  (packed at seq_len=$MAX_SEQ_LEN)"
echo "  Total steps    : ~$TOTAL_STEPS  (~${TOTAL_HOURS}hr @ 42s/it on A100 80GB)"
echo "  Learning rate  : 2e-5  (cosine schedule)"
echo "  Warmup steps   : 100"
echo "  Max seq len    : $MAX_SEQ_LEN"
echo "  Precision      : BFloat16"
echo "  Trainer        : Unsloth SFTTrainer (pre-packed input_ids)"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
echo "=== Launching training ==="
echo "$(date)"
echo ""

python3 "${SCRIPT_DIR}/train.py" \
    --model-id "$MODEL_ID" \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --test-file "$TEST_FILE" \
    --tokenized-dir "$TOKENIZED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --merged-dir "$MERGED_DIR" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout 0.0 \
    --epochs "$EPOCHS" \
    --lr 2e-5 \
    --per-gpu-batch "$PER_GPU_BATCH" \
    --per-gpu-eval-batch "$PER_GPU_EVAL_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --warmup-steps 100 \
    --save-steps 500 \
    --eval-steps 500 \
    --log-steps 10 \
    --max-val-samples 5000 \
    --seed 42 \
    "$@"

echo ""
echo "=== Training finished: $(date) ==="
echo "  LoRA adapter  : $OUTPUT_DIR/final"
echo "  Merged model  : $MERGED_DIR"
echo "  Test metrics  : $OUTPUT_DIR/test_metrics.json"
