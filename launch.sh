#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Mythos v3 — Training launch script
#  Config  : LoRA r=128 α=256 | ZeRO-2 | bf16 | flash-attn | no CPU offload
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
export NCCL_DEBUG=WARN

# Uncomment if using WandB:
# export WANDB_PROJECT="mythos-v3"
# export WANDB_RUN_NAME="lora-r128-zero2"
export WANDB_DISABLED=true

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
required = ["peft", "trl", "deepspeed", "flash_attn"]
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
DS_CONFIG="${SCRIPT_DIR}/deepspeed_zero2.json"
OUTPUT_DIR="${SCRIPT_DIR}/mythos-v3-lora"
MERGED_DIR="${SCRIPT_DIR}/mythos-v3-merged"

# ── Verify raw data and config exist ─────────────────────────────────────────
for f in "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE" "$DS_CONFIG"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

# ── Pre-tokenization (run once, skipped automatically on subsequent runs) ─────
echo ""
echo "=== Pre-tokenization ==="
if [ -d "${TOKENIZED_DIR}/train" ] && [ -d "${TOKENIZED_DIR}/val" ]; then
    echo "  Pre-formatted data found at ${TOKENIZED_DIR} — skipping"
else
    echo "  Running pretokenize.py (this takes ~5-10 min, only needed once)..."
    python3 "${SCRIPT_DIR}/pretokenize.py" \
        --data-dir   "${SCRIPT_DIR}/training_data" \
        --output-dir "$TOKENIZED_DIR" \
        --num-proc   6
    echo "  Pre-tokenization complete."
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
PER_GPU_BATCH=${PER_GPU_BATCH:-4}
GRAD_ACCUM=${GRAD_ACCUM:-24}
EPOCHS=3
EFF_BATCH=$(python3 -c "print($PER_GPU_BATCH * $NUM_GPUS * $GRAD_ACCUM)")
TOTAL_STEPS=$(python3 -c "import math; print(math.ceil($TRAIN_LINES / $EFF_BATCH) * $EPOCHS)")

echo ""
echo "=== Training configuration ==="
echo "  Model          : Qwen/Qwen2.5-7B-Instruct"
echo "  LoRA           : r=128  alpha=256  dropout=0.05"
echo "  Per-GPU batch  : $PER_GPU_BATCH"
echo "  Grad accumul.  : $GRAD_ACCUM"
echo "  Effective batch: $EFF_BATCH"
echo "  GPUs           : $NUM_GPUS"
echo "  Epochs         : $EPOCHS"
echo "  Total steps    : ~$TOTAL_STEPS"
echo "  Learning rate  : 2e-5  (cosine schedule)"
echo "  Warmup steps   : 100"
echo "  Precision      : BFloat16"
echo "  ZeRO stage     : 2"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
echo "=== Launching training ==="
echo "$(date)"
echo ""

deepspeed \
    --num_gpus="$NUM_GPUS" \
    --master_port=29500 \
    "${SCRIPT_DIR}/train.py" \
    --model-id "Qwen/Qwen2.5-7B-Instruct" \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --test-file "$TEST_FILE" \
    --tokenized-dir "$TOKENIZED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --merged-dir "$MERGED_DIR" \
    --lora-r 128 \
    --lora-alpha 256 \
    --lora-dropout 0.05 \
    --epochs "$EPOCHS" \
    --lr 2e-5 \
    --per-gpu-batch "$PER_GPU_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --max-seq-len 2048 \
    --warmup-steps 100 \
    --save-steps 500 \
    --eval-steps 500 \
    --log-steps 10 \
    --max-val-samples 5000 \
    --seed 42 \
    --deepspeed "$DS_CONFIG" \
    "$@"

echo ""
echo "=== Training finished: $(date) ==="
echo "  LoRA adapter  : $OUTPUT_DIR/final"
echo "  Merged model  : $MERGED_DIR"
echo "  Test metrics  : $OUTPUT_DIR/test_metrics.json"
