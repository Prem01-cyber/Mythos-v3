#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Mythos v3 — Evaluation launcher
#
#  Benchmarks fine-tuned model vs base Qwen2.5-7B-Instruct using vLLM.
#  vLLM auto-detects all available GPUs via tensor_parallel.
#
#  Prerequisites (run once):
#    pip install -r eval/requirements_eval.txt
#
#  Usage:
#    # Full eval — fine-tuned vs base (recommended, ~45-60 min)
#    bash eval/run_eval.sh
#
#    # Quick smoke test (~10 min)
#    bash eval/run_eval.sh --quick
#
#    # Fine-tuned only, no comparison
#    bash eval/run_eval.sh --no-base
#
#    # Specific benchmarks
#    bash eval/run_eval.sh --benchmarks secbench_mcq custom
#
#    # Custom model paths
#    FINETUNED=/path/to/merged BASEMODEL=Qwen/Qwen2.5-7B-Instruct bash eval/run_eval.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# vLLM 0.20+ attempts a DeepGEMM (FP8) kernel warmup even on non-Hopper GPUs.
# Disable it — the A100 doesn't support FP8 hardware anyway.
export VLLM_USE_DEEP_GEMM=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASEMODEL=${BASEMODEL:-"Qwen/Qwen2.5-7B-Instruct"}
OUTPUT_DIR="${PROJECT_DIR}/eval_results"
MAX_MCQ=${MAX_MCQ:-500}           # SecBench MCQ questions per model

# ── GPU auto-detect ───────────────────────────────────────────────────────────
N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
GPU_UTIL=${GPU_UTIL:-0.90}        # vLLM memory utilisation per GPU

# ── Auto-detect: merged model or LoRA adapter? ───────────────────────────────
# Priority: FINETUNED env var → mythos-v3-7b-merged → mythos-v3-merged → LoRA adapter
MODEL_FLAG=""
if [ -n "${FINETUNED:-}" ] && [ -d "$FINETUNED" ]; then
    MODEL_FLAG="--finetuned $FINETUNED"
    MODEL_DISPLAY="$FINETUNED (merged)"
elif [ -d "${PROJECT_DIR}/mythos-v3-7b-merged" ]; then
    MODEL_FLAG="--finetuned ${PROJECT_DIR}/mythos-v3-7b-merged"
    MODEL_DISPLAY="${PROJECT_DIR}/mythos-v3-7b-merged (merged)"
elif [ -d "${PROJECT_DIR}/mythos-v3-merged" ]; then
    MODEL_FLAG="--finetuned ${PROJECT_DIR}/mythos-v3-merged"
    MODEL_DISPLAY="${PROJECT_DIR}/mythos-v3-merged (merged)"
elif [ -n "${ADAPTER:-}" ] && [ -d "$ADAPTER" ]; then
    MODEL_FLAG="--adapter $ADAPTER"
    MODEL_DISPLAY="$ADAPTER (LoRA adapter, no merge)"
elif [ -d "${PROJECT_DIR}/mythos-v3-7b-lora/final" ]; then
    ADAPTER="${PROJECT_DIR}/mythos-v3-7b-lora/final"
    MODEL_FLAG="--adapter $ADAPTER"
    MODEL_DISPLAY="$ADAPTER (LoRA adapter, no merge)"
else
    echo "ERROR: No fine-tuned model found. Tried:"
    echo "  \$FINETUNED, mythos-v3-7b-merged, mythos-v3-merged, mythos-v3-7b-lora/final"
    echo ""
    echo "  To merge the LoRA adapter into a full model first:"
    echo "    python3 eval/merge.py"
    echo ""
    echo "  Or run eval directly from the adapter (no merge needed):"
    echo "    ADAPTER=./mythos-v3-7b-lora/final bash eval/run_eval.sh"
    exit 1
fi

echo ""
echo "=== Mythos v3 Evaluation ==="
echo "  Model      : $MODEL_DISPLAY"
echo "  Base model : $BASEMODEL"
echo "  GPUs       : $N_GPUS"
echo "  GPU util   : $GPU_UTIL"
echo "  Output dir : $OUTPUT_DIR"
echo "  Max MCQ    : $MAX_MCQ"
echo ""

# ── Check dependencies ────────────────────────────────────────────────────────
python3 - <<'PYCHECK'
missing = []
for pkg in ["vllm", "datasets", "transformers"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"Missing packages: {' '.join(missing)}")
    print("Run: pip install -r eval/requirements_eval.txt")
    import sys; sys.exit(1)
print("  Dependencies OK")
PYCHECK

mkdir -p "$OUTPUT_DIR"

# ── Run evaluation ────────────────────────────────────────────────────────────
echo "=== Starting evaluation: $(date) ==="
echo ""

python3 "${SCRIPT_DIR}/eval.py" \
    $MODEL_FLAG \
    --base       "$BASEMODEL" \
    --output-dir "$OUTPUT_DIR" \
    --max-mcq    "$MAX_MCQ" \
    --gpu-util   "$GPU_UTIL" \
    "$@"

echo ""
echo "=== Evaluation complete: $(date) ==="
echo "  Results → $OUTPUT_DIR"
echo ""

# Show the latest markdown report
LATEST_REPORT=$(ls -t "${OUTPUT_DIR}"/eval_*.md 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    echo "─── Report summary ─────────────────────────────────────────────"
    # Print everything up to and including the first table
    head -40 "$LATEST_REPORT"
    echo "..."
    echo "Full report: $LATEST_REPORT"
fi
