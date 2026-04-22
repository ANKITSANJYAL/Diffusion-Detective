#!/usr/bin/env bash
# ============================================================================
# Diffusion Detective — Preflight → Full Ablation (one-shot)
# ============================================================================
# This script:
#   1. Runs the preflight smoke test (1 prompt × 24 treatments) to verify
#      the split baseline/intervention pipeline works end-to-end.
#   2. If smoke test passes, immediately launches the full SDXL ablation
#      (5000 prompts × 24 treatments = 120K interventions + 5K baselines).
#   3. Then launches the SD1.5 model ablation for comparison.
#
# Usage:
#   conda activate diff_cvpr
#   nohup bash run_ablation.sh > ablation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   tail -f ablation_*.log
# ============================================================================

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

NGPU=2
LOG_DIR="$PROJ_DIR/outputs/ablation_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Diffusion Detective — Full Ablation Pipeline"
echo "  Started: $(date)"
echo "  GPUs: $NGPU"
echo "  Log dir: $LOG_DIR"
echo "============================================================"


# ────────────────────────────────────────────────────────────────
# PHASE 0: PREFLIGHT SMOKE TEST
# ────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  PHASE 0/3: PREFLIGHT SMOKE TEST"
echo "  1 prompt × 24 treatments = 24 intervention passes"
echo "  Expected: ~5 min on 2×A100"
echo "═══════════════════════════════════════════════════════════"

SMOKE_START=$(date +%s)

torchrun --nproc_per_node=$NGPU -m experiments.run_experiment \
    --config-name=smoke_test 2>&1 | tee "$LOG_DIR/00_smoke_test.log"

SMOKE_END=$(date +%s)
SMOKE_ELAPSED=$(( SMOKE_END - SMOKE_START ))
echo "  ✓ Smoke test completed in ${SMOKE_ELAPSED}s"

# ── Validate smoke test output ──────────────────────────────────
# Find the most recent results directory
SMOKE_DIR=$(ls -td "$PROJ_DIR/experiments/results/preflight-smoke"* 2>/dev/null | head -1)

if [ -z "$SMOKE_DIR" ]; then
    echo "  ✗ FATAL: No smoke test results directory found!"
    exit 1
fi

JSONL="$SMOKE_DIR/aggregated_metrics.jsonl"
if [ ! -f "$JSONL" ]; then
    echo "  ✗ FATAL: No JSONL file found at $JSONL"
    exit 1
fi

ROW_COUNT=$(wc -l < "$JSONL")
echo ""
echo "  ── Smoke Test Validation ──"
echo "  Results dir: $SMOKE_DIR"
echo "  JSONL rows:  $ROW_COUNT  (expected: 24)"

if [ "$ROW_COUNT" -lt 24 ]; then
    echo "  ✗ FATAL: Expected 24 rows (6 strengths × 4 windows), got $ROW_COUNT"
    echo "  Aborting full ablation. Check $LOG_DIR/00_smoke_test.log"
    exit 1
fi

# Check that metrics are populated (not all null)
NULL_CLIPS=$(grep -c '"delta_clip": null' "$JSONL" || true)
echo "  Null Δ-CLIP: $NULL_CLIPS / $ROW_COUNT"

if [ "$NULL_CLIPS" -eq "$ROW_COUNT" ]; then
    echo "  ✗ FATAL: ALL delta_clip values are null — metrics pipeline broken"
    exit 1
fi

# Quick peek at metric ranges
echo ""
echo "  ── Metric Summary (first 5 rows) ──"
head -5 "$JSONL" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    m = d['metrics']
    tag = d['condition_tag']
    dc = m.get('delta_clip')
    lp = m.get('lpips')
    dc_str = f'{dc:+.4f}' if dc is not None else 'null'
    lp_str = f'{lp:.4f}' if lp is not None else 'null'
    print(f'    {tag:40s}  Δ-CLIP={dc_str}  LPIPS={lp_str}')
"

# Check saved images
IMG_COUNT=$(find "$SMOKE_DIR" -name "*.png" 2>/dev/null | wc -l)
echo ""
echo "  Saved images: $IMG_COUNT  (expected: ~48 baseline+steered)"
echo ""
echo "  ✓ PREFLIGHT PASSED — all systems go"
echo ""


# ────────────────────────────────────────────────────────────────
# PHASE 1: FULL SDXL ABLATION
# ────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  PHASE 1/3: SDXL ABLATION"
echo "  5000 prompts × 24 treatments = 120,000 interventions"
echo "  Start: $(date)"
echo "═══════════════════════════════════════════════════════════"

SDXL_START=$(date +%s)

torchrun --nproc_per_node=$NGPU -m experiments.run_experiment \
    --config-name=ablation 2>&1 | tee "$LOG_DIR/01_sdxl_ablation.log"

SDXL_END=$(date +%s)
SDXL_ELAPSED=$(( SDXL_END - SDXL_START ))
echo "  ✓ SDXL ablation completed in $(( SDXL_ELAPSED / 3600 ))h $(( (SDXL_ELAPSED % 3600) / 60 ))m"


# ────────────────────────────────────────────────────────────────
# PHASE 2: SD1.5 MODEL ABLATION
# ────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  PHASE 2/3: SD1.5 MODEL ABLATION"
echo "  5000 prompts × 24 treatments = 120,000 interventions"
echo "  Start: $(date)"
echo "═══════════════════════════════════════════════════════════"

SD15_START=$(date +%s)

torchrun --nproc_per_node=$NGPU -m experiments.run_experiment \
    --config-name=ablation \
    pipeline.model_id=runwayml/stable-diffusion-v1-5 \
    experiment_name=unified-ablation-sd15 2>&1 | tee "$LOG_DIR/02_sd15_ablation.log"

SD15_END=$(date +%s)
SD15_ELAPSED=$(( SD15_END - SD15_START ))
echo "  ✓ SD1.5 ablation completed in $(( SD15_ELAPSED / 3600 ))h $(( (SD15_ELAPSED % 3600) / 60 ))m"


# ────────────────────────────────────────────────────────────────
# PHASE 3: ATTRIBUTE DIVERSITY (SKIPPED — too expensive for now)
# Uncomment below to enable: 5000 × 288 = 1.44M interventions
# ────────────────────────────────────────────────────────────────
# echo ""
# echo "═══════════════════════════════════════════════════════════"
# echo "  PHASE 3/3: ATTRIBUTE DIVERSITY (SDXL)"
# echo "  5000 prompts × 288 treatments = 1,440,000 interventions"
# echo "  Start: $(date)"
# echo "═══════════════════════════════════════════════════════════"
#
# ATTR_START=$(date +%s)
#
# torchrun --nproc_per_node=$NGPU -m experiments.run_experiment \
#     --config-name=ablation \
#     sweep.sweep_attribute=true \
#     experiment_name=unified-ablation-attr-diversity 2>&1 | tee "$LOG_DIR/03_attr_diversity.log"
#
# ATTR_END=$(date +%s)
# ATTR_ELAPSED=$(( ATTR_END - ATTR_START ))
# echo "  ✓ Attribute diversity completed in $(( ATTR_ELAPSED / 3600 ))h $(( (ATTR_ELAPSED % 3600) / 60 ))m"


# ────────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - SMOKE_START ))
echo ""
echo "============================================================"
echo "  ALL ABLATIONS COMPLETE"
echo "  Total wall time: $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m"
echo "============================================================"
echo ""
echo "  Phase 0 — Smoke test:           1 × 24   =          24  (${SMOKE_ELAPSED}s)"
echo "  Phase 1 — SDXL ablation:    5,000 × 24   =     120,000"
echo "  Phase 2 — SD1.5 ablation:   5,000 × 24   =     120,000"
echo "  Phase 3 — Attr diversity:   SKIPPED (uncomment to enable)"
echo "  ────────────────────────────────────────────────────────"
echo "  GRAND TOTAL:                              ≈     240,024"
echo ""
echo "  Results: experiments/results/"
echo "  Logs:    $LOG_DIR/"
echo "============================================================"
