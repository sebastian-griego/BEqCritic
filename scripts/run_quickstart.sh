#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"

RUN_DIR="${RUN_DIR:-runs/quickstart}"
DATASET="${DATASET:-PAug/ProofNetVerif}"
SEED="${SEED:-0}"

TRAIN_SPLIT="${TRAIN_SPLIT:-valid}"
TEST_SPLIT="${TEST_SPLIT:-test}"
DEVICE="${BEQCRITIC_DEVICE:-}"
BASE_MODEL="${BEQCRITIC_BASE_MODEL:-microsoft/deberta-v3-small}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-0}"
TRAIN_MAX_PROBLEMS="${TRAIN_MAX_PROBLEMS:-0}"
TEST_MAX_PROBLEMS="${TEST_MAX_PROBLEMS:-0}"
AB_BOOTSTRAP="${AB_BOOTSTRAP:-2000}"

if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$ROOT/$RUN_DIR"
fi
mkdir -p "$RUN_DIR"

export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$RUN_DIR/hf_home}"

device_args=()
if [[ -n "$DEVICE" ]]; then
  device_args=(--device "$DEVICE")
fi

run_step() {
  local name="$1"
  shift
  echo "==> $name"
  local t0
  t0="$(date +%s)"
  "$@" 2>&1 | tee "$RUN_DIR/${name}.log"
  local t1
  t1="$(date +%s)"
  echo "${name}_seconds=$((t1 - t0))" >>"$RUN_DIR/timing.txt"
}

"$PYTHON_BIN" -m beqcritic.smoke >"$RUN_DIR/smoke.json"

CKPT_DIR="$RUN_DIR/checkpoints/beqcritic_deberta"
CAND="$RUN_DIR/proofnetverif_${TEST_SPLIT}_candidates.jsonl"
SEL_BEQ="$RUN_DIR/proofnetverif_${TEST_SPLIT}_selection_beqcritic.jsonl"
SEL_SELF="$RUN_DIR/proofnetverif_${TEST_SPLIT}_selection_selfbleu.jsonl"

run_step train_beqcritic \
  "$PYTHON_BIN" -m beqcritic.train_beq_critic \
  --dataset "$DATASET" \
  --split "$TRAIN_SPLIT" \
  --pred-key lean4_prediction \
  --ref-key lean4_formalization \
  --label-key correct \
  --problem-id-key id \
  --base-model "$BASE_MODEL" \
  --max-rows "$TRAIN_MAX_ROWS" \
  --max-problems "$TRAIN_MAX_PROBLEMS" \
  --output-dir "$CKPT_DIR" \
  --task-mix pred_vs_ref,cand_vs_cand \
  --epochs 1 \
  --batch-size 8 \
  --seed "$SEED" \
  --write-split-ids

run_step make_grouped_candidates \
  "$PYTHON_BIN" -m beqcritic.make_grouped_candidates \
  --dataset "$DATASET" \
  --split "$TEST_SPLIT" \
  --pred-key lean4_prediction \
  --ref-key lean4_formalization \
  --label-key correct \
  --problem-id-key id \
  --max-problems "$TEST_MAX_PROBLEMS" \
  --output "$CAND"

run_step select_beqcritic \
  "$PYTHON_BIN" -m beqcritic.score_and_select \
  --model "$CKPT_DIR" \
  --input "$CAND" \
  --output "$SEL_BEQ" \
  "${device_args[@]}" \
  --similarity critic \
  --threshold 0.5 \
  --tie-break medoid \
  --cluster-rank size_then_cohesion \
  --triangle-prune-margin 0.2 \
  --emit-stats

run_step eval_beqcritic \
  "$PYTHON_BIN" -m beqcritic.evaluate_selection \
  --candidates "$CAND" \
  --selections "$SEL_BEQ"

run_step select_selfbleu \
  "$PYTHON_BIN" -m beqcritic.self_bleu_select \
  --input "$CAND" \
  --output "$SEL_SELF"

run_step eval_selfbleu \
  "$PYTHON_BIN" -m beqcritic.evaluate_selection \
  --candidates "$CAND" \
  --selections "$SEL_SELF"

run_step ab_compare \
  "$PYTHON_BIN" -m beqcritic.evaluate_ab \
  --candidates "$CAND" \
  --selections-a "$SEL_SELF" --a-name selfbleu \
  --selections-b "$SEL_BEQ" --b-name beqcritic \
  --bootstrap "$AB_BOOTSTRAP" \
  --seed "$SEED" \
  --timing "$RUN_DIR/timing.txt" \
  --output-json "$RUN_DIR/ab_metrics.json" \
  --output-md "$RUN_DIR/ab_metrics.md"

echo "Done. Outputs in $RUN_DIR"
