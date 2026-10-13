#!/usr/bin/env bash
# The manual execution order, one stage per invocation.
#
#   bash scripts/run_pipeline.sh                 # everything up to (not including) training
#   bash scripts/run_pipeline.sh --with-train    # also run the smoke profile, after a prompt
#
# Every stage is skipped if its artifact already exists, so the script is safe to
# re-run after a failure. Delete the artifact to force a stage to repeat.
#
# This script does NOT start GPU training unless you pass --with-train, and even
# then only the smoke profile runs, and only after you type the checkpoint path.

set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
# Prefer the project venv, for the same reason check_env.sh does: reporting a
# missing dependency for an installed one trains the operator to ignore errors.
if [ -z "${PYTHON:-}" ] && [ -x .venv/bin/python ]; then
  PY="$PWD/.venv/bin/python"
elif [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  PY="$(command -v python3)"
fi
WITH_TRAIN=0
SMOKE_OUT="${SMOKE_OUT:-outputs/checkpoints/smoke}"

for arg in "$@"; do
  case "$arg" in
    --with-train) WITH_TRAIN=1 ;;
    --config=*) CONFIG="${arg#*=}" ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

run() {
  local stage="$1"; shift
  echo
  echo "=============================================================="
  echo "  $stage"
  echo "  \$ $*"
  echo "=============================================================="
  "$@"
}

# 0: one second. A mistyped model name otherwise fails inside the retry loop,
# a hundred seconds per chunk, for the whole of `generate`.
run "preflight [ollama]    one second; refuses to continue on a blocking finding" \
                         "$PY" -m laya_pipeline --config "$CONFIG" preflight

# 1-2: local, fast.
run "scan      [local]"    "$PY" -m laya_pipeline --config "$CONFIG" scan
run "process   [local]     downloads the tokenizer once" \
                         "$PY" -m laya_pipeline --config "$CONFIG" process

# 3: the first slow stage. Minutes to hours depending on the corpus and the model.
run "generate  [ollama]    SLOW — questions and key points" \
                         "$PY" -m laya_pipeline --config "$CONFIG" generate

# 4: local, and it must happen before negatives (leakage prevention).
run "split     [local]     document-level train/validation/test" \
                         "$PY" -m laya_pipeline --config "$CONFIG" split

# 5-7: the rest of the slow stages.
run "negatives [ollama]    SLOW — mined inside each split's own chunk pool" \
                         "$PY" -m laya_pipeline --config "$CONFIG" negatives
run "label     [ollama]    SLOW — teacher labeling and the review queue" \
                         "$PY" -m laya_pipeline --config "$CONFIG" label

if [ -f outputs/review/review.csv ]; then
  echo
  echo ">>> HUMAN REVIEW STEP"
  echo "    outputs/review/review.csv is waiting for you."
  echo "    Correct the final_label column, save it, then:"
  echo "      $PY -c \"from laya_pipeline.config import load_config; from laya_pipeline.labeling import apply_corrections, read_review_csv; ...\""
  echo "    See the README section 'Manual label review' for the exact command."
  echo "    Continuing without it is allowed; those labels stay as the teacher set them."
fi

run "tasks sufficiency [ollama] SLOW" \
                         "$PY" -m laya_pipeline --config "$CONFIG" tasks --task sufficiency
run "tasks routing      [ollama] SLOW" \
                         "$PY" -m laya_pipeline --config "$CONFIG" tasks --task routing

# 8-10: local.
run "build     [local]"    "$PY" -m laya_pipeline --config "$CONFIG" build --task all
run "validate  [local]"   "$PY" -m laya_pipeline --config "$CONFIG" validate --task all
run "stats     [local]"    "$PY" -m laya_pipeline --config "$CONFIG" stats --task all

# 11: loads the tokenizer, no weights, no training. Always safe.
run "train dry-run [weights, no training]" \
                         "$PY" -m laya_pipeline --config "$CONFIG" train \
                           --task relevance --profile configs/dry_run.yaml --dry-run

if [ "$WITH_TRAIN" -eq 1 ]; then
  echo
  echo ">>> GPU TRAINING"
  echo "    About to run the smoke profile: one epoch, encoder frozen."
  echo "    Output directory: $SMOKE_OUT"
  read -r -p "    Type the output directory to confirm, or anything else to abort: " confirm
  if [ "$confirm" != "$SMOKE_OUT" ]; then
    echo "aborted"
    exit 1
  fi
  run "train smoke [gpu-train] MINUTES" \
                         "$PY" -m laya_pipeline --config "$CONFIG" train \
                           --task relevance --profile configs/smoke.yaml \
                           --out "$SMOKE_OUT" --yes
fi

echo
echo "done. next: read README.md section 'Fine-tuning commands'."
echo "nothing was trained unless you passed --with-train."