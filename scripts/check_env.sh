#!/usr/bin/env bash
# Check that everything the pipeline needs is present, before you spend time on it.
#
#   bash scripts/check_env.sh
#
# Read-only: installs nothing, starts nothing, contacts nothing except the local
# Ollama server (which it pings with GET /api/tags).

set -uo pipefail

ok=0
warn=0
fail=0

say_ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; ok=$((ok+1)); }
say_warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; warn=$((warn+1)); }
say_fail() { printf '  \033[31mfail\033[0m  %s\n' "$1"; fail=$((fail+1)); }

echo
echo "python"
echo "------"

# Prefer the project venv. Checking the system interpreter reports missing
# packages for a project whose dependencies are perfectly installed, which
# trains the operator to ignore this script.
if [ -x .venv/bin/python ]; then
  PYTHON_BIN="$PWD/.venv/bin/python"
  say_ok "using the project venv: .venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
  say_warn "no .venv; falling back to $(basename "$PYTHON_BIN")"
else
  PYTHON_BIN=""
  say_fail "python3 not found"
fi

if [ -n "$PYTHON_BIN" ]; then
  version=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  if "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    say_ok "python $version (>= 3.10 required)"
  else
    say_fail "python $version is too old; 3.10 or newer is required"
  fi
fi

if [ -d .venv ]; then
  say_ok ".venv exists; activate it with: source .venv/bin/activate"
else
  say_warn "no .venv; create one with: python3 -m venv .venv && source .venv/bin/activate"
fi

echo
echo "pipeline dependencies"
echo "---------------------"
for module in yaml httpx numpy; do
  if [ -n "$PYTHON_BIN" ] && "$PYTHON_BIN" -c "import $module" >/dev/null 2>&1; then
    say_ok "$module"
  else
    say_fail "$module not installed; run: pip install -e ."
  fi
done

if [ -n "$PYTHON_BIN" ] && "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
  say_ok "pytest (dev)"
else
  say_warn "pytest not installed; run: pip install -e '.[dev]'"
fi

echo
echo "laya and torch (training and evaluation only)"
echo "--------------------------------------------"
if [ -n "$PYTHON_BIN" ] && "$PYTHON_BIN" -c "import laya" >/dev/null 2>&1; then
  "$PYTHON_BIN" - <<'PY'
import laya
print(f"  ok    laya {laya.__version__}")
PY
else
  say_warn "laya not installed. Stages 1-10 still work without it."
  say_warn "install it with: pip install -e '.[laya]'"
fi

if [ -n "$PYTHON_BIN" ] && "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  "$PYTHON_BIN" - <<'PY'
import torch
print(f"  ok    torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gib = props.total_memory / (1024 ** 3)
        print(f"  ok    gpu{i}: {props.name} ({gib:.1f} GiB, capability {props.major}.{props.minor})")
PY
else
  say_warn "torch not installed; only needed for training and evaluation"
fi

echo
echo "ollama (question generation, labeling, embeddings)"
echo "------------------------------------------------"
if command -v ollama >/dev/null 2>&1; then
  say_ok "ollama CLI present"
else
  say_warn "ollama CLI not on PATH (the API may still be running)"
fi

endpoint="${LAYA_OLLAMA_ENDPOINT:-http://127.0.0.1:11434}"
if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 5 "$endpoint/api/tags" >/dev/null 2>&1; then
  say_ok "ollama reachable at $endpoint"
  # A missing or mistyped model does not fail on load; it fails on the first
  # call, inside a retry loop, about a hundred seconds later. `preflight` is
  # the one implementation of that check, so the script delegates to it rather
  # than growing a second copy that can drift.
  if [ -x .venv/bin/laya-pipeline ]; then
    if .venv/bin/laya-pipeline preflight >/tmp/laya-preflight.log 2>&1; then
      say_ok "preflight passed (models, think flag, context window)"
    else
      say_fail "preflight found problems; see below"
      fail=$((fail+1))
    fi
    grep -oE 'finding=.*' /tmp/laya-preflight.log | sed 's/^finding=/          /' || true
    rm -f /tmp/laya-preflight.log
  else
    say_warn "preflight skipped; activate the venv or: pip install -e ."
  fi
else
  say_warn "ollama not reachable at $endpoint (stages 1, 2 and 11+ still work)"
fi

echo
echo "project files"
echo "-------------"
for path in configs/default.yaml prompts/questions.fa.md corpus AGENTS.md; do
  if [ -e "$path" ]; then
    say_ok "$path"
  else
    say_fail "$path is missing"
  fi
done

echo
echo "summary"
echo "-------"
printf '  %d ok, %d warnings, %d failures\n\n' "$ok" "$warn" "$fail"
if [ "$fail" -gt 0 ]; then
  exit 1
fi
exit 0