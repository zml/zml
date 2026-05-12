#!/usr/bin/env bash
set -euo pipefail

TARGET_MODEL="${TARGET_MODEL:-/Users/tristan/models/meta-llama/Llama-3.1-8B-Instruct}"
DFLASH_MODEL="${DFLASH_MODEL:-/Users/tristan/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat}"
ACTIVATIONS="${ACTIVATIONS:-/tmp/dflash_activations.safetensors}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ZML_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"

"$PYTHON" "$SCRIPT_DIR/extract_reference.py" \
  --target-model "$TARGET_MODEL" \
  --dflash-model "$DFLASH_MODEL" \
  --out "$ACTIVATIONS"

cd "$ZML_ROOT"
bazel run //examples/dflash:dflash_compare_activations -- \
  --model="$DFLASH_MODEL" \
  --activations="$ACTIVATIONS"
