#!/usr/bin/env bash
# One-shot kernel comparison: dump Python + Zig TTIR, push both through XLA,
# extract per-stage IR, and diff every stage by default. Pass `--kernel NAME`
# to do just one kernel, or `--stage S` to diff only one stage.
#
# Uses `python` from PATH — activate your venv first, or have `triton`,
# `torch`, `numpy` installed at the system level.
# Required platform: CUDA or ROCm (XLA's Triton pipeline is GPU-only).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"

KERNEL=""
STAGE=""
CLEAN=0
SHOW_DIFFS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel) KERNEL="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --clean) CLEAN=1; shift ;;
        --show-diffs) SHOW_DIFFS=1; shift ;;
        -h|--help)
            echo "usage: $0 [--kernel NAME] [--stage S] [--show-diffs] [--clean]"
            echo "  --clean       remove all generated output dirs and exit"
            echo "  --show-diffs  print full diff bodies for every divergence"
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

OUT="examples/triton_emitter"
PY_IR="$OUT/py_ir"
ZIG_IR="$OUT/zig_ir"
XLA_PY="$OUT/xla_dump_py"
XLA_ZIG="$OUT/xla_dump_zig"
XLA_PY_OUT="$OUT/xla_py"
XLA_ZIG_OUT="$OUT/xla_zig"

if [[ $CLEAN -eq 1 ]]; then
    rm -rf "$PY_IR" "$ZIG_IR" "$XLA_PY" "$XLA_ZIG" "$XLA_PY_OUT" "$XLA_ZIG_OUT"
    echo "cleaned: $PY_IR $ZIG_IR $XLA_PY $XLA_ZIG $XLA_PY_OUT $XLA_ZIG_OUT"
    exit 0
fi

filter=()
# Use `--kernel=NAME` (single arg) so both Python (argparse) and the Zig tools
# (`zig-args`, which only recognizes `=`-separated values) accept the same flag.
[[ -n "$KERNEL" ]] && filter=("--kernel=$KERNEL")

echo "=== 1. dump_python ==="
python "$OUT/dump_python_ir.py" --out-dir "$PY_IR" "${filter[@]}"

echo "=== 2. dump_zig ==="
bazel run //examples/triton_emitter:dump_zig_ir -- --out-dir="$(pwd)/$ZIG_IR" "${filter[@]}"

echo "=== 3. lower py_ir via XLA ==="
rm -rf "$XLA_PY"
bazel run //examples/triton_emitter:dump_via_xla --//platforms:cuda=True -- \
    --in-dir="$(pwd)/$PY_IR" --out-dir="$(pwd)/$XLA_PY" "${filter[@]}"

echo "=== 4. lower zig_ir via XLA ==="
rm -rf "$XLA_ZIG"
bazel run //examples/triton_emitter:dump_via_xla --//platforms:cuda=True -- \
    --in-dir="$(pwd)/$ZIG_IR" --out-dir="$(pwd)/$XLA_ZIG" "${filter[@]}"

echo "=== 5. extract per-stage IR ==="
rm -rf "$XLA_PY_OUT" "$XLA_ZIG_OUT"
python "$OUT/extract_xla_dump.py" --in-dir "$XLA_PY"  --out-dir "$XLA_PY_OUT"
python "$OUT/extract_xla_dump.py" --in-dir "$XLA_ZIG" --out-dir "$XLA_ZIG_OUT"

echo "=== 6. compare ==="
cmp_args=(--left "$XLA_PY_OUT" --right "$XLA_ZIG_OUT" "${filter[@]}")
[[ -n "$STAGE" ]] && cmp_args+=(--stage "$STAGE")
[[ $SHOW_DIFFS -eq 1 ]] && cmp_args+=(--show-diffs)
python "$OUT/compare_ir.py" "${cmp_args[@]}"
