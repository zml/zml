#!/usr/bin/env bash
#
# Apples-to-apples diff between the Zig DSL output and Pallas's `debug=True`
# IR after both go through `canonicalize,cse,canonicalize`.
#
# Usage:
#   compare.sh                                   # default sweep
#   compare.sh HEAD_DIM KV_DTYPE                 # single combo
#   COMBOS="128:bf16 256:bf16 128:f32" compare.sh
#
# Defaults assume:
#   - Zig binary built at `bazel-bin/examples/mosaic_ragged_paged/ragged_paged_attention`
#     (already canonicalizes via `finishOpts(.{ .canonicalize = true })`).
#   - Pallas IR is generated on-the-fly by running `kernel.py HEAD_DIM KV_DTYPE`
#     and grep'ing `^module {` … `^}$` from the `debug=True` output.
#   - The `canonicalize` companion tool built (//examples/mosaic_ragged_paged:canonicalize).
#
# Per combo, writes:
#   /tmp/rpa_compare/<combo>/zig_canon.mlir       — Zig output (already canonicalized).
#   /tmp/rpa_compare/<combo>/pallas_canon.mlir    — Pallas IR after the same passes.
#   /tmp/rpa_compare/<combo>/zig_canon.norm.mlir  — `%foo`-stripped form of the above.
#   /tmp/rpa_compare/<combo>/pallas_canon.norm.mlir
#
# And prints a structural diff (everything stripped down to op shape +
# attributes; SSA value names and ordering of independent ops are noise).

set -uo pipefail

OUT=/tmp/rpa_compare
mkdir -p "$OUT"

cd "$(git rev-parse --show-toplevel)"

bazel build //examples/mosaic_ragged_paged:ragged_paged_attention //examples/mosaic_ragged_paged:canonicalize >/dev/null

if [[ $# -eq 2 ]]; then
  COMBOS="$1:$2"
fi
: "${COMBOS:=128:bf16 256:bf16 128:f16 256:f16 128:f32 256:f32 128:i32 256:i32 128:i8 256:i8 128:f8e4m3fn 256:f8e4m3fn}"

run_combo() {
  local head_dim="$1"
  local kv_dtype="$2"
  local combo="${head_dim}_${kv_dtype}"
  local dir="$OUT/$combo"
  mkdir -p "$dir"

  bazel-bin/examples/mosaic_ragged_paged/ragged_paged_attention "$head_dim" "$kv_dtype" \
    2>/dev/null > "$dir/zig_canon.mlir"

  if [[ ! -s "$dir/pallas_mosaic.mlir" || -n "${REGEN_PALLAS:-}" ]]; then
    ( cd examples/mosaic_ragged_paged \
      && uv run --with jax --with jaxlib python kernel.py "$head_dim" "$kv_dtype" 2>&1 \
        | sed -n '/^module {/,/^}$/p' > "$dir/pallas_mosaic.mlir" )
  fi
  if [[ ! -s "$dir/pallas_mosaic.mlir" ]]; then
    echo "  pallas IR generation failed for $combo — check kernel.py output" >&2
    return 1
  fi

  bazel-bin/examples/mosaic_ragged_paged/canonicalize "$dir/pallas_mosaic.mlir" \
    2>/dev/null > "$dir/pallas_canon.mlir"

  sed -E 's/%[a-zA-Z_0-9]+/%X/g' "$dir/zig_canon.mlir"    > "$dir/zig_canon.norm.mlir"
  sed -E 's/%[a-zA-Z_0-9]+/%X/g' "$dir/pallas_canon.mlir" > "$dir/pallas_canon.norm.mlir"

  local zig_lines pal_lines raw_diff norm_diff
  zig_lines=$(wc -l < "$dir/zig_canon.mlir")
  pal_lines=$(wc -l < "$dir/pallas_canon.mlir")
  raw_diff=$(diff "$dir/zig_canon.mlir" "$dir/pallas_canon.mlir" | wc -l | tr -d ' ')
  norm_diff=$(diff "$dir/zig_canon.norm.mlir" "$dir/pallas_canon.norm.mlir" | wc -l | tr -d ' ')

  echo "==== $combo ============================================================="
  printf "  zig_canon.mlir       %4d lines\n" "$zig_lines"
  printf "  pallas_canon.mlir    %4d lines\n" "$pal_lines"
  printf "  raw diff             %4d lines  (includes SSA-name shuffling)\n" "$raw_diff"
  printf "  ssa-stripped diff    %4d lines  (true structural delta)\n" "$norm_diff"

  echo "  ---- op-count delta (zig vs pallas) ----"
  {
    echo "OP|ZIG|PAL"
    for op in \
      "scf.while" "scf.if" "scf.condition" "scf.yield" \
      "func.func" "func.return" \
      "tpu.matmul" "tpu.iota" "tpu.bitcast" "tpu.strided_load" \
      "tpu.memref_slice" "tpu.memref_squeeze" "tpu.memref_reshape" "tpu.memref_bitcast" \
      "tpu.enqueue_dma" "tpu.wait_dma2" "tpu.trace_start" "tpu.trace_stop" \
      "tpu.vector_store" "tpu.concatenate" \
      "vector.load" "vector.broadcast" "vector.shape_cast" "vector.multi_reduction" \
      "memref.load" "memref.store" \
      "math.exp" \
      "arith.constant" "arith.muli" "arith.addi" "arith.subi" "arith.divsi" "arith.remsi" \
      "arith.minsi" "arith.maxsi" "arith.shli" "arith.shrui" "arith.andi" "arith.extui" "arith.trunci" \
      "arith.cmpi" "arith.cmpf" "arith.select" "arith.index_cast" \
      "arith.mulf" "arith.addf" "arith.subf" "arith.divf" "arith.maximumf" \
      "arith.extf" "arith.truncf"; \
    do
      z=$(grep -c "$op" "$dir/zig_canon.mlir" || true)
      p=$(grep -c "$op" "$dir/pallas_canon.mlir" || true)
      if [[ "$z" != "$p" ]]; then
        echo "$op|$z|$p"
      fi
    done
  } | column -t -s '|' | sed 's/^/    /'
  echo
}

for combo in $COMBOS; do
  IFS=':' read -r head_dim kv_dtype <<<"$combo"
  run_combo "$head_dim" "$kv_dtype" || true
done
