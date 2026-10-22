#!/usr/bin/env bash
# Build gt/layout_gt (DeepGEMM's layout/scheduler headers + the vLLM DeepGEMM fork's host
# heuristics and buffer-layout functions + the kernel's SharedStorage, extracted verbatim),
# run it, and regenerate ../layout_test.zig.
# Usage: gen_layout_gt.sh <vllm>/vllm/third_party/deep_gemm/include <vLLM DeepGEMM fork checkout> <work dir>
set -euo pipefail
INC="$1"; FORK="$2"; WORK="$3"; HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK"
awk '/^static MmaKind parse_mma_kind/,/^static MegaMoEConfig get_mega_moe_config/' "$FORK/csrc/jit_kernels/heuristics/mega_moe.hpp" | sed '$d' > "$WORK/heur_extract.inc"
awk '/^static int checked_int/,/^static std::tuple<int64_t, std::vector<int64_t>> get_symm_buffer_size_for_mega_moe/' "$FORK/csrc/apis/mega_moe.hpp" | sed '$d' > "$WORK/api_extract.inc"
awk '/    struct SharedStorage \{/,/^    \};/' "$INC/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh" > "$WORK/smem_extract.inc"
cp "$HERE/layout_gt.cu" "$WORK/"
/usr/local/cuda/bin/nvcc -std=c++20 -O1 --expt-relaxed-constexpr -I "$INC" -o "$WORK/layout_gt" "$WORK/layout_gt.cu"
"$WORK/layout_gt" > "$WORK/layout_gt.json"
python3 "$HERE/gen_layout_test.py" "$WORK/layout_gt.json" "$HERE/../layout_test.zig"
