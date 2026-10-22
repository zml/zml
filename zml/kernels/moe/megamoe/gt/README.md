# Ground truth for the MegaMoE kernel building blocks

Everything here produces DeepGEMM's own numbers for the tests next door; none of it is built by
Bazel.

| File | Produces | Consumed by |
|---|---|---|
| `gen_layout_gt.sh`, `layout_gt.cu`, `gen_layout_test.py` | DeepGEMM's layout/heuristics/`SharedStorage` numbers (its headers and host functions compiled with nvcc) | `../layout_test.zig` (generated) |
| `dg_layout_check.py` | Cross-check of `layout_gt.json` against DeepGEMM's compiled `_C` (buffer sizes, BLOCK_M) on a GPU | manual |
| `numerics_ref.{cu,py}` | L1 epilogue, L2 packing, combine; stmatrix and TMEM register maps; from DeepGEMM's device functions | `../gpu_test.zig` |
| `tma_ref.{cu,py}` | 2-CTA TMA loads/stores and bulk copies with DeepGEMM's `tma::copy` on DeepGEMM-encoded descriptors | `../gpu_test.zig` |

`numerics_ref.py` and `tma_ref.py` take the DeepGEMM include directory
(`<vllm>/vllm/third_party/deep_gemm/include`) and an output directory, need torch and
`CUDA_HOME`, and run on one GPU. Point `MEGAMOE_REF_DIR` at the output directory when running
`//zml/kernels/moe/megamoe:gpu_test`.
