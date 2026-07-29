# FlashInfer CUTLASS MoE development dependency

Build the three producer libraries in the sibling FlashInfer checkout:

```shell
cd ../flashinfer
bazel build -c opt //:flashinfer_cutlass_moe_so
```

ZML's local repository then exposes only the public C header and the three
shared libraries. Override the default sibling checkout when necessary:

```shell
export ZML_FLASHINFER_CUTLASS_MOE_ROOT=/path/to/flashinfer
```

The runtime loader chooses `sm90`, `sm100`, or `sm120` for each CUDA device.
No Torch, TVM-FFI, FlashInfer Python package, generated cubin repository, or
TRTLLM Gen runtime files are linked into ZML.

`//zml:flashinfer_cutlass_moe` exposes the after-routing BF16 and NVFP4 custom
calls. BF16 is available in all three libraries. NVFP4 W4A4 is selected only on
SM100/SM120, takes ordinary packed E2M1 expert weights, and requires the E4M3
block scales to have been converted to FlashInfer's
`block_scale_interleave` layout. Unlike TRTLLM Gen, the CUTLASS path does not
shuffle weights to BlockMajorK.
