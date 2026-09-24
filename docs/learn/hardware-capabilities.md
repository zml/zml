# Hardware capabilities

`platforms.ComputeCapability` identifies hardware with a tagged union:

```zig
const cc = platform.computeCapability() orelse return error.UnknownHardware;
for (cc.matmulCapabilities()) |operation| {
    // Inspect operation.instruction, formats, native_shapes and execution.
}
```

`Device.hardware` retains the backend's device name and raw attributes alongside
an optional recognized `capability`. Its strings and attribute slices borrow PJRT
storage for the client's lifetime. `Device.computeCapability()` and
`platform.computeCapability()` return `?ComputeCapability`. The platform value is
null if the addressable devices are unknown, mixed, or empty; unknown devices
never inherit a peer's capabilities. An unrecognized architecture no longer
prevents initialization.

Hardware support decisions must use exhaustive switches without `else`, either
on the device enum, on `architecture()`, or through `matmulCapabilities()`.
Keep backend support switches beside the backend implementation. Adding a device
must require an explicit mapping; adding an architecture must require every
architecture-level decision to be reviewed. Avoid enum equality and numeric SM
comparisons for hardware-capability decisions. Unknown hardware needs an explicit
fallback. A backend-wide target check such as `platform.target == .cuda` does not
need a redundant switch over every CUDA device.

Explicit device equality is appropriate for tuning measured on a particular device
(such as GB300 attention tile sizes), with a comment explaining that scope. It must
not stand in for a capability or backend support query.

Every matrix path identifies an instruction form, including its encoding/layout
generation, formats, accumulator, native shapes, and execution requirements.
`supportsMatmulInstruction()` queries the forms available on the device.
CDNA1 can compose a BF16 16×16×16 tile, but lacks
`mfma_f32_16x16x16bf16_1k`. Fly MoE and sparse MLA require that exact form and
use the same identifier when constructing their Fly atom. Forms that gfx940
removed, the gfx908 two-element BF16 and 16×16×16 INT8 MFMA, stay on CDNA1 and
CDNA2 only; CDNA3 and CDNA4 list `gfx940_mfma_i32_i8` instead, so a newer
generation never inherits an older one's table.

`executionCapabilities()` independently describes supported hardware wave modes:
CDNA have wave64; RDNA supports wave32 and wave64; CUDA has warp32. Unknown
or unmodeled execution information returns null. Intel DPAS SIMD width is recorded
on the instruction; it is not treated as a general GPU thread-count guarantee.
Fly currently selects wave32 on RDNA and wave64 on CDNA. The ROCm execution
test probes `rocdl.wavefrontsize` and the actual block size to check that the
compiler agrees. Its compiler mapping and kernel tuning remain backend concerns.
`zml.kernel.fly.isAvailable()` requires a recognized ROCm device with a known
wave mode, so launch sizing never guesses; `zml.kernel.fly.supportsMmaAtom()`
additionally admits each Fly atom family per validated architecture.

Hardware recognition, native matrix instructions, and backend execution support
are separate decisions:

- An unknown architecture has a null capability and retains its raw identity.
- A known device without modeled matrix instructions, such as Meteor Lake or
  Radeon VII, has a non-null capability and an empty `matmulCapabilities()` table.
  Ordinary vector arithmetic can still implement matmul.
- A matching matrix path describes hardware arithmetic only. The installed
  compiler or kernel backend must support the requested operation, formats,
  layouts, and resource requirements. Generic operations can be attempted on
  unknown hardware; specialized paths must explicitly check their prerequisites.

The current selection rules follow the implementations and pinned dependencies:

| Decision | Requirement and evidence |
| --- | --- |
| FA2 routing | CUDA-wide backend gate, including unknown CUDA identities. The [pinned FA2 package](https://github.com/zml/flash-attention/blob/v0.0.6-rc9/BUILD.bazel) builds SM80 + PTX and Blackwell kernels; this gate does not claim that historical pre-Ampere GPUs can execute those binaries. |
| FA3 routing | Hopper. The same package builds FA3 only for `sm_90a` and disables its SM8x path. [NVIDIA documents that `compute_90a` is incompatible with Blackwell](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/). A source-level `arch >= 90` condition is insufficient to extend the packaged backend. |
| Triton routing | CUDA, ROCm, OneAPI. Metal, TPU, Neuron and CPU cannot run this Triton backend. Sparse MLA and MXFP4 MoE reject these unsupported targets. |
| Triton FP4 × FP8 MoE | SM100, SM101, SM103, SM110. The [pinned lowering](https://github.com/triton-lang/triton/blob/c05aa65087a9a1a6b8a08fdbb474aba834d5cddf/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp) accepts mixed scaled operands on SM100–119; its SM12x path accepts only FP4 × FP4 or FP8 × FP8. Native FP4 support alone is insufficient. |
| CUTLASS MoE | SM90, SM100, SM103, SM120, with loaded runners. The [pinned C API](https://github.com/zml/flashinfer/blob/cutlass-moe-v0.2.0/capi/flashinfer_cutlass_moe.cu) checks exact major/minor matches. NVFP4 additionally requires a native NVFP4 matrix path. |
| Fly BF16 MoE / sparse MLA | CDNA3. These kernels use wave64 and `mfma_f32_16x16x16bf16_1k`. The [Fly atom lowering](https://github.com/ROCm/FlyDSL/blob/c62159d0c18e232794a1903f192fc094148f4a43/lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp) emits this instruction, which [LLVM enables on gfx90a+](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/VOP3PInstructions.td), but the `FlyROCDL/CDNA3` atoms have only been validated on MI300. `supportsMmaAtom()` keeps CDNA2 and CDNA4 on Triton until they are validated. CDNA1's smaller BF16 atom and RDNA's wave32 WMMA do not meet this contract. Shape and layout restrictions still apply. |
| Block-FP8 input quantization | CUDA/ROCm plus a modeled native FP8 path. Select OCP E4M3FN or FNUZ from `matmulCapabilities()`; retain unquantized inputs when neither is available. Pre-Ada CUDA has no modeled FP8 path, so inputs stay BF16 there. |
| Generic Fly tiled-MMA example | Its concrete atom is f32 16×16×4 MFMA on CDNA3, f16 16×16×16 WMMA on RDNA3+. The example runs only where `supportsMmaAtom()` admits the atom family; this is not a claim that every operation in the `CDNA3` dialect is valid on every CDNA generation. |

The Triton fused-experts implementation accepts BF16, F16, F32 or E4M3 FP8
weights, plus packed MXFP4 through `autoMxfp4()`. Its auto-selection rejects E8M0
weight tensors, which only ever hold scales. Metal and TPU retain their own type
contracts. Sparse MLA's `Backend.auto()` returns an error when no backend is
available.

These are source-verified eligibility rules. They do not replace GPU execution
validation, particularly for the newly admitted SM110 Triton MXFP4 path.

OpenXLA OneAPI builds can report `compute_capability=unknown`. Discovery first
uses optional integer plugin attributes `ip_version` or `pci_device_id` (with
`vendor_id` when available), then a recognized capability string, then an exact
known product name. `pci_device_id` is a PCI product identifier, **not** PJRT's
logical device ID or local hardware ordinal. Unrecognized numeric identifiers or
explicit future architecture names remain unknown instead of falling back to a
potentially misleading product name. Existing plugins that do not expose numeric
identifiers leave these fields null; this change does not add Level Zero queries
to those plugins. Generic `Intel(R) Arc(TM) Graphics` names remain unknown without
additional identifying attributes.

The Intel catalog covers DG2 (including Arc Pro A and Flex), PVC, BMG, Meteor Lake,
Arrow Lake H, Lunar Lake, Panther Lake, and Wildcat Lake. Arrow Lake U/S use `.mtl`
graphics; `.arl_h` identifies the XMX-equipped variant. B370/B390 map to `.ptl`,
while discrete B570/B580 and Pro B products map to `.bmg`. DG2/ARL-H DPAS uses
SIMD8; PVC, BMG, LNL, PTL, and WCL use SIMD16. The tables currently model their
FP16, BF16, and INT8 paths, not every available instruction precision.

The CUDA catalog starts at SM70 (Volta). The AMD catalog starts at CDNA1
(`gfx908`) and RDNA2 (`gfx1030`–`gfx1036`), retaining newer CDNA/RDNA families.
Pre-SM70 CUDA, GCN5, and RDNA1 devices are outside the catalog and parse as
unknown hardware. Recent non-matrix hardware, including RDNA2 and Intel Meteor
Lake, remains recognized with an empty matrix table. In particular,
`gfx1151` identifies Strix Halo / Ryzen AI Max, separately from Strix Point
(`gfx1150`), Krackan Point (`gfx1152`), and Radeon 820M (`gfx1153`). Recognition does
not assert support by the installed ROCm release. Product-to-target mappings come
from [AMD's target lookup](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#gfx-target-lookup-table).

Numeric SM ordering is not a feature compatibility test.

The vendor enums expose an `Architecture` enum and `architecture()`: for example,
CUDA `.sm80`, `.sm86` and `.sm87` return `.ampere`, ROCm `.gfx1100` and
`.gfx1101` return `.rdna3`; OneAPI `.dg2`, `.pvc`, and `.bmg` return `.xe_hpg`,
`.xe_hpc`, and `.xe2_hpg`. Switch on the vendor capability's
`architecture()` when choosing behavior shared by an architecture. Each CUDA
SM's documentation names a representative GPU or embedded module, based on NVIDIA's
[current](https://developer.nvidia.com/cuda/gpus) and
[legacy](https://developer.nvidia.com/cuda/gpus/legacy) GPU tables.

Architecture families group products; they do not imply identical instruction
support or membership in the same CUDA `f` compiler-target family. In particular, Blackwell includes SM100/103, Thor SM110 (formerly
SM101), and SM120/121, whose matrix constraints differ. Continue querying the
exact compute capability when generating kernels.

## Selecting a matmul kernel

`cc.matmulCapabilities()` describes native dense matmul paths using these fields:

- `formats`: allowed types for either operand; every A/B combination within the
  set is legal. Separate records prevent mixing BF16 with FP16 or NVFP4 with MXFP4.
- `accumulator`: the accumulation precision for the path.
- `instruction`: a mandatory encoding/layout form, shared with DSL adapters.
- `native_shapes`: legal instruction M/N/K shapes, not arbitrary multiples.
- `execution`: subgroup/SIMD width, issue scope, and CTA grouping. A warpgroup
  instruction is issued by 128 threads; tcgen05 can be issued by one thread.
  Neither number changes CUDA's 32-thread warp width.

Hopper FP8 is modeled as both the `mma_fp8_f32` mma.sync form and the warpgroup
`wgmma_fp8_f32` form (64×N×32). tcgen05 forms list M=128 with `cta_group::1`
only; the M=64 and `cta_group::2` M=256 shapes are not modeled yet.
`tcgen05_nvfp4_k96_f32` is the SM103 `scale_vec::6X` variant with K=96.

`cc.supportsMatmulFormat(.mxfp4)` is a quick format query. Kernel selection must
also check the operand pair, accumulator and tile:

```zig
const hw = zml.platform.capabilities;
// If unknown, use a generic path or let this specialized operation decline.
const cc = platform.computeCapability() orelse return error.UnknownHardware;
const tile: hw.Tile = .{ .m = 128, .n = 128, .k = 128 };
for (cc.matmulCapabilities()) |cap| {
    if (cap.accumulator != .f32) continue;
    if (!cap.supports(.mxfp4, .mxfp8_e4m3, tile)) continue;
    // The hardware has a suitable arithmetic path.
    // Now check the chosen backend's support and tile/resource restrictions.
}
```

`cap.supportsNative(a, b, accumulator, shape)` checks an exact native shape.
`cap.supports(a, b, tile)` checks whether the arithmetic can be composed from
one of the listed shapes; the caller also checks accumulation precision.
Neither query establishes backend execution support or selects an optimal tile.
For example, gfx950 scaled MFMA supports 16×16×128 and 32×32×64, which does not
imply a native 16×16×64 instruction.

The independent `platforms/matmul` module owns shared instruction descriptions;
capabilities maps devices to them. Fly's adapter owns MLIR types and supports
only explicitly implemented instruction forms. Its tiled-MMA example owns
block K, atom tiling, and copy strategy. Triton tuning is unchanged.

The catalog covers useful native dense paths, not sparse operations, every
accumulator variant, conversions, or software dequantization. For example,
gfx942 can implement MXFP4 weight-only matmul by dequantizing, without native
MXFP4 arithmetic. A missing format does not prohibit such a kernel.

## Scaling and backend constraints

`Format.scaling()` gives the scale encoding and block length along K, with a
scale for each row of A or column of B. MX formats use UE8M0 with blocks of 32;
NVFP4 uses UE4M3 with blocks of 16. NVFP4's tensor-wide scale is handled by the
algorithm separately. `Format.bits()` describes the logical element width.

cuTile, Triton and FlyDSL own instruction lowering, packing, scale layouts,
synchronization and compiler-target selection. Each backend must additionally
check its supported operand layouts, alignment, tile restrictions and resource
requirements. Hardware support alone does not guarantee a backend can generate
the kernel. Shared-memory limits should come from device/backend resource
queries when selecting a launch configuration; they are not inferred from
these arithmetic tile multiples.

## Sources and tests

The hardware tables derive from [NVIDIA's PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/),
[LLVM's AMD instruction definitions](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/VOP3PInstructions.td),
and [Intel Triton's DPAS lowering](https://github.com/intel/intel-xpu-backend-for-triton).
Instruction variants whose shapes are multiples of an existing path are omitted
when they do not add format or tile coverage. The catalog does not rank paths by
throughput.

Run `./bazel.sh test //platforms:capabilities_test //zml:test` to check parsing,
format boundaries, operand pairs, scaling, tile constraints and existing callers.
