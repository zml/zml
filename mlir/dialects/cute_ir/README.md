# CuTe dialects and Zig bindings

`mlir/dialects/cute_ir` is the CuTe dialect of nvidia-cutlass-dsl 4.8.0. The
dialect, attributes, public types and `cutegen` layout algebra come verbatim
from the CUTLASS release's `cutlass_compiler/` tree (`@cute_ir`,
pinned in `third_party/cute_ir`, one patch giving cutegen the
`?{div=N}` leaf); this tree carries what differs from it: `CuteTypes.cpp` with
the compiler's remaining types, attributes and enums and ratio strides
allowed, the compiler's attributes, types and enums beyond the release
(`CuteAttrsCompiler.td`, `CuteTypesCompiler.td`, `CuteEnums.td` and the
`../cute_nvgpu` definitions), and every operation of the compiler's registry in
`CuteOps.td` and the companion `CuteNVGPUOps.td`.

The companion dialects live in `../cute_nvgpu`, `../nvvm`, and `../cuda`,
with separate Bazel targets and Zig modules. NVIDIA's `nvvm` differs from
upstream's. The CuTe builder registers these dialects together with upstream
`llvm`.

Updating to a new DSL release means refreshing the compiler entry-point
offsets and regenerating the `.td` files from the release's Python bindings
and compiler. The C API (except the algebra attribute functions in `capi.cc`)
and the Zig types, attributes, enums and operation builders are generated
from those `.td` files and follow them.

## Zig

Depend on `//mlir/dialects/cute_ir` and import `mlir/dialects/cute_ir`.
`cute.registerDialect(registry)` registers `cute`.

Import each companion dialect explicitly and add its Bazel dependency:
`mlir/dialects/cute_nvgpu`, `mlir/dialects/nvvm`, and `mlir/dialects/cuda`.
CUDA and NVVM expose generated bindings in `.ops` and convenience builders
at the module root. NVIDIA's NVVM handle is `cute_nvvm`; its dialect cannot
share a registry with upstream NVVM.

```zig
const cute = @import("mlir/dialects/cute_ir");

const pointer = try cute.PtrType.get(ctx, .{
    .valueType = .float(ctx, .f32),
    .addressSpace = .smem,
    .alignment = 16,
});
const layout_attr = try cute.LayoutAttr.get(ctx, "4:1");
const layout = try cute.LayoutType.get(ctx, .{ .attr = layout_attr.attribute() });
const memref = try cute.MemRefType.get(ctx, .{
    .ptr = pointer.type_(),
    .layout = layout.type_(),
});
const alloc = cute.memref_alloc_smem(ctx, memref.type_(), loc).appendTo(block);
```

The bindings follow `mlir/dialects/README.md`. Algebra types take an algebra attribute, `cute.LayoutType.get(ctx, .{ .attr =
layout_attr })`; algebra attribute constructors call the C++ algebra parser
with the raw expression and `value()` returns the canonical text. NVGPU atom
types take their typed parameters, such as `.valType` and `.copyBits`;
`cute_nvgpu.typeFromPayload(T, ctx, "<f32>")` (from
`@import("mlir/dialects/cute_nvgpu")`) parses one from the text after its
mnemonic. For reading existing IR use MLIR's generic `Type.parse`.

Operation builders take the operands, then the result types, then the
attributes, then the location, and call `mlir.Operation.make`. Parameters
follow ODS names, with a trailing underscore where needed. Result parameters
append `_type` or `_types`; attributes take `*const mlir.Attribute`. Optional
operands, attributes and result types go in a trailing `opts` struct whose
fields default to null; pass `.{}` to omit them, and `&.{}` for empty variadic
operands. Dots in
mnemonics become underscores. Variable-length operand and result groups go
through `Operation.make`'s `.variadic` API, which writes the segment-size
attributes. Builders verify the created operations; to defer verification, use
`mlir.Operation.make` directly with `.verify = false`.

Builders of operations with regions (`cuda.kernel`) and of terminators
(`cuda.return`) do not verify on creation, since their blocks are completed
afterwards; the module is verified instead.

Each dialect exposes a `:capi` target and a package-local `capi.h`, independently
of Zig: `//mlir/dialects/cute_ir:capi`, `//mlir/dialects/cute_nvgpu:capi`,
`//mlir/dialects/nvvm:capi`, and `//mlir/dialects/cuda:capi`. C API declarations
and definitions are grouped by registration, attributes, and types in
`capi.h` and `capi.cc`. Constructors return a null handle for invalid parameters.

C++ implementations, headers, and TableGen sources live directly in each
dialect package. Generated `.inc` files are emitted into the same package
under Bazel's output tree. Virtual headers preserve upstream CuTe include
paths required by the release sources.

```sh
bazel test //mlir/dialects/cute_ir:test //mlir/dialects/cute_nvgpu:test \
  //mlir/dialects/nvvm:test //mlir/dialects/cuda:test
```

These tests only check the bindings: every bound operation is registered,
every type and attribute rebuilds from its getters, and a sample of the nvvm
and cuda attributes and types print the compiler's own text byte for byte
(`#cuda<device_attributes{…}>` and `#cuda<func_attributes{…}>` print with a
space the compiler omits; it parses both). The dialect itself is
tested where it is maintained. The dialects here do not implement the
compiler's semantic verification or lowering.
