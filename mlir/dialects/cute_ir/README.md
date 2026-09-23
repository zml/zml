# CuTe dialects and Zig bindings

`mlir/dialects/cute_ir` is the CuTe dialect of nvidia-cutlass-dsl 4.8.0. The
dialect, attributes, public types and `cutegen` layout algebra come verbatim
from the CUTLASS release's `cutlass_compiler/` tree (`@cute_ir`,
pinned in `third_party/cute_ir`, one patch giving cutegen the
`?{div=N}` leaf); this tree carries what differs from it: `CuteTypes.cpp` with
the compiler's remaining types, attributes and enums and ratio strides
allowed, the compiler's attributes, types and enums beyond the release
(`CuteAttrsCompiler.td`, `CuteTypesCompiler.td`, `CuteEnums.td` and the
`CuteNVGPU*.td` files), and every operation of the compiler's registry in
`CuteOps.td` and `CuteNVGPUOps.td`.

Updating to a new DSL release means refreshing the compiler entry-point
offsets and regenerating the `.td` files from the release's Python bindings
and compiler. The C API (except the algebra attributes, `CuteAttributes.*`)
and the Zig types, attributes, enums and operation builders are generated
from those `.td` files and follow them.

## Zig

Depend on `//mlir/dialects/cute_ir` and import `mlir/dialects/cute_ir`, or use
`@import("mlir/dialects").cute` through the dialect aggregator.

`cute.registerDialects(registry)` registers both `cute` and `cute_nvgpu`.
Cute builders, types, and attributes live directly in `cute`; CuteNVGPU builders
and types live in `cute.nvgpu`.

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

Type and attribute constructors call the C++ `get`/`getChecked` functions,
not a parser. `InitArgs` fields are the ODS parameter names in lower camel
case; getters are the C++ accessor names without underscores. Optional and defaulted parameters have defaults in `InitArgs`. Enums
are Zig enums with the compiler's values (the C API takes the integer value).
`get` returns `error.InvalidMlir` for invalid parameters or mismatched contexts.
`type_()` converts a typed wrapper to `*const mlir.Type` for operation builders.
Algebra types take an algebra attribute, `cute.LayoutType.get(ctx, .{ .attr =
layout_attr })`; algebra attribute constructors call the C++ algebra parser
with the raw expression and `getValue()` returns the canonical text. NVGPU atom
types take their typed parameters, such as `.valType` and `.copyBits`;
`cute.nvgpu.typeFromPayload(T, ctx, "<f32>")` parses one from the text after
its mnemonic. For reading existing IR use MLIR's generic `Type.parse`.

Operation builders take the operands, then the result types, then the
attributes, then the location, and call `mlir.Operation.make`. Parameters
follow ODS names, with a trailing underscore where needed. Result parameters
append `_type` or `_types`; attributes take `*const mlir.Attribute`. Pass `null`
for absent optional parameters and `&.{}` for empty variadic ones. Dots in
mnemonics become underscores. Variable-length operand and result groups go
through `Operation.make`'s `.variadic` API, which writes the segment-size
attributes. Builders verify the created operations; to defer verification, use
`mlir.Operation.make` directly with `.verify = false`.

The C API is also available independently of Zig from `lib/CAPI:cute` and
`lib/CAPI:cute_nvgpu`, via `cute_ir-c/Dialect/Cute.h` and `CuteNVGPU.h`;
constructors return a null handle for invalid parameters.

```sh
bazel test //mlir/dialects/cute_ir:test
```

These tests only check the bindings: every bound operation is registered and
every type and attribute rebuilds from its getters. The dialect itself is
tested where it is maintained. The dialects here do not implement the
compiler's semantic verification or lowering.
