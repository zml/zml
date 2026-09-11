# CuTe Zig Bindings

Depend on `//mlir/dialects/cute_ir` and import `mlir/dialects/cute_ir`, or use
`@import("mlir/dialects").cute` through the dialect aggregator.

`cute.registerDialects(registry)` registers both `cute` and `cute_nvgpu`.
Cute builders, types, and attributes live directly in `cute`; CuteNVGPU builders
and types live in `cute.nvgpu`. Cute bindings are in `cute_ir.zig`, and CuteNVGPU
bindings are in `cute_nvgpu.zig`. Public and recovered operations belong to the same `cute`
dialect, without a separate private Zig namespace.

```zig
const cute = @import("mlir/dialects/cute_ir");

const pointer = try cute.PtrType.get(ctx, .{
    .valueType = .float(ctx, .f32),
    .memorySpace = .string(ctx, "smem"),
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

The checked-in operation builders follow the pinned OSS `CuteOps.td` and the
local `CuteOpsPrivate.td` and `CuteNVGPUOps.td` schemas. Update the bindings when
those schemas change. There is no build-time Zig generator. Attribute and type
wrappers call `CuteAttributes.h/.cpp`, `CuteTypes.h/.cpp`, and
`CuteNVGPUTypes.h/.cpp`. C++ dialect definitions still use TableGen.

Type constructors call the actual C++ `get`/`getChecked` functions, not a type
parser. Parameters follow ODS names; getters follow the C++ accessor names.
`get` returns `error.InvalidMlir` for invalid parameters or mismatched contexts.
`type_()` converts a typed wrapper to `*const mlir.Type` for operation builders.

The C API is also available independently of Zig from `lib/CAPI:cute` and
`lib/CAPI:cute_nvgpu`, via `cute_ir-c/Dialect/Cute.h` and `CuteNVGPU.h`.

Algebra types take an existing algebra attribute, for example
`cute.LayoutType.get(ctx, .{ .attr = layout_attr })`. Attribute constructors
call the C++ algebra parser with the raw expression, without building or parsing
MLIR assembly. `getValue()` returns canonical algebra text owned by the context.
The `algebraAttribute` and `algebraType` helpers dispatch to these C bindings.
Recovered NVGPU payload types take a `StringAttr` containing the
complete payload, including angle brackets, such as `<f32>`; the current schemas
preserve those architecture-specific contents without interpreting them.
Type construction uses the typed C API directly, with no temporary MLIR assembly
strings. For explicitly reading existing IR, use MLIR's generic `Type.parse`
or `mlirTypeParseGet`; there are no CuTe-specific parsing helpers.

Operation builders take explicit parameters, followed by the location, and call
`mlir.Operation.make`. Parameters follow ODS names, with a trailing underscore
where needed to avoid Zig name collisions. Result parameters append `_type` or
`_types`; attributes take `*const mlir.Attribute`. Pass `null` for absent
optional parameters and `&.{}` for empty variadic parameters.
Dots in operation mnemonics become underscores, except for the private
`cute.@"tuple.product"` and `cute.@"tuple.product_each"` builders, whose literal
names distinguish them from the public `tuple_product` and `tuple_product_each`.
Builders compute variadic segment sizes as dense-array attributes and verify
the created operations. To defer verification, use `mlir.Operation.make`
directly with `.verify = false`.

Builders pass variable-length operand and result groups through
`Operation.make`'s `.variadic` API, without temporary heap allocations or a
fixed operand-count limit in the Zig builder. MLIR still allocates its own
operation storage, and the dialect/compiler determines legal fragment sizes.

Tests construct public/private/NVGPU operations, verify their registered schemas,
and round-trip MLIR bytecode into a fresh context:

```sh
bazel test //mlir/dialects/cute_ir:test //mlir/dialects/cute_ir/test:all
```

These tests do not invoke the NVIDIA compiler or execute GPU kernels. Recovered
private schemas do not implement NVIDIA's complete semantic verification or
lowering pipeline.

## Compiler Offset Inspection

`tools:cute-objdump-offsets` invokes GNU objdump and scans its disassembly for
the eight known private compiler entry-point signatures. It requires Perl
(core modules only) and a GNU objdump supporting AArch64; it neither uses Python
nor loads the library. For example:

```sh
bazel run //mlir/dialects/cute_ir/tools:cute-objdump-offsets -- \
  --input /absolute/path/to/_cutlass_ir.so --output /tmp/cute-offsets.json
bazel test //mlir/dialects/cute_ir/tools:cute-objdump-offsets-test
```

Use `--objdump /path/to/aarch64-linux-gnu-objdump` or `OBJDUMP` to select the
disassembler. The JSON report records the binary SHA-256, symbol availability,
all candidate addresses, and matching instructions. Addresses are ELF virtual
addresses, not file offsets. Missing or ambiguous matches produce exit status
1 and no `offsets` table, while retaining candidates for inspection.

The signatures come from the native offset generator in `cute-native`, targeting
the 4.6.2 cu13/AArch64 package. This is an independent disassembly-based check
of those signatures, not fresh semantic identification or proof of ABI
compatibility with other builds. Some signatures contain build-specific PC-relative
instructions or neighboring code. This inspection tool does not replace the
native compiler bridge's build-time header generator or runtime binary checks.
