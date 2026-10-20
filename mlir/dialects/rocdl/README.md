# ROCDL

Depend on `//mlir/dialects/rocdl` and import `mlir/dialects/rocdl` for LLVM's
AMDGPU intrinsic dialect. The package links LLVM's `ROCDLDialect` and
`CAPIROCDL`. `fly_rocdl` is a separate dialect and package.

The pinned LLVM revision supplies 374 operations, 15 attributes, and 12 enums.

- `ops` provides every operation builder. Names replace dots with underscores,
  such as `ops.raw_buffer_load` and `ops.mfma_f32_16x16x4f32`. Arguments are
  operands, explicit result types, required attributes, an `opts` struct when
  there are optional attributes, and the location, preceded by the MLIR context.
  Attribute names match ODS. Optional and default-valued attributes are fields
  of `opts` with `null` defaults; pass `.{}` to omit them and let MLIR supply
  its defaults. Builders verify the resulting operation.
- Enum types and attribute wrappers are available at the module root and in
  `enums`/`attributes`. Attribute constructors return `error.InvalidMlir` for
  invalid enum values or flag bits. Bit enums support `combine` and `contains`.
- `TargetAttr` constructs and reads `#rocdl.target`, including optimization
  level, triple, chip, features, ABI, flags, and link files. Defaults match LLVM.
  Invalid fields and attributes from another context are rejected.
- `annotations` supplies function, module, and memory annotations. Apply
  function annotations to `llvm.func`, and `xnack`/`sramecc` to LLVM-compatible
  modules. A missing module option preserves the target default; `false`
  explicitly disables the option.
- `types` supplies the builtin vector types named in ROCDL's TableGen records
  and LLVM pointers in ROCDL address spaces. ROCDL defines no independent
  MLIR types.
- `registerDialect` registers ROCDL; loading it also loads its LLVM dependency.
  The existing `cvtPkF32Fp8` convenience builder remains available.

```zig
const rocdl = @import("mlir/dialects/rocdl");

const policy = try rocdl.Gfx942CachePolicyAttr.get(ctx, .combine(&.{ .sc0, .nt }));
const load = rocdl.ops.raw_buffer_load(
    ctx, resource, offset, scalar_offset,
    mlir.Type.float(ctx, .f32), .{ .aux = policy.attribute() }, location,
);
```

The `:capi` target exposes registration and attribute constructors, predicates,
and getters through `capi.h`. Declarations and implementations are grouped by
registration, attributes, and target configuration. Getters require a handle
accepted by the matching predicate. Contexts own all returned attributes and
strings. There are no additional C operation wrappers: Zig builders use MLIR's
standard operation API.

Tests compare the generated operation list with the loaded dialect's registered
operation count, check every operation name, round-trip every named enum value,
reject invalid enum/target inputs, and verify and reparse representative barrier,
scheduling, buffer, and matrix operations in an LLVM function. The tests require
no AMD GPU or ROCm runtime.
