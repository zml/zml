# MLIR dialects

One Bazel package per MLIR dialect that ZML emits. The rules come in three
tiers:

- **Guarantees** hold for every package. Tests and the conformance checks
  below enforce them, and callers rely on them.
- **Conventions** are the defaults. Follow them in new code; existing code
  that differs is fine unless it confuses callers.
- **Exceptions** list the known, justified departures from the conventions.

References: `cuda/`, `nvvm/` and `cute_nvgpu/` (generated from ODS, with
their own TableGen and C API), `gpu/`, `arith/` and `func/` (upstream
dialects, with a small C API for what upstream lacks).

## Guarantees

- **Registration.** The package exposes `registerDialect(registry)`, which
  calls `registry.registerDialect("<handle>")` with a string literal, where
  `<handle>` is the second argument of
  `MLIR_DECLARE_CAPI_DIALECT_REGISTRATION`. Tests and consumers register
  through `<pkg>.registerDialect`, never through raw strings or
  `DialectHandle.fromString`. A DSL exposes `registerDialects(registry)`,
  which calls the `registerDialect` of each package it emits.
- **Operation list.** A package that builds operations has, in its
  `test.zig`, a sorted `const operation_names: []const []const u8` naming
  every operation its builders create. `test "every bound operation is
  registered"` checks each name against the registry, and checks that
  unregistered dialects are not allowed. A builder for an operation the pinned dialect does not register
  does not exist; a DSL method builds it from the operations the Python DSL
  lowers it to instead.
- **Bindings, not text.** A dialect package builds attributes and types
  through C API constructors (`get`/`getChecked`), never by printing and
  parsing text. A type or attribute the C API lacks gets a binding (a C
  constructor and a Zig wrapper) before anything uses it. `mlir.Type.parse`
  and `mlir.Attribute.parse` are for tests. Consumers use the bindings too.
- **Everything compiles.** `test.zig` runs `std.testing.refAllDecls` on the
  package and on each of its wrapper types, so no declaration hides behind
  lazy analysis.
- **Argument order.** A builder takes `ctx`, the required operands, the
  result types (omitted when inferred), the required attributes, then the
  optional arguments (an `opts` struct), then `location`, which is always
  last.
- **Verification.** A builder verifies the operation it creates. One that
  cannot (its region is filled later, or it only verifies inside a parent)
  passes `.verify = false` and says so in its doc comment.
- **C API safety.** Every declaration is `MLIR_CAPI_EXPORTED`. Constructors
  load the dialect, return a null handle for a null context, and verifying
  constructors (`getChecked`) return a null handle for invalid input. No
  `alwayslink` and no static initializers: the Zig side calls the
  registration function by name, so the linker keeps what it uses.
- **Hygiene.** LLVM file banners and include-path guards (below), clean under
  `mlir/.clang-format` and the Bazel toolchain's `zig fmt` (0.17), tests only
  in `test.zig`, and the BUILD essentials: `capi` and the `zig_library`
  tagged `manual`, an explicit `srcs` list in `td_files`, no `glob(`.

## Conventions

### Package layout

- `mlir/dialects/<pkg>/` is imported as `@import("mlir/dialects/<pkg>")`,
  main file `<pkg>.zig`. Generated builders go in `ops.zig`, exposed as
  `<pkg>.ops`, with hand-written helpers in `<pkg>.zig`.
- The C API is `capi.h` / `capi.cc`. Dialects written here use
  `<Dialect>{Dialect,Ops,Types,Attributes,Enums}.{td,h,cpp}`; vendored
  sources keep their upstream file names.
- A generated file starts with a header naming its source and saying it is
  generated. The package that generates it has a `REGENERATING.md` of agent
  instructions (sources, generated vs maintained files, steps, checks) that
  points here for conventions. Generator scripts are not committed.

### BUILD.bazel

Targets in this order: `capi` (`cc_library`, deps `:dialect`, `:headers`,
`@llvm-project//mlir:CAPIIR`), the `zig_library` (public, deps `:capi`,
`//mlir`), `test` (`zig_test`, `main = "test.zig"`, plus `:capi` when
`test.zig` imports `c`), then `td_files` (`includes = ["/"]`, direct td
deps such as `@llvm-project//mlir:OpBaseTdFiles`), the `gentbl_cc_library`
targets the dialect needs (`dialect_inc_gen`, `enums_inc_gen`,
`attrs_inc_gen`, `types_inc_gen`, `ops_inc_gen`), `headers` and `dialect`.
Every dep a source includes or imports is listed directly. Format with
`./tools/buildifier.sh`.

### C and C++

- The first line is an 80-column LLVM banner:
  `//===- capi.h - C API for NVIDIA's cuda dialect -------------------*- C -*-===//`.
  The tag is `*- C -*-` for C headers, `*- C++ -*-` for C++ headers and
  `*- tablegen -*-` for `.td`; `.cc`/`.cpp`/`.inc` banners end in `---===//`.
  An optional `//` description follows, then a blank line.
- The guard is the include path in upper case with `/` and `.` as `_`, no
  `ZML_` prefix (`MLIR_DIALECTS_CUDA_CUDADIALECT_H`, `..._TD`), closed with
  `#endif // <GUARD>`.
- Includes follow LLVM order: own header, package headers, `mlir/` and
  `llvm/`, then system headers. `capi.h` includes `mlir-c/IR.h` (which
  provides `stdbool` and `stdint`) and the upstream `mlir-c/Dialect/<X>.h` it
  extends, inside `extern "C"` under `#ifdef __cplusplus`.
- Hand-written C APIs use fixed-width integers (`uint32_t`, `int64_t`),
  parameters `MlirContext context`, `MlirAttribute attr`, `MlirType type`,
  and C++ locals `MLIRContext *ctx`. `capi.cc` uses `static` helpers, not
  anonymous namespaces, and one static helper loads the dialect
  (`loadGPU` in `gpu/capi.cc`).

### Names and where they come from

`<C>` is the dialect's C prefix, the first argument of
`MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(<C>, <handle>)`: `Cuda`, `CuteNVGPU`,
`Triton`, `ROCDL`. Everything else is named as ODS names it:

| ODS | C API | Zig |
|---|---|---|
| `AttrDef<D, "Name">` (class `NameAttr`) | `mlirAttributeIsA<C>Name`, `mlir<C>NameAttrGet` | `NameAttr`, `NameAttr.get(ctx, .{ ... })` |
| `TypeDef<D, "Name">` (class `NameType`) | `mlirTypeIsA<C>Name`, `mlir<C>NameTypeGet` | `NameType` |
| parameter `$elementType` | `mlir<C>NameAttrGetElementType` | field `.elementType`, accessor `elementType()` |
| list parameter `$archs` | `...GetNumArchs`, `...GetArchs(attr, pos)` | `numArchs()`, `arch(i)` |
| `EnumAttr` / `I32EnumAttr` | `...AttrGet(context, value)`, `...AttrGetValue` | Zig enum, `XAttr.get(ctx, .{ .value = ... })`, `value()` |
| op mnemonic `fast_sin.f32` | generic operation API | builder `fast_sin_f32` |

- Predicates have no `Attr`/`Type` suffix (`mlirTypeIsACudaEvent`).
- C getters keep MLIR's `Get<Param>`. Zig accessors are nouns, as in the rest
  of ZML (`value()`, `elementType()`, `numAxes()`, `axis(i)`), never
  `get`-prefixed. Predicates keep `is`/`has`.
- Enum fields use the IR spelling (`@"and"`, `@"false"`).
- A builder is named after its mnemonic with dots turned into underscores,
  case kept. Its parameters are the ODS operand and attribute names, and its
  result types `<result>_type` (`<result>_types` when variadic).
- A name that is a Zig keyword, or that collides with a member, a declaration
  of the file or a builder local, takes a trailing underscore (`for_`,
  `return_`, `ptr_()`, `format_()`). It is quoted only when that name is taken
  too (`TupleType.@"type"(i)`, next to the `type_()` upcast).
- Alternative constructors are `mlir<C>NameAttr<Form>Get` in C, as MLIR's
  `mlirDenseElementsAttrSplatGet`, and `get<Form>(ctx, ...)` in Zig.

### Zig API

```zig
pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("cuda");
}

/// `#cuda.executable`: Format of a compiled executable
pub const ExecutableAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaExecutable;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;
    pub const InitArgs = struct { format: ExecutableFormat };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self { ... }
    pub fn attribute(self: *const Self) *const mlir.Attribute { ... }
    pub fn format_(self: *const Self) ExecutableFormat { ... }
};
```

- Wrappers use `mlir.Methods` with `c.MlirAttribute` or `c.MlirType`, upcast
  with `attribute()` or `type_()`, and build with `get(ctx, InitArgs)`, a
  struct even for one field. `get` returns `mlir.Error!*const X` when the C
  getter can return null, a plain pointer otherwise.
- Free helpers are `<x>Attr(ctx, ...)` and `<x>Type(ctx, ...)`.
- Optional builder arguments go in `opts`: generated builders declare it
  inline (`opts: struct { ... }`), hand-written ones may name it `<Op>Opts`.
  Its fields are the optional operands, result types and optional, defaulted
  or unit attributes, each defaulting to `null` (or the ODS default), so
  callers pass `.{}`.
- Sorted imports (`std` first), no unused declarations.

### Comments

- Porting adds no explanatory comments; doc comments on public API stay
  short.
- A generated wrapper's doc comment is the IR spelling, a colon and the ODS
  summary (``/// `#cuda.executable`: Format of a compiled executable``); a
  generated builder's is the IR name, a period, then the summary.

### Tests

Beyond the guaranteed `refAllDecls` and registration tests: a
`testContext()` that registers the package and its dependencies and calls
`mlir.Context.init(.{ .registry = registry, .threading = false })` and
`loadAllAvailableDialects()`; attributes and types round-tripped (build with
`get`, print with `{f}`, reparse, compare with `eql`); operations built and
verified.

## Exceptions

- Generated `.td` files start with a "generated" header instead of a banner.
- Generated C APIs keep ODS's integer types (`int`) and repeat the null check
  and dialect load in every constructor.
- `cute_ir/` and `cute_nvgpu/` keep their generated builders in `<pkg>.zig`;
  `llvm/` keeps its builders in a private `ops.zig`.
- `cuda/`, `cute_nvgpu/` and `nvvm/` are regenerated through
  `cute_ir/REGENERATING.md`, `fly_rocdl/` through `fly/REGENERATING.md`.
- `cute_ir:capi` and `cute_nvgpu:capi` each depend on the other's `:dialect`:
  their C++ sources use each other's types.
- `cf`, `llvm`, `math`, `memref`, `scf`, `shardy` and `stablehlo` have no
  `capi` of their own and bind upstream's C API target directly.
- Hand-written builders with many parameters take an `<Op>Args` struct
  (`func.func`, `gpu.func`, `ttir.func`, `cuda.kernel`, `cuda_tile.entry`,
  `memref.subview`, `vector.transfer_read`, ...); function-like ones carry
  `location` inside it.
- Some hand-written `Opts` hold required or non-null defaulted fields where
  named arguments read better (`stablehlo.PadOpts`,
  `mosaic_tpu.LoadOpts.sublane_mask`, `affine.ForOpts.step`).
- stablehlo, shardy, cuda_tile and ttir getters return plain pointers (their
  C calls cannot fail); `ttir.pointerType` and `ttir.tensorDescType` take no
  `ctx` (the C API derives it from the element type).
- Enum cases whose IR spelling is not an identifier use the C++ symbol
  (rocdl's `volatile_op`), as do the generated cute_nvgpu and ttir enums.
- fly's attributes and cute_ir's algebra attributes have named constructors
  (`getStatic`, `getDynamic`, `getTuple`, ...); fly's generated builders use
  snake_case parameters.

## Porting checklist

1. Choose `<pkg>`, the C prefix and the handle; create the package with the
   BUILD targets above.
2. Vendor or write the `.td`/C++ sources and list every `.td` in `td_files`.
3. Write `capi.h`/`capi.cc` for registration and whatever upstream's C API
   lacks: per attribute and type, a predicate, a `Get` and the parameter
   getters.
4. Bind in `<pkg>.zig`, or generate `ops.zig`: `registerDialect`, enums,
   wrappers, helpers, builders.
5. Write `test.zig`, with the `operation_names` list.
6. If anything is generated, add a `REGENERATING.md`.
7. Make consumers depend on `//mlir/dialects/<pkg>` and call
   `<pkg>.registerDialect(registry)`.
8. Run the verification below.

## Verification

Run from the repository root, with `ZIG` set to the Bazel toolchain's `zig`
(`AGENTS.md` shows how to find its standard library; `zig` is next to it).

```bash
#!/usr/bin/env bash
d=mlir/dialects/<pkg>
bazel test "//$d:test" && bazel test //mlir/... && bazel build //...
./tools/buildifier.sh -mode=check -r "$PWD/$d"
"$ZIG" fmt --check "$d"
src=(); while IFS= read -r f; do src+=("$f"); done < <(find "$d" -maxdepth 1 \( -name '*.h' -o -name '*.cc' -o -name '*.cpp' -o -name '*.td' -o -name '*.inc' \))
cc=(); for f in "${src[@]}"; do [[ $f == *.td || $f == *.inc ]] || cc+=("$f"); done
[ ${#cc[@]} -eq 0 ] || clang-format --dry-run --Werror --style=file:mlir/.clang-format "${cc[@]}"

# Conformance: prints nothing on a conforming package.
for f in "${src[@]}"; do l=$(head -1 "$f"); [ ${#l} -eq 80 ] || grep -q generated <<<"$l" || echo "banner: $f"; done
grep -rln --include='*.h' --include='*.td' 'ZML_' "$d"
grep -nE 'alwayslink|glob\(' "$d/BUILD.bazel"
[ -f "$d/capi.cc" ] && grep -nE 'extern "C"|^namespace \{' "$d/capi.cc"
[ -f "$d/capi.h" ] && grep -nw unsigned "$d/capi.h"
grep -rnE --include='*.zig' 'pub fn get[A-Z][A-Za-z]*\(self' "$d"
grep -rnE --include='*.zig' --exclude=test.zig '(Type|Attribute)\.parse\(' "$d"
grep -rln --include='*.zig' --exclude=test.zig '^test ' "$d"
grep -rqF --include='*.zig' 'pub fn registerDialect' "$d" || echo "missing: registerDialect"
if grep -rqF --include='*.zig' '.make(' "$d"; then
  grep -qF 'const operation_names' "$d/test.zig" || echo "missing: operation_names"
  grep -qF 'test "every bound operation is registered"' "$d/test.zig" || echo "missing: registration test"
fi
git grep -n 'DialectHandle.fromString' -- dsl zml examples
exit 0
```
