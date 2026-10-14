# Regenerating the dialect definitions

These are instructions for an agent. They describe what this directory must
contain after a CuTe DSL release bump, what is available to derive it, and how
to check the result. How to get there is left to you: combine the sources as
you see fit, write throwaway scripts, probe the compiler. Scripts and
intermediate tables are not committed; only the definitions are.

Paths below are relative to this directory (the one holding this file).

## Goal

Every dialect this directory carries (`cute`, `cute_nvgpu`, and NVIDIA's
`nvvm` and `cuda`, see below) must describe the DSL compiler's
registry as completely and as precisely as the release allows: every
operation, type, attribute and enum the compiler registers, with every fact
that can be established about each of them. Nothing is opaque unless it cannot
be recovered, and each remaining gap is stated in the file where it occurs.

The DSL compiler stays the authority for semantics and lowering. These
definitions exist so that IR can be built, parsed, printed and type-inferred
without it, and so that anything built here is accepted by it unchanged.

## What lives here

- **Taken from the release archive.** Its `cutlass_compiler/` tree provides the
  cute dialect declaration, the public types and attributes and their `.td`
  files, the type interfaces, `CuteDialect.h`, `CuteAttributes.cpp`,
  `CuteDialect.cpp` and the `cutegen` layout algebra. The build reads them
  from the archive; they are never copied or edited here. One patch on the
  archive (`cutegen_dynamic_divisibility.patch`, next to the archive's
  repository rule) teaches `cutegen` the `?{div=N}` leaf. Refresh it if it no
  longer applies. The archive's `CuteOps.td` is not used: its operations
  differ from the compiler's.
- **Maintained here.** `CuteTypes.cpp` started as the release's copy and holds
  the parsers, printers and verifiers of every cute type, fixed wherever the
  release's parsing diverges from the compiler's. `CuteInference.{h,cpp}` holds
  result type inference; the other dialect `.cpp` files and the `BUILD.bazel`
  files complete them. Edit them as the new definitions require. For `nvvm`
  and `cuda` these are `NVVMDialect.{h,cpp}` (with the `#nvvm.target`
  verifier) and `CudaDialect.{h,cpp}` under `include/` and `lib/Dialect/`.
- **Generated: what you produce.** Every `.td` file under `include/` that is not
  a dialect declaration: the operations (`CuteOps.td`, `CuteNVGPUOps.td`,
  `NVVMOps.td`, `CudaOps.td`), the compiler's types, attributes and enums
  beyond the release (`CuteTypesCompiler.td`, `CuteAttrsCompiler.td`,
  `CuteEnums.td`, `CuteNVGPUTypes.td`, `CuteNVGPUAttrs.td`, `CuteNVGPUEnums.td`,
  `NVVMAttrs.td`, `NVVMEnums.td`, `CudaTypes.td`, `CudaAttrs.td`,
  `CudaEnums.td`), the inference of `nvvm` and `cuda` operations whose results
  are not one buildable type (`NVVMInference.cpp`, `CudaInference.cpp`), and
  any new `.td` you need to split them sensibly. The dialect declarations (`*Dialect.td`) change only as far as the
  generated files need.
- **Language bindings, if present.** When the tree carries bindings derived
  from these definitions (a C API under `lib/CAPI`, builder files next to this
  directory), regenerate them from the records of the new definitions
  (`llvm-tblgen --dump-json` gives them all; do not regex the `.td` files).
  When a class changes shape (opaque to parametric, merged, split, renamed
  accessors), change the bindings to match, and keep the existing
  convenience helpers working where it costs little.

Consumers outside this directory use the dialect classes and must keep
compiling: search the repository for the dialect namespaces
(`cutlass_compiler::cute`, `cute_nvgpu`, `nvvm`, `cuda`), for the Zig modules
(`cute.nvgpu`, `cute.nvvm`, `cute.cuda`, and the convenience layers over them
in `mlir/dialects/nvvm.zig` and `mlir/dialects/cuda.zig`) and for the headers
under `include/cute_ir`. When a definition changes shape, update those users in the
same change. Tighter operand constraints also break consumers that built IR
with loose types, and so does an operation that stops inferring its results:
fix the consumer, not the constraint.

## NVIDIA's `nvvm` and `cuda` dialects

The kernels the CuTe builder emits (`kernels/cute`) use two more dialects of
the compiler: `nvvm` (special registers, mbarriers, tcgen05, ...) and `cuda`
(kernels, launch configurations, host runtime calls). Both are carried here
and must be regenerated with every release, like `cute` and `cute_nvgpu`.

- **Neither is upstream's.** The compiler is built on NVIDIA's own LLVM
  revision (the `.so` names it: `LLVM version 23.0.0git (<hash>)`, a hash
  upstream does not have), and its `nvvm` differs from upstream NVVM: it has
  operations, operands and enums upstream lacks (`nvvm.mbarrier.wait.parity`,
  `#nvvm.mbar_wait`) and lacks some upstream has. `cuda` exists only in the
  compiler. No `.td` ships with the release for either: rebuild both from the
  bindings and the live compiler, as for the cute dialects. Upstream's NVVM
  `.td` is neither a source nor a substitute.
- **Where they live.** `include/cute_ir/Dialect/{NVVM,Cuda}/IR` (declarations
  and generated `.td`), `lib/Dialect/{NVVM,Cuda}/IR` (dialect and inference
  `.cpp`), `include/cute_ir-c/Dialect/{NVVM,Cuda}*.h` and
  `lib/CAPI/{NVVM,Cuda}*.cpp` (C API), and `nvvm.zig`, `cuda.zig` (Zig,
  `cute.nvvm` and `cute.cuda`).
- **Names.** The C++ namespaces are `mlir::cutlass_compiler::nvvm` and
  `mlir::cutlass_compiler::cuda`. The dialects are named `nvvm` and `cuda` as
  in the compiler, so a registry holds this `nvvm` or upstream's, never both;
  upstream's NVVM C++ is linked into the tree and must stay unregistered where
  this one is. The C dialect handles are `cute_nvvm` and `cuda`, and the C API
  of `nvvm` is prefixed `CuteNVVM` (`mlirAttributeIsACuteNVVM<Name>`,
  `mlirCuteNVVM<Name>AttrGet`) so that neither collides with upstream's
  `mlirGetDialectHandle__nvvm__` or a future upstream NVVM C API; `cuda`'s is
  prefixed `Cuda`.
- **`llvm` is upstream's.** NVVM operands are `!llvm.ptr<N>` and LLVM
  structs, and the builder emits `llvm.*` operations. The CuTe context
  registers the upstream LLVM dialect, bound by `mlir/dialects/llvm_dialect.zig`
  (its builders, and `mlirLLVMPointerTypeGet` for `!llvm.ptr<N>`); the nvvm
  and cuda constraints name `::mlir::LLVM` types. NVIDIA's `llvm` is not
  vendored: its types print and parse the same, and the compiler reads the
  `llvm.*` operations upstream prints (custom and generic forms, with the
  properties upstream fills in) into the same module. This holds only while
  the compatibility check below passes; when it stops holding, vendor NVIDIA's
  `llvm` the same way (its bindings are `_llvm_ops_gen.py` and
  `_llvm_enum_gen.py`) rather than patching around it.

Sources specific to them, besides those in the table below:

- `_nvvm_ops_gen.py`, `_cuda_ops_gen.py`: the operations. Operand kinds come
  from how `__init__` appends them (and `_ODS_OPERAND_SEGMENTS`), result names
  and kinds from the result properties, inference from whether `__init__`
  builds a `results` list or passes its `results=` argument through.
- `_nvvm_enum_gen.py`, `_cuda_enum_gen.py`: the enums. A builder registered
  under the enum's own name builds the plain `I32EnumAttr` (it follows its
  enum class in the file); one registered as `nvvm.<Name>Attr` builds the
  dialect attribute and gives its mnemonic and form: `#nvvm.<mnemonic><case>`
  or `#nvvm<mnemonic case>`. Some enums only have the latter. Parsing an
  invalid case names the C++ enum (`expected ::mlir::NVVM::<Enum> to be one
  of: ...`) or its summary, which maps each attribute to its enum.
- `cutlass._mlir._mlir_libs._cutlass_ir._mlirDialectsCuda`: the typed cuda
  type and attribute classes and their `get()` parameters. The other cuda
  types and attributes, and the three non-enum nvvm attributes (`shape`,
  `ld_st_matrix_shape`, `target`), are found by trying the `.so`'s
  identifiers as mnemonics (`!cuda.<word>`, `#nvvm.<word>`); `nvvm` has no
  types.
- Attribute constraints the bindings name but no builder covers
  (`nvvm.FPArithRoundingMode`, the `nvvm.Multimem*` orders, anonymous
  `TypeAttr` constraints) are recovered by probing values: the verifier's
  `failed to satisfy constraint: ...` names the rule, and the values it lets
  through give the predicate.
- The compiler aborts on some malformed probes (`nvvm.mma.sync`,
  `nvvm.mma.sp.sync`): run those checks one process each.
- Results the compiler infers but that are not one buildable type get an
  inference rule only where varying the operand types shows the result is one
  operand's type; elsewhere inference fails and the types written stand
  (`<Dialect>_InferredOp`, as in cute). Vary the attributes too: a rule must
  hold for every attribute value (`nvvm.mul` with `mode = wide` doubles the
  width, `nvvm.shfl.sync` with `return_value_and_is_valid` returns a struct),
  or inference must fail.

The bindings follow the cute ones: the C API and the Zig enums, typed
attribute and type wrappers and operation builders are generated from
`llvm-tblgen --dump-json` of the generated `.td` files, then formatted with
`zig fmt`. Builders of operations with regions, and of terminators, do not
verify on creation (their blocks are filled, or their block is completed,
afterwards). `mlir/dialects/nvvm.zig` and `mlir/dialects/cuda.zig` are the
hand-written convenience layer the builder calls; they fix the attributes the
kernels do not choose and must keep building no text.

## Sources

Pin everything to one release version, read from the repository rule that
fetches the release archive. The same version names the Python wheels.

The packaging below is the one of the release these instructions were last
run against. Releases move things around: when a source is missing or has
changed shape, find where the same fact now lives, and update this file.

| Source | Where | What it gives |
|---|---|---|
| Release archive | `https://github.com/NVIDIA/cutlass/archive/refs/tags/v<version>.tar.gz`, `cutlass_compiler/` | The release's own `.td` files: what is already defined, their style, the parameter C++ types and helpers (`Cute_Param`, interfaces). Anything defined there is not redefined here. |
| Operation bindings | wheel `nvidia-cutlass-dsl-libs-base`, `cutlass/_mlir/dialects/_<dialect>_ops_gen.py` | Per operation: name, class name, docstring (summary, description, examples), operands and results in order with single, optional or variadic kind, segment attributes, attribute names with their ODS constraint (`AttrBuilder.get('<Constraint>')`), whether the builder infers result types, region count, successors. |
| Enum bindings | same wheel, `_<dialect>_enum_gen.py` | Every enum: cases, integer values, spellings, and for the ones that are attributes, the mnemonic they print under. |
| Compiler library | wheel `nvidia-cutlass-dsl-libs-cu13` (or `-cu12`), `cutlass/_mlir/_mlir_libs/_cutlass_ir.cu13.*.so` | Its strings: type and attribute class names, parameter names and C++ types (from the generated parser messages `failed to parse <Class> parameter '<p>' which is to be a <T>`), mnemonics, verifier messages and constraint summaries. The strings are not in declaration order, and parameters read by custom directives have no message at all. |
| Typed Python classes | the same `.so`, as `cutlass._mlir.dialects.cute`, `cute_nvgpu`, `cuda` | For every compiler type and attribute: a `get(...)` with its parameters in order, and getters. Building a value and printing it shows the canonical syntax of each parameter value. |
| The examples | the release archive's `examples/python/CuTeDSL` | Real compiler-accepted modules, the round-trip corpus (see below). |
| The live compiler | the same `.so`, loaded from Python | Ground truth for everything above. |

Loading the compiler. The `.so` only runs on Linux, on the architecture it was
built for. Two ways:

- **Full install** (needed for the typed classes): a venv with
  `nvidia-cutlass-dsl==<version>` installed, then `import cutlass` before
  anything else.
- **Bare wheels** (generic API only): unpack the two wheels into one directory,
  put its `nvidia_cutlass_dsl/dsl_packages` on `sys.path`, `import
  cutlass._mlir._mlir_libs`. The typed classes fail to import this way.

Then use `cutlass._mlir.ir` with a fresh `ir.Context()`. The compiler
segfaults on some malformed inputs (some bitlayouts, some bad inference
inputs): run probes in a forked process.

What the live compiler tells you:

- `ir.Type.parse` / `ir.Attribute.parse` accept or reject a spelling, and on
  rejection say where and what they expected; `str()` gives the canonical
  printed form. Together with the typed classes this establishes each syntax:
  the generated parser messages name each missing parameter, and the bespoke
  custom directives (`128x128x32`, `32 DP`, `x32`, trailing flags such as
  `mcast` or `t`) show up as `expected ...` errors. Every mnemonic a parser
  knows can be found by trying every identifier-like string of the `.so`.
- `ir.Module.parse` on generic-form operations with wrong types yields the
  verifier's constraint for each operand and result, in the compiler's words
  (`operand #0 must be ... , but got ...`). Probe each operand with one
  instance of every type class to learn exactly which classes it accepts.
- `op.has_trait(ir.IsTerminatorTrait)`, `ir.NoTerminatorTrait`; constructing
  `ir.InferTypeOpInterface(op)` or `ir.MemoryEffectsOpInterface(op)` fails when
  the operation lacks the interface.
- `ir.InferTypeOpInterface(<OpView class>).inferReturnTypes(operands=...,
  attributes=..., context=...)` is the inference oracle, with operand values
  taken as block arguments of a dummy region. It cannot take properties:
  for operations with attributes, parse and verify a module instead.
- Loop-invariant code motion hoists `Pure` operations, `canonicalize` removes
  unused ones that do not write, and `cse` merges ones that at most read.
- Printing a parsed generic operation shows its custom assembly.

The round-trip corpus: run every example of the archive's
`examples/python/CuTeDSL` with `CUTE_DSL_KEEP=ir-debug`, which keeps each
compiled module's IR; set `CUTE_DSL_ARCH` for the architecture-restricted
ones. This gives hundreds of real modules covering far more operations than
the fixtures in the repository (the testdata under `tools/` and the kernels'
fixtures), which are added to it.

When sources disagree, the live compiler wins; record the disagreement in a
comment.

## What each definition must carry

Operations:

- The operation name, and the binding's class name as the def name.
- `summary` and `description` from the docstring, kept whole, including its
  examples. Reflow nothing that is code. Every example must parse and verify in
  the compiler; fix the ones that do not and say so.
- Every operand and result, in order, with its name, its kind (single,
  `Optional`, `Variadic`), and a type constraint that accepts exactly the type
  classes the compiler's verifier accepts, with the verifier's wording as its
  summary. Where a class or interface is not defined here or in the release,
  or no probe type satisfies it, the predicate accepts any type and the file
  says so. Constraints on types of unregistered dialects (such as `ptr`) must
  also match the opaque form those types take here. A constraint must never
  reject what the compiler accepts.
- Every attribute with its constraint: the ODS constraint the bindings name,
  or the generated enum or dialect attribute it corresponds to; `OptionalAttr`
  and `DefaultValuedAttr` as the bindings show.
- `AttrSizedOperandSegments` / `AttrSizedResultSegments`, regions (count and
  kind), successors, terminator traits.
- `Pure` (or the memory effects) when established; none when unknown.
- `InferTypeOpInterface` exactly on the operations whose result types the
  compiler infers. ODS generates the inference when all results are buildable
  types; the others are written in `CuteInference.cpp`. The compiler's
  inference is not a function of types alone: it folds constant operands and
  keeps divisibility that `cutegen` drops. Where a rule is not recovered,
  inference must fail rather than guess, and those operations override
  `refineReturnTypes` so written result types stand: compiler-accepted IR is
  never rejected for want of an inference rule.
- Operations keep the generic assembly form unless an `assemblyFormat` has been
  checked to round-trip against the compiler (see below); the custom syntax
  stays in the description.

Types and attributes:

- Every class the compiler has that the release lacks, with its mnemonic,
  summary, parameters in declaration order (from the typed `get()`s and the
  order parsing asks for them) with their C++ types and optional or defaulted
  status, and a printed example in the description. C++ types of parameters
  read by custom directives are not observable: choose the natural one.
- The syntax exactly as the compiler prints and parses it: a declarative
  `assemblyFormat` when it can express it, otherwise a custom parser and
  printer. Parametric beats opaque: a verbatim string payload is acceptable
  only when the syntax cannot be reproduced, and the file says why. A hidden
  parameter that never prints is left out: keeping it would make a built value
  differ from its own printed text.
- The compiler's type aliases (`!memref_smem_f16`, `!copy_simt`, ...) are part
  of its printed form: reproduce them through the dialect's
  `OpAsmDialectInterface`.
- Verifiers (`genVerifyDecl`) where the compiler rejects parameter values, when
  the rule is known. Large legality tables (instruction shapes and forms) may
  be left to the compiler; the file says which.
- Enums as `I32EnumAttr` with the compiler's values and spellings, and their
  `EnumAttr` with the compiler's mnemonic and `<case>` form.

Every generated file starts with a short header naming the release version and
the sources it came from, saying it is generated, and listing what it leaves
out or leaves unconstrained, and why.

ODS pitfalls met so far:

- Naming `InferTypeOpInterface` in the traits of an operation with buildable
  results suppresses the `inferReturnTypes` ODS would generate.
- `mlir-tblgen -gen-enum-decls` emits every enum it sees, so a dialect's enums
  file must not include another dialect's.
- The release's `CuteDialect.h` includes the attributes before the types, so
  enums the attributes use must be emitted ahead of the attributes' include.
- A mnemonic that varies with a value (`!cute.i32`, `!cute.i7`) needs a hook
  in front of the generated type parser.
- An attribute whose body is not in `<...>` prints in the opaque form
  (`#dialect<...>`); its pretty form does not parse.

## Checking the result

A change is done when all of these hold:

1. **Coverage.** Every operation in the bindings, every enum, and every
   mnemonic the compiler's parsers know is either defined here, defined by the
   release, or listed in the file's header as left out with the reason.
2. **Round trip against the compiler,** on the corpus and on sweeps of
   spellings for every type and attribute (combinations of parameter values):
   whatever the compiler accepts, this dialect accepts, and prints text the
   compiler reads back to its own canonical form. Where the compiler's printed
   form is itself reparseable, this dialect prints it byte-identical. Where
   the compiler prints text its own parser rejects, or its parser drops data,
   the reparseable, lossless form wins; list those cases.
3. **Inference agrees** with the compiler on every result the corpus and
   crafted oracle cases produce. The inference test holds recorded compiler
   results, since the compiler cannot be asked at test time: add a case for
   every inferred operation the emitted modules do not already cover.
4. **The repository's tests pass**, including the dialect's own tests and the
   tests of every consumer found above (kernel emitters, host tools, language
   bindings), built with the tree's usual Bazel configuration.
5. **NVIDIA's compiler reads what the kernels emit.** Dump the IR of every
   kernel the repository's tests emit through `kernels/cute`: the builder's
   own tests, the MXFP4 emission tests under `zml/moe/cute_kernels`, and the
   FlashMLA program (`zml/attention/cute_kernels/flashmla_sm100.zig`, whose
   `Program.emit` needs a temporary test with a few configurations, single-CTA,
   clustered and persistent). Dump both with the previous definitions and with
   the new ones. In the release's venv, `import cutlass`, then parse and
   verify each module with `cutlass._mlir.ir.Module.parse`, and compare the
   generic print (`get_asm(print_generic_op_form=True)`) of each module before
   and after: it must be identical, and a module the compiler rejected before
   must be rejected the same way after. This is also what keeps upstream
   `llvm` acceptable (see above).
6. **Both copies agree.** When the same dialect is carried by more than one
   repository, the shared files are identical in each, and only
   repository-specific files (`BUILD.bazel`, bindings) differ.

Report, at the end: what changed in each file, the coverage figures, the
round-trip and inference figures, what is still opaque or unconstrained and
why, the test results, and anything in these instructions that was wrong,
missing or unclear.
