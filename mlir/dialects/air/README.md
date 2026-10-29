# Metal AIR dialect

The Metal-specific part of AIR, the LLVM IR Apple's Metal compiler emits (not
AMD's asynchronous AIR dialect). Namespace `air`, C++ namespace
`mlir::metal::air`, C API prefix `mlirMetalAIR`. `dsl/metal` emits it, and
XLA's `__gpu$xla.gpu.air` custom call lowers it, from a vendored copy, to a
metallib.

## Operations

Each operation is an `air.*` intrinsic, named as Apple's compiler names it
minus the type suffix its operand and result types determine:

| Operation | Intrinsic |
|---|---|
| `air.fast_exp %x : vector<2xf32>` | `@air.fast_exp.v2f32` |
| `air.max.u %a, %b : i32` | `@air.max.u.i32` |
| `air.simd_shuffle_xor %x, %lane : f16` | `@air.simd_shuffle_xor.f16` |
| `air.convert.s %x : i32 to f32` | `@air.convert.f.f32.s.i32` |
| `air.simdgroup_matrix_8x8_load %p, ... : !llvm.ptr<3> -> vector<64xbf16>` | `@air.simdgroup_matrix_8x8_load.v64bf16.p3bf16` |
| `air.wg.barrier <threadgroup>` | `@air.wg.barrier(i32 2, i32 1)` |

- Float math: `exp`, `exp2`, `log`, `log2`, `tanh`, `sqrt`, `rsqrt`, `fabs`,
  `ceil`, `floor`, `round`, `trunc`, `sin`, `cos`, `fmax`, `fmin`, `pow`,
  `fmod`, `fma`, `clamp` and `dot` on f16 and f32. The `fast_` variants are
  f32 only: under fast math (MSL's default) `exp(x)` is `fast_exp`, and
  `metal::precise::exp` is `exp`.
- Integer math: `max`, `min`, `clamp` and `abs`, as `.s` and `.u`
  overloads.
- SIMD-group functions: `simd_sum`, `simd_max`, `simd_min`,
  `simd_prefix_exclusive_sum`, `simd_shuffle`, `simd_shuffle_xor`,
  `simd_shuffle_down` and `simd_broadcast` on floats, the integer ones as
  `.s` and `.u`, and `simd_any`, `simd_all`. There are no bfloat overloads.
- `convert` between float types, and `convert.s`, `convert.u` between floats
  and integers. Apple's compiler uses LLVM casts for scalar f32 to and from
  f16 or bf16, and between integers.
- `wg.barrier` and `simdgroup.barrier` (`threadgroup_barrier`,
  `simdgroup_barrier`), with `#air.mem_flags`.
- `simdgroup_matrix_8x8_load`, `_store` and `_multiply_accumulate`. A
  `simdgroup_matrix<T, 8, 8>` is a `vector<64xT>`, and `thread_elements()` its
  elements 0 and 1.

Float operations carry `fastmath` flags, which become the call's flags (Apple's
compiler marks every float call `fast` under fast math). SIMD-group, barrier
and matrix operations have unknown effects, so they are not speculated or
moved across divergent control flow. Arithmetic, memory and control flow use
the arith, cf, scf, vector and llvm dialects, with pointers as `!llvm.ptr` in
AIR's address spaces: 0 thread, 1 device, 2 constant, 3 threadgroup (a buffer
is never in thread memory).

## Kernels

A kernel is a void `func.func` with `air.kernel`. Each argument carries one
`!air.kernel` descriptor:

- `air.builtin = #air.builtin<threadgroup_position_in_grid>`: the position and
  size builtins are `uint` or `ushort` scalars or 2- and 3-element vectors,
  the index builtins (`thread_index_in_simdgroup`, ...) scalars.
- `air.buffer = #air.buffer<...>` on an `!llvm.ptr`: the `[[buffer(n)]]` or
  `[[threadgroup(n)]]` index, the address space, the access (`const` pointers
  are `read`), the pointee type with its Metal name, size and alignment, the
  size of a constant reference, `noalias` for `__restrict__`, and the fields
  of a constant struct (`air.struct_type_info`). Device and constant buffers
  share indices; threadgroup buffers have their own.
- `air.arg_name`, optionally, the argument's name.

`air.fast_math_enable` on the module records that it was compiled with fast
math (`air.compile.fast_math_enable`).

## Evidence

Names, signatures, argument descriptors and flags follow what Apple's Metal
compiler 32023.830 (the `@metal_toolchain` repository) emits for MSL probes and
for XLA's Metal kernels, with `metal -std=metal4.0 -S -emit-llvm` (fast math,
-O2, as XLA compiles them). Only the overloads it emits are accepted.

## Build and test

```sh
bazel build //mlir/dialects/air //mlir/dialects/air:capi
bazel test //mlir/dialects/air:test
```

The Zig library is `:air`, the C++ dialect `:dialect` and its C API `:capi`.
`registerDialect(registry)` registers AIR, which loads the LLVM dialect;
`kernel` builds a `func.func`, so callers also register `func`. Attributes are
built through the C API (`BuiltinAttr`, `MemFlagsAttr`, `StructFieldAttr`,
`BufferAttr`), kernel arguments through `builtinArgumentAttr` and
`bufferArgumentAttr`, and every operation has a builder named after it
(`fast_exp`, `max_u`, `wg_barrier`, `simdgroup_matrix_8x8_load`).
