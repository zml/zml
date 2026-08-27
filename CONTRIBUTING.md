# Contributing to ZML

Thanks for contributing. This file is the short entry point; day-to-day agent and
contributor conventions also live in [`AGENTS.md`](./AGENTS.md).

## Layout

- `zml/` — core library (Tensor, Shape, Buffer, compile, IO, attention, MoE, …)
- `stdx/` — shared Zig utilities
- `ffi/`, `upb/` — C helpers and protobuf (XLA compile options)
- `mlir/`, `pjrt/`, `platforms/` — MLIR bindings, PJRT, accelerator plugins
- `examples/`, `docs/` — runnable examples and documentation
- `bazel/`, `third_party/` — build wiring and vendored deps

## Build and test

```bash
bazel build //zml
bazel test //zml:test
bazel test //stdx:test
bazel test //zml/tokenizer:test
bazel run //examples/mnist
```

Platform flags when relevant, for example:

```bash
--@zml//platforms:cuda=true
--@zml//platforms:rocm=true
--@zml//platforms:tpu=true
--@zml//platforms:cpu=false
```

Format Zig with `zig fmt` (CI runs `zig fmt --check` outside `third_party/`).
Format Starlark with `./tools/buildifier.sh`.

## Style

Follow the [Zig style guide](https://ziglang.org/documentation/master/#Style-Guide)
and [`docs/misc/style_guide.md`](./docs/misc/style_guide.md). Prefer
`const x: Foo = .{ .bar = 1 };`, `pub fn method(self: Foo)`, PascalCase types,
lowerCamelCase functions/fields.

## Pull requests

- Scoped, imperative subjects (e.g. `zml/tensor: add Tensor.onMemory()`).
- Short description, linked issue if any, platform impact, and the exact
  `bazel build` / `bazel test` commands you ran.
- Keep docs in sync with public APIs (`zml.Slice`, `zml.Exe`, `zml.Bufferized`,
  `zml.io.TensorStore` / `Loader` — not older names like HostBuffer / Executable / aio).

## License

Contributions are under the [Apache 2.0 license](./LICENSE).
