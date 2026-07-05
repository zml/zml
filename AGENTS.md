# Repository Guidelines

## Project Structure & Module Organization

ZML is a Zig project built with Bazel. Core library code lives in `zml/`, with IO under `zml/io/`, tokenizers under `zml/tokenizer/`, attention implementations under `zml/attention/`, and MoE support under `zml/moe/`. Shared Zig utilities are in `stdx/`; FFI and C helpers are in `ffi/` and `upb/`. Bazel rules and workspace helpers live in `bazel/` and `third_party/`. Runnable examples are in `examples/`, while user-facing documentation is in `docs/`. Keep tests close to implementation files unless a package already has a separate test file.

## Build, Test, and Development Commands

- `bazel build //zml`: builds the main ZML library.
- `bazel test //zml:test`: runs the core Zig test target.
- `bazel test //stdx:test`: runs shared utility tests.
- `bazel test //zml/tokenizer:test`: runs tokenizer tests.
- `bazel run //examples/mnist`: runs the quick MNIST smoke test.
- `bazel run //examples/llm -- --model=hf://... --prompt="..."`: runs the LLM example.
- `bazel build //...` and `bazel test //...`: use before broad or shared changes.
- `./tools/buildifier.sh`: formats Bazel/Starlark files.
- `echo "$(bazel info output_base)/$(bazel cquery --output=files "$(bazel cquery "filter('zig_toolchain', deps(//zml))" 2>/dev/null | cut -d' ' -f1 | head -n1)" 2>/dev/null | rg "/lib\$")/std"`: find the current Zig standard library used by Bazel.

Append platform flags when relevant, for example `--@zml//platforms:cuda=true`, `--@zml//platforms:rocm=true`, `--@zml//platforms:tpu=true`, or `--@zml//platforms:cpu=false`.

## Zabel Acceptance Target

Use `//examples/llm` as the Zabel acceptance build target. Use
`deps(//examples/llm) except //examples/llm` as the exact query, cquery, and
aquery expression.

Build the pinned `@bazel//src:bazel` executable from `/Users/dzbarsky/zabel`
and run the Bazel oracle commands from this repository:

```bash
<pinned-bazel> --output_base=<zml-bazel-output-base> query \
  'deps(//examples/llm) except //examples/llm'

<pinned-bazel> --output_base=<zml-bazel-output-base> cquery \
  --config=remote \
  --platforms=//platforms:linux_amd64 \
  --@zml//platforms:cuda=true \
  'deps(//examples/llm) except //examples/llm'

<pinned-bazel> --output_base=<zml-bazel-output-base> aquery \
  --config=remote \
  --platforms=//platforms:linux_amd64 \
  --@zml//platforms:cuda=true \
  --output=text \
  'deps(//examples/llm) except //examples/llm'

<pinned-bazel> --output_base=<zml-bazel-output-base> build \
  --config=remote \
  --platforms=//platforms:linux_amd64 \
  --@zml//platforms:cuda=true \
  //examples/llm
```

Use the same expression and target configuration for Zabel comparisons.
Bazel `query` rejects `--platforms` and `--@zml//platforms:cuda=true`, so do
not pass either configuration option to Bazel or Zabel `query`. Let the pinned
Bazel update `MODULE.bazel.lock` before comparing Zabel.

## Coding Style & Naming Conventions

Follow the Zig style guide and format Zig code with `zig fmt`; CI checks `zig fmt --check` outside `third_party/`. Use four-space indentation. Prefer `const x: Foo = .{ .bar = 1 };`, `pub fn method(self: Foo)`, and module imports such as `const foo = @import("foo.zig"); foo.bar()`. Use `PascalCase` for types and `lowerCamelCase` for functions, fields, and locals. Add comments only for non-obvious invariants, ownership, tensor shape semantics, or platform behavior.

## Testing Guidelines

Tests are generally inline Zig `test "..." {}` blocks near the code they cover. Use descriptive test names and narrow Bazel targets first, then broaden when touching shared APIs. For tensor or graph changes, cover shape/tag inference and runtime behavior when practical. CI builds all targets and runs tests on CPU, CUDA, and ROCm where available.

## Commit & Pull Request Guidelines

Recent history uses scoped, imperative commit subjects such as `zml/tensor: add Tensor.onMemory()` or `workspace: use latest version of upstreamable rules_zig`. Keep subjects concise and mention the affected area first. Pull requests should include a short description, relevant linked issue, platform impact if any, and the exact `bazel build` or `bazel test` commands run. Include screenshots only for docs or UI-visible changes.
