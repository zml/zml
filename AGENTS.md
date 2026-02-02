# AGENTS.md — ZML Codebase Guide

## Overview

ZML is a high-performance AI inference stack written in **Zig**, using **MLIR** (via StableHLO) for compilation and **PJRT** for accelerator execution. It targets CPU, NVIDIA CUDA, AMD ROCm, Google TPU, and AWS Neuron backends.

The build system is **Bazel** (via Bazelisk). The Zig toolchain, LLVM, XLA, and all other dependencies are managed hermetically through Bazel modules.

---

## Essential Commands

### Prerequisites

Install Bazelisk, or use the included `bazel.sh` wrapper (downloads Bazelisk automatically):

```bash
# macOS
brew install bazelisk

# Or use the wrapper (any platform)
./bazel.sh <command>
```

Bazel version is pinned in `.bazelversion` (currently **8.5.0**).

### Build

```bash
# Build everything (debug by default)
bazel build //...

# Build in release mode
bazel build --config=release //...

# Build a specific target
bazel build //zml
bazel build //examples/llama
bazel build //examples/mnist
```

### Test

```bash
# Run all ZML core tests
bazel test //zml:test

# Run stdx tests
bazel test //stdx:test

# Run all tests in the repo
bazel test //...

# Run tests with a specific filter (test runner accepts a substring filter)
bazel test //zml:test --test_arg=<substring>
```

Test output shows errors by default (`--test_output=errors` in `.bazelrc`).

### Run Examples

```bash
# MNIST (downloads pre-trained model automatically)
bazel run --config=release //examples/mnist

# LLaMA (requires model download first)
bazel run //tools/hf -- download meta-llama/Llama-3.1-8B-Instruct --local-dir $HOME/Llama-3.1-8B-Instruct --exclude='*.pth'
bazel run --config=release //examples/llama -- --model=$HOME/Llama-3.1-8B-Instruct
```

### Accelerator Targets

Append platform flags to compile for specific hardware:

```bash
--@zml//platforms:cuda=true     # NVIDIA CUDA
--@zml//platforms:rocm=true     # AMD ROCm
--@zml//platforms:tpu=true      # Google TPU
--@zml//platforms:neuron=true   # AWS Neuron
--@zml//platforms:cpu=false     # Skip CPU (faster compilation)
```

### Build Configurations

| Config | Description |
|---|---|
| (default) | Debug mode (`--compilation_mode=dbg`) |
| `--config=debug` | Optimized backend, debug frontend Zig code |
| `--config=alldebug` | Everything in debug mode |
| `--config=release` | Full optimization (`-Ofast`, `release_safe`) |
| `--config=native` | Optimized for local CPU (`-march=native`) |
| `--config=silent` | Suppresses progress output |

### Format / Lint

```bash
# Format Bazel BUILD files (buildifier)
./tools/buildifier.sh

# Zig code is formatted by ZLS (language server)
# No separate zig fmt command — use ZLS auto-format on save
```

### ZLS (Zig Language Server)

```bash
# Run ZLS through Bazel (for editor integration)
./tools/zls.sh
```

Editor configs in `.zed/settings.json` and `.nvim.lua` point to this script.

---

## Code Organization

```
zml/                    # Root
├── zml/                # Core ZML library (the main package)
│   ├── zml.zig         # Root module — re-exports all public API
│   ├── tensor.zig      # Tensor type (graph node representing computation)
│   ├── buffer.zig      # Buffer type (device-allocated multi-dim array)
│   ├── shape.zig       # Shape type (metadata: dimensions + dtype + tags)
│   ├── dtype.zig       # DataType definitions (f32, f16, bf16, etc.)
│   ├── nn.zig          # Neural network layers (Linear, LayerNorm, RMSNorm, RoPE, etc.)
│   ├── ops.zig         # Low-level MLIR operations (reduce, etc.)
│   ├── module.zig      # Compilation context & zml.module.compile()
│   ├── platform.zig    # Platform/Device/Memory abstraction over PJRT
│   ├── exe.zig         # Compiled executable wrapper
│   ├── io.zig          # TensorStore — weight loading from safetensors
│   ├── mem.zig         # DMA allocators, Bufferized type mapper
│   ├── meta.zig        # Type-level programming (MapType, visit, etc.)
│   ├── safetensors.zig # Safetensors file parser
│   ├── floats.zig      # Float conversion utilities (f8, bf16, etc.)
│   ├── mlirx.zig       # MLIR extensions / helpers
│   ├── pjrtx.zig       # PJRT extensions / helpers
│   ├── slice.zig       # Host-side multi-dim array view
│   ├── testing.zig     # Test utilities (platform auto-detect, approxEq, expectClose)
│   ├── test_runner.zig # Custom Zig test runner (used by all zig_test targets)
│   ├── constants.zig   # Global constants (MAX_RANK=8)
│   ├── attention/      # Attention backends (vanilla, FlashAttention 2/3)
│   ├── io/             # VFS layer (local files, HuggingFace, HTTP, S3)
│   └── tokenizer/      # Tokenizer implementations (HF tokenizers, SentencePiece, homemade)
├── stdx/               # Extended standard library utilities
│   ├── stdx.zig        # Root — BoundedArray, debug, flags, fmt, json, meta, etc.
│   └── *.zig           # Individual utility modules
├── mlir/               # MLIR C API Zig bindings
│   ├── dialects/       # StableHLO and other dialect bindings
│   └── mlir.zig        # Core MLIR type wrappers
├── pjrt/               # PJRT (Pluggable JIT Runtime) Zig bindings
├── platforms/           # Platform-specific PJRT plugin loading
│   ├── cpu/
│   ├── cuda/
│   ├── rocm/
│   ├── tpu/
│   └── neuron/
├── ffi/                # Foreign function interface helpers (C ↔ Zig)
├── upb/                # Protobuf µpb bindings
├── examples/           # Example model implementations
│   ├── mnist/          # MNIST handwritten digit recognition
│   ├── llama/          # LLaMA family (1B–70B)
│   ├── lfm/            # Liquid Foundation Model
│   ├── benchmark/
│   └── vfs/
├── tools/              # Developer tools
│   ├── hf/             # HuggingFace model downloader
│   ├── buildifier.sh   # Bazel file formatter
│   └── zls.sh          # ZLS launcher via Bazel
├── third_party/        # Vendored / external dependencies
│   ├── xla/            # XLA/MLIR/LLVM integration
│   ├── zls/            # ZLS toolchain
│   └── ...
├── bazel/              # Bazel build rules and helpers
├── docs/               # Documentation
├── MODULE.bazel        # Bazel module definition (dependency list)
├── BUILD.bazel         # Root build target (ZLS completion)
├── .bazelrc            # Bazel configuration flags
└── bazel.sh            # Bazelisk bootstrap script
```

---

## Key Abstractions

### Tensor vs Buffer vs Shape

This is the most important concept in ZML:

| Type | What it is | Where it lives | When used |
|---|---|---|---|
| `Shape` | Metadata only (dimensions + dtype + tags) | Nowhere (no memory) | Model definition, compilation |
| `Tensor` | Shape + MLIR value (computation graph node) | Compilation context | Inside `forward()` functions during compilation |
| `Buffer` | Shape + device memory handle | Accelerator (GPU/TPU/etc.) | At runtime, after loading weights / executing |
| `Slice` | Shape + host memory pointer | CPU host memory | For host-side inspection of results |

### Bufferized(T)

`zml.Bufferized(T)` is a compile-time type transformation: given a struct `T` whose fields are `Tensor`, it produces a new struct with those fields replaced by `Buffer`. This maps model definitions to their loaded/executable form.

### Model Lifecycle

1. Parse safetensors → get `TensorStore` (shapes only, weights stay on disk)
2. Instantiate model struct with `Tensor` fields (using shapes from store)
3. Compile `forward()` via `zml.module.compile()` → `Exe`
4. Load weights from disk to device → `Bufferized(Model)`
5. Execute with `Exe` + `Buffer` inputs → `Buffer` outputs
6. Read results back to host

### TensorStore and Weight Loading

```zig
// Typical pattern
var store: zml.io.TensorStore.View = root_store.withPrefix("model.layers.0");
const weight = store.createTensorWithTags("weight", .{ .d_out, .d });
```

Weights are loaded via `zml.io.load()` with DMA options for parallelism.

### Compilation

```zig
// Compile a function into an accelerator executable
var exe = try zml.module.compile(allocator, io, Model.forward, .{ model, input }, platform);
```

The `forward` function operates on `Tensor` values (building an MLIR graph). At compilation time, tensors don't hold data — they record operations via StableHLO dialects.

---

## Naming Conventions and Style

### Zig Style

Follows the [Zig Style Guide](https://ziglang.org/documentation/0.13.0/#Style-Guide) with these house rules:

```zig
// Prefer anonymous struct init
const x: Foo = .{ .bar = 1 };
// Over: const x = Foo{ .bar = 1 };

// Use concrete type name for self
pub fn method(self: Foo) void {}
// Over: pub fn method(self: Self) void {}

// Import the module, access through namespace
const foo = @import("foo.zig"); foo.bar();
// Over: const bar = @import("foo.zig").bar;

// Import types directly only for very frequent types
const Tensor = @import("tensor.zig").Tensor;
```

### Scoped Logging

Every module uses scoped logging:

```zig
const log = std.log.scoped(.@"zml/nn");
const log = std.log.scoped(.@"zml/io");
const log = std.log.scoped(.llama);
```

### Tagged Tensors

Tensor dimensions use semantic tags (`.b`, `.s`, `.d`, `.h`, `.hd`, `.k`, `.q`, `.voc`, `.d_out`, etc.) instead of raw integer indices:

```zig
const q = q_.withTags(.{ .b, .h, .q, .hd });
const result = x.dot(weight, .d);  // Contract over dimension tagged .d
```

### Module/Package Structure

- Root module file exports all public types: `pub const Tensor = @import("tensor.zig").Tensor;`
- Internal imports use relative paths: `const Shape = @import("shape.zig").Shape;`
- Each Bazel package has a `BUILD.bazel` defining `zig_library` and optionally `zig_test`

### Assertions

Use `stdx.debug.assert` (formatted) instead of `std.debug.assert`:

```zig
stdx.debug.assert(tokens_.dtype() == .u32, "Expected u32 tokens, got: {f}", .{tokens_});
```

### Binary Entry Points

Binaries define `pub const std_options` and use `stdx.flags.parseProcessArgs`:

```zig
pub const std_options: std.Options = .{ .log_level = .info };

const CliArgs = struct {
    model: []const u8,
    prompt: ?[]const u8 = null,
    pub const help = \\Usage: ...;
};

pub fn main() !void {
    const args = stdx.flags.parseProcessArgs(CliArgs);
    // ...
}
```

---

## Testing Patterns

### Inline Tests (Zig standard)

Tests are defined inline in source files using `test "name" { ... }`:

```zig
test normalizeL2 {
    const platform = zml.testing.env();
    const input: zml.Tensor = .init(.{ 2, 2 }, .f32);
    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, normalizeL2, .{ input, 1e-12 }, platform);
    defer exe.deinit();
    // ... create buffers, execute, check results
    try zml.testing.expectClose(std.testing.io, expectation, res, 1e-4);
}
```

### Test Utilities

- `zml.testing.env()` — auto-detects available platform (CPU, CUDA, etc.)
- `zml.testing.expectClose()` — approximate floating-point comparison with tolerance
- `zml.testing.approxEq()` — relative + absolute precision check
- `zml.testing.autoCall()` — convenience to call compiled functions in tests

### Bazel Test Targets

Tests are defined in `BUILD.bazel` using `zig_test`:

```python
zig_test(
    name = "test",
    test_runner = ":test_runner",  # Custom runner at zml/test_runner.zig
    deps = ["//zml"],
)
```

The custom test runner at `zml/test_runner.zig` adds:
- Progress reporting
- Substring-based test filtering via CLI args
- Memory leak detection
- Explicit `exit(0)` for clean async thread shutdown

### Running Tests

```bash
bazel test //zml:test                    # Core ZML tests
bazel test //stdx:test                   # stdx utility tests
bazel test //mlir:test                   # MLIR binding tests
bazel test //zml/tokenizer:test          # Tokenizer tests
bazel test //...                         # Everything
```

---

## Important Gotchas

### Tensor Operations Are Graph Construction, Not Execution

Inside `forward()` functions, `Tensor` operations don't compute values — they build an MLIR computation graph. Actual execution happens when calling the compiled `Exe`. You cannot inspect tensor values during compilation.

### MAX_RANK = 8

All tensors are limited to 8 dimensions maximum (defined in `constants.zig`). This is a fixed limit using stack-allocated bounded arrays throughout the codebase.

### Bazel Is Required

There is no standalone Zig build — `build.zig` is empty. All compilation goes through Bazel, which manages the Zig toolchain, LLVM, XLA, PJRT plugins, and all other dependencies hermetically.

### Zig Version

The project tracks a **Zig development version** (`0.16.0-dev.1912+0cbaaa5eb`), not a stable release. API may differ from stable Zig documentation.

### MLIR Thread-Local Context

`CompilationContext` uses a thread-local `_current` variable. Only one compilation can be active per thread. The context must be explicitly activated/deactivated.

### Safetensors Format

ZML uses the HuggingFace **safetensors** format for model weights (not PyTorch `.pth` or `.bin`). The `--exclude='*.pth'` flag is used when downloading models.

### Platform Auto-Detection

`Platform.auto()` scans for available accelerators at runtime and picks the best one. Tests use `zml.testing.env()` which calls this internally.

### Cross-Platform Builds

The project supports cross-compilation via Bazel platform definitions:
- `//platforms:linux_amd64`
- `//platforms:linux_arm64`
- `//platforms:macos_arm64`
- `//platforms:macos_amd64`

### VFS Layer

Model loading goes through a VFS abstraction (`zml.io.VFS`) that supports local files, HuggingFace Hub (`hf://`), HTTP, and S3 backends transparently.

### Docker/OCI Images

Examples like LLaMA include OCI image build rules for containerized deployment using `rules_oci` and distroless base images.

### No CI in Repo

CI configuration is managed externally — there are no `.github/workflows`, Jenkinsfile, or similar files in the repository.

---

## Dependency Map

| Dependency | Purpose |
|---|---|
| `rules_zig` | Zig Bazel rules (custom fork at `zml/rules_zig`) |
| XLA / StableHLO | ML compiler infrastructure, HLO dialect |
| LLVM | Compiler backend (used via `-fllvm` Zig flag) |
| PJRT | Pluggable JIT Runtime for device execution |
| Protobuf / upb | XLA serialization |
| `rules_rust` / Cargo | HuggingFace tokenizer (Rust FFI) |
| SentencePiece | Alternative tokenizer (C++ FFI) |
| FlashAttention | Optimized attention kernels (CUDA) |
| `rules_oci` | Container image building |
| Buildifier | Bazel file formatting |

---

## Adding a New Model

Follow the pattern in `examples/`:

1. Create `examples/<name>/BUILD.bazel` with a `zig_binary` target depending on `//zml`
2. Create a model definition file (e.g., `model.zig`) with:
   - A config struct parsed from JSON (HuggingFace `config.json`)
   - Model struct with `Tensor` fields
   - `init()` method loading shapes from `TensorStore.View`
   - `forward()` method performing the computation
   - `load()` / `loadBuffers()` method using `zml.io.load()`
3. Create `main.zig` with:
   - `CliArgs` struct with `pub const help`
   - Platform initialization via `zml.Platform.auto()`
   - VFS setup for model loading
   - Compilation via `zml.module.compile()`
   - Weight loading and execution loop
