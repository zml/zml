<div align="center">
  <img src="https://raw.githubusercontent.com/zml/zml.github.io/refs/heads/main/docs-assets/zml-banner.png" style="width:100%; height:120px;">
  <a href="https://zml.ai">Website</a>
  | <a href="#getting-started">Getting Started</a>
  | <a href="./docs/README.md">Documentation</a>
  | <a href="https://discord.gg/6y72SN2E7H">Discord</a>
  | <a href="./CONTRIBUTING.md">Contributing</a>
</div>

[ZML]: https://zml.ai/
[Getting Started]: #getting-started
[Documentation]: ./docs/README.md
[Contributing]: ./CONTRIBUTING.md
[Discord]: https://discord.gg/6y72SN2E7H

# About

ZML is a production inference stack, purpose-built to decouple AI workloads from proprietary hardware.

Any model, many hardwares, one codebase, peak performance.

Compiled directly to NVIDIA, AMD, TPU, Trainium for peak hardware performance on any accelerator. No rewriting.

It is built using the
[Zig](https://ziglang.org) language, [MLIR](https://mlir.llvm.org), and [Bazel](https://bazel.build).

# Getting Started

## Prerequisites

We use `bazel` to build ZML and its dependencies. The only prerequisite is
`bazel`, which we recommend installing through `bazelisk`.

### macOS

```bash
brew install bazelisk
```

### Linux

```bash
curl -L -o /usr/local/bin/bazel 'https://github.com/bazelbuild/bazelisk/releases/download/v1.28.0/bazelisk-linux-amd64'
chmod +x /usr/local/bin/bazel
```

## 30-Second Smoke Test

Run the MNIST example:

```bash
bazel run //examples/mnist
```

This downloads a small pretrained MNIST model, compiles it, loads the weights, and
classifies a random handwritten digit.

## LLM Quickstart

The main LLM example is [`//examples/llm`](./examples/llm). It currently supports:

- Llama 3.1 / 3.2
- Qwen 3.5
- LFM 2.5

Authenticate with Hugging Face if you want to load gated repos such as Meta
Llama:

```bash
bazel run //tools/hf -- auth login
```

Alternatively, set the `HF_TOKEN` environment variable.

Then run a prompt directly:

```bash
bazel run //examples/llm -- --model=hf://meta-llama/Llama-3.2-1B-Instruct --prompt="What is the capital of France?"
```

Open the interactive chat loop by omitting `--prompt`:

```bash
bazel run //examples/llm -- --model=hf://meta-llama/Llama-3.2-1B-Instruct
```

You can also load from:

- a local directory: `--model=/var/models/meta-llama/Llama-3.2-1B-Instruct`
- S3: `--model=s3://bucket/path/to/model`

## Running Models on GPU / TPU

Append one or more platform flags when compiling or running:

- NVIDIA CUDA: `--@zml//platforms:cuda=true`
- AMD RoCM: `--@zml//platforms:rocm=true`
- Google TPU: `--@zml//platforms:tpu=true`
- AWS Trainium / Inferentia 2: `--@zml//platforms:neuron=true`
- Disable CPU compilation: `--@zml//platforms:cpu=false`

Example on CUDA:

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://meta-llama/Llama-3.2-1B-Instruct --prompt="Write a haiku about Zig"
```

Example on ROCm:

```bash
bazel run //examples/llm --@zml//platforms:rocm=true -- --model=hf://meta-llama/Llama-3.2-1B-Instruct --prompt="Write a haiku about Zig"
```

## Run Tests

```bash
bazel test //zml:test
```

# Examples

- [`examples/llm`](./examples/llm): unified LLM CLI for Llama, Qwen, and LFM
- [`examples/mnist`](./examples/mnist): smallest end-to-end model run
- [`examples/sharding`](./examples/sharding): logical mesh, partitioners, shard-local execution, profiler output
- [`examples/io`](./examples/io): inspect and load local, `hf://`, `https://`, and `s3://` repositories through the VFS layer
- [`examples/benchmark`](./examples/benchmark): measure loading and execution performance

# A Taste Of ZML

```zig
const Mnist = struct {
    fc1: Layer,
    fc2: Layer,

    const Layer = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn init(store: zml.io.TensorStore.View) Layer {
            return .{
                .weight = store.createTensor("weight", .{ .d_out, .d }, null),
                .bias = store.createTensor("bias", .{.d_out}, null),
            };
        }

        pub fn forward(self: Layer, input: zml.Tensor) zml.Tensor {
            return self.weight.dot(input, .d).add(self.bias).relu().withTags(.{.d});
        }
    };

    pub fn init(store: zml.io.TensorStore.View) Mnist {
        return .{
            .fc1 = .init(store.withPrefix("fc1")),
            .fc2 = .init(store.withPrefix("fc2")),
        };
    }

    pub fn load(
        self: *const Mnist,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(Mnist) {
        return zml.io.load(Mnist, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mnist)) void {
        self.fc1.weight.deinit();
        self.fc1.bias.deinit();
        self.fc2.weight.deinit();
        self.fc2.bias.deinit();
    }

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: zml.Tensor) zml.Tensor {
        var x = input.flatten().convert(.f32).withTags(.{.d});
        const layers: []const Layer = &.{ self.fc1, self.fc2 };
        for (layers) |layer| {
            x = layer.forward(x);
        }
        return x.argMax(0).indices.convert(.u8);
    }
};
```

For a full walkthrough, see:

- [Getting Started](./docs/tutorials/getting_started.md)
- [Writing your first model](./docs/tutorials/write_first_model.md)
- [ZML Concepts](./docs/learn/concepts.md)
- [Deploying on a server](./docs/howtos/deploy_on_server.md)

# Where To Go Next

- Run more examples in [`./examples`](./examples)
- Read the example-specific notes in [`examples/llm/README.md`](./examples/llm/README.md)
- Learn tagged dimensions in [`working_with_tensors.md`](./docs/tutorials/working_with_tensors.md)
- Start building a model with [`write_first_model.md`](./docs/tutorials/write_first_model.md)
- Explore deployment in [`deploy_on_server.md`](./docs/howtos/deploy_on_server.md)

# Contributing

See [here][Contributing].

# License

ZML is licensed under the [Apache 2.0 license](./LICENSE).

# Thanks To Our Contributors

<a href="https://github.com/zml/zml/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zml/zml" />
</a>
