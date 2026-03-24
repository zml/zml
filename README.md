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

ZML is a production inference stack built close to the hardware.

It lowers models directly onto NVIDIA, AMD, TPU, and Trainium targets from a single codebase, without depending on and suffering from the Python-heavy runtime layers that most of the ecosystem is built around.

It is built using the
[Zig](https://ziglang.org) language, [MLIR](https://mlir.llvm.org), and [Bazel](https://bazel.build).

# Getting started

## Prerequisites

We use `bazel` to build ZML and its dependencies. The only prerequisite is
`bazel`, which we recommend to download through `bazelisk`, a version manager
for `bazel`.

**Install Bazel** (recommended):

### macOS
```
brew install bazelisk
```

### Linux

```
curl -L -o /usr/local/bin/bazel 'https://github.com/bazelbuild/bazelisk/releases/download/v1.28.0/bazelisk-linux-amd64'
chmod +x /usr/local/bin/bazel
```

## Run a pre-packaged model

We have implemented a variety of example models in ZML. See our reference
implementations in the
[examples](https://github.com/zml/zml/tree/master/examples/) folder.

### MNIST

The [classic](https://en.wikipedia.org/wiki/MNIST_database) handwritten digits
recognition task. The model is tasked to recognize a handwritten digit, which
has been converted to a 28x28 pixel monochrome image. `Bazel` will download a
pre-trained model, and the test dataset. The program will load the model,
compile it, and classify a randomly picked example from the test dataset.

On the command line:

```
bazel run //examples/mnist
```

### Meta Llama 3.2 1B

This model has restrictions, see
[here](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). It **requires
approval from Meta on Hugging Face**, which can take a few hours to get granted.

Ensure you are authenticated with the Hugging Face CLI:
```
hf auth login
```
Alternatively, set the `HF_TOKEN` environment variable.

Now, you can run the model like so:
```
bazel run //examples/llm -- --model=hf://meta-llama/Llama-3.2-1B-Instruct --prompt="What is the capital of France?"
```

For a larger 3.2 model, you can also try `Llama-3.2-3B-Instruct`.


### Meta Llama 3.1 8B

Like the 1B model above, this model also requires approval. See
[here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for access requirements.


```
bazel run //examples/llm -- --model=hf://meta-llama/Llama-3.1-8B-Instruct --prompt="What is the capital of France?"
```

You can also try `Llama-3.1-70B-Instruct` if you have enough memory.

## Running Models on GPU / TPU

You can compile models for accelerator runtimes by appending one or more of the
following arguments to the command line when compiling / running a model:

- NVIDIA CUDA: `--@zml//platforms:cuda=true`
- AMD RoCM: `--@zml//platforms:rocm=true`
- Google TPU: `--@zml//platforms:tpu=true`
- AWS Trainium/Inferentia 2: `--@zml//platforms:neuron=true`
- **AVOID CPU:** `--@zml//platforms:cpu=false`

The latter, avoiding compilation for CPU, cuts down compilation time.

So, to run the Llama 3.1 8B model from above on your host supporting an NVIDIA GPU,
run the following:

```
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
```

And on your host supporting an AMD GPU:

```
bazel run //examples/llm --@zml//platforms:rocm=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
```

Same goes for all supported platforms.

## Run Tests

```
bazel test //zml:test
```

# A taste of ZML

## MNIST


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

# Where to go next:

You might want to check out more [examples](./examples), read through the
[documentation directly on GitHub](./docs/README.md).

# Contributing

See [here][Contributing].

# License

ZML is licensed under the [Apache 2.0 license](./LICENSE).

# Thanks to our contributors

<a href="https://github.com/zml/zml/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zml/zml" />
</a>
