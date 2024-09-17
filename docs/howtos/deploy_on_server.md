
# Deploying Models on a Server

To run models on remote GPU/TPU machines, it is inconvenient to have to check
out your project’s repository and compile it on every target. Instead, you more
likely want to cross-compile right from your development machine, **for every**
supported target architecture and accelerator.

See [Getting Started with ZML](../tutorials/getting_started.md) if you need more
information on how to compile a model.

**Here's a quick recap:**

You can compile models for accelerator runtimes by appending one or more of the
following arguments to the command line when compiling / running a model:

- NVIDIA CUDA: `--@zml//runtimes:cuda=true`
- AMD RoCM: `--@zml//runtimes:rocm=true`
- Google TPU: `--@zml//runtimes:tpu=true`
- **AVOID CPU:** `--@zml//runtimes:cpu=false`

So, to run the OpenLLama model from above **on your development machine**
housing an NVIDIA GPU, run the following:

```
cd examples
bazel run -c opt //llama:OpenLLaMA-3B --@zml//runtimes:cuda=true
```


## Cross-Compiling and creating a TAR for your server

Currently, ZML lets you cross-compile to one of the following target
architectures:

- Linux X86_64: `--platforms=@zml//platforms:linux_amd64`
- Linux ARM64: `--platforms=@zml//platforms:linux_arm64`
- MacOS ARM64: `--platforms=@zml//platforms:macos_arm64`

As an example, here is how you build above OpenLLama for CUDA on Linux X86_64:

```
cd examples
bazel build -c opt //llama:OpenLLaMA-3B               \
            --@zml//runtimes:cuda=true                \
            --@zml//runtimes:cpu=false                \
            --platforms=@zml//platforms:linux_amd64
```

### Creating the TAR

When cross-compiling, it is convenient to produce a compressed TAR file that
you can copy to the target host, so you can unpack it there and run the model.

Let's use MNIST as example.

If not present already, add an "archive" target to the model's `BUILD.bazel`,
like this:

```python
load("@aspect_bazel_lib//lib:tar.bzl", "mtree_spec", "tar")

# Manifest, required for building the tar archive
mtree_spec(
    name = "mtree",
    srcs = [":mnist"],
)

# Create a tar archive from the above manifest
tar(
    name = "archive",
    srcs = [":mnist"],
    args = [
        "--options",
        "zstd:compression-level=9",
    ],
    compress = "zstd",
    mtree = ":mtree",
)
```

... and then build the TAR archive:

```
# cd examples
bazel build -c opt //mnist:archive                    \
            --@zml//runtimes:cuda=true                \
            --@zml//runtimes:cpu=false                \
            --platforms=@zml//platforms:linux_amd64
```

Note the `//mnist:archive` notation.

The resulting tar file will be in `bazel-bin/mnist/archive.tar.zst`.

### Run it on the server

You can copy the TAR archive onto your Linux X86_64 NVIDIA GPU server, untar
and run it:

```bash
# on your machine
scp bazel-bin/mnist/archive.tar.zst destination-server:
ssh destination-server   # to enter the server

# ... on the server
tar xvf archive.tar.zst
./mnist \
    'mnist.runfiles/_main~_repo_rules~com_github_ggerganov_ggml_mnist/file/mnist.pt' \
    'mnist.runfiles/_main~_repo_rules~com_github_ggerganov_ggml_mnist_data/file/mnist.ylc'
```

The easiest way to figure out the commandline arguments of an example model is
to consult the model's `BUILD.bazel` and check out its `args` section. It will
reference e.g. weights files that are defined either in the same `BUILD.bazel`
file or in a `weights.bzl` file.

You can also consult the console output when running your model locally:

```bash
bazel run //mnist

INFO: Analyzed target //mnist:mnist (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //mnist:mnist up-to-date:
  bazel-bin/mnist/mnist
INFO: Elapsed time: 0.302s, Critical Path: 0.00s
INFO: 3 processes: 3 internal.
INFO: Build completed successfully, 3 total actions
INFO: Running command line: bazel-bin/mnist/mnist ../_main~_repo_rules~com_github_ggerganov_ggml_mnist/file/mnist.pt ../_main~_repo_rules~com_github_ggerganov_ggml_mnist_data/file/mnist.ylc
# ...
```

You see the command line right up there. On the server, you just need to replace
`../` with the 'runfiles' directory of your TAR.

