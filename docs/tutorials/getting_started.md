
# Getting Started with ZML

In this tutorial, we will install `ZML` and run a few models locally.

## Prerequisites

First, let's checkout the ZML codebase. In a terminal, run:

```
git clone https://github.com/zml/zml.git
cd zml/
```

We use `bazel` to build ZML and its dependencies. We recommend to download it
through `bazelisk`, a version manager for `bazel`.


### Install Bazel:

**macOs:**

```
    brew install bazelisk
```

**Linux:**

```
    curl -L -o /usr/local/bin/bazel 'https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64'
    chmod +x /usr/local/bin/bazel
```



## Run a pre-packaged model

ZML comes with a variety of model examples. See also our reference implementations in the [examples](https://github.com/zml/zml/tree/master/examples/) folder.

### MNIST

The [classic](https://en.wikipedia.org/wiki/MNIST_database) handwritten digits
recognition task. The model is tasked to recognize a handwritten digit, which
has been converted to a 28x28 pixel monochrome image. `Bazel` will download a
pre-trained model, and the test dataset. The program will load the model,
compile it, and classify a randomly picked example from the test dataset.


On the command line:

```
cd examples
bazel run --config=release //mnist
```

### Llama

Llama is a family of "Large Language Models", trained to generate text, based
on the beginning of a sentence/book/article. This "beginning" is generally
referred to as the "prompt".

#### Meta Llama 3.1 8B

This model has restrictions, see
[here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). It **requires
approval from Meta on Huggingface**, which can take a few hours to get granted.

While waiting for approval, you can already
[generate your Huggingface access token](../howtos/huggingface_access_token.md).

Once you've been granted access, you're ready to download a gated model like
`Meta-Llama-3.1-8B-Instruct`!

```
# requires token in $HOME/.cache/huggingface/token, as created by the
# `huggingface-cli login` command, or the `HUGGINGFACE_TOKEN` environment variable.
cd examples
bazel run --config=release //llama:Llama-3.1-8B-Instruct
bazel run --config=release //llama:Llama-3.1-8B-Instruct -- --prompt="What is the capital of France?"
```

You can also try `Llama-3.1-70B-Instruct` if you have enough memory.

### Meta Llama 3.2 1B

Like the 8B model above, this model also requires approval. See
[here](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) for access requirements.

```
cd examples
bazel run --config=release //llama:Llama-3.2-1B-Instruct
bazel run --config=release //llama:Llama-3.2-1B-Instruct -- --prompt="What is the capital of France?"
```

For a larger 3.2 model, you can also try `Llama-3.2-3B-Instruct`.


## Run Tests

```
bazel test //zml:test
```

## Running Models on GPU / TPU

You can compile models for accelerator runtimes by appending one or more of the
following arguments to the command line when compiling or running a model:

- NVIDIA CUDA: `--@zml//runtimes:cuda=true`
- AMD RoCM: `--@zml//runtimes:rocm=true`
- Google TPU: `--@zml//runtimes:tpu=true`
- AWS Trainium/Inferentia 2: `--@zml//runtimes:neuron=true`
- **AVOID CPU:** `--@zml//runtimes:cpu=false`

The latter, avoiding compilation for CPU, cuts down compilation time.


So, to run the OpenLLama model from above on your host sporting an NVIDIA GPU,
run the following:

```
cd examples
bazel run --config=release //llama:Llama-3.2-1B-Instruct            \
          --@zml//runtimes:cuda=true                      \
          -- --prompt="What is the capital of France?"
```


## Where to go next:

In [Deploying Models on a Server](../howtos/deploy_on_server.md), we show how you can
cross-compile and package for a specific architecture, then deploy and run your
model. Alternatively, you can also [dockerize](../howtos/dockerize_models.md) your
model.

You might also want to check out the
[examples](https://github.com/zml/zml/tree/master/examples), read through the
[documentation](../README.md), start
[writing your first model](../tutorials/write_first_model.md), or read about more
high-level [ZML concepts](../learn/concepts.md).

