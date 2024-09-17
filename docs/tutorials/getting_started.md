
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
    curl -L -o /usr/local/bin/bazel 'https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64'
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
bazel run -c opt //mnist
```

### Llama

Llama is a family of "Large Language Models", trained to generate text, based
on the beginning of a sentence/book/article. This "beginning" is generally
referred to as the "prompt".

#### TinyLlama, Stories 15M

To start, you can use a small model trained specifically on children's history
books. This model has been trained by [Andrej Karpathy](https://x.com/karpathy);
you can read more about it on his 
[Github](https://github.com/karpathy/llama2.c).

```
cd examples
bazel run -c opt //llama:TinyLlama-Stories-15M
bazel run -c opt //llama:TinyLlama-Stories-15M -- --prompt="Once upon a time, there was a cute little dragon"
```

#### OpenLLama 3B

```
cd examples
bazel run -c opt //llama:OpenLLaMA-3B
bazel run -c opt //llama:OpenLLaMA-3B -- --prompt="Once upon a time,"
```

#### Meta Llama 3 8B

This model has restrictions, see
[here](https://huggingface.co/meta-llama/Meta-Llama-3-8B): it **requires
approval from Meta on Huggingface**, which can take a few hours to get granted.

While waiting for approval, you can already 
[generate your Huggingface access token](../howtos/huggingface_access_token.md).

Once you've been granted access, you're ready to download a gated model like
`Meta-Llama-3-8b`!

```
# requires token in $HOME/.cache/huggingface/token
cd examples
bazel run -c opt //llama:Meta-Llama-3-8b
bazel run -c opt //llama:Meta-Llama-3-8b -- --promt="Once upon a time,"
```


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
- **AVOID CPU:** `--@zml//runtimes:cpu=false`

The latter, avoiding compilation for CPU, cuts down compilation time.


So, to run the OpenLLama model from above on your host sporting an NVIDIA GPU,
run the following:

```
cd examples
bazel run -c opt //llama:OpenLLaMA-3B             \
          --@zml//runtimes:cuda=true              \
          -- --prompt="Once upon a time,"
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

