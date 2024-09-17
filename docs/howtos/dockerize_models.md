
# Containerize a Model

A convenient way of [deploying a model](../howtos/deploy_on_server.md) is packaging
it up in a Docker container. Thanks to bazel, this is really easy to do. You
just have to append a few lines to your model's `BUILD.bazel`. Here is how it's
done.

**Note:** This walkthrough will work with your installed container runtime, no
matter if it's **Docker or e.g. Podman.**  Also, we'll create images in the 
[OCI](https://github.com/opencontainers/image-spec) open image format.

Let's try containerizing our [first model](../tutorials/write_first_model.md), as it
doesn't need any additional weights files. We'll see [down below](#adding-weights-and-data) 
how to add those. We'll also see how to add GPU/TPU support for our container
there.

Bazel creates images from `.TAR` archives.

The steps required for containerization are:

1. Let bazel create a MANIFEST for the tar file to come.
2. Let bazel create a TAR archive of everything needed for the model to run.
    - see also: [Deploying Models on a Server](../howtos/deploy_on_server.md), where
      we prepare a TAR file, and copy it to and run it on a remote GPU server.
3. Let bazel create a container image for Linux X86_64.
4. Let bazel load the image _(OPTIONAL)_.
5. Let bazel push the image straight to the Docker registry.
6. Let bazel [add weights and data](#adding-weights-and-data), GPU/TPU support
   _(OPTIONAL)_.

**Note:** every TAR archive we create (one in this example) becomes its own
layer in the container image.

## Dockerizing our first model

We need to add a few "imports" at the beginning of our `BUILD.bazel` so we can
use their rules to define our 5 additional targets:

```python
load("@aspect_bazel_lib//lib:tar.bzl", "mtree_spec", "tar")
load("@aspect_bazel_lib//lib:transitions.bzl", "platform_transition_filegroup")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load", "oci_push")

zig_cc_binary(
    name = "simple_layer",
    main = "main.zig",
    deps = [
        "@zml//async",
        "@zml//zml",
    ],
)
```


### 1. The Manifest

To get started, let's make bazel generate a manifest that will be used when
creating the TAR archive. 

```python
# Manifest created from the simple_layer binary and friends
mtree_spec(
    name = "mtree",
    srcs = [":simple_layer"],
)
```

It is as easy as that: we define that we want everything needed for our binary
to be included in the manifest.


### 2. The TAR

Creating the TAR archive is equally easy; it's just a few more lines of bazel:

```python
# Create a tar archive from the above manifest
tar(
    name = "archive",
    srcs = [":simple_layer"],
    args = [
        "--options",
        "zstd:compression-level=9",
    ],
    compress = "zstd",
    mtree = ":mtree",
)
```

Note that we specify high **zstd** compression, which serves two purposes:
avoiding large TAR files, and also: creating TAR files that are quick to
extract.


### 3. The Image

Creating the actual image is a two-step process:

- First, we use a rule that creates an
  [OCI](https://github.com/opencontainers/image-spec) image (open image
  format). But we're not done yet.
- Second, we force the actual OCI image to be built for `Linux X86_64` always,
  regardless of the host we're building the image **on**.

```python
# The actual docker image, with entrypoint, created from tar archive
oci_image(
    name = "image_",
    base = "@distroless_cc_debian12",
    entrypoint = ["./{}/simple_layer".format(package_name())],
    tars = [":archive"],
)
```

See how we use string interpolation to fill in the folder name for the
container's entrypoint?


Next, we use a transition rule to force the container to be built for 
Linux X86_64:

```python
# We always want to create the image for Linux
platform_transition_filegroup(
    name = "image",
    srcs = [":image_"],
    target_platform = "@zml//platforms:linux_amd64",
)
```

And that's almost it! You can already build the image:

```
# cd examples
bazel build -c opt //simple_layer:image

INFO: Analyzed target //simple_layer:image (1 packages loaded, 8 targets configured).
INFO: Found 1 target...
Target //simple_layer:image up-to-date:
  bazel-out/k8-dbg-ST-f832ad0148ae/bin/simple_layer/image_
INFO: Elapsed time: 0.279s, Critical Path: 0.00s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
```

... and inspect `./bazel-out`. Bazel tells you the exact path to the `image_`.


### 4. The Load

While inspecting the image is surely interesting, we usually want to load the 
image so we can run it.

There is a bazel rule for that: `oci_load`. When we append the following lines 
to `BUILD.bazel`:

```python
# Load will immediately load the image (eg: docker load)
oci_load(
    name = "load",
    image = ":image",
    repo_tags = [
        "distroless/simple_layer:latest",
    ],
)
```
... then we can load the image and run it with the following commands:

```
bazel run -c opt //simple_layer:load
docker run --rm distroless/simple_layer:latest
```


### 5. The Push

We just need to add one more target to the build file before we can push the
image to a container registry:

```python
# Bazel target for pushing the Linux image to the docker registry
oci_push(
    name = "push",
    image = ":image",
    remote_tags = ["latest"],
    # override with -- --repository foo.bar/org/image
    repository = "index.docker.io/renerocksai/simple_layer",
)
```

This will push the `simple_layer` image with the tag `latest` (you can add more)
to the docker registry:

```
bazel run -c opt //simple_layer:push
```

When dealing with maybe a public and a private container registry - or if you
just want to try it out **right now**, you can always override the repository on
the command line:

```
bazel run -c opt //simple_layer:push -- --repository my.server.com/org/image
```


## Adding weights and data

Dockerizing a model that doesn't need any weights was easy. But what if you want
to create a complete care-free package of a model plus all required weights and
supporting files?

We'll use the [MNIST
example](https://github.com/zml/zml/tree/master/examples/mnist) to illustrate
how to build Docker images that also contain data files.

You can `bazel run -c opt //mnist:push -- --repository
index.docker.io/my_org/zml_mnist` in the `./examples` folder if you want to try
it out. 

**Note: Please add one more of the following parameters to specify all the
platforms your containerized model should support.**

- NVIDIA CUDA: `--@zml//runtimes:cuda=true`
- AMD RoCM: `--@zml//runtimes:rocm=true`
- Google TPU: `--@zml//runtimes:tpu=true`
- **AVOID CPU:** `--@zml//runtimes:cpu=false`

**Example:**

```
bazel run //mnist:push -c opt --@zml//runtimes:cuda=true -- --repository index.docker.io/my_org/zml_mnist
```


### Manifest and Archive

We only add one more target to the `BUILD.bazel` to construct the commandline
for the `entrypoint` of the container. All other steps basically remain the
same.

Let's start with creating the manifest and archive:

```python
load("@aspect_bazel_lib//lib:expand_template.bzl", "expand_template")
load("@aspect_bazel_lib//lib:tar.bzl", "mtree_spec", "tar")
load("@aspect_bazel_lib//lib:transitions.bzl", "platform_transition_filegroup")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load", "oci_push")
load("@zml//bazel:zig.bzl", "zig_cc_binary")

# The executable
zig_cc_binary(
    name = "mnist",
    args = [
        "$(location @com_github_ggerganov_ggml_mnist//file)",
        "$(location @com_github_ggerganov_ggml_mnist_data//file)",
    ],
    data = [
        "@com_github_ggerganov_ggml_mnist//file",
        "@com_github_ggerganov_ggml_mnist_data//file",
    ],
    main = "mnist.zig",
    deps = [
        "@zml//async",
        "@zml//zml",
    ],
)

# Manifest created from the executable (incl. its data:  weights and dataset)
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

### Entrypoint

Our container entrypoint commandline is not just the name of the executable
anymore, as we need to pass the weights file and the test dataset to MNIST. A
simple string interpolation will not be enough.

For this reason, we use the `expand_template` rule, like this:

```python
# A convenience template for creating the "command line" for the entrypoint
expand_template(
    name = "entrypoint",
    data = [
        ":mnist",
        "@com_github_ggerganov_ggml_mnist//file",
        "@com_github_ggerganov_ggml_mnist_data//file",
    ],
    substitutions = {
        ":model": "$(rlocationpath @com_github_ggerganov_ggml_mnist//file)",
        ":data": "$(rlocationpath @com_github_ggerganov_ggml_mnist_data//file)",
    },
    template = [
        "./{}/mnist".format(package_name()),
        "./{}/mnist.runfiles/:model".format(package_name()),
        "./{}/mnist.runfiles/:data".format(package_name()),
    ],
)
```

- `data`, which is identical to `data` in the `mnist` target used for running
  the model, tells bazel which files are needed.
- in `substitutions` we define what `:model` and `:data` need to be replaced
  with
- in `template`, we construct the actual entrypoint conmandline


### Image, Push

From here on, everything is analog to the `simple_layer` example, with one
exception: in the `image_` target, we don't fill in the `entrypoint` directly,
but use the expanded template, which we conveniently named `entrypoint` above.


```python

# The actual docker image, with entrypoint, created from tar archive
oci_image(
    name = "image_",
    base = "@distroless_cc_debian12",
    # the entrypoint comes from the expand_template rule `entrypoint` above
    entrypoint = ":entrypoint", 
    tars = [":archive"],
)

# We always want to create the image for Linux
platform_transition_filegroup(
    name = "image",
    srcs = [":image_"],
    target_platform = "@zml//platforms:linux_amd64",
)

# Load will immediately load the image (eg: docker load)
oci_load(
    name = "load",
    image = ":image",
    repo_tags = [
        "distroless/mnist:latest",
    ],
)

# Bazel target for pushing the Linux image to our docker registry
oci_push(
    name = "push",
    image = ":image",
    remote_tags = ["latest"],
    # override with -- --repository foo.bar/org/image
    repository = "index.docker.io/steeve/mnist",
)
```


And that's it! With one simple bazel command, you can push a neatly packaged
MNIST model, including weights and dataset, to the docker registry:

```
bazel run //mnist:push --@zml//runtimes:cuda=true -- --repository index.docker.io/my_org/zml_mnist
```

