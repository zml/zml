
# Adding Weights Files

Our [first model](../tutorials/write_first_model.md) did not need any weights files.
We just created weights and biases at runtime.

But real-world models typically need weights files, and maybe some other
supporting files.

We recommend, for easy deployments, you upload those files. In many instances,
you will use a site like [🤗 Hugging Face](https://huggingface.co).

We also recommend to add a `weights.bzl` file to your project root directory, so
you don't "pollute" your build file with long URLs and SHAs:

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _weights_impl(mctx):
    http_archive(
        name = "mnist",
        sha256 = "075905e433ea0cce13c3fc08832448ab86225d089b5d412be67f59c29388fb19",
        url = "https://mirror.zml.ai/data/mnist.tar.zst",
        build_file_content = """exports_files(glob(["**"]), visibility = ["//visibility:public"])""",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

weights = module_extension(
    implementation = _weights_impl,
)
```

The above `weights.bzl` shows how we load files for MNIST:

- `mnist.pt` (model weights)
- `mnist.ylc` (dataset for picking sample images)

Then, in your `BUILD.bazel`, you can refer to the files you defined above, in
the following way:

```python
zig_binary(
    name = "mnist",
    args = [
        "$(location @mnist//:mnist.pt)",
        "$(location @mnist//:t10k-images.idx3-ubyte)",
    ],
    data = [
        "@mnist//:mnist.pt",
        "@mnist//:t10k-images.idx3-ubyte",
    ],
    main = "mnist.zig",
    deps = [
        "//async",
        "//zml",
    ],
)
```

See how:

- we use `data = [ ... ]` to reference the files in `weights.bzl`
- we use `args = [ ... ]` to pass the files as command-line arguments to the
  MNIST executable at runtime, automatically.
