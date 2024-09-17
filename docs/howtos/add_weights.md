
# Adding Weights Files

Our [first model](../tutorials/write_first_model.md) did not need any weights files.
We just created weights and bias at runtime.

But real-world models typically need weights files, and maybe some other
supporting files. 

We recommend, for easy deployments, you upload those files. In many instances,
you will use a site like [🤗 Hugging Face](https://huggingface.co).

We also recommend to add a `weights.bzl` file to your project root directory, so
you don't "pollute" your build file with long URLs and SHAs:

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

def _weights_impl(mctx):
    http_file(
        name = "com_github_zml_cdn_mnist",
        downloaded_file_path = "mnist.pt",
        sha256 = "d8a25252e28915e147720c19223721f0f53e3317493727ca754a2dd672450ba9",
        url = "https://github.com/ggerganov/ggml/raw/18703ad600cc68dbdb04d57434c876989a841d12/examples/mnist/models/mnist/mnist_model.state_dict",
    )

    http_file(
        name = "com_github_zml_cdn_mnist_data",
        downloaded_file_path = "mnist.ylc",
        sha256 = "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
        url = "https://github.com/ggerganov/ggml/raw/18703ad600cc68dbdb04d57434c876989a841d12/examples/mnist/models/mnist/t10k-images.idx3-ubyte",
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
zig_cc_binary(
    name = "mnist",
    args = [
        "$(location @com_github_zml_cdn_mnist//file)",
        "$(location @com_github_zml_cdn_mnist_data//file)",
    ],
    data = [
        "@com_github_zml_cdn_mnist//file",
        "@com_github_zml_cdn_mnist_data//file",
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

