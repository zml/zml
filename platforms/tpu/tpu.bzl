load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tpu_impl(mctx):
    # https://storage.googleapis.com/jax-releases/libtpu_releases.html
    http_archive(
        name = "libzml_tpu",
        url = "https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu/libtpu-0.0.50.dev20261005+nightly-cp314-cp314t-manylinux_2_31_x86_64.whl",
        type = "zip",
        sha256 = "a1e8615a33f60e1bfce15ec9d570642f882a8bec1d46763531e3b831edb2bb42",
        build_file = "libzml_tpu.BUILD.bazel",
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_tpu"],
        root_module_direct_dev_deps = [],
    )

tpu_packages = module_extension(
    implementation = _tpu_impl,
)
