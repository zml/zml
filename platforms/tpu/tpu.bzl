load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _tpu_impl(mctx):
    # https://storage.googleapis.com/jax-releases/libtpu_releases.html
    http_archive(
        name = "libpjrt_tpu",
        url = "https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu/libtpu-0.0.33.dev20251218+nightly-cp314-cp314t-manylinux_2_31_x86_64.whl",
        type = "zip",
        sha256 = "b7d9855c74a7fc56d11c4c7a2c55d5395c80594d43caec1549c69d5f908583cf",
        build_file = "libpjrt_tpu.BUILD.bazel",
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_tpu"],
        root_module_direct_dev_deps = [],
    )

tpu_packages = module_extension(
    implementation = _tpu_impl,
)
