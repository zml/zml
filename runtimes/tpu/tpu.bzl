load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _tpu_impl(mctx):
    # https://storage.googleapis.com/jax-releases/libtpu_releases.html
    http_archive(
        name = "libpjrt_tpu",
        url = "https://storage.googleapis.com/libtpu-lts-releases/wheels/libtpu/libtpu-0.0.19.1-py3-none-manylinux_2_31_x86_64.whl",
        type = "zip",
        sha256 = "373368791794e2ffa20d48ae522232c3146918e138825fe8f7c29b9eae8544e5",
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
