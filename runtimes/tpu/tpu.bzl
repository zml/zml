load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _tpu_impl(mctx):
    # https://storage.googleapis.com/jax-releases/libtpu_releases.html
    http_archive(
        name = "libpjrt_tpu",
        url = "https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20250807+nightly-py3-none-manylinux_2_31_x86_64.whl",
        type = "zip",
        sha256 = "41c19fa5ae4a32fbd05f0260527ba2d93afb6cf128e6c4de7773e9011c7b3df5",
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
