load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tpu_impl(mctx):
    # https://storage.googleapis.com/jax-releases/libtpu_releases.html
    http_archive(
        name = "libpjrt_tpu",
        url = "https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu/libtpu-0.0.42.dev20260613+nightly-cp314-cp314t-manylinux_2_31_x86_64.whl",
        type = "zip",
        sha256 = "1d0bbb1608154bda6944902abd6f56f956638539d7503db4894927e21a41baed",
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
