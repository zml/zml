load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tpu_impl(mctx):
    # https://storage.googleapis.com/jax-releases/libtpu_releases.html
    http_archive(
        name = "libpjrt_tpu",
        url = "https://files.pythonhosted.org/packages/5e/ea/6271a8906d5509349dde55da1f516ac232e73b3cd8e1fba095f4132142bf/libtpu-0.0.36-cp314-cp314t-manylinux_2_31_x86_64.whl",
        type = "zip",
        sha256 = "6d0e6a2ec26b851f5a00c74933738b2b185af47c4eacbd161e23954b1d911ae4",
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
