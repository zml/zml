load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _tpu_impl(mctx):
    # https://storage.googleapis.com/jax-releases/libtpu_releases.html
    # Keep requirement.in jax version in sync with the version here.
    http_archive(
        name = "libpjrt_tpu",
        url = "https://files.pythonhosted.org/packages/3e/88/d10f7a8429502759e72078d08213fd07eadc023091516b95717a8f506e61/libtpu-0.0.37-cp314-cp314t-manylinux_2_31_x86_64.whl",
        type = "zip",
        sha256 = "4d61b54e2c9a6be86a86436f55dffd89a47a299b46b20919a201e957b702b2ad",
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
