"""Consume the complete Furiosa sandbox produced by XLA."""

load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _furiosa_impl(mctx):
    http_archive(
        name = "libzml_furiosa",
        build_file = "libzml_furiosa.BUILD.bazel",
        url = "https://mirror.zml.ai/plugins/202610091428.181.1.bde986653bf6/zml-furiosa-linux-amd64.tar.zst",
        sha256 = "757493c17aa7fbd6217a1d8f5f9172fea67a19eeb7da0c92e7aebff6033d27ef",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_furiosa"],
        root_module_direct_dev_deps = [],
    )

furiosa_packages = module_extension(
    implementation = _furiosa_impl,
)
