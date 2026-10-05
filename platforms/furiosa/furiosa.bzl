"""Consume the complete Furiosa sandbox produced by XLA."""

load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _furiosa_impl(mctx):
    # Update URL and SHA together after publishing XLA's complete archive.
    # Until then, test with --override_repository=libzml_furiosa=... .
    http_archive(
        name = "libzml_furiosa",
        build_file = "libzml_furiosa.BUILD.bazel",
        url = "https://mirror.zml.ai/plugins/202610021553.141.1.aa72501708f2/zml-furiosa-linux-amd64.tar.zst",
        sha256 = "f8909dc9c06709b50bf89679231dd8a959e9117027c184b41367fe2ab133f4da",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_furiosa"],
        root_module_direct_dev_deps = [],
    )

furiosa_packages = module_extension(
    implementation = _furiosa_impl,
)
