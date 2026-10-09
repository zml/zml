"""Consume the complete Furiosa sandbox produced by XLA."""

load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _furiosa_impl(mctx):
    http_archive(
        name = "libzml_furiosa",
        build_file = "libzml_furiosa.BUILD.bazel",
        url = "https://mirror.zml.ai/plugins/202610081500.175.1.4a98481373f6/zml-furiosa-linux-amd64.tar.zst",
        sha256 = "fb5569b2c3ab8604924fb66b2e9f6fd3fe38de32f58aaec25ef1aa510dfe10ee",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_furiosa"],
        root_module_direct_dev_deps = [],
    )

furiosa_packages = module_extension(
    implementation = _furiosa_impl,
)
