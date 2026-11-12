"""Consume the complete Furiosa sandbox produced by XLA."""

load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _furiosa_impl(mctx):
    http_archive(
        name = "libzml_furiosa",
        build_file = "libzml_furiosa.BUILD.bazel",
        url = "https://mirror.zml.ai/plugins/202611121233.371.1.9ff7ddf1e19f/zml-furiosa-linux-amd64.tar.zst",
        sha256 = "d732b56832b1f009cbe4d808c70c6f8d1dc3857a83eb06804fc53b39bef9dd6b",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_furiosa"],
        root_module_direct_dev_deps = [],
    )

furiosa_packages = module_extension(
    implementation = _furiosa_impl,
)
