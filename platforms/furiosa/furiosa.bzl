"""Consume the complete Furiosa sandbox produced by XLA."""

load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

def _furiosa_impl(mctx):
    http_archive(
        name = "libzml_furiosa",
        build_file = "libzml_furiosa.BUILD.bazel",
        url = "https://mirror.zml.ai/plugins/202610191219.248.1.c2619532effe/zml-furiosa-linux-amd64.tar.zst",
        sha256 = "3913a592fbc35a50ac60700011529addb9c18a51b614dfc5d1aa4f346b500712",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_furiosa"],
        root_module_direct_dev_deps = [],
    )

furiosa_packages = module_extension(
    implementation = _furiosa_impl,
)
