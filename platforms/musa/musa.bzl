"""Consume the complete MUSA sandbox produced by XLA."""

load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

_PLUGIN_SHA256 = "9212c278e1ccc68854ed4aa2abfa73e749dce88f69a0d11d9e45466aca757f6a"

def _musa_impl(mctx):
    http_archive(
        name = "libzml_musa",
        build_file = "libzml_musa.BUILD.bazel",
        url = "https://mirror.zml.ai/plugins/sha256/{}/zml-musa-linux-amd64.tar.zst".format(_PLUGIN_SHA256),
        sha256 = _PLUGIN_SHA256,
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_musa"],
        root_module_direct_dev_deps = [],
    )

musa_packages = module_extension(
    implementation = _musa_impl,
)
