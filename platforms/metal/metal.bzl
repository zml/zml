load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libzml_metal",
        build_file = "libzml_metal.BUILD.bazel",
        sha256 = "b7c1a5c4b30fb70c7901f41a003da6cb5bb4bb2f37f965b92c14656ae7cc6dca",
        url = "https://mirror.zml.ai/plugins/202610141143.207.1.b026d193d9cb/zml-metal-darwin-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
