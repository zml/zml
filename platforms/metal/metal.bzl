load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libzml_metal",
        build_file = "libzml_metal.BUILD.bazel",
        sha256 = "89881d1a1db336e3337a9373db2c4e94f71ddfb77619dee75c0c12e2903913ae",
        url = "https://mirror.zml.ai/plugins/202611101651.363.1.d46c7471c419/zml-metal-darwin-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
