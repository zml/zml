load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libzml_metal",
        build_file = "libzml_metal.BUILD.bazel",
        sha256 = "176b9d08cba22871a5c53e33da26235509612f6ce22247e3ee3d6ea8f5d206b0",
        url = "https://mirror.zml.ai/plugins/202609250917.103.1.3c8a2e14f8d9/zml-metal-darwin-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
