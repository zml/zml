load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libzml_metal",
        build_file = "libzml_metal.BUILD.bazel",
        sha256 = "bdd2de3ec8293faa25b358cdf35ee3e75d69be73a4acea87b6e007a18b8f10e8",
        url = "https://mirror.zml.ai/plugins/202611041808.348.1.a83ad22d7ed5/zml-metal-darwin-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
