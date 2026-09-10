load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libpjrt_metal",
        build_file = "libpjrt_metal.BUILD.bazel",
        sha256 = "30ba8ecc14d38c609dd5ead280ae55424d6d7e25a11c2236ec172a818271fb95",
        url = "https://mirror.zml.ai/pjrt-plugins/20260910.3cf307d5d5df.17.1/pjrt-metal-darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
