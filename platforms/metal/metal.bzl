load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libpjrt_metal",
        build_file = "libpjrt_metal.BUILD.bazel",
        sha256 = "0751be83d707fdc74e307b48b719a032d9f5c96d7be4c123e0b1b3b8c8906eea",
        url = "https://mirror.zml.ai/pjrt-plugins/202609100934.19.1.c22f39a3b260/pjrt-metal-darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
