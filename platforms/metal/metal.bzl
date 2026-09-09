load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libpjrt_metal",
        build_file = "libpjrt_metal.BUILD.bazel",
        sha256 = "ef66435cc0290a608ddbdd69f3b1e8a10c461d9cab73e6ed35e976b96e13ea50",
        url = "https://mirror.zml.ai/pjrt-plugins/20260909.69bab05e48b6.14.1/pjrt-metal-darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
