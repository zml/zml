load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libpjrt_metal",
        build_file = "libpjrt_metal.BUILD.bazel",
        sha256 = "f37434d65a486f0896e9f98b4a5c27eafc50372b0986886b4df62e4f2edb68b1",
        url = "https://mirror.zml.ai/pjrt-plugins/20260910.c6a043b8e354.16.1/pjrt-metal-darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
