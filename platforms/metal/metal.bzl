load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libpjrt_metal",
        build_file = "libpjrt_metal.BUILD.bazel",
        sha256 = "d6dac3b0df3390f56e4f0fa41241e949ce32d6266b2f7ed2aa677be4c942acce",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-06-23T00-20-00Z/pjrt-metal_macos-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
