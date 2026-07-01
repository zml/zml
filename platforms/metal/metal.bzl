load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libpjrt_metal",
        build_file = "libpjrt_metal.BUILD.bazel",
        sha256 = "9222a79886019a50ccfb15ce815acbad348210a6e8afc02d8f850e6bc8244da3",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-06-30T00-16-30Z/pjrt-metal_macos-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
