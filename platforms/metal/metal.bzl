load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libpjrt_metal",
        build_file = "libpjrt_metal.BUILD.bazel",
        sha256 = "f55356328d9d814c5a961b6c38741cff2f98b4043846a7ac3865e2a23549b767",
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
