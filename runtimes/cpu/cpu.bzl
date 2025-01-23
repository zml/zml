load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_BUILD = """\
cc_import(
    name = "libpjrt_cpu",
    shared_library = "libpjrt_cpu.{ext}",
    visibility = ["//visibility:public"],
)
"""

def _cpu_pjrt_plugin_impl(mctx):
    http_archive(
        name = "libpjrt_cpu_linux_amd64",
        build_file_content = _BUILD.format(ext = "so"),
        sha256 = "0f2cb204015e062df5d1cbd39d8c01c076ab2b004d0f4f37f6d5e120d3cd7087",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v5.0.0/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD.format(ext = "dylib"),
        sha256 = "2ddb66a93c8a913e3bc8f291e01df59aa297592cc91e05aab2dd4813884098cb",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v5.0.0/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
