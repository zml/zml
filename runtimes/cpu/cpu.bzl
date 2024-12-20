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
        sha256 = "646b8ea61e690af0e4133637343674fb072e7d5e3a29694e6f84bb66ea75a6f0",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v3.0.0/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD.format(ext = "dylib"),
        sha256 = "33b1a5c670d0dda8852cfe9d0ef68467003e0669a7a4b043e0954852385c91df",
        url = "http://localhost:8000/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
