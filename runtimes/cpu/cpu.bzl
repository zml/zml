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
        sha256 = "14f0cd3541532b648b61c4f7ad66c27f37be9051876f71196c692abd9dd1a13b",  # todo: change checksum
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v4.0.0/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD.format(ext = "dylib"),
        sha256 = "0993df85f38efc2d630b1412e3cde6e131933ccaa5502444e35e4aff5ec84de3",  # todo: change checksum
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v4.0.0/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
