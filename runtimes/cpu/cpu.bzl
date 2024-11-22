load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_BUILD = """\
cc_import(
    name = "libpjrt_cpu",
    shared_library = "libpjrt_cpu.{ext}",
    visibility = ["//visibility:public"],
)
"""

def _cpu_pjrt_plugin_impl(mctx):
    # custom plugin serving : tar -zc libpjrt_cpu.so > pjrt-cpu_linux-amd64.tar.gz | sha256sum pjrt-cpu_linux-amd64.tar.gz |  python3 -m http.server 8000
    http_archive(
        name = "libpjrt_cpu_linux_amd64",
        build_file_content = _BUILD.format(ext = "so"),
        sha256 = "e09e8e8c0dee87c782f3a1ae19663d55df387c3f11d20704077818cd7f37e9c7",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v0.2.4/pjrt-cpu_linux-amd64.tar.gz",
    )

    # custom plugin serving : tar -zc libpjrt_cpu.dylib > pjrt-cpu_darwin-arm64.tar.gz | sha256sum pjrt-cpu_darwin-arm64.tar.gz |  python3 -m http.server 8000
    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD.format(ext = "dylib"),
        sha256 = "09a23422377ff80a42162f13ec0084b3c125dec6e15765ae6027d94b8f223dda",
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
