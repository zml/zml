load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_BUILD_LINUX = """\
load("@zml//bazel:cc_import.bzl", "cc_import")
cc_import(
    name = "libpjrt_cpu",
    shared_library = "libpjrt_cpu.so",
    soname = "libpjrt_cpu.so",
    visibility = ["@zml//runtimes/cpu:__subpackages__"],
)
"""

_BUILD_DARWIN = """\
cc_import(
    name = "libpjrt_cpu",
    shared_library = "libpjrt_cpu.dylib",
    visibility = ["@zml//runtimes/cpu:__subpackages__"],
)
"""

def _cpu_pjrt_plugin_impl(mctx):
    http_archive(
        name = "libpjrt_cpu_linux_amd64",
        build_file_content = _BUILD_LINUX,
        sha256 = "e17d15331a3f42b90c8131459c235a9cf1145913a581c8b192845c50d313a7d6",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v6.0.0/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_DARWIN,
        sha256 = "2e18f8426ec5fa698163d55871fdbaed3616325cab10fad54f9d14f265fbf00d",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v6.0.0/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
