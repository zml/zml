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
        sha256 = "c40cd5d6c6210c714e5508e36f86a2d363a2043cc56cf028e70ef075b32bca32",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v5.1.0/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_DARWIN,
        sha256 = "e5a69f7f7d3663fc8779b1e756bd2941bfef6618b64da4dfd311c2fa39e3708f",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v5.1.0/pjrt-cpu_darwin-arm64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_amd64",
        build_file_content = _BUILD_DARWIN,
        sha256 = "3ddcb1642b06b0c6d2528790648454d68a67bd113af1eb3a7b7f2e3f03606a44",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v5.1.0/pjrt-cpu_darwin-amd64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
