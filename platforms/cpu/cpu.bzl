load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//platforms:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_BUILD_LINUX = "\n".join([
    packages.filegroup(
        name = "libpjrt_cpu",
        srcs = ["libpjrt_cpu.so"],
        visibility = ["@zml//platforms/cpu:__subpackages__"],
    ),
])

_BUILD_DARWIN = packages.filegroup(
    name = "libpjrt_cpu",
    srcs = ["libpjrt_cpu.dylib"],
    visibility = ["@zml//platforms/cpu:__subpackages__"],
)

def _cpu_pjrt_plugin_impl(mctx):
    http_archive(
        name = "libpjrt_cpu_linux_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_LINUX,
        sha256 = "65e631db0f842845e7799d245a414b361a3c3e77bf4cc0547c20c71f28a9fd70",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-07-03T00-10-30Z/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "22130f752abdaaa0f3ff48e71b9191ce74b29cd3fdb0c4784f04e4c4a436f25e",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-07-03T00-10-30Z/pjrt-cpu_darwin-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "14c85504d801c75fa8d157ce951a2644d8a8d7983346b3ac281aa7f64abf8390",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-07-03T00-10-30Z/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
