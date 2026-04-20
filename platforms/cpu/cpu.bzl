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
        sha256 = "867ea7129a61eccb2170fb928cf008577431b1ed5b1ef317766e14deb9660513",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/nightly-2026-04-20/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "sha256:12515794094962e37b15c2527e4580b8802f83839d13801937191c1ab4b49a2e",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/nightly-2026-04-20/pjrt-cpu_darwin-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "e1a648ce12d25ce04aa107b4cfa3aa4c5f675968488779fcbf17b3a98c207da3",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/nightly-2026-04-20/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
