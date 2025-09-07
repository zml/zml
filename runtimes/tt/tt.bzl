load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//runtimes/common:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_UBUNTU_PACKAGES = {
    "libopenmpi3": """\
filegroup(
    name = "libopenmpi3",
    srcs = glob(["usr/lib/x86_64-linux-gnu/**"]),
)
""",
    # "libhwloc15": packages.filegroup(name = "libhwloc15", srcs = ["usr/lib/x86_64-linux-gnu/libhwloc.so.15"]),
    # "libhwloc-plugins": packages.filegroup(name = "libhwloc-plugins", srcs = packages.glob(["usr/lib/x86_64-linux-gnu/**"])),
    "libprotobuf23": packages.filegroup(name = "libprotobuf23", srcs = ["usr/lib/x86_64-linux-gnu/libprotobuf.so.23"]),
    # "libnsl2": packages.filegroup(name = "libnsl2", srcs = ["usr/lib/x86_64-linux-gnu/libnsl.so.2"]),
    # "zlib1g": packages.filegroup(name = "zlib1g", srcs = ["lib/x86_64-linux-gnu/libz.so.1"]),
}


def _tt_impl(mctx):
    loaded_packages = packages.read(mctx, [
        "@zml//runtimes/tt:packages.lock.json",
    ])

    for pkg_name, build_file_content in _UBUNTU_PACKAGES.items():
        pkg = loaded_packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    http_archive(
        name = "libpjrt_tt",
        build_file = "libpjrt_tt.BUILD.bazel",
        strip_prefix = "pjrt_plugin_tt-0.3.0.data",
        url = "https://github.com/tenstorrent/tt-xla/releases/download/0.3.0/pjrt_plugin_tt-0.3.0-py3-none-linux_x86_64.whl",
        sha256 = "986c8b06d34bbf322f0f6c5d8646e3903cd6e334d0ca927143d612f810c4ba1e",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_tt"],
        root_module_direct_dev_deps = [],
    )

tt_packages = module_extension(
    implementation = _tt_impl,
)
