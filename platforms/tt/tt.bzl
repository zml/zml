load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_UBUNTU_PACKAGES = {
    "libopenmpi3": """\
load("@zml//bazel:patchelf.bzl", "patchelf")

filegroup(
    name = "libopenmpi3",
    srcs = glob(["usr/lib/x86_64-linux-gnu/**"]),
)
""",
    "libevent-pthreads-2.1-7": """\
filegroup(
    name = "libevent-pthreads-2.1-7",
    srcs = glob(["usr/lib/x86_64-linux-gnu/**"]),
    )
""",
    "libhwloc15": packages.filegroup(name = "libhwloc15", srcs = ["usr/lib/x86_64-linux-gnu/libhwloc.so.15"]),
    "libhwloc-plugins": """\
filegroup(
    name = "libhwloc-plugins",
    srcs = glob(["usr/lib/x86_64-linux-gnu/**"]),
)
""",
    "libprotobuf23": packages.filegroup(name = "libprotobuf23", srcs = ["usr/lib/x86_64-linux-gnu/libprotobuf.so.23"]),
    "libnsl2": packages.filegroup(name = "libnsl2", srcs = ["usr/lib/x86_64-linux-gnu/libnsl.so.2"]),
    "zlib1g": packages.filegroup(name = "zlib1g", srcs = ["lib/x86_64-linux-gnu/libz.so.1"]),
}


def _tt_impl(mctx):
    loaded_packages = packages.read(mctx, [
        "@zml//platforms/tt:packages.lock.json",
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
        strip_prefix = "pjrt_plugin_tt",
        url = "https://github.com/tenstorrent/tt-xla/releases/download/0.8.0.dev20260120/pjrt_plugin_tt-0.8.0.dev20260120-cp311-cp311-linux_x86_64.whl",
        sha256 = "9175b12d94bc9795e51792e6339c99eadd81c1dd9d37cc0cc6285705eba4520a",
        patch_cmds = [
            "find tt-metal -name 'BUILD.bazel' -delete",
        ],
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_tt"],
        root_module_direct_dev_deps = [],
    )

tt_packages = module_extension(
    implementation = _tt_impl,
)
