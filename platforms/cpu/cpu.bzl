load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_BUILD_LINUX = "\n".join([
    packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
    packages.patchelf(
        name = "libzml_cpu_so",
        src = "lib/libzml_cpu.so",
        set_rpath = "$ORIGIN",
    ),
    packages.filegroup(
        name = "libzml_cpu",
        srcs = [":libzml_cpu_so", "@llvm-libunwind1//:libunwind"],
        visibility = ["@zml//platforms/cpu:__subpackages__"],
    ),
])

_BUILD_DARWIN = packages.filegroup(
    name = "libzml_cpu",
    srcs = ["lib/libzml_cpu.dylib"],
    visibility = ["@zml//platforms/cpu:__subpackages__"],
)

def _cpu_plugin_impl(mctx):
    loaded_packages = packages.read(mctx, ["@zml//platforms/cpu:packages.lock.json"])
    pkg = loaded_packages["llvm-libunwind1"]["amd64"]
    http_deb_archive(
        name = "llvm-libunwind1",
        urls = pkg["urls"],
        sha256 = pkg["sha256"],
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + packages.filegroup(
            name = "libunwind",
            srcs = ["usr/lib/x86_64-linux-gnu/libunwind.so.1"],
        ),
    )

    http_archive(
        name = "libzml_cpu_linux_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_LINUX,
        sha256 = "3d6a6d1b047a0943b50fd9e63e5f6dff6b7cf184b186c0cc1052de40f0dbec92",
        url = "https://mirror.zml.ai/plugins/202610281508.306.1.6bf391c563e0/zml-cpu-linux-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "d6b4b15b13a2d1a3e0b84f143f568cd572bef5db91784f9f7a826173ed626c7e",
        url = "https://mirror.zml.ai/plugins/202610281508.306.1.6bf391c563e0/zml-cpu-darwin-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "c06babc001edeab313bc63daab3aa3d40e087266a3a29634d182628fb65fad14",
        url = "https://mirror.zml.ai/plugins/202610281508.306.1.6bf391c563e0/zml-cpu-darwin-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "libzml_cpu_linux_amd64",
            "libzml_cpu_darwin_amd64",
            "libzml_cpu_darwin_arm64",
        ],
        root_module_direct_dev_deps = [],
    )

cpu_plugin = module_extension(
    implementation = _cpu_plugin_impl,
)
