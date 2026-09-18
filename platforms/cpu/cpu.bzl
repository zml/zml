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
        sha256 = "5d6783ccd775a74dbe8d03e9f35e7a5a422ef051f56bb4266ac918bcb5f56970",
        url = "https://mirror.zml.ai/plugins/202609161604.45.1.80cbf79212e9/zml-cpu-linux-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "3535dee8b26bcf3e1906779b103bcbf8903c88ca93b85487d98fd2181e4f99be",
        url = "https://mirror.zml.ai/plugins/202609161604.45.1.80cbf79212e9/zml-cpu-darwin-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "bf5c036ba2a8195f8449119fa6138c7b83d7382c5732c2149d74a7b901095ffd",
        url = "https://mirror.zml.ai/plugins/202609161604.45.1.80cbf79212e9/zml-cpu-darwin-arm64.tar.zst",
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
