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
        sha256 = "4de73a067e552e83ec0188a75f408df048ae7fe0011695a811ea4eedbd686a84",
        url = "https://mirror.zml.ai/plugins/202611251235.407.1.475061b3c599/zml-cpu-linux-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "183cafb0d1e069d455b3ed2c8cbe51ee5e49f4ddefb39ed385b44d97d27f1e41",
        url = "https://mirror.zml.ai/plugins/202611251235.407.1.475061b3c599/zml-cpu-darwin-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "8109a9657b8bc8403570d6ef1234da84fe41f0ab2ce78f87173da689a9bd8e7e",
        url = "https://mirror.zml.ai/plugins/202611251235.407.1.475061b3c599/zml-cpu-darwin-arm64.tar.zst",
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
