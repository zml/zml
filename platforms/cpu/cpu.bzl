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
        sha256 = "6397644e129e198431e56d8fee6871d5d9a71550dc6ca03a5abbf0a9ef3a3eaf",
        url = "https://mirror.zml.ai/plugins/202610091329.178.1.75e89af70664/zml-cpu-linux-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "8276a919970b9d5c2edf4cf6f962c689fa67c45d325f70849bfdd38403f8fb8b",
        url = "https://mirror.zml.ai/plugins/202610091329.178.1.75e89af70664/zml-cpu-darwin-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "e08eb1986a8705c66f065d15c0cacf228219118aca7c44148e47502f011982b1",
        url = "https://mirror.zml.ai/plugins/202610091329.178.1.75e89af70664/zml-cpu-darwin-arm64.tar.zst",
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
