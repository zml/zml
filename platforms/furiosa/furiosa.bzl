"""Pinned host libraries and hermetic target headers for the Furiosa sandbox."""

load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("@with_cfg.bzl", "with_cfg")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

# TCC targets AArch64 even when the plugin runs on an x86-64 host.
aarch64_headers, _aarch64_headers = with_cfg(native.filegroup).set(
    "platforms",
    [Label("@llvm//platforms:linux_aarch64_gnu.2.39")],
).build()

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

# Compiler binaries, C++ runtimes and target headers come from the LLVM toolchain.
# TCC and the plugin come from XLA; their remaining host libraries are pinned here.

_DEB_PACKAGES = {
    "libc6": """
exports_files([
    "usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
    "usr/lib/x86_64-linux-gnu/libc.so.6",
    "usr/lib/x86_64-linux-gnu/libdl.so.2",
    "usr/lib/x86_64-linux-gnu/libm.so.6",
    "usr/lib/x86_64-linux-gnu/libpthread.so.0",
])

filegroup(
    name = "files",
    srcs = [
        "usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
        "usr/lib/x86_64-linux-gnu/libc.so.6",
        "usr/lib/x86_64-linux-gnu/libdl.so.2",
        "usr/lib/x86_64-linux-gnu/libm.so.6",
        "usr/lib/x86_64-linux-gnu/libpthread.so.0",
    ],
)
""",
    "libgcc-s1": """
filegroup(
    name = "files",
    srcs = ["usr/lib/x86_64-linux-gnu/libgcc_s.so.1"],
)
""",
}

def _read_packages(mctx, labels):
    ret = {}
    for label in labels:
        data = json.decode(mctx.read(Label(label)))
        for pkg in data["packages"]:
            ret.setdefault(pkg["name"], {})[pkg["arch"]] = pkg
    return ret

def _furiosa_impl(mctx):
    loaded_packages = _read_packages(mctx, [
        "@zml//platforms/furiosa:packages.lock.json",
    ])

    http_archive(
        name = "libzml_furiosa",
        build_file = "libzml_furiosa.BUILD.bazel",
        url = "https://mirror.zml.ai/plugins/202610021553.141.1.aa72501708f2/zml-furiosa-linux-amd64.tar.zst",
        sha256 = "f8909dc9c06709b50bf89679231dd8a959e9117027c184b41367fe2ab133f4da",
    )

    for pkg_name, build_file_content in _DEB_PACKAGES.items():
        pkg = loaded_packages[pkg_name]["amd64"]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_furiosa"],
        root_module_direct_dev_deps = [],
    )

furiosa_packages = module_extension(
    implementation = _furiosa_impl,
)
