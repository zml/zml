"""Pinned libraries and target headers for the Furiosa compiler sandbox."""

load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

# The compiler binaries come from ZML's hermetic LLVM toolchain. TCC and the
# plugin come from the XLA archive; their host libraries are pinned here.

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
    "libc6-dev-arm64-cross": """
# Transitive headers used by TCC's pe.h/tuc.h with the interceptor's C11 flags.
filegroup(
    name = "headers",
    srcs = [
        "usr/aarch64-linux-gnu/include/assert.h",
        "usr/aarch64-linux-gnu/include/bits/libc-header-start.h",
        "usr/aarch64-linux-gnu/include/bits/long-double.h",
        "usr/aarch64-linux-gnu/include/bits/stdint-intn.h",
        "usr/aarch64-linux-gnu/include/bits/stdint-least.h",
        "usr/aarch64-linux-gnu/include/bits/stdint-uintn.h",
        "usr/aarch64-linux-gnu/include/bits/time64.h",
        "usr/aarch64-linux-gnu/include/bits/timesize.h",
        "usr/aarch64-linux-gnu/include/bits/types.h",
        "usr/aarch64-linux-gnu/include/bits/typesizes.h",
        "usr/aarch64-linux-gnu/include/bits/wchar.h",
        "usr/aarch64-linux-gnu/include/bits/wordsize.h",
        "usr/aarch64-linux-gnu/include/features-time64.h",
        "usr/aarch64-linux-gnu/include/features.h",
        "usr/aarch64-linux-gnu/include/gnu/stubs-lp64.h",
        "usr/aarch64-linux-gnu/include/gnu/stubs.h",
        "usr/aarch64-linux-gnu/include/stdc-predef.h",
        "usr/aarch64-linux-gnu/include/stdint.h",
        "usr/aarch64-linux-gnu/include/string.h",
        "usr/aarch64-linux-gnu/include/sys/cdefs.h",
    ],
)
""",
    "libgcc-s1": """
filegroup(
    name = "files",
    srcs = ["usr/lib/x86_64-linux-gnu/libgcc_s.so.1"],
)
""",
    "libstdc++6": """
filegroup(
    name = "files",
    srcs = ["usr/lib/x86_64-linux-gnu/libstdc++.so.6"],
)
""",
    "llvm-libunwind1": """
filegroup(
    name = "runtime",
    srcs = ["usr/lib/x86_64-linux-gnu/libunwind.so.1"],
)
""",
}

_REPO_NAMES = {
    "libc6": "libc6",
    "libc6-dev-arm64-cross": "libc6-dev-arm64-cross",
    "libgcc-s1": "libgcc-s1",
    "libstdc++6": "libstdcpp6",
    "llvm-libunwind1": "llvm-libunwind1",
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
            name = _REPO_NAMES[pkg_name],
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
