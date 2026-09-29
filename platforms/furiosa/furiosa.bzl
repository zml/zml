"""Pinned compiler dependencies and a locally built PJRT plugin."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//bazel:simple_repository.bzl", "simple_repository")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

# TCC links with -nostdlib. Sanitizer and other optional target runtimes from
# the full package lock are not needed by the compiler's generated C code.
# libstdc++ and libgcc_s are shared with the plugin through furiosa_runtime.
_TOOLCHAIN_PACKAGES = [
    "binutils-aarch64-linux-gnu",
    "cpp-13-aarch64-linux-gnu",
    "gcc-13-aarch64-linux-gnu",
    "libc6-dev-arm64-cross",
    "libgcc-13-dev-arm64-cross",
    "libgmp10",
    "libisl23",
    "libjansson4",
    "libmpc3",
    "libmpfr6",
    "libsframe1",
    "libzstd1",
    "zlib1g",
]

# Explicit inputs for TCC 2026.3's generated C: compiler/linker binaries,
# their shared libraries and the pe.h standard-header dependencies.
# Revalidate this list with fresh compilation when upgrading the SDK.
_DEB_PACKAGES = {
    "binutils-aarch64-linux-gnu": """
filegroup(
    name = "files",
    srcs = [
        "usr/aarch64-linux-gnu/bin/as",
        "usr/aarch64-linux-gnu/bin/ld",
        "usr/lib/x86_64-linux-gnu/libbfd-2.42-arm64.so",
        "usr/lib/x86_64-linux-gnu/libctf-arm64.so.0",
        "usr/lib/x86_64-linux-gnu/libopcodes-2.42-arm64.so",
    ],
)""",
    "cpp-13-aarch64-linux-gnu": """
filegroup(
    name = "files",
    srcs = ["usr/libexec/gcc-cross/aarch64-linux-gnu/13/cc1"],
)""",
    "gcc-13-aarch64-linux-gnu": """
filegroup(
    name = "files",
    srcs = [
        "usr/libexec/gcc-cross/aarch64-linux-gnu/13/collect2",
    ],
)
exports_files(["usr/bin/aarch64-linux-gnu-gcc-13"])""",
    "libc6-dev-arm64-cross": """
filegroup(
    name = "files",
    srcs = [
        "usr/aarch64-linux-gnu/include/assert.h",
        "usr/aarch64-linux-gnu/include/bits/libc-header-start.h",
        "usr/aarch64-linux-gnu/include/bits/long-double.h",
        "usr/aarch64-linux-gnu/include/bits/stdint-intn.h",
        "usr/aarch64-linux-gnu/include/bits/stdint-least.h",
        "usr/aarch64-linux-gnu/include/bits/stdint-uintn.h",
        "usr/aarch64-linux-gnu/include/bits/string_fortified.h",
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
)""",
    "libgcc-13-dev-arm64-cross": """
filegroup(
    name = "files",
    srcs = [
        "usr/lib/gcc-cross/aarch64-linux-gnu/13/include/stdalign.h",
        "usr/lib/gcc-cross/aarch64-linux-gnu/13/include/stdarg.h",
        "usr/lib/gcc-cross/aarch64-linux-gnu/13/include/stdbool.h",
        "usr/lib/gcc-cross/aarch64-linux-gnu/13/include/stddef.h",
        "usr/lib/gcc-cross/aarch64-linux-gnu/13/include/stdint.h",
    ],
)""",
    "libgcc-s1": """
filegroup(
    name = "files",
    srcs = ["usr/lib/x86_64-linux-gnu/libgcc_s.so.1"],
)""",
    "libgmp10": """
filegroup(
    name = "files",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libgmp.so.10",
    ],
)""",
    "libisl23": """
filegroup(
    name = "files",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libisl.so.23",
    ],
)""",
    "libjansson4": """
filegroup(
    name = "files",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libjansson.so.4",
    ],
)""",
    "libmpc3": """
filegroup(
    name = "files",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libmpc.so.3",
    ],
)""",
    "libmpfr6": """
filegroup(
    name = "files",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libmpfr.so.6",
    ],
)""",
    "libsframe1": """
filegroup(
    name = "files",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libsframe.so.1",
    ],
)""",
    "libstdc++6": """
filegroup(
    name = "files",
    srcs = ["usr/lib/x86_64-linux-gnu/libstdc++.so.6"],
)""",
    "libzstd1": """
filegroup(
    name = "files",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libzstd.so.1",
    ],
)""",
    "zlib1g": """
filegroup(
    name = "files",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libz.so.1",
    ],
)""",
    "llvm-libunwind1": """
filegroup(
    name = "runtime",
    srcs = ["usr/lib/x86_64-linux-gnu/libunwind.so.1"],
)""",
}

_RUNTIME_BUILD_FILE = """
load("@zml//bazel:patchelf.bzl", "patchelf")

package(default_visibility = ["//visibility:public"])

patchelf(
    name = "libstdc++.so.6",
    src = "@furiosa_deb_libstdcpp6//:files",
    set_rpath = "$ORIGIN",
)
patchelf(
    name = "libgcc_s.so.1",
    src = "@furiosa_deb_libgcc_s1//:files",
    set_rpath = "$ORIGIN",
)
patchelf(
    name = "libunwind.so.1",
    src = "@furiosa_deb_llvm_libunwind1//:runtime",
    set_rpath = "$ORIGIN",
)
filegroup(
    name = "files",
    srcs = [
        ":libstdc++.so.6",
        ":libgcc_s.so.1",
        ":libunwind.so.1",
    ],
)
"""

_ROOT_MODULE_DIRECT_DEPS = [
    "libzml_furiosa",
    "furiosa_tcc",
    "furiosa_toolchain",
    "furiosa_runtime",
]

def _read_packages(mctx, labels):
    ret = {}
    for label in labels:
        data = json.decode(mctx.read(Label(label)))
        for pkg in data["packages"]:
            ret.setdefault(pkg["name"], {})[pkg["arch"]] = pkg
    return ret

def _placeholder_impl(rctx):
    rctx.file("BUILD.bazel", """
load(":missing.bzl", "missing_plugin")

package(default_visibility = ["//visibility:public"])

missing_plugin(name = "libzml_furiosa")
alias(
    name = "lib/libpjrt_c_api_furiosa_plugin.so",
    actual = ":libzml_furiosa",
)
""")
    rctx.file("missing.bzl", """
def _impl(ctx):
    fail("Furiosa requires a staged PJRT plugin: use --override_repository=libzml_furiosa=/path/to/xla-override")

missing_plugin = rule(implementation = _impl)
""")

_placeholder = repository_rule(
    implementation = _placeholder_impl,
)

def _furiosa_impl(mctx):
    loaded_packages = _read_packages(mctx, [
        "@zml//platforms/furiosa:packages.lock.json",
    ])

    _placeholder(name = "libzml_furiosa")

    http_archive(
        name = "furiosa_tcc",
        urls = ["https://files.pythonhosted.org/packages/48/45/dfc28eedd7cbdc4cb80bbf17d6221a501d911ab2fb53bdced28339c368b6/furiosa_tcc-2026.3.0-py3-none-manylinux2014_x86_64.whl"],
        sha256 = "61ca63feb35f04c1998def992141f134eb78d9a783ca031c87b5298129776ef1",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + """
filegroup(
    name = "compiler",
    srcs = ["furiosa_tcc-2026.3.0.data/scripts/furiosa-tcc"],
)
""",
    )

    toolchain_targets = []
    for pkg_name, build_file_content in _DEB_PACKAGES.items():
        pkg = loaded_packages[pkg_name]["amd64"]
        repo = "furiosa_deb_" + pkg_name.replace("+", "p").replace("-", "_")
        if pkg_name in _TOOLCHAIN_PACKAGES:
            toolchain_targets.append("@{}//:files".format(repo))
        http_deb_archive(
            name = repo,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    simple_repository(
        name = "furiosa_toolchain",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + """
filegroup(
    name = "files",
    srcs = {toolchain_targets},
)
alias(
    name = "gcc",
    actual = "@furiosa_deb_gcc_13_aarch64_linux_gnu//:usr/bin/aarch64-linux-gnu-gcc-13",
)
""".format(toolchain_targets = repr(toolchain_targets)),
    )
    simple_repository(
        name = "furiosa_runtime",
        build_file_content = _RUNTIME_BUILD_FILE,
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = _ROOT_MODULE_DIRECT_DEPS,
        root_module_direct_dev_deps = [],
    )

furiosa_packages = module_extension(
    implementation = _furiosa_impl,
)
