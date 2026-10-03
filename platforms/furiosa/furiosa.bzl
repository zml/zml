"""Pinned host runtime libraries and a locally built PJRT plugin/SDK."""

load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//bazel:simple_repository.bzl", "simple_repository")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

# The compiler SDK is supplied with the plugin. Only its host plugin runtime
# libraries are assembled here.
_DEB_PACKAGES = {
    "libgcc-s1": 'filegroup(name = "files", srcs = ["usr/lib/x86_64-linux-gnu/libgcc_s.so.1"])',
    "libstdc++6": 'filegroup(name = "files", srcs = ["usr/lib/x86_64-linux-gnu/libstdc++.so.6"])',
    "llvm-libunwind1": 'filegroup(name = "runtime", srcs = ["usr/lib/x86_64-linux-gnu/libunwind.so.1"])',
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
alias(name = "compiler_sdk", actual = ":libzml_furiosa")
alias(
    name = "lib/libpjrt_c_api_furiosa_plugin.so",
    actual = ":libzml_furiosa",
)
""")
    rctx.file("missing.bzl", """
def _impl(ctx):
    fail("Furiosa requires a matching staged PJRT plugin and compiler_sdk: use --override_repository=libzml_furiosa=/path/to/xla-override")

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

    for pkg_name, build_file_content in _DEB_PACKAGES.items():
        pkg = loaded_packages[pkg_name]["amd64"]
        repo = "furiosa_deb_" + pkg_name.replace("+", "p").replace("-", "_")
        http_deb_archive(
            name = repo,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
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
