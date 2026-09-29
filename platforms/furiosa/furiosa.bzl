"""Pinned compiler dependencies and a locally built PJRT/runtime payload."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//bazel:simple_repository.bzl", "simple_repository")
load("//platforms:packages.bzl", "packages")

# TCC links with -nostdlib. Sanitizer and other optional target runtimes from
# the full package lock are not needed by the compiler's generated C code.
_TOOLCHAIN_PACKAGES = [
    "binutils-aarch64-linux-gnu",
    "cpp-13-aarch64-linux-gnu",
    "gcc-13-aarch64-linux-gnu",
    "gcc-13-aarch64-linux-gnu-base",
    "gcc-13-cross-base",
    "gcc-14-base",
    "libc6-arm64-cross",
    "libc6-dev-arm64-cross",
    "libgcc-13-dev-arm64-cross",
    "libgcc-s1",
    "libgmp10",
    "libisl23",
    "libjansson4",
    "libmpc3",
    "libmpfr6",
    "libsframe1",
    "libstdc++6",
    "libzstd1",
    "linux-libc-dev-arm64-cross",
    "zlib1g",
]

_LIBRARIES = [
    "libpjrt_c_api_furiosa_plugin.so",
    "libdevice_runtime.so",
    "libstdc++.so.6",
    "libunwind.so.1",
]

def _placeholder_impl(ctx):
    ctx.file("BUILD.bazel", '''
load(":missing.bzl", "missing_plugin")
missing_plugin(name = "libzml_furiosa", visibility = ["//visibility:public"])
''' + "\n".join([
        'alias(name = "lib/{}", actual = ":libzml_furiosa", visibility = ["//visibility:public"])'.format(name)
        for name in _LIBRARIES
    ]))
    ctx.file("missing.bzl", '''
def _impl(ctx):
    fail("Furiosa requires a staged PJRT plugin and bridge: use --override_repository=libzml_furiosa=/path/to/xla-override (see platforms/furiosa/README.md)")
missing_plugin = rule(implementation = _impl)
''')

_placeholder = repository_rule(implementation = _placeholder_impl)

def _furiosa_impl(ctx):
    _placeholder(name = "libzml_furiosa")
    http_archive(
        name = "furiosa_tcc",
        urls = ["https://files.pythonhosted.org/packages/48/45/dfc28eedd7cbdc4cb80bbf17d6221a501d911ab2fb53bdced28339c368b6/furiosa_tcc-2026.3.0-py3-none-manylinux2014_x86_64.whl"],
        sha256 = "61ca63feb35f04c1998def992141f134eb78d9a783ca031c87b5298129776ef1",
        build_file_content = '''
package(default_visibility = ["//visibility:public"])
filegroup(name = "compiler", srcs = ["furiosa_tcc-2026.3.0.data/scripts/furiosa-tcc"])
filegroup(name = "licenses", srcs = glob(["**/LICENSE*", "**/NOTICE*", "**/METADATA"]))
''',
    )
    locked = packages.read(ctx, ["//platforms/furiosa:packages.lock.json"])
    repos = []
    for name in _TOOLCHAIN_PACKAGES:
        pkg = locked[name]["amd64"]
        repo = "furiosa_deb_" + name.replace("+", "p").replace("-", "_")
        repos.append("@{}//:files".format(repo))
        http_deb_archive(
            name = repo,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = '''
filegroup(
    name = "files",
    srcs = glob([
        "usr/bin/aarch64-linux-gnu-*",
        "usr/libexec/gcc-cross/**",
        "usr/lib/gcc-cross/**/include/**",
        "usr/lib/gcc-cross/**/libgcc*.a",
        "usr/aarch64-linux-gnu/**",
        "usr/lib/x86_64-linux-gnu/**",
        "lib/x86_64-linux-gnu/**",
        "usr/share/doc/**/copyright",
    ]),
    visibility = ["//visibility:public"],
)
''',
        )
    simple_repository(
        name = "furiosa_toolchain",
        build_file_content = 'filegroup(name = "files", srcs = {}, visibility = ["//visibility:public"])'.format(repr(repos)),
    )
    return ctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_furiosa", "furiosa_tcc", "furiosa_toolchain"],
        root_module_direct_dev_deps = [],
    )

furiosa_packages = module_extension(implementation = _furiosa_impl)
