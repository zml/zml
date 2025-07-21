load("@python_versions//3.11:defs.bzl", _py_binary = "py_binary")
load("@rules_python//python:defs.bzl", "PyInfo")
load("@with_cfg.bzl", "with_cfg")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//bazel:patchelf.bzl", "patchelf")
load("//runtimes/common:packages.bzl", "packages")

BASE_URL = "https://apt.repos.neuron.amazonaws.com"
STRIP_PREFIX = "opt/aws/neuron"

BUILD_FILE_PRELUDE = """\
"""

_PACKAGES = {
    "aws-neuronx-runtime-lib": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.cc_library_hdrs_glob(
            name = "libnrt_headers",
            hdrs_glob = [
                "include/ndl/**/*.h",
                "include/nrt/**/*.h",
            ],
            includes = ["include"],
            visibility = ["//visibility:public"],
        ),
        packages.patchelf(
            name = "libnrt.patchelf",
            shared_library = "lib/libnrt.so.1",
            set_rpath = '$ORIGIN',
            add_needed = [
                # readelf -d ./opt/aws/neuron/libl/libncfw.so
                "libncfw.so.2",
            ],
            rename_dynamic_symbols = {
                "dlopen": "zmlxneuron_dlopen",
            },
            visibility = ["@zml//runtimes/neuron:__subpackages__"],
        ),
        packages.patchelf(
            name = "libncfw.patchelf",
            shared_library = "lib/libncfw.so",
            soname = "libncfw.so.2",
            visibility = ["@zml//runtimes/neuron:__subpackages__"],
        ),
    ]),
    "aws-neuronx-collectives": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.filegroup(
            name = "libnccom",
            srcs = ["lib/libnccom.so.2"],
            visibility = ["@zml//runtimes/neuron:__subpackages__"],
        ),
        packages.patchelf(
            name = "libnccom-net.patchelf",
            shared_library = "lib/libnccom-net.so",
            soname = "libnccom-net.so.0",
            visibility = ["@zml//runtimes/neuron:__subpackages__"],
        ),
    ]),
}

def _neuron_impl(mctx):
    loaded_packages = packages.read(mctx, [
        "@zml//runtimes/neuron:packages.lock.json",
    ])
    for pkg_name, build_file_content in _PACKAGES.items():
        pkg = loaded_packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            strip_prefix = STRIP_PREFIX,
            build_file_content = BUILD_FILE_PRELUDE + build_file_content,
        )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

neuron_packages = module_extension(
    implementation = _neuron_impl,
)

py_binary_with_script, _py_binary_internal = with_cfg(
    kind = _py_binary,
    extra_providers = [PyInfo],
).set(Label("@rules_python//python/config_settings:bootstrap_impl"), "script").build()
