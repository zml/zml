load("@python_versions//3.12:defs.bzl", _py_binary = "py_binary")
load("@rules_python//python:defs.bzl", "PyInfo")
load("@with_cfg.bzl", "with_cfg")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//bazel:simple_repository.bzl", "simple_repository")
load("//platforms:packages.bzl", "packages")

BASE_URL = "https://apt.repos.neuron.amazonaws.com"
STRIP_PREFIX = "./opt/aws/neuron"

_BUILD_FILE_PRELUDE = """\
package(default_visibility = ["//visibility:public"])
"""

_UBUNTU_PACKAGES = {
    "libgcc_s1": {
        "package": "libgcc-s1",
        "build": packages.filegroup(name = "libgcc_s1", srcs = ["lib/x86_64-linux-gnu/libgcc_s.so.1"]),
    },
    "libgomp1": {
        "package": "libgomp1",
        "build": packages.filegroup(name = "libgomp1", srcs = ["usr/lib/x86_64-linux-gnu/libgomp.so.1"]),
    },
    "libstdcpp6": {
        "package": "libstdc++6",
        "build": packages.filegroup(name = "libstdcpp6", srcs = ["usr/lib/x86_64-linux-gnu/libstdc++.so.6"]),
    },
    "zlib1g": {
        "package": "zlib1g",
        "build": packages.filegroup(name = "zlib1g", srcs = ["lib/x86_64-linux-gnu/libz.so.1"]),
    },
}

_NEURON_PACKAGES = {
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
            src = "lib/libnrt.so.1",
            set_rpath = "$ORIGIN",
            add_needed = [
                # readelf -d ./opt/aws/neuron/libl/libncfw.so
                "libncfw.so.2",
            ],
            rename_dynamic_symbols = {
                "dlopen": "zmlxneuron_dlopen",
            },
        ),
        packages.patchelf(
            name = "libncfw.patchelf",
            src = "lib/libncfw.so",
            soname = "libncfw.so.2",
        ),
        packages.filegroup(
            name = "libnrtucode_extisa",
            srcs = ["lib/libnrtucode_extisa.so"],
        ),
    ]),
    "aws-neuronx-collectives": "\n".join([
        packages.filegroup(
            name = "libnccom",
            srcs = ["lib/libnccom.so.2"],
        ),
    ]),
    "aws-neuronx-tools": "\n".join([
        packages.filegroup(
            name = "neuron-explorer",
            srcs = ["bin/neuron-explorer"],
        ),
    ]),
}

def _neuron_impl(mctx):
    loaded_packages = packages.read(mctx, [
        "@zml//platforms/neuron:packages.lock.json",
    ])

    simple_repository(
        name = "libpjrt_neuron",
        build_file = ":libpjrt_neuron.BUILD.bazel",
    )

    for repo_name, package in _UBUNTU_PACKAGES.items():
        pkg = loaded_packages[package["package"]]["amd64"]
        http_deb_archive(
            name = repo_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_PRELUDE + package["build"],
        )

    for pkg_name, build_file_content in _NEURON_PACKAGES.items():
        pkg = loaded_packages[pkg_name]["amd64"]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            strip_prefix = STRIP_PREFIX,
            build_file_content = _BUILD_FILE_PRELUDE + build_file_content,
        )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "aws-neuronx-collectives",
            "aws-neuronx-runtime-lib",
            "aws-neuronx-tools",
            "libgcc_s1",
            "libgomp1",
            "libstdcpp6",
            "libpjrt_neuron",
        ],
        root_module_direct_dev_deps = [],
    )

neuron_packages = module_extension(
    implementation = _neuron_impl,
)

py_binary_with_script, _py_binary_internal = with_cfg(
    kind = _py_binary,
    extra_providers = [PyInfo],
).build()
