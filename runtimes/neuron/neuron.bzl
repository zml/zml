load("//bazel:dpkg.bzl", "dpkg")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

BASE_URL = "https://apt.repos.neuron.amazonaws.com"
STRIP_PREFIX = "opt/aws/neuron"

_PACKAGES = {
    "aws-neuronx-runtime-lib": struct(
        version = "2.22.14.0-6e27b8d5b",
        build_file_content = """\
load("@zml//bazel:cc_import.bzl", "cc_import")
cc_import(
    name = "aws-neuronx-runtime-lib",
    shared_library = "lib/libnrt.so.1",
    rename_dynamic_symbols = {
        "dlopen": "zmlxneuron_dlopen",
    },
    visibility = ["@zml//runtimes/neuron:__subpackages__"],
    deps = ["@aws-neuronx-collectives//:libnccom"],
)
""",
    ),
    "aws-neuronx-collectives": struct(
        version = "2.22.26.0-17a033bc8",
        build_file_content = """\
cc_import(
    name = "libnccom",
    shared_library = "lib/libnccom.so.2",
    visibility = ["@aws-neuronx-runtime-lib//:__subpackages__"],
)

cc_import(
    name = "libnccom-net",
    shared_library = "lib/libnccom-net.so.0",
)
""",
    ),
}

def _neuron_impl(mctx):
    packages = dpkg.read_packages(mctx, "@zml//runtimes/neuron:packages.amd64.txt")
    for pkg_name, pkg_data in _PACKAGES.items():
        pkg = packages[pkg_name][pkg_data.version]
        http_deb_archive(
            name = pkg.Package,
            urls = [BASE_URL + "/" + pkg.Filename],
            sha256 = pkg.SHA256,
            strip_prefix = STRIP_PREFIX,
            build_file_content = pkg_data.build_file_content,
        )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

neuron_packages = module_extension(
    implementation = _neuron_impl,
)
