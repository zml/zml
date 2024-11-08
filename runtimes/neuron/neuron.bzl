load("@python_versions//3.11:defs.bzl", _py_binary = "py_binary")
load("@rules_python//python:defs.bzl", "PyInfo")
load("@with_cfg.bzl", "with_cfg")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

BASE_URL = "https://apt.repos.neuron.amazonaws.com"
STRIP_PREFIX = "opt/aws/neuron"

_PACKAGES = {
    "aws-neuronx-runtime-lib": """\
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
    "aws-neuronx-collectives": """\
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
}

def _neuron_impl(mctx):
    packages_json = json.decode(mctx.read(Label("@zml//runtimes/neuron:packages.lock.json")))
    packages = {
        pkg["name"]: pkg
        for pkg in packages_json["packages"]
    }
    for pkg_name, build_file_content in _PACKAGES.items():
        pkg = packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = [pkg["url"]],
            sha256 = pkg["sha256"],
            strip_prefix = STRIP_PREFIX,
            build_file_content = build_file_content,
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
