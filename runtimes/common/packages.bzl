load("//bazel:http_deb_archive.bzl", "http_deb_archive")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

def _kwargs(**kwargs):
    return repr(struct(**kwargs))[len("struct("):-1]

def _cc_import(**kwargs):
    return """cc_import({})""".format(_kwargs(**kwargs))

def _cc_library(**kwargs):
    return """cc_library({})""".format(_kwargs(**kwargs))

def _cc_import_glob_hdrs(name, hdrs_glob, shared_library, deps = [], **kwargs):
    return """\
cc_import(
    name = "{name}",
    shared_library = {shared_library},
    hdrs = glob({hdrs_glob}),
    deps = {deps},
    {kwargs}
)
""".format(name = name, hdrs_glob = repr(hdrs_glob), shared_library = repr(shared_library), deps = repr(deps), kwargs = _kwargs(**kwargs))

def _filegroup(**kwargs):
    return """filegroup({})""".format(_kwargs(**kwargs))

def _patchelf(**kwargs):
    return """patchelf({})""".format(_kwargs(**kwargs))

def _load(bzl, name):
    return """load({}, {})""".format(repr(bzl), repr(name))

def _read(mctx, labels):
    ret = {}
    for label in labels:
        data = json.decode(mctx.read(Label(label)))
        ret.update({
            pkg["name"]: pkg
            for pkg in data["packages"]
        })
    return ret

_DEBIAN_PACKAGES = {
    "zlib1g": _filegroup(name = "zlib1g", srcs = ["lib/x86_64-linux-gnu/libz.so.1"]),
}

def _common_apt_packages_impl(mctx):
    loaded_packages = packages.read(mctx, ["packages.lock.json"])
    for pkg_name, build_file_content in _DEBIAN_PACKAGES.items():
        pkg = loaded_packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

common_apt_packages = module_extension(
    implementation = _common_apt_packages_impl,
)

packages = struct(
    read = _read,
    cc_import = _cc_import,
    cc_import_glob_hdrs = _cc_import_glob_hdrs,
    cc_library = _cc_library,
    filegroup = _filegroup,
    load_ = _load,
    patchelf = _patchelf,
)
