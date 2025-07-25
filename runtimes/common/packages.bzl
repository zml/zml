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

def _cc_library_hdrs_glob(name, hdrs_glob, deps = [], **kwargs):
    return """\
cc_library(
    name = "{name}",
    hdrs = glob({hdrs_glob}),
    deps = {deps},
    {kwargs}
)
""".format(name = name, hdrs_glob = repr(hdrs_glob), deps = repr(deps), kwargs = _kwargs(**kwargs))

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

packages = struct(
    read = _read,
    cc_import = _cc_import,
    cc_import_glob_hdrs = _cc_import_glob_hdrs,
    cc_library = _cc_library,
    cc_library_hdrs_glob = _cc_library_hdrs_glob,
    filegroup = _filegroup,
    load_ = _load,
    patchelf = _patchelf,
)
