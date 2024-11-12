load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_ROCM_STRIP_PREFIX = "opt/rocm-6.3.4"

def _kwargs(**kwargs):
    return repr(struct(**kwargs))[len("struct("):-1]

def _cc_import(**kwargs):
    return """cc_import({})""".format(_kwargs(**kwargs))

def _filegroup(**kwargs):
    return """filegroup({})""".format(_kwargs(**kwargs))

def _load(bzl, name):
    return """load({}, {})""".format(repr(bzl), repr(name))

_UBUNTU_PACKAGES = {
    "libdrm2": _cc_import(name = "libdrm2", shared_library = "usr/lib/x86_64-linux-gnu/libdrm.so.2"),
    "libelf1": _cc_import(name = "libelf1", shared_library = "usr/lib/x86_64-linux-gnu/libelf.so.1"),
    "libnuma1": _cc_import(name = "libnuma1", shared_library = "usr/lib/x86_64-linux-gnu/libnuma.so.1"),
    "libzstd1": _cc_import(name = "libzstd1", shared_library = "usr/lib/x86_64-linux-gnu/libzstd.so.1"),
    "libdrm-amdgpu1": _cc_import(name = "libdrm-amdgpu1", shared_library = "usr/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1"),
    "libtinfo6": _cc_import(name = "libtinfo6", shared_library = "lib/x86_64-linux-gnu/libtinfo.so.6"),
    "zlib1g": _cc_import(name = "zlib1g", shared_library = "lib/x86_64-linux-gnu/libz.so.1"),
}

_ROCM_PACKAGES = {
    "rocm-core": _cc_import(name = "rocm-core", shared_library = "lib/librocm-core.so.1"),
    "rocm-smi-lib": _cc_import(name = "rocm_smi", shared_library = "lib/librocm_smi64.so.7"),
    "hsa-rocr": _cc_import(name = "hsa-runtime", shared_library = "lib/libhsa-runtime64.so.1"),
    "hsa-amd-aqlprofile": _cc_import(name = "hsa-amd-aqlprofile", shared_library = "lib/libhsa-amd-aqlprofile64.so.1"),
    "comgr": _cc_import(name = "amd_comgr", shared_library = "lib/libamd_comgr.so.2"),
    "rocprofiler-register": _cc_import(name = "rocprofiler-register", shared_library = "lib/librocprofiler-register.so.0"),
    "miopen-hip": "\n".join([
        _cc_import(name = "MIOpen", shared_library = "lib/libMIOpen.so.1"),
        """filegroup(name = "runfiles", srcs = glob(["share/miopen/**"]))""",
    ]),
    "rccl": _cc_import(name = "rccl", shared_library = "lib/librccl.so.1"),
    "rocm-device-libs": """filegroup(name = "runfiles", srcs = glob(["amdgcn/**"]))""",
    "hip-dev": """filegroup(name = "runfiles", srcs = glob(["share/**"]))""",
    "rocblas": "\n".join([
        _load("@zml//bazel:cc_import.bzl", "cc_import"),
        _load("@zml//runtimes/rocm:gfx.bzl", "bytecode_select"),
        _cc_import(
            name = "rocblas",
            shared_library = "lib/librocblas.so.4",
            add_needed = ["libzmlxrocm.so.0"],
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        """bytecode_select(
            name = "runfiles",
            srcs = glob(["lib/rocblas/library/*"]),
            enabled_gfx = "@libpjrt_rocm//:gfx",
        )""",
    ]),
    "roctracer": "\n".join([
        _cc_import(name = "roctracer", shared_library = "lib/libroctracer64.so.4", deps = [":roctx"]),
        _cc_import(name = "roctx", shared_library = "lib/libroctx64.so.4"),
    ]),
    "hipblaslt": "\n".join([
        _load("@zml//bazel:cc_import.bzl", "cc_import"),
        _load("@zml//runtimes/rocm:gfx.bzl", "bytecode_select"),
        _cc_import(
            name = "hipblaslt",
            shared_library = "lib/libhipblaslt.so.0",
            add_needed = ["libzmlxrocm.so.0"],
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        """bytecode_select(
            name = "bytecodes",
            srcs = glob(
                include = ["lib/hipblaslt/library/*"],
                exclude = ["lib/hipblaslt/library/hipblasltExtOpLibrary.dat"],
            ),
            enabled_gfx = "@libpjrt_rocm//:gfx",
        )""",
        _filegroup(
            name = "runfiles",
            srcs = [
                "lib/hipblaslt/library/hipblasltExtOpLibrary.dat",
                ":bytecodes",
            ],
        ),
    ]),
    "hip-runtime-amd": "\n".join([
        _cc_import(name = "amdhip", shared_library = "lib/libamdhip64.so.6", deps = [":hiprtc"]),
        _cc_import(name = "hiprtc", shared_library = "lib/libhiprtc.so.6"),
    ]),
    "rocm-llvm": _filegroup(name = "lld", srcs = ["llvm/bin/ld.lld"], visibility = ["//visibility:public"]),
}

def _read_packages_json(mctx, packages_json):
    data = json.decode(mctx.read(Label(packages_json)))
    return {
        pkg["name"]: pkg
        for pkg in data["packages"]
    }

def _rocm_impl(mctx):
    packages = _read_packages_json(mctx, "@zml//runtimes/rocm:packages.lock.json")
    packages.update(_read_packages_json(mctx, "@zml//runtimes/common:packages.lock.json"))

    for pkg_name, build_file_content in _ROCM_PACKAGES.items():
        pkg = packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            strip_prefix = _ROCM_STRIP_PREFIX,
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    for pkg_name, build_file_content in _UBUNTU_PACKAGES.items():
        pkg = packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    http_archive(
        name = "libpjrt_rocm",
        build_file = "libpjrt_rocm.BUILD.bazel",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v6.0.0/pjrt-rocm_linux-amd64.tar.gz",
        sha256 = "0a41e5b255d0b3284e029c11c2362d777557f87778e1531a2117bc46b835395a",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_rocm", "hipblaslt", "rocblas"],
        root_module_direct_dev_deps = [],
    )

rocm_packages = module_extension(
    implementation = _rocm_impl,
)
