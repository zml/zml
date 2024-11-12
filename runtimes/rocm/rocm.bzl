load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

_ROCM_STRIP_PREFIX = "opt/rocm-6.2.4"

def _kwargs(**kwargs):
    return repr(struct(**kwargs))[len("struct("):-1]

def _cc_import(**kwargs):
    return """cc_import({})""".format(_kwargs(**kwargs))

def _filegroup(**kwargs):
    return """filegroup({})""".format(_kwargs(**kwargs))

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
    "rccl": "\n".join([
        _cc_import(name = "rccl", shared_library = "lib/librccl.so.1"),
        """filegroup(name = "runfiles", srcs = glob(["share/rccl/msccl-algorithms/**"]))""",
    ]),
    "rocm-device-libs": _filegroup(name = "runfiles", glob = ["amdgcn/**"]),
    "hip-dev": _filegroup(name = "runfiles", glob = ["share/**"]),
    "rocblas": "\n".join([
        """load("@zml//bazel:cc_import.bzl", "cc_import")""",
        """load("@zml//runtimes/rocm:gfx.bzl", "bytecode_select")""",
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
        """load("@zml//bazel:cc_import.bzl", "cc_import")""",
        _cc_import(
            name = "hipblaslt",
            shared_library = "lib/libhipblaslt.so.0",
            add_needed = ["libzmlxrocm.so.0"],
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
    ]),
    "hipblaslt-dev": "\n".join([
        """load("@zml//runtimes/rocm:gfx.bzl", "bytecode_select")""",
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
    "rocm-llvm": _filegroup(name = "lld", srcs = ["llvm/bin/ld.lld"]),
}

def _rocm_impl(mctx):
    packages_json = json.decode(mctx.read(Label("@zml//runtimes/rocm:packages.lock.json")))
    packages = {
        pkg["name"]: pkg
        for pkg in packages_json["packages"]
    }
    for pkg_name, build_file_content in _ROCM_PACKAGES.items():
        pkg = packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = [pkg["url"]],
            sha256 = pkg["sha256"],
            strip_prefix = _ROCM_STRIP_PREFIX,
            build_file_content = build_file_content,
        )

    for pkg_name, build_file_content in _UBUNTU_PACKAGES.items():
        pkg = packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = [pkg["url"]],
            sha256 = pkg["sha256"],
            build_file_content = build_file_content,
        )

    http_archive(
        name = "libpjrt_rocm",
        build_file = "libpjrt_rocm.BUILD.bazel",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v3.0.0/pjrt-rocm_linux-amd64.tar.gz",
        sha256 = "a7da45dfca820d3defa6de8e782cc334a3f6bdffe65fa972c048994923c2e110",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_rocm"],
        root_module_direct_dev_deps = [],
    )

rocm_packages = module_extension(
    implementation = _rocm_impl,
)
