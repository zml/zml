load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

ROCM_VERSION = "6.2"
BASE_URL = "https://repo.radeon.com/rocm/apt/{}".format(ROCM_VERSION)
STRIP_PREFIX = "opt/rocm-6.2.0"

def pkg_kwargs(pkg, packages):
    return {
        "name": pkg,
        "urls": [BASE_URL + "/" + packages[pkg]["Filename"]],
        "sha256": packages[pkg]["SHA256"],
        "strip_prefix": STRIP_PREFIX,
    }

def _ubuntu_package(path, deb_path, sha256, name, shared_library):
    return {
        "urls": ["http://archive.ubuntu.com/ubuntu/pool/main/{}".format(path)],
        "sha256": sha256,
        "build_file_content": """\
cc_import(
    name = {name},
    shared_library = "{deb_path}{shared_library}",
    visibility = ["//visibility:public"],
)
""".format(name = repr(name), shared_library = shared_library, deb_path = deb_path),
    }

_UBUNTU_PACKAGES = {
    "libdrm": _ubuntu_package(
        path = "libd/libdrm/libdrm2_2.4.107-8ubuntu1~20.04.2_amd64.deb",
        deb_path = "usr/lib/x86_64-linux-gnu/",
        sha256 = "9b01d73313841abe8e3f24c2715edced675fbe329bbd10be912a5b135cd51fb6",
        name = "libdrm",
        shared_library = "libdrm.so.2",
    ),
    "libelf": _ubuntu_package(
        path = "e/elfutils/libelf1_0.176-1.1build1_amd64.deb",
        deb_path = "usr/lib/x86_64-linux-gnu/",
        sha256 = "78a8761227efc04a1e37527f2f33ba608c6fb5d6c911616346ada5d7b9b72ee3",
        name = "libelf",
        shared_library = "libelf.so.1",
    ),
    "libnuma": _ubuntu_package(
        path = "n/numactl/libnuma1_2.0.12-1_amd64.deb",
        deb_path = "usr/lib/x86_64-linux-gnu/",
        sha256 = "0b1edf08cf9befecd21fe94e298ac25e476f87fd876ddd4adf42ef713449e637",
        name = "libnuma",
        shared_library = "libnuma.so.1",
    ),
    "libzstd": _ubuntu_package(
        path = "libz/libzstd/libzstd1_1.4.4+dfsg-3ubuntu0.1_amd64.deb",
        deb_path = "usr/lib/x86_64-linux-gnu/",
        sha256 = "7a4422dadb90510dc90765c308d65e61a3e244ceb3886394335e48cff7559e69",
        name = "libzstd",
        shared_library = "libzstd.so.1",
    ),
    "libdrm-amdgpu": _ubuntu_package(
        path = "libd/libdrm/libdrm-amdgpu1_2.4.107-8ubuntu1~20.04.2_amd64.deb",
        deb_path = "usr/lib/x86_64-linux-gnu/",
        sha256 = "0d95779b581f344e3d658e0f21f6e4b57da6eb3606c0bcb8cb874c12f5754bf2",
        name = "libdrm-amdgpu",
        shared_library = "libdrm_amdgpu.so.1",
    ),
    "libtinfo": _ubuntu_package(
        path = "n/ncurses/libtinfo6_6.2-0ubuntu2.1_amd64.deb",
        deb_path = "lib/x86_64-linux-gnu/",
        sha256 = "711a3a901c3a71561565558865699efa9c07a99fdc810ffe086a5636f89c6431",
        name = "libtinfo",
        shared_library = "libtinfo.so.6",
    ),
    "zlib1g": _ubuntu_package(
        path = "z/zlib/zlib1g_1.2.11.dfsg-2ubuntu1.5_amd64.deb",
        deb_path = "lib/x86_64-linux-gnu/",
        sha256 = "bf67018f5303466eb468680b637a5d3f3bb17b9d44decf3d82d40b35babcd3e0",
        name = "zlib1g",
        shared_library = "libz.so.1",
    ),
}

_CC_IMPORT_TPL = """\
cc_import(
    name = "{name}",
    shared_library = "lib/{shared_library}",
    visibility = ["@libpjrt_rocm//:__subpackages__"],
)
"""

_RUNFILES_TPL = """\
filegroup(
    name = "{name}",
    srcs = glob({glob}),
    visibility = ["@libpjrt_rocm//:__subpackages__"],
)
"""

_PACKAGES = {
    "rocm-core": _CC_IMPORT_TPL.format(name = "rocm-core", shared_library = "librocm-core.so.1"),
    "rocm-smi-lib": _CC_IMPORT_TPL.format(name = "rocm_smi", shared_library = "librocm_smi64.so.7"),
    "hsa-rocr": _CC_IMPORT_TPL.format(name = "hsa-runtime", shared_library = "libhsa-runtime64.so.1"),
    "hsa-amd-aqlprofile": _CC_IMPORT_TPL.format(name = "hsa-amd-aqlprofile", shared_library = "libhsa-amd-aqlprofile64.so.1"),
    "comgr": _CC_IMPORT_TPL.format(name = "amd_comgr", shared_library = "libamd_comgr.so.2"),
    "rocprofiler-register": _CC_IMPORT_TPL.format(name = "rocprofiler-register", shared_library = "librocprofiler-register.so.0"),
    "miopen-hip": "".join([
        _CC_IMPORT_TPL.format(name = "MIOpen", shared_library = "libMIOpen.so.1"),
        _RUNFILES_TPL.format(name = "runfiles", glob = repr(["share/miopen/**"])),
    ]),
    "rccl": "".join([
        _CC_IMPORT_TPL.format(name = "rccl", shared_library = "librccl.so.1"),
        _RUNFILES_TPL.format(name = "runfiles", glob = repr(["share/rccl/msccl-algorithms/**"])),
    ]),
    "rocm-device-libs": _RUNFILES_TPL.format(name = "runfiles", glob = repr(["amdgcn/**"])),
    "hip-dev": _RUNFILES_TPL.format(name = "runfiles", glob = repr(["share/**"])),
    "rocblas": """\
load("@zml//bazel:cc_import.bzl", "cc_import")
load("@zml//runtimes/rocm:gfx.bzl", "bytecode_select")

cc_import(
    name = "rocblas",
    shared_library = "lib/librocblas.so.4",
    add_needed = ["libzmlrocmhooks.so.0"],
    visibility = ["@libpjrt_rocm//:__subpackages__"],
)

bytecode_select(
    name = "runfiles",
    bytecodes = glob(["lib/rocblas/library/*"]),
    enabled_gfx = "@libpjrt_rocm//:gfx",
    visibility = ["@libpjrt_rocm//:__subpackages__"],
)
""",
    "roctracer": """\
cc_import(
    name = "roctracer",
    shared_library = "lib/libroctracer64.so.4",
    visibility = ["@libpjrt_rocm//:__subpackages__"],
    deps = [":roctx"],
)

cc_import(
    name = "roctx",
    shared_library = "lib/libroctx64.so.4",
)
""",
    "hipblaslt": """\
load("@zml//bazel:cc_import.bzl", "cc_import")
cc_import(
    name = "hipblaslt",
    shared_library = "lib/libhipblaslt.so.0",
    add_needed = ["libzmlrocmhooks.so.0"],
    visibility = ["@libpjrt_rocm//:__subpackages__"],
)
""",
    "hipblaslt-dev": """\
load("@zml//runtimes/rocm:gfx.bzl", "bytecode_select")

bytecode_select(
    name = "bytecodes",
    bytecodes = glob(
        include = ["lib/hipblaslt/library/*"],
        exclude = ["lib/hipblaslt/library/hipblasltExtOpLibrary.dat"],
    ),
    enabled_gfx = "@libpjrt_rocm//:gfx",
)

filegroup(
    name = "runfiles",
    srcs = [
        "lib/hipblaslt/library/hipblasltExtOpLibrary.dat",
        ":bytecodes",
    ],
    visibility = ["@libpjrt_rocm//:__subpackages__"],
)
""",
    "hip-runtime-amd": """\
cc_import(
    name = "amdhip",
    shared_library = "lib/libamdhip64.so.6",
    visibility = ["@libpjrt_rocm//:__subpackages__"],
    deps = [":hiprtc"],
)
cc_import(
    name = "hiprtc",
    shared_library = "lib/libhiprtc.so.6",
)
""",
    "rocm-llvm": """\
filegroup(
    name = "lld",
    srcs = ["llvm/bin/ld.lld"],
    visibility = ["@libpjrt_rocm//:__subpackages__"],
)
""",
}

def _packages_to_dict(txt):
    packages = {}
    current_pkg = {}
    for line in txt.splitlines():
        if line == "":
            if current_pkg:
                packages[current_pkg["Package"]] = current_pkg
            current_pkg = {}
            continue
        if line.startswith(" "):
            current_pkg[key] += line
            continue
        split = line.split(": ", 1)
        key = split[0]
        value = len(split) > 1 and split[1] or ""
        current_pkg[key] = value
    return packages

def _rocm_impl(mctx):
    data = mctx.read(Label("@zml//runtimes/rocm:packages.amd64.txt"))
    PACKAGES = _packages_to_dict(data)

    for pkg, build_file_content in _PACKAGES.items():
        http_deb_archive(
            build_file_content = build_file_content,
            **pkg_kwargs(pkg, PACKAGES)
        )

    for repository, kwargs in _UBUNTU_PACKAGES.items():
        http_deb_archive(name = repository, **kwargs)

    http_archive(
        name = "libpjrt_rocm",
        build_file = "libpjrt_rocm.BUILD.bazel",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v0.1.13/pjrt-rocm_linux-amd64.tar.gz",
        sha256 = "5900cec41274e80ab799bc13f31cdc87202f8e168d7e753b1c10796912f5ebef",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_rocm"],
        root_module_direct_dev_deps = [],
    )

rocm_packages = module_extension(
    implementation = _rocm_impl,
)
