load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//runtimes/common:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_ROCM_STRIP_PREFIX = "opt/rocm-6.3.4"

_UBUNTU_PACKAGES = {
    "libdrm2": packages.filegroup(name = "libdrm2", srcs = ["usr/lib/x86_64-linux-gnu/libdrm.so.2"]),
    "libelf1": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "libelf1",
            shared_library = "usr/lib/x86_64-linux-gnu/libelf.so.1",
            set_rpath = '$ORIGIN',
        ),
    ]),
    "libnuma1": packages.filegroup(name = "libnuma1", srcs = ["usr/lib/x86_64-linux-gnu/libnuma.so.1"]),
    "libzstd1": packages.filegroup(name = "libzstd1", srcs = ["usr/lib/x86_64-linux-gnu/libzstd.so.1"]),
    "libdrm-amdgpu1": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "libdrm-amdgpu1",
            shared_library = "usr/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1",
            set_rpath = '$ORIGIN',
        ),
    ]),
    "libtinfo6": packages.filegroup(name = "libtinfo6", srcs = ["lib/x86_64-linux-gnu/libtinfo.so.6"]),
    "zlib1g": packages.filegroup(name = "zlib1g", srcs = ["lib/x86_64-linux-gnu/libz.so.1"]),
}

_ROCM_PACKAGES = {
    "rocm-core": packages.filegroup(name = "rocm-core", srcs = ["lib/librocm-core.so.1"]),
    "rocm-smi-lib": packages.filegroup(name = "rocm_smi", srcs = ["lib/librocm_smi64.so.7"]),
    "hsa-rocr": packages.filegroup(name = "hsa-runtime", srcs = ["lib/libhsa-runtime64.so.1"]),
    "hsa-amd-aqlprofile": packages.filegroup(name = "hsa-amd-aqlprofile", srcs = ["lib/libhsa-amd-aqlprofile64.so.1"]),
    "comgr": packages.filegroup(name = "amd_comgr", srcs = ["lib/libamd_comgr.so.2"]),
    "rocprofiler-register": packages.filegroup(name = "rocprofiler-register", srcs = ["lib/librocprofiler-register.so.0"]),
    "miopen-hip": "\n".join([
        packages.filegroup(name = "MIOpen", srcs = ["lib/libMIOpen.so.1"]),
        """filegroup(name = "runfiles", srcs = glob(["share/miopen/**"]))""",
    ]),
    "rccl": packages.filegroup(name = "rccl", srcs = ["lib/librccl.so.1"]),
    "rocm-device-libs": """filegroup(name = "runfiles", srcs = glob(["amdgcn/**"]))""",
    "hip-dev": """filegroup(name = "runfiles", srcs = glob(["share/**"]))""",
    "rocblas": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.load_("@zml//runtimes/rocm:gfx.bzl", "bytecode_select"),
        packages.patchelf(
            name = "rocblas",
            shared_library = "lib/librocblas.so.4",
            add_needed = ["libzmlxrocm.so.0"],
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        """bytecode_select(
            name = "bytecodes",
            srcs = glob(["lib/rocblas/library/*"]),
            enabled_gfx = "@libpjrt_rocm//:gfx",
        )
        """,
        packages.filegroup(
            name = "runfiles",
            srcs = [
                "lib/rocblas/library/TensileManifest.txt",
                ":bytecodes",
            ],
        ),
    ]),
    "rocfft": packages.filegroup(name = "rocfft", srcs = ["lib/librocfft.so.0"]),
    "rocsolver": packages.filegroup(name = "rocsolver", srcs = ["lib/librocsolver.so.0"]),
    "roctracer": "\n".join([
        packages.filegroup(name = "roctracer", srcs = ["lib/libroctracer64.so.4"]),
        packages.filegroup(name = "roctx", srcs = ["lib/libroctx64.so.4"]),
    ]),
    "hipblaslt": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.load_("@zml//runtimes/rocm:gfx.bzl", "bytecode_select"),
        packages.patchelf(
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
        )
        """,
        packages.filegroup(
            name = "runfiles",
            srcs = [
                "lib/hipblaslt/library/hipblasltExtOpLibrary.dat",
                "lib/hipblaslt/library/TensileManifest.txt",
                ":bytecodes",
            ],
        ),
    ]),
    "hipfft": packages.filegroup(name = "hipfft", srcs = ["lib/libhipfft.so.0"]),
    "hip-runtime-amd": "\n".join([
        packages.filegroup(name = "amdhip", srcs = ["lib/libamdhip64.so.6"]),
        packages.filegroup(name = "hiprtc", srcs = ["lib/libhiprtc.so.6"]),
    ]),
    "hipsolver": packages.filegroup(name = "hipsolver", srcs = ["lib/libhipsolver.so.0"]),
}

def _rocm_impl(mctx):
    loaded_packages = packages.read(mctx, [
        "@zml//runtimes/rocm:packages.lock.json",
    ])

    for pkg_name, build_file_content in _UBUNTU_PACKAGES.items():
        pkg = loaded_packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    for pkg_name, build_file_content in _ROCM_PACKAGES.items():
        pkg = loaded_packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            strip_prefix = _ROCM_STRIP_PREFIX,
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    http_archive(
        name = "libpjrt_rocm",
        build_file = "libpjrt_rocm.BUILD.bazel",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v11.0.0/pjrt-rocm_linux-amd64.tar.gz",
        sha256 = "a6d8ef38ae4deda244856549271a1b1a6f46499e9efb64fb71a12fd6ae792d3b",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_rocm", "hipblaslt", "rocblas"],
        root_module_direct_dev_deps = [],
    )

rocm_packages = module_extension(
    implementation = _rocm_impl,
)
