load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_ROCM_STRIP_PREFIX = "./opt/rocm-7.2.1"

def _rocm_dlopen_patchelf(name, src):
    return "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = name,
            src = src,
            add_needed = ["libzmlxrocm.so.0"],
            set_rpath = "$ORIGIN",
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
    ])

_UBUNTU_PACKAGES = {
    "libbz2-1.0": packages.filegroup(name = "libbz2-1.0", srcs = ["lib/x86_64-linux-gnu/libbz2.so.1.0"]),
    "libdrm2-amdgpu": packages.filegroup(name = "libdrm2-amdgpu", srcs = ["opt/amdgpu/lib/x86_64-linux-gnu/libdrm.so.2"]),
    "libelf1": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "libelf1",
            src = "usr/lib/x86_64-linux-gnu/libelf.so.1",
            set_rpath = "$ORIGIN",
        ),
    ]),
    "libdrm-amdgpu-common": packages.filegroup(name = "amdgpu_ids", srcs = ["opt/amdgpu/share/libdrm/amdgpu.ids"]),
    "libnuma1": packages.filegroup(name = "libnuma1", srcs = ["usr/lib/x86_64-linux-gnu/libnuma.so.1"]),
    "liblzma5": packages.filegroup(name = "liblzma5", srcs = ["lib/x86_64-linux-gnu/liblzma.so.5"]),
    "libzstd1": packages.filegroup(name = "libzstd1", srcs = ["usr/lib/x86_64-linux-gnu/libzstd.so.1"]),
    "libdrm-amdgpu-amdgpu1": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "libdrm-amdgpu-amdgpu1",
            src = "opt/amdgpu/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1",
            add_needed = ["libzmlxrocm.so.0"],
            set_rpath = "$ORIGIN",
            rename_dynamic_symbols = {
                "fopen64": "zmlxrocm_fopen64",
            },
        ),
    ]),
    "libtinfo6": packages.filegroup(name = "libtinfo6", srcs = ["lib/x86_64-linux-gnu/libtinfo.so.6"]),
    "zlib1g": packages.filegroup(name = "zlib1g", srcs = ["lib/x86_64-linux-gnu/libz.so.1"]),
    "libdw1": packages.filegroup(name = "libdw1", srcs = ["usr/lib/x86_64-linux-gnu/libdw.so.1"]),
}

_ROCM_PACKAGES = {
    "rocm-core": _rocm_dlopen_patchelf(
        name = "rocm-core",
        src = "lib/librocm-core.so.1",
    ),
    "rocm-smi-lib": _rocm_dlopen_patchelf(
        name = "rocm_smi",
        src = "lib/librocm_smi64.so.1",
    ),
    "amd-smi-lib": packages.filegroup(name = "amdsmi", srcs = ["lib/libamd_smi.so"]),
    "rocprofiler-sdk": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "rocprofiler-sdk_so",
            src = "lib/librocprofiler-sdk.so.1",
            add_needed = ["libzmlxrocm.so.0"],
            set_rpath = "$ORIGIN",
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        packages.filegroup(name = "rocprofiler-sdk-attach", srcs = ["lib/librocprofiler-sdk-attach.so.1"]),
        packages.filegroup(name = "rocprofiler-sdk", srcs = [
            ":rocprofiler-sdk_so",
            ":rocprofiler-sdk-attach",
        ]),
    ]),
    "rocprofiler-sdk-rocpd": packages.filegroup(name = "rocprofiler-sdk-rocpd", srcs = ["lib/librocprofiler-sdk-rocpd.so.1"]),
    "rocprofiler-sdk-roctx": packages.filegroup(name = "rocprofiler-sdk-roctx", srcs = ["lib/librocprofiler-sdk-roctx.so.1"]),
    "hsa-rocr": _rocm_dlopen_patchelf(
        name = "hsa-runtime",
        src = "lib/libhsa-runtime64.so.1",
    ),
    "hsa-amd-aqlprofile": _rocm_dlopen_patchelf(
        name = "hsa-amd-aqlprofile",
        src = "lib/libhsa-amd-aqlprofile64.so.1",
    ),
    "comgr": _rocm_dlopen_patchelf(
        name = "amd_comgr",
        src = "lib/libamd_comgr.so.3",
    ),
    "rocprofiler-register": _rocm_dlopen_patchelf(
        name = "rocprofiler-register",
        src = "lib/librocprofiler-register.so.0",
    ),
    "miopen-hip": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "MIOpen",
            src = "lib/libMIOpen.so.1",
            add_needed = ["libzmlxrocm.so.0"],
            set_rpath = "$ORIGIN",
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        """filegroup(name = "runfiles", srcs = glob(["share/miopen/**"]))""",
    ]),
    "rccl": _rocm_dlopen_patchelf(
        name = "rccl",
        src = "lib/librccl.so.1",
    ),
    "rocm-device-libs": """filegroup(name = "runfiles", srcs = glob(["amdgcn/**"]))""",
    "hip-dev": """filegroup(name = "runfiles", srcs = glob(["share/**"]))""",
    "rocblas": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "rocblas",
            src = "lib/librocblas.so.5",
            add_needed = ["libzmlxrocm.so.0"],
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        """filegroup(
            name = "runfiles",
            srcs = glob(["lib/rocblas/library/**"]),
        )
        """,
    ]),
    "rocfft": packages.filegroup(name = "rocfft", srcs = ["lib/librocfft.so.0"]),
    "rocsolver": _rocm_dlopen_patchelf(
        name = "rocsolver",
        src = "lib/librocsolver.so.0",
    ),
    "roctracer": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "roctracer",
            src = "lib/libroctracer64.so.4",
            add_needed = ["libzmlxrocm.so.0"],
            set_rpath = "$ORIGIN",
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        packages.patchelf(
            name = "roctx",
            src = "lib/libroctx64.so.4",
            add_needed = ["libzmlxrocm.so.0"],
            set_rpath = "$ORIGIN",
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
    ]),
    "hipblaslt": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "hipblaslt",
            src = "lib/libhipblaslt.so.1",
            add_needed = ["libzmlxrocm.so.0"],
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        packages.patchelf(
            name = "rocroller",
            src = "lib/librocroller.so.1",
            set_rpath = "$ORIGIN",
            add_needed = ["libzmlxrocm.so.0"],
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        """filegroup(
            name = "runfiles",
            srcs = glob(["lib/hipblaslt/library/**"]),
        )
        """,
    ]),
    "hipfft": packages.filegroup(name = "hipfft", srcs = ["lib/libhipfft.so.0"]),
    "hip-runtime-amd": "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = "amdhip",
            src = "lib/libamdhip64.so.7",
            add_needed = ["libzmlxrocm.so.0"],
            set_rpath = "$ORIGIN",
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        packages.patchelf(
            name = "hiprtc_so",
            src = "lib/libhiprtc.so.7",
            add_needed = ["libzmlxrocm.so.0"],
            set_rpath = "$ORIGIN",
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
            },
        ),
        packages.filegroup(name = "hiprtc", srcs = [":hiprtc_so", "lib/libhiprtc-builtins.so.7"]),
    ]),
    "hipsolver": _rocm_dlopen_patchelf(
        name = "hipsolver",
        src = "lib/libhipsolver.so.1",
    ),
    "rocsparse": packages.filegroup(name = "rocsparse", srcs = ["lib/librocsparse.so.1"]),
}

def _rocm_impl(mctx):
    loaded_packages = packages.read(mctx, [
        "@zml//platforms/rocm:packages.lock.json",
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
        url = "https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-03-26T11-30-00Z/pjrt-rocm_linux-amd64.tar.gz",
        sha256 = "6649e89831570926bf127f7e57f25dca4f526e22764f2df0d689818badc1d4fe",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_rocm", "hipblaslt", "rocblas", "amd-smi-lib"],
        root_module_direct_dev_deps = [],
    )

rocm_packages = module_extension(
    implementation = _rocm_impl,
)
