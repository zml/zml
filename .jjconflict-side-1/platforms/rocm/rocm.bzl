load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:cc_library.bzl", "cc_library")
"""

_ROCM_VERSION = "7.14"
_ROCM_STRIP_PREFIX = "./opt/rocm/core-" + _ROCM_VERSION
_PJRT_ROCM_URL = "https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-07-20T15-30-00Z/pjrt-rocm_linux-amd64.tar.gz"
_PJRT_ROCM_SHA256 = "6fd0515beb299550e298996f6919db09aec79859feee7362443bf2ebff900d0f"

def _rocm_package_name(name):
    return name + _ROCM_VERSION

def _rocm_repo_name(package_name):
    return package_name.replace(_ROCM_VERSION, "")

_UBUNTU_PACKAGES = {
    "libatomic1": packages.filegroup(name = "libatomic1", srcs = ["usr/lib/x86_64-linux-gnu/libatomic.so.1"]),
    "libdrm-common": packages.filegroup(name = "amdgpu_ids", srcs = ["usr/share/libdrm/amdgpu.ids"]),
}

def _rocm_dlopen_patchelf(name, src):
    return "\n".join([
        packages.load_("@zml//bazel:patchelf.bzl", "patchelf"),
        packages.patchelf(
            name = name,
            src = src,
            add_needed = ["libzmlxrocm.so.0"],
            set_rpath = "$ORIGIN:$ORIGIN/rocm_sysdeps/lib",
            rename_dynamic_symbols = {
                "dlopen": "zmlxrocm_dlopen",
                "pthread_attr_setstacksize": "zmlxrocm_pthread_attr_setstacksize",
            },
        ),
    ])

def _glob_filegroup(name, srcs):
    return """filegroup(
    name = "{name}",
    srcs = glob({srcs}),
)""".format(name = name, srcs = repr(srcs))

def _family_runfile_package_names(loaded_packages, family):
    prefix = "amdrocm-{}{}-gfx".format(family, _ROCM_VERSION)
    return sorted([
        pkg_name
        for pkg_name in loaded_packages.keys()
        if pkg_name.startswith(prefix)
    ])

def _family_runfile_labels(loaded_packages, family):
    return [
        "@{}//:runfiles".format(_rocm_repo_name(pkg_name))
        for pkg_name in _family_runfile_package_names(loaded_packages, family)
    ]

def _rocm_runfile_build_files(loaded_packages):
    rocm_packages = {}

    for pkg_name in _family_runfile_package_names(loaded_packages, "blas"):
        rocm_packages[pkg_name] = _glob_filegroup("runfiles", [
            ".kpack/**",
            "lib/hipblaslt/library/**",
            "lib/rocblas/library/**",
        ])

    for pkg_name in _family_runfile_package_names(loaded_packages, "dnn"):
        rocm_packages[pkg_name] = _glob_filegroup("runfiles", [
            "lib/libMIOpenCKGroupedConv_*.so",
            "share/miopen/db/**",
        ])

    for pkg_name in _family_runfile_package_names(loaded_packages, "fft"):
        rocm_packages[pkg_name] = _glob_filegroup("runfiles", [".kpack/**"])

    for pkg_name in _family_runfile_package_names(loaded_packages, "rccl"):
        rocm_packages[pkg_name] = _glob_filegroup("runfiles", [".kpack/**"])

    return rocm_packages

def _rocm_base_build_files(loaded_packages):
    return {
        _rocm_package_name("amdrocm-amdsmi"): "\n\n".join([
            packages.cc_library(name = "amdsmi", hdrs = ["include/amd_smi/amdsmi.h"], includes = ["include/amd_smi"]),
            packages.filegroup(name = "libamd_smi", srcs = ["lib/libamd_smi.so.26"]),
        ]),
        _rocm_package_name("amdrocm-base"): "\n\n".join([
            _rocm_dlopen_patchelf(name = "rocm-core", src = "lib/librocm-core.so.1"),
            _rocm_dlopen_patchelf(name = "rocm_smi", src = "lib/librocm_smi64.so.1"),
            _rocm_dlopen_patchelf(name = "rocprofiler-register", src = "lib/librocprofiler-register.so.0"),
        ]),
        _rocm_package_name("amdrocm-blas-dev"): _glob_filegroup("headers", ["include/**"]),
        _rocm_package_name("amdrocm-blas-host"): "\n\n".join([
            _rocm_dlopen_patchelf(name = "hipblas", src = "lib/libhipblas.so.3"),
            packages.filegroup(name = "hipblaslt", srcs = ["lib/libhipblaslt.so.1"]),
            _rocm_dlopen_patchelf(name = "hipsparselt", src = "lib/libhipsparselt.so.0"),
            packages.filegroup(name = "rocblas", srcs = ["lib/librocblas.so.5"]),
            _rocm_dlopen_patchelf(name = "rocroller", src = "lib/librocroller.so.1"),
            _glob_filegroup("hipblaslt_support", [
                "lib/hipdnn_plugins/engines/libhipblaslt_plugin.so",
                "share/hipblaslt/**",
            ]),
        ]),
        _rocm_package_name("amdrocm-blas"): "\n\n".join([
            packages.filegroup(name = "hipblaslt", srcs = [
                "@amdrocm-blas-host//:hipblaslt",
                "@amdrocm-blas-host//:rocroller",
            ]),
            packages.filegroup(name = "hipblaslt_runfiles", srcs = _family_runfile_labels(loaded_packages, "blas") + [
                "@amdrocm-blas-host//:hipblaslt_support",
            ]),
            packages.filegroup(name = "rocblas", srcs = [
                "@amdrocm-blas-host//:hipblas",
                "@amdrocm-blas-host//:hipsparselt",
                "@amdrocm-blas-host//:rocblas",
            ]),
            packages.filegroup(name = "rocblas_runfiles", srcs = _family_runfile_labels(loaded_packages, "blas")),
        ]),
        _rocm_package_name("amdrocm-dnn-host"): "\n\n".join([
            _rocm_dlopen_patchelf(name = "MIOpen", src = "lib/libMIOpen.so.1"),
            _rocm_dlopen_patchelf(name = "hipdnn_backend", src = "lib/libhipdnn_backend.so"),
            _glob_filegroup("runfiles", [
                "lib/hipdnn_plugins/**",
                "share/miopen/**",
            ]),
        ]),
        _rocm_package_name("amdrocm-dnn"): packages.filegroup(name = "MIOpen", srcs = [
            "@amdrocm-dnn-host//:MIOpen",
            "@amdrocm-dnn-host//:hipdnn_backend",
            "@amdrocm-dnn-host//:runfiles",
        ] + _family_runfile_labels(loaded_packages, "dnn")),
        _rocm_package_name("amdrocm-fft-host"): "\n\n".join([
            _rocm_dlopen_patchelf(name = "hipfft", src = "lib/libhipfft.so.0"),
            _rocm_dlopen_patchelf(name = "hipfftw", src = "lib/libhipfftw.so.0"),
            packages.filegroup(name = "rocfft", srcs = ["lib/librocfft.so.0"]),
        ]),
        _rocm_package_name("amdrocm-fft"): "\n\n".join([
            packages.filegroup(name = "hipfft", srcs = [
                "@amdrocm-fft-host//:hipfft",
                "@amdrocm-fft-host//:hipfftw",
            ]),
            packages.filegroup(name = "rocfft", srcs = [
                "@amdrocm-fft-host//:rocfft",
            ] + _family_runfile_labels(loaded_packages, "fft")),
        ]),
        _rocm_package_name("amdrocm-llvm"): "\n\n".join([
            packages.filegroup(name = "llvm_libs", srcs = [
                "lib/llvm/lib/libLLVM.so.23.0git",
                "lib/llvm/lib/libclang-cpp.so.23.0git",
            ]),
            _glob_filegroup("rocm_device_libs_runfiles", ["lib/llvm/amdgcn/**"]),
        ]),
        _rocm_package_name("amdrocm-math-common"): _glob_filegroup("host_math_libs", ["lib/host-math/lib/*.so*"]),
        _rocm_package_name("amdrocm-profiler-base"): "\n\n".join([
            _rocm_dlopen_patchelf(name = "hsa-amd-aqlprofile", src = "lib/libhsa-amd-aqlprofile64.so.1"),
            _rocm_dlopen_patchelf(name = "rocprofiler-sdk_so", src = "lib/librocprofiler-sdk.so.1"),
            packages.filegroup(name = "rocprofiler-sdk-attach", srcs = ["lib/librocprofiler-sdk-attach.so.1"]),
            packages.filegroup(name = "rocprofiler-sdk-rocattach", srcs = ["lib/librocprofiler-sdk-rocattach.so.1"]),
            _rocm_dlopen_patchelf(name = "roctracer", src = "lib/libroctracer64.so.4"),
            _rocm_dlopen_patchelf(name = "roctx", src = "lib/libroctx64.so.4"),
            packages.cc_library_hdrs_glob(
                name = "roctx_headers",
                hdrs_glob = ["include/rocprofiler-sdk-roctx/**/*.h"],
                includes = ["include"],
            ),
            packages.filegroup(name = "rocprofiler-sdk", srcs = [
                ":rocprofiler-sdk_so",
                ":rocprofiler-sdk-attach",
                ":rocprofiler-sdk-rocattach",
            ]),
            _glob_filegroup("rocprofv3_tool", [
                "bin/rocprof*",
                "bin/rocpd*",
                "libexec/rocprofiler-sdk/**",
                "share/rocprofiler-sdk/**",
            ]),
        ]),
        _rocm_package_name("amdrocm-rccl-host"): packages.filegroup(name = "rccl", srcs = ["lib/librccl.so.1"]),
        _rocm_package_name("amdrocm-rccl"): packages.filegroup(name = "rccl", srcs = [
            "@amdrocm-rccl-host//:rccl",
        ] + _family_runfile_labels(loaded_packages, "rccl")),
        _rocm_package_name("amdrocm-runtime"): "\n\n".join([
            _rocm_dlopen_patchelf(name = "amd_comgr", src = "lib/libamd_comgr.so.3"),
            _rocm_dlopen_patchelf(name = "amdhip_so", src = "lib/libamdhip64.so.7"),
            _rocm_dlopen_patchelf(name = "hiprtc_so", src = "lib/libhiprtc.so.7"),
            _rocm_dlopen_patchelf(name = "hsa-runtime", src = "lib/libhsa-runtime64.so.1"),
            packages.filegroup(name = "amdhip", srcs = [
                ":amdhip_so",
                "lib/librocm_kpack.so.0",
            ]),
            packages.filegroup(name = "hiprtc", srcs = [
                ":hiprtc_so",
                "lib/libhiprtc-builtins.so.7",
            ]),
        ]),
        _rocm_package_name("amdrocm-runtime-dev"): _glob_filegroup("headers", ["include/**"]),
        _rocm_package_name("amdrocm-solver-host"): "\n\n".join([
            _rocm_dlopen_patchelf(name = "hipsolver", src = "lib/libhipsolver.so.1"),
            packages.filegroup(name = "rocsolver", srcs = ["lib/librocsolver.so.0"]),
        ]),
        _rocm_package_name("amdrocm-solver"): "\n\n".join([
            packages.filegroup(name = "hipsolver", srcs = ["@amdrocm-solver-host//:hipsolver"]),
            packages.filegroup(name = "rocsolver", srcs = [
                "@amdrocm-solver-host//:rocsolver",
                "@amdrocm-math-common//:host_math_libs",
            ]),
        ]),
        _rocm_package_name("amdrocm-sparse-host"): "\n\n".join([
            _rocm_dlopen_patchelf(name = "hipsparse", src = "lib/libhipsparse.so.4"),
            packages.filegroup(name = "rocsparse", srcs = ["lib/librocsparse.so.1"]),
        ]),
        _rocm_package_name("amdrocm-sparse"): packages.filegroup(name = "rocsparse", srcs = [
            "@amdrocm-sparse-host//:hipsparse",
            "@amdrocm-sparse-host//:rocsparse",
        ]),
        _rocm_package_name("amdrocm-sysdeps"): _glob_filegroup("system_libs", [
            "lib/rocm_sysdeps/lib/*.so*",
            "lib/rocm_sysdeps/share/**",
        ]),
    }

def _rocm_build_files(loaded_packages):
    rocm_packages = _rocm_base_build_files(loaded_packages)
    rocm_packages.update(_rocm_runfile_build_files(loaded_packages))
    return rocm_packages

def _rocm_impl(mctx):
    loaded_packages = packages.read(mctx, [
        "@zml//platforms/rocm:packages.lock.json",
    ])

    for pkg_name, build_file_content in _UBUNTU_PACKAGES.items():
        pkg = loaded_packages[pkg_name]["amd64"]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    for pkg_name, build_file_content in _rocm_build_files(loaded_packages).items():
        pkg = loaded_packages[pkg_name]["amd64"]
        http_deb_archive(
            name = _rocm_repo_name(pkg_name),
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            strip_prefix = _ROCM_STRIP_PREFIX,
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    http_file(
        name = "libdrm_mesa_amdgpu_ids",
        url = "https://gitlab.freedesktop.org/mesa/libdrm/-/raw/979f607906ad64f629967ac1f3ba3590e756442c/data/amdgpu.ids?inline=false",
        sha256 = "ffd2a8f1bfa755f4d90f537b4969fc4676f116e5af051ce2f18ef93a96d8beb6",
        downloaded_file_path = "amdgpu.ids",
    )

    http_archive(
        name = "libpjrt_rocm",
        build_file = "libpjrt_rocm.BUILD.bazel",
        url = _PJRT_ROCM_URL,
        sha256 = _PJRT_ROCM_SHA256,
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "amdrocm-amdsmi",
            "amdrocm-base",
            "amdrocm-blas",
            "amdrocm-dnn",
            "amdrocm-fft",
            "amdrocm-llvm",
            "amdrocm-math-common",
            "amdrocm-profiler-base",
            "amdrocm-rccl",
            "amdrocm-runtime",
            "amdrocm-solver",
            "amdrocm-sparse",
            "amdrocm-sysdeps",
            "libatomic1",
            "libdrm-common",
            "libpjrt_rocm",
        ],
        root_module_direct_dev_deps = [],
    )

rocm_packages = module_extension(
    implementation = _rocm_impl,
)
