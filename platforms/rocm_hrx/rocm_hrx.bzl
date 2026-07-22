load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

_PJRT_ROCM_HRX_URL = "https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-07-22T11-00-00Z/pjrt-rocm_hrx_linux-amd64.tar.gz"
_PJRT_ROCM_HRX_SHA256 = "a916d565752e7ec67ca94e656520020b66dbe8f20dbe98331f1ba762c4258c04"
_ROCM_HRX_URL = "https://github.com/zml/hrx-system/releases/download/v0.2/hrx-rocm-linux-amd64.tar.gz"
_ROCM_HRX_SHA256 = "f00c2aec515b494fee891acaffaad43b93584c331f550ce19716735922f1c526"

def _rocm_hrx_impl(mctx):
    http_archive(
        name = "libpjrt_rocm_hrx",
        build_file = "//platforms/rocm_hrx:libpjrt_rocm_hrx.BUILD.bazel",
        url = _PJRT_ROCM_HRX_URL,
        sha256 = _PJRT_ROCM_HRX_SHA256,
    )

    http_archive(
        name = "rocm_hrx",
        build_file = "//platforms/rocm_hrx:libpjrt_rocm_hrx.BUILD.bazel",
        url = _ROCM_HRX_URL,
        sha256 = _ROCM_HRX_SHA256,
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "libpjrt_rocm_hrx",
            "rocm_hrx",
        ],
        root_module_direct_dev_deps = [],
    )

rocm_hrx_packages = module_extension(
    implementation = _rocm_hrx_impl,
)
