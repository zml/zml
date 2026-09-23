load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

PJRT_MUSA_RELEASE = "202609221224.68.1.97e1cddb43de"
PJRT_MUSA_ARTIFACT_SHA256 = "f87883e1ec332d8dfd5a2943067dea9423ff1c2be298fcc4f2fc3209f66ce4b9"

PJRT_MUSA_ARTIFACT_URL = "https://mirror.zml.ai/plugins/{release}/zml-musa-linux-amd64.tar.zst".format(
    release = PJRT_MUSA_RELEASE,
)

MUSA_SDK_VERSION = "5.1.0"
MUSA_SDK_PACKAGE = "musa_sdk_5_1_0_cc2_2_deb"
MUSA_SDK_RELEASE = "musa-v{MUSA_SDK_VERSION}-{MUSA_SDK_PACKAGE}-ubuntu-x86_64".format(
    MUSA_SDK_PACKAGE = MUSA_SDK_PACKAGE,
    MUSA_SDK_VERSION = MUSA_SDK_VERSION,
)
MUSA_TOOLKIT_RELEASE = "musa-toolkit-{MUSA_SDK_VERSION}-{MUSA_SDK_PACKAGE}-ubuntu-x86_64.tar.zst".format(
    MUSA_SDK_PACKAGE = MUSA_SDK_PACKAGE,
    MUSA_SDK_VERSION = MUSA_SDK_VERSION,
)
MUSA_SDK_URL = "https://github.com/neudinger/rules-ml-toolchain-redists/releases/download/{MUSA_SDK_RELEASE}/{MUSA_TOOLKIT_RELEASE}".format(
    MUSA_SDK_RELEASE = MUSA_SDK_RELEASE,
    MUSA_TOOLKIT_RELEASE = MUSA_TOOLKIT_RELEASE,
)
MUSA_SDK_SHA256 = "afa05b1e73c4816e063fb695c889e37877599aa021a4ef8dba08998c1f3b1f9f"
MUSA_SDK_STRIP_PREFIX = "musa"

_MUSA_SDK_BUILD_FILE_CONTENT = """\
package(default_visibility = ["//visibility:public"])

exports_files(["toolchain-identity.txt"])

filegroup(
    name = "runtime_libs",
    srcs = [
        "lib/libmccl.so.2",
        "lib/libmublas.so.1",
        "lib/libmudnn.so.3",
        "lib/libmudnn_base.so.3",
        "lib/libmudnn_ops.so.3",
        "lib/libmudnn_tensor.so.3",
        "lib/libmudnn_tensor_binary.so.3",
        "lib/libmudnn_tensor_reduce.so.3",
        "lib/libmudnn_tensor_unary.so.3",
        "lib/libmudnn_xmma.so.3",
        "lib/libmufft.so.1",
        "lib/libmusart.so.5",
    ] + glob([
        "lib/libmtfft-device-*.so*",
    ]),
)

filegroup(
    name = "mcc",
    # Package the canonical compiler binary rather than the SDK's mcc -> clang
    # symlink. copy_to_directory dereferences that symlink, so retaining the
    # mcc filename would fail the bridge's pinned-SDK closure validation.
    srcs = ["bin/clang-14"],
)

filegroup(
    name = "clang_offload_bundler",
    srcs = ["bin/clang-offload-bundler"],
)

filegroup(
    name = "lld",
    srcs = ["bin/lld"],
)

filegroup(
    name = "llvm_readobj",
    srcs = ["bin/llvm-readobj"],
)

filegroup(
    name = "libclang_cpp",
    srcs = ["lib/libclang-cpp.so.14"],
)

filegroup(
    name = "libdevice",
    srcs = ["mtgpu/bitcode/libdevice.bc"],
)

filegroup(
    name = "intrinsics_musa_td",
    srcs = ["include/llvm/IR/IntrinsicsMUSA.td"],
)

filegroup(
    name = "builtins_mtgpu_def",
    srcs = ["include/clang/Basic/BuiltinsMTGPU.def"],
)

"""

def _musa_impl(mctx):
    sdk_sha256 = mctx.getenv("MUSA_DISTRO_HASH", MUSA_SDK_SHA256)
    http_archive(
        name = "libzml_musa",
        build_file = "//platforms/musa:libzml_musa.BUILD.bazel",
        sha256 = PJRT_MUSA_ARTIFACT_SHA256,
        url = PJRT_MUSA_ARTIFACT_URL,
    )
    http_archive(
        name = "musa_sdk",
        build_file_content = _MUSA_SDK_BUILD_FILE_CONTENT,
        # Fingerprint the selected SDK, independently of the plugin archive.
        generated_files = {
            "toolchain-identity.txt": "\n".join([
                "schema=xla-musa-toolchain-v1",
                "musa_version=5.1.0",
                "musa_version_number=50100",
                "musa_device=S4000",
                "musa_gpu_architectures=mp_22",
                "distro_sha256=" + sdk_sha256,
                "",
            ]),
        },
        sha256 = sdk_sha256,
        strip_prefix = mctx.getenv("MUSA_DISTRO_ROOT", MUSA_SDK_STRIP_PREFIX),
        url = mctx.getenv("MUSA_DISTRO_URL", MUSA_SDK_URL),
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "libzml_musa",
        ],
        root_module_direct_dev_deps = [],
    )

musa_packages = module_extension(
    implementation = _musa_impl,
)
