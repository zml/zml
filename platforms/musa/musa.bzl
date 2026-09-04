load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

# bazel run //examples/benchmark \
#   --@zml//platforms:cpu=false \
#   --@zml//platforms:musa=true \
#   --override_repository=+musa_packages+libpjrt_musa=/home/kevin/musa/xla-override/libpjrt_musa/
PJRT_MUSA_RELEASE = "musa-5.1.0-s4000-local"
# Replace with the sha256sum of pjrt-musa_linux-amd64.tar.gz after publishing this release.
PJRT_MUSA_ARTIFACT_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
PJRT_MUSA_ARTIFACT_URL = "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-musa_linux-amd64.tar.gz".format(
    release = PJRT_MUSA_RELEASE,
)

MUSA_SDK_VERSION = "5.1.0"
MUSA_SDK_PACKAGE = "musa_sdk_5_1_0_cc2_2_deb"
MUSA_SDK_RELEASE = "musa-v{MUSA_SDK_VERSION}-{MUSA_SDK_PACKAGE}-ubuntu-x86_64".format(
    MUSA_SDK_PACKAGE = MUSA_SDK_PACKAGE,
    MUSA_SDK_VERSION = MUSA_SDK_VERSION
)
MUSA_TOOLKIT_RELEASE = "musa-toolkit-{MUSA_SDK_VERSION}-{MUSA_SDK_PACKAGE}-ubuntu-x86_64.tar.zst".format(
    MUSA_SDK_PACKAGE = MUSA_SDK_PACKAGE,
    MUSA_SDK_VERSION = MUSA_SDK_VERSION
)
MUSA_SDK_URL = "https://github.com/neudinger/rules-ml-toolchain-redists/releases/download/{MUSA_SDK_RELEASE}/{MUSA_TOOLKIT_RELEASE}".format(
    MUSA_SDK_RELEASE = MUSA_SDK_RELEASE,
    MUSA_TOOLKIT_RELEASE = MUSA_TOOLKIT_RELEASE
)
MUSA_SDK_SHA256 = "5407266eab8fe42caee83f6a7a979edeaa9ea6e542e197bf65fb5f394f1980b2"
MUSA_SDK_STRIP_PREFIX = "musa"

_MUSA_SDK_BUILD_FILE_CONTENT = """\
package(default_visibility = ["//visibility:public"])

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
        "lib/libmusa.so.1",
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

filegroup(
    name = "libmusart_5_1_0",
    srcs = ["lib/libmusart.so.5.1.0"],
)
"""

def _musa_impl(mctx):
    http_archive(
        name = "libpjrt_musa",
        # Use an explicit label so edits to the package BUILD definition are
        # repository-rule inputs and invalidate the generated repository.
        build_file = Label("//platforms/musa:libpjrt_musa.BUILD.bazel"),
        sha256 = PJRT_MUSA_ARTIFACT_SHA256,
        url = PJRT_MUSA_ARTIFACT_URL,
    )
    http_archive(
        name = "musa_sdk",
        build_file_content = _MUSA_SDK_BUILD_FILE_CONTENT,
        sha256 = mctx.getenv("MUSA_DISTRO_HASH", MUSA_SDK_SHA256),
        strip_prefix = mctx.getenv("MUSA_DISTRO_ROOT", MUSA_SDK_STRIP_PREFIX),
        url = mctx.getenv("MUSA_DISTRO_URL", MUSA_SDK_URL),
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "libpjrt_musa",
        ],
        root_module_direct_dev_deps = [],
    )

musa_packages = module_extension(
    implementation = _musa_impl,
)
