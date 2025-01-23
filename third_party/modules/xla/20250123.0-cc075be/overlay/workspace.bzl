load("@tsl//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("@tsl//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("@tsl//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("@tsl//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("@tsl//tools/toolchains/remote:configure.bzl", "remote_execution_configure")

def _xla_workspace_impl(mctx):
    cuda_configure(name = "local_config_cuda")
    remote_execution_configure(name = "local_config_remote_execution")
    rocm_configure(name = "local_config_rocm")
    tensorrt_configure(name = "local_config_tensorrt")
    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "b956598d8cbe168b5ee717b5dafa56563eb5201a947856a6688bbeac9cac4e1f",
        strip_prefix = "grpc-b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd",
        system_build_file = "@tsl//third_party/systemlibs:grpc.BUILD",
        patch_file = [
            "@tsl//third_party/grpc:generate_cc_env_fix.patch",
            "@tsl//third_party/grpc:register_go_toolchain.patch",
        ],
        system_link_files = {
            "@tsl//third_party/systemlibs:BUILD": "bazel/BUILD",
            "@tsl//third_party/systemlibs:grpc.BUILD": "src/compiler/BUILD",
            "@tsl//third_party/systemlibs:grpc.bazel.grpc_deps.bzl": "bazel/grpc_deps.bzl",
            "@tsl//third_party/systemlibs:grpc.bazel.grpc_extra_deps.bzl": "bazel/grpc_extra_deps.bzl",
            "@tsl//third_party/systemlibs:grpc.bazel.cc_grpc_library.bzl": "bazel/cc_grpc_library.bzl",
            "@tsl//third_party/systemlibs:grpc.bazel.generate_cc.bzl": "bazel/generate_cc.bzl",
            "@tsl//third_party/systemlibs:grpc.bazel.protobuf.bzl": "bazel/protobuf.bzl",
        },
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz"),
    )
    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = ["@tsl//third_party/protobuf:protobuf.patch"],
        sha256 = "f66073dee0bc159157b0bd7f502d7d1ee0bc76b3c1eac9836927511bdc4b3fc1",
        strip_prefix = "protobuf-3.21.9",
        system_build_file = "@tsl//third_party/systemlibs:protobuf.BUILD",
        system_link_files = {
            "@tsl//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
            "@tsl//third_party/systemlibs:protobuf_deps.bzl": "protobuf_deps.bzl",
        },
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/v3.21.9.zip"),
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

xla_workspace = module_extension(
    implementation = _xla_workspace_impl,
)
