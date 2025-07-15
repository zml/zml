load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls", "tf_vendored")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/py:python_repo.bzl", "python_repository")
load("//third_party/pybind11_bazel:workspace.bzl", pybind11_bazel = "repo")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("//tools/toolchains/remote:configure.bzl", "remote_execution_configure")

def _workspace_private_impl(mctx):
    http_archive(
        name = "rules_ml_toolchain",
        sha256 = "fb78d09234528aef2be856820b69b76486829f65e4eb3c7ffaa5803b667fa441",
        strip_prefix = "rules_ml_toolchain-f4ad89fa906be2c1374785a79335c8a7dcd49df7",
        urls = [
            "https://github.com/zml/rules_ml_toolchain/archive/f4ad89fa906be2c1374785a79335c8a7dcd49df7.tar.gz",
        ],
    )

    # Use cuda_configure from XLA to make it work with bzlmod.
    # A pure bzlmod solution for rules_ml_toolchain is impossible because of the legacy design.
    # It relies on a "generate-then-load" pattern that creates a deadlock in Bazel's architecture:
    # - Generate: First, it runs a rule to generate a .bzl file containing configuration data.
    # - Load: Then, it requires a load() statement to load that same file to continue the setup.
    # This fails in bzlmod because Bazel's Loading Phase (when load() statements are processed) happens before
    # the Analysis Phase (when repository rules are run).
    # This creates a fundamental chicken-and-egg problem: the build tries to load a file that has not been generated yet.
    # Without using the official WORKSPACE.bzlmod escape hatch,
    # this incompatibility cannot be resolved without modifying the upstream rules.

    cuda_configure(name = "local_config_cuda")
    remote_execution_configure(name = "local_config_remote_execution")
    rocm_configure(name = "local_config_rocm")
    tensorrt_configure(name = "local_config_tensorrt")
    tf_vendored(name = "tsl", relpath = "third_party/tsl")
    pybind11_bazel()
    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "afbc5d78d6ba6d509cc6e264de0d49dcd7304db435cbf2d630385bacf49e066c",
        strip_prefix = "grpc-1.68.2",
        patch_file = [
            "//third_party/grpc:grpc.patch",
        ],
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/refs/tags/v1.68.2.tar.gz"),
    )
    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = ["//third_party/protobuf:protobuf.patch"],
        sha256 = "f645e6e42745ce922ca5388b1883ca583bafe4366cc74cf35c3c9299005136e2",
        strip_prefix = "protobuf-5.28.3",
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/refs/tags/v5.28.3.zip"),
    )

    python_repository(
        name = "python_version_repo",
        requirements_versions = ["3.11"],
        requirements_locks = ["//:requirements_lock_3_11.txt"],
        local_wheel_workspaces = [],
        local_wheel_dist_folder = None,
        default_python_version = None,
        local_wheel_inclusion_list = ["*"],
        local_wheel_exclusion_list = [],
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

workspace_private = module_extension(
    implementation = _workspace_private_impl,
)
