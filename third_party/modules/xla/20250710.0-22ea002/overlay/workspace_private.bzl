load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls", "tf_vendored")
load("//third_party/py:python_repo.bzl", "python_repository")
load("//third_party/pybind11_bazel:workspace.bzl", pybind11_bazel = "repo")

def _workspace_private_impl(mctx):
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
