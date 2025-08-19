load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "ef07e787ea1303fa2f8d8a175d24d434bfb84107",
        patch_args = ["-p1"],
        patches = [
            "//third_party/xla:patches/0001-bazel-migration-to-bazel-8.1.1.patch",
            "//third_party/xla:patches/0002-Added-FFI-handler-registration-API-to-the-FFI-PjRt.patch",
            "//third_party/xla:patches/0003-Remove-unconventional-C-code-in-headers.patch",
        ],
    )
