load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "b3fbfeeb076f2b536897180f4a274680ed9d52eb",
        patch_args = ["-p1"],
        patches = [
            # patches live in the patches directory
        ],
    )
