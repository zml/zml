load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "01da52b9afe3a2e694bd926323a649a6e63a3785",
    )
