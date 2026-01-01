load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "arocc",
        remote = "https://github.com/zml/arocc.git",
        commit = "7e60c78a9660016e46d8be8907591b143ba2e700",
        build_file = "//third_party/arocc:arocc.bazel",
    )
