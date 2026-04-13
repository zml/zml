load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "arocc",
        remote = "https://github.com/Vexu/arocc.git",
        commit = "5f5a050569a95ecc40a426f0c3666ae7ef987ede",
        build_file = "//third_party/arocc:arocc.bazel",
    )
