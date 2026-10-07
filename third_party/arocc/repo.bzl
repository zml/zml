load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "arocc",
        remote = "https://github.com/Vexu/arocc.git",
        commit = "5f5a050569a95ecc40a426f0c3666ae7ef987ede",
        build_file = Label("//third_party/arocc:arocc.bazel"),
    )
