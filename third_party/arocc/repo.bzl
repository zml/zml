load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "arocc",
        remote = "https://github.com/Vexu/arocc.git",
        commit = "8a5da9a689c03ee1abec54767df15079d16ea030",
        build_file = Label("//third_party/arocc:arocc.bazel"),
    )
