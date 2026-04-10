load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "arocc",
        remote = "https://github.com/Vexu/arocc.git",
        commit = "a08e2ca4b0e83059c3892b8b3e026754467e0ed1",
        build_file = "//third_party/arocc:arocc.bazel",
    )
