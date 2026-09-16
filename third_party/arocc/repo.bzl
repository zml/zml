load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "arocc",
        remote = "https://github.com/Vexu/arocc.git",
        commit = "ec463262c14c1111fc9323086b708ad3b0b9ca11",
        build_file = Label("//third_party/arocc:arocc.bazel"),
    )
