load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "arocc",
        remote = "https://github.com/Vexu/arocc.git",
        commit = "3c7b4545b2ac30b9da41dbc8eacd2f14af44bc6c",
        build_file = "//third_party/arocc:arocc.bazel",
    )
