load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    # https://github.com/zml/arocc/releases/tag/0.17.0
    git_repository(
        name = "arocc",
        remote = "https://github.com/zml/arocc",
        commit = "d0c8c4d9c55daa7ef6e40cf0f630a5b5e900989b",
        build_file = Label("//third_party/arocc:arocc.bazel"),
    )
