load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "translate-c",
        remote = "https://codeberg.org/ziglang/translate-c",
        commit = "46b5609b5ac4c0a896217d1d984f3ae50e4810b5",
        build_file = Label("//third_party/translate-c:translate-c.bazel"),
    )
