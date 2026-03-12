load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "translate-c",
        remote = "https://codeberg.org/ziglang/translate-c.git",
        commit = "41c10fa66ac81343c33f2b8c746f181b41eaaa27",
        build_file = "//third_party/translate-c:translate-c.bazel",
    )
