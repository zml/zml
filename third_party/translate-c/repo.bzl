load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "translate-c",
        remote = "https://codeberg.org/ziglang/translate-c.git",
        commit = "d2be2f19ef7c9caa1561d38c96581ac79dd4c654",
        build_file = "//third_party/translate-c:translate-c.bazel",
    )
