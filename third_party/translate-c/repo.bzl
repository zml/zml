load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "translate-c",
        remote = "https://codeberg.org/ziglang/translate-c",
        commit = "5ac39f77661a216b75b195fe74ce7d0a04b33b7d",
        build_file = "//third_party/translate-c:translate-c.bazel",
    )
