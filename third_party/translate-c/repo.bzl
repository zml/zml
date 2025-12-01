load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "translate-c",
        remote = "https://codeberg.org/ziglang/translate-c.git",
        commit = "55f225f0a37b9627d459ed23b7df21f458f492d9",
        build_file = "//third_party/translate-c:translate-c.bazel",
    )
