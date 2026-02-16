load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "linenoise",
        remote = "https://github.com/antirez/linenoise.git",
        commit = "dc83cc373ac2058030eb3cf5e404959a26fef112",
        build_file = "//third_party/linenoise:linenoise.BUILD",
    )
