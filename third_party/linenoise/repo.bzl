load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "linenoise",
        remote = "https://github.com/antirez/linenoise.git",
        commit = "452e3793858a15cc006bb3aa3ea06378eb98c5b5",
        build_file = "//third_party/linenoise:linenoise.BUILD",
        # patches = ["//third_party/linenoise:hint_key.patch"],
        # patch_args = ["-p1"],
    )
