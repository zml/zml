load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "com_github_hejsil_clap",
        remote = "https://github.com/Hejsil/zig-clap.git",
        commit = "5289e0753cd274d65344bef1c114284c633536ea",
        build_file = "//:third_party/com_github_hejsil_clap/clap.bazel",
    )
