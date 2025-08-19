load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "com_github_hejsil_clap",
        remote = "https://github.com/Hejsil/zig-clap.git",
        commit = "068c38f89814079635692c7d0be9f58508c86173",
        build_file = "//:third_party/com_github_hejsil_clap/clap.bazel",
    )
