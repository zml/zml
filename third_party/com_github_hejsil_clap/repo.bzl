load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "com_github_hejsil_clap",
        remote = "https://github.com/devoc09/zig-clap.git",
        commit = "3c35ff18bb32fec2f53e12e6bdbb00422d6c33fe",
        build_file = "//:third_party/com_github_hejsil_clap/clap.bazel",
    )
