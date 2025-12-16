load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "com_github_hejsil_clap",
        remote = "https://github.com/Hejsil/zig-clap.git",
        commit = "3ae92228409d82a2c906fafc228b7920d9ff046b",
        build_file = "//:third_party/com_github_hejsil_clap/clap.bazel",
    )
