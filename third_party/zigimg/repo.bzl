load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "zigimg",
        remote = "https://github.com/elogir/zigimg.git",
        commit = "0658b2c93b48db0ce2e363a5f7d1c971e64ae8f4",
        build_file = "//third_party/zigimg:zigimg.bazel",
    )
