load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "arocc",
        remote = "https://github.com/zml/arocc.git",
        commit = "73acf3aa8164cc8fee16b0f79036b92b07dab053",
        build_file = "//third_party/arocc:arocc.bazel",
    )
