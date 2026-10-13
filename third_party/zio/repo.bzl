load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zio",
        remote = "https://github.com/lalinsky/zio",
        commit = "1b4e9787f8a4a67d91372a603b23258364f89bf0",
        build_file = Label("//third_party/zio:zio.bazel"),
    )
