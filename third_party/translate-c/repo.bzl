load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "translate-c",
        remote = "https://github.com/zml/translate-c.git",
        commit = "5308ec6eba13a96dcaec452bffd0cc946384909a",
        build_file = "//third_party/translate-c:translate-c.bazel",
    )
