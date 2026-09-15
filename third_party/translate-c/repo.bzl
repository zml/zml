load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "translate-c",
        remote = "https://codeberg.org/ziglang/translate-c",
        commit = "0944784e197e419433a21d4b28bfc65e48e7d514",
        build_file = Label("//third_party/translate-c:translate-c.bazel"),
    )
