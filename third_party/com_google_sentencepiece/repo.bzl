load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "com_google_sentencepiece",
        remote = "https://github.com/google/sentencepiece.git",
        commit = "d8f741853847553169444afc12c00f4bbff3e9ce",
        build_file = "//third_party/com_google_sentencepiece:sentencepiece.bazel",
    )
