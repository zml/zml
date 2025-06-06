load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "com_google_sentencepiece",
        remote = "https://github.com/google/sentencepiece.git",
        commit = "273449044caa593c2fd7eb7550cb3ab2cff93f1a",
        build_file = "//third_party/com_google_sentencepiece:sentencepiece.bazel",
    )
