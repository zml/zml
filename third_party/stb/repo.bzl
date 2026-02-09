load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "stb",
        remote = "https://github.com/nothings/stb.git",
        build_file = "//third_party/stb:stb.BUILD",
       commit = "f1c79c02822848a9bed4315b12c8c8f3761e1296",
    )
