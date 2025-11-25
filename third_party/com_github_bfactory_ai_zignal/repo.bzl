load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "com_github_bfactory_ai_zignal",
        remote = "https://github.com/loupicaaa/zignal.git",
        commit = "21553a48014add0e7f069f8c72b9277786185127",
        build_file = "//:third_party/com_github_bfactory_ai_zignal/zignal.bazel",
    )
