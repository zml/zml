load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zigimg",
        remote = "https://github.com/zigimg/zigimg.git",
        commit = "7b98e82621fe302a9edc147df1191f4d1b7ff7a5",
        build_file = Label("//third_party/zigimg:zigimg.bazel"),
        patches = [
            Label("//third_party/zigimg:zig-0.17.patch"),
        ],
    )
