load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zio",
        remote = "https://github.com/lalinsky/zio",
        commit = "417757935ef6d0f0a2396d4023a44ca82ac1901c",
        build_file = Label("//third_party/zio:zio.bazel"),
        patches = [
            Label("//third_party/zio:progress-parent-file.patch"),
            Label("//third_party/zio:runtime-page-size.patch"),
        ],
        patch_args = ["-p1"],
    )
