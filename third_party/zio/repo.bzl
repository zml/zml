load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zio",
        remote = "https://github.com/lalinsky/zio",
        commit = "54cb18f05d3feaec431353de85d5d10dac93d37b",
        build_file = Label("//third_party/zio:zio.bazel"),
        patches = [Label("//third_party/zio:progress-parent-file.patch")],
        patch_args = ["-p1"],
    )
