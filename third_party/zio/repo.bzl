load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zio",
        remote = "https://github.com/lalinsky/zio",
        commit = "281d969fdefe04f2e8b28d1f45fac0011f881c73",  # v0.19.0
        build_file = Label("//third_party/zio:zio.bazel"),
    )
