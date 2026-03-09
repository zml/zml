load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "libvaxis",
        remote = "https://github.com/elogir/libvaxis.git",
        commit = "f88c96524af51ca08faa192a58565ab98383202d",
        build_file = "//third_party/libvaxis:libvaxis.bazel",
    )
