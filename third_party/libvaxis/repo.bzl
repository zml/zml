load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "libvaxis",
        remote = "https://github.com/elogir/libvaxis.git",
        commit = "102b13d199720dd93ec8cd9324148f77e3bf6ded",
        build_file = "//third_party/libvaxis:libvaxis.bazel",
    )
