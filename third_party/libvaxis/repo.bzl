load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "libvaxis",
        remote = "https://github.com/rockorager/libvaxis.git",
        commit = "25b4bd7371cc3c32049e103ab292a525a1ecb7a7",
        build_file = "//third_party/libvaxis:libvaxis.bazel",
        patches = ["//third_party/libvaxis:kitty-quiet.patch"],
        patch_args = ["-p1"],
    )
