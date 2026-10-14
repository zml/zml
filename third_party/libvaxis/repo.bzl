load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "libvaxis",
        remote = "https://github.com/rockorager/libvaxis.git",
        commit = "14b7684272234a3d66316c0d1bbdc876d4e8997c",
        build_file = "//third_party/libvaxis:libvaxis.bazel",
        patches = ["//third_party/libvaxis:fixes.patch"],
        patch_args = ["-p1"],
    )
