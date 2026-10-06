load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "libvaxis",
        remote = "https://github.com/rockorager/libvaxis.git",
        commit = "173a890d1394946b5d7623c66cd34bcd36d8eeb8",
        build_file = "//third_party/libvaxis:libvaxis.bazel",
        patches = ["//third_party/libvaxis:fixes.patch"],
        patch_args = ["-p1"],
    )
