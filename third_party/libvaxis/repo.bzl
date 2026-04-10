load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    new_git_repository(
        name = "libvaxis",
        remote = "https://github.com/rockorager/libvaxis.git",
        commit = "a3ae1d53feeeeaeb6218de3d38837559811acae4",
        build_file = "//third_party/libvaxis:libvaxis.bazel",
        patches = ["//third_party/libvaxis:bump-zig.patch", "//third_party/libvaxis:fixes.patch"],
        patch_args = ["-p1"],
    )
