load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "47f005bb8150a13cb0217c2d7daf108bcdca34cc",
        patch_args = ["-p0"],
        patches = ["//third_party/xla:xspace_to_perfetto.patch"],
    )
