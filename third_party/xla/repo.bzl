load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "f3dee9ae87745e151837b2ff1b0485928f8e5a5d",
        patches = [
            "//third_party/xla:cuda-root-path-local-defines.patch",
        ],
        patch_args = ["-p1"],
    )
