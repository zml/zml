load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "fc31477d352ed8edc223648524382f9b10a7699b",
        patches = [
            "//third_party/xla:cuda-root-path-local-defines.patch",
            "//third_party/xla:host-alwayslink.patch",
        ],
        patch_args = ["-p1"],
    )
