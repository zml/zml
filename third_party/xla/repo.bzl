load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "41370d1124c74d7b93a207136a636d8c631cbed9",
        patches = [
            "//third_party/xla:cuda-root-path-local-defines.patch",
        ],
        patch_args = ["-p1"],
    )
