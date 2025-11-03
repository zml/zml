load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "9a77a882bb2bc75cb8c29620ff8cd0fd089bdc86",
        patch_args = ["-p1"],
        patches = [
            "third_party/xla/patches/0001-PjRT-C-API-male-header-C-compliant-for-PJRT-FFI-exte.patch",
        ],
    )
