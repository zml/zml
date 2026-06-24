load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "6d7284490a942e5be5a771ded9181f9c602fd692",
        patches = [
            "//third_party/xla:pjrt-device-event-typedef.patch",
            "//third_party/xla:cuda-root-path-local-defines.patch",
        ],
        patch_args = ["-p1"],
    )
