load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "65aed3e0ed7b0f965825676dd27e312965c846c8",
        patches = [
            "//third_party/xla:cuda-root-path-local-defines.patch",
            "//third_party/xla:cpu-profiler.patch",
            "//third_party/xla:cpu-convolution-source-metadata.patch",
            "//third_party/xla:cpu-runtime-source-metadata.patch",
        ],
        patch_args = ["-p1"],
    )
