load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# Match the CUDA Tile revision and patches used by the XLA plugins.
def repo():
    git_repository(
        name = "cuda_tile",
        remote = "https://github.com/NVIDIA/cuda-tile.git",
        commit = "e11729ed4e228af005dc8cc113d7b84157e9d654",
        build_file = "//third_party/cuda_tile:cuda_tile.bazel",
        patches = [
            "//third_party/cuda_tile:constructor.patch",
            "//third_party/cuda_tile:unused_variable.patch",
        ],
        patch_args = ["-p1"],
    )
