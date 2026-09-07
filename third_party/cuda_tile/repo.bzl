load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# v13.3.3
def repo():
    git_repository(
        name = "cuda_tile",
        remote = "https://github.com/NVIDIA/cuda-tile.git",
        commit = "af2417041cc939b87ef56d92cfdcf61737c5457e",
        build_file = "//third_party/cuda_tile:cuda_tile.bazel",
    )
