load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# v13.4.0 with upstream LLVM compatibility fixes
def repo():
    git_repository(
        name = "cuda_tile",
        remote = "https://github.com/NVIDIA/cuda-tile.git",
        commit = "5b3fa4c0dd04d90edeb935a1887c20bf14a86ebe",
        build_file = "//third_party/cuda_tile:cuda_tile.bazel",
    )
