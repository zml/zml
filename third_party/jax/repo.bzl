load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "jax",
        remote = "https://github.com/jax-ml/jax.git",
        commit = "92348e8631f89bc9e4071f918524b432bcb73f97",
        build_file = "//third_party/jax:jax.bazel",
        patch_cmds = ["find . -name BUILD -delete"],
    )
