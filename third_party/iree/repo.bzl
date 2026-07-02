load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def repo():
    git_repository(
        name = "iree",
        remote = "https://github.com/iree-org/iree.git",
        commit = "4d4e97d00f099a21f38eeff26f82a6d9e3643a11",
        sparse_checkout_patterns = [
            "runtime/src/**",
            "build_tools/bazel/**",
            "build_tools/third_party/libbacktrace/**",
            "BUILD.bazel",
        ],
        patches = [
            "//third_party/iree:tokenizer-only.patch",
            "//third_party/iree:fix-added-token-matching.patch",
            "//third_party/iree:match-hf-tokenizer.patch",
        ],
        patch_args = ["-p1"],
    )

    # Use this if you want to use a local copy of IREE instead of the git repository. Make sure to update the path to point to your local IREE checkout.
    #new_local_repository(name = "iree", build_file="//:third_party/iree/BUILD.bazel", path="/home/corendos/iree/")
