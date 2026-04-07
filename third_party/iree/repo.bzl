load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "iree",
        remote = "https://github.com/iree-org/iree.git",
        commit = "71af3a5e41a8e265330bc693194c708cf6df4724",
        sparse_checkout_patterns = [
            "runtime/src/**",
            "build_tools/bazel/**",
            "build_tools/third_party/libbacktrace/**",
            "BUILD.bazel",
        ],
        patches = [
            "//third_party/iree:tokenizer-only.patch",
        ],
        patch_args = ["-p1"],
    )

    #new_local_repository(name = "flashattn", build_file="//:third_party/flashattn/BUILD.bazel", path="/home/corendos/flashattn/")

