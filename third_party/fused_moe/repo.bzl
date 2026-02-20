load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def repo():
    new_local_repository(
        name = "fused_moe",
        path = "/home/louislechevalier/fused-moe",
        build_file = "//:third_party/fused_moe/BUILD.bazel",
    )
