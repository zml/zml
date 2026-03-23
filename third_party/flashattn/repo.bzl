load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def repo():
    http_archive(
        name = "flashattn",
        url = "https://github.com/zml/flash-attention/releases/download/v2026.3.23/flashattn-linux-amd64.tar.zst",
        sha256 = "79496eaa5a373170a201a31abe822c1774138fce8ffcd2aa39c6f0fc46c79e2d",
        build_file = "//:third_party/flashattn/flashattention.BUILD.bazel",
    )

    #new_local_repository(name = "flashattn", build_file="//:third_party/flashattn/BUILD.bazel", path="/home/corendos/flashattn/")
