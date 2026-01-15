load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def repo():
    http_archive(
        name = "flashattn",
        url = "https://github.com/zml/flash-attention/releases/download/v0.0.6-rc6/flashattn.tar.gz",
        sha256 = "ff44c0657c1e0d41883b6bc034340f68afe9c4f8cddfd9232c039f4c6b5e1da6",
        build_file = "//:third_party/flashattn/BUILD.bazel",
    )

    #new_local_repository(name = "flashattn", build_file="//:third_party/flashattn/BUILD.bazel", path="/home/corendos/flashattn/")

