load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def repo():
    http_archive(
        name = "flashattn",
        url = "https://github.com/zml/flash-attention/releases/download/v0.0.6-rc8/flashattn.tar.gz",
        sha256 = "360caace18b84f03779fe7af8aa1cdff64e99b6c17cbe05774be100aaf23b579",
        build_file = "//:third_party/flashattn/flashattention.BUILD.bazel",
    )

    #new_local_repository(name = "flashattn", build_file="//:third_party/flashattn/BUILD.bazel", path="/home/corendos/flashattn/")

