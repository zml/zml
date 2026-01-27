load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def repo():
    http_archive(
        name = "flashattn",
        url = "https://github.com/zml/flash-attention/releases/download/v0.0.6-rc7/flashattn.tar.gz",
        sha256 = "6b8195a5ab285404fcb084122c1c32f342ff695040b8652c9144afd2289b71b3",
        build_file = "//:third_party/flashattn/BUILD.bazel",
    )

    #new_local_repository(name = "flashattn", build_file="//:third_party/flashattn/BUILD.bazel", path="/home/corendos/flashattn/")

