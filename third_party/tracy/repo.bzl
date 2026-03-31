load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "tracy",
        url = "https://github.com/wolfpld/tracy/archive/refs/tags/v0.13.1.tar.gz",
        strip_prefix = "tracy-0.13.1",
        build_file = "//third_party/tracy:tracy.bazel",
    )
