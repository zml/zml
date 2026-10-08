load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "lunasvg",
        urls = ["https://github.com/sammycage/lunasvg/archive/refs/tags/v3.5.0.tar.gz"],
        sha256 = "1abf1472ee6c4d19797916e8cc3c2e4b628e0d81178ffac60bdb0d457e32c690",
        strip_prefix = "lunasvg-3.5.0",
        build_file = "//third_party/lunasvg:lunasvg.bazel",
    )
