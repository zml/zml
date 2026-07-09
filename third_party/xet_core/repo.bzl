load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "xet_core",
        build_file = "//third_party/xet_core:xet_core.bazel",
        sha256 = "c083e5316c8f9ea4ed71c9b179462c72828990f75c2c107374c45e473faf13f2",
        strip_prefix = "xet-core-40865553bac8b9db25ec4974bb6923b86fee70df",
        urls = ["https://github.com/huggingface/xet-core/archive/40865553bac8b9db25ec4974bb6923b86fee70df.tar.gz"],
    )
