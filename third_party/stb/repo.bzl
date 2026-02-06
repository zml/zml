load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "stb",
        build_file = "//third_party/stb:stb.BUILD",
        urls = ["https://github.com/nothings/stb/archive/master.tar.gz"],
        strip_prefix = "stb-master",
        sha256 = "aa1cd65973cf814b11e5823889cb10650cf25c1badf11a80bbbc23e0c32622ee",
    )
