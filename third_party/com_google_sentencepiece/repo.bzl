load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "com_google_sentencepiece",
        url = "https://github.com/google/sentencepiece/releases/download/v0.2.1/sentencepiece-0.2.1.tar.gz",
        strip_prefix = "sentencepiece-0.2.1/sentencepiece",
        sha256 = "8138cec27c2f2282f4a34d9a016e3374cd40e5c6e9cb335063db66a0a3b71fad",
        build_file = "//third_party/com_google_sentencepiece:sentencepiece.bazel",
    )
