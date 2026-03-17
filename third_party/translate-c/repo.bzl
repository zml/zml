load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "translate-c",
        build_file = "//third_party/translate-c:translate-c.bazel",
        sha256 = "506643ec817620025d341efd87169d90013aa369db28780e853cf18bff174326",
        url = "https://codeberg.org/ziglang/translate-c/archive/d2be2f19ef7c9caa1561d38c96581ac79dd4c654.tar.gz",
        strip_prefix = "translate-c",
    )
