load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "org_swig_swig",
        url = "http://prdownloads.sourceforge.net/swig/swig-4.4.1.tar.gz",
        sha256 = "40162a706c56f7592d08fd52ef5511cb7ac191f3593cf07306a0a554c6281fcf",
        strip_prefix = "swig-4.4.1",
        build_file = "//:third_party/org_swig_swig/swig.bazel",
    )
