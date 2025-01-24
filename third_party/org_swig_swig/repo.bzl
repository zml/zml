load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "org_swig_swig",
        url = "http://prdownloads.sourceforge.net/swig/swig-4.3.0.tar.gz",
        sha256 = "f7203ef796f61af986c70c05816236cbd0d31b7aa9631e5ab53020ab7804aa9e",
        strip_prefix = "swig-4.3.0",
        build_file = "//:third_party/org_swig_swig/swig.bazel",
    )
