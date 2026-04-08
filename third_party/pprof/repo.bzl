load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_PPROF_COMMIT = "545e8a4df9364095d66e521b8f515f7af961e653"
_PPROF_SHA256 = "cbbaa59897eb698ff8d0c96568f87f8df89663cac60fdd4d691dcd5395724152"

def repo():
    http_archive(
        name = "pprof",
        sha256 = _PPROF_SHA256,
        strip_prefix = "pprof-" + _PPROF_COMMIT,
        urls = ["https://github.com/google/pprof/archive/" + _PPROF_COMMIT + ".tar.gz"],
        build_file = "//third_party/pprof:pprof.bazel",
    )
