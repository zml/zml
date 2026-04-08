load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_TRACY_PUBLIC_COMMIT = "00a069d6088ff8d93304eaac4d925cece0e9081c"
_TRACY_PUBLIC_SHA256 = "fff8de03b18c9ea739c05dc471c3cd38a4f51c15910bc5cb4002d5d1b4d69325"

def repo():
    http_archive(
        name = "tracy",
        sha256 = _TRACY_PUBLIC_SHA256,
        strip_prefix = "tracy-" + _TRACY_PUBLIC_COMMIT,
        urls = ["https://github.com/wolfpld/tracy/archive/" + _TRACY_PUBLIC_COMMIT + ".tar.gz"],
        build_file = "//third_party/tracy:tracy.bazel",
    )
