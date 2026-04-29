load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

_FLASHATTN_RELEASE = "v0.0.6-rc9"

_FLASHATTN_ASSETS = {
    "amd64": {
        "sha256": "cb341c14ee0fc5f02a262497cbc80b552c23bce594d087a2aa278f14cfc82ef2",
        "url": "https://github.com/zml/flash-attention/releases/download/{release}/flash-attention_linux-amd64.tar.gz",
    },
    "arm64": {
        "sha256": "55feeccb4b3728dec08da2aa051109d0b9fbd8c0d0786c878976c1b713e3b407",
        "url": "https://github.com/zml/flash-attention/releases/download/{release}/flash-attention_linux-arm64.tar.gz",
    },
}

def flasshattn_archive(name, arch):
    asset = _FLASHATTN_ASSETS[arch]
    http_archive(
        name = name,
        url = asset["url"].format(release = _FLASHATTN_RELEASE),
        sha256 = asset["sha256"],
        build_file = "//:third_party/flashattn/flashattention.BUILD.bazel",
    )

def repo():
    flasshattn_archive("flashattn_linux_amd64", "amd64")
    flasshattn_archive("flashattn_linux_arm64", "arm64")

    #new_local_repository(name = "flashattn", build_file="//:third_party/flashattn/BUILD.bazel", path="/home/corendos/flashattn/")
