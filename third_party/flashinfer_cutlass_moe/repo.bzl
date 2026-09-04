load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_RELEASE = "cutlass-moe-v0.2.0"

_ASSETS = {
    "amd64": {
        "sha256": "5c3a29da0fd91e2281d046eb2e467a7a40aeb24a0be8ba0487006356d45fecc4",
        "url": "https://github.com/zml/flashinfer/releases/download/{release}/flashinfer-cutlass-moe_linux-amd64.tar.gz",
    },
    "arm64": {
        "sha256": "60b6fdbd193e11c24589add1eb206f59a3900cde4760e53719d6b2add492343e",
        "url": "https://github.com/zml/flashinfer/releases/download/{release}/flashinfer-cutlass-moe_linux-arm64.tar.gz",
    },
}

def _archive(name, arch):
    asset = _ASSETS[arch]

    http_archive(
        name = name,
        urls = [
            asset["url"].format(release = _RELEASE),
        ],
        sha256 = asset["sha256"],
        build_file = Label(
            "//third_party/flashinfer_cutlass_moe:flashinfer_cutlass_moe.BUILD.bazel",
        ),
    )

def repo():
    _archive(
        name = "flashinfer_cutlass_moe_linux_amd64",
        arch = "amd64",
    )
    _archive(
        name = "flashinfer_cutlass_moe_linux_arm64",
        arch = "arm64",
    )
