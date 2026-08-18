load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_RELEASE = "cutlass-moe-v0.1.0"

_ASSETS = {
    "amd64": {
        "sha256": "d40348ad771e2bc34b3dd59602ed5f3ef47b58da06ddad582a1fd993f872fb40",
        "url": "https://github.com/zml/flashinfer/releases/download/{release}/flashinfer-cutlass-moe_linux-amd64.tar.gz",
    },
    "arm64": {
        "sha256": "4278b3900d789e2ad6c207f7e698728a0b68ab200a0fc815625dfc25a2a6e1a8",
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
