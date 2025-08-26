load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "mnist",
        sha256 = "9c7d9a5ef9c245084996f6d2ec66ef176e51186e6a5b22efdcc3828d644941ca",
        url = "https://mirror.zml.ai/data/mnist_safetensors.tar.zst",
        build_file_content = """exports_files(glob(["**"]), visibility = ["//visibility:public"])""",
    )
