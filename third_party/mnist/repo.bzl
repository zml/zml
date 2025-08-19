load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "mnist",
        sha256 = "075905e433ea0cce13c3fc08832448ab86225d089b5d412be67f59c29388fb19",
        url = "https://mirror.zml.ai/data/mnist.tar.zst",
        build_file_content = """exports_files(glob(["**"]), visibility = ["//visibility:public"])""",
    )
