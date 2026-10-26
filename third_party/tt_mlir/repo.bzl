load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Same TT-MLIR and LLVM revisions as the Tenstorrent XLA plugin.
def repo():
    http_archive(
        name = "tt_mlir",
        type = "tar.gz",
        urls = ["https://codeload.github.com/tenstorrent/tt-mlir/tar.gz/e2b21a721955b2b68540d9e6879a215ce3dcfae5"],
        sha256 = "e135e2a24c0c458451abffde3ded48f88c30b044af90b7be9daf587abf266089",
        strip_prefix = "tt-mlir-e2b21a721955b2b68540d9e6879a215ce3dcfae5",
        build_file = "//third_party/tt_mlir:tt_mlir.bazel",
        patches = ["//third_party/tt_mlir:llvm.patch", "//third_party/tt_mlir:standalone_ttir.patch"],
        patch_args = ["-p1"],
    )

    # Use the native C++ targets without FlatBuffers' unrelated JS toolchains.
    http_archive(
        name = "tt_flatbuffers",
        urls = ["https://github.com/google/flatbuffers/archive/refs/tags/v25.9.23.tar.gz"],
        integrity = "sha256-kQIlMhTepq4QwqyWbqHtIVXSIgI5C1MtHepkk1xRito=",
        strip_prefix = "flatbuffers-25.9.23",
    )
