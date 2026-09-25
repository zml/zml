load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zig-protobuf",
        remote = "https://github.com/Arwalk/zig-protobuf",
        commit = "b794f99323cead7f1794ae68554d0311cc309857",
        # XLA ShapeProto and LayoutProto are mutually recursive.
        patches = [Label("//third_party/zig-protobuf:mutual-recursion.patch")],
        patch_args = ["-p1"],
        build_file = Label("//third_party/zig-protobuf:zig-protobuf.bazel"),
    )
