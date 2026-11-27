load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "uucode",
        remote = "https://github.com/jacobsandlund/uucode.git",
        # Match libvaxis build.zig.zon.
        commit = "1fb73433bba5d93366c57f23ff2e9d7939746500",
        build_file = Label("//third_party/uucode:uucode.bazel"),
        # Upstream build.zig relies on a cyclic config.zig <-> storage.zig
        # module import that Bazel's DAG cannot represent.
        patches = ["//third_party/uucode:config-storage-module.patch"],
        patch_args = ["-p1"],
    )
