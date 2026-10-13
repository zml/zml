load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zigimg",
        remote = "https://github.com/zigimg/zigimg.git",
        # Match libvaxis build.zig.zon.
        commit = "d695acd97c02e57bb151e8f659d1280f5cd6ca70",
        build_file = Label("//third_party/zigimg:zigimg.bazel"),
        patches = ["//third_party/zigimg:zig-0.17.patch"],
        patch_args = ["-p1"],
    )
