load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zigimg",
        remote = "https://github.com/zigimg/zigimg.git",
        # Match libvaxis build.zig.zon.
        commit = "734fa919fb15b7917d4eb124f53600348f4ffa30",
        build_file = Label("//third_party/zigimg:zigimg.bazel"),
    )
