load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zio",
        remote = "https://github.com/lalinsky/zio",
        # zig-0.17 branch
        commit = "b3475afacc7674f01842a9b1e7499f0976972f22",
        build_file = Label("//third_party/zio:zio.bazel"),
        patches = [
            Label("//third_party/zio:progress-parent-file.patch"),
            Label("//third_party/zio:runtime-page-size.patch"),
        ],
        patch_args = ["-p1"],
    )
