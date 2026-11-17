load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    # ZLS 0.17.0 and the dependencies pinned in its build.zig.zon.
    git_repository(
        name = "zls",
        remote = "https://github.com/zigtools/zls.git",
        commit = "542462c4ddf045493fa9ca028ee6bc3f38b2754a",
        build_file = Label("//third_party/zls:zls.bazel"),
        # Bazel's module paths are absolute, including generated and external files.
        patch_args = ["-p1"],
        patches = [Label("//third_party/zls:absolute-build-paths.patch")],
    )
    git_repository(
        name = "zls_lsp",
        remote = "https://github.com/zigtools/lsp-kit.git",
        commit = "3e1058db66079911fdde82f833fdce6856e0a706",
        build_file = Label("//third_party/zls:lsp.bazel"),
    )
    git_repository(
        name = "zls_diffz",
        remote = "https://github.com/ziglibs/diffz.git",
        commit = "aac8aa99c436ab8277b0711922aad062c0167b12",
        build_file = Label("//third_party/zls:diffz.bazel"),
    )
    git_repository(
        name = "zls_known_folders",
        remote = "https://github.com/ziglibs/known-folders.git",
        commit = "6da0d0c41b78b9ed2d34fa364fcb81b5ebece6c4",
        build_file = Label("//third_party/zls:known_folders.bazel"),
    )
