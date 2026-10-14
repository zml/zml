load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    # ZLS 0.17 development revision; dependencies match its build.zig.zon.
    git_repository(
        name = "zls",
        remote = "https://github.com/zigtools/zls.git",
        commit = "eab2be0fd74443a27662da809b771c4ad0d2afcf",
        build_file = Label("//third_party/zls:zls.bazel"),
    )
    git_repository(
        name = "zls_lsp",
        remote = "https://github.com/zigtools/lsp-kit.git",
        commit = "d148676ace1eaecf3c703ce7d350d0e94e502ed7",
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
