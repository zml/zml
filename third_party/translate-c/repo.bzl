load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "translate-c",
        # Use ZML Github mirror of translate-c because Codeber is too unstable
        remote = "https://github.com/zml/translate-c",
        commit = "6f9af6cd3579883ed816f14799474d8ad286a7ef",
        build_file = Label("//third_party/translate-c:translate-c.bazel"),
        patches = [Label("//third_party/translate-c:latest-arocc.patch")],
        patch_args = ["-p1"],
    )
