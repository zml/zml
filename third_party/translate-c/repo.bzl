load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "translate-c",
        # Use the ZML GitHub mirror because Codeberg is unreliable.
        remote = "https://github.com/zml/translate-c",
        # 2.0.0
        commit = "0da7a16c3235b935b82421646076e0657cda21f6",
        build_file = Label("//third_party/translate-c:translate-c.bazel"),
    )
