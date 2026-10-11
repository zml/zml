load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "tt_lang",
        remote = "https://github.com/tenstorrent/tt-lang.git",
        commit = "9b7ad24d1bd6a89dc1f8f885686bc4db86ac7244",
        build_file = "//third_party/tt_lang:tt_lang.bazel",
        patches = [
            "//third_party/tt_lang:mlir_legacy_assembly.patch",
            "//third_party/tt_lang:opaque_call_extensions.patch",
            "//third_party/tt_lang:reduce_defaults_guard.patch",
        ],
        patch_args = ["-p1"],
    )
