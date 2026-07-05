load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "41370d1124c74d7b93a207136a636d8c631cbed9",
        patches = [
            # Bazel a3dc34c545 removed AutoloadSymbols, so these BUILD files
            # must load the rules they call.
            # 13cee57bf26faa5fb7c19ac7a293074f0d073264 deleted
            # xspace_to_perfetto.patch; keep its unrelated rule loads here.
            "//third_party/xla:explicit-rule-loads.patch",
            "//third_party/xla:cuda-root-path-local-defines.patch",
        ],
        patch_args = ["-p1"],
    )
