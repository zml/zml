load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "cute_ir",
        remote = "https://github.com/NVIDIA/cutlass.git",
        commit = "147295a3d4b75f3aeff247c25b8927cea9a7006a",
        build_file = Label("//third_party/cute_ir:cute_ir.bazel"),
        strip_prefix = "cutlass_compiler",
    )
