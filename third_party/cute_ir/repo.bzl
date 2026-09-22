load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    # The cutlass_compiler/ tree of the CUTLASS v4.8.0 release: the open-source
    # part of the CuTe dialect that mlir/dialects/cute_ir builds on.
    git_repository(
        name = "cute_ir",
        remote = "https://github.com/NVIDIA/cutlass.git",
        commit = "098de2a652cf8f00fd70b2df54051c7eccbb855a",
        build_file = Label("//third_party/cute_ir:cute_ir.bazel"),
        patches = [
            Label("//third_party/cute_ir:slice_tensor.patch"),
            Label("//third_party/cute_ir:dynamic_divisibility.patch"),
        ],
        patch_args = ["-p1"],
        strip_prefix = "cutlass_compiler",
        # The DSL spells a dynamic leaf it knows to be a multiple of N as
        # `?{div=N}`; the release's cutegen reads only the width.
        patches = [Label("//third_party/cute_ir:cutegen_dynamic_divisibility.patch")],
        patch_args = ["-p2"],
    )
