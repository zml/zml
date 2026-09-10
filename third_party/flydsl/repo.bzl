load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "flydsl",
        remote = "https://github.com/ROCm/FlyDSL.git",
        commit = "c62159d0c18e232794a1903f192fc094148f4a43",
        build_file = "//third_party/flydsl:flydsl.bazel",
        patches = [
            "//third_party/flydsl:atomic_exchange.patch",
            "//third_party/flydsl:mlir_property_ref.patch",
            "//third_party/flydsl:vector_atomic.patch",
            "//third_party/flydsl:xf32_mfma.patch",
        ],
        patch_args = ["-p1"],
    )
