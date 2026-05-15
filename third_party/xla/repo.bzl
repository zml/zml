load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "47f005bb8150a13cb0217c2d7daf108bcdca34cc",
        patch_args = ["-p1"],
        patches = [
            "//third_party/xla:eigen_bazel9_loads.patch",
            "//third_party/xla:farmhash_bazel9_loads.patch",
            # "//third_party/xla:grpc_bazel9_native_cc.patch",
            "//third_party/xla:ml_dtypes_bazel9_loads.patch",
            "//third_party/xla:tsl_bazel9_loads.patch",
            # "//third_party/xla:xspace_to_perfetto.patch",
            "//third_party/xla:shardy_bazel9_loads.patch",
        ],
    )
