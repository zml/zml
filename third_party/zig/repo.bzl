load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def repo():
    # REQUIRED FOR KVX
    new_git_repository(
        name = "zig",
        remote = "https://codeberg.org/ziglang/zig.git",
        commit = "24fdd5b7a4c1c8b5deb5b56756b9dbc8e08c86a8", # 0.16.0
        build_file = "//third_party/zig:zig.bazel",
        patch_args = ["-p1"],
        patches = [
            "//third_party/zig/patches:0.16.0/0001-ZML-kvx-add-support-for-LLVM-backend.patch",
            "//third_party/zig/patches:0.16.0/0002-ZML-sema-target-authorize-Zig-types-in-kvx-callconv.patch",
            "//third_party/zig/patches:0.16.0/0003-ZML-kvx-add-ClusterOS-target.patch",
            "//third_party/zig/patches:0.16.0/0004-ZML-kvx-add-addrspace-support.patch",
            "//third_party/zig/patches:0.16.0/0005-ZML-zig-enable-configurable-LLVM-Clang-and-LLVM-AR.patch",
            ### Do not use this one until we bump to LLVM 22
            # "//third_party/zig/patches:0.16.0/0006-ZML-zig-adapt-to-LLVM-22-OptBisect-API-change.patch",
            ###
            "//third_party/zig/patches:0.16.0/0007-llvm-do-not-use-LLVM-aliases-for-nvptx-exports.patch",
            "//third_party/zig/patches:0.16.0/0008-sema-allow-packed-structs-as-Vector-element.patch",
            "//third_party/zig/patches:0.16.0/0009-codegen-llvm-allow-overriding-packed-struct-llvm-typ.patch",
            "//third_party/zig/patches:0.16.0/0010-Revert-llvm-disable-loop-vectorization-for-now.patch",
        ],
    )
    # new_local_repository(
    #     name = "zig",
    #     path = "./ziglang",
    #     build_file = "//third_party/zig:zig.bazel",
    # )
