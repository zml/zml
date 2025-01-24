load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def _non_module_deps_impl(mctx):

    new_git_repository(
        name = "com_github_hejsil_clap",
        remote = "https://github.com/Hejsil/zig-clap.git",
        commit = "d71cc39a94f3e6ccbad00c25d350c9147de4df9f",
        build_file = "//:third_party/com_github_hejsil_clap/clap.bazel",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

non_module_deps = module_extension(
    implementation = _non_module_deps_impl,
)
