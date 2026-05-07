load(":repo.bzl", llvm = "repo")

def _llvm_repos_impl(mctx):
    llvm()
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

llvm_21_kvx = module_extension(
    implementation = _llvm_repos_impl,
)
