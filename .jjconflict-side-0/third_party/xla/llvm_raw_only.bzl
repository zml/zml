load("@xla//third_party/llvm:workspace.bzl", llvm = "repo")

def _llvm_raw_only_impl(mctx):
    llvm("llvm-raw")
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["llvm-raw"],
        root_module_direct_dev_deps = [],
    )

llvm_raw_only = module_extension(
    implementation = _llvm_raw_only_impl,
)
