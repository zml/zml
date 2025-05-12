load("@llvm-raw//utils/bazel:configure.bzl", _llvm_configure = "llvm_configure")

def _llvm_configure_impl(mctx):
    _llvm_configure(name = "llvm-project")
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

llvm_configure = module_extension(
    implementation = _llvm_configure_impl,
)
