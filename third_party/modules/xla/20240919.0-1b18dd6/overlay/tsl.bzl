load("//third_party:repo.bzl", "tf_vendored")

def _tsl_impl(mctx):
    tf_vendored(name = "tsl", relpath = "third_party/tsl")
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

tsl = module_extension(
    implementation = _tsl_impl,
)
