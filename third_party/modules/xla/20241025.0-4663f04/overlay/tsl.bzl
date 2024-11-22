load("//third_party:repo.bzl", "tf_vendored")
load("//third_party/py:python_init_repositories.bzl", "python_init_repositories")

def _tsl_impl(mctx):
    python_init_repositories(
        requirements = {
            "3.11": "//:requirements_lock_3_11.txt",
        },
    )
    tf_vendored(name = "tsl", relpath = "third_party/tsl")
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["tsl"],
        root_module_direct_dev_deps = [],
    )

tsl = module_extension(
    implementation = _tsl_impl,
)
