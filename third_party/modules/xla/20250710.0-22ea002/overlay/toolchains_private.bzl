load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _toolchains_private_impl(mctx):
    http_archive(
        name = "rules_ml_toolchain",
        sha256 = "fb78d09234528aef2be856820b69b76486829f65e4eb3c7ffaa5803b667fa441",
        strip_prefix = "rules_ml_toolchain-f4ad89fa906be2c1374785a79335c8a7dcd49df7",
        urls = [
            "https://github.com/zml/rules_ml_toolchain/archive/f4ad89fa906be2c1374785a79335c8a7dcd49df7.tar.gz",
        ],
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

toolchains_private = module_extension(
    implementation = _toolchains_private_impl,
)
