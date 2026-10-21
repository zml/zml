load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _metal_impl(mctx):
    http_archive(
        name = "libzml_metal",
        build_file = "libzml_metal.BUILD.bazel",
        sha256 = "e484f78bb14df98c4271f0d91760c32d167716b936ac465258fb7874282b3b2a",
        url = "https://mirror.zml.ai/plugins/202610191219.248.1.c2619532effe/zml-metal-darwin-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
