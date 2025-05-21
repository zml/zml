load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
def _metal_cpp_impl(mctx):
    http_archive(
        name = "metal_cpp",
        build_file_content = """\
cc_library(
    name = "metal_cpp",
    hdrs = glob(["metal-cpp/**/*.hpp"]),
    strip_include_prefix = "metal-cpp",
    visibility = ["//visibility:public"],
)
""",
        sha256 = "0433df1e0ab13c2b0becbd78665071e3fa28381e9714a3fce28a497892b8a184",
        url = "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip",
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )
metal_cpp = module_extension(
    implementation = _metal_cpp_impl,
)
