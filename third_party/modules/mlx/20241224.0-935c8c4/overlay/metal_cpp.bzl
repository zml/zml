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
        sha256 = "d0a7990f43c7ce666036b5649283c9965df2f19a4a41570af0617bbe93b4a6e5",
        url = "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18-beta.zip",
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )
metal_cpp = module_extension(
    implementation = _metal_cpp_impl,
)
