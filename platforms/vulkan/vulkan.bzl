_STAGED_SO_PATH = "/home/reesechong/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"

_BUILD_FILE_CONTENT = """\
load("@bazel_lib//lib:copy_to_directory.bzl", "copy_to_directory")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

copy_file(
    name = "libpjrt_vulkan.so",
    src = "pjrt_c_api_gpu_plugin.so",
    out = "lib/pjrt_c_api_gpu_plugin.so",
    allow_symlink = True,
)

copy_to_directory(
    name = "sandbox",
    srcs = [
        ":libpjrt_vulkan.so",
    ],
    add_directory_to_runfiles = False,
    include_external_repositories = ["**"],
)

cc_library(
    name = "libpjrt_vulkan",
    data = [":sandbox"],
    visibility = ["@zml//platforms/vulkan:__subpackages__"],
)
"""

def _libpjrt_vulkan_repo_impl(rctx):
    so_path = rctx.path(rctx.attr.so_path)
    if not so_path.exists:
        fail("Vulkan-enabled XLA PJRT plugin not found at {}. Build //xla/pjrt/c:pjrt_c_api_gpu_plugin first.".format(so_path))
    rctx.symlink(so_path, "pjrt_c_api_gpu_plugin.so")
    rctx.file("BUILD.bazel", _BUILD_FILE_CONTENT)

libpjrt_vulkan_repository = repository_rule(
    implementation = _libpjrt_vulkan_repo_impl,
    attrs = {
        "so_path": attr.string(
            mandatory = True,
            doc = "Absolute path to a locally built Vulkan-enabled XLA PJRT GPU plugin .so.",
        ),
    },
    local = True,
    doc = "Imports a locally built Vulkan-enabled XLA PJRT GPU plugin by absolute path.",
)

def _vulkan_impl(mctx):
    libpjrt_vulkan_repository(
        name = "libpjrt_vulkan",
        so_path = _STAGED_SO_PATH,
    )

    return mctx.extension_metadata(
        reproducible = False,
        root_module_direct_deps = ["libpjrt_vulkan"],
        root_module_direct_dev_deps = [],
    )

vulkan_packages = module_extension(
    implementation = _vulkan_impl,
)
