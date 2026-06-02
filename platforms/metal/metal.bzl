_STAGED_DYLIB_PATH = "/Users/raph/Documents/Git-Repos/xla/bazel-bin/xla/pjrt/c/libpjrt_c_api_gpu_plugin.dylib"

_BUILD_FILE_CONTENT = """\
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "libpjrt_metal",
    srcs = ["libpjrt_metal.dylib"],
    visibility = ["@zml//platforms/metal:__subpackages__"],
)
"""

def _libpjrt_metal_repo_impl(rctx):
    # Symlink the locally-staged dylib into the repo under the canonical name
    # libpjrt_metal.dylib (matches the dylib's install_name and what metal.zig
    # appends to the sandbox path).
    rctx.symlink(rctx.attr.dylib_path, "libpjrt_metal.dylib")
    rctx.file("BUILD.bazel", _BUILD_FILE_CONTENT)

libpjrt_metal_repository = repository_rule(
    implementation = _libpjrt_metal_repo_impl,
    attrs = {
        "dylib_path": attr.string(
            mandatory = True,
            doc = "Absolute path to the locally-staged libpjrt_metal.dylib.",
        ),
    },
    local = True,
    doc = "Imports a locally-built Metal PJRT plugin dylib by absolute path.",
)

def _metal_impl(mctx):
    libpjrt_metal_repository(
        name = "libpjrt_metal",
        dylib_path = _STAGED_DYLIB_PATH,
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
