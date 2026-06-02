"""Metal PJRT plugin module extension.

Unlike the other platforms (cpu/oneapi/...), the Metal plugin is NOT fetched from
a remote tarball -- it is built locally from the raphael/metal XLA fork. This
extension exposes the locally-staged dylib as a Bazel repo (@libpjrt_metal)
containing a single filegroup ":libpjrt_metal" -> libpjrt_metal.dylib.

Design default: the staged dylib path is hardcoded (plan option (a), "local,
fastest for dev"). The build script stages it to
platforms/metal/prebuilt/libpjrt_metal.dylib (the install_name basename). To
update the plugin, rebuild it in the XLA fork and re-copy it to that path; no
sha256/URL bump is needed. A future reproducible variant would publish a tarball
and http_archive it (plan option (b)); deferred until the local path works.
"""

# Absolute path to the locally-staged Metal PJRT plugin dylib.
#
# This is the install_name-matching copy made from the XLA fork build output
# (xla/bazel-bin/xla/pjrt/c/libpjrt_c_api_gpu_plugin.dylib). We point at the
# stable staged copy under the ZML repo rather than directly at bazel-bin, which
# is an unstable symlink tree. The path is hardcoded for dev convenience; the
# repo rule below symlinks it (no copy) and renames it to libpjrt_metal.dylib.
_STAGED_DYLIB_PATH = "/Users/raph/Documents/Git-Repos/zml_raphael-metal/platforms/metal/prebuilt/libpjrt_metal.dylib"

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
