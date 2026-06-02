"""Metal PJRT plugin module extension.

Unlike the other platforms (cpu/oneapi/...), the Metal plugin is NOT fetched from
a remote tarball -- it is built locally from the raphael/metal XLA fork. This
extension exposes the locally-staged dylib as a Bazel repo (@libpjrt_metal)
containing a single filegroup ":libpjrt_metal" -> libpjrt_metal.dylib.

Design default: the dylib path is hardcoded (plan option (a), "local, fastest for
dev") and points DIRECTLY at the XLA fork's build output. To update the plugin,
just rebuild it in the XLA fork (`bazel build //xla/pjrt/c:pjrt_c_api_gpu_plugin`)
and re-run here — no copy/stage step, no sha256/URL bump. The repo rule symlinks
the path (lazily resolved) and renames it to libpjrt_metal.dylib; Bazel re-hashes
the symlinked source on each build, so XLA rebuilds propagate into the sandbox.
A future reproducible variant would publish a tarball and http_archive it (plan
option (b)); deferred until the local path works.
"""

# Absolute path to the locally-built Metal PJRT plugin dylib.
#
# Points straight at the XLA fork's build output: the `bazel-bin` convenience
# symlink is a STABLE logical path (it absorbs output-base churn), and
# repository_ctx.symlink resolves lazily at use time, so this is a live link to
# the latest plugin build with no copy step. The dylib's install_name is
# @rpath/libpjrt_metal.dylib; the repo rule renames the symlink to that basename.
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
