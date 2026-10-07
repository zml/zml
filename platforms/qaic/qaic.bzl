"""libzml_qaic: the Qualcomm Cloud AI 100 PJRT plugin (the zml/xla fork's //xla/pjrt/c:pjrt_c_api_qaic_graph_plugin).

No plugin is published yet, so the only mode is the local one:

    bazel build ... --@zml//platforms:qaic=true --repo_env=ZML_QAIC_PLUGIN_DIR=<dir>

`<dir>` is a directory written by the fork's `xla/backends/qaic/tools/publish_local.sh`:
`lib/libzml_qaic.so` and `lib/libzml_qaic.so.sha256`. The repository copies the plugin (never a
symlink, so a rebuild of the fork cannot change a running binary) and verifies it against the
`.sha256`, which it watches: publishing a new plugin re-fetches the repository.
"""

_PLUGIN_ENV = "ZML_QAIC_PLUGIN_DIR"

def _libzml_qaic_impl(rctx):
    plugin_dir = rctx.getenv(_PLUGIN_ENV)
    if not plugin_dir:
        fail("no published libzml_qaic; set {}".format(_PLUGIN_ENV))

    root = rctx.path(plugin_dir)
    so = root.get_child("lib").get_child("libzml_qaic.so")
    sha_file = root.get_child("lib").get_child("libzml_qaic.so.sha256")
    rctx.watch(sha_file)
    if not sha_file.exists:
        fail("{}={}: {} is missing (publish the plugin with publish_local.sh)".format(_PLUGIN_ENV, plugin_dir, sha_file))
    if not so.exists:
        fail("{}={}: {} is missing (publish the plugin with publish_local.sh)".format(_PLUGIN_ENV, plugin_dir, so))

    # `sha256sum` format: "<hex>  <name>"; a bare hex digest is accepted too.
    sha256 = rctx.read(sha_file).strip().split(" ")[0]
    if len(sha256) != 64:
        fail("{}: not a sha256 digest: {}".format(sha_file, sha256))

    # A file:// download is a verified copy: a stale or half-written .so fails here.
    rctx.download(
        url = "file://{}".format(so.realpath),
        output = "lib/libzml_qaic.so",
        sha256 = sha256,
    )
    rctx.file("lib/libzml_qaic.so.sha256", "{}  libzml_qaic.so\n".format(sha256), executable = False)
    rctx.template("BUILD.bazel", rctx.attr.build_file)

_libzml_qaic = repository_rule(
    implementation = _libzml_qaic_impl,
    attrs = {
        "build_file": attr.label(allow_single_file = True),
    },
)

def _qaic_impl(mctx):
    # No published plugin yet: the repository fails, lazily, when ZML_QAIC_PLUGIN_DIR is unset.
    _libzml_qaic(
        name = "libzml_qaic",
        build_file = "//platforms/qaic:libzml_qaic.BUILD.bazel",
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_qaic"],
        root_module_direct_dev_deps = [],
    )

qaic_packages = module_extension(
    implementation = _qaic_impl,
)
