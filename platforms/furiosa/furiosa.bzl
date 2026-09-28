"""Local PJRT plugin repository, supplied using --override_repository."""

def _placeholder_impl(ctx):
    ctx.file("BUILD.bazel", '''
load(":missing.bzl", "missing_plugin")
missing_plugin(name = "libzml_furiosa", visibility = ["//visibility:public"])
''')
    ctx.file("missing.bzl", '''
def _impl(ctx):
    fail("Furiosa requires a built PJRT plugin: use --override_repository=libzml_furiosa=/path/to/xla-override (see platforms/furiosa/README.md)")
missing_plugin = rule(implementation = _impl)
''')

_placeholder = repository_rule(implementation = _placeholder_impl)

def _furiosa_impl(ctx):
    _placeholder(name = "libzml_furiosa")
    return ctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libzml_furiosa"],
        root_module_direct_dev_deps = [],
    )

furiosa_packages = module_extension(implementation = _furiosa_impl)
