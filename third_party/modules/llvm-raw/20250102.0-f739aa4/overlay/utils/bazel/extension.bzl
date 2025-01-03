load("//utils/bazel:configure.bzl", _llvm_configure = "llvm_configure")

def _llvm_impl(mctx):
    _targets = {}
    for mod in mctx.modules:
        for conf in mod.tags.configure:
            for target in conf.targets:
                _targets[target] = True
    _llvm_configure(
        name = "llvm-project",
        targets = _targets.keys(),
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

llvm = module_extension(
    implementation = _llvm_impl,
    tag_classes = {
        "configure": tag_class(
            attrs = {
                "targets": attr.string_list(mandatory = True),
            },
        ),
    },
)
