load("@llvm-raw//utils/bazel:configure.bzl", _llvm_configure = "llvm_configure")

def _llvm_zlib_compat_impl(repository_ctx):
    repository_ctx.file("BUILD.bazel", """
alias(
    name = "zlib-ng",
    actual = "@zlib//:zlib",
    visibility = ["//visibility:public"],
)
""")

_llvm_zlib_compat = repository_rule(
    implementation = _llvm_zlib_compat_impl,
)

def _llvm_zstd_compat_impl(repository_ctx):
    repository_ctx.file("BUILD.bazel", """
alias(
    name = "zstd",
    actual = "@net_zstd//:zstd",
    visibility = ["//visibility:public"],
)
""")

_llvm_zstd_compat = repository_rule(
    implementation = _llvm_zstd_compat_impl,
)

def _llvm_impl(mctx):
    _targets = {}
    for mod in mctx.modules:
        for conf in mod.tags.configure:
            for target in conf.targets:
                _targets[target] = True
    _llvm_zlib_compat(name = "llvm_zlib")
    _llvm_zstd_compat(name = "llvm_zstd")
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
                "targets": attr.string_list(
                    default = [],
                ),
            },
        ),
    },
)
