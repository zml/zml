"""Generates the ZLS build config file."""

load("@apple_support//lib:apple_support.bzl", "apple_support")
load("@rules_cc//cc:find_cc_toolchain.bzl", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_zig//zig/private:cc_helper.bzl", "need_translate_c")
load("@rules_zig//zig/private/common:translate_c.bzl", "zig_translate_c")
load("@rules_zig//zig/private/common:zig_cache.bzl", "zig_cache_output")
load("@rules_zig//zig/private/common:zig_lib_dir.bzl", "zig_lib_dir")
load("@rules_zig//zig/private/providers:zig_module_info.bzl", "ZigModuleInfo", "zig_module_info")
load(
    "@rules_zig//zig/private/providers:zig_target_info.bzl",
    "zig_target_platform",
)

def _zls_construct_zig_module_info_impl(target, ctx):
    """Aspect that constructs ZigModuleInfo for zig_binary and zig_library rules."""
    if ZigModuleInfo in target:
        return []

    if ctx.rule.kind not in ("zig_binary", "zig_static_library", "zig_shared_library"):
        return []

    cdeps = []
    zdeps = []
    for dep in ctx.rule.attr.deps:
        if ZigModuleInfo in dep:
            zdeps.append(dep[ZigModuleInfo])
        elif CcInfo in dep:
            cdeps.append(dep[CcInfo])

    root_module_is_only_dep = len(ctx.rule.attr.deps) == 1 and ZigModuleInfo in ctx.rule.attr.deps[0]
    if root_module_is_only_dep:
        root_module = ctx.rule.attr.deps[0][ZigModuleInfo]
    else:
        root_module = zig_module_info(
            name = ctx.rule.attr.name,
            canonical_name = target.label.name,
            main = ctx.rule.file.main,
            srcs = ctx.rule.files.srcs,
            extra_srcs = ctx.rule.files.extra_srcs,
            deps = zdeps,  # [bazel_builtin_module(ctx)],
            cdeps = cdeps,
            zigopts = [],
        )

    return [
        root_module,
    ]

zls_construct_zig_module_info = aspect(
    implementation = _zls_construct_zig_module_info_impl,
)

def format_main_file(main):
    prefix = "@@__BUILD_WORKSPACE_DIRECTORY__@@/"
    if (main.startswith("bazel-out/") or main.startswith("external/")):
        prefix = "@@__BAZEL_EXECUTION_ROOT__@@/"
    return prefix + main

def _zls_write_build_config_impl(ctx):
    zigtoolchaininfo = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    zigtargetinfo = ctx.toolchains["@rules_zig//zig/target:toolchain_type"].zigtargetinfo

    c_module_contexts = []
    c_module_inputs = []
    cc_info = cc_common.merge_cc_infos(direct_cc_infos = [dep[ZigModuleInfo].cc_info for dep in ctx.attr.deps if dep[ZigModuleInfo].cc_info])
    if need_translate_c(cc_info):
        global_args = ctx.actions.args()
        global_args.use_param_file("@%s")

        zig_target_platform(
            target = zigtargetinfo,
            args = global_args,
        )

        zig_lib_dir(
            zigtoolchaininfo = zigtoolchaininfo,
            args = global_args,
        )

        zig_cache_output(
            zigtoolchaininfo = zigtoolchaininfo,
            args = global_args,
        )

        c_module = zig_translate_c(
            ctx = ctx,
            name = "c",
            canonical_name = "c",
            zigtoolchaininfo = zigtoolchaininfo,
            global_args = global_args,
            cc_infos = [cc_info],
        )

        c_module_contexts = [depset(
            direct = [c_module.module_context],
            transitive = [c_module.transitive_module_contexts],
        )]
        c_module_inputs = [c_module.transitive_inputs]

    contexts = depset(
        direct = [dep[ZigModuleInfo].module_context for dep in ctx.attr.deps],
        transitive = [dep[ZigModuleInfo].transitive_module_contexts for dep in ctx.attr.deps] + c_module_contexts,
    )

    modules = {
        mod.canonical_name: mod
        for mod in contexts.to_list()
    }

    output = {
        "dependencies": {},
        "modules": {
            format_main_file(mod.main): {
                "import_table": {
                    dep.name: format_main_file(modules.get(dep.canonical_name, "").main)
                    for dep in mod.dependency_mappings
                },
                "c_macros": [],
                "include_dirs": [],
            }
            for mod in modules.values()
        },
        "compilations": [],
        "top_level_steps": [],
        "available_options": {},
    }

    config = ctx.outputs.out
    ctx.actions.write(
        output = config,
        content = json.encode(output),
    )

    return [
        DefaultInfo(
            files = depset(direct = [config]),
            # Not really runfiles but we need to pull the output of translate_c...
            runfiles = ctx.runfiles(transitive_files = depset(transitive = c_module_inputs)),
        ),
    ]

zls_write_build_config = rule(
    implementation = _zls_write_build_config_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [ZigModuleInfo],
            mandatory = True,
            aspects = [zls_construct_zig_module_info],
        ),
        "out": attr.output(
            mandatory = True,
        ),
        "_zls_completion_tpl": attr.label(
            default = ":zls.completion.tpl",
            allow_single_file = True,
        ),
        "_translate_c": attr.label(
            default = Label("@rules_zig//zig/private/common:translate-c"),
            cfg = "exec",
            executable = True,
        ),
        "_c_helpers": attr.label(
            default = Label("@rules_zig//zig/private/common:helpers"),
        ),
        "_c_builtins": attr.label(
            default = Label("@rules_zig//zig/private/common:c_builtins"),
        ),
    } | apple_support.action_required_attrs() | apple_support.platform_constraint_attrs(),
    toolchains = [
        "@rules_zig//zig:toolchain_type",
        "@rules_zig//zig/target:toolchain_type",
    ] + use_cc_toolchain(mandatory = False),
    fragments = ["cpp", "apple"],
)
