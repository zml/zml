"""Generates the ZLS build config file."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_zig//zig/private:cc_helper.bzl", "need_translate_c")
load("@rules_zig//zig/private/common:translate_c.bzl", "zig_translate_c")
load("@rules_zig//zig/private/common:zig_cache.bzl", "zig_cache_output")
load("@rules_zig//zig/private/common:zig_lib_dir.bzl", "zig_lib_dir")
load("@rules_zig//zig/private/providers:zig_module_info.bzl", "ZigModuleInfo")
load(
    "@rules_zig//zig/private/providers:zig_target_info.bzl",
    "zig_target_platform",
)

def _add_context(context):
    return json.encode(struct(name = context.name, path = "@@__BUILD_WORKSPACE_DIRECTORY__@@/" + context.main))

def _zls_write_build_config_impl(ctx):
    zigtoolchaininfo = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    zigtargetinfo = ctx.toolchains["@rules_zig//zig/target:toolchain_type"].zigtargetinfo

    c_module_contexts = []
    c_module_inputs = []
    cc_info = cc_common.merge_cc_infos(cc_infos = [dep[ZigModuleInfo].cc_info for dep in ctx.attr.deps if dep[ZigModuleInfo].cc_info])
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
            zigtoolchaininfo = zigtoolchaininfo,
            global_args = global_args,
            cc_infos = [cc_info],
        )

        c_module_contexts = [depset(direct = [c_module.module_context])]
        c_module_inputs = [c_module.transitive_inputs]

    contexts = depset(
        direct = [dep[ZigModuleInfo].module_context for dep in ctx.attr.deps],
        transitive = [dep[ZigModuleInfo].transitive_module_contexts for dep in ctx.attr.deps] + c_module_contexts,
    )

    template_dict = ctx.actions.template_dict()
    template_dict.add_joined("@@PACKAGES@@", contexts, map_each = _add_context, join_with = ",\n        ")

    config = ctx.outputs.out
    ctx.actions.expand_template(
        output = config,
        template = ctx.file._zls_completion_tpl,
        computed_substitutions = template_dict,
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
        ),
        "out": attr.output(
            mandatory = True,
        ),
        "_zls_completion_tpl": attr.label(
            default = ":zls.completion.tpl",
            allow_single_file = True,
        ),
    },
    toolchains = [
        "@rules_zig//zig:toolchain_type",
        "@rules_zig//zig/target:toolchain_type",
    ],
    fragments = ["cpp"],
)
