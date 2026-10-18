"""Bazel-owned editor metadata. No application compilation is requested."""

load("@rules_cc//cc:find_cc_toolchain.bzl", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_zig//zig/private:cc_helper.bzl", "need_translate_c")
load("@rules_zig//zig/private/common:escape_label.bzl", "escape_label")
load("@rules_zig//zig/private/common:exec_groups.bzl", "ZIG_EXEC_GROUPS", "translate_c_exec_group_toolchain", "zig_exec_group_toolchain")
load("@rules_zig//zig/private/common:translate_c.bzl", "zig_translate_c")
load("@rules_zig//zig/private/common:zig_cache.bzl", "zig_cache_output")
load("@rules_zig//zig/private/common:zig_lib_dir.bzl", "zig_lib_dir")
load("@rules_zig//zig/private/providers:zig_module_info.bzl", "ZigModuleInfo", "zig_module_info")
load("@rules_zig//zig/private/providers:zig_settings_info.bzl", "ZigSettingsInfo")
load("@rules_zig//zig/private/providers:zig_target_info.bzl", "zig_target_platform")
load("@rules_zig//zig/private/providers:zig_toolchain_info.bzl", "zig_toolchain_executable_path", "zig_toolchain_lib_dir_path")

_Compilation = provider(fields = ["module", "kind", "label"])
_KINDS = {"zig_binary": "exe", "zig_test": "test", "zig_static_library": "static", "zig_shared_library": "shared"}

def _compilation_impl(target, ctx):
    kind = _KINDS.get(ctx.rule.kind)
    if not kind:
        fail("Editor roots must be real Zig compilations: %s (%s)" % (target.label, ctx.rule.kind))
    zdeps = [d[ZigModuleInfo] for d in ctx.rule.attr.deps if ZigModuleInfo in d]
    if not ctx.rule.file.main:
        module = zdeps[0]
    else:
        # Binary rules do not expose their root module provider. Reconstruct only
        # that descriptor, using the same provider factory as rules_zig.
        name = escape_label(label = target.label)
        builtin = ctx.actions.declare_file(target.label.name + ".editor_builtin.zig")
        ctx.actions.write(builtin, "pub const current_repository: []const u8 = %s;\npub const current_package: []const u8 = %s;\npub const current_target: []const u8 = %s;\n" % (
            json.encode(target.label.repo_name),
            json.encode(target.label.package),
            json.encode(target.label.name),
        ))
        builtin_module = zig_module_info(name = "bazel_builtin", canonical_name = "editor_builtin_" + name, main = builtin)
        module = zig_module_info(
            name = target.label.name,
            canonical_name = name,
            main = ctx.rule.file.main,
            srcs = ctx.rule.files.srcs,
            extra_srcs = ctx.rule.files.extra_srcs,
            deps = zdeps + [builtin_module],
            cdeps = [d[CcInfo] for d in ctx.rule.attr.deps if ZigModuleInfo not in d and CcInfo in d],
            zigopts = ctx.rule.attr.zigopts,
            import_names = {d[ZigModuleInfo].canonical_name: alias for d, alias in ctx.rule.attr.import_names.items()},
        )
    return [_Compilation(module = module, kind = kind, label = str(target.label))]

_compilation = aspect(implementation = _compilation_impl)

def _metadata_impl(ctx):
    zig = zig_exec_group_toolchain(ctx)
    target = ctx.toolchains["@rules_zig//zig/target:toolchain_type"].zigtargetinfo
    settings = ctx.attr._settings[ZigSettingsInfo]
    roots = [dep[_Compilation] for dep in ctx.attr.roots]
    providers = [root.module for root in roots] + [dep[ZigModuleInfo] for dep in ctx.attr.deps]
    inputs = [module.transitive_inputs for module in providers]
    contexts = [module.module_context for module in providers]
    transitive = [module.transitive_module_contexts for module in providers]
    cc = cc_common.merge_cc_infos(direct_cc_infos = [m.cc_info for m in providers if m.cc_info])
    if need_translate_c(cc):
        args = ctx.actions.args()
        args.use_param_file("@%s")
        zig_target_platform(target = target, args = args)
        zig_lib_dir(zigtoolchaininfo = zig, args = args)
        zig_cache_output(zigtoolchaininfo = zig, args = args)
        translate = translate_c_exec_group_toolchain(ctx)
        c = zig_translate_c(
            ctx = ctx,
            name = "c",
            canonical_name = "c",
            zigtoolchaininfo = zig,
            global_args = args,
            cc_infos = [cc],
            translatectoolchaininfo = translate.translatectoolchaininfo if translate else None,
        )
        contexts.append(c.module_context)
        transitive.append(c.transitive_module_contexts)
        inputs.append(c.transitive_inputs)
    modules = {}
    for module in depset(contexts, transitive = transitive).to_list():
        imports = {}
        for edge in module.dependency_mappings:
            if edge.name in imports and imports[edge.name] != edge.canonical_name:
                fail("Conflicting import alias %s in %s" % (edge.name, module.main))
            imports[edge.name] = edge.canonical_name
        description = struct(
            id = module.canonical_name,
            source = module.main,
            imports = imports,
            zigopts = module.zigopts,
        )
        previous = modules.get(module.canonical_name)
        if previous and previous != description:
            fail("Conflicting configured module identity: %s" % module.canonical_name)
        modules[module.canonical_name] = description
    for module in modules.values():
        for dep in module.imports.values():
            if dep not in modules:
                fail("Dangling module edge: %s -> %s" % (module.id, dep))
    sources = depset(transitive = inputs)
    output = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(output, json.encode({
        "schema_version": 1,
        "target": target.target,
        "target_args": target.args,
        "settings": settings,
        "configuration": ctx.bin_dir.path,
        "modules": [modules[key] for key in sorted(modules)],
        "compilations": [struct(label = root.label, kind = root.kind, root = root.module.canonical_name) for root in roots],
        "inputs": sorted([file.path for file in sources.to_list()]),
        "tools": {
            "zig": zig_toolchain_executable_path(zig),
            "zig_lib": zig_toolchain_lib_dir_path(zig),
            "zig_version": zig.zig_version,
            "zls": ctx.executable._zls.path,
        },
    }))
    tools = [ctx.executable._zls]
    if zig.zig_exe.file:
        tools.extend([zig.zig_exe.file, zig.zig_lib.file])

    # Top-level outputs force local materialization even with remote_download_toplevel.
    return [DefaultInfo(files = depset([output] + tools, transitive = [sources])), OutputGroupInfo(manifest = depset([output]))]

zls_metadata = rule(
    implementation = _metadata_impl,
    attrs = {
        "roots": attr.label_list(mandatory = True, aspects = [_compilation]),
        "deps": attr.label_list(providers = [ZigModuleInfo]),
        "_settings": attr.label(default = "@rules_zig//zig/settings"),
        "_zls": attr.label(default = "@zls//:zls", executable = True, cfg = "exec"),
    },
    toolchains = ["@rules_zig//zig/target:toolchain_type"] + use_cc_toolchain(mandatory = False),
    exec_groups = ZIG_EXEC_GROUPS,
    fragments = ["cpp", "apple"],
)
