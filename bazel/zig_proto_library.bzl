"""Starlark implementation of zig_proto_library"""

load("@rules_proto//proto:defs.bzl", "proto_common")
load(
    "@rules_zig//zig/private/providers:zig_module_info.bzl",
    "ZigModuleInfo",
    "zig_module_info",
)

def _zig_proto_library_impl(ctx):
    if len(ctx.attr.deps) != 1:
        fail("zig_proto_library '{}' requires exactly one 'deps'".format(ctx.label.name))

    (dep,) = ctx.attr.deps

    # the aspect already generated for us the Zig module
    # we just change the import name to make it match what the user chose.
    module = dep[ZigModuleInfo]
    import_name = ctx.attr.import_name or ctx.label.name

    if import_name:
        keys = ["canonical_name", "cdeps", "copts", "deps", "extra_srcs", "linkopts", "main", "srcs"]
        args = {k: getattr(module, k) for k in keys}
        args["name"] = import_name
        module = zig_module_info(**args)
    return [module]

def zig_proto_library_aspect_impl(target, ctx):
    """
    For each `.proto` in the given target dependencies,
    generate a `.pb.zig` file, and a `zig_module` to import it.
    """
    toolchain = ctx.attr._zig_proto_toolchain[proto_common.ProtoLangToolchainInfo]
    proto_info = target[ProtoInfo]

    # assert len(proto_info.direct_sources) == 1, "Can only compile .proto files one by one"
    (proto_src,) = proto_info.direct_sources
    pb_zig_name = proto_src.basename[:-len(".proto")] + ".pb.zig"
    zig_src = ctx.actions.declare_file(pb_zig_name, sibling = proto_src)

    ctx.actions.run
    proto_common.compile(
        ctx.actions,
        proto_info = proto_info,
        proto_lang_toolchain_info = toolchain,
        generated_files = [zig_src],
    )

    zig_proto_modules = [p[ZigModuleInfo] for p in ctx.rule.attr.deps]
    import_name = get_import_name(target, proto_src)

    module = zig_module_info(
        name = import_name,
        canonical_name = str(target.label),
        main = zig_src,
        srcs = [],
        extra_srcs = [],
        copts = [],
        linkopts = [],
        deps = [toolchain.runtime[ZigModuleInfo]] + zig_proto_modules,
        cdeps = [],
    )
    return [module]

def get_import_name(target, proto_src):
    """
    When the Zig protoc plugin is generating .pb.zig files,
    it generates import names based on the path received from protoc.
    We need to create Zig modules with the same name.
    """
    name = str(target.label)

    # special handling of builtin types
    if "com_google_protobuf//:" in name:
        name = "google_protobuf_" + proto_src.basename
    else:
        name = name.rsplit("//")[-1]
    name = name.rsplit("//")[-1]
    return name.replace(".", "_").replace(":", "_").replace("/", "_")

zig_proto_library_aspect = aspect(
    attrs = {
        "_zig_proto_toolchain": attr.label(
            default = "@zig-protobuf//:zig_toolchain",
            providers = [proto_common.ProtoLangToolchainInfo],
        ),
    },
    implementation = zig_proto_library_aspect_impl,
    provides = [ZigModuleInfo],
    attr_aspects = ["deps"],
)

zig_proto_library = rule(
    doc = """
    Converts a single `proto_library` into a zig module.
    """,
    implementation = _zig_proto_library_impl,
    attrs = {
        "deps": attr.label_list(
            aspects = [zig_proto_library_aspect],
            providers = [ProtoInfo],
        ),
        "import_name": attr.string(
            doc = "The import name of the Zig module.",
            default = "",
        ),
    },
    provides = [ZigModuleInfo],
)
