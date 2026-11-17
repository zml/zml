"""Generate a Zig module with upstream zig-protobuf's package-based layout."""

load("@com_google_protobuf//bazel/common:proto_common.bzl", "proto_common")
load("@com_google_protobuf//bazel/common:proto_info.bzl", "ProtoInfo")
load("@rules_zig//zig/private/providers:zig_module_info.bzl", "ZigModuleInfo", "zig_module_info")

def _zig_proto_library_impl(ctx):
    if len(ctx.attr.deps) != 1:
        fail("zig_proto_library requires exactly one proto_library dependency")

    # Upstream merges declarations by protobuf package and emits all imported
    # packages too. Generate the entire closure together so relative imports and
    # declarations shared by multiple .proto files resolve within one Zig module.
    generated = ctx.actions.declare_directory(ctx.label.name + "_generated")
    toolchain = ctx.attr._zig_proto_toolchain[proto_common.ProtoLangToolchainInfo]
    proto_common.compile(
        ctx.actions,
        proto_info = ctx.attr.deps[0][ProtoInfo],
        proto_lang_toolchain_info = toolchain,
        generated_files = [generated],
        plugin_output = generated.path,
    )

    # A root next to the generated tree keeps every package inside Zig's module
    # path. Re-exporting through a named declaration avoids copying generated code.
    root = ctx.actions.declare_file(ctx.label.name + ".zig")
    ctx.actions.write(root, 'pub const proto = @import("{}/{}.pb.zig");\n'.format(
        generated.basename,
        ctx.attr.proto_package.replace(".", "/"),
    ))
    return [
        DefaultInfo(files = depset([root, generated])),
        zig_module_info(
            name = ctx.attr.import_name or ctx.label.name,
            canonical_name = str(ctx.label),
            main = root,
            srcs = [generated],
            deps = [toolchain.runtime[ZigModuleInfo]],
        ),
    ]

zig_proto_library = rule(
    doc = "Generates a Zig module exposing the selected protobuf package as `proto`.",
    implementation = _zig_proto_library_impl,
    attrs = {
        "deps": attr.label_list(providers = [ProtoInfo]),
        "import_name": attr.string(),
        "proto_package": attr.string(mandatory = True, doc = "Protobuf package to expose, e.g. xla."),
        "_zig_proto_toolchain": attr.label(
            default = "@zig-protobuf//:zig_toolchain",
            providers = [proto_common.ProtoLangToolchainInfo],
        ),
    },
    provides = [ZigModuleInfo],
)
