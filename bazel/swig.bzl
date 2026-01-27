load("@rules_cc//cc:action_names.bzl", "C_COMPILE_ACTION_NAME")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:defs.bzl", "CcInfo", "cc_common")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")

def _swig_cc_library_impl(ctx):
    args = ctx.actions.args()

    if ctx.attr.cpp:
        args.add("-c++")

    args.add("-std=c++17")
    args.add("-c")
    args.add("-O")
    args.add("-module", ctx.attr.module)
    args.add_joined("-features", ctx.attr.enabled_features, join_with = ",")

    if ctx.attr.defines:
        args.add_all(ctx.attr.defines, format_each = "-D%s")

    cc_toolchain = find_cc_toolchain(ctx)
    if (cc_toolchain):
        args.add_all(cc_toolchain.built_in_include_directories, format_each = "-I%s")

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    c_compile_variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = ctx.fragments.cpp.copts + ctx.fragments.cpp.conlyopts,
    )
    cc_compile_command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = C_COMPILE_ACTION_NAME,
        variables = c_compile_variables,
    )
    for arg in cc_compile_command_line:
        if (arg.startswith("-I") or arg.startswith("-D")):
            args.add(arg)

    cc_info = cc_common.merge_cc_infos(direct_cc_infos = [dep[CcInfo] for dep in ctx.attr.deps])
    args.add_all(cc_info.compilation_context.defines, format_each = "-D%s")
    args.add_all(cc_info.compilation_context.local_defines, format_each = "-D%s")
    args.add_all(cc_info.compilation_context.framework_includes, format_each = "-I%s")
    args.add_all(cc_info.compilation_context.includes, format_each = "-I%s")
    args.add_all(cc_info.compilation_context.quote_includes, format_each = "-I%s")
    args.add_all(cc_info.compilation_context.system_includes, format_each = "-I%s")

    output_cpp = ctx.actions.declare_file("%s.cpp" % ctx.attr.module)
    output_h = ctx.actions.declare_file("%s.h" % ctx.attr.module)
    args.add("-outdir", output_h.dirname)

    outputs = [
        output_cpp,
        output_h,
    ]
    args.add("-o", output_cpp)
    args.add("-w-305")
    args.add(ctx.file.interface)

    inputs = depset(ctx.attr.srcs, transitive = [
        ctx.attr.interface.files,
        cc_info.compilation_context.headers,
        ctx.attr._swig_lib.files,
    ])

    ctx.actions.run(
        inputs = inputs,
        outputs = outputs,
        executable = ctx.executable._swig,
        arguments = [args],
        env = {
            "SWIG_LIB": ctx.files._swig_lib[0].dirname,
        },
        mnemonic = "SwigC",
    )

    return [
        DefaultInfo(
            files = depset(outputs),
        ),
        OutputGroupInfo(
            hdrs = depset([output_h]),
            srcs = depset([output_cpp]),
        ),
    ]

_swig_cc_library = rule(
    _swig_cc_library_impl,
    attrs = {
        "interface": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "srcs": attr.label_list(
            allow_files = True,
        ),
        "deps": attr.label_list(
            providers = [CcInfo],
        ),
        "defines": attr.string_list(),
        "enabled_features": attr.string_list(),
        "module": attr.string(
            mandatory = True,
        ),
        "cpp": attr.bool(
            default = True,
        ),
        "intgosize": attr.int(
            default = 64,
        ),
        "_swig": attr.label(
            default = "@org_swig_swig//:swig",
            cfg = "exec",
            executable = True,
        ),
        "_swig_lib": attr.label(
            default = "@org_swig_swig//:lib",
            allow_files = True,
        ),
    },
    toolchains = use_cc_toolchain(),
    fragments = ["cpp"],
)

def swig_cc_library(name, deps = [], **kwargs):
    _swig_cc_library(
        name = "{}.swig".format(name),
        deps = deps,
        **kwargs
    )
    native.filegroup(
        name = "{}.hdrs".format(name),
        srcs = [":{}.swig".format(name)],
        output_group = "hdrs",
    )
    native.filegroup(
        name = "{}.srcs".format(name),
        srcs = [":{}.swig".format(name)],
        output_group = "srcs",
    )
    cc_library(
        name = name,
        hdrs = [":{}.hdrs".format(name)],
        srcs = [":{}.srcs".format(name)],
        deps = deps,
    )
