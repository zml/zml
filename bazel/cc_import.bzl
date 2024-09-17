load("@bazel_tools//tools/build_defs/cc:cc_import.bzl", _cc_import = "cc_import")
load(":patchelf.bzl", "patchelf")

def _cc_import_runfiles_impl(ctx):
    runfiles = ctx.runfiles(files = ctx.files.data)
    transitive_runfiles_list = []
    if ctx.attr.static_library:
        transitive_runfiles_list.append(ctx.attr.static_library[DefaultInfo].default_runfiles)
    if ctx.attr.pic_static_library:
        transitive_runfiles_list.append(ctx.attr.pic_static_library[DefaultInfo].default_runfiles)
    if ctx.attr.shared_library:
        transitive_runfiles_list.append(ctx.attr.shared_library[DefaultInfo].default_runfiles)
    if ctx.attr.interface_library:
        transitive_runfiles_list.append(ctx.attr.interface_library[DefaultInfo].default_runfiles)
    for dep in ctx.attr.deps:
        transitive_runfiles_list.append(dep[DefaultInfo].default_runfiles)

    for maybe_runfiles in transitive_runfiles_list:
        if maybe_runfiles:
            runfiles = runfiles.merge(maybe_runfiles)

    default_info = DefaultInfo(runfiles = runfiles)
    return [ctx.attr.src[CcInfo], default_info]

_cc_import_runfiles = rule(
    implementation = _cc_import_runfiles_impl,
    attrs = {
        "src": attr.label(providers = [CcInfo]),
        "static_library": attr.label(allow_single_file = [".a", ".lib"]),
        "pic_static_library": attr.label(allow_single_file = [".pic.a", ".pic.lib"]),
        "shared_library": attr.label(allow_single_file = True),
        "interface_library": attr.label(allow_single_file = [".ifso", ".tbd", ".lib", ".so", ".dylib"]),
        "data": attr.label_list(allow_files = True),
        "deps": attr.label_list(),
    },
)

def cc_import(
    name,
    static_library = None,
    pic_static_library = None,
    shared_library = None,
    interface_library = None,
    data = None,
    deps = None,
    visibility = None,
    soname = None,
    add_needed = None,
    remove_needed = None,
    replace_needed = None,
    **kwargs):
    if shared_library and (soname or add_needed or remove_needed or replace_needed):
        patched_name = "{}_patchelf".format(name)
        patchelf(
            name = patched_name,
            shared_library = shared_library,
            soname = soname,
            add_needed = add_needed,
            remove_needed = remove_needed,
            replace_needed = replace_needed,
        )
        shared_library = ":" + patched_name
    if data:
        _cc_import(
            name = name + "_no_runfiles",
            static_library = static_library,
            pic_static_library = pic_static_library,
            shared_library = shared_library,
            interface_library = interface_library,
            data = data,
            deps = deps,
            **kwargs
        )
        _cc_import_runfiles(
            name = name,
            src = ":{}_no_runfiles".format(name),
            static_library = static_library,
            pic_static_library = pic_static_library,
            shared_library = shared_library,
            interface_library = interface_library,
            data = data,
            deps = deps,
            visibility = visibility,
        )
    else:
        _cc_import(
            name = name,
            static_library = static_library,
            pic_static_library = pic_static_library,
            shared_library = shared_library,
            interface_library = interface_library,
            data = data,
            deps = deps,
            visibility = visibility,
            **kwargs
        )
