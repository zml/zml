load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

_ALL_GFX = ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx1010", "gfx1012", "gfx1030", "gfx1100", "gfx1101", "gfx1102"]

def _compute_enabled_gfx(values):
    ret = {}
    for v in values:
        if (v == "all"):
            ret = {gfx: True for gfx in _ALL_GFX}
        elif (v == "none"):
            ret = {}
        else:
            ret[v] = True
    return ret

def _gfx_from_file(file):
    return file.basename[:-len(file.extension) - 1].rpartition("_")[-1].partition("-")[0]

def _is_file_enabled(file, enabled_gfx):
    gfx = _gfx_from_file(file)
    return gfx in enabled_gfx or gfx == "fallback"

def _bytecode_select_impl(ctx):
    enabled_gfx = _compute_enabled_gfx(ctx.attr.enabled_gfx[BuildSettingInfo].value)
    return [
        DefaultInfo(
            files = depset([
                file
                for file in ctx.files.bytecodes
                if _is_file_enabled(file, enabled_gfx)
            ]),
        ),
    ]

bytecode_select = rule(
    implementation = _bytecode_select_impl,
    attrs = {
        "bytecodes": attr.label_list(allow_files = True),
        "enabled_gfx": attr.label(mandatory = True),
    },
)


def if_gfx(gfx, value):
    return select({
        "@zml//runtimes/rocm:_{}".format(gfx): value,
        "//conditions:default": [],
    })

