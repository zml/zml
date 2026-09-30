"""Build settings for Zig options."""

load("@bazel_skylib//lib:types.bzl", "types")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_zig//zig:defs.bzl", "zig_library")

ZigOption = provider(
    doc = "A Zig option's configured value and allowed string values.",
    fields = {
        "name": "The flag's target name, or this option's target name for explicit values.",
        "module_name": "The Bazel module declaring the flag, or empty when unavailable.",
        "value": "The configured build setting or explicit value.",
        "type": "The Zig integer type, or None for other flag types.",
        "allowed_values": "Allowed values for string flags; an empty list allows any string. None for other flag types.",
        "enum_name": "Optional exported enum type name for a constrained string flag.",
        "nullable": "Whether an unconstrained string emits null for an empty value.",
    },
)

_FlagInfo = provider(fields = ["allowed_values"])

def _flag_info_impl(_target, ctx):
    return [_FlagInfo(allowed_values = getattr(ctx.rule.attr, "values", []))]

_flag_info = aspect(implementation = _flag_info_impl)

def _zig_option_impl(ctx):
    flag = ctx.attr.flag
    if ctx.attr.value_source == "flag":
        if flag == None:
            fail("flag must resolve to a build setting")
        value = flag[BuildSettingInfo].value
    else:
        value = getattr(ctx.attr, ctx.attr.value_source)
    if not (types.is_bool(value) or types.is_int(value) or types.is_string(value)):
        fail("Zig options require an integer, boolean, or string value")
    allowed_values = None
    if types.is_string(value):
        allowed_values = flag[_FlagInfo].allowed_values if flag else []
    if ctx.attr.type and not types.is_int(value):
        fail("type is only supported for integer options")
    if ctx.attr.nullable and not types.is_string(value):
        fail("nullable is only supported for string options")
    if ctx.attr.nullable and allowed_values:
        fail("nullable strings cannot have enum values")
    if ctx.attr.enum_name and not allowed_values:
        fail("enum_name requires a string flag with nonempty values")
    return [ZigOption(
        name = flag.label.name if flag else ctx.label.name,
        module_name = "",
        value = value,
        type = (ctx.attr.type or "usize") if types.is_int(value) else None,
        allowed_values = allowed_values,
        enum_name = ctx.attr.enum_name,
        nullable = ctx.attr.nullable,
    )]

def _zig_string(value):
    # JSON and Zig differ in their control-character and Unicode escapes.
    escaped = []
    for char in value.elems():
        part = json.encode(char)[1:-1]
        if part.startswith("\\u"):
            part = "\\u{" + part[2:] + "}"
        escaped.append(part.replace("\\b", "\\x08").replace("\\f", "\\x0c"))
    return '"' + "".join(escaped) + '"'

def _zig_options_impl(ctx):
    lines = []
    names = {}
    for setting in ctx.attr.settings:
        option = setting[ZigOption]
        name = option.name
        if name in names:
            fail("Duplicate Zig declaration: {}".format(name))
        names[name] = True
        value = option.value
        if types.is_bool(value):
            zig_type = "bool"
            literal = "true" if value else "false"
        elif types.is_int(value):
            zig_type = option.type
            literal = str(value)
        elif option.allowed_values:
            zig_type = "enum { " + ", ".join(["@" + _zig_string(v) for v in option.allowed_values]) + " }"
            if option.enum_name:
                if option.enum_name in names:
                    fail("Duplicate Zig declaration: {}".format(option.enum_name))
                names[option.enum_name] = True
                lines.append("pub const @{} = {};".format(_zig_string(option.enum_name), zig_type))
                zig_type = "@" + _zig_string(option.enum_name)
            literal = ".@" + _zig_string(value)
        else:
            zig_type = "?[]const u8" if option.nullable else "[]const u8"
            literal = "null" if option.nullable and not value else _zig_string(value)
        lines.append("pub const @{}: {} = {};".format(_zig_string(name), zig_type, literal))
    ctx.actions.write(ctx.outputs.out, "\n".join(lines) + "\n")
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

_zig_options_source = rule(
    implementation = _zig_options_impl,
    attrs = {
        "out": attr.output(mandatory = True),
        "settings": attr.label_list(
            mandatory = True,
            allow_files = False,
            providers = [ZigOption],
        ),
    },
)

def zig_options(name, settings, **kwargs):
    """Generate an options module and expose it as a zig_library.

    Args:
        name: Library target name; the generated source is <name>.zig.
        settings: zig_option targets; flag names or explicit option names become quoted Zig constants.
        **kwargs: Attributes forwarded to zig_library, except main which is generated.
    """
    if "main" in kwargs:
        fail("zig_options generates its own main source")
    source_attrs = {key: kwargs[key] for key in ["testonly", "tags", "compatible_with", "restricted_to", "target_compatible_with"] if key in kwargs}
    _zig_options_source(
        name = name + "_source",
        settings = settings,
        out = name + ".zig",
        visibility = ["//visibility:private"],
        **source_attrs
    )
    zig_library(name = name, main = ":" + name + "_source", **kwargs)

_zig_option = rule(
    implementation = _zig_option_impl,
    attrs = {
        "flag": attr.label(
            providers = [BuildSettingInfo],
            aspects = [_flag_info],
            doc = "Boolean, integer, or string build setting. An aspect discovers allowed string values.",
        ),
        "bool": attr.bool(),
        "string": attr.string(),
        "int": attr.int(),
        "value_source": attr.string(mandatory = True, values = ["flag", "bool", "string", "int"]),
        "type": attr.string(doc = "Zig integer type; defaults to usize. Only valid for integer options."),
        "nullable": attr.bool(doc = "Emit an optional string; an empty value becomes null. Cannot be combined with values."),
        "enum_name": attr.string(doc = "Optional exported Zig enum type name; requires nonempty values."),
    },
    doc = "Expose a build setting or explicit value as a Zig option, named after the flag or this target respectively.",
)

def zig_option(name, flag = None, bool = None, string = None, int = None, **kwargs):
    """Expose exactly one build setting or explicit value as a Zig option.

    Args:
        name: Target name; also the generated Zig constant name for explicit values.
        flag: Boolean, integer, or string build setting to read.
        bool: Explicit boolean value, which may use select().
        string: Explicit string value, which may use select().
        int: Explicit integer value, which may use select().
        **kwargs: Attributes forwarded to the rule, including type, nullable, and enum_name.
    """
    sources = {key: value for key, value in {
        "flag": flag,
        "bool": bool,
        "string": string,
        "int": int,
    }.items() if value != None}
    if len(sources) != 1:
        fail("zig_option requires exactly one of flag, bool, string, or int")
    _zig_option(name = name, value_source = sources.keys()[0], **dict(sources, **kwargs))
