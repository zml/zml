"""Build settings for Zig options."""

load("@bazel_skylib//lib:types.bzl", "types")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_zig//zig:defs.bzl", "zig_library")

ZigOption = provider(
    doc = "A Zig option's configured value and allowed string values.",
    fields = {
        "module_name": "The Bazel module declaring the flag, or empty when unavailable.",
        "value": "The configured build setting value.",
        "type": "The Zig integer type, or None for other flag types.",
        "allowed_values": "Allowed values for string flags; an empty list allows any string. None for other flag types.",
        "enum_name": "Optional exported enum type name for a constrained string flag.",
        "nullable": "Whether an unconstrained string emits null for an empty value.",
    },
)

_MAKE_VARIABLE_ATTR = attr.string(
    doc = "Optional Make variable name exposed to rules that list this flag in toolchains.",
)

_MODULE_NAME_ATTR = attr.string(
    doc = "Declaring module name, captured by zig_option during package evaluation.",
)

_SCOPE_ATTR = attr.string(
    doc = "The scope indicates where a flag can propagate to",
    default = "target",
)

def _flag_impl(ctx):
    value = ctx.build_setting_value
    providers = [
        BuildSettingInfo(value = value),
        ZigOption(
            module_name = ctx.attr.module_name,
            value = value,
            type = getattr(ctx.attr, "type", None),
            allowed_values = getattr(ctx.attr, "values", None),
            enum_name = getattr(ctx.attr, "enum_name", ""),
            nullable = getattr(ctx.attr, "nullable", False),
        ),
    ]
    if ctx.attr.make_variable:
        providers.append(platform_common.TemplateVariableInfo({
            ctx.attr.make_variable: str(value).lower() if type(value) == "bool" else str(value),
        }))
    return providers

def _string_impl(ctx):
    if ctx.attr.nullable and ctx.attr.values:
        fail("nullable strings cannot have enum values")
    if ctx.attr.enum_name and not ctx.attr.values:
        fail("enum_name requires nonempty values")
    if ctx.attr.values and ctx.build_setting_value not in ctx.attr.values:
        fail("{}: invalid value '{}'; expected one of {}".format(
            ctx.label,
            ctx.build_setting_value,
            ctx.attr.values,
        ))
    return _flag_impl(ctx)

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
        name = setting.label.name
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
        settings: Option flags; target names become quoted Zig constant names.
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

def zig_option(name, build_setting_default, type = None, values = None, enum_name = None, **kwargs):
    """Define a command-line option, inferring its kind from the default.

    Args:
        name: Flag target name and generated Zig constant name.
        build_setting_default: Integer, boolean, or string default value.
        type: Zig integer type; defaults to usize. Only valid for integers.
        values: Allowed string values, emitted as enum fields when nonempty.
        enum_name: Optional exported enum type name; requires a string with nonempty values.
        **kwargs: Attributes forwarded to the underlying flag rule.
    """
    if type != None and not types.is_int(build_setting_default):
        fail("type is only supported for integer options")
    if values != None and not types.is_string(build_setting_default):
        fail("values is only supported for string options")
    if enum_name != None and (not types.is_string(build_setting_default) or not values):
        fail("enum_name requires a string option with nonempty values")
    if types.is_bool(build_setting_default):
        flag = bool_flag
    elif types.is_int(build_setting_default):
        flag = int_flag
        kwargs["type"] = type if type != None else "usize"
    elif types.is_string(build_setting_default):
        flag = string_flag
        kwargs["values"] = values if values != None else []
        kwargs["enum_name"] = enum_name if enum_name != None else ""
    else:
        fail("Zig options require an integer, boolean, or string default")
    flag(name = name, build_setting_default = build_setting_default, module_name = native.module_name() or "", **kwargs)

int_flag = rule(
    implementation = _flag_impl,
    build_setting = config.int(flag = True),
    attrs = {
        "type": attr.string(default = "usize", doc = "Zig integer type, such as usize, u32, or i64."),
        "module_name": _MODULE_NAME_ATTR,
        "make_variable": _MAKE_VARIABLE_ATTR,
        "scope": _SCOPE_ATTR,
    },
    doc = "An integer build setting that can be set on the command line",
)

bool_flag = rule(
    implementation = _flag_impl,
    build_setting = config.bool(flag = True),
    attrs = {
        "module_name": _MODULE_NAME_ATTR,
        "make_variable": _MAKE_VARIABLE_ATTR,
        "scope": _SCOPE_ATTR,
    },
    doc = "A boolean build setting that can be set on the command line",
)

string_flag = rule(
    implementation = _string_impl,
    build_setting = config.string(flag = True),
    attrs = {
        "nullable": attr.bool(doc = "Emit an optional string; an empty value becomes null. Cannot be combined with values."),
        "enum_name": attr.string(doc = "Optional exported Zig enum type name; requires nonempty values."),
        "module_name": _MODULE_NAME_ATTR,
        "values": attr.string_list(
            doc = "The list of allowed values for this setting. An error is raised if any other value is given.",
        ),
        "make_variable": _MAKE_VARIABLE_ATTR,
        "scope": _SCOPE_ATTR,
    },
    doc = "A string-typed build setting that can be set on the command line",
)
