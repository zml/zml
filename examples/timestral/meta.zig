/// Builds a `Subset` by copying, field by field, the matching fields of `config`
/// (a struct value, or a single-item pointer to one).
/// Fields of `Subset` missing from `config` use their default value, or trigger a compile error.
pub fn shallowPick(config: anytype, comptime Subset: type) Subset {
    const Config = switch (@typeInfo(@TypeOf(config))) {
        .pointer => |p| blk: {
            if (p.size != .one) {
                @compileError(
                    "pick: expected a struct or a single-item pointer to one, got " ++ @typeName(@TypeOf(config)),
                );
            }
            break :blk p.child;
        },
        else => @TypeOf(config),
    };

    const info = @typeInfo(Subset).@"struct";
    var result: Subset = undefined;
    inline for (info.field_names, info.field_types, info.field_attrs) |name, Field, attrs| {
        if (@hasField(Config, name)) {
            @field(result, name) = @field(config, name);
        } else if (attrs.defaultValue(Field)) |default| {
            @field(result, name) = default;
        } else {
            @compileError("pick: field '" ++ name ++ "' of " ++ @typeName(Subset) ++
                " not found in " ++ @typeName(Config) ++ " and has no default value");
        }
    }
    return result;
}

