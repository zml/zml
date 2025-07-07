const std = @import("std");
const log = std.log.scoped(.@"zml/runtime/common");

fn symbolPredicate(comptime decl_name: []const u8, comptime prefix: []const u8, module: type) bool {
    return std.mem.startsWith(u8, decl_name, prefix) and @typeInfo(@TypeOf(@field(module, decl_name))) == .@"fn";
}

pub fn weakifySymbols(module: type, prefix: []const u8) type {

    @setEvalBranchQuota(200000);

    var len = 0;
    for (@typeInfo(module).@"struct".decls) |decl| {
        if (symbolPredicate(decl.name, prefix, module)) {
            len += 1;
        }
    }

    var fields: [len]std.builtin.Type.StructField = undefined;
    var i = 0;
    for (@typeInfo(module).@"struct".decls) |decl| {
        if (symbolPredicate(decl.name, prefix, module)) {
            const field = std.builtin.Type.StructField{
                .name = decl.name,
                .type = @Type(.{
                    .pointer = std.builtin.Type.Pointer{
                        .child = @TypeOf(@field(module, decl.name)),
                        .is_const = true,
                        .is_volatile = false,
                        .is_allowzero = false,
                        .size = .one,
                        .sentinel_ptr = null,
                        .address_space = .generic,
                        .alignment = @alignOf(@TypeOf(@field(module, decl.name))),
                    },
                }),
                .default_value_ptr = null,
                .is_comptime = false,
                .alignment = 8,
            };
            fields[i] = field;
            i += 1;
        }
    }

    return @Type(.{
        .@"struct" = .{
            .is_tuple = false,
            .layout = .auto,
            .decls = &.{},
            .fields = &fields,
        },
    });
}

pub fn bindWeakSymbols(container: anytype, module: type, comptime prefix: []const u8) !void {
    if (@typeInfo(@TypeOf(container)) != .@"pointer") {
        @compileError("container must be a pointer type");
    }
    @setEvalBranchQuota(200000);
    inline for (@typeInfo(module).@"struct".decls) |decl| {
        if (comptime symbolPredicate(decl.name, prefix, module)) {
            const symb = std.c.dlsym(null, decl.name) orelse {
                log.err("Unable to find symbol {s} in {s}", .{decl.name, @typeName(module)});
                return error.FileNotFound;
            };
            
            @field(container.*, decl.name) = @ptrCast(symb);
        }
    }
}
