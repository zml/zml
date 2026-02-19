const std = @import("std");

pub const Namespace = struct {
    map: std.StringHashMap(Value),

    pub fn init(allocator: std.mem.Allocator) Namespace {
        return .{ .map = std.StringHashMap(Value).init(allocator) };
    }
};

pub const LoopMeta = struct {
    index0: i64,
    index: i64,
    first: bool,
    last: bool,
    previtem: Value,
    nextitem: Value,
};

pub const Value = union(enum) {
    undefined,
    null,
    bool: bool,
    int: i64,
    float: f64,
    str: []const u8,
    list: []const Value,
    map: []const Entry,
    namespace: *Namespace,
    loopmeta: *LoopMeta,

    pub const Entry = struct {
        key: []const u8,
        value: Value,
    };

    pub fn fromBool(v: bool) Value {
        return .{ .bool = v };
    }

    pub fn fromInt(v: i64) Value {
        return .{ .int = v };
    }

    pub fn fromString(v: []const u8) Value {
        return .{ .str = v };
    }

    pub fn fromList(v: []const Value) Value {
        return .{ .list = v };
    }

    pub fn fromMap(v: []const Entry) Value {
        return .{ .map = v };
    }

    pub fn isTruthy(self: Value) bool {
        return switch (self) {
            .undefined, .null => false,
            .bool => |v| v,
            .int => |v| v != 0,
            .float => |v| v != 0.0,
            .str => |v| v.len > 0,
            .list => |v| v.len > 0,
            .map => |v| v.len > 0,
            .namespace => |ns| ns.map.count() > 0,
            .loopmeta => true,
        };
    }

    pub fn getAttr(self: Value, key: []const u8) ?Value {
        return switch (self) {
            .map => |entries| mapGet(entries, key),
            .namespace => |ns| ns.map.get(key),
            .loopmeta => |lp| loopMetaGet(lp, key),
            else => null,
        };
    }

    pub fn setAttr(self: Value, key: []const u8, value: Value) !void {
        switch (self) {
            .namespace => |ns| try ns.map.put(key, value),
            else => return error.TypeError,
        }
    }

    pub fn eql(a: Value, b: Value) bool {
        return switch (a) {
            .undefined => switch (b) {
                .undefined => true,
                else => false,
            },
            .null => switch (b) {
                .null => true,
                else => false,
            },
            .bool => |av| switch (b) {
                .bool => |bv| av == bv,
                else => false,
            },
            .int => |av| switch (b) {
                .int => |bv| av == bv,
                .float => |bv| @as(f64, @floatFromInt(av)) == bv,
                else => false,
            },
            .float => |av| switch (b) {
                .int => |bv| av == @as(f64, @floatFromInt(bv)),
                .float => |bv| av == bv,
                else => false,
            },
            .str => |av| switch (b) {
                .str => |bv| std.mem.eql(u8, av, bv),
                else => false,
            },
            .list => false,
            .map => false,
            .namespace => false,
            .loopmeta => false,
        };
    }
};

const Counts = struct {
    values: usize = 0,
    entries: usize = 0,
};

fn typeCounts(comptime T: type) Counts {
    if (T == Value or T == Value.Entry) return .{};
    return switch (@typeInfo(T)) {
        .bool, .int, .comptime_int, .float, .comptime_float => .{},
        .@"enum" => .{},
        .optional => |opt| typeCounts(opt.child),
        .pointer => |ptr| switch (ptr.size) {
            .one => typeCounts(ptr.child),
            .slice => if (ptr.child == u8) .{} else @compileError("zero-alloc struct context does not support non-string slices"),
            else => @compileError("unsupported pointer type for zero-alloc struct context"),
        },
        .array => |arr| blk: {
            if (arr.child == u8) break :blk .{};
            const child_counts = typeCounts(arr.child);
            break :blk .{
                .values = arr.len + child_counts.values * arr.len,
                .entries = child_counts.entries * arr.len,
            };
        },
        .@"struct" => |s| blk: {
            var out = Counts{ .entries = s.fields.len };
            inline for (s.fields) |f| {
                const fc = typeCounts(f.type);
                out.values += fc.values;
                out.entries += fc.entries;
            }
            break :blk out;
        },
        else => @compileError("unsupported type for zero-alloc struct context: " ++ @typeName(T)),
    };
}

pub fn StructContext(comptime T: type) type {
    const counts = typeCounts(T);

    return struct {
        const Self = @This();

        values: [counts.values]Value = undefined,
        entries: [counts.entries]Value.Entry = undefined,
        values_len: usize = 0,
        entries_len: usize = 0,

        pub fn from(self: *Self, src: *const T) []const Value.Entry {
            self.values_len = 0;
            self.entries_len = 0;
            const root = self.toValue(T, src);
            return root.map;
        }

        fn toValue(self: *Self, comptime U: type, src: *const U) Value {
            if (U == Value) return src.*;
            return switch (@typeInfo(U)) {
                .bool => .{ .bool = src.* },
                .int, .comptime_int => .{ .int = @intCast(src.*) },
                .float, .comptime_float => .{ .float = @floatCast(src.*) },
                .@"enum" => .{ .str = @tagName(src.*) },
                .optional => |opt| if (src.*) |v| self.toValue(opt.child, &v) else .null,
                .pointer => |ptr| switch (ptr.size) {
                    .one => self.toValue(ptr.child, src.*),
                    .slice => if (ptr.child == u8) .{ .str = src.* } else @compileError("zero-alloc struct context does not support non-string slices"),
                    else => @compileError("unsupported pointer type for zero-alloc struct context"),
                },
                .array => |arr| blk: {
                    if (arr.child == u8) break :blk .{ .str = src.*[0..] };
                    const start = self.values_len;
                    self.values_len += arr.len;
                    inline for (0..arr.len) |i| {
                        self.values[start + i] = self.toValue(arr.child, &src.*[i]);
                    }
                    break :blk .{ .list = self.values[start .. start + arr.len] };
                },
                .@"struct" => |s| blk: {
                    const start = self.entries_len;
                    self.entries_len += s.fields.len;
                    inline for (s.fields, 0..) |f, i| {
                        self.entries[start + i] = .{
                            .key = f.name,
                            .value = self.toValue(f.type, &@field(src.*, f.name)),
                        };
                    }
                    break :blk .{ .map = self.entries[start .. start + s.fields.len] };
                },
                else => @compileError("unsupported type for zero-alloc struct context: " ++ @typeName(U)),
            };
        }
    };
}

fn mapGet(entries: []const Value.Entry, key: []const u8) ?Value {
    for (entries) |entry| {
        if (entry.key.ptr == key.ptr and entry.key.len == key.len) return entry.value;
        if (entry.key.len != key.len) continue;
        if (entry.key.len > 0 and entry.key[0] != key[0]) continue;
        if (std.mem.eql(u8, entry.key, key)) return entry.value;
    }
    return null;
}

fn loopMetaGet(lp: *const LoopMeta, key: []const u8) ?Value {
    if (key.len == 0) return null;
    switch (key[0]) {
        'f' => if (std.mem.eql(u8, key, "first")) return .{ .bool = lp.first },
        'i' => {
            if (std.mem.eql(u8, key, "index")) return .{ .int = lp.index };
            if (std.mem.eql(u8, key, "index0")) return .{ .int = lp.index0 };
        },
        'l' => if (std.mem.eql(u8, key, "last")) return .{ .bool = lp.last },
        'n' => if (std.mem.eql(u8, key, "nextitem")) return lp.nextitem,
        'p' => if (std.mem.eql(u8, key, "previtem")) return lp.previtem,
        else => {},
    }
    return null;
}

pub fn writeValue(writer: anytype, value: Value) !void {
    switch (value) {
        .undefined, .null => try writer.writeAll(""),
        .bool => |v| {
            if (v) {
                try writer.writeAll("true");
            } else {
                try writer.writeAll("false");
            }
        },
        .int => |v| {
            var buf: [32]u8 = undefined;
            const rendered = try std.fmt.bufPrint(&buf, "{}", .{v});
            try writer.writeAll(rendered);
        },
        .float => |v| {
            var buf: [64]u8 = undefined;
            const rendered = try std.fmt.bufPrint(&buf, "{}", .{v});
            try writer.writeAll(rendered);
        },
        .str => |v| try writer.writeAll(v),
        .list => |v| {
            try writer.writeByte('[');
            for (v, 0..) |item, idx| {
                if (idx > 0) try writer.writeAll(", ");
                try writeValue(writer, item);
            }
            try writer.writeByte(']');
        },
        .map => |entries| {
            try writer.writeByte('{');
            for (entries, 0..) |entry, idx| {
                if (idx > 0) try writer.writeAll(", ");
                try writer.writeAll(entry.key);
                try writer.writeAll(": ");
                try writeValue(writer, entry.value);
            }
            try writer.writeByte('}');
        },
        .namespace => |ns| {
            try writer.writeByte('{');
            var it = ns.map.iterator();
            var idx: usize = 0;
            while (it.next()) |entry| {
                if (idx > 0) try writer.writeAll(", ");
                idx += 1;
                try writer.writeAll(entry.key_ptr.*);
                try writer.writeAll(": ");
                try writeValue(writer, entry.value_ptr.*);
            }
            try writer.writeByte('}');
        },
        .loopmeta => |lp| {
            try writer.writeByte('{');
            try writer.writeAll("index0: ");
            try writeValue(writer, .{ .int = lp.index0 });
            try writer.writeAll(", index: ");
            try writeValue(writer, .{ .int = lp.index });
            try writer.writeAll(", first: ");
            try writeValue(writer, .{ .bool = lp.first });
            try writer.writeAll(", last: ");
            try writeValue(writer, .{ .bool = lp.last });
            try writer.writeByte('}');
        },
    }
}
