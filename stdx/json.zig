pub const std = @import("std");

pub fn Union(comptime T: type) type {
    return struct {
        const Self = @This();

        value: T,

        pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Self {
            return jsonParseFromValue(
                allocator,
                try std.json.innerParse(
                    std.json.Value,
                    allocator,
                    source,
                    options,
                ),
                options,
            );
        }

        pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) !Self {
            inline for (std.meta.fields(T)) |field| {
                switch (field.type) {
                    bool => if (source == .bool) return .{ .value = @unionInit(T, field.name, source.bool) },
                    []const u8 => switch (source) {
                        .string => |v| return .{ .value = @unionInit(T, field.name, v) },
                        .number_string => |v| return .{ .value = @unionInit(T, field.name, v) },
                        else => {},
                    },
                    else => switch (@typeInfo(field.type)) {
                        .int => if (source == .integer) return .{ .value = @unionInit(T, field.name, @intCast(source.integer)) },
                        .float => if (source == .float) return .{ .value = @unionInit(T, field.name, @floatCast(source.float)) },
                        .@"struct" => if (source == .object) return .{ .value = @unionInit(T, field.name, try std.json.innerParseFromValue(field.type, allocator, source.object, options)) },
                        inline else => switch (source) {
                            .number_string, .array => return .{ .value = @unionInit(T, field.name, try std.json.innerParseFromValue(field.type, allocator, source, options)) },
                            else => {},
                        },
                    },
                }
            }
            return error.UnexpectedToken;
        }
    };
}

pub fn NeverNull(comptime T: type, comptime default_value: T) type {
    return struct {
        const Self = @This();

        value: T = default_value,

        pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !Self {
            return .{ .value = (try std.json.innerParse(?T, allocator, source, options)) orelse default_value };
        }

        pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) !Self {
            return .{ .value = (try std.json.innerParseFromValue(?T, allocator, source, options)) orelse default_value };
        }
    };
}

pub fn fillDefaultStructValues(comptime T: type, r: *T) !void {
    inline for (@typeInfo(T).Struct.fields) |field| {
        if (field.default_value) |default_ptr| {
            if (@field(r, field.name) == null) {
                const default = @as(*align(1) const field.type, @ptrCast(default_ptr)).*;
                @field(r, field.name) = default;
            }
        }
    }
}
