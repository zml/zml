pub const std = @import("std");
const ParseFromValueError = std.json.ParseFromValueError;

/// Handle json fields that can have different Zig types depending on the message.
/// Each union field should have a unique Zig type.
///
/// Example json:
///
/// ```json
/// [
///    { "question": "How old are you ?", "answer": 5 },
///    { "question": "Count to three.", "answer": [1, 2, 3] },
/// ]
/// ```
///
/// Corresponding Zig code:
///
/// ```zig
/// const Answer = union {
///    number: i32,
///    numbers: []const i32,
/// };
///
/// const Message = struct {
///    question: []const u8;
///    answer: stdx.json.Union(Answer);
/// }
/// ```
pub fn Union(comptime T: type) type {
    return struct {
        const Self = @This();

        value: T,

        pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) std.json.ParseError(@TypeOf(source.*))!Self {
            return jsonParseFromValue(
                allocator,
                try std.json.innerParse(std.json.Value, allocator, source, options),
                options,
            );
        }

        pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) ParseFromValueError!Self {
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
            return error.UnknownField;
        }
    };
}

/// Handle json fields that can have different Zig types depending on another field in the same message.
/// This is translated to a Zig tagged union.
///
/// Example json:
///
/// ```json
/// [
///    { "type": "faq", "question": "How old are you ?", "answer": 5 },
///    { "type": "address", "city": "NYC", "zipcode": "49130"},
/// ]
/// ```
///
/// Corresponding Zig struct:
///
/// ```zig
/// const Entry = union {
///    faq: struct { question: []const u8, answer: u32 },
///    address: struct { city: []const u8, zipcode: []const u8 },
/// };
///
/// const Message = []const stdx.json.TaggedUnion(Entry, "type");
/// ```
pub fn TaggedUnion(comptime T: type, comptime tag_name: [:0]const u8) type {
    return struct {
        const Self = @This();

        value: T,

        pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) std.json.ParseError(@TypeOf(source.*))!Self {
            return jsonParseFromValue(
                allocator,
                try std.json.innerParse(std.json.Value, allocator, source, options),
                options,
            );
        }

        pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) ParseFromValueError!Self {
            errdefer std.log.warn("failed to parse: {} as {s}", .{ source, @typeName(T) });
            if (source != .object) return error.UnexpectedToken;
            const o = source.object;
            const tag = o.get(tag_name) orelse return error.MissingField;
            for (o.keys(), o.values()) |k, v| {
                std.log.warn("object['{s}'] = {}", .{ k, v });
            }
            if (tag != .string) return error.LengthMismatch;
            inline for (std.meta.fields(T)) |field| {
                if (std.mem.eql(u8, field.name, tag.string)) {
                    const inner_source = o.get(field.name) orelse return error.MissingField;
                    const inner: field.type = std.json.innerParseFromValue(field.type, allocator, inner_source, options) catch |err| {
                        std.log.warn("failed to interpret {s} as a {s}: {}", .{ tag.string, @typeName(field.type), err });
                        return err;
                    };
                    return .{ .value = @unionInit(T, field.name, inner) };
                }
            }
            return error.InvalidEnumTag;
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
