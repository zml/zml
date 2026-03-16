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

        pub fn jsonStringify(self: Self, jw: anytype) !void {
            return switch (self.value) {
                inline else => |v| jw.write(v),
            };
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
    comptime {
        if (@typeInfo(T) != .@"union") {
            @compileError("TaggedUnion expects a union type, found " ++ @typeName(T));
        }
        for (std.meta.fields(T)) |field| {
            if (@typeInfo(field.type) != .@"struct") {
                @compileError("TaggedUnion member '" ++ field.name ++ "' must be a struct, found " ++ @typeName(field.type));
            }
        }
    }

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
            var o = source.object;
            const tag = (o.fetchSwapRemove(tag_name) orelse return error.MissingField).value;
            if (tag != .string) return error.LengthMismatch;
            inline for (std.meta.fields(T)) |field| {
                if (std.mem.eql(u8, field.name, tag.string)) {
                    const inner: field.type = std.json.parseFromValueLeaky(field.type, allocator, .{ .object = o }, options) catch |err| {
                        std.log.warn("failed to interpret {s} as a {s}: {}", .{ tag.string, @typeName(field.type), err });
                        return err;
                    };
                    return .{ .value = @unionInit(T, field.name, inner) };
                }
            }
            return error.InvalidEnumTag;
        }

        pub fn jsonStringify(self: Self, jw: anytype) !void {
            try jw.beginObject();
            switch (self.value) {
                inline else => |v, tag| {
                    const field_name = @tagName(tag);
                    try jw.objectField(tag_name);
                    try jw.write(field_name);
                    switch (@typeInfo(@TypeOf(v))) {
                        .@"struct" => inline for (std.meta.fields(@TypeOf(v))) |field| {
                            try jw.objectField(field.name);
                            try jw.write(@field(v, field.name));
                        },
                        else => unreachable, // Would have failed with comptime check.
                    }
                },
            }
            try jw.endObject();
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

        pub fn jsonStringify(self: Self, jw: anytype) !void {
            try jw.write(self.value);
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

test "union" {
    const Answer = union(enum) {
        number: i32,
        numbers: []const i32,
        flag: bool,
        ratio: f64,
        text: []const u8,
    };

    const Message = struct {
        question: []const u8,
        answer: Union(Answer),
    };

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const number_case = "{\"question\":\"How old are you?\",\"answer\":5}";
    const parsed_number = try std.json.parseFromSliceLeaky(Message, allocator, number_case, .{});
    try std.testing.expectEqualStrings("How old are you?", parsed_number.question);
    try std.testing.expectEqual(@as(i32, 5), parsed_number.answer.value.number);
    try std.testing.expectFmt(number_case, "{f}", .{std.json.fmt(parsed_number, .{})});

    const numbers_case = "{\"question\":\"Count to three.\",\"answer\":[1,2,3]}";
    const parsed_numbers = try std.json.parseFromSliceLeaky(Message, allocator, numbers_case, .{});
    try std.testing.expectEqual(@as(usize, 3), parsed_numbers.answer.value.numbers.len);
    try std.testing.expectEqual(@as(i32, 1), parsed_numbers.answer.value.numbers[0]);
    try std.testing.expectEqual(@as(i32, 2), parsed_numbers.answer.value.numbers[1]);
    try std.testing.expectEqual(@as(i32, 3), parsed_numbers.answer.value.numbers[2]);
    try std.testing.expectFmt(numbers_case, "{f}", .{std.json.fmt(parsed_numbers, .{})});

    const flag_case = "{\"question\":\"Are you ready?\",\"answer\":true}";
    const parsed_flag = try std.json.parseFromSliceLeaky(Message, allocator, flag_case, .{});
    try std.testing.expectEqual(true, parsed_flag.answer.value.flag);
    try std.testing.expectFmt(flag_case, "{f}", .{std.json.fmt(parsed_flag, .{})});

    const ratio_case = "{\"question\":\"What is pi?\",\"answer\":3.5}";
    const parsed_ratio = try std.json.parseFromSliceLeaky(Message, allocator, ratio_case, .{});
    try std.testing.expectEqual(@as(f64, 3.5), parsed_ratio.answer.value.ratio);
    try std.testing.expectFmt(ratio_case, "{f}", .{std.json.fmt(parsed_ratio, .{})});

    const text_case = "{\"question\":\"Your name?\",\"answer\":\"zml\"}";
    const parsed_text = try std.json.parseFromSliceLeaky(Message, allocator, text_case, .{});
    try std.testing.expectEqualStrings("zml", parsed_text.answer.value.text);
    try std.testing.expectFmt(text_case, "{f}", .{std.json.fmt(parsed_text, .{})});

    const null_value = try std.json.parseFromSliceLeaky(std.json.Value, allocator, "null", .{});
    try std.testing.expectError(error.UnknownField, Union(Answer).jsonParseFromValue(allocator, null_value, .{}));
}

test "tagged union" {
    const Entry = union(enum) {
        faq: struct {
            question: []const u8,
            answer: u32,
        },
        address: struct {
            city: []const u8,
            zipcode: []const u8,
        },
    };

    const Message = TaggedUnion(Entry, "type");

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const faq_case = "{\"type\":\"faq\",\"question\":\"How old are you?\",\"answer\":5}";
    const parsed_faq = try std.json.parseFromSliceLeaky(Message, allocator, faq_case, .{});
    try std.testing.expectEqualStrings("How old are you?", parsed_faq.value.faq.question);
    try std.testing.expectEqual(@as(u32, 5), parsed_faq.value.faq.answer);
    try std.testing.expectFmt(faq_case, "{f}", .{std.json.fmt(parsed_faq, .{})});

    const address_case = "{\"type\":\"address\",\"city\":\"NYC\",\"zipcode\":\"49130\"}";
    const parsed_address = try std.json.parseFromSliceLeaky(Message, allocator, address_case, .{});
    try std.testing.expectEqualStrings("NYC", parsed_address.value.address.city);
    try std.testing.expectEqualStrings("49130", parsed_address.value.address.zipcode);
    try std.testing.expectFmt(address_case, "{f}", .{std.json.fmt(parsed_address, .{})});
}

test "never null" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const non_null = try std.json.parseFromSliceLeaky(NeverNull(i32, 10), allocator, "3", .{});
    try std.testing.expectEqual(@as(i32, 3), non_null.value);
    try std.testing.expectFmt("3", "{f}", .{std.json.fmt(non_null, .{})});

    const was_null = try std.json.parseFromSliceLeaky(NeverNull(i32, 10), allocator, "null", .{});
    try std.testing.expectEqual(@as(i32, 10), was_null.value);
    try std.testing.expectFmt("10", "{f}", .{std.json.fmt(was_null, .{})});
}
