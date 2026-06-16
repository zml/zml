const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const platforms = @import("platforms");
const zffi = @import("ffi");

pub const Span = struct {
    inner: ?*c.zml_traceme = null,

    /// Name is copied underneath
    pub fn start(name: []const u8) Span {
        return .{
            .inner = c.zml_traceme_start(zffi.ZigSlice.from(name)),
        };
    }

    pub fn end(self: *Span) void {
        if (self.inner) |inner| {
            c.zml_traceme_stop(inner);
            self.inner = null;
        }
    }
};

// TraceMe is always the base host tracing path. The C tracing bridge adds NVTX
// ranges on CUDA/Linux builds, roctx ranges on ROCm/Linux builds, and
// os_signpost intervals on macOS.
pub const supportsDeviceAnnotations = switch (builtin.os.tag) {
    .macos => true,
    .linux => platforms.target == .cuda or platforms.target == .rocm,
    else => false,
};

pub fn enabled() bool {
    return c.zml_traceme_enabled();
}

/// Creates a Span with encoded metadata. The metadata encoded size must be known at compile time.
/// If it's not your case, use `formatSpanName` to generate the span name before and use void for the metadata.
pub fn span(comptime name: []const u8, metadata: anytype) Span {
    switch (@typeInfo(@TypeOf(metadata))) {
        .void => {
            return .start(name);
        },
        .@"struct" => |info| {
            if (info.fields.len == 0) {
                return .start(name);
            }

            var buffer: [computeEncodedNameLen(name, @TypeOf(metadata))]u8 = undefined;
            var encoded: std.ArrayList(u8) = .initBuffer(&buffer);

            appendEncodedName(&encoded, .failing, name, info, metadata) catch unreachable;
            return .start(encoded.items);
        },
        else => @compileError("Unsupported metadata."),
    }
}

/// Returns an allocator-owned encoded span name for metadata that cannot be
/// bounded by `span` at comptime. The caller owns the returned slice.
pub fn formatSpanName(allocator: std.mem.Allocator, name: []const u8, metadata: anytype) ![]const u8 {
    switch (@typeInfo(@TypeOf(metadata))) {
        .@"struct" => |info| {
            var encoded: std.ArrayList(u8) = .empty;
            errdefer encoded.deinit(allocator);

            try appendEncodedName(&encoded, allocator, name, info, metadata);
            return try encoded.toOwnedSlice(allocator);
        },
        else => @compileError("trace metadata must be a struct literal like .{ .step_num = 42 }"),
    }
}

fn appendEncodedName(buffer: *std.ArrayList(u8), allocator: std.mem.Allocator, name: []const u8, info: std.builtin.Type.Struct, metadata: anytype) !void {
    try buffer.appendSlice(allocator, name);

    var field_count: usize = 0;
    try buffer.append(allocator, '#');
    inline for (info.fields) |field| {
        if (shouldEncodeField(field, @field(metadata, field.name))) {
            if (field_count != 0) {
                try buffer.append(allocator, ',');
            }
            field_count += 1;

            const key = comptime metadataKey(field.name);
            validateMetadataToken(key);
            try buffer.appendSlice(allocator, key);
            try buffer.append(allocator, '=');
            try appendMetadataValue(buffer, allocator, @field(metadata, field.name), field.name);
        }
    }

    // We may not have written any fields because of optional values.
    if (field_count == 0) {
        buffer.items.len = name.len;
    } else {
        try buffer.append(allocator, '#');
    }
}

fn validateMetadataToken(token: []const u8) void {
    if (std.mem.indexOfScalar(u8, token, '#') != null) {
        @panic("trace metadata cannot contain '#'");
    }
}

fn metadataKey(comptime field_name: []const u8) []const u8 {
    return if (comptime std.mem.eql(u8, field_name, "root")) "_r" else field_name;
}

fn computeEncodedNameLen(comptime name: []const u8, comptime Metadata: type) usize {
    switch (@typeInfo(Metadata)) {
        .@"struct" => |info| {
            if (info.fields.len == 0) return name.len;

            // name#[attrs]#
            var max_len = name.len + 2;
            inline for (info.fields) |field| {
                const key = comptime metadataKey(field.name);
                validateMetadataToken(key);

                // [key]=[value],
                max_len += key.len + 2;
                max_len += switch (@typeInfo(field.type)) {
                    .bool => 1,
                    .int => @max(
                        std.fmt.comptimePrint("{}", .{std.math.minInt(field.type)}).len,
                        std.fmt.comptimePrint("{}", .{std.math.maxInt(field.type)}).len,
                    ),
                    .@"enum" => |ty| b: {
                        var max_enum_len: usize = 0;
                        for (ty.fields) |enum_field| {
                            max_enum_len = @max(max_enum_len, enum_field.name.len);
                        }
                        break :b max_enum_len;
                    },
                    .comptime_int => std.fmt.comptimePrint("{}", .{defaultMetadataValue(field).?}).len,
                    .comptime_float => std.fmt.comptimePrint("{d}", .{defaultMetadataValue(field).?}).len,
                    .enum_literal => @tagName(defaultMetadataValue(field).?).len,
                    else => @compileError("trace metadata field '" ++ field.name ++ "' is not statically bounded; use formatspanName"),
                };
            }
            return max_len;
        },
        else => @compileError("trace metadata must be a struct literal like .{ .step_num = 42 }"),
    }
}

fn defaultMetadataValue(comptime field: std.builtin.Type.StructField) ?field.type {
    return if (field.default_value_ptr) |default_opaque|
        @as(*const field.type, @ptrCast(@alignCast(default_opaque))).*
    else
        null;
}

fn shouldEncodeField(field: std.builtin.Type.StructField, value: anytype) bool {
    return switch (@typeInfo(field.type)) {
        .optional => value != null,
        .bool => if (comptime std.mem.eql(u8, field.name, "root")) value else true,
        else => true,
    };
}

fn appendMetadataValue(buffer: *std.ArrayList(u8), allocator: std.mem.Allocator, value: anytype, comptime field_name: []const u8) !void {
    const T = @TypeOf(value);
    switch (@typeInfo(T)) {
        .optional => {
            try appendMetadataValue(buffer, allocator, value.?, field_name);
        },
        .bool => {
            if (comptime std.mem.eql(u8, field_name, "root")) {
                try buffer.append(allocator, '1');
            } else {
                try buffer.appendSlice(allocator, if (value) "1" else "0");
            }
        },
        .int, .comptime_int => try buffer.print(allocator, "{}", .{value}),
        .float, .comptime_float => try buffer.print(allocator, "{d}", .{value}),
        .enum_literal => try buffer.appendSlice(allocator, @tagName(value)),
        .@"enum" => try buffer.appendSlice(allocator, @tagName(value)),
        .pointer => |ptr| {
            if (ptr.size == .slice and ptr.child == u8) {
                validateMetadataToken(value);
                try buffer.appendSlice(allocator, value);
                return;
            }

            if (ptr.size == .one) switch (@typeInfo(ptr.child)) {
                .array => |arr| {
                    if (arr.child == u8) {
                        const slice = value[0..];
                        validateMetadataToken(slice);
                        try buffer.appendSlice(allocator, slice);
                        return;
                    }
                },
                else => {},
            };

            @compileError("trace metadata pointers must be []const u8 or string literals");
        },
        .array => |arr| {
            if (arr.child != u8) {
                @compileError("trace metadata arrays must be [N]u8");
            }
            validateMetadataToken(value[0..]);
            try buffer.appendSlice(allocator, value[0..]);
        },
        else => @compileError("unsupported trace metadata type for field '" ++ field_name ++ "'"),
    }
}

test "formatspanName encodes metadata" {
    const formatted = try formatSpanName(std.testing.allocator, "test.span", .{
        .root = true,
        .count = @as(u8, 42),
        .label = "batch",
    });
    defer std.testing.allocator.free(formatted);

    try std.testing.expectEqualStrings("test.span#_r=1,count=42,label=batch#", formatted);
}

test "formatspanName omits empty metadata" {
    const formatted = try formatSpanName(std.testing.allocator, "test.span", .{
        .root = false,
        .optional = @as(?u8, null),
    });
    defer std.testing.allocator.free(formatted);

    try std.testing.expectEqualStrings("test.span", formatted);
}
