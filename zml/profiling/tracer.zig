const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const platforms = @import("platforms");
const stdx = @import("stdx");
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
            if (info.field_names.len == 0) {
                return .start(name);
            }

            var buffer: [computeEncodedNameLen(name, @TypeOf(metadata))]u8 = undefined;
            var encoded: std.ArrayList(u8) = .initBuffer(&buffer);

            appendEncodedName(&encoded, .failing, name, metadata) catch unreachable;
            return .start(encoded.items);
        },
        else => @compileError("Unsupported metadata."),
    }
}

/// Returns an allocator-owned encoded span name for metadata that cannot be
/// bounded by `span` at comptime. The caller owns the returned slice.
pub fn formatSpanName(allocator: std.mem.Allocator, name: []const u8, metadata: anytype) ![]const u8 {
    switch (@typeInfo(@TypeOf(metadata))) {
        .@"struct" => {
            var encoded: std.ArrayList(u8) = .empty;
            errdefer encoded.deinit(allocator);

            try appendEncodedName(&encoded, allocator, name, metadata);
            return try encoded.toOwnedSlice(allocator);
        },
        else => @compileError("trace metadata must be a struct literal like .{ .step_num = 42 }"),
    }
}

fn appendEncodedName(buffer: *std.ArrayList(u8), allocator: std.mem.Allocator, name: []const u8, metadata: anytype) !void {
    try buffer.appendSlice(allocator, name);

    var field_count: usize = 0;
    try buffer.append(allocator, '#');
    inline for (comptime std.meta.fieldNames(@TypeOf(metadata))) |field_name| {
        if (shouldEncodeField(field_name, @field(metadata, field_name))) {
            if (field_count != 0) {
                try buffer.append(allocator, ',');
            }
            field_count += 1;

            const key = comptime metadataKey(field_name);
            validateMetadataToken(key);
            try buffer.appendSlice(allocator, key);
            try buffer.append(allocator, '=');
            try appendMetadataValue(buffer, allocator, @field(metadata, field_name), field_name);
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
            if (info.field_names.len == 0) return name.len;

            // name#[attrs]#
            var max_len = name.len + 2;
            inline for (
                info.field_names,
                info.field_types,
                info.field_attrs,
            ) |field_name, FieldType, field_attr| {
                const key = comptime metadataKey(field_name);
                validateMetadataToken(key);

                const default: ?FieldType = field_attr.defaultValue(FieldType);
                // [key]=[value],
                max_len += key.len + 2;
                max_len += switch (@typeInfo(FieldType)) {
                    .bool => 1,
                    .int => @max(
                        std.fmt.comptimePrint("{}", .{std.math.minInt(FieldType)}).len,
                        std.fmt.comptimePrint("{}", .{std.math.maxInt(FieldType)}).len,
                    ),
                    .@"enum" => |ty| b: {
                        var max_enum_len: usize = 0;
                        for (ty.field_names) |enum_field_name| {
                            max_enum_len = @max(max_enum_len, enum_field_name.len);
                        }
                        break :b max_enum_len;
                    },
                    .comptime_int => std.fmt.comptimePrint("{}", .{default.?}).len,
                    .comptime_float => std.fmt.comptimePrint("{d}", .{default.?}).len,
                    .enum_literal => @tagName(default.?).len,
                    else => @compileError("trace metadata field '" ++ field_name ++ "' is not statically bounded; use formatspanName"),
                };
            }
            return max_len;
        },
        else => @compileError("trace metadata must be a struct literal like .{ .step_num = 42 }"),
    }
}

fn shouldEncodeField(comptime field_name: []const u8, value: anytype) bool {
    const FieldType = @TypeOf(value);
    return switch (@typeInfo(FieldType)) {
        .optional => value != null,
        .bool => if (comptime std.mem.eql(u8, field_name, "root")) value else true,
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
