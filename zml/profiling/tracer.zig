const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const platforms = @import("platforms");
const zffi = @import("ffi");

pub const Scope = struct {
    inner: ?*c.zml_traceme = null,

    pub fn end(self: *Scope) void {
        if (self.inner) |inner| {
            c.zml_traceme_stop(inner);
            self.inner = null;
        }
    }

    pub fn deinit(self: *Scope) void {
        self.end();
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
const inline_scope_name_buffer_size = 1024;

pub fn enabled() bool {
    return c.zml_traceme_enabled();
}

pub fn scope(name: []const u8, metadata: anytype) !Scope {
    const Metadata = @TypeOf(metadata);
    comptime assertMetadataStruct(Metadata);

    if (comptime metadataFieldCount(Metadata) == 0) {
        return .{
            .inner = c.zml_traceme_start(zffi.ZigSlice.from(name)),
        };
    }

    var inline_buffer: [inline_scope_name_buffer_size]u8 = undefined;
    var fixed_buffer_allocator: std.heap.FixedBufferAllocator = .init(&inline_buffer);
    const fixed_allocator = fixed_buffer_allocator.allocator();

    var encoded: std.ArrayList(u8) = .empty;
    defer encoded.deinit(fixed_allocator);

    try appendEncodedName(&encoded, fixed_allocator, name, metadata);
    return .{
        .inner = c.zml_traceme_start(zffi.ZigSlice.from(encoded.items)),
    };
}

fn appendEncodedName(buffer: *std.ArrayList(u8), allocator: std.mem.Allocator, name: []const u8, metadata: anytype) !void {
    const Metadata = @TypeOf(metadata);
    try buffer.appendSlice(allocator, name);
    if (comptime metadataFieldCount(Metadata) == 0) return;

    var field_count: usize = 0;
    try buffer.append(allocator, '#');
    inline for (std.meta.fields(Metadata)) |field| {
        if (shouldEncodeField(@TypeOf(@field(metadata, field.name)), @field(metadata, field.name), field.name)) {
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
    if (field_count == 0) {
        buffer.items.len = name.len;
        return;
    }
    try buffer.append(allocator, '#');
}

fn validateMetadataToken(token: []const u8) void {
    if (std.mem.indexOfScalar(u8, token, '#') != null) {
        @panic("trace metadata cannot contain '#'");
    }
}

fn assertMetadataStruct(comptime T: type) void {
    switch (@typeInfo(T)) {
        .@"struct" => {},
        else => @compileError("trace metadata must be a struct literal like .{ .step_num = 42 }"),
    }
}

fn metadataFieldCount(comptime T: type) usize {
    assertMetadataStruct(T);
    return std.meta.fields(T).len;
}

fn metadataKey(comptime field_name: []const u8) []const u8 {
    return if (comptime std.mem.eql(u8, field_name, "root")) "_r" else field_name;
}

fn shouldEncodeField(comptime T: type, value: T, comptime field_name: []const u8) bool {
    return switch (@typeInfo(T)) {
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
