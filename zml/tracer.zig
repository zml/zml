const builtin = @import("builtin");
const std = @import("std");

const c = @import("c");
const zffi = @import("ffi");

pub const Tracer = switch (builtin.os.tag) {
    // TODO(cerisier): fix MacOsTracer
    // .macos => MacOsTracer,
    .linux => if (@hasDecl(c, "ZML_RUNTIME_CUDA")) CudaTracer else FakeTracer,
    else => FakeTracer,
};

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

pub fn enabled() bool {
    return c.zml_traceme_enabled();
}

pub fn scope(name: []const u8) Scope {
    return .{
        .inner = c.zml_traceme_start(zffi.ZigSlice.from(name)),
    };
}

pub fn scopeWith(allocator: std.mem.Allocator, name: []const u8, metadata: anytype) !Scope {
    const Metadata = @TypeOf(metadata);
    comptime assertMetadataStruct(Metadata);

    if (comptime metadataFieldCount(Metadata) == 0) {
        return scope(name);
    }

    var encoded = std.ArrayList(u8).empty;
    defer encoded.deinit(allocator);

    try appendEncodedName(&encoded, allocator, name, metadata);
    return scope(encoded.items);
}

pub fn step(allocator: std.mem.Allocator, name: []const u8, step_num: u64) !Scope {
    return scopeWith(allocator, name, .{
        .root = true,
        .step_num = step_num,
    });
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

const CudaTracer = struct {

    // Those symbols are defined in cudaProfiler.h but their implementation is in libcuda.so
    // They will be bound at call time after libcuda.so is loaded (as a needed dependency of libpjrt_cuda.so).
    const cuProfilerStart = @extern(*const fn () callconv(.c) c_int, .{ .name = "cuProfilerStart", .linkage = .weak }) orelse unreachable;
    const cuProfilerStop = @extern(*const fn () callconv(.c) c_int, .{ .name = "cuProfilerStop", .linkage = .weak }) orelse unreachable;

    // Those symbols are defined in nvToolsExt.h which we don't want to provide.
    // However, we link with libnvToolsExt.so which provides them.
    // They will be bound at call time after libnvToolsExt.so is loaded (manually dlopen'ed by us).
    const nvtxMarkA = @extern(*const fn ([*:0]const u8) callconv(.c) void, .{ .name = "nvtxMarkA", .linkage = .weak }) orelse unreachable;
    const nvtxRangeStartA = @extern(*const fn ([*:0]const u8) callconv(.c) c_int, .{ .name = "nvtxRangeStartA", .linkage = .weak }) orelse unreachable;
    const nvtxRangeEnd = @extern(*const fn (c_int) callconv(.c) void, .{ .name = "nvtxRangeEnd", .linkage = .weak }) orelse unreachable;

    pub fn event(message: [:0]const u8) void {
        nvtxMarkA(message.ptr);
    }

    pub fn frameStart(message: [:0]const u8) u64 {
        return @intCast(nvtxRangeStartA(message.ptr));
    }

    pub fn frameEnd(interval_id: u64, message: [:0]const u8) void {
        _ = message;
        nvtxRangeEnd(@intCast(interval_id));
        return;
    }
};

const MacOsTracer = struct {
    logger: c.os_log_t,

    pub fn event(self: *const MacOsTracer, message: [:0]const u8) void {
        const interval_id = c.os_signpost_id_generate(self.logger);
        c.zml_os_signpost_event(self.logger, interval_id, message);
    }

    pub fn frameStart(self: *const MacOsTracer, message: [:0]const u8) c.os_signpost_id_t {
        const interval_id = c.os_signpost_id_generate(self.logger);
        c.zml_os_signpost_interval_begin(self.logger, interval_id, message);
        return interval_id;
    }

    pub fn frameEnd(self: *const MacOsTracer, interval_id: c.os_signpost_id_t, message: [:0]const u8) void {
        c.zml_os_signpost_interval_end(self.logger, interval_id, message);
    }
};

/// Mock tracer for OS which don't have an impl.
const FakeTracer = struct {
    pub fn event(message: [:0]const u8) void {
        _ = message;
        return;
    }

    pub fn frameStart(message: [:0]const u8) u64 {
        _ = message;
        return 0;
    }

    pub fn frameEnd(interval_id: u64, message: [:0]const u8) void {
        _ = interval_id;
        _ = message;
        return;
    }
};

test "scopeWith encodes metadata like TraceMe" {
    var buffer = std.ArrayList(u8).empty;
    defer buffer.deinit(std.testing.allocator);

    try appendEncodedName(&buffer, std.testing.allocator, "train", .{
        .root = true,
        .step_num = @as(u64, 42),
        .enabled = true,
    });

    try std.testing.expectEqualStrings("train#_r=1,step_num=42,enabled=1#", buffer.items);
}

test "scopeWith skips null optionals and false root" {
    var buffer = std.ArrayList(u8).empty;
    defer buffer.deinit(std.testing.allocator);

    try appendEncodedName(&buffer, std.testing.allocator, "load", .{
        .root = false,
        .step_num = @as(?u64, null),
        .stage = "prefill",
    });

    try std.testing.expectEqualStrings("load#stage=prefill#", buffer.items);
}
