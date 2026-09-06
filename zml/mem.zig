//! Generic buffer storage and conversion from tensor models to buffer models.

const std = @import("std");

const Buffer = @import("buffer.zig").Buffer;
const meta = @import("meta.zig");
const Tensor = @import("tensor.zig").Tensor;

/// Placement of page-backed host arenas used for device transfers.
pub const NumaPlacement = union(enum) {
    /// Interleave page by page over every host node that has memory
    /// (`/sys/devices/system/node/has_memory`). A single node, or no
    /// readable list, applies no policy.
    memory_nodes,
    /// One bit per node: a single bit binds, several interleave.
    nodes: u64,
    /// No policy: wherever the kernel and the driver put the pages.
    none,
};

/// Return a clone of a type with Tensors replaced by Buffer.
/// Non-Tensor metadata is stripped out of the resulting struct.
/// Recursively descends into the type.
pub fn Bufferized(comptime T: type) type {
    @setEvalBranchQuota(10_000);
    return meta.MapRestrict(Tensor, Buffer).map(T);
}

/// Convert a model to its bufferized form by replacing Tensor fields with Buffer
/// and allocating any required slices using the provided allocator.
pub inline fn bufferize(allocator: std.mem.Allocator, comptime ModelType: type, model: *const ModelType) !Bufferized(ModelType) {
    var bufferized: Bufferized(ModelType) = undefined;
    try bufferizeInner(allocator, model.*, &bufferized);
    return bufferized;
}

/// Deinitializes every accelerator buffer and frees the recursive slice
/// storage allocated by `bufferize`.
pub fn deinitBufferized(allocator: std.mem.Allocator, comptime ModelType: type, bufferized: *Bufferized(ModelType)) void {
    deinitBufferizedInner(allocator, bufferized);
}

pub const FixedBufferPool = struct {
    buffer: []u8,
    block_size: usize,
    q_buf: []const u16,
    q: std.Io.Queue(u16),

    pub fn init(allocator: std.mem.Allocator, buffer_: []u8, blocks_: u16) !FixedBufferPool {
        const block_size = @divExact(buffer_.len, blocks_);
        const q_buf = try allocator.alloc(u16, blocks_);
        for (q_buf, 0..) |*idx, i| {
            idx.* = @intCast(i);
        }
        var q: std.Io.Queue(u16) = .init(q_buf);
        q.type_erased.len = q.type_erased.buffer.len; // make the queue full
        return .{
            .buffer = buffer_,
            .block_size = block_size,
            .q = q,
            .q_buf = q_buf,
        };
    }

    pub fn deinit(self: *FixedBufferPool, allocator: std.mem.Allocator) void {
        allocator.free(self.q_buf);
    }

    pub fn get(self: *FixedBufferPool, io: std.Io) ![]u8 {
        const idx = try self.q.getOneUncancelable(io);
        return self.buffer[idx * self.block_size ..][0..self.block_size];
    }

    pub fn put(self: *FixedBufferPool, io: std.Io, buf: []u8) void {
        std.debug.assert(inRange(buf, self.buffer));
        const idx = @divExact(@intFromPtr(buf.ptr) - @intFromPtr(self.buffer.ptr), self.block_size);
        self.q.putOneUncancelable(io, @intCast(idx)) catch unreachable;
    }

    fn inRange(sub_buffer: []const u8, buffer: []const u8) bool {
        return @intFromPtr(sub_buffer.ptr) >= @intFromPtr(buffer.ptr) and
            @intFromPtr(sub_buffer[sub_buffer.len - 1 ..].ptr) <= @intFromPtr(buffer[buffer.len - 1 ..].ptr);
    }
};

fn bufferizeInner(allocator: std.mem.Allocator, model: anytype, bufferized_: *Bufferized(@TypeOf(model))) !void {
    @setEvalBranchQuota(10_000);
    const Model = @TypeOf(model);
    const ModelBufferized = Bufferized(Model);

    if (ModelBufferized == Buffer) {
        bufferized_.* = .{
            // I'm not sure I like that. I'd rather set all fields but _shards than leaving most of the m undefined.
            ._shape = model._shape,
            ._shards = .empty,
            ._sharding = undefined,
            ._platform = undefined,
        };
        return;
    }

    const type_info = @typeInfo(ModelBufferized);
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            var initialized_fields: usize = 0;
            errdefer inline for (struct_type_info.fields, 0..) |field, index| {
                if (index < initialized_fields)
                    deinitBufferizedInner(allocator, &@field(bufferized_, field.name));
            };
            inline for (struct_type_info.fields) |field| {
                try bufferizeInner(allocator, @field(model, field.name), &@field(bufferized_, field.name));
                initialized_fields += 1;
            }
        },
        .@"union" => {
            switch (model) {
                inline else => |v, tag| {
                    bufferized_.* = @unionInit(ModelBufferized, @tagName(tag), undefined);
                    try bufferizeInner(allocator, v, &@field(bufferized_, @tagName(tag)));
                },
            }
        },
        .optional => |optional_type_info| {
            if (model == null) {
                bufferized_.* = null;
            } else {
                bufferized_.* = @as(optional_type_info.child, undefined);
                try bufferizeInner(allocator, model.?, &bufferized_.*.?);
            }
        },
        .pointer => |p| {
            switch (p.size) {
                .slice => {
                    const allocated = try allocator.alignedAlloc(p.child, .fromByteUnits(p.alignment orelse @alignOf(p.child)), model.len);
                    var initialized: usize = 0;
                    errdefer {
                        for (allocated[0..initialized]) |*element| deinitBufferizedInner(allocator, element);
                        allocator.free(allocated);
                    }
                    for (model, allocated) |src, *dst| {
                        try bufferizeInner(allocator, src, dst);
                        initialized += 1;
                    }
                    bufferized_.* = allocated;
                },
                else => unreachable,
            }
        },
        .array => |info| {
            var initialized: usize = 0;
            errdefer for (bufferized_.*[0..initialized]) |*element| deinitBufferizedInner(allocator, element);
            inline for (0..info.len) |index| {
                try bufferizeInner(allocator, model[index], &bufferized_.*[index]);
                initialized = index + 1;
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .vector => {},
        else => unreachable,
    }
}

fn deinitBufferizedInner(allocator: std.mem.Allocator, value: anytype) void {
    const Ptr = @TypeOf(value);
    const T = @typeInfo(Ptr).pointer.child;
    if (T == Buffer) {
        const buffer: *Buffer = @constCast(value);
        buffer.deinit();
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| inline for (info.fields) |field| {
            deinitBufferizedInner(allocator, &@field(value, field.name));
        },
        .@"union" => switch (value.*) {
            inline else => |*payload| deinitBufferizedInner(allocator, payload),
        },
        .optional => if (value.*) |*payload| {
            deinitBufferizedInner(allocator, payload);
        },
        .pointer => |info| switch (info.size) {
            .slice => {
                for (value.*) |*element| deinitBufferizedInner(allocator, element);
                allocator.free(value.*);
            },
            else => unreachable,
        },
        .array => for (&value.*) |*element| deinitBufferizedInner(allocator, element),
        .void, .int, .@"enum", .bool, .enum_literal, .float, .vector => {},
        else => unreachable,
    }
}

test "bufferize rolls back earlier slice fields on allocation failure" {
    const Model = struct {
        first: []const Tensor,
        second: []const Tensor,
    };
    var tensors: [1]Tensor = undefined;
    const model: Model = .{ .first = &tensors, .second = &tensors };
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 1 });

    if (bufferize(failing.allocator(), Model, &model)) |result| {
        var unexpected = result;
        deinitBufferized(failing.allocator(), Model, &unexpected);
        return error.ExpectedOutOfMemory;
    } else |err| {
        try std.testing.expect(err == error.OutOfMemory);
    }
    try std.testing.expect(failing.has_induced_failure);
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}

test "deinitBufferized frees recursive const slices" {
    const Layer = struct { weights: []const Tensor };
    const Model = struct {
        layers: []const Layer,
        fixed: [2]Tensor,
    };
    var weights: [2]Tensor = undefined;
    const layers = [_]Layer{
        .{ .weights = weights[0..1] },
        .{ .weights = weights[1..2] },
    };
    const model: Model = .{ .layers = &layers, .fixed = undefined };
    var bufferized = try bufferize(std.testing.allocator, Model, &model);
    deinitBufferized(std.testing.allocator, Model, &bufferized);
}
