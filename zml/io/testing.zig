//! Test support for the loader: platforms with a given device count, real
//! safetensors files and per-shard read-back. Only tests import this.
const std = @import("std");

const Buffer = @import("../buffer.zig").Buffer;
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const zml_testing = @import("../testing.zig");
const TensorStore = @import("TensorStore.zig");

/// `device_count` devices for a test: the accelerator the tests run on
/// (`zml.testing.env`, shared and never freed) when it has that many,
/// otherwise a CPU platform with exactly that many, freed by `deinit`.
/// `physical` lays the first `device_count` devices on one axis, so a mesh
/// over it (`mesh`) places on exactly those, whatever the platform has.
/// Must not move after `init`: meshes point at `physical`.
/// `initCpu` always creates the CPU platform, for tests of code that places
/// buffers on the platform's own mesh.
pub const Devices = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    owned: ?*Platform,
    physical: Sharding.PhysicalMesh,

    pub fn init(self: *Devices, allocator: std.mem.Allocator, io: std.Io, device_count: usize) !void {
        return self.initWith(allocator, io, device_count, false);
    }

    pub fn initCpu(self: *Devices, allocator: std.mem.Allocator, io: std.Io, device_count: usize) !void {
        return self.initWith(allocator, io, device_count, true);
    }

    fn initWith(self: *Devices, allocator: std.mem.Allocator, io: std.Io, device_count: usize, force_cpu: bool) !void {
        const shared = zml_testing.env();
        const owned: ?*Platform = if (!force_cpu and shared.target != .cpu and shared.devices.len >= device_count)
            null
        else
            try Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = @intCast(device_count) } });
        errdefer if (owned) |platform| platform.deinit(allocator, io);
        const platform: *const Platform = if (owned) |p| p else shared;
        const nodes = try allocator.alloc(Sharding.PhysicalNode, device_count);
        defer allocator.free(nodes);
        for (nodes, platform.devices[0..device_count]) |*node, device| node.* = .device(device);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .owned = owned,
            .physical = try .fromTree(allocator, platform.target, .axis(.bus, .tree, nodes)),
        };
    }

    pub fn deinit(self: *Devices) void {
        self.physical.deinit(self.allocator);
        if (self.owned) |platform| platform.deinit(self.allocator, self.io);
        self.* = undefined;
    }

    /// A mesh named `name` whose one axis, `model`, spans the devices.
    pub fn mesh(self: *const Devices, name: []const u8) !Sharding.Mesh {
        return .init(name, &self.physical, .mesh(.{ .model = .high_bandwidth }), .parseBindings(.{ .model = .bus }));
    }
};

pub const TensorBytes = struct {
    name: []const u8,
    /// The safetensors dtype name, such as "U16".
    dtype: []const u8,
    dims: []const i64,
    bytes: []const u8,
};

/// Writes `tensors` as the safetensors file `sub_path` of `dir`, in order,
/// and registers them from it.
pub fn writeSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    sub_path: []const u8,
    registry: *safetensors.TensorRegistry,
    tensors: []const TensorBytes,
) !void {
    var header: std.Io.Writer.Allocating = .init(allocator);
    defer header.deinit();
    const writer = &header.writer;
    try writer.writeByte('{');
    var offset: usize = 0;
    for (tensors, 0..) |tensor, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ tensor.name, tensor.dtype });
        for (tensor.dims, 0..) |dim, axis| try writer.print("{s}{d}", .{ if (axis != 0) "," else "", dim });
        try writer.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + tensor.bytes.len });
        offset += tensor.bytes.len;
    }
    try writer.writeByte('}');

    const file = try dir.createFile(io, sub_path, .{ .read = true });
    defer file.close(io);
    var header_len: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_len, header.written().len, .little);
    try file.writePositionalAll(io, &header_len, 0);
    var position: u64 = header_len.len;
    try file.writePositionalAll(io, header.written(), position);
    position += header.written().len;
    for (tensors) |tensor| {
        try file.writePositionalAll(io, tensor.bytes, position);
        position += tensor.bytes.len;
    }
    try safetensors.parseSafetensors(allocator, io, registry, file, "");
}

/// A store over a copy of `registry`, for a test that builds several stores
/// over one checkpoint: a store owns its registry.
pub fn storeOf(allocator: std.mem.Allocator, registry: *const safetensors.TensorRegistry, meshes: []const *const Sharding.Mesh) !TensorStore {
    var copy: safetensors.TensorRegistry = .init(allocator);
    errdefer copy.deinit();
    for (registry.tensors.values()) |tensor| try copy.registerTensor(tensor);
    return .init(allocator, copy, .{ .meshes = meshes });
}

/// Distinct, nonzero bytes for test tensor number `seed`.
pub fn pattern(bytes: []u8, seed: u8) void {
    for (bytes, 0..) |*byte, index| byte.* = (@as(u8, @truncate(index *% 29)) +% seed *% 53) | 1;
}

/// Reads every shard of the rank-2 `buffer` back, in its mesh's device
/// order, and checks it holds that device's slice of `global`, the whole
/// tensor's row-major bytes. Independent of `Buffer.toSliceAlloc`, which
/// assumes the platform's own device order.
pub fn expectShards(allocator: std.mem.Allocator, io: std.Io, buffer: *const Buffer, global: []const u8) !void {
    const shape = buffer._platform.placedShape(buffer.shape());
    try std.testing.expectEqual(@as(usize, 2), shape.rank());
    try std.testing.expectEqual(shape.byteSize(), global.len);
    const placement: Sharding.Placement = try .init(shape);
    const devices = placement.mesh().?.devicesInCanonicalOrder();
    try std.testing.expectEqual(devices.len, buffer._shards.len);
    const shard_bytes = placement.shape.byteSize();
    const actual = try allocator.alloc(u8, shard_bytes);
    defer allocator.free(actual);
    const expected = try allocator.alloc(u8, shard_bytes);
    defer allocator.free(expected);
    const element_bytes: usize = shape.dtype().sizeOf();
    const row_bytes: usize = @as(usize, @intCast(shape.dim(1))) * element_bytes;
    var shards = buffer.shards();
    for (devices) |device| {
        try shards.next().?.toHost(io, actual);
        const slices = placement.slices(device.coords);
        const rows = slices.get(0);
        const columns = slices.get(1);
        const len = @as(usize, @intCast(columns.size)) * element_bytes;
        const column_offset = @as(usize, @intCast(columns.start)) * element_bytes;
        var cursor: usize = 0;
        for (@intCast(rows.start)..@intCast(rows.start + rows.size)) |row| {
            @memcpy(expected[cursor..][0..len], global[row * row_bytes + column_offset ..][0..len]);
            cursor += len;
        }
        try std.testing.expectEqualSlices(u8, expected, actual);
    }
}
