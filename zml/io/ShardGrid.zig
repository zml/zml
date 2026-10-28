//! Where a destination's logical bytes land. Its devices hold the boxes of an
//! even grid over the tensor (`Sharding.Placement` splits every axis into
//! equal shards), each box packed row-major on its devices, and devices
//! holding the same box are one writer set. Box order and logical order agree
//! on a box's elements, so a contiguous logical range lands as one contiguous
//! range of every box it touches: ranges are placed by arithmetic, never by
//! listing rows.
//!
//! For a [4, 8] tensor whose columns are split in two, each letter one
//! element:
//!
//!     Logical bytes              Box 0       Box 1
//!     a b c d e f g h            a b c d     e f g h
//!     i j k l m n o p            i j k l     m n o p
//!     q r s t u v w x            q r s t     u v w x
//!     y z A B C D E F            y z A B     C D E F
//!
//! The logical range from c to n lands as c..l on box 0 and e..n on box 1.
//! Runs are the aligned pieces that stay in one box (`run`: a shard of the
//! innermost split axis, four elements here); a worker walks them to place
//! a range's bytes, the planner only counts them.

const std = @import("std");

const constants = @import("../constants.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");

pub const InitError = error{IncompatibleSharding};

const ShardGrid = @This();

const max_rank = constants.MAX_RANK;
/// Every box has at least one device, and a writer mask holds 64.
pub const max_boxes = 64;

rank: usize,
/// Elements per axis, of the tensor and of one box.
dims: [max_rank]u64 = @splat(1),
shard: [max_rank]u64 = @splat(1),
/// Byte strides, logical and inside a box.
strides: [max_rank]u64 = @splat(0),
box_strides: [max_rank]u64 = @splat(0),
/// A box's index is row-major over the shard counts of the axes.
index_strides: [max_rank]u64 = @splat(0),
element: u64,
total: u64,
box_bytes: u64,
/// Aligned runs of `run` logical bytes each land in one box, contiguous.
run: u64,
box_count: usize,
/// The canonical device indices holding each box.
writers: [max_boxes]u64 = @splat(0),

/// `shape`'s sharding must name a mesh (`Platform.placedShape`): its
/// devices, in canonical order, are the writers the masks index.
pub fn init(shape: Shape) InitError!ShardGrid {
    const placement: Sharding.Placement = try .init(&shape);
    const local = placement.localShape(shape);
    const devices = placement.mesh().?.devicesInCanonicalOrder();
    std.debug.assert(devices.len <= max_boxes);
    const element: u64 = shape.dtype().sizeOf();
    var grid: ShardGrid = .{
        .rank = shape.rank(),
        .element = element,
        .total = shape.byteSize(),
        .box_bytes = local.byteSize(),
        .run = shape.byteSize(),
        .box_count = 1,
    };
    const strides = shape.computeByteStrides();
    var box_stride = element;
    var index_stride: u64 = 1;
    var axis = grid.rank;
    while (axis > 0) {
        axis -= 1;
        const dim: u64 = @intCast(shape.dim(axis));
        const shard: u64 = @intCast(local.dim(axis));
        grid.dims[axis] = dim;
        grid.shard[axis] = shard;
        grid.strides[axis] = @intCast(strides.get(axis));
        grid.box_strides[axis] = box_stride;
        grid.index_strides[axis] = index_stride;
        box_stride *= shard;
        const count = dim / shard;
        index_stride *= count;
        // The innermost split axis cuts the runs.
        if (count > 1 and grid.run == grid.total) grid.run = shard * grid.strides[axis];
    }
    if (index_stride > max_boxes) return error.IncompatibleSharding;
    grid.box_count = @intCast(index_stride);
    for (0..devices.len) |writer| {
        const slices = placement.slices(writer, shape);
        var box: u64 = 0;
        for (slices.constSlice(), 0..) |slice, i| box += @as(u64, @intCast(slice.start)) / grid.shard[i] * grid.index_strides[i];
        grid.writers[@intCast(box)] |= @as(u64, 1) << @intCast(writer);
    }
    for (grid.writers[0..grid.box_count]) |mask| {
        if (mask == 0) return error.IncompatibleSharding;
    }
    return grid;
}

fn coordinate(self: *const ShardGrid, offset: u64, axis: usize) u64 {
    return offset / self.strides[axis] % self.dims[axis];
}

/// The box holding logical byte `offset`.
pub fn boxOf(self: *const ShardGrid, offset: u64) usize {
    var box: u64 = 0;
    for (0..self.rank) |i| box += self.coordinate(offset, i) / self.shard[i] * self.index_strides[i];
    return @intCast(box);
}

/// Where logical byte `offset` lands inside its box.
pub fn boxOffset(self: *const ShardGrid, offset: u64) u64 {
    var at = offset % self.element;
    for (0..self.rank) |i| at += self.coordinate(offset, i) % self.shard[i] * self.box_strides[i];
    return at;
}

/// The bytes of box `box` before logical offset `offset`: a logical range
/// `[start, end)` lands on that box at `[rankIn(start), rankIn(end))`.
/// Counted axis by axis, as in a lexicographic comparison.
pub fn rankIn(self: *const ShardGrid, box: usize, offset: u64) u64 {
    if (offset >= self.total) return self.box_bytes;
    var before: u64 = 0;
    for (0..self.rank) |i| {
        const c = self.coordinate(offset, i);
        const first = @as(u64, box) / self.index_strides[i] % (self.dims[i] / self.shard[i]) * self.shard[i];
        before += @min(c -| first, self.shard[i]) * self.box_strides[i];
        if (c < first or c >= first + self.shard[i]) return before;
    }
    return before + offset % self.element;
}

/// The end of the run holding logical byte `offset`, at most `limit`.
pub fn runEnd(self: *const ShardGrid, offset: u64, limit: u64) u64 {
    return @min(limit, (offset / self.run + 1) * self.run);
}

/// The runs with a byte in `[start, end)`, `start < end`.
pub fn runCount(self: *const ShardGrid, start: u64, end: u64) u64 {
    return (end + self.run - 1) / self.run - start / self.run;
}

/// Devices per box: the grid is even, so every box has as many.
pub fn replicas(self: *const ShardGrid) usize {
    return @popCount(self.writers[0]);
}

const testing = struct {
    const platform_mod = @import("../platform.zig");
    const CreateOptions = platform_mod.CreateOptions;
    const Device = platform_mod.Device;
    const Platform = platform_mod.Platform;
    const Target = platform_mod.Target;

    const Scenario = struct {
        name: []const u8,
        device_count: u32,
        physical_mesh: CreateOptions.PhysicalMesh = .auto,
        shape: Shape,
        logical_mesh: Sharding.LogicalMesh,
        strategy: Sharding.Strategy,
        /// Logical ranges are cut every `step` bytes, unaligned to rows.
        step: u64,
        packed_storage: bool = false,
    };

    fn buildMesh2x2(allocator: std.mem.Allocator, target: Target, devices: []const Device) !Sharding.PhysicalMesh {
        if (devices.len < 4) return error.NotEnoughDevices;
        return .fromTree(allocator, target, .axis(.link_x, .{ .mesh = .torus }, &.{
            .axis(.link_y, .{ .mesh = .torus }, &.{ .device(devices[0]), .device(devices[1]) }),
            .axis(.link_y, .{ .mesh = .torus }, &.{ .device(devices[2]), .device(devices[3]) }),
        }));
    }

    fn buildMesh2x2x2(allocator: std.mem.Allocator, target: Target, devices: []const Device) !Sharding.PhysicalMesh {
        if (devices.len < 8) return error.NotEnoughDevices;
        return .fromTree(allocator, target, .axis(.link_x, .{ .mesh = .torus }, &.{
            .axis(.link_y, .{ .mesh = .torus }, &.{
                .axis(.link_z, .{ .mesh = .torus }, &.{ .device(devices[0]), .device(devices[1]) }),
                .axis(.link_z, .{ .mesh = .torus }, &.{ .device(devices[2]), .device(devices[3]) }),
            }),
            .axis(.link_y, .{ .mesh = .torus }, &.{
                .axis(.link_z, .{ .mesh = .torus }, &.{ .device(devices[4]), .device(devices[5]) }),
                .axis(.link_z, .{ .mesh = .torus }, &.{ .device(devices[6]), .device(devices[7]) }),
            }),
        }));
    }

    fn run(scenario: Scenario, partitioning: anytype) !void {
        const allocator = std.testing.allocator;
        const io = std.testing.io;
        var platform = Platform.init(allocator, io, .cpu, .{
            .physical_mesh = scenario.physical_mesh,
            .cpu = .{ .device_count = scenario.device_count },
        }) catch return error.SkipZigTest;
        defer platform.deinit(allocator, io);
        const mesh: Sharding.Mesh = try .init(scenario.name, &platform.physical_mesh, scenario.logical_mesh, scenario.strategy);
        const partitioned = scenario.shape.withPartitioning(&mesh, partitioning);
        try expectPlaces(allocator, if (scenario.packed_storage) partitioned.packedShape() else partitioned, scenario.step);
    }

    /// Walks the runs of every `step`-byte range into the writers' boxes and
    /// compares with each device's slices, element by element; checks that
    /// `rankIn` bounds the range's bytes on each box and that `runCount`
    /// counts the runs.
    fn expectPlaces(allocator: std.mem.Allocator, shape: Shape, step: u64) !void {
        const grid: ShardGrid = try .init(shape);
        const placement: Sharding.Placement = try .init(&shape);
        const devices = placement.mesh().?.devicesInCanonicalOrder();
        const box_bytes: usize = @intCast(grid.box_bytes);
        const total: usize = @intCast(grid.total);
        const source = try allocator.alloc(u8, total);
        defer allocator.free(source);
        for (source, 0..) |*byte, i| byte.* = @truncate(i *% 131 +% 17);

        const expected = try allocator.alloc(u8, devices.len * box_bytes);
        defer allocator.free(expected);
        const element: usize = @intCast(grid.element);
        for (0..devices.len) |writer| {
            const slices = placement.slices(writer, shape);
            var index: [max_rank]u64 = @splat(0);
            var cursor: usize = 0;
            while (cursor < box_bytes) : (cursor += element) {
                var logical: u64 = 0;
                for (0..grid.rank) |i| logical += (@as(u64, @intCast(slices.get(i).start)) + index[i]) * grid.strides[i];
                @memcpy(expected[writer * box_bytes + cursor ..][0..element], source[@intCast(logical)..][0..element]);
                var axis = grid.rank;
                while (axis > 0) {
                    axis -= 1;
                    index[axis] += 1;
                    if (index[axis] < grid.shard[axis]) break;
                    index[axis] = 0;
                }
            }
        }

        const actual = try allocator.alloc(u8, expected.len);
        defer allocator.free(actual);
        @memset(actual, 0);
        var start: u64 = 0;
        while (start < grid.total) : (start += step) {
            const end = @min(grid.total, start + step);
            var low: [max_boxes]u64 = @splat(std.math.maxInt(u64));
            var high: [max_boxes]u64 = @splat(0);
            var runs: u64 = 0;
            var at = start;
            while (at < end) : (runs += 1) {
                const stop = grid.runEnd(at, end);
                const box = grid.boxOf(at);
                const landing = grid.boxOffset(at);
                low[box] = @min(low[box], landing);
                high[box] = @max(high[box], landing + stop - at);
                var mask = grid.writers[box];
                while (mask != 0) : (mask &= mask - 1) {
                    const writer: usize = @ctz(mask);
                    @memcpy(actual[writer * box_bytes + @as(usize, @intCast(landing)) ..][0..@intCast(stop - at)], source[@intCast(at)..@intCast(stop)]);
                }
                at = stop;
            }
            try std.testing.expectEqual(runs, grid.runCount(start, end));
            for (0..grid.box_count) |box| {
                const first = grid.rankIn(box, start);
                const last = grid.rankIn(box, end);
                if (high[box] == 0) {
                    try std.testing.expectEqual(first, last);
                } else {
                    try std.testing.expectEqual(low[box], first);
                    try std.testing.expectEqual(high[box], last);
                }
            }
        }
        try std.testing.expectEqualSlices(u8, expected, actual);
    }
};

test "a grid places replicated tensors, across range boundaries" {
    try testing.run(.{
        .name = "replicated",
        .device_count = 4,
        .shape = Shape.init(.{ .rows = 9, .cols = 257 }, .f32),
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
        .step = 773,
    }, .{ .rows = .replicated, .cols = .replicated });
}

test "a grid places packed sub-byte storage" {
    const logical = Shape.init(.{ .rows = 9, .cols = 256 }, .u2);
    try testing.run(.{
        .name = "packed_u2",
        .device_count = 4,
        .shape = logical,
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
        .step = 131,
        .packed_storage = true,
    }, .{ .rows = .replicated, .cols = .x });
}

test "a grid places 1D mirrored and folded sharding" {
    try testing.run(.{
        .name = "mirrored_1d",
        .device_count = 4,
        .physical_mesh = .{ .custom = testing.buildMesh2x2 },
        .shape = Shape.init(.{ .rows = 7, .model = 1024 }, .f32),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .model = .link_x }),
        .step = 2053,
    }, .{ .rows = .replicated, .model = .model });
    try testing.run(.{
        .name = "folded_1d",
        .device_count = 4,
        .physical_mesh = .{ .custom = testing.buildMesh2x2 },
        .shape = Shape.init(.{ .model = 4096 }, .f32),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_y });
            break :blk strategy;
        },
        .step = 3001,
    }, .{ .model = .model });
}

test "a grid places 2D and 3D sharding" {
    try testing.run(.{
        .name = "batch_model_2d",
        .device_count = 4,
        .physical_mesh = .{ .custom = testing.buildMesh2x2 },
        .shape = Shape.init(.{ .batch = 8, .model = 1024 }, .f32),
        .logical_mesh = .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .batch = .link_x, .model = .link_y }),
        .step = 4093,
    }, .{ .batch = .batch, .model = .model });
    try testing.run(.{
        .name = "folded_model_3d",
        .device_count = 8,
        .physical_mesh = .{ .custom = testing.buildMesh2x2x2 },
        .shape = Shape.init(.{ .batch = 16, .model = 4096 }, .f32),
        .logical_mesh = .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_z });
            break :blk strategy;
        },
        .step = 8191,
    }, .{ .batch = .replicated, .model = .model });
    // Three axes, the middle and the last split: runs are a quarter row.
    try testing.run(.{
        .name = "middle_and_last_3d",
        .device_count = 4,
        .physical_mesh = .{ .custom = testing.buildMesh2x2 },
        .shape = Shape.init(.{ .a = 3, .b = 4, .c = 8 }, .u16),
        .logical_mesh = .mesh(.{ .x = .low_bandwidth, .y = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x, .y = .link_y }),
        .step = 37,
    }, .{ .a = .replicated, .b = .x, .c = .y });
}

test "a grid places a scalar on every device" {
    try testing.run(.{
        .name = "scalar",
        .device_count = 4,
        .shape = Shape.init(.{}, .f32),
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
        .step = 3,
    }, .{});
}
