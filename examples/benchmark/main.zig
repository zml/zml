//! Run with: bazel run --config=release //examples/benchmark
const std = @import("std");
const builtin = @import("builtin");
const zml = @import("zml");
const Sharding = zml.Sharding;
const pjrt = zml.pjrt;

comptime {
    _ = zml.bazel;
}

const CliArgs = struct {
    pub const help = "benchmark --iterations=20000 --samples=7 --warmup=2000";
    iterations: usize = 20_000,
    samples: usize = 7,
    warmup: usize = 2_000,
};

const layouts = [_]Layout{
    .{ .name = "cpu-1", .target = .cpu, .sizes = &.{1} },
    .{ .name = "cuda-8", .target = .cuda, .sizes = &.{8} },
    .{ .name = "cuda-2x2", .target = .cuda, .sizes = &.{ 2, 2 } },
    .{ .name = "rocm-2x2x2", .target = .rocm, .sizes = &.{ 2, 2, 2 } },
    .{ .name = "neuron-2x2x2x2", .target = .neuron, .sizes = &.{ 2, 2, 2, 2 } },
    .{ .name = "cuda-4x2x2x2-reversed", .target = .cuda, .sizes = &.{ 4, 2, 2, 2 }, .reverse_ids = true },
};

const expected = [_]Expected{
    .{ .layout = "cpu-1", .case = "replicated", .offsets = &.{0} },
    .{ .layout = "cpu-1", .case = "partial-model", .offsets = &.{0} },
    .{ .layout = "cpu-1", .case = "all-axes", .offsets = &.{0} },
    .{ .layout = "cpu-1", .case = "folded-model", .offsets = &.{0} },
    .{ .layout = "cpu-1", .case = "auto-model-data", .offsets = &.{0} },
    .{ .layout = "cuda-8", .case = "replicated", .offsets = &.{ 0, 0, 0, 0, 0, 0, 0, 0 } },
    .{ .layout = "cuda-8", .case = "partial-model", .offsets = &.{ 0, 430080, 860160, 1290240, 1720320, 2150400, 2580480, 3010560 } },
    .{ .layout = "cuda-8", .case = "all-axes", .offsets = &.{ 0, 430080, 860160, 1290240, 1720320, 2150400, 2580480, 3010560 } },
    .{ .layout = "cuda-8", .case = "folded-model", .offsets = &.{ 0, 430080, 860160, 1290240, 1720320, 2150400, 2580480, 3010560 } },
    .{ .layout = "cuda-8", .case = "auto-model-data", .offsets = &.{ 0, 430080, 860160, 1290240, 1720320, 2150400, 2580480, 3010560 } },
    .{ .layout = "cuda-2x2", .case = "replicated", .offsets = &.{ 0, 0, 0, 0 } },
    .{ .layout = "cuda-2x2", .case = "partial-model", .offsets = &.{ 0, 0, 1720320, 1720320 } },
    .{ .layout = "cuda-2x2", .case = "all-axes", .offsets = &.{ 0, 860160, 1720320, 2580480 } },
    .{ .layout = "cuda-2x2", .case = "folded-model", .offsets = &.{ 0, 860160, 1720320, 2580480 } },
    .{ .layout = "cuda-2x2", .case = "auto-model-data", .offsets = &.{ 0, 107520, 1720320, 1827840 } },
    .{ .layout = "cuda-2x2", .case = "reordered-2d", .offsets = &.{ 0, 1720320, 107520, 1827840 } },
    .{ .layout = "rocm-2x2x2", .case = "replicated", .offsets = &.{ 0, 0, 0, 0, 0, 0, 0, 0 } },
    .{ .layout = "rocm-2x2x2", .case = "partial-model", .offsets = &.{ 0, 0, 0, 0, 1720320, 1720320, 1720320, 1720320 } },
    .{ .layout = "rocm-2x2x2", .case = "all-axes", .offsets = &.{ 0, 430080, 860160, 1290240, 1720320, 2150400, 2580480, 3010560 } },
    .{ .layout = "rocm-2x2x2", .case = "folded-model", .offsets = &.{ 0, 430080, 860160, 1290240, 1720320, 2150400, 2580480, 3010560 } },
    .{ .layout = "rocm-2x2x2", .case = "auto-model-data", .offsets = &.{ 0, 107520, 0, 107520, 1720320, 1827840, 1720320, 1827840 } },
    .{ .layout = "rocm-2x2x2", .case = "reordered-2d", .offsets = &.{ 0, 0, 1720320, 1720320, 107520, 107520, 1827840, 1827840 } },
    .{ .layout = "rocm-2x2x2", .case = "logical-3d", .offsets = &.{ 0, 4480, 107520, 112000, 1720320, 1724800, 1827840, 1832320 } },
    .{ .layout = "neuron-2x2x2x2", .case = "replicated", .offsets = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    .{ .layout = "neuron-2x2x2x2", .case = "partial-model", .offsets = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 1720320, 1720320, 1720320, 1720320, 1720320, 1720320, 1720320, 1720320 } },
    .{ .layout = "neuron-2x2x2x2", .case = "all-axes", .offsets = &.{ 0, 215040, 430080, 645120, 860160, 1075200, 1290240, 1505280, 1720320, 1935360, 2150400, 2365440, 2580480, 2795520, 3010560, 3225600 } },
    .{ .layout = "neuron-2x2x2x2", .case = "folded-model", .offsets = &.{ 0, 215040, 430080, 645120, 860160, 1075200, 1290240, 1505280, 1720320, 1935360, 2150400, 2365440, 2580480, 2795520, 3010560, 3225600 } },
    .{ .layout = "neuron-2x2x2x2", .case = "auto-model-data", .offsets = &.{ 0, 0, 107520, 107520, 0, 0, 107520, 107520, 1720320, 1720320, 1827840, 1827840, 1720320, 1720320, 1827840, 1827840 } },
    .{ .layout = "neuron-2x2x2x2", .case = "reordered-2d", .offsets = &.{ 0, 0, 0, 0, 1720320, 1720320, 1720320, 1720320, 107520, 107520, 107520, 107520, 1827840, 1827840, 1827840, 1827840 } },
    .{ .layout = "neuron-2x2x2x2", .case = "logical-3d", .offsets = &.{ 0, 0, 4480, 4480, 107520, 107520, 112000, 112000, 1720320, 1720320, 1724800, 1724800, 1827840, 1827840, 1832320, 1832320 } },
    .{ .layout = "neuron-2x2x2x2", .case = "logical-4d", .offsets = &.{ 0, 112, 4480, 4592, 107520, 107632, 112000, 112112, 1720320, 1720432, 1724800, 1724912, 1827840, 1827952, 1832320, 1832432 } },
    .{ .layout = "cuda-4x2x2x2-reversed", .case = "replicated", .offsets = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    .{ .layout = "cuda-4x2x2x2-reversed", .case = "partial-model", .offsets = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 860160, 860160, 860160, 860160, 860160, 860160, 860160, 860160, 1720320, 1720320, 1720320, 1720320, 1720320, 1720320, 1720320, 1720320, 2580480, 2580480, 2580480, 2580480, 2580480, 2580480, 2580480, 2580480 } },
    .{ .layout = "cuda-4x2x2x2-reversed", .case = "auto-model-data", .offsets = &.{ 0, 107520, 0, 107520, 0, 107520, 0, 107520, 860160, 967680, 860160, 967680, 860160, 967680, 860160, 967680, 1720320, 1827840, 1720320, 1827840, 1720320, 1827840, 1720320, 1827840, 2580480, 2688000, 2580480, 2688000, 2580480, 2688000, 2580480, 2688000 } },
    .{ .layout = "cuda-4x2x2x2-reversed", .case = "reordered-2d", .offsets = &.{ 0, 0, 0, 0, 1720320, 1720320, 1720320, 1720320, 53760, 53760, 53760, 53760, 1774080, 1774080, 1774080, 1774080, 107520, 107520, 107520, 107520, 1827840, 1827840, 1827840, 1827840, 161280, 161280, 161280, 161280, 1881600, 1881600, 1881600, 1881600 } },
    .{ .layout = "cuda-4x2x2x2-reversed", .case = "logical-3d", .offsets = &.{ 0, 0, 4480, 4480, 107520, 107520, 112000, 112000, 860160, 860160, 864640, 864640, 967680, 967680, 972160, 972160, 1720320, 1720320, 1724800, 1724800, 1827840, 1827840, 1832320, 1832320, 2580480, 2580480, 2584960, 2584960, 2688000, 2688000, 2692480, 2692480 } },
    .{ .layout = "cuda-4x2x2x2-reversed", .case = "logical-4d", .offsets = &.{ 0, 112, 4480, 4592, 107520, 107632, 112000, 112112, 860160, 860272, 864640, 864752, 967680, 967792, 972160, 972272, 1720320, 1720432, 1724800, 1724912, 1827840, 1827952, 1832320, 1832432, 2580480, 2580592, 2584960, 2585072, 2688000, 2688112, 2692480, 2692592 } },
};

pub fn main(init: std.process.Init) !void {
    const args = zml.stdx.flags.parse(init.minimal.args, CliArgs);
    if (args.iterations == 0 or args.samples == 0) return error.InvalidBenchmarkCount;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    var buffers: std.ArrayList(zml.Buffer) = .empty;
    try buffers.ensureTotalCapacity(init.gpa, @max(args.iterations, args.warmup));
    defer buffers.deinit(init.gpa);

    const data = try allocator.alloc(u8, (2 * 8) * (3 * 8) * (5 * 8) * (7 * 8) * @sizeOf(f32));
    @memset(data, 0);
    std.log.info("Buffer.from metadata only; mode={s}; iterations={d}; samples={d}; warmup={d}", .{ @tagName(builtin.mode), args.iterations, args.samples, args.warmup });
    std.log.info("Synthetic platforms; no PJRT, transfers, events or waits. Setup and pointer checks excluded. Compiler barriers and buffer storage included.", .{});
    std.log.info("layout,devices,mesh,avg_ns_per_call,min_sample_ns,max_sample_ns,avg_ns_per_device", .{});
    const iterations: u64 = @intCast(args.iterations);
    const calls_per_case = iterations * @as(u64, @intCast(args.samples));
    var total: u64 = 0;
    var count: u64 = 0;
    var device_count: u64 = 0;
    for (layouts) |layout| {
        const platform = try syntheticPlatform(allocator, layout);
        const cases = try casesFor(allocator, platform, layout.sizes.len);
        for (cases) |case| {
            try batch(platform, case.shape, data, args.warmup, &buffers);
            try checkBuffers(buffers.items, data, layout.name, case.name);
            buffers.clearRetainingCapacity();
            var sum: u64 = 0;
            var min: u64 = std.math.maxInt(u64);
            var max: u64 = 0;
            for (0..args.samples) |_| {
                const start: std.Io.Timestamp = .now(init.io, .awake);
                try batch(platform, case.shape, data, args.iterations, &buffers);
                const ns: u64 = @intCast(start.untilNow(init.io, .awake).toNanoseconds());
                try checkBuffers(buffers.items, data, layout.name, case.name);
                buffers.clearRetainingCapacity();
                sum += ns;
                min = @min(min, ns);
                max = @max(max, ns);
            }
            const avg = sum / calls_per_case;
            std.log.info("{s},{d},{s},{d},{d},{d},{d}", .{ layout.name, platform.devices.len, case.name, avg, min / iterations, max / iterations, sum / (calls_per_case * @as(u64, @intCast(platform.devices.len))) });
            total += sum;
            count += 1;
            device_count += calls_per_case * @as(u64, @intCast(platform.devices.len));
        }
    }
    std.log.info("Equal-weight mean across {d} cases: {d} ns/call", .{ count, total / (count * calls_per_case) });
    std.log.info("Overall mean: {d} ns/device (total elapsed / total device preparations)", .{total / device_count});
}

// Mirrors Buffer.from and emptyShell. Keep this in sync with zml/buffer.zig.
// Synthetic handles are never dereferenced or passed to PJRT, including deinit.
// Compiler barriers stand in for the external call consuming each upload's args.
noinline fn bufferFromMetadata(platform: *const zml.Platform, shape: zml.Shape, data: []const u8, opts: zml.Buffer.FromOptions) !zml.Buffer {
    var buffer: zml.Buffer = .{ ._platform = platform, ._shape = shape, ._shards = .empty };
    const packed_shape = shape.packedShape();
    var placed_shape = packed_shape;
    if (placed_shape._sharding.mesh == null) placed_shape._sharding.mesh = platform.replicated_mesh;
    const placement = try Sharding.Placement.init(placed_shape);
    const ty = zml.pjrtx.bufferTypeFromDtype(packed_shape.dtype());
    const layout = platform.defaultMemoryLayout(placement.shape.dims(), packed_shape.dtype());
    zml.stdx.debug.assert(platform.devices[0].memory(opts.memory) != null, "Device doesn't have {} memory", .{opts.memory});
    const slice = zml.Slice.init(packed_shape, data);

    for (platform.physical_mesh.devices_in_canonical_order) |device| {
        const memory = platform.devices[device.id].memory(opts.memory).?;
        const args: pjrt.Client.BufferFromHostBufferArgs = .{
            .data = placement.shardPtr(device.coords, slice),
            .dst = .{ .memory = memory.pjrt_memory },
            .layout = layout,
            .dims = placement.shape.dims(),
            .buffer_type = ty,
            .byte_strides = slice.byte_strides.constSlice(),
            .host_buffer_semantics = .ImmutableUntilTransferCompletes,
        };
        std.mem.doNotOptimizeAway(&args);
        // Smuggle the "data" pointer passed to BufferFromHostBufferArgs into the result Buffer object,
        // to allow checking the correctness of the operation `checkBuffers`
        buffer._shards.appendAssumeCapacity(@ptrCast(@constCast(args.data)));
    }
    return buffer;
}

const Layout = struct {
    name: []const u8,
    target: zml.Target,
    sizes: []const usize,
    reverse_ids: bool = false,
};

const physical_tags = [_]Sharding.PhysicalAxisTag{ .link_x, .link_y, .link_z, .bus };

fn physicalTree(allocator: std.mem.Allocator, sizes: []const usize, depth: usize, next_id: *u32) !Sharding.PhysicalNode {
    if (depth == sizes.len) {
        defer next_id.* += 1;
        // Let PhysicalMesh.fromTree assign coordinates from the tree path.
        return .{ .leaf = .{ .id = next_id.*, .coords = @splat(0xff) } };
    }
    const children = try allocator.alloc(Sharding.PhysicalNode, sizes[depth]);
    for (children) |*child| child.* = try physicalTree(allocator, sizes, depth + 1, next_id);
    return .axis(physical_tags[depth], .{ .mesh = .torus }, children);
}

fn syntheticPlatform(allocator: std.mem.Allocator, layout: Layout) !*zml.Platform {
    const platform = try allocator.create(zml.Platform);
    var next_id: u32 = 0;
    const root = try physicalTree(allocator, layout.sizes, 0, &next_id);
    // Only metadata fields are initialized. Runtime/client fields must never be used.
    platform.* = undefined;
    platform.target = layout.target;
    platform.physical_mesh = try .fromTree(allocator, layout.target, root);
    if (layout.reverse_ids) {
        for (platform.physical_mesh.devices_in_canonical_order) |*device| device.id = next_id - 1 - device.id;
    }
    const devices = try allocator.alloc(zml.platform.Device, next_id);
    const memories = try allocator.alloc(zml.Memory, next_id);
    for (devices, memories, 0..) |*device, *memory, i| {
        memory.* = .{
            .pjrt_memory = @ptrFromInt((i + 1) * 4096),
            .platform = platform,
            .addressable_by_devices = &.{},
        };
        device.* = undefined;
        device.platform = platform;
        device.memory_by_kind = .initFill(memory);
    }
    platform.devices = devices;
    platform.memories = memories;
    const replicated = try allocator.create(zml.Mesh);
    replicated.* = try .init("replicated", &platform.physical_mesh, .mesh(.{ .all = .high_bandwidth }), .parseBindings(.{ .all = .link_x }));
    platform.replicated_mesh = replicated;
    return platform;
}

const Case = struct {
    name: []const u8,
    shape: zml.Shape,
};

fn addCase(allocator: std.mem.Allocator, cases: *std.ArrayList(Case), platform: *const zml.Platform, name: []const u8, logical: Sharding.LogicalMesh, strategy: Sharding.Strategy, partitioning: anytype) !void {
    const mesh = try allocator.create(zml.Mesh);
    mesh.* = try .init(name, &platform.physical_mesh, logical, strategy);
    try cases.append(allocator, .{
        .name = name,
        .shape = zml.Shape.init(.{ .m = 2 * 8, .n = 3 * 8, .e = 5 * 8, .p = 7 * 8 }, .f32).withPartitioning(mesh, partitioning),
    });
}

fn casesFor(allocator: std.mem.Allocator, platform: *const zml.Platform, rank: usize) ![]const Case {
    var cases: std.ArrayList(Case) = .empty;
    try cases.append(allocator, .{ .name = "replicated", .shape = .init(.{ .m = 2 * 8, .n = 3 * 8, .e = 5 * 8, .p = 7 * 8 }, .f32) });
    const model: Sharding.LogicalMesh = .mesh(.{ .model = .high_bandwidth });
    try addCase(allocator, &cases, platform, "partial-model", model, .parseBindings(.{ .model = .link_x }), .{ .m = .model });

    // All-axis partitioning of m = 16 requires at most 16 devices.
    if (platform.devices.len <= 16) {
        var all: Sharding.Strategy = .init;
        for (physical_tags[0..rank]) |tag| all.addBinding(.model, tag);
        try addCase(allocator, &cases, platform, "all-axes", model, all, .{ .m = .model });
        var folded: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
        folded.addFold(.link_x, physical_tags[0..rank]);
        try addCase(allocator, &cases, platform, "folded-model", model, folded, .{ .m = .model });
    }

    const two: Sharding.LogicalMesh = .mesh(.{ .model = .high_bandwidth, .data = .low_bandwidth });
    try addCase(allocator, &cases, platform, "auto-model-data", two, .suggest(two, &platform.physical_mesh), .{ .m = .model, .n = .data });
    if (rank >= 2) {
        try addCase(allocator, &cases, platform, "reordered-2d", two, .parseBindings(.{ .model = .link_y, .data = .link_x }), .{ .m = .model, .n = .data });
    }
    if (rank >= 3) {
        try addCase(allocator, &cases, platform, "logical-3d", .mesh(.{ .model = .high_bandwidth, .data = .low_bandwidth, .experts = .balanced }), .parseBindings(.{ .model = .link_x, .data = .link_y, .experts = .link_z }), .{ .m = .model, .n = .data, .e = .experts });
    }
    if (rank == 4) {
        try addCase(allocator, &cases, platform, "logical-4d", .mesh(.{ .model = .high_bandwidth, .data = .low_bandwidth, .experts = .balanced, .pipeline = .low_bandwidth }), .parseBindings(.{ .model = .link_x, .data = .link_y, .experts = .link_z, .pipeline = .bus }), .{ .m = .model, .n = .data, .e = .experts, .p = .pipeline });
    }
    return cases.items;
}

fn batch(platform: *const zml.Platform, shape: zml.Shape, data: []const u8, iterations: usize, out: *std.ArrayList(zml.Buffer)) !void {
    for (0..iterations) |_| {
        const buffer = try bufferFromMetadata(platform, shape, data, .{ .wait = false });
        out.appendAssumeCapacity(buffer);
    }
}

// Byte offsets into main's 16 x 24 x 40 x 56 f32 allocation, in canonical
// device order (also for the layout with reversed device IDs).
const Expected = struct {
    layout: []const u8,
    case: []const u8,
    offsets: []const usize,
};

pub fn checkBuffers(buffers: []const zml.Buffer, data: []const u8, layout: []const u8, case: []const u8) !void {
    const offsets = for (expected) |entry| {
        if (std.mem.eql(u8, entry.layout, layout) and std.mem.eql(u8, entry.case, case)) break entry.offsets;
    } else return error.MissingExpectedOffsets;

    // These shards smuggle host pointers; never call PJRT methods or Buffer.deinit.
    for (buffers) |buffer| {
        if (buffer.numShards() != offsets.len or buffer.numShards() != buffer._platform.devices.len) return error.InvalidShardCount;
        for (buffer._shards.constSlice(), offsets, 0..) |shard, offset, deviceIndex| {
            if (offset >= data.len) return error.InvalidExpectedOffset;
            const actual = @intFromPtr(shard);
            const want = @intFromPtr(data.ptr + offset);
            if (actual != want) {
                std.log.err("{s}/{s}: canonical device {d}, expected data + {d} ({x}), got {x}", .{ layout, case, deviceIndex, offset, want, actual });
                return error.InvalidShardPointer;
            }
        }
    }
}
