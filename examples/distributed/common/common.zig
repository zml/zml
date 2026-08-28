const std = @import("std");

const zml = @import("zml");

pub const Job = struct {
    coordinator_address: std.Io.net.IpAddress,
    process_index: usize,
    process_count: usize,
    namespace: []const u8,

    pub fn parse(init: std.process.Init) !Job {
        var iterator = init.minimal.args.iterate();
        _ = iterator.next();

        const coordinator = iterator.next() orelse return usage();
        const process_index = iterator.next() orelse return usage();
        const process_count = iterator.next() orelse return usage();
        const namespace = iterator.next() orelse return usage();
        if (iterator.next() != null) return usage();

        const rank = try std.fmt.parseInt(usize, process_index, 10);
        const world = try std.fmt.parseInt(usize, process_count, 10);
        if (world == 0 or rank >= world or namespace.len == 0) {
            return error.InvalidDistributedJob;
        }

        return .{
            .coordinator_address = try .parseLiteral(coordinator),
            .process_index = rank,
            .process_count = world,
            .namespace = namespace,
        };
    }

    pub fn openPlatform(
        self: Job,
        allocator: std.mem.Allocator,
        io: std.Io,
    ) !*zml.Platform {
        return zml.Platform.init(allocator, io, .cuda, .{
            .distributed = .{
                .coordinator_address = self.coordinator_address,
                .process_index = self.process_index,
                .process_count = self.process_count,
                .namespace = self.namespace,
                .local_device_ids = &.{ 0, 1 },
            },
            .xla_gpu = .{
                .allocator = .{
                    .bfc = .{ .preallocate = false },
                },
            },
        });
    }
};

pub fn dataSharding(platform: *zml.Platform) !zml.Sharding {
    return platform.registerShardingWithStrategy(
        "data",
        .mesh(.{ .data = .low_bandwidth }),
        .parseBindings(.{
            .data = .{ .network, .link },
        }),
    );
}

pub fn hostGpuSharding(platform: *zml.Platform) !zml.Sharding {
    return platform.registerShardingWithStrategy(
        "host-gpu",
        .mesh(.{
            .host = .low_bandwidth,
            .gpu = .high_bandwidth,
        }),
        .parseBindings(.{
            .host = .network,
            .gpu = .link,
        }),
    );
}

pub fn hybridSharding(platform: *zml.Platform) !zml.Sharding {
    return platform.registerShardingWithStrategy(
        "data-model",
        .mesh(.{
            .data = .low_bandwidth,
            .model = .high_bandwidth,
        }),
        .parseBindings(.{
            .data = .network,
            .model = .link,
        }),
    );
}

pub fn allocateValues(
    allocator: std.mem.Allocator,
    shape: zml.Shape,
    mode: enum { ones, sequence },
) !zml.Slice {
    const result = try zml.Slice.alloc(allocator, shape);
    for (result.items(f32), 0..) |*value, index| {
        value.* = switch (mode) {
            .ones => 1,
            .sequence => @floatFromInt(index),
        };
    }
    return result;
}

pub fn expectTopology(
    platform: *const zml.Platform,
    global: usize,
    local: usize,
) !void {
    if (platform.globalDevices().len != global or
        platform.addressableDevices().len != local)
    {
        return error.UnexpectedTopology;
    }
}

pub fn expectShardCounts(
    buffer: *const zml.Buffer,
    global: usize,
    local: usize,
) !void {
    if (buffer.numGlobalShards() != @as(u32, @intCast(global)) or
        buffer.numShards() != @as(u32, @intCast(local)))
    {
        return error.UnexpectedShardCount;
    }
}

pub fn expectAddressable(
    platform: *const zml.Platform,
    buffer: *const zml.Buffer,
) !void {
    var shards = buffer.shards();
    while (shards.next()) |shard| {
        for (platform.addressableDevices()) |device| {
            if (device.id() == shard.globalDeviceId()) break;
        } else return error.RemoteBufferShard;
    }
}

pub fn printLocalShards(
    allocator: std.mem.Allocator,
    io: std.Io,
    buffer: *const zml.Buffer,
    label: []const u8,
) !void {
    var shards = buffer.shards();
    while (shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        std.debug.print(
            "{s}: device={d} slices={any} shape={f} values={any}\n",
            .{
                label,
                shard.globalDeviceId(),
                shard.globalSlices().constSlice(),
                shard.shape(),
                local.constItems(f32),
            },
        );
    }
}

fn usage() error{InvalidArguments} {
    std.debug.print(
        \\Usage: EXAMPLE COORDINATOR RANK PROCESS_COUNT NAMESPACE
        \\
        \\Example:
        \\  EXAMPLE 100.80.27.10:8910 0 2 zml-run-001
        \\
    , .{});
    return error.InvalidArguments;
}
