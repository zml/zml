//! Shared helpers for the JAX-equivalent interface previews.
//! This file intentionally has no Bazel target.

const std = @import("std");

const preview = @import("../interface_preview/common.zig");
const zml = @import("zml");

pub const Job = preview.Job;
pub const dataSharding = preview.dataSharding;

pub fn hostGpuSharding(
    platform: *zml.Platform,
) !zml.Sharding {
    return platform.registerShardingWithStrategy(
        "host-gpu",
        .mesh(.{
            .host = .low_bandwidth,
            .gpu = .high_bandwidth,
        }),
        .parseBindings(.{
            .host = .{.network},
            .gpu = .{.link},
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
                shard.globalSlices(),
                shard.shape(),
                local.items(f32),
            },
        );
    }
}
