const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const zml = @import("zml");

const smoke = @import("smoke.zig");

const AxisKind = zml.platform.tenstorrent.AxisKind;

const Fixture = struct {
    file: []const u8,
    /// Forces the view through `tt_mesh_shape`; empty takes the plugin's default.
    mesh_shape: []const i64 = &.{},
    /// The view the plugin is expected to serve.
    shape: []const u32,
    kinds: []const AxisKind,
};

const fixtures = [_]Fixture{
    .{ .file = "blackhole_P150.yaml", .shape = &.{1}, .kinds = &.{.singleton} },
    .{ .file = "box_2xp150_capture.yaml", .shape = &.{2}, .kinds = &.{.line} },
    // No power-of-two device count on this platform.
    .{ .file = "3xp150_line.yaml", .shape = &.{3}, .kinds = &.{.line} },
    .{ .file = "blackhole_4xP150.yaml", .shape = &.{4}, .kinds = &.{.ring} },
    .{ .file = "blackhole_8xP150.yaml", .shape = &.{8}, .kinds = &.{.ring} },
    // Two-axis views: coords of rank 2 and link_y. The plugin serves at most
    // two axes (a [2,2,2] view of the 8 cards is UNIMPLEMENTED there).
    .{ .file = "blackhole_4xP150.yaml", .mesh_shape = &.{ 2, 2 }, .shape = &.{ 2, 2 }, .kinds = &.{ .line, .line } },
    .{ .file = "blackhole_8xP150.yaml", .mesh_shape = &.{ 2, 4 }, .shape = &.{ 2, 4 }, .kinds = &.{ .line, .ring } },
};

fn fixturePath(buf: []u8, file: []const u8) ![]const u8 {
    const r = try zml.bazel.runfiles(bazel_builtin.current_repository);
    var name_buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "zml/platforms/tenstorrent/fixtures/{s}", .{file});
    return try r.rlocation(name, buf) orelse error.FileNotFound;
}

test "device-free Tenstorrent platforms of 1, 2, 3, 4 and 8 devices, in one or two axes, run y = x * 2 + 1 like the CPU" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    for (fixtures) |fixture| {
        errdefer std.log.err("fixture {s}, mesh_shape {any}", .{ fixture.file, fixture.mesh_shape });
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const platform: *zml.Platform = try .init(allocator, io, .tenstorrent, .{
            .tenstorrent = .{
                .runtime = .fake,
                .cluster_descriptor = try fixturePath(&path_buf, fixture.file),
                .mesh_shape = fixture.mesh_shape,
            },
        });
        defer platform.deinit(allocator, io);

        try expectView(platform, fixture);
        try std.testing.expectEqualStrings("device", platform.memoryKind(.device));
        try std.testing.expectEqualStrings("pinned_host", platform.memoryKind(.host_pinned));
        for (platform.devices) |device| try std.testing.expect(device.memory(.host_pinned) != null);

        // Buffers take the plugin's default layout: 32x32 tiles of 16x16 faces where shapes allow.
        const tiled = (try platform.defaultMemoryLayout(&.{ 64, 96 }, .f32)).tiled;
        try std.testing.expectEqualSlices(i64, &.{ 32, 32, 16, 16 }, tiled.tile_dims);
        try std.testing.expectEqual(0, (try platform.defaultMemoryLayout(&.{ 5, 7 }, .f32)).tiled.tile_dims.len);

        try smoke.expectMatchesCpu(allocator, io, platform);
        try smoke.expectLoaderRoundTrip(allocator, io, platform);
        try smoke.expectLoaderRoundTripIn(allocator, io, platform, .host_pinned);
    }
}

/// Checks the device attributes against the expected view, and ZML's physical
/// mesh against the pinned convention of DESIGN 7.4: the non-singleton view
/// axes are link_x, link_y, link_z in order, `coords` is the row-major
/// position on them, and device ids follow that order.
fn expectView(platform: *const zml.Platform, fixture: Fixture) !void {
    var device_count: usize = 1;
    for (fixture.shape) |extent| device_count *= extent;
    try std.testing.expectEqual(device_count, platform.devices.len);

    var extents: zml.stdx.BoundedArray(u32, 4) = .empty;
    for (fixture.shape) |extent| if (extent > 1) extents.appendAssumeCapacity(extent);

    for (platform.devices, 0..) |device, i| {
        const info: zml.platform.tenstorrent.DeviceInfo = try .init(platform, device);
        try std.testing.expectEqual(fixture.shape.len, info.view.axes.len);
        for (info.view.axes.constSlice(), fixture.shape, fixture.kinds) |ax, extent, kind| {
            try std.testing.expectEqual(extent, ax.extent);
            try std.testing.expectEqual(kind, ax.kind);
        }
        try std.testing.expect(info.dram_gbps > 0);
        try std.testing.expect(info.ccl_base_us >= 0);

        var expected_coords: [4]u32 = @splat(0);
        var rest = i;
        var axis = extents.len;
        while (axis > 0) {
            axis -= 1;
            expected_coords[axis] = @intCast(rest % extents.get(axis));
            rest /= extents.get(axis);
        }
        try std.testing.expectEqualSlices(u32, expected_coords[0..@max(extents.len, 1)], info.coords.constSlice());
    }

    const mesh = platform.physical_mesh;
    try std.testing.expectEqual(device_count, mesh.countDevices());
    const tags = [_]zml.Sharding.PhysicalAxisTag{ .link_x, .link_y, .link_z };
    var tag_index: usize = 0;
    for (fixture.shape, fixture.kinds) |extent, kind| {
        if (kind == .singleton) continue;
        const tag = tags[tag_index];
        tag_index += 1;
        try std.testing.expectEqual(@as(i64, extent), mesh.axis(tag));
        const geometry: zml.Sharding.AxisGeometry = if (kind == .ring) .{ .ring = .closed_ring } else .{ .ring = .linear };
        try std.testing.expectEqual(geometry, mesh.geometry(tag).?);
    }
    if (tag_index == 0) {
        // A single chip is one link_x device.
        try std.testing.expectEqual(1, mesh.axis(.link_x));
        try std.testing.expectEqual(zml.Sharding.AxisGeometry.point_to_point, mesh.geometry(.link_x).?);
        tag_index = 1;
    }
    for (tags[tag_index..]) |tag| try std.testing.expect(!mesh.hasAxis(tag));
    for (mesh.devices_in_canonical_order, 0..) |device, i| {
        try std.testing.expectEqual(@as(u32, @intCast(i)), device.id);
    }
}

test "every Tenstorrent client option reaches the plugin" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const cluster_descriptor = try fixturePath(&path_buf, "box_2xp150_capture.yaml");

    // The plugin rejects unknown options and wrong types: every name and type is checked.
    var options: zml.platform.CreateOptions.Tenstorrent = .{
        .runtime = .fake,
        .cluster_descriptor = cluster_descriptor,
        .mesh_shape = &.{2},
        .mesh_hint = &.{ 1, 2 },
        .visible_chips = "4411763081529,4411763081535",
        .runtime_root = "/nonexistent/runtime-root",
        .cache_dir = "/nonexistent/cache",
        .logs_dir = "/nonexistent/logs",
        .dram_reserve_bytes = 64 << 20,
        .trace_region_bytes = 0,
        .arena_bytes = 32 << 20,
    };
    {
        const platform: *zml.Platform = try .init(allocator, io, .tenstorrent, .{ .tenstorrent = options });
        defer platform.deinit(allocator, io);
        try std.testing.expectEqual(2, platform.devices.len);
    }

    options.mesh_shape = &.{};
    options.mesh_hint = &.{};
    options.visible_chips = "0x40331924139";
    const platform: *zml.Platform = try .init(allocator, io, .tenstorrent, .{ .tenstorrent = options });
    defer platform.deinit(allocator, io);
    try std.testing.expectEqual(1, platform.devices.len);
    const info: zml.platform.tenstorrent.DeviceInfo = try .init(platform, platform.devices[0]);
    try std.testing.expectEqual(4411763081529, info.board_id);
}

test "a buffer of a type the plugin cannot hold yet is an error, not a crash" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const platform: *zml.Platform = try .init(allocator, io, .tenstorrent, .{
        .tenstorrent = .{ .runtime = .fake, .cluster_descriptor = try fixturePath(&path_buf, "box_2xp150_capture.yaml") },
    });
    defer platform.deinit(allocator, io);

    // The plugin holds f32, bf16, i32, u32, u16, u8, bool and, as 32-bit
    // halves, i64 and u64; it answers the others' layout query with
    // UNIMPLEMENTED.
    const bytes: [64 * 96 * 8]u8 = @splat(0);
    for ([_]zml.DataType{ .f16, .f64, .i16 }) |dtype| {
        const shape = zml.Shape.init(.{ .b = 64, .d = 96 }, dtype);
        try expectUnimplemented(zml.Buffer.fromBytes(io, platform, shape, bytes[0..shape.byteSize()]));
        try expectUnimplemented(zml.Buffer.uninitialized(io, platform, shape, .{}));
    }
}

/// std.testing.expectError would format the Buffer, and with it the whole platform state.
fn expectUnimplemented(result: anytype) !void {
    if (result) |buffer| {
        var b = buffer;
        b.deinit();
        return error.TestExpectedError;
    } else |err| try std.testing.expectEqual(error.Unimplemented, err);
}
