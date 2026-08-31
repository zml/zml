const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.metal_donate);

pub const std_options: std.Options = .{ .log_level = .info };

fn donateAdd(x: zml.Tensor) zml.Tensor {
    return x.addConstant(1).reuseBuffer(x);
}

fn copyAdd(x: zml.Tensor) zml.Tensor {
    return x.addConstant(1);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(gpa, io, .{});
    defer platform.deinit(gpa, io);
    log.info("{f}", .{platform});

    try run(gpa, io, platform, donateAdd, "donate_add");
    try run(gpa, io, platform, copyAdd, "copy_add");
}

fn run(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    name: []const u8,
) !void {
    const shape = zml.Shape.init(.{ .n = 4 }, .f32);
    const vals = [_]f32{ 1, 2, 3, 4 };

    var exe = try platform.compileFn(gpa, io, func, .{
        zml.Tensor.fromShape(shape),
    }, .{ .program_name = name });
    defer exe.deinit();

    var src = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(&vals));
    var args = try exe.args(gpa);
    defer args.deinit(gpa);
    var results = try exe.results(gpa);
    defer results.deinit(gpa);
    args.set(.{src});
    exe.callOpts(io, args, &results, .{ .wait = true });

    const api = src._platform.pjrt_api;
    const shard = src._shards.get(0);
    const donated = shard.isDeleted(api);
    log.info("{s} input deleted={any}", .{ name, donated });
    if (shard.opaqueDeviceMemoryDataPointer(api)) |ptr| {
        log.info("{s} OpaqueDeviceMemoryDataPointer 0x{x}", .{ name, @intFromPtr(ptr) });
    } else |err| {
        log.err("{s} OpaqueDeviceMemoryDataPointer {t}", .{ name, err });
    }
    if (shard.increaseExternalReferenceCount(api)) |_| {
        shard.decreaseExternalReferenceCount(api) catch {};
    } else |err| {
        log.err("{s} IncreaseExternalReferenceCount {t}", .{ name, err });
    }

    const in_host = src.toSliceAlloc(gpa, io) catch |err| {
        log.err("{s} donated-input ToHostBuffer {t}", .{ name, err });
        if (!donated) src.deinit();
        var out = results.get(zml.Buffer);
        out.deinit();
        return err;
    };
    defer in_host.free(gpa);
    if (!donated) src.deinit();

    var out = results.get(zml.Buffer);
    defer out.deinit();
    const host = try out.toSliceAlloc(gpa, io);
    defer host.free(gpa);
    const got = host.items(f32);
    log.info("{s} {d} {d} {d} {d} want 2 3 4 5", .{ name, got[0], got[1], got[2], got[3] });
    if (got[0] != 2 or got[1] != 3 or got[2] != 4 or got[3] != 5) return error.DonateMismatch;
}
