const std = @import("std");
const zml = @import("zml");
const stdx = @import("stdx");
const asynk = @import("async");
const builtin = @import("builtin");

// Set log level to info to see the output. Use .debug to see the generated IR.
pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

const log = std.log.scoped(.@"examples/collectives");

const AllReduceModel = struct {
    pub fn forward(x: zml.Tensor) zml.Tensor {
        const mesh = x.mesh();
        const reduced = zml.ops.allReduce(x, .x, mesh, 1);

        const replicated = reduced.withSharding(.{});
        return replicated.add(zml.Tensor.constant(replicated.shape(), replicated.dtype().zero()));
    }
};

const AllGatherModel = struct {
    pub fn forward(x_sharded: zml.Tensor) zml.Tensor {
        const mesh = x_sharded.mesh();
        const gathered = zml.ops.allGather(x_sharded, .m, mesh);
        const replicated = gathered.withSharding(.{});
        return replicated.add(zml.Tensor.constant(replicated.shape(), replicated.dtype().zero()));
    }
};

const ReduceScatterModel = struct {
    pub fn forward(x_replicated: zml.Tensor) zml.Tensor {
        const mesh = x_replicated.mesh();
        return zml.ops.reduceScatter(x_replicated, .x, mesh);
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{ .cpu = .{ .cpu_device_count = 8 } }).withCompilationOptions(.{
        .sharding_enabled = true,
        .xla_dump_to = "/tmp/zml/collectives",
    });

    const num_devices = platform.getDevices().len;
    if (num_devices < 2) {
        log.warn("This example requires at least 2 devices. Skipping actual run.", .{});
        return;
    }
    context.printAvailablePlatforms(platform);

    log.info("Running collectives example on {d} devices", .{num_devices});
    const mesh = zml.Mesh.init(.{ .x = num_devices });

    {
        log.info("--- Running AllReduce ---", .{});
        const shape = zml.Shape.init(.{ .m = 8, .n = num_devices }, .i32).withPartitioning(.{ .n = .x });
        const mod = try zml.compileFn(allocator, AllReduceModel.forward, .{shape}, mesh, platform);
        defer mod.deinit();

        const input_data = try zml.slice.arange(allocator, shape, .{});
        defer allocator.free(input_data);
        const input_buffer = try zml.Buffer.from(platform, .init(mesh, shape), input_data, .{});
        defer input_buffer.deinit();

        const result_buffer = mod.call(.{input_buffer});
        defer result_buffer.deinit();

        const result_data = try result_buffer.toHost(allocator);
        defer allocator.free(result_data);
        log.info("AllReduce Result (first 8 elements): {any}", .{result_data[0..@min(result_data.len, 8)]});
    }

    {
        log.info("--- Running AllGather ---", .{});
        const global_shape = zml.Shape.init(.{ .m = 8 * num_devices, .n = 2 }, .i32);
        const input_shape = global_shape.withPartitioning(.{ .m = .x });

        const mod = try zml.compileFn(allocator, AllGatherModel.forward, .{input_shape}, mesh, platform);
        defer mod.deinit();

        const input_data = try zml.slice.arange(allocator, global_shape, .{});
        defer allocator.free(input_data);
        const input_buffer = try zml.Buffer.from(platform, .init(mesh, input_shape), input_data, .{});
        defer input_buffer.deinit();

        const result_buffer = mod.call(.{input_buffer});
        defer result_buffer.deinit();

        const result_data = try result_buffer.toHost(allocator);
        defer allocator.free(result_data);
        log.info("AllGather Result (first 8 elements): {any}", .{result_data[0..@min(result_data.len, 8)]});
    }

    {
        log.info("--- Running ReduceScatter ---", .{});
        const input_shape = zml.Shape.init(.{ .m = 8 * num_devices, .n = 2 }, .i32).withReplicatedPartitioning();
        const mod = try zml.compileFn(allocator, ReduceScatterModel.forward, .{input_shape}, mesh, platform);
        defer mod.deinit();

        const input_data = try zml.slice.arange(allocator, input_shape, .{});
        defer allocator.free(input_data);
        const input_buffer = try zml.Buffer.from(platform, .init(mesh, input_shape), input_data, .{});
        defer input_buffer.deinit();

        const result_buffer = mod.call(.{input_buffer});
        defer result_buffer.deinit();

        const result_data = try result_buffer.toHost(allocator);
        defer allocator.free(result_data);
        log.info("ReduceScatter Result (first 8 elements): {any}", .{result_data[0..@min(result_data.len, 8)]});
    }
}
