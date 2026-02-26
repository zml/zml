const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const testing = zml.testing;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const inputs_bytes = @embedFile("safetensors/hello_world_inputs.safetensors");
const outputs_bytes = @embedFile("safetensors/hello_world_output.safetensors");

const cfg = struct {
    const M = 256;
    const N = 256;
    const K = 256;
    const BLOCK_M = 128;
    const BLOCK_N = 128;
};

pub fn wrappedHelloWorld(a: Tensor, b: Tensor, out: Tensor) Tensor {
    const grid: [3]i32 = .{
        @intCast(@divExact(cfg.M, cfg.BLOCK_M)),
        @intCast(@divExact(cfg.N, cfg.BLOCK_N)),
        1,
    };

    const target = zml.module.CompilationContext.current().platform.target;
    const num_warps: i32 = switch (target) {
        .rocm => 8,
        .cuda => 8,
        else => 8,
    };

    return zml.ops.triton(.{
        a,
        b,
    }, .{out.shape()}, .{
        .name = "matmul_fixed_kernel",
        .ir = @embedFile("hello_world_kernel.ttir"),
        .grid = grid,
        .num_stages = 1,
        .num_warps = num_warps,
        .debug = true,
        .output_operand_aliases = &.{},
    })[0];
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    log.info("\n{f}", .{platform.fmtVerbose()});

    if (platform.target != .cuda and platform.target != .rocm) {
        log.err("ttir_test requires CUDA/ROCm target, got {s}", .{@tagName(platform.target)});
        return;
    }

    const a_shape: zml.Tensor = .init(.{ cfg.M, cfg.K }, .f32);
    const b_shape: zml.Tensor = .init(.{ cfg.K, cfg.N }, .f32);
    const c_shape: zml.Tensor = .init(.{ cfg.M, cfg.N }, .f32);

    var exe = try platform.compileFn(allocator, io, wrappedHelloWorld, .{
        a_shape,
        b_shape,
        c_shape,
    });
    defer exe.deinit();

    const inputs_path = try writeEmbeddedSafetensors(allocator, io, inputs_bytes, "hello_world_inputs.safetensors");
    defer allocator.free(inputs_path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, inputs_path);
    defer registry.deinit();

    var a = try loadBufferFromRegistry(allocator, io, platform, &registry, "a");
    defer a.deinit();
    var b = try loadBufferFromRegistry(allocator, io, platform, &registry, "b");
    defer b.deinit();
    var c = try zeroBuffer(allocator, io, platform, c_shape.shape());
    defer c.deinit();

    log.info("a shape: {f}, device: {s}", .{ a.shape(), @tagName(platform.target) });
    log.info("b shape: {f}, device: {s}", .{ b.shape(), @tagName(platform.target) });
    log.info("c shape: {f}, device: {s}", .{ c.shape(), @tagName(platform.target) });

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{
        a,
        b,
        c,
    });

    exe.call(exe_args, &exe_results);
    var result: zml.Buffer = exe_results.get(zml.Buffer);
    defer result.deinit();

    const expected_path = try writeEmbeddedSafetensors(allocator, io, outputs_bytes, "hello_world_output.safetensors");
    defer allocator.free(expected_path);
    var expected_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, expected_path);
    defer expected_registry.deinit();
    var expected = try loadBufferFromRegistry(allocator, io, platform, &expected_registry, "c");
    defer expected.deinit();

    var output = try result.toSliceAlloc(allocator, io);
    defer output.free(allocator);

    const output_items = output.constItems(f32);
    if (output_items.len == 0) {
        std.debug.print("c[0:10,0:10] after matmul: <empty>\n", .{});
        return;
    }

    const rows: usize = @intCast(result.shape().dim(0));
    const cols: usize = @intCast(result.shape().dim(1));
    const rmax = @min(rows, 10);
    const cmax = @min(cols, 10);

    std.debug.print("c[0:10,0:10] after matmul:\n", .{});
    for (0..rmax) |r| {
        const row_start = r * cols;
        for (0..cmax) |col| {
            const v = output_items[row_start + col];
            if (col == 0) {
                std.debug.print("{d:.5}", .{v});
            } else {
                std.debug.print(" {d:.5}", .{v});
            }
        }
        std.debug.print("\n", .{});
    }

    var matches = true;
    testing.expectClose(io, result, expected, .{}) catch {
        matches = false;
    };
    std.debug.print("\n\n", .{});
    if (matches) {
        std.debug.print("Output matches expected tensor\n", .{});
    } else {
        std.debug.print("Output does not match expected tensor\n", .{});
    }
}

fn writeEmbeddedSafetensors(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8, filename: []const u8) ![]const u8 {
    const path = filename;
    const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer file.close(io);

    var writer = file.writer(io, &.{});
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();

    var real_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const real_len = try file.realPath(io, &real_buf);
    return try allocator.dupe(u8, real_buf[0..real_len]);
}

fn loadBufferFromRegistry(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    registry: *zml.safetensors.TensorRegistry,
    key: []const u8,
) !zml.Buffer {
    const tensor_desc = registry.tensors.get(key) orelse return error.NotFound;
    const shape = tensor_desc.shape;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try registry.reader(io, key, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, host_bytes);
}

fn zeroBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    @memset(slice.data(), 0);
    return zml.Buffer.fromSlice(io, platform, slice);
}
