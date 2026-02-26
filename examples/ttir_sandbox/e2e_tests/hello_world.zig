const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const ttir_compile_sandbox = @import("../ttir_compile_sandbox.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const cfg = struct {
    const M = 256;
    const N = 256;
    const K = 256;
    const BLOCK_M = 128;
    const BLOCK_N = 128;
    const BLOCK_K = 32;
};

pub fn wrappedHelloWorld(kernel_ttir: [:0]const u8, a: Tensor, b: Tensor, out: Tensor) Tensor {
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

    return zml.ops.triton(.{ a, b }, .{out.shape()}, .{
        .name = "matmul_fixed_kernel",
        .ir = kernel_ttir,
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

    const kernel_params =
        "{" ++
        "\"M\":256," ++
        "\"N\":256," ++
        "\"K\":256," ++
        "\"BLOCK_M\":128," ++
        "\"BLOCK_N\":128," ++
        "\"BLOCK_K\":32," ++
        "\"num_warps\":8" ++
        "}";

    const kernel_ttir = try ttir_compile_sandbox.getHelloWorldMatmulTtir(allocator, io, kernel_params);
    defer allocator.free(kernel_ttir);

    const a_shape: zml.Tensor = .init(.{ cfg.M, cfg.K }, .f32);
    const b_shape: zml.Tensor = .init(.{ cfg.K, cfg.N }, .f32);
    const c_shape: zml.Tensor = .init(.{ cfg.M, cfg.N }, .f32);

    var exe = try platform.compileFn(allocator, io, wrappedHelloWorld, .{
        kernel_ttir,
        a_shape,
        b_shape,
        c_shape,
    });
    defer exe.deinit();

    var a = try zeroBuffer(allocator, io, platform, a_shape.shape());
    defer a.deinit();
    var b = try zeroBuffer(allocator, io, platform, b_shape.shape());
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

    exe_args.set(.{ a, b, c });
    exe.call(exe_args, &exe_results);

    var result: zml.Buffer = exe_results.get(zml.Buffer);
    defer result.deinit();

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
}

fn zeroBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    @memset(slice.data(), 0);
    return zml.Buffer.fromSlice(io, platform, slice);
}
