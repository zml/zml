//! Local FP32-to-BF16 linear projection using bounded native TCL tiles.

const std = @import("std");
const zml = @import("../../../zml.zig");
const tcl = zml.kernel.tcl;
const Tensor = zml.Tensor;

// TODO: Move this row tiling and fused bias/conversion into Furiosa XLA
// lowering so ordinary linear projections no longer need this custom kernel.
const rowTile = 512;

pub const Inputs = struct {
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
};

const Config = struct {
    tiles: i64,
    inputWidth: i64,
    outputWidth: i64,
};

/// Projects local FP32 inputs to a BF16 output shape.
/// Requires an explicit manual-computation boundary.
pub fn forward(input: Inputs, output: zml.Shape) Tensor {
    const width = input.x.dim(-1);
    const rows = @divExact(@as(i64, @intCast(input.x.shape().count())), width);
    const tiles = std.math.divCeil(i64, rows, rowTile) catch unreachable;
    const x = input.x.reshape(.{ .row = rows, .k = width })
        .pad(0, .{ .row = Tensor.Pad{ .high = tiles * rowTile - rows } })
        .reshape(.{ .tile = tiles, .row = rowTile, .k = width });
    const projected = Projection.call(.{ .x = x, .weight = input.weight, .bias = input.bias }, .{
        .out = .init(.{ .tile = tiles, .row = rowTile, .n = input.weight.dim(0) }, .bf16),
    }, .{ .cfg = .{ .tiles = tiles, .inputWidth = width, .outputWidth = input.weight.dim(0) } }).out;
    return projected.reshape(.{ .row = tiles * rowTile, .n = input.weight.dim(0) })
        .slice(.row, .{ .end = rows }).reshape(output);
}

const Projection = tcl.Kernel(Config, .{
    .name = "row_tiled_projection",
    .inputs = &.{ "x", "weight", "bias" },
    .outputs = &.{"out"},
    .run = run,
});

fn run(b: *tcl.Builder, cfg: Config) tcl.FinishError!void {
    b.compilerConfig(.{ .hierarchical_tactic_population = true });
    const T = b.axis("Tiles", cfg.tiles);
    const R = b.axis("Rows", rowTile);
    const K = b.axis("K", cfg.inputWidth);
    const N = b.axis("N", cfg.outputWidth);
    const input = try b.declareArgs(.{
        .x = .{ .dtype = .f32, .axes = &.{ T, R, K } },
        .weight = .{ .dtype = .f32, .axes = &.{ N, K } },
        .bias = .{ .dtype = .f32, .axes = &.{N} },
    });

    // Match XLA's FP32 VE lowering: split the output width into powers of two.
    // Keep the entire reduction axis, so row tiling does not split the sum.
    const partCount = @popCount(cfg.outputWidth); //For instance, 5376 = 4096 + 1024 + 256, so partCount = 3
    // Allocate host side TCL tensor descriptors for each output column group weights and results
    const weights = try b.arena.allocator().alloc(@TypeOf(input.weight), partCount);
    const outputs = try b.arena.allocator().alloc(@TypeOf(input.weight), partCount);
    var offset: i64 = 0;
    // Populate weights with descriptors for each part
    for (weights, 0..) |*weight, i| {
        const size = std.math.floorPowerOfTwo(i64, cfg.outputWidth - offset);
        var name: [24]u8 = undefined;
        const part = if (partCount == 1) N else b.axis(std.fmt.bufPrint(&name, "N{d}", .{i}) catch unreachable, size);
        weight.* = if (partCount == 1) input.weight else try b.slice(input.weight, N, offset, &.{ part, K }, .{});
        offset += size;
    }

    var loop = b.openFor(T, .{try b.scratchpad(.bf16, &.{ T, R, N })});
    const row = try b.indexRead(input.x, loop.iv); // reads input.x at index loop.iv
    for (weights, outputs) |weight, *output| {
        // [512, K] × transpose([group_width, K]) --> [512, group_width]
        const op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
        output.* = try op.commit(op.fetch(row, .{}).mulf(op.fetch(weight, .{})).reduce(&.{K}, .addf), .{
            .axes = &.{ R, weight.axes[0] },
        });
    }
    // For instannce: [512,4096] + [512,1024] + [512,256] → [512,5376]
    const projected = if (partCount == 1) outputs[0] else try b.concat(outputs, &.{ R, N }, .{});
    const op = b.tensorOperation(.{ .tactic = .elementwise });
    const y = try op.commit(op.fetch(projected, .{}).addf(op.fetch(input.bias, .{})), .{ .dtype = .bf16 }); // Add bias
    try loop.yield(.{try b.indexWrite(loop.carried[0], loop.iv, y)});
    b.ret(&.{loop.results[0]});
}
