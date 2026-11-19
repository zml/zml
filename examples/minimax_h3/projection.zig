//! Linear projections using bounded native TCL tiles.

const std = @import("std");
const zml = @import("zml");
const tcl = zml.kernel.tcl;
const Tensor = zml.Tensor;

const rowTile = 512;

const Inputs = struct {
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
};

const Config = struct {
    tiles: i64,
    inputWidth: i64,
    outputWidth: i64,
};

/// x[..., K], weight[N, K] and bias[N] are FP32; all dimensions are replicated.
/// Preserve x's leading dimensions and tags, replacing its final width with N.
pub fn forward(x: Tensor, weight: Tensor, bias: Tensor, mesh: *const zml.Mesh) Tensor {
    return zml.ops.manualComputation2(local, .{ .x = x, .weight = weight, .bias = bias }, x.shape().setDim(-1, weight.dim(0)).withDtype(.bf16), mesh, .{
        .in_specs = .{ .x = .replicated, .weight = .replicated, .bias = .replicated },
        .out_specs = .replicated,
    });
}

fn local(input: Inputs, output: zml.Shape) Tensor {
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

/// FP32 x[..., K], weight[N, K] and bias[N], with batch sharded on .model.
/// N must be divisible by 256; leading dimensions and tags are preserved.
/// Shards the batch dimension across the model mesh. Bias and weights are replicated.
pub fn forwardBatchF32(x: Tensor, weight: Tensor, bias: Tensor, mesh: *const zml.Mesh) Tensor {
    return zml.ops.manualComputation2(BatchF32.local, .{ .x = x, .weight = weight, .bias = bias }, x.shape().setDim(-1, weight.dim(0)), mesh, .{
        .in_specs = .{ .x = .parse(.{ .b = .model }), .weight = .replicated, .bias = .replicated },
        .out_specs = .parse(.{ .b = .model }),
    });
}

const BatchF32 = struct {
    const chunkRows = 384;
    const tileRows = 128;
    const tileColumns = 256;
    const BatchConfig = struct { columns: i64, width: i64 };

    fn local(input: Inputs, output: zml.Shape) Tensor {
        const width = input.x.dim(-1);
        const outputWidth = input.weight.dim(0);
        const rows = @divExact(@as(i64, @intCast(input.x.shape().count())), width);
        const padding = @mod(-rows, chunkRows);
        const chunkCount = @divExact(rows + padding, chunkRows);
        const columns = @divExact(outputWidth, tileColumns);
        const x = input.x.reshape(.{ .row = rows, .k = width })
            .pad(0, .{ .row = Tensor.Pad{ .high = padding } });
        const weight = input.weight.reshape(.{ .column = columns, .n = tileColumns, .k = width });
        const bias = input.bias.reshape(.{ .column = columns, .n = tileColumns });

        // Emit fixed calls to one bounded kernel; grouped runtime while loops are unsupported.
        const pieces = zml.Compiler.current().alloc(Tensor, @intCast(chunkCount));
        // Iterate over each chunk of the input batch and apply the projection kernel
        for (pieces, 0..) |*piece, i| {
            const start: i64 = @intCast(i * chunkRows);
            const chunk = x.slice(.row, .{ .start = start, .end = start + chunkRows })
                .reshape(.{ .tile = @divExact(chunkRows, tileRows), .row = tileRows, .k = width });
            const projected = Kernel.call(.{ .x = chunk, .weight = weight, .bias = bias }, .{
                .out = .init(.{ .column = columns, .tile = @divExact(chunkRows, tileRows), .row = tileRows, .n = tileColumns }, .f32),
            }, .{ .cfg = .{ .columns = columns, .width = width } }).out;
            piece.* = projected.transpose(.{ .tile, .row, .column, .n })
                .reshape(.{ .row = chunkRows, .n = outputWidth });
        }
        return Tensor.concatenate(pieces, .row).slice(.row, .{ .end = rows }).reshape(output);
    }

    const Kernel = tcl.Kernel(BatchConfig, .{
        .name = "batch_f32_projection",
        .inputs = &.{ "x", "weight", "bias" },
        .outputs = &.{"out"},
        .run = build,
    });

    fn build(b: *tcl.Builder, cfg: BatchConfig) tcl.FinishError!void {
        b.compilerConfig(.{ .hierarchical_tactic_population = true });
        const C = b.axis("Columns", cfg.columns);
        const T = b.axis("Tiles", @divExact(chunkRows, tileRows));
        const R = b.axis("Rows", tileRows);
        const K = b.axis("K", cfg.width);
        const N = b.axis("N", tileColumns);
        const input = try b.declareArgs(.{
            .x = .{ .dtype = .f32, .axes = &.{ T, R, K } },
            .weight = .{ .dtype = .f32, .axes = &.{ C, N, K } },
            .bias = .{ .dtype = .f32, .axes = &.{ C, N } },
        });

        var columns = b.openFor(C, .{try b.scratchpad(.f32, &.{ C, T, R, N })}); // Move horizontally, 256 columns at a time
        const weight = try b.indexRead(input.weight, columns.iv);
        const bias = try b.indexRead(input.bias, columns.iv);
        var tiles = b.openFor(T, .{try b.scratchpad(.f32, &.{ T, R, N })}); // Moves vertically, 128 rows at a time
        const row = try b.indexRead(input.x, tiles.iv);
        const dot = b.tensorOperation(.{ .tactic = .einsum_by_ve });
        const projected = try dot.commit(dot.fetch(row, .{}).mulf(dot.fetch(weight, .{})).reduce(&.{K}, .addf), .{ .axes = &.{ R, N } });
        const add = b.tensorOperation(.{ .tactic = .elementwise });
        const y = try add.commit(add.fetch(projected, .{}).addf(add.fetch(bias, .{})), .{ .dtype = .f32, .axes = &.{ R, N } });
        try tiles.yield(.{try b.indexWrite(tiles.carried[0], tiles.iv, y)});
        try columns.yield(.{try b.indexWrite(columns.carried[0], columns.iv, tiles.results[0])});
        b.ret(&.{columns.results[0]});
    }
};
