//! FP32 online-softmax attention with bounded query/key tiles.

const std = @import("std");

const zml = @import("../../../zml.zig");
const tcl = zml.kernel.tcl;
const online_softmax = @import("online_softmax.zig");
const Tensor = zml.Tensor;

const queryRows = 32;
const keyRows = 128;
pub const queryChunk = 128;

const Config = struct { batches: i64, keys: i64, heads: i64, dim: i64 };

/// Noncausal FP32 attention on a local {b, q/k, h, hd} shard.
/// The dispatcher requires at least queryChunk queries.
pub fn attention(q: Tensor, k: Tensor, v: Tensor) Tensor {
    const cfg: Config = .{ .batches = q.dim(.b), .keys = k.dim(.k), .heads = q.dim(.h), .dim = q.dim(.hd) };
    const queries = q.dim(.q);
    const count = std.math.divCeil(i64, queries, queryChunk) catch unreachable;
    const keyTiles = std.math.divCeil(i64, cfg.keys, keyRows) catch unreachable;
    const kvShape = .{ .b = cfg.batches, .tile = keyTiles, .key = keyRows, .h = cfg.heads, .hd = cfg.dim };
    const paddedK = k.pad(0, .{ .k = Tensor.Pad{ .high = keyTiles * keyRows - cfg.keys } }).reshape(kvShape);
    const paddedV = v.pad(0, .{ .k = Tensor.Pad{ .high = keyTiles * keyRows - cfg.keys } }).reshape(kvShape);
    const pieces = zml.Compiler.current().alloc(Tensor, @intCast(count));
    for (pieces, 0..) |*piece, i| {
        // Overlap the final window, then discard duplicate query outputs.
        const wanted: i64 = @intCast(i * queryChunk);
        const start = @min(wanted, queries - queryChunk);
        const chunk = q.slice(.q, .{ .start = start, .end = start + queryChunk })
            .reshape(.{ .b = cfg.batches, .tile = @divExact(queryChunk, queryRows), .q = queryRows, .h = cfg.heads, .hd = cfg.dim });
        piece.* = Kernel.call(.{ .q = chunk, .k = paddedK, .v = paddedV }, .{ .out = chunk.shape() }, .{ .cfg = cfg }).out
            .reshape(.{ .b = cfg.batches, .q = queryChunk, .h = cfg.heads, .hd = cfg.dim })
            .slice(.q, .{ .start = wanted - start });
    }
    return Tensor.concatenate(pieces, .q);
}

const Kernel = tcl.Kernel(Config, .{
    .name = "dense_flash_attention_f32",
    .inputs = &.{ "q", "k", "v" },
    .outputs = &.{"out"},
    .run = build,
});

fn build(b: *tcl.Builder, cfg: Config) tcl.FinishError!void {
    b.compilerConfig(.{ .hierarchical_tactic_population = true });
    const B = b.axis("Batch", cfg.batches);
    const T = b.axis("Queries", @divExact(queryChunk, queryRows));
    const I = b.axis("I", queryRows);
    const C = b.axis("Keys", std.math.divCeil(i64, cfg.keys, keyRows) catch unreachable);
    const J = b.axis("J", keyRows);
    const H = b.axis("H", cfg.heads);
    const D = b.axis("D", cfg.dim);
    const input = try b.declareArgs(.{
        .q = .{ .dtype = .f32, .axes = &.{ B, T, I, H, D } },
        .k = .{ .dtype = .f32, .axes = &.{ B, C, J, H, D } },
        .v = .{ .dtype = .f32, .axes = &.{ B, C, J, H, D } },
    });
    const positions = try b.reshape(try b.arange(C.size * J.size, .i32, &.{b.axis("Positions", C.size * J.size)}, .{}), &.{ C, J }, .{});
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(cfg.dim))));
    const sentinel = -std.math.floatMax(f32);

    var batches = b.openFor(B, .{try b.scratchpad(.f32, &.{ B, T, I, H, D })});
    const queries = try b.indexRead(input.q, batches.iv);
    const keys = try b.indexRead(input.k, batches.iv);
    const values = try b.indexRead(input.v, batches.iv);
    var tiles = b.openFor(T, .{try b.scratchpad(.f32, &.{ T, I, H, D })});
    const q = try b.indexRead(queries, tiles.iv);

    var loop = b.openFor(C, .{ try b.full(0, .f32, &.{ I, H }), try b.full(0, .f32, &.{ I, H, D }), try b.full(sentinel, .f32, &.{ I, H }) });
    const sumAcc, const outAcc, const maxAcc = loop.carried;
    const k = try b.indexRead(keys, loop.iv);
    const v = try b.indexRead(values, loop.iv);
    const position = try b.indexRead(positions, loop.iv);
    var op = b.tensorOperation(.{ .tactic = .elementwise });
    const mask = try op.commit(op.fetch(position, .{}).muli(-1).addi(cfg.keys).binary(.maxi, 0).binary(.mini, 1), .{});
    op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
    const scores = try op.commit(op.fetch(q, .{}).mulf(op.fetch(k, .{})).reduce(&.{D}, .addf), .{ .axes = &.{ I, H, J } });
    op = b.tensorOperation(.{ .tactic = .elementwise });
    const scaled = op.fetch(scores, .{}).mulf(scale);
    const masked = try op.commit(op.where(op.fetch(mask, .{}), .ne, 0, scaled, sentinel), .{});
    const update = try online_softmax.update(b, masked, mask, J, sumAcc, maxAcc, .{ .tactic = .elementwise });
    op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
    const outCur = try op.commit(op.fetch(update.weights, .{}).mulf(op.fetch(v, .{})).reduce(&.{J}, .addf), .{ .axes = &.{ I, H, D } });
    const outNew = try update.accumulate(b, outAcc, outCur, .{ .tactic = .elementwise });
    try loop.yield(.{ update.sum, outNew, update.max });

    op = b.tensorOperation(.{ .tactic = .elementwise });
    const result = try op.commit(op.fetch(loop.results[1], .{}).divf(op.fetch(loop.results[0], .{})), .{});
    try tiles.yield(.{try b.indexWrite(tiles.carried[0], tiles.iv, result)});
    try batches.yield(.{try b.indexWrite(batches.carried[0], batches.iv, tiles.results[0])});
    b.ret(&.{batches.results[0]});
}
