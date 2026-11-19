//! Dense noncausal attention: bounded query blocks and a native TCL key loop.
const std = @import("std");

const zml = @import("../../../zml.zig");
const Tensor = zml.Tensor;
const tcl = zml.kernel.tcl;
const online_softmax = @import("online_softmax.zig");

const queryBlock = 32;
const keyBlock = 64;
const maxKeys = 2048;
const maxPairs = 512 * maxKeys;

const Config = struct {
    batch: i64,
    queries: i64,
    keys: i64,
    heads: i64,
    headDim: i64,
    queryBlocks: i64,
    blockRows: i64,
    blocks: i64,
    keyBlocks: i64,
    chunk: i64,
};

/// Inputs are a local shard in {b, q/k, h, hd} order.
pub fn attention(q: Tensor, k: Tensor, v: Tensor) Tensor {
    const batches: i64 = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(q.dim(.b))) catch unreachable);
    // TCC 2026.5.0b0 overflows SPM offsets with 2048 query blocks in one unit.
    const queries = @max(queryBlock, @divTrunc(maxPairs / maxKeys, batches));
    const count: usize = @intCast(std.math.divCeil(i64, q.dim(.q), queries) catch unreachable);
    const outputs = zml.Compiler.current().alloc(Tensor, count);
    for (outputs, 0..) |*output, i| {
        const start = @as(i64, @intCast(i)) * queries;
        output.* = attentionChunk(q.slice(.q, .{ .start = start, .end = @min(start + queries, q.dim(.q)) }), k, v, batches);
    }
    return if (count == 1) outputs[0] else Tensor.concatenate(outputs, .q);
}

fn attentionChunk(q: Tensor, k: Tensor, v: Tensor, batches: i64) Tensor {
    // Power-of-two block counts bound TCC's layout search. Native gathers repeat
    // tail queries; those independent outputs are discarded after the call.
    const blockRows: i64 = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(@min(q.dim(.q), queryBlock))) catch unreachable);
    const queryBlocks: i64 = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(std.math.divCeil(i64, q.dim(.q), blockRows) catch unreachable)) catch unreachable);
    const blocks = batches * queryBlocks;
    const keyBlocks = std.math.divCeil(i64, k.dim(.k), keyBlock) catch unreachable;
    const chunk = @min(keyBlocks * keyBlock, @max(keyBlock, @divTrunc(@min(maxKeys, @divTrunc(maxPairs, blocks * blockRows)), keyBlock) * keyBlock));
    const cfg: Config = .{
        .batch = q.dim(.b),
        .queries = q.dim(.q),
        .keys = k.dim(.k),
        .heads = q.dim(.h),
        .headDim = q.dim(.hd),
        .queryBlocks = queryBlocks,
        .blockRows = blockRows,
        .blocks = blocks,
        .keyBlocks = keyBlocks,
        .chunk = chunk,
    };
    const kvShape = .{ .block = cfg.batch * keyBlocks, .key = keyBlock, .h = cfg.heads, .hd = cfg.headDim };
    const padding = .{ .k = Tensor.Pad{ .high = keyBlocks * keyBlock - cfg.keys } };
    const out = Flash.call(.{
        .q = q.reshape(.{ .t = cfg.batch * cfg.queries, .h = cfg.heads, .g = 1, .hd = cfg.headDim }),
        .k = k.pad(0, padding).reshape(kvShape),
        .v = v.pad(0, padding).reshape(kvShape),
    }, .{ .out = .init(.{ .t = blocks * blockRows, .h = cfg.heads, .g = 1, .hd = cfg.headDim }, .bf16) }, .{ .cfg = cfg }).out;
    return out.reshape(.{ .b = batches, .q = queryBlocks * blockRows, .h = cfg.heads, .hd = cfg.headDim })
        .slice(.b, .{ .end = cfg.batch }).slice(.q, .{ .end = cfg.queries });
}

const Flash = tcl.Kernel(Config, .{
    .name = "dense_flash_attention",
    .inputs = &.{ "q", "k", "v" },
    .outputs = &.{"out"},
    .run = run,
});

fn run(b: *tcl.Builder, cfg: Config) tcl.FinishError!void {
    // Match the existing TCL prefill schedule without importing paged policies.
    b.compilerConfig(.{
        .enable_einsum_fusion = true,
        .padding_policy = .{ .Small = 1.1 },
        .instruction_mem_budget = 0xB0000,
        .reshape_einsum_mode = .{ .Reshape = .{ .permute = false } },
        .tensor_unit_bridge_threshold_in_page = 12,
        .allow_external_operators = false,
        .scheduler_beam_search = true,
        .allow_reduce_by_ve_cluster_chip_reduce = false,
        .allow_reduce_by_ve_cluster_chip_reduce_base_population = false,
        .tactic_hint = .{ .DefaultWithMaxNumUniqueShapes = @as(i64, if (cfg.heads * cfg.headDim >= 1024) 16 else 8) },
        .dma_preference = 1.2,
    });
    const T = b.axis("T", cfg.batch * cfg.queries);
    const B = b.axis("B", cfg.blocks);
    const I = b.axis("I", cfg.blockRows);
    const H = b.axis("H", cfg.heads);
    const G = b.axis("G", 1);
    const D = b.axis("D", cfg.headDim);
    const N = b.axis("N", cfg.batch * cfg.keyBlocks);
    const L = b.axis("L", keyBlock);
    const C = b.axis("C", std.math.divCeil(i64, cfg.keys, cfg.chunk) catch unreachable);
    const J = b.axis("J", @divExact(cfg.chunk, keyBlock));
    const K = b.axis("K", cfg.chunk);
    const inputs = try b.declareArgs(.{
        .q = .{ .dtype = .bf16, .axes = &.{ T, H, G, D } },
        .k = .{ .dtype = .bf16, .axes = &.{ N, L, H, D } },
        .v = .{ .dtype = .bf16, .axes = &.{ N, L, H, D } },
    });

    const blocks = try b.arange(cfg.blocks, .i32, &.{B}, .{});
    const within = try b.arange(cfg.blockRows, .i32, &.{I}, .{});
    var op = b.tensorOperation(.{});
    const batch = try op.commit(op.fetch(blocks, .{}).binary(.shr_arith, std.math.log2_int(u64, @intCast(cfg.queryBlocks))).binary(.mini, cfg.batch - 1), .{});
    op = b.tensorOperation(.{});
    const firstQuery = try op.commit(op.fetch(blocks, .{}).binary(.bit_andi, cfg.queryBlocks - 1).muli(cfg.blockRows), .{});
    op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
    const row = try op.commit(op.fetch(firstQuery, .{}).addi(op.fetch(within, .{})), .{});
    op = b.tensorOperation(.{});
    const queryBase = try op.commit(op.fetch(batch, .{}).muli(cfg.queries), .{});
    op = b.tensorOperation(.{ .tactic = .elementwise });
    const queryIndices = try op.commit(op.fetch(row, .{}).binary(.mini, cfg.queries - 1).addi(op.fetch(queryBase, .{})), .{});
    const q = if (T.size == B.size * I.size)
        try b.reshape(inputs.q, &.{ B, I, H, G, D }, .{})
    else
        try b.gather(inputs.q, queryIndices, T, &.{ B, I, H, G, D }, .{});

    // K/V are contiguous 64-key blocks. Compute addresses once, outside the
    // native loop; no page table or sequence metadata is supplied by the caller.
    const keyIndices = try b.reshape(try b.arange(C.size * J.size, .i32, &.{b.axis("KeyBlocks", C.size * J.size)}, .{}), &.{ C, J }, .{});
    op = b.tensorOperation(.{});
    const keyBase = try op.commit(op.fetch(batch, .{}).muli(cfg.keyBlocks), .{});
    op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
    const blockIndices = try op.commit(op.fetch(keyIndices, .{}).binary(.mini, cfg.keyBlocks - 1).addi(op.fetch(keyBase, .{})), .{ .axes = &.{ C, B, J } });
    const positions = try b.reshape(try b.arange(C.size * K.size, .i32, &.{b.axis("Keys", C.size * K.size)}, .{}), &.{ C, K }, .{});
    const stats = &.{ H, G, B, I };
    const outAxes = &.{ B, I, H, G, D };
    const scoreMin = -std.math.floatMax(f32);
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(cfg.headDim))));
    var loop = b.openFor(C, .{ try b.full(0, .f32, stats), try b.full(0, .f32, outAxes), try b.full(scoreMin, .f32, stats) });
    {
        const sumAcc, const outAcc, const maxAcc = loop.carried;
        const indices = try b.indexRead(blockIndices, loop.iv);
        const position = try b.indexRead(positions, loop.iv);
        // A single K/V block is shared by every query block. Broadcast it in
        // the contractions instead of asking TCC to compile an identity gather.
        const k = if (N.size == 1)
            try b.reshape(inputs.k, &.{ K, H, D }, .{})
        else
            try b.reshape(try b.gather(inputs.k, indices, N, &.{ B, J, L, H, D }, .{}), &.{ B, K, H, D }, .{});
        const v = if (N.size == 1)
            try b.reshape(inputs.v, &.{ K, H, D }, .{})
        else
            try b.reshape(try b.gather(inputs.v, indices, N, &.{ B, J, L, H, D }, .{}), &.{ B, K, H, D }, .{});
        op = b.tensorOperation(.{});
        const mask = try op.commit(op.fetch(position, .{}).muli(-1).addi(cfg.keys).binary(.maxi, 0).binary(.mini, 1), .{});
        op = b.tensorOperation(.{});
        const scores = try op.commit(op.contract(q, k, &.{ H, G, B, I, K }).mulf(scale), .{});
        op = b.tensorOperation(.{});
        const s = op.fetch(scores, .{});
        const masked = try op.commit(op.where(op.fetch(mask, .{}), .ne, 0, s, scoreMin), .{});
        const update = try online_softmax.update(b, masked, mask, K, sumAcc, maxAcc, .{});
        op = b.tensorOperation(.{});
        // Materialize the BF16 conversion through the vector engine before PV.
        const p = try op.commit(op.fetch(update.weights, .{}).mulf(1.0), .{ .dtype = .bf16 });
        op = b.tensorOperation(.{});
        const outCur = try op.commit(op.contract(p, v, outAxes), .{});
        const outNew = try update.accumulate(b, outAcc, outCur, .{});
        try loop.yield(.{ update.sum, outNew, update.max });
    }
    op = b.tensorOperation(.{});
    const sum = op.fetch(loop.results[0], .{});
    const denominator = try op.commit(op.where(sum, .gt, 0, sum, 1.0), .{});
    op = b.tensorOperation(.{});
    const output = try op.commit(op.fetch(loop.results[1], .{}).divf(op.fetch(denominator, .{})), .{ .dtype = .bf16 });
    b.ret(&.{try b.reshape(output, &.{ b.axis("TOut", cfg.blocks * cfg.blockRows), H, G, D }, .{})});
}
