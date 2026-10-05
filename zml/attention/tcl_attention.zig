//! Furiosa RNGD paged attention in TCL, after Furiosa's production kernels
//! (`furiosa.kernels.common.attention`, `exaone_moe/.../flash_attention`).
//!
//! Queries are cut in blocks of rows of one sequence (one token per block for
//! decode). A context of at most `chunk_keys` keys runs Furiosa's one-shot
//! attention: gather the keys by token slot, contract, mask, one softmax,
//! contract. Longer contexts run Furiosa's flash attention: a device loop over
//! chunks of `chunk_keys` keys up to the longest sequence, gathering whole
//! pages, with an online softmax.
const std = @import("std");

const builder = @import("kernels/tcl/builder");

const zml = @import("../zml.zig");
const Tensor = zml.Tensor;
const tcl = zml.kernel.tcl;
const paged_attention = @import("paged_attention.zig");
const triton = @import("triton_attention.zig");

const chunk_keys = 4096;
const prefill_block = 32;

pub const paged = struct {
    pub const Options = triton.paged.Options;
    pub const Parameters = triton.paged.Parameters;

    /// `q` is `{.b = tokens, .hkv, .hg, .hd}`, the caches
    /// `{.page, .k_chunk, .hkv, .hd}`. Causal, without sliding window or sinks.
    pub fn pagedAttention(parameters: Parameters, q: Tensor, k_cache: Tensor, v_cache: Tensor, opts: paged_attention.AttentionOptions) Tensor {
        if (!opts.is_causal or opts.sliding_window >= 0 or opts.sink != null)
            std.debug.panic("TCL paged attention is causal, without sliding window or sinks", .{});
        const page = k_cache.dim(.k_chunk);
        const keys = parameters.block_table.dim(.p) * page;
        const decode = !parameters.options_.is_prefill and q.dim(.b) == parameters.seq_lens.dim(.b);
        const blocks = if (decode) decodeBlocks(parameters, q) else prefillBlocks(parameters, q);
        const cfg: Cfg = .{
            .b = blocks.q.dim(.nb),
            .i = blocks.q.dim(.qb),
            .a = if (keys <= chunk_keys) keys else (std.math.divCeil(i64, keys, chunk_keys) catch unreachable) * chunk_keys,
            .chunk = @min(keys, chunk_keys),
            .hkv = q.dim(.hkv),
            .g = q.dim(.hg),
            .d = q.dim(.hd),
            .pages = k_cache.dim(.page),
            .pg = page,
            .scale = opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd))))),
            .decode = decode,
        };
        const table = parameters.block_table.withTags(.{ .s, .p }).gather(.{ .s = blocks.seq }, .{}).pad(-1, .{ .p = Tensor.Pad{ .high = @divExact(cfg.a, page) - parameters.block_table.dim(.p) } });
        const position = Tensor.iota(.init(.{ .nb = cfg.b, .qb = cfg.i, .a = cfg.a }, .i32), .a);
        const mask = position.cmp(.LE, blocks.qpos.insertAxes(.last, .{.a}).broad(position.shape()));
        const q_bf16 = blocks.q.convert(.bf16);
        const out_shape = blocks.q.shape().withDtype(.bf16);
        const out = if (cfg.a == cfg.chunk) out: {
            // Token slots of the keys; padding keys read distinct slots.
            const keys_shape: zml.Shape = .init(.{ .nb = cfg.b, .a = cfg.a }, .i32);
            const a = Tensor.iota(keys_shape, .a);
            const pages = table.rename(.{ .p = .pp }).gather(.{ .pp = a.divByConst(page) }, .{});
            const slots = pages.cmp(.GE, .scalar(0, .i32)).select(pages.scale(page).add(a.remainderConst(page)), a.remainderConst(cfg.pages * page));
            break :out OneShot.call(.{ .q = q_bf16, .slots = slots, .k_cache = k_cache.convert(.bf16), .v_cache = v_cache.convert(.bf16), .mask = mask }, .{ .out = out_shape }, .{ .cfg = cfg }).out;
        } else out: {
            const chunk_pages = @divExact(cfg.chunk, page);
            const chunks = @divExact(cfg.a, cfg.chunk);
            const pages = table.maximum(.scalar(0, .i32)).splitAxis(.p, .{ .c = chunks, .cp = chunk_pages }).transpose(.{ .c, .nb, .cp });
            const used = blocks.kvlen.addConstant(page - 1).divByConst(page).insertAxes(0, .{.c}).broad(.init(.{ .c = chunks, .nb = cfg.b }, .i32));
            const chunk_valid = used.sub(Tensor.iota(used.shape(), .c).scale(chunk_pages)).clamp(.scalar(0, .i32), .scalar(chunk_pages, .i32));
            const chunk_mask = mask.splitAxis(.a, .{ .c = chunks, .ck = cfg.chunk }).transpose(.{ .c, .nb, .qb, .ck });
            break :out Flash.call(.{ .q = q_bf16, .pages = pages, .chunk_valid = chunk_valid, .k_cache = k_cache.convert(.bf16), .v_cache = v_cache.convert(.bf16), .mask = chunk_mask }, .{ .out = out_shape }, .{ .cfg = cfg }).out;
        };
        const rows = out.merge(.{ .row = .{ .nb, .qb } });
        return rows.gather(.{ .row = blocks.row }, .{}).withTags(.{ .b, .hkv, .hg, .hd }).convert(q.dtype());
    }

    /// Blocks of query rows of one sequence each.
    const Blocks = struct {
        /// `{.nb, .qb, .hkv, .hg, .hd}` queries.
        q: Tensor,
        /// `{.nb}` sequence of each block, clamped to a valid one.
        seq: Tensor,
        /// `{.nb, .qb}` position of each query row, -1 for padding rows.
        qpos: Tensor,
        /// `{.nb}` visible keys of each block, 0 for padding blocks.
        kvlen: Tensor,
        /// `{.b}` row of each token in the flattened `{.nb, .qb}` blocks.
        row: Tensor,
    };

    /// Token `b` is the query of sequence `b`.
    fn decodeBlocks(parameters: Parameters, q: Tensor) Blocks {
        const seq_lens = parameters.seq_lens.withTags(.{.nb});
        const n = q.dim(.b);
        return .{
            .q = q.rename(.{ .b = .nb }).insertAxes(.hkv, .{.qb}),
            .seq = Tensor.iota(.init(.{ .nb = n }, .i32), .nb),
            .qpos = seq_lens.addConstant(-1).insertAxes(.last, .{.qb}),
            .kvlen = seq_lens,
            .row = Tensor.iota(.init(.{ .b = n }, .i32), .b),
        };
    }

    /// Each sequence starts a block: at most one partial block each.
    fn prefillBlocks(parameters: Parameters, q: Tensor) Blocks {
        const tokens = q.dim(.b);
        const batch = parameters.seq_lens.dim(.b);
        const nb = (std.math.divCeil(i64, tokens, prefill_block) catch unreachable) + batch;
        const starts = parameters.query_start_len.slice(.b, .{ .end = batch }).withTags(.{.s});
        const ends = parameters.query_start_len.slice(.b, .{ .start = 1 }).withTags(.{.s});
        const qlen = ends.sub(starts);
        const seq_lens = parameters.seq_lens.withTags(.{.s});
        const blocks = qlen.addConstant(prefill_block - 1).divByConst(prefill_block);
        const block_end = blocks.cumulativeSum(.s);
        const block_start = block_end.sub(blocks);
        const pick = struct {
            fn f(t: Tensor, idx: Tensor) Tensor {
                return t.gather(.{ .s = idx }, .{});
            }
        }.f;

        // Block j belongs to the sequence whose blocks end after j.
        const nb_shape: zml.Shape = .init(.{ .nb = nb, .s = batch }, .i32);
        const seq_of_block = block_end.broad(nb_shape).cmp(.LE, Tensor.iota(nb_shape, .nb)).convert(.i32).sum(.s).squeeze(.s);
        const live_block = seq_of_block.cmp(.LT, .scalar(batch, .i32));
        const seq = seq_of_block.minimum(.scalar(batch - 1, .i32));
        const rows: zml.Shape = .init(.{ .nb = nb, .qb = prefill_block }, .i32);
        const first = Tensor.iota(.init(.{ .nb = nb }, .i32), .nb).sub(pick(block_start, seq)).scale(prefill_block);
        const qi = first.broad(rows).add(Tensor.iota(rows, .qb));
        const qlen_b = pick(qlen, seq).broad(rows);
        const live = qi.cmp(.LT, qlen_b).logical(.AND, live_block.broad(rows.withDtype(.bool)));
        const token = live.select(pick(starts, seq).broad(rows).add(qi), Tensor.zeroes(rows));

        // Token t is row (t - start) % block of block start_block + (t - start) / block.
        const ts: zml.Shape = .init(.{ .b = tokens, .s = batch }, .i32);
        const seq_of_token = ends.broad(ts).cmp(.LE, Tensor.iota(ts, .b)).convert(.i32).sum(.s).squeeze(.s).minimum(.scalar(batch - 1, .i32));
        const qi_t = Tensor.iota(.init(.{ .b = tokens }, .i32), .b).sub(pick(starts, seq_of_token));
        return .{
            .q = q.gather(.{ .b = token }, .{}),
            .seq = seq,
            .qpos = live.select(pick(seq_lens, seq).broad(rows).sub(qlen_b).add(qi), Tensor.scalar(-1, .i32).broad(rows)),
            .kvlen = live_block.select(pick(seq_lens, seq), Tensor.zeroes(.init(.{ .nb = nb }, .i32))),
            .row = pick(block_start, seq_of_token).scale(prefill_block).add(qi_t).clamp(.scalar(0, .i32), .scalar(nb * prefill_block - 1, .i32)),
        };
    }
};

pub const Cfg = struct {
    /// Query blocks and query rows per block.
    b: i64,
    i: i64,
    /// Keys per block (padded to whole chunks) and keys per chunk.
    a: i64,
    chunk: i64,
    hkv: i64,
    g: i64,
    d: i64,
    pages: i64,
    pg: i64,
    scale: f32,
    decode: bool,
};

/// Furiosa's compiler configuration for attention kernels
/// (`furiosa.kernels.llama.config`, `common.attention`).
fn configure(b: *tcl.Builder, cfg: Cfg) void {
    const lowering_mode = if (cfg.b * cfg.i >= 8) "Heuristic" else "Default";
    if (cfg.decode) b.compilerConfig(.{
        .enable_einsum_fusion = true,
        .padding_policy = .{ .Small = 1.1 },
        .instruction_mem_budget = 0xB0000,
        .reshape_einsum_mode = .{ .Reshape = .{ .permute = false } },
        .tensor_unit_bridge_threshold_in_page = 12,
        .allow_external_operators = false,
        .enable_tactic_pruning = false,
        .scheduler_beam_search = true,
        .allow_reduce_by_ve_cluster_chip_reduce = false,
        .allow_reduce_by_ve_cluster_chip_reduce_base_population = false,
        // tcc's fused attention kernel fails on some llmd shapes ("Vrf output
        // tensor of pass chunk cannot be segmented").
        .use_attention_kernel = false,
        .propagate_sparse_axis_from_op = "Gather",
        .tactic_hint = .{ .ForLlmModelIOBound = 200 },
        .apply_adaptive_einsum_by_pattern = true,
        .num_transaction_simulation_per_pe = 1024,
        .dma_preference = 0.8,
        .enable_vrf_half_mode = true,
        .lowering_mode = lowering_mode,
    }) else b.compilerConfig(.{
        .enable_einsum_fusion = true,
        .padding_policy = .{ .Small = 1.1 },
        .instruction_mem_budget = 0xB0000,
        .reshape_einsum_mode = .{ .Reshape = .{ .permute = false } },
        .tensor_unit_bridge_threshold_in_page = 12,
        .allow_external_operators = false,
        .enable_tactic_pruning = false,
        .scheduler_beam_search = true,
        .allow_reduce_by_ve_cluster_chip_reduce = false,
        .allow_reduce_by_ve_cluster_chip_reduce_base_population = false,
        .use_attention_kernel = false,
        .tactic_hint = .{ .ForLlmModelComputeBound = 200 },
        .dma_preference = 1.2,
        .lowering_mode = lowering_mode,
    });
}

const Axes = struct {
    B: tcl.Axis,
    I: tcl.Axis,
    Hkv: tcl.Axis,
    G: tcl.Axis,
    D: tcl.Axis,
    P: tcl.Axis,
    PG: tcl.Axis,

    fn init(b: *tcl.Builder, cfg: Cfg) Axes {
        return .{
            .B = b.axis("B", cfg.b),
            .I = b.axis("I", cfg.i),
            .Hkv = b.axis("Hkv", cfg.hkv),
            .G = b.axis("G", cfg.g),
            .D = b.axis("D", cfg.d),
            .P = b.axis("P", cfg.pages),
            .PG = b.axis("PG", cfg.pg),
        };
    }
};

const bf16_min = -3.3895313892515355e+38;

/// `scores = q k^T * scale`, masked with `bf16.MIN`, as Furiosa's
/// `einsum_dpe_scale` and `elementwise_attention_mask`.
fn maskedScores(b: *tcl.Builder, a: Axes, cfg: Cfg, q: builder.Tensor, k: builder.Tensor, mask: builder.Tensor, keys: tcl.Axis) tcl.FinishError!builder.Tensor {
    const scores_axes = &.{ a.Hkv, a.G, a.B, a.I, keys };
    var op = b.tensorOperation(.{});
    const scores = try op.commit(op.contract(q, k, scores_axes).mulf(cfg.scale), .{ .dtype = .bf16 });
    op = b.tensorOperation(.{});
    const s = op.fetch(scores, .{});
    return op.commit(op.where(op.fetch(mask, .{}), .ne, 0, s, bf16_min), .{ .dtype = .bf16 });
}

pub const OneShot = tcl.Kernel(Cfg, .{
    .name = "attention",
    .inputs = &.{ "q", "slots", "k_cache", "v_cache", "mask" },
    .outputs = &.{"out"},
    .run = struct {
        fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
            configure(b, cfg);
            const a: Axes = .init(b, cfg);
            const A = b.axis("A", cfg.a);
            const Nb = b.axis("Nb", cfg.pages * cfg.pg);
            const t = try b.declareArgs(.{
                .q = .{ .dtype = .bf16, .axes = &.{ a.B, a.I, a.Hkv, a.G, a.D } },
                .slots = .{ .dtype = .i32, .axes = &.{ a.B, A } },
                .k_cache = .{ .dtype = .bf16, .axes = &.{ a.P, a.PG, a.Hkv, a.D } },
                .v_cache = .{ .dtype = .bf16, .axes = &.{ a.P, a.PG, a.Hkv, a.D } },
                .mask = .{ .dtype = .bool, .axes = &.{ a.B, a.I, A } },
            });
            const kc = try b.reshape(t.k_cache, &.{ Nb, a.Hkv, a.D }, .{});
            const vc = try b.reshape(t.v_cache, &.{ Nb, a.Hkv, a.D }, .{});
            const k = try b.gather(kc, t.slots, Nb, &.{ a.B, A, a.Hkv, a.D }, .{});
            const v = try b.gather(vc, t.slots, Nb, &.{ a.B, A, a.Hkv, a.D }, .{});
            const masked = try maskedScores(b, a, cfg, t.q, k, t.mask, A);
            var op = b.tensorOperation(.{});
            const x = op.fetch(masked, .{});
            const e = x.subf(x.reduce(&.{A}, .maxf)).exp();
            const p = try op.commit(e.divf(e.reduce(&.{A}, .addf)), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            b.ret(&.{try op.commit(op.contract(v, p, &.{ a.B, a.I, a.Hkv, a.G, a.D }), .{ .dtype = .bf16 })});
        }
    }.run,
});

pub const Flash = tcl.Kernel(Cfg, .{
    .name = "flash_attention",
    .inputs = &.{ "q", "pages", "chunk_valid", "k_cache", "v_cache", "mask" },
    .outputs = &.{"out"},
    .run = struct {
        fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
            configure(b, cfg);
            const a: Axes = .init(b, cfg);
            const C = b.axis("C", @divExact(cfg.a, cfg.chunk));
            const Ck = b.axis("Ck", cfg.chunk);
            const Cp = b.axis("Cp", @divExact(cfg.chunk, cfg.pg));
            const t = try b.declareArgs(.{
                .q = .{ .dtype = .bf16, .axes = &.{ a.B, a.I, a.Hkv, a.G, a.D } },
                .pages = .{ .dtype = .i32, .axes = &.{ C, a.B, Cp } },
                .chunk_valid = .{ .dtype = .i32, .axes = &.{ C, a.B } },
                .k_cache = .{ .dtype = .bf16, .axes = &.{ a.P, a.PG, a.Hkv, a.D } },
                .v_cache = .{ .dtype = .bf16, .axes = &.{ a.P, a.PG, a.Hkv, a.D } },
                .mask = .{ .dtype = .bool, .axes = &.{ C, a.B, a.I, Ck } },
            });
            const stats = &.{ a.Hkv, a.G, a.B, a.I };
            const out_axes = &.{ a.B, a.I, a.Hkv, a.G, a.D };
            var op = b.tensorOperation(.{});
            const valid = try op.commit(op.fetch(t.chunk_valid, .{}).reduce(&.{C}, .addi), .{});
            const n = try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(valid), @divExact(cfg.chunk, cfg.pg) - 1), @divExact(cfg.chunk, cfg.pg));
            var loop = b.openFor(n, .{ try b.full(0, .bf16, stats), try b.full(0, .bf16, out_axes), try b.full(bf16_min, .bf16, stats) });
            const sum_acc, const qkv_acc, const max_acc = loop.carried;
            const cvl = try b.indexRead(t.chunk_valid, loop.iv);
            const idx = try b.indexRead(t.pages, loop.iv);
            const mask = try b.indexRead(t.mask, loop.iv);
            const k = try b.reshape(try b.gather(t.k_cache, idx, a.P, &.{ a.B, Cp, a.PG, a.Hkv, a.D }, .{ .valid_length = cvl }), &.{ a.B, Ck, a.Hkv, a.D }, .{});
            const v = try b.reshape(try b.gather(t.v_cache, idx, a.P, &.{ a.B, Cp, a.PG, a.Hkv, a.D }, .{ .valid_length = cvl }), &.{ a.B, Ck, a.Hkv, a.D }, .{});
            const masked = try maskedScores(b, a, cfg, t.q, k, mask, Ck);
            op = b.tensorOperation(.{});
            const max_cur = try op.commit(op.fetch(masked, .{}).reduce(&.{Ck}, .maxf), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            const max_new = try op.commit(op.fetch(max_cur, .{}).maxf(op.fetch(max_acc, .{})), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            const rescale = try op.commit(op.fetch(max_acc, .{}).subf(op.fetch(max_new, .{})).exp(), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            const e = try op.commit(op.fetch(masked, .{}).subf(op.fetch(max_new, .{})).exp(), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            const sum_cur = try op.commit(op.fetch(e, .{}).reduce(&.{Ck}, .addf), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            const sum_new = try op.commit(op.fetch(sum_acc, .{}).mulf(op.fetch(rescale, .{})).addf(op.fetch(sum_cur, .{})), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            const qkv_cur = try op.commit(op.contract(e, v, out_axes), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            const qkv_new = try op.commit(op.fetch(qkv_acc, .{}).mulf(op.fetch(rescale, .{})).addf(op.fetch(qkv_cur, .{})), .{ .dtype = .bf16 });
            try loop.yield(.{ sum_new, qkv_new, max_new });

            op = b.tensorOperation(.{});
            b.ret(&.{try op.commit(op.fetch(loop.results[1], .{}).divf(op.fetch(loop.results[0], .{})), .{ .dtype = .bf16 })});
        }
    }.run,
});

test "tcl paged attention runs on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const batch = 4;
    // (query tokens, context including them) per sequence. Up to 16 pages
    // per sequence is one chunk (one-shot), 320 pages is flash attention.
    const Case = struct { prefill: bool, seqs: [batch][2]i32 };
    inline for (.{ 16, 320 }) |max_pages| {
        const num_pages = 2 * max_pages;
        const cases = if (max_pages == 16) [_]Case{
            .{ .prefill = true, .seqs = .{ .{ 37, 200 }, .{ 1, 77 }, .{ 20, 20 }, .{ 0, 0 } } },
            .{ .prefill = false, .seqs = .{ .{ 1, 250 }, .{ 1, 3 }, .{ 1, 64 }, .{ 1, 129 } } },
        } else [_]Case{
            .{ .prefill = true, .seqs = .{ .{ 37, 4500 }, .{ 1, 77 }, .{ 70, 70 }, .{ 0, 0 } } },
            .{ .prefill = false, .seqs = .{ .{ 1, 5000 }, .{ 1, 3 }, .{ 1, 4097 }, .{ 1, 129 } } },
        };
        try runCases(max_pages, num_pages, &cases);
    }
}

fn runCases(comptime max_pages: usize, comptime num_pages: usize, cases: anytype) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bf16 = zml.floats.BFloat16;
    const hkv = 2;
    const g = 4;
    const d = 64;
    const page = 16;
    const batch = 4;
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    const kc = try allocator.alloc(bf16, num_pages * page * hkv * d);
    defer allocator.free(kc);
    const vc = try allocator.alloc(bf16, num_pages * page * hkv * d);
    defer allocator.free(vc);
    for (kc) |*v| v.* = .fromF32(r.floatNorm(f32));
    for (vc) |*v| v.* = .fromF32(r.floatNorm(f32));
    const k_t: Tensor = .init(.{ .page = num_pages, .k_chunk = page, .hkv = hkv, .hd = d }, .bf16);
    var bk: zml.Buffer = try .fromBytes(io, platform, k_t.shape(), .replicated, std.mem.sliceAsBytes(kc));
    defer bk.deinit();
    var bv: zml.Buffer = try .fromBytes(io, platform, k_t.shape(), .replicated, std.mem.sliceAsBytes(vc));
    defer bv.deinit();

    for (cases) |case| {
        var tokens: i64 = 0;
        for (case.seqs) |s| tokens += s[0];
        const params: paged.Parameters = .init(.{ .batch_size = batch, .max_num_pages = max_pages, .max_seqlen_q = 64, .is_prefill = case.prefill });
        const q_t: Tensor = .init(.{ .b = tokens, .hkv = hkv, .hg = g, .hd = d }, .bf16);
        const Mod = struct {
            pub fn forward(p: paged.Parameters, q: Tensor, k: Tensor, v: Tensor) Tensor {
                return paged.pagedAttention(p, q, k, v, .{});
            }
        };
        const v_t: Tensor = .init(k_t.shape(), .bf16);
        var exe = try zml.module.compile(allocator, io, Mod.forward, .{ params, q_t, k_t, v_t }, platform, .{});
        defer exe.deinit();

        const q = try allocator.alloc(bf16, @intCast(tokens * hkv * g * d));
        defer allocator.free(q);
        for (q) |*v| v.* = .fromF32(r.floatNorm(f32));
        var table: [batch][max_pages]i32 = @splat(@splat(-1));
        var seq_lens: [batch]i32 = undefined;
        var starts: [batch + 1]i32 = undefined;
        starts[0] = 0;
        var next_page: i32 = 0;
        for (case.seqs, 0..) |s, b| {
            seq_lens[b] = s[1];
            starts[b + 1] = starts[b] + s[0];
            for (0..@intCast(std.math.divCeil(i32, s[1], page) catch unreachable)) |p| {
                table[b][p] = @mod(next_page * 7, @as(i32, num_pages));
                next_page += 1;
            }
        }
        var bq: zml.Buffer = try .fromBytes(io, platform, q_t.shape(), .replicated, std.mem.sliceAsBytes(q));
        defer bq.deinit();
        var bp: zml.Bufferized(paged.Parameters) = .{
            .block_table = try .fromBytes(io, platform, params.block_table.shape(), .replicated, std.mem.sliceAsBytes(&table)),
            .seq_lens = try .fromBytes(io, platform, params.seq_lens.shape(), .replicated, std.mem.sliceAsBytes(&seq_lens)),
            .query_start_len = try .fromBytes(io, platform, params.query_start_len.shape(), .replicated, std.mem.sliceAsBytes(&starts)),
        };
        defer zml.Buffer.deinitAll(paged.Parameters, &bp);
        var result = try exe.eval(allocator, io, .{ bp, bq, bk, bv });
        defer result.deinit();
        var host = try result.toSliceAlloc(allocator, io);
        defer host.free(allocator);
        const out = host.items(bf16);

        const scale = 1.0 / @sqrt(@as(f32, d));
        for (case.seqs, 0..) |s, b| for (0..@intCast(s[0])) |i| {
            const tok: usize = @intCast(starts[b] + @as(i32, @intCast(i)));
            const visible: usize = @intCast(s[1] - s[0] + @as(i32, @intCast(i)) + 1);
            for (0..hkv) |h| for (0..g) |gi| {
                const qrow = q[((tok * hkv + h) * g + gi) * d ..][0..d];
                var scores: [max_pages * page]f32 = undefined;
                var max: f32 = -std.math.inf(f32);
                for (0..visible) |key| {
                    const slot: usize = @intCast(table[b][key / page]);
                    const krow = kc[((slot * page + key % page) * hkv + h) * d ..][0..d];
                    var dot: f32 = 0;
                    for (qrow, krow) |x, y| dot += x.toF32() * y.toF32();
                    scores[key] = dot * scale;
                    max = @max(max, scores[key]);
                }
                var sum: f32 = 0;
                for (scores[0..visible]) |*sc| {
                    sc.* = @exp(sc.* - max);
                    sum += sc.*;
                }
                for (0..d) |dd| {
                    var want: f32 = 0;
                    for (0..visible) |key| {
                        const slot: usize = @intCast(table[b][key / page]);
                        want += scores[key] * vc[((slot * page + key % page) * hkv + h) * d + dd].toF32();
                    }
                    want /= sum;
                    const got = out[((tok * hkv + h) * g + gi) * d + dd].toF32();
                    try std.testing.expectApproxEqAbs(want, got, 3e-2 + 3e-2 * @abs(want));
                }
            };
        };
    }
}
