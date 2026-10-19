//! Furiosa RNGD paged attention in TCL, using the common/flash attention
//! algorithms from Furiosa's SDK (`furiosa-kernels 2026.4.0` reference).
//! This implementation retains FP32 normalization/accumulation and BF16
//! contractions. The exact common SDK operation materializes scores in BF16;
//! it fails the required unit-scale tolerance and is not a selectable backend.
//!
//! Queries are cut in blocks of rows of one sequence (one token per block for
//! decode). A context of at most `chunk_keys` keys runs Furiosa's one-shot
//! attention: gather the keys by token slot, contract, mask, one softmax,
//! contract. Longer contexts run Furiosa's flash attention: a device loop over
//! chunks of `chunk_keys` keys up to the longest sequence, gathering whole
//! pages and accumulating the softmax statistics in f32.
const std = @import("std");

const stdx = @import("stdx");
const builder = @import("kernels/tcl/builder");

const zml = @import("../../../zml.zig");
const Tensor = zml.Tensor;
const tcl = zml.kernel.tcl;
const paged_attention = @import("../paged_attention.zig");
const triton = @import("../triton/triton.zig");

const chunk_keys = 4096;
const prefill_block = 32;

pub const paged = struct {
    pub const Options = triton.paged.Options;
    pub const Parameters = triton.paged.Parameters;

    /// `q` is `{.b = tokens, .hkv, .hg, .hd}`, the caches
    /// `{.page, .k_chunk, .hkv, .hd}`. Sinks are one logit per query head.
    pub fn pagedAttention(parameters: Parameters, q: Tensor, k_cache: Tensor, v_cache: Tensor, opts: paged_attention.AttentionOptions) Tensor {
        if (opts.mask == .indices) @panic("Sparse attention masks are not supported by TCL paged attention");
        stdx.debug.assert(q.rank() == 4 and k_cache.rank() == 4 and v_cache.rank() == 4 and k_cache.shape().eqlDims(v_cache.shape()), "TCL attention requires matching rank-four split K/V caches", .{});
        stdx.debug.assert(q.dtype().isFloat() and k_cache.dtype().isFloat() and v_cache.dtype().isFloat(), "TCL attention requires floating-point queries and caches", .{});
        stdx.debug.assert(q.dim(.hkv) == k_cache.dim(.hkv) and q.dim(.hd) == k_cache.dim(.hd) and q.dim(.hg) > 0, "TCL attention query and cache head dimensions must match", .{});
        stdx.debug.assert(parameters.block_table.dtype() == .i32 and parameters.seq_lens.dtype() == .i32 and parameters.query_start_len.dtype() == .i32, "TCL attention metadata must be I32", .{});
        stdx.debug.assert(parameters.block_table.dim(.b) == parameters.seq_lens.dim(.b) and parameters.query_start_len.dim(.b) == parameters.seq_lens.dim(.b) + 1, "TCL attention metadata batch dimensions must match", .{});
        stdx.debug.assert(parameters.seq_lens.dim(.b) > 0 and parameters.block_table.dim(.p) > 0 and k_cache.dim(.page) > 0 and k_cache.dim(.k_chunk) > 0, "TCL attention requires nonempty sequence and cache capacities", .{});
        stdx.debug.assert(k_cache.dim(.page) * k_cache.dim(.k_chunk) <= std.math.maxInt(i32) and q.dim(.b) <= std.math.maxInt(i32), "TCL attention indices must fit in I32", .{});
        const compiler = zml.Compiler.current();
        stdx.debug.assert(compiler.platform.target == .furiosa and compiler.meshes[0].numDevices() == 1, "TCL attention requires one Furiosa chip", .{});
        const attributes = compiler.platform.devices[0].pjrt_device.attributes(compiler.platform.pjrt_api) catch
            std.debug.panic("TCL attention could not verify the Furiosa device's PE range", .{});
        defer attributes.deinit();
        const first_pe = attributes.get("pe_start");
        const last_pe = attributes.get("pe_end");
        stdx.debug.assert(first_pe != null and first_pe.? == .int64 and first_pe.?.int64 == 0 and
            last_pe != null and last_pe.? == .int64 and last_pe.?.int64 == 7, "TCL attention requires a complete 8-PE RNGD chip", .{});
        if (q.dim(.b) == 0) return Tensor.zeroes(q.shape());
        const page = k_cache.dim(.k_chunk);
        // The 2026.4 compiler has no DPE tactic for 32-key attention. Pad to
        // at least 64 keys; the extra pages remain invalid and contribute zero.
        const keys = @max(parameters.block_table.dim(.p), std.math.divCeil(i64, 64, page) catch unreachable) * page;
        const chunk = @min(keys, @max(1, @divTrunc(chunk_keys, page)) * page);
        const decode = !parameters.options_.is_prefill and q.dim(.b) == parameters.seq_lens.dim(.b);
        const blocks = if (decode) decodeBlocks(parameters, q) else prefillBlocks(parameters, q);
        const cfg: Cfg = .{
            .b = blocks.q.dim(.nb),
            .i = blocks.q.dim(.qb),
            .a = (std.math.divCeil(i64, keys, chunk) catch unreachable) * chunk,
            .chunk = chunk,
            .hkv = q.dim(.hkv),
            .g = q.dim(.hg),
            .d = q.dim(.hd),
            .pages = k_cache.dim(.page),
            .pg = page,
            .scale = opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd))))),
            .decode = decode,
            .use_sink = opts.sink != null,
            .is_causal = opts.mask.isCausal(),
            .window = @intCast(opts.mask.slidingWindow() orelse 0),
        };
        const block_table = if (decode) parameters.block_table.withTags(.{ .nb, .p }) else parameters.block_table.withTags(.{ .s, .p }).gather(.{ .s = blocks.seq }, .{});
        const table = block_table.pad(-1, .{ .p = Tensor.Pad{ .high = @divExact(cfg.a, page) - parameters.block_table.dim(.p) } });
        const q_bf16 = blocks.q.convert(.bf16);
        const sink = if (opts.sink) |s| s.convert(.f32).reshape(.{ .hkv = cfg.hkv, .hg = cfg.g }) else Tensor.zeroes(.init(.{ .hkv = cfg.hkv, .hg = cfg.g }, .f32));
        const out_shape = blocks.q.shape().withDtype(.bf16);
        const out = if (cfg.a == cfg.chunk) out: {
            const key_pos = Tensor.iota(.init(.{ .nb = cfg.b, .a = cfg.a }, .i32), .a);
            const key_pages = table.rename(.{ .p = .pp }).gather(.{ .pp = key_pos.divByConst(page) }, .{});
            const valid_page = key_pages.cmp(.GE, .scalar(0, .i32)).logical(.AND, key_pages.cmp(.LT, .scalar(cfg.pages, .i32)));
            // Sanitize padding before multiplying: maxInt(i32) must not wrap
            // into an apparently valid cache address. Physical slot zero is valid.
            const slots = key_pages.clamp(.scalar(0, .i32), .scalar(cfg.pages - 1, .i32)).scale(page).add(key_pos.remainderConst(page)).mask(valid_page, -1);
            break :out OneShot.call(.{ .q = q_bf16, .slots = slots, .k_cache = k_cache.convert(.bf16), .v_cache = v_cache.convert(.bf16), .qpos = blocks.qpos, .kvlen = blocks.kvlen, .sink = sink }, .{ .out = out_shape }, .{ .cfg = cfg }).out;
        } else out: {
            const chunk_pages = @divExact(cfg.chunk, page);
            const chunks = @divExact(cfg.a, cfg.chunk);
            const pages = table.splitAxis(.p, .{ .c = chunks, .cp = chunk_pages }).transpose(.{ .c, .nb, .cp });
            const used = blocks.kvlen.addConstant(page - 1).divByConst(page).insertAxes(0, .{.c}).broad(.init(.{ .c = chunks, .nb = cfg.b }, .i32));
            const chunk_valid = used.sub(Tensor.iota(used.shape(), .c).scale(chunk_pages)).clamp(.scalar(0, .i32), .scalar(chunk_pages, .i32));
            break :out Flash.call(.{ .q = q_bf16, .pages = pages, .chunk_valid = chunk_valid, .k_cache = k_cache.convert(.bf16), .v_cache = v_cache.convert(.bf16), .qpos = blocks.qpos, .kvlen = blocks.kvlen, .sink = sink }, .{ .out = out_shape }, .{ .cfg = cfg }).out;
        };
        // A single decode row is already in packed order, even when inactive.
        if (decode and q.dim(.b) == 1) return out.reshape(q.shape().withDtype(.bf16)).convert(q.dtype());
        // The plugin cannot gather some padded BF16 prefill layouts directly.
        const rows = out.convert(.f32).merge(.{ .row = .{ .nb, .qb } });
        const active_count = parameters.query_start_len.slice(.b, .single(parameters.seq_lens.dim(.b))).asScalar();
        const active = Tensor.iota(.init(.{ .b = q.dim(.b) }, .i32), .b).cmp(.LT, active_count);
        return rows.gather(.{ .row = blocks.row }, .{}).withTags(.{ .b, .hkv, .hg, .hd }).mask(active, 0).convert(q.dtype());
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

    /// One query per active sequence, packed even when a sequence is idle.
    fn decodeBlocks(parameters: Parameters, q: Tensor) Blocks {
        const seq_lens = parameters.seq_lens.withTags(.{.nb});
        const n = q.dim(.b);
        const starts = parameters.query_start_len.slice(.b, .{ .end = n }).withTags(.{.nb});
        const ends = parameters.query_start_len.slice(.b, .{ .start = 1 }).withTags(.{.nb});
        const live = ends.cmp(.GT, starts);
        const ts: zml.Shape = .init(.{ .b = n, .nb = n }, .i32);
        const row = ends.broad(ts).cmp(.LE, Tensor.iota(ts, .b)).convert(.i32).sum(.nb).squeeze(.nb).minimum(.scalar(n - 1, .i32));
        return .{
            .q = (if (n == 1) q.withTags(.{ .nb, .hkv, .hg, .hd }) else q.gather(.{ .b = starts.minimum(.scalar(n - 1, .i32)) }, .{})).insertAxes(.hkv, .{.qb}),
            .seq = Tensor.iota(.init(.{ .nb = n }, .i32), .nb),
            .qpos = live.select(seq_lens.addConstant(-1), .scalar(-1, .i32)).insertAxes(.last, .{.qb}),
            .kvlen = live.select(seq_lens, .scalar(0, .i32)),
            .row = row,
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

const Cfg = struct {
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
    use_sink: bool = false,
    is_causal: bool = true,
    window: i32 = -1,
};

/// Attention tuning inherited from the original TCL backend, informed by
/// `furiosa.kernels.llama.config` and `common.attention`. Keep the compiler's
/// default lowering mode and omit the adaptive-einsum and half-VRF settings
/// removed in TCC 2026.5.
fn configure(b: *tcl.Builder, cfg: Cfg) void {
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
        .num_transaction_simulation_per_pe = 1024,
        .dma_preference = 0.8,
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

const score_min = -std.math.floatMax(f32);

fn validIndices(b: *tcl.Builder, indices: builder.Tensor, count: i64) tcl.FinishError!struct { safe: builder.Tensor, valid: builder.Tensor } {
    var op = b.tensorOperation(.{});
    const safe = try op.commit(op.fetch(indices, .{}).binary(.maxi, 0).binary(.mini, count - 1), .{});
    op = b.tensorOperation(.{});
    const nonnegative = try op.commit(op.fetch(indices, .{}).addi(1).binary(.maxi, 0).binary(.mini, 1), .{});
    op = b.tensorOperation(.{ .tactic = .elementwise });
    const valid = try op.commit(op.fetch(indices, .{}).muli(-1).addi(count).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(nonnegative, .{})), .{});
    return .{ .safe = safe, .valid = valid };
}

/// Build the sequence, query, page and window mask for the current keys.
fn attentionMask(b: *tcl.Builder, cfg: Cfg, qpos: builder.Tensor, kvlen: builder.Tensor, position: builder.Tensor, valid_keys: builder.Tensor) tcl.FinishError!builder.Tensor {
    const axes = &.{ qpos.axes[0], qpos.axes[1], position.axes[0] };
    const read: builder.TensorOperation.FetchOptions = .{ .broadcast_to = axes };
    // Elementwise operations require their first read to cover the output
    // axes, and each VE instruction must consume the current accumulator.
    var op = b.tensorOperation(.{});
    const pos = try op.commit(op.fetch(position, read), .{});
    op = b.tensorOperation(.{});
    const query = try op.commit(op.fetch(qpos, read), .{});
    op = b.tensorOperation(.{});
    var mask = try op.commit(op.fetch(valid_keys, read), .{});
    op = b.tensorOperation(.{ .tactic = .elementwise });
    mask = try op.commit(op.fetch(pos, .{}).muli(-1).addi(op.fetch(kvlen, .{})).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(mask, .{})), .{});
    op = b.tensorOperation(.{ .tactic = .elementwise });
    mask = try op.commit(op.fetch(query, .{}).addi(1).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(mask, .{})), .{});
    if (cfg.is_causal) {
        op = b.tensorOperation(.{ .tactic = .elementwise });
        mask = try op.commit(op.fetch(query, .{}).subi(op.fetch(pos, .{})).addi(1).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(mask, .{})), .{});
    }
    if (cfg.window > 0) {
        op = b.tensorOperation(.{ .tactic = .elementwise });
        mask = try op.commit(op.fetch(pos, .{}).subi(op.fetch(query, .{})).addi(cfg.window).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(mask, .{})), .{});
    }
    return mask;
}

/// Keep the contraction accumulator in f32 through masking and softmax.
/// A finite sentinel also makes entirely masked chunks safe to rescale.
fn maskedScores(b: *tcl.Builder, a: Axes, cfg: Cfg, q: builder.Tensor, k: builder.Tensor, mask: builder.Tensor, keys: tcl.Axis) tcl.FinishError!builder.Tensor {
    const scores_axes = &.{ a.Hkv, a.G, a.B, a.I, keys };
    var op = b.tensorOperation(.{});
    const scores = try op.commit(op.contract(q, k, scores_axes).mulf(cfg.scale), .{});
    op = b.tensorOperation(.{});
    const s = op.fetch(scores, .{});
    return op.commit(op.where(op.fetch(mask, .{}), .ne, 0, s, score_min), .{});
}

fn maskedExp(b: *tcl.Builder, scores: builder.Tensor, maximum: builder.Tensor, mask: builder.Tensor) tcl.FinishError!builder.Tensor {
    const op = b.tensorOperation(.{});
    const e = op.fetch(scores, .{}).subf(op.fetch(maximum, .{})).exp();
    // Subtracting the sentinel from itself gives exp(0), so masking only
    // the logits is insufficient for empty rows and fully masked chunks.
    return op.commit(op.where(op.fetch(mask, .{}), .ne, 0, e, 0.0), .{});
}

fn expDifference(b: *tcl.Builder, x: builder.Tensor, maximum: builder.Tensor) tcl.FinishError!builder.Tensor {
    const op = b.tensorOperation(.{ .tactic = .elementwise });
    const m = op.fetch(maximum, .{});
    return op.commit(op.fetch(x, .{}).subf(m).exp(), .{});
}

fn safeDenominator(b: *tcl.Builder, sum: builder.Tensor) tcl.FinishError!builder.Tensor {
    const op = b.tensorOperation(.{});
    const s = op.fetch(sum, .{});
    return op.commit(op.where(s, .gt, 0, s, 1.0), .{});
}

const OneShot = tcl.Kernel(Cfg, .{
    .name = "attention",
    .inputs = &.{ "q", "slots", "k_cache", "v_cache", "qpos", "kvlen", "sink" },
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
                .qpos = .{ .dtype = .i32, .axes = &.{ a.B, a.I } },
                .kvlen = .{ .dtype = .i32, .axes = &.{a.B} },
                .sink = .{ .dtype = .f32, .axes = &.{ a.Hkv, a.G } },
            });
            const kc = try b.reshape(t.k_cache, &.{ Nb, a.Hkv, a.D }, .{});
            const vc = try b.reshape(t.v_cache, &.{ Nb, a.Hkv, a.D }, .{});
            const slots = try validIndices(b, t.slots, cfg.pages * cfg.pg);
            // Match the SDK's valid-length gathers for bandwidth-bound decode.
            const valid_length = if (cfg.decode) t.kvlen else null;
            const k = try b.gather(kc, slots.safe, Nb, &.{ a.B, A, a.Hkv, a.D }, .{ .valid_length = valid_length });
            const v = try b.gather(vc, slots.safe, Nb, &.{ a.B, A, a.Hkv, a.D }, .{ .valid_length = valid_length });
            const mask = try attentionMask(b, cfg, t.qpos, t.kvlen, try b.arange(cfg.a, .i32, &.{A}, .{}), slots.valid);
            const masked = try maskedScores(b, a, cfg, t.q, k, mask, A);
            var op = b.tensorOperation(.{});
            const p = if (cfg.decode) compact: {
                // Furiosa's common.softmax keeps normalization inside one VE
                // operation. FP32 intermediates only narrow at the DPE input.
                const exp_sink = if (cfg.use_sink) sink: {
                    const maximum = try op.commit(op.fetch(masked, .{}).reduce(&.{A}, .maxf).maxf(op.fetch(t.sink, .{})), .{});
                    break :sink try expDifference(b, t.sink, maximum);
                } else null;
                op = b.tensorOperation(.{});
                const scores = op.fetch(masked, .{});
                var maximum = scores.reduce(&.{A}, .maxf);
                if (cfg.use_sink) maximum = maximum.maxf(op.fetch(t.sink, .{}));
                const e = scores.subf(maximum).exp();
                var sum = e.reduce(&.{A}, .addf);
                if (exp_sink) |s| sum = sum.addf(op.fetch(s, .{}));
                const probabilities = try op.commit(e.divf(sum), .{ .dtype = .bf16 });
                // Empty rows have uniform probabilities from finite sentinels.
                // Clear masked probabilities before contracting with values.
                op = b.tensorOperation(.{});
                break :compact try op.commit(op.where(op.fetch(mask, .{}), .ne, 0, op.fetch(probabilities, .{}), 0.0), .{ .dtype = .bf16 });
            } else tiled: {
                // Separate reductions schedule better for multi-query blocks
                // with the current compiler; keep their intermediate state FP32.
                var m = op.fetch(masked, .{}).reduce(&.{A}, .maxf);
                if (cfg.use_sink) m = m.maxf(op.fetch(t.sink, .{}));
                const maximum = try op.commit(m, .{});
                const e = try maskedExp(b, masked, maximum, mask);
                const exp_sink = if (cfg.use_sink) try expDifference(b, t.sink, maximum) else null;
                op = b.tensorOperation(.{});
                var sum = op.fetch(e, .{}).reduce(&.{A}, .addf);
                if (exp_sink) |s| sum = sum.addf(op.fetch(s, .{}));
                const denominator = try safeDenominator(b, try op.commit(sum, .{}));
                op = b.tensorOperation(.{});
                break :tiled try op.commit(op.fetch(e, .{}).divf(op.fetch(denominator, .{})), .{ .dtype = .bf16 });
            };
            op = b.tensorOperation(.{});
            b.ret(&.{try op.commit(op.contract(v, p, &.{ a.B, a.I, a.Hkv, a.G, a.D }), .{ .dtype = .bf16 })});
        }
    }.run,
});

const Flash = tcl.Kernel(Cfg, .{
    .name = "flash_attention",
    .inputs = &.{ "q", "pages", "chunk_valid", "k_cache", "v_cache", "qpos", "kvlen", "sink" },
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
                .qpos = .{ .dtype = .i32, .axes = &.{ a.B, a.I } },
                .kvlen = .{ .dtype = .i32, .axes = &.{a.B} },
                .sink = .{ .dtype = .f32, .axes = &.{ a.Hkv, a.G } },
            });
            const stats = &.{ a.Hkv, a.G, a.B, a.I };
            const out_axes = &.{ a.B, a.I, a.Hkv, a.G, a.D };
            var op = b.tensorOperation(.{});
            const valid = try op.commit(op.fetch(t.chunk_valid, .{}).reduce(&.{C}, .addi), .{});
            const n = try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(valid), @divExact(cfg.chunk, cfg.pg) - 1), @divExact(cfg.chunk, cfg.pg));
            var loop = b.openFor(n, .{ try b.full(0, .f32, stats), try b.full(0, .f32, out_axes), try b.full(score_min, .f32, stats) });
            const sum_acc, const qkv_acc, const max_acc = loop.carried;
            const cvl = try b.indexRead(t.chunk_valid, loop.iv);
            const indices = try validIndices(b, try b.indexRead(t.pages, loop.iv), cfg.pages);
            const idx = indices.safe;
            op = b.tensorOperation(.{});
            const valid_pages = try op.commit(op.fetch(indices.valid, .{ .broadcast_to = &.{ a.B, Cp, a.PG } }), .{});
            const valid_keys = try b.reshape(valid_pages, &.{ a.B, Ck }, .{});
            const offset = try b.symExpr(.mul, loop.iv, cfg.chunk);
            const local_pos = try b.arange(cfg.chunk, .i32, &.{Ck}, .{});
            op = b.tensorOperation(.{});
            const position = try op.commit(op.fetch(local_pos, .{}).addi(op.fetch(offset, .{})), .{});
            const mask = try attentionMask(b, cfg, t.qpos, t.kvlen, position, valid_keys);
            const k = try b.reshape(try b.gather(t.k_cache, idx, a.P, &.{ a.B, Cp, a.PG, a.Hkv, a.D }, .{ .valid_length = cvl }), &.{ a.B, Ck, a.Hkv, a.D }, .{});
            const v = try b.reshape(try b.gather(t.v_cache, idx, a.P, &.{ a.B, Cp, a.PG, a.Hkv, a.D }, .{ .valid_length = cvl }), &.{ a.B, Ck, a.Hkv, a.D }, .{});
            const masked = try maskedScores(b, a, cfg, t.q, k, mask, Ck);
            op = b.tensorOperation(.{});
            const max_cur = try op.commit(op.fetch(masked, .{}).reduce(&.{Ck}, .maxf), .{});
            op = b.tensorOperation(.{ .tactic = .elementwise });
            const max_new = try op.commit(op.fetch(max_cur, .{}).maxf(op.fetch(max_acc, .{})), .{});
            const rescale = try expDifference(b, max_acc, max_new);
            const e = try maskedExp(b, masked, max_new, mask);
            op = b.tensorOperation(.{});
            const sum_cur = try op.commit(op.fetch(e, .{}).reduce(&.{Ck}, .addf), .{});
            op = b.tensorOperation(.{});
            const sum_new = try op.commit(op.fetch(sum_acc, .{}).mulf(op.fetch(rescale, .{})).addf(op.fetch(sum_cur, .{})), .{});
            op = b.tensorOperation(.{});
            // Materialize the narrowing conversion through the vector engine.
            const p = try op.commit(op.fetch(e, .{}).mulf(1.0), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            const qkv_cur = try op.commit(op.contract(p, v, out_axes), .{});
            op = b.tensorOperation(.{});
            const qkv_new = try op.commit(op.fetch(qkv_acc, .{}).mulf(op.fetch(rescale, .{})).addf(op.fetch(qkv_cur, .{})), .{});
            try loop.yield(.{ sum_new, qkv_new, max_new });

            var sum = loop.results[0];
            var numerator = loop.results[1];
            if (cfg.use_sink) {
                // Merge the zero-valued sink once, after all key chunks.
                op = b.tensorOperation(.{});
                const maximum = try op.commit(op.fetch(loop.results[2], .{}).maxf(op.fetch(t.sink, .{})), .{});
                const alpha = try expDifference(b, loop.results[2], maximum);
                const exp_sink = try expDifference(b, t.sink, maximum);
                op = b.tensorOperation(.{});
                sum = try op.commit(op.fetch(sum, .{}).mulf(op.fetch(alpha, .{})).addf(op.fetch(exp_sink, .{})), .{});
                op = b.tensorOperation(.{});
                numerator = try op.commit(op.fetch(numerator, .{}).mulf(op.fetch(alpha, .{})), .{});
            }
            const denominator = try safeDenominator(b, sum);
            op = b.tensorOperation(.{});
            b.ret(&.{try op.commit(op.fetch(numerator, .{}).divf(op.fetch(denominator, .{})), .{ .dtype = .bf16 })});
        }
    }.run,
});

test {
    std.testing.refAllDecls(attention_test);
}

const attention_test = struct {
    pub const Options = struct {
        is_causal: bool = true,
        sliding_window: i32 = -1,
        scale: ?f32 = null,
        sink: ?[]const f32 = null,
    };

    pub fn reference(allocator: std.mem.Allocator, q: zml.Slice, k: zml.Slice, v: zml.Slice, table: []const i32, lengths: []const i32, starts: []const i32, opts: Options) !zml.Slice {
        const bf16 = zml.floats.BFloat16;
        const hkv: usize = @intCast(q.shape.dim(.hkv));
        const groups: usize = @intCast(q.shape.dim(.hg));
        const d: usize = @intCast(q.shape.dim(.hd));
        const page: usize = @intCast(k.shape.dim(.k_chunk));
        const pages: usize = @intCast(k.shape.dim(.page));
        const max_pages = table.len / lengths.len;
        const scale = opts.scale orelse 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        const queries = q.constItems(bf16);
        const keys = k.constItems(bf16);
        const values = v.constItems(bf16);
        const result = try zml.Slice.alloc(allocator, q.shape.withDtype(.f32));
        errdefer result.free(allocator);
        const out = result.items(f32);
        @memset(out, 0);
        const scores = try allocator.alloc(f32, max_pages * page);
        defer allocator.free(scores);

        for (lengths, 0..) |length, seq| {
            const qlen = starts[seq + 1] - starts[seq];
            for (@intCast(starts[seq])..@intCast(starts[seq + 1])) |token| {
                const qpos = length - qlen + @as(i32, @intCast(token)) - starts[seq];
                if (qpos < 0) continue;
                for (0..hkv) |h| for (0..groups) |g| {
                    const offset = ((token * hkv + h) * groups + g) * d;
                    const sink = if (opts.sink) |s| s[h * groups + g] else -std.math.inf(f32);
                    var maximum = sink;
                    @memset(scores, -std.math.inf(f32));
                    for (0..@intCast(length)) |key| {
                        const pos: i32 = @intCast(key);
                        if (opts.is_causal and pos > qpos) continue;
                        if (opts.sliding_window > 0 and qpos - pos >= opts.sliding_window) continue;
                        const physical = table[seq * max_pages + key / page];
                        if (physical < 0 or physical >= pages) continue;
                        const kv_offset = ((@as(usize, @intCast(physical)) * page + key % page) * hkv + h) * d;
                        var dot: f32 = 0;
                        for (0..d) |i| dot += queries[offset + i].toF32() * keys[kv_offset + i].toF32();
                        scores[key] = dot * scale;
                        maximum = @max(maximum, scores[key]);
                    }
                    if (maximum == -std.math.inf(f32)) continue;
                    var denominator = @exp(sink - maximum);
                    for (scores[0..@intCast(length)]) |score| denominator += @exp(score - maximum);
                    for (0..@intCast(length)) |key| {
                        if (scores[key] == -std.math.inf(f32)) continue;
                        const physical: usize = @intCast(table[seq * max_pages + key / page]);
                        const kv_offset = ((physical * page + key % page) * hkv + h) * d;
                        const probability = @exp(scores[key] - maximum) / denominator;
                        for (0..d) |i| out[offset + i] += probability * values[kv_offset + i].toF32();
                    }
                };
            }
        }
        return result;
    }

    test "paged attention host reference scale and sink normalization" {
        const bf16 = zml.floats.BFloat16;
        const queries: [3]bf16 = @splat(.fromF32(1));
        const keys = [_]bf16{ .fromF32(0), .fromF32(1) };
        const values = [_]bf16{ .fromF32(2), .fromF32(4) };
        const q = zml.Slice.initConst(.init(.{ .b = 3, .hkv = 1, .hg = 1, .hd = 1 }, .bf16), std.mem.sliceAsBytes(&queries));
        const k = zml.Slice.initConst(.init(.{ .page = 1, .k_chunk = 2, .hkv = 1, .hd = 1 }, .bf16), std.mem.sliceAsBytes(&keys));
        const v = zml.Slice.initConst(k.shape, std.mem.sliceAsBytes(&values));
        const cases = [_]struct { opts: Options, expected: f32 }{
            .{ .opts = .{ .is_causal = false, .scale = 0 }, .expected = 3 },
            .{ .opts = .{ .is_causal = false, .scale = @log(@as(f32, 3)) }, .expected = 3.5 },
            .{ .opts = .{ .is_causal = false, .scale = @log(@as(f32, 3)), .sink = &.{0} }, .expected = 2.8 },
        };
        for (cases) |case| {
            const result = try reference(std.testing.allocator, q, k, v, &.{0}, &.{2}, &.{ 0, 2 }, case.opts);
            defer result.free(std.testing.allocator);
            for (result.constItems(f32)[0..2]) |value| try std.testing.expectApproxEqAbs(case.expected, value, 1e-6);
            try std.testing.expectEqual(@as(f32, 0), result.constItems(f32)[2]);
        }
    }

    pub fn expectClose(expected: zml.Slice, actual: zml.Slice) !void {
        try std.testing.expectEqualSlices(i64, expected.shape.dims(), actual.shape.dims());
        for (expected.constItems(f32), 0..) |want, i| {
            const got = floatAt(actual, i);
            try std.testing.expect(std.math.isFinite(got));
            std.testing.expectApproxEqAbs(want, got, 3e-2 + 3e-2 * @abs(want)) catch |err| {
                std.log.err("paged attention mismatch at element {d}: expected {d}, got {d}", .{ i, want, got });
                return err;
            };
        }
    }

    fn floatAt(slice: zml.Slice, index: usize) f32 {
        return switch (slice.shape.dtype()) {
            .bf16 => slice.constItems(zml.floats.BFloat16)[index].toF32(),
            .f16 => @floatCast(slice.constItems(f16)[index]),
            .f32 => slice.constItems(f32)[index],
            else => unreachable,
        };
    }

    test "paged attention host reference windows sinks and padding" {
        const bf16 = zml.floats.BFloat16;
        const queries: [3]bf16 = @splat(.fromF32(0));
        const keys: [2]bf16 = @splat(.fromF32(0));
        const values = [_]bf16{ .fromF32(2), .fromF32(4) };
        const q = zml.Slice.initConst(.init(.{ .b = 3, .hkv = 1, .hg = 1, .hd = 1 }, .bf16), std.mem.sliceAsBytes(&queries));
        const k = zml.Slice.initConst(.init(.{ .page = 1, .k_chunk = 2, .hkv = 1, .hd = 1 }, .bf16), std.mem.sliceAsBytes(&keys));
        const v = zml.Slice.initConst(k.shape, std.mem.sliceAsBytes(&values));
        const cases = [_]struct { opts: Options, expected: [3]f32 }{
            .{ .opts = .{}, .expected = .{ 2, 3, 0 } },
            .{ .opts = .{ .sliding_window = 1 }, .expected = .{ 2, 4, 0 } },
            .{ .opts = .{ .is_causal = false, .sliding_window = 1 }, .expected = .{ 3, 4, 0 } },
            .{ .opts = .{ .sink = &.{0} }, .expected = .{ 1, 2, 0 } },
            .{ .opts = .{ .sink = &.{-std.math.inf(f32)} }, .expected = .{ 2, 3, 0 } },
        };
        for (cases) |case| {
            const result = try reference(std.testing.allocator, q, k, v, &.{0}, &.{2}, &.{ 0, 2 }, case.opts);
            defer result.free(std.testing.allocator);
            try std.testing.expectEqualSlices(f32, &case.expected, result.constItems(f32));
        }
    }
};

const TestCase = struct {
    prefill: bool,
    seqs: [4][2]i32,
    padding: i64 = 0,
    opts: attention_test.Options = .{},
    mask_all: bool = false,
    padding_page: i32 = -1,
};

const TestDims = struct {
    batch: usize = 4,
    hkv: usize = 2,
    g: usize = 4,
    d: usize = 64,
    page: usize = 16,
    uniform: bool = false,
    uniform_scores: bool = false,
    dtype: zml.DataType = .bf16,
};

test "tcl paged attention unit scale and dtype boundaries on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    try runCases(16, 32, &.{.{ .prefill = true, .seqs = @splat(.{ 32, 256 }), .opts = .{ .scale = 1.0 } }}, .{});
    inline for (.{ .f16, .f32 }) |dtype| {
        try runCases(2, 8, &.{.{ .prefill = false, .seqs = .{ .{ 1, 17 }, .{ 1, 1 }, .{ 1, 3 }, .{ 0, 0 } }, .padding = 1, .opts = .{ .sink = &.{ 0, -std.math.inf(f32) } } }}, .{ .hkv = 1, .g = 2, .d = 32, .dtype = dtype });
    }
}

test "tcl paged attention single decode row needs no rearrangement" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    try runCases(16, 32, &.{
        .{ .prefill = false, .seqs = .{ .{ 1, 251 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } } },
        .{ .prefill = false, .seqs = @splat(.{ 0, 0 }), .padding = 1 },
    }, .{ .batch = 1 });
}

test "tcl paged attention invalid pages cannot wrap into slot zero" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const cases = [_]TestCase{
        .{ .prefill = false, .seqs = @splat(.{ 1, 17 }), .mask_all = true, .padding_page = 1 << 28 },
        .{ .prefill = false, .seqs = @splat(.{ 1, 17 }), .mask_all = true, .padding_page = std.math.maxInt(i32) },
    };
    try runCases(2, 8, &cases, .{ .hkv = 1, .g = 1, .d = 32 });
}

test "tcl paged attention emits decode and prefill kernels with and without sinks" {
    inline for (.{ false, true }) |decode| {
        inline for (.{ false, true }) |use_sink| {
            const cfg: Cfg = .{ .b = 2, .i = if (decode) 1 else 4, .a = 32, .chunk = 16, .hkv = 2, .g = 4, .d = 32, .pages = 8, .pg = 16, .scale = 0.125, .decode = decode, .use_sink = use_sink };
            const one_shot = try OneShot.emit(std.testing.allocator, cfg);
            defer std.testing.allocator.free(one_shot);
            const flash = try Flash.emit(std.testing.allocator, cfg);
            defer std.testing.allocator.free(flash);
            for ([_][]const u8{ one_shot, flash }) |ir| {
                try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "\"tcl.dpe\""));
                try std.testing.expect(std.mem.indexOf(u8, ir, "!tcl.logical<f32") != null);
                try std.testing.expect(std.mem.indexOf(u8, ir, "!tcl.logical<bf16") != null);
                try std.testing.expect(std.mem.indexOf(u8, ir, "allow_external_operators = false") != null);
                try std.testing.expect(std.mem.indexOf(u8, ir, "lowering_mode") == null);
            }
            try std.testing.expect(std.mem.indexOf(u8, one_shot, "\"tcl.graph.for\"") == null);
            try std.testing.expect(std.mem.indexOf(u8, flash, "\"tcl.graph.for\"") != null);
        }
    }
}

test "tcl paged attention runs on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    inline for (.{ 16, 320 }) |max_pages| {
        const cases = if (max_pages == 16) [_]TestCase{
            .{ .prefill = true, .seqs = .{ .{ 37, 200 }, .{ 1, 77 }, .{ 20, 20 }, .{ 0, 0 } } },
            .{ .prefill = false, .seqs = .{ .{ 1, 250 }, .{ 1, 3 }, .{ 1, 64 }, .{ 1, 129 } } },
        } else [_]TestCase{
            .{ .prefill = true, .seqs = .{ .{ 37, 4500 }, .{ 1, 77 }, .{ 70, 70 }, .{ 0, 0 } } },
            .{ .prefill = false, .seqs = .{ .{ 1, 5000 }, .{ 1, 3 }, .{ 1, 4097 }, .{ 1, 129 } } },
        };
        try runCases(max_pages, 2 * max_pages, &cases, .{});
    }
}

test "tcl paged attention options and inactive queries on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const sinks = [_]f32{ -std.math.inf(f32), -5, 0, 2, 8, -1, 1, 20 };
    const disabled: [8]f32 = @splat(-std.math.inf(f32));
    const mixed = [4][2]i32{ .{ 37, 200 }, .{ 0, 0 }, .{ 1, 77 }, .{ 20, 20 } };
    const cases = [_]TestCase{
        .{ .prefill = true, .seqs = mixed, .padding = 5, .opts = .{ .sliding_window = 17, .scale = 0.2, .sink = &sinks } },
        .{ .prefill = true, .seqs = mixed, .padding = 5, .opts = .{ .is_causal = false, .sliding_window = 17, .scale = 0.2, .sink = &sinks } },
        .{ .prefill = true, .seqs = mixed, .padding = 5, .opts = .{ .is_causal = false, .sliding_window = 0 } },
        .{ .prefill = true, .seqs = mixed, .padding = 5, .opts = .{ .sliding_window = 1, .sink = &disabled } },
        .{ .prefill = false, .seqs = .{ .{ 1, 251 }, .{ 0, 73 }, .{ 1, 1 }, .{ 0, 17 } }, .padding = 2, .opts = .{ .sink = &sinks } },
        .{ .prefill = false, .seqs = @splat(.{ 0, 0 }), .padding = 4 },
        .{ .prefill = false, .seqs = @splat(.{ 1, 17 }), .mask_all = true },
        .{ .prefill = false, .seqs = @splat(.{ 1, 17 }), .mask_all = true, .opts = .{ .sink = &sinks } },
        .{ .prefill = false, .seqs = .{ .{ 1, 251 }, .{ 0, 73 }, .{ 1, 20 }, .{ 0, 17 } }, .padding = 2, .opts = .{ .is_causal = false, .sliding_window = 17, .scale = 0.2, .sink = &sinks } },
        .{ .prefill = false, .seqs = .{ .{ 1, 251 }, .{ 0, 73 }, .{ 1, 20 }, .{ 0, 17 } }, .padding = 2, .opts = .{ .sliding_window = 1, .sink = &disabled } },
    };
    try runCases(16, 32, &cases, .{});
}

test "tcl paged attention chunk boundaries on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const cases = [_]TestCase{
        .{ .prefill = true, .seqs = .{ .{ 37, 4097 }, .{ 0, 0 }, .{ 1, 4096 }, .{ 2, 4095 } }, .padding = 7, .opts = .{ .is_causal = false, .sliding_window = 33, .scale = 0.2, .sink = &.{ 0, 10 } } },
        .{ .prefill = false, .seqs = .{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 1, 4097 }, .{ 0, 0 } }, .padding = 1, .opts = .{ .sliding_window = 17, .sink = &.{ -std.math.inf(f32), 0 } } },
        .{ .prefill = false, .seqs = @splat(.{ 0, 0 }), .padding = 4 },
        .{ .prefill = false, .seqs = @splat(.{ 1, 4097 }), .mask_all = true, .opts = .{ .sink = &.{ -std.math.inf(f32), 0 } } },
    };
    try runCases(257, 514, &cases, .{ .hkv = 1, .g = 2, .d = 32 });
    try runCases(256, 512, &.{.{ .prefill = false, .seqs = .{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 0, 0 }, .{ 0, 0 } }, .padding = 2 }}, .{ .hkv = 1, .g = 1, .d = 32 });
    try runCases(86, 172, &.{.{ .prefill = false, .seqs = .{ .{ 1, 4079 }, .{ 1, 4080 }, .{ 1, 4081 }, .{ 0, 0 } }, .padding = 1, .opts = .{ .sliding_window = 49 } }}, .{ .hkv = 1, .g = 1, .d = 32, .page = 48 });
}

test "tcl paged attention preserves constant values across chunks" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const cases = [_]TestCase{
        .{ .prefill = false, .seqs = .{ .{ 1, 64 }, .{ 1, 65 }, .{ 1, 66 }, .{ 0, 0 } }, .padding = 1 },
        .{ .prefill = false, .seqs = .{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 1, 4097 }, .{ 0, 0 } }, .padding = 1 },
        .{ .prefill = false, .seqs = .{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 1, 4097 }, .{ 0, 0 } }, .padding = 1, .opts = .{ .sliding_window = 1, .sink = &.{0} } },
        .{ .prefill = true, .seqs = .{ .{ 37, 4097 }, .{ 0, 0 }, .{ 1, 77 }, .{ 2, 2 } }, .padding = 3 },
    };
    try runCases(257, 514, &cases, .{ .hkv = 1, .g = 1, .d = 32, .uniform = true });
}

test "tcl paged attention chunked decode on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const cases = [_]TestCase{
        .{ .prefill = false, .seqs = .{ .{ 1, 64 }, .{ 1, 65 }, .{ 1, 66 }, .{ 0, 0 } }, .padding = 1 },
        .{ .prefill = false, .seqs = .{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 1, 4097 }, .{ 0, 0 } }, .padding = 1 },
    };
    try runCases(257, 514, &cases, .{ .hkv = 1, .g = 1, .d = 32 });
}

test "tcl paged attention uniform probabilities on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    try runCases(257, 514, &.{.{ .prefill = false, .seqs = .{ .{ 1, 64 }, .{ 1, 65 }, .{ 1, 66 }, .{ 0, 0 } }, .padding = 1 }}, .{ .hkv = 1, .g = 1, .d = 32, .uniform_scores = true });
}

fn runCases(comptime max_pages: usize, comptime num_pages: usize, cases: []const TestCase, comptime dims: TestDims) !void {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bf16 = zml.floats.BFloat16;
    const hkv = dims.hkv;
    const g = dims.g;
    const d = dims.d;
    const page = dims.page;
    const batch = dims.batch;
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    const kc = try allocator.alloc(bf16, num_pages * page * hkv * d);
    defer allocator.free(kc);
    const vc = try allocator.alloc(bf16, num_pages * page * hkv * d);
    defer allocator.free(vc);
    for (kc) |*v| v.* = .fromF32(if (dims.uniform or dims.uniform_scores) 0 else r.floatNorm(f32));
    for (vc, 0..) |*v, i| v.* = .fromF32(if (dims.uniform) @as(f32, @floatFromInt(i % d)) / d else r.floatNorm(f32));
    const k_t: Tensor = .init(.{ .page = num_pages, .k_chunk = page, .hkv = hkv, .hd = d }, dims.dtype);
    var bk = try testBuffer(platform, k_t.shape(), kc);
    defer bk.deinit();
    var bv = try testBuffer(platform, k_t.shape(), vc);
    defer bv.deinit();

    for (cases) |case| {
        var tokens: i64 = case.padding;
        for (case.seqs[0..batch]) |s| tokens += s[0];
        const params: paged.Parameters = .init(.{ .batch_size = batch, .max_num_pages = max_pages, .max_seqlen_q = 128, .is_prefill = case.prefill });
        const q_t: Tensor = .init(.{ .b = tokens, .hkv = hkv, .hg = g, .hd = d }, dims.dtype);
        const Mod = struct {
            pub fn forward(p: paged.Parameters, q: Tensor, k: Tensor, v: Tensor, opts: paged_attention.AttentionOptions) Tensor {
                return paged_attention.pagedAttention(.{ .tcl = p }, q, k, v, .{ .split = .{ .k = k, .v = v } }, opts);
            }
        };
        const opts: paged_attention.AttentionOptions = .{
            .mask = .{ .standard = .{
                .causal = case.opts.is_causal,
                .sliding_window = if (case.opts.sliding_window > 0) @intCast(case.opts.sliding_window) else null,
            } },
            .scale = case.opts.scale,
            .sink = if (case.opts.sink != null) Tensor.init(.{ .h = hkv * g }, .f32) else null,
        };
        const v_t: Tensor = .init(k_t.shape(), dims.dtype);
        var exe = try zml.module.compile(allocator, io, Mod.forward, .{ params, q_t, k_t, v_t, opts }, platform, .{});
        defer exe.deinit();

        const q = try allocator.alloc(bf16, @intCast(tokens * hkv * g * d));
        defer allocator.free(q);
        for (q) |*v| v.* = .fromF32(if (dims.uniform or dims.uniform_scores) 0 else r.floatNorm(f32));
        var table: [batch][max_pages]i32 = @splat(@splat(case.padding_page));
        var seq_lens: [batch]i32 = undefined;
        var starts: [batch + 1]i32 = undefined;
        starts[0] = 0;
        var next_page: i32 = 0;
        for (case.seqs[0..batch], 0..) |s, seq| {
            seq_lens[seq] = s[1];
            starts[seq + 1] = starts[seq] + s[0];
            if (!case.mask_all) {
                for (0..@intCast(std.math.divCeil(i32, s[1], page) catch unreachable)) |p| {
                    table[seq][p] = @mod(next_page * 7, @as(i32, num_pages));
                    next_page += 1;
                }
            }
        }
        var bq = try testBuffer(platform, q_t.shape(), q);
        defer bq.deinit();
        var bp: zml.Bufferized(paged.Parameters) = .{
            .block_table = try .fromBytes(io, platform, params.block_table.shape(), std.mem.sliceAsBytes(&table)),
            .seq_lens = try .fromBytes(io, platform, params.seq_lens.shape(), std.mem.sliceAsBytes(&seq_lens)),
            .query_start_len = try .fromBytes(io, platform, params.query_start_len.shape(), std.mem.sliceAsBytes(&starts)),
        };
        defer zml.Buffer.deinitAll(paged.Parameters, &bp);
        var opts_d: zml.Bufferized(paged_attention.AttentionOptions) = .{
            .mask = .{ .standard = {} },
            .sink = if (case.opts.sink) |s| try .fromBytes(io, platform, opts.sink.?.shape(), std.mem.sliceAsBytes(s)) else null,
        };
        defer zml.Buffer.deinitAll(paged_attention.AttentionOptions, &opts_d);
        var result = try exe.eval(allocator, io, .{ bp, bq, bk, bv, opts_d });
        defer result.deinit();
        const host = try result.toSliceAlloc(allocator, io);
        defer host.free(allocator);
        const expected = try attention_test.reference(
            allocator,
            .initConst(q_t.shape().withDtype(.bf16), std.mem.sliceAsBytes(q)),
            .initConst(k_t.shape().withDtype(.bf16), std.mem.sliceAsBytes(kc)),
            .initConst(k_t.shape().withDtype(.bf16), std.mem.sliceAsBytes(vc)),
            @as([*]const i32, @ptrCast(&table))[0 .. batch * max_pages],
            &seq_lens,
            &starts,
            case.opts,
        );
        defer expected.free(allocator);
        try std.testing.expectEqual(dims.dtype, host.shape.dtype());
        attention_test.expectClose(expected, host) catch |err| {
            std.log.err("TCL attention fixture: {any}", .{case});
            return err;
        };
        const active_elements = if (case.mask_all) 0 else @as(usize, @intCast(starts[batch])) * hkv * g * d;
        for (active_elements..@intCast(tokens * hkv * g * d)) |index| try std.testing.expectEqual(@as(f32, 0), attention_test.floatAt(host, index));
        for (case.seqs[0..batch], 0..) |seq, index| {
            const empty_queries: usize = @intCast(@max(0, seq[0] - seq[1]));
            const begin: usize = @as(usize, @intCast(starts[index])) * hkv * g * d;
            for (begin..begin + empty_queries * hkv * g * d) |element| try std.testing.expectEqual(@as(f32, 0), attention_test.floatAt(host, element));
        }
    }
}

fn testBuffer(platform: *const zml.Platform, shape: zml.Shape, values: []const zml.floats.BFloat16) !zml.Buffer {
    const slice = try zml.Slice.alloc(std.testing.allocator, shape);
    defer slice.free(std.testing.allocator);
    switch (shape.dtype()) {
        .bf16 => @memcpy(slice.items(zml.floats.BFloat16), values),
        .f16 => for (slice.items(f16), values) |*out, value| {
            out.* = @floatCast(value.toF32());
        },
        .f32 => for (slice.items(f32), values) |*out, value| {
            out.* = value.toF32();
        },
        else => unreachable,
    }
    return .fromBytes(std.testing.io, platform, shape, slice.bytes);
}
