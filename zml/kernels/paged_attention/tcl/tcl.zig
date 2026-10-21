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
//!
//! Decode without sinks or windows, up to `decode_keys` keys, instead runs one
//! fusible kernel per layer over the whole pages each token sees, with the
//! token metadata every layer shares computed once per forward pass.
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

    /// Token counts up to which `writeCache` writes rows one by one.
    const row_writes = 32;

    /// Writes token rows `{.b, .hkv, .hd}` at their `page` and `offset`.
    /// Tokens without a slot carry indices past the cache, which both paths
    /// clamp onto the cache's last page: the caller must keep that page out of
    /// its allocation. A few rows become dynamic updates, which compile into
    /// the unit that computes them; more rows go through one in-place scatter
    /// kernel instead of XLA's scatter, which the plugin expands into a host
    /// loop over rows.
    pub fn writeCache(cache: Tensor, rows: Tensor, page: Tensor, offset: Tensor) Tensor {
        if (rows.dim(.b) > row_writes) {
            const last_page = Tensor.scalar(cache.dim(.page) - 1, .i32);
            const last_offset = Tensor.scalar(cache.dim(.k_chunk) - 1, .i32);
            const slots = page.convert(.i32).clamp(Tensor.scalar(0, .i32), last_page).scale(cache.dim(.k_chunk))
                .add(offset.convert(.i32).clamp(Tensor.scalar(0, .i32), last_offset));
            return WriteRows.call(
                .{ .cache = cache, .rows = rows.convert(cache.dtype()), .slots = slots },
                .{ .out = cache.shape() },
                .{ .cfg = .{ .t = rows.dim(.b), .pages = cache.dim(.page), .pg = cache.dim(.k_chunk), .hkv = cache.dim(.hkv), .d = cache.dim(.hd), .dtype = tcl.from(cache.dtype()) } },
            ).out;
        }
        var out = cache;
        for (0..@intCast(rows.dim(.b))) |t| {
            const token: Tensor.Slice = .{ .start = @intCast(t), .end = @intCast(t + 1) };
            out = out.dynamicUpdateSlice(.{
                .page = page.slice(.b, token).squeeze(.b),
                .k_chunk = offset.slice(.b, token).squeeze(.b),
            }, rows.slice(.b, token).squeeze(.b));
        }
        return out;
    }

    /// `q` is `{.b = tokens, .hkv, .hg, .hd}`, the caches
    /// `{.page, .k_chunk, .hkv, .hd}`. Sinks are one logit per query head.
    /// Decode reads whole pages, past each sequence's end and from pages of
    /// other sequences: the caches must hold numbers, so zero them once.
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
        // Chunks hold whole pages and a multiple of 64 keys: TCC 2026.5 fails
        // to allocate a chunk of 4080 keys (85 pages of 48).
        const key_alignment = @divExact(page, @as(i64, @intCast(std.math.gcd(@as(u64, @intCast(page)), 64)))) * 64;
        const alignment = if (key_alignment <= chunk_keys) key_alignment else page;
        const chunk = if (keys <= chunk_keys) keys else @min(keys, @max(1, @divTrunc(chunk_keys, alignment)) * alignment);
        const decode = !parameters.options_.is_prefill and q.dim(.b) == parameters.seq_lens.dim(.b);
        const blocks = (std.math.divCeil(i64, q.dim(.b), prefill_block) catch unreachable) + parameters.seq_lens.dim(.b);
        // Regular block capacities limit layout-search cost for the fused graph.
        // Extra blocks retain zero valid lengths and invalid query positions.
        const cfg: Cfg = .{
            .tokens = q.dim(.b),
            .sequences = parameters.seq_lens.dim(.b),
            .b = if (decode) q.dim(.b) else @intCast(std.math.ceilPowerOfTwo(u64, @intCast(blocks)) catch unreachable),
            .i = if (decode) 1 else prefill_block,
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
        // TCC 2026.5 fails on the decode metadata for some page counts (257,
        // 320): round them up to a power of two.
        const decode_width = @as(i64, @intCast(std.math.ceilPowerOfTwo(u64, @intCast(@divExact(keys, page))) catch unreachable)) * page;
        if (decode and !cfg.use_sink and cfg.window == 0 and decode_width <= decode_keys) {
            return decodeAttention(parameters, q, k_cache, v_cache, .{
                .t = cfg.tokens,
                .a = decode_width,
                .hkv = cfg.hkv,
                .g = cfg.g,
                .d = cfg.d,
                .pages = cfg.pages,
                .pg = page,
                .scale = cfg.scale,
                .is_causal = cfg.is_causal,
            }).convert(q.dtype());
        }
        const table = parameters.block_table.pad(-1, .{ .p = Tensor.Pad{ .high = @divExact(cfg.a, page) - parameters.block_table.dim(.p) } });
        const inputs: OneShot.Inputs = .{
            .q = q.convert(.bf16),
            .pages = table,
            .seq_lens = parameters.seq_lens,
            .query_offsets = parameters.query_start_len,
            .k_cache = k_cache.convert(.bf16),
            .v_cache = v_cache.convert(.bf16),
            .sink = if (opts.sink) |s| s.convert(.f32).reshape(.{ .hkv = cfg.hkv, .hg = cfg.g }) else Tensor.zeroes(.init(.{ .hkv = cfg.hkv, .hg = cfg.g }, .f32)),
        };
        const outputs: OneShot.Outputs = .{ .out = q.shape().withDtype(.bf16) };
        const out = if (cfg.a == cfg.chunk)
            OneShot.call(inputs, outputs, .{ .cfg = cfg }).out
        else
            Flash.call(inputs, outputs, .{ .cfg = cfg }).out;
        return out.convert(q.dtype());
    }
};

const WriteCfg = struct {
    t: i64,
    pages: i64,
    pg: i64,
    hkv: i64,
    d: i64,
    dtype: tcl.DType,
};

/// `cache[slots[t]] = rows[t]` over the cache's token slots, in place.
const WriteRows = tcl.Kernel(WriteCfg, .{
    .name = "paged_write_rows",
    .inputs = &.{ "cache", "rows", "slots" },
    .outputs = &.{"out"},
    .run = struct {
        fn run(b: *tcl.Builder, cfg: WriteCfg) tcl.FinishError!void {
            const T = b.axis("T", cfg.t);
            const P = b.axis("P", cfg.pages);
            const PG = b.axis("PG", cfg.pg);
            const N = b.axis("N", cfg.pages * cfg.pg);
            const Hkv = b.axis("Hkv", cfg.hkv);
            const D = b.axis("D", cfg.d);
            const t = try b.declareArgs(.{
                .cache = .{ .dtype = cfg.dtype, .axes = &.{ P, PG, Hkv, D }, .dram = .{} },
                .rows = .{ .dtype = cfg.dtype, .axes = &.{ T, Hkv, D } },
                .slots = .{ .dtype = .i32, .axes = &.{T} },
            });
            const flat = try b.reshape(t.cache, &.{ N, Hkv, D }, .{});
            const written = try b.scatter(t.rows, t.slots, N, &.{ N, Hkv, D }, .{ .init = flat });
            b.ret(&.{try b.update(0, try b.reshape(written, &.{ P, PG, Hkv, D }, .{}))});
        }
    }.run,
});

/// Keys a decode kernel attends to in one pass: TCC 2026.5 fails on the
/// decode metadata for 131072 keys in pages of 16.
const decode_keys = 65536;

const DecodeCfg = struct {
    t: i64,
    a: i64,
    hkv: i64,
    g: i64,
    d: i64,
    pages: i64,
    pg: i64,
    scale: f32,
    is_causal: bool,
};

/// Decode attention in Furiosa's form: the kernel gathers whole pages, then
/// runs one contraction, a masked softmax and a second contraction. It reads
/// past each sequence's end, so the caches must hold numbers (zero them once).
/// The token metadata depends only on the attention parameters, which every
/// layer shares: one kernel computes it once per forward pass.
fn decodeAttention(parameters: paged.Parameters, q: Tensor, k_cache: Tensor, v_cache: Tensor, cfg: DecodeCfg) Tensor {
    const columns = @divExact(cfg.a, cfg.pg);
    const table = parameters.block_table.pad(0, .{ .p = Tensor.Pad{ .high = columns - parameters.block_table.dim(.p) } });
    const meta = DecodeMeta.call(.{ .seq_lens = parameters.seq_lens, .query_offsets = parameters.query_start_len, .table = table }, .{
        .pages = zml.Shape.init(.{ .t = cfg.t, .p = columns }, .i32),
        .bounds = zml.Shape.init(.{ .t = cfg.t }, .i32),
        .mask = zml.Shape.init(.{ .t = cfg.t, .a = cfg.a }, .i32),
        .keyed = zml.Shape.init(.{ .t = cfg.t }, .f32),
    }, .{ .cfg = .{ .t = cfg.t, .s = parameters.seq_lens.dim(.b), .ap = columns, .pg = cfg.pg, .pages = cfg.pages, .is_causal = cfg.is_causal } });
    return Decode.call(.{
        .q = q.convert(.bf16),
        .pages = meta.pages,
        .bounds = meta.bounds,
        .mask = meta.mask,
        .keyed = meta.keyed,
        .k_cache = k_cache.convert(.bf16),
        .v_cache = v_cache.convert(.bf16),
    }, .{ .out = q.shape().withDtype(.bf16) }, .{ .cfg = cfg }).out;
}

const MetaCfg = struct { t: i64, s: i64, ap: i64, pg: i64, pages: i64, is_causal: bool };

/// For each token: the cache page of each column, how many columns hold keys
/// it sees, the keys it sees (below its limit, in pages inside the cache) and
/// whether it sees any. Columns past the sequence's
/// pages read page `p`, so that the masked padding spreads over many pages
/// instead of hammering one. Rows are selected by one-hot products and
/// comparisons follow `attentionMask`'s clamp arithmetic: TCC 2026.5 has no
/// tactics for small integer gathers and fails on the plugin's lowering of the
/// same comparisons for several tokens.
const DecodeMeta = tcl.Kernel(MetaCfg, .{
    .name = "paged_decode_metadata",
    .inputs = &.{ "seq_lens", "query_offsets", "table" },
    .outputs = &.{ "pages", "bounds", "mask", "keyed" },
    .run = struct {
        fn run(b: *tcl.Builder, cfg: MetaCfg) tcl.FinishError!void {
            const T = b.axis("T", cfg.t);
            const S = b.axis("S", cfg.s);
            const O = b.axis("O", cfg.s + 1);
            const AP = b.axis("AP", cfg.ap);
            const PG = b.axis("PG", cfg.pg);
            const A = b.axis("A", cfg.ap * cfg.pg);
            const t = try b.declareArgs(.{
                .seq_lens = .{ .dtype = .i32, .axes = &.{S} },
                .query_offsets = .{ .dtype = .i32, .axes = &.{O} },
                .table = .{ .dtype = .i32, .axes = &.{ S, AP } },
            });
            const ends = try selectMetadata(b, t.query_offsets, try selector(b, try b.arange(cfg.s + 1, .i32, &.{S}, .{ .start = 1 }), O, cfg.s + 1), O);
            const tokens = try b.arange(cfg.t, .i32, &.{T}, .{});
            // Padding tokens follow every sequence's end and belong to none.
            const seq = try countBefore(b, ends, tokens, S);
            const own = try selector(b, seq, S, cfg.s);
            var op = b.tensorOperation(.{});
            const active = try op.commit(op.fetch(seq, .{}).muli(-1).addi(cfg.s).binary(.maxi, 0).binary(.mini, 1), .{});
            var visible = try selectMetadata(b, t.seq_lens, own, S);
            if (cfg.is_causal) {
                // Causal tokens see their sequence up to their own position.
                const end = try selectMetadata(b, ends, own, S);
                op = b.tensorOperation(.{});
                visible = try op.commit(op.fetch(visible, .{}).subi(op.fetch(end, .{})).addi(op.fetch(tokens, .{})).addi(1), .{});
            }
            op = b.tensorOperation(.{});
            const limit = try op.commit(op.fetch(visible, .{}).binary(.maxi, 0).binary(.mini, cfg.ap * cfg.pg).muli(op.fetch(active, .{})), .{});
            const entries = try selectMetadata(b, t.table, own, S);
            const pages = try validIndices(b, entries, cfg.pages);
            const firsts = try b.arange(cfg.ap * cfg.pg, .i32, &.{AP}, .{ .step = cfg.pg });
            op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
            const used = try op.commit(op.fetch(limit, .{}).subi(op.fetch(firsts, .{})).binary(.maxi, 0).binary(.mini, 1), .{});
            op = b.tensorOperation(.{});
            const bounds = try op.commit(op.fetch(used, .{}).reduce(&.{AP}, .addi), .{});
            op = b.tensorOperation(.{});
            const spread = try op.commit(op.fetch(try b.arange(cfg.ap, .i32, &.{AP}, .{}), .{}).binary(.mini, cfg.pages - 1), .{});
            op = b.tensorOperation(.{});
            const read = try op.commit(op.where(op.fetch(used, .{}), .ne, 0, op.fetch(pages.safe, .{}), op.fetch(spread, .{ .broadcast_to = &.{ T, AP } })), .{});
            // Tokens without keys have no defined attention: the attention
            // kernel zeroes them.
            op = b.tensorOperation(.{});
            const keyed = try op.commit(op.fetch(used, .{}).muli(op.fetch(pages.valid, .{})).reduce(&.{AP}, .maxi).toFloat(), .{});
            op = b.tensorOperation(.{});
            const valid = try op.commit(op.fetch(pages.valid, .{ .broadcast_to = &.{ T, AP, PG } }), .{});
            op = b.tensorOperation(.{});
            const position = try op.commit(op.fetch(try b.arange(cfg.ap * cfg.pg, .i32, &.{A}, .{}), .{ .broadcast_to = &.{ T, A } }), .{});
            op = b.tensorOperation(.{ .tactic = .elementwise });
            const mask = try op.commit(op.fetch(position, .{}).muli(-1).addi(op.fetch(limit, .{})).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(try b.reshape(valid, &.{ T, A }, .{}), .{})), .{});
            b.ret(&.{ read, bounds, mask, keyed });
        }
    }.run,
});

/// Gathers the key and value pages a token sees, contracts with the query,
/// masks keys past the sequence, normalizes in one softmax times `keyed` (zero
/// for tokens without keys, whose softmax is uniform) and contracts with the
/// values. A token's pages are gathered up to its bound, and the operations
/// reading them skip the rest: the work follows the sequence's length, not the
/// table's. Fusible: it joins the layer around it, one launch per layer.
const Decode = tcl.Kernel(DecodeCfg, .{
    .name = "paged_decode_attention",
    .inputs = &.{ "q", "pages", "bounds", "mask", "keyed", "k_cache", "v_cache" },
    .outputs = &.{"out"},
    .run = struct {
        fn run(b: *tcl.Builder, cfg: DecodeCfg) tcl.FinishError!void {
            b.fuse();
            configure(b, .{ .tokens = cfg.t, .sequences = cfg.t, .b = cfg.t, .i = 1, .a = cfg.a, .chunk = cfg.a, .hkv = cfg.hkv, .g = cfg.g, .d = cfg.d, .pages = cfg.pages, .pg = cfg.pg, .scale = cfg.scale, .decode = true });
            const T = b.axis("T", cfg.t);
            const A = b.axis("A", cfg.a);
            const AP = b.axis("AP", @divExact(cfg.a, cfg.pg));
            const Hkv = b.axis("Hkv", cfg.hkv);
            const G = b.axis("G", cfg.g);
            const D = b.axis("D", cfg.d);
            const P = b.axis("P", cfg.pages);
            const PG = b.axis("PG", cfg.pg);
            const t = try b.declareArgs(.{
                .q = .{ .dtype = .bf16, .axes = &.{ T, Hkv, G, D } },
                .pages = .{ .dtype = .i32, .axes = &.{ T, AP } },
                .bounds = .{ .dtype = .i32, .axes = &.{T} },
                .mask = .{ .dtype = .i32, .axes = &.{ T, A } },
                .keyed = .{ .dtype = .f32, .axes = &.{T} },
                .k_cache = .{ .dtype = .bf16, .axes = &.{ P, PG, Hkv, D } },
                .v_cache = .{ .dtype = .bf16, .axes = &.{ P, PG, Hkv, D } },
            });
            const k = try b.gather(t.k_cache, t.pages, P, &.{ T, AP, PG, Hkv, D }, .{ .valid_length = t.bounds });
            const v = try b.gather(t.v_cache, t.pages, P, &.{ T, AP, PG, Hkv, D }, .{ .valid_length = t.bounds });
            const mask = try b.reshape(t.mask, &.{ T, AP, PG }, .{});
            const keys: []const tcl.Axis = &.{ AP, PG };
            var op = b.tensorOperation(.{});
            const scores = try op.commit(op.contract(t.q, k, &.{ Hkv, G, T, AP, PG }).mulf(cfg.scale), .{});
            op = b.tensorOperation(.{});
            const s = op.fetch(scores, .{});
            const masked = try op.commit(op.where(op.fetch(mask, .{}), .ne, 0, s, score_min), .{});
            op = b.tensorOperation(.{});
            const x = op.fetch(masked, .{});
            const e = x.subf(x.reduce(keys, .maxf)).exp();
            const p = try op.commit(e.divf(e.reduce(keys, .addf)).mulf(op.fetch(t.keyed, .{})), .{ .dtype = .bf16 });
            op = b.tensorOperation(.{});
            b.ret(&.{try op.commit(op.contract(v, p, &.{ T, Hkv, G, D }), .{ .dtype = .bf16 })});
        }
    }.run,
});

const Cfg = struct {
    tokens: i64,
    sequences: i64,
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
    T: tcl.Axis,
    S: tcl.Axis,
    O: tcl.Axis,
    AP: tcl.Axis,
    B: tcl.Axis,
    I: tcl.Axis,
    Hkv: tcl.Axis,
    G: tcl.Axis,
    D: tcl.Axis,
    P: tcl.Axis,
    PG: tcl.Axis,

    fn init(b: *tcl.Builder, cfg: Cfg) Axes {
        return .{
            .T = b.axis("T", cfg.tokens),
            .S = b.axis("S", cfg.sequences),
            .O = b.axis("O", cfg.sequences + 1),
            .AP = b.axis("AP", @divExact(cfg.a, cfg.pg)),
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

const NativeInputs = struct {
    q: builder.Tensor,
    pages: builder.Tensor,
    seq_lens: builder.Tensor,
    query_offsets: builder.Tensor,
    k_cache: builder.Tensor,
    v_cache: builder.Tensor,
    sink: builder.Tensor,
};

fn declareInputs(b: *tcl.Builder, a: Axes) tcl.FinishError!NativeInputs {
    const t = try b.declareArgs(.{
        .q = .{ .dtype = .bf16, .axes = &.{ a.T, a.Hkv, a.G, a.D } },
        .pages = .{ .dtype = .i32, .axes = &.{ a.S, a.AP } },
        .seq_lens = .{ .dtype = .i32, .axes = &.{a.S} },
        .query_offsets = .{ .dtype = .i32, .axes = &.{a.O} },
        .k_cache = .{ .dtype = .bf16, .axes = &.{ a.P, a.PG, a.Hkv, a.D } },
        .v_cache = .{ .dtype = .bf16, .axes = &.{ a.P, a.PG, a.Hkv, a.D } },
        .sink = .{ .dtype = .f32, .axes = &.{ a.Hkv, a.G } },
    });
    return .{ .q = t.q, .pages = t.pages, .seq_lens = t.seq_lens, .query_offsets = t.query_offsets, .k_cache = t.k_cache, .v_cache = t.v_cache, .sink = t.sink };
}

const Prepared = struct {
    q: builder.Tensor,
    pages: builder.Tensor,
    qpos: builder.Tensor,
    kvlen: builder.Tensor,
    row: builder.Tensor,
    active: builder.Tensor,
};

fn countBefore(b: *tcl.Builder, ends: builder.Tensor, positions: builder.Tensor, axis: tcl.Axis) tcl.FinishError!builder.Tensor {
    var op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
    const distance = try op.commit(op.fetch(ends, .{}).subi(op.fetch(positions, .{})), .{});
    op = b.tensorOperation(.{});
    const passed = try op.commit(op.where(distance, .le, 0, 1, 0), .{});
    op = b.tensorOperation(.{});
    return op.commit(op.fetch(passed, .{}).reduce(&.{axis}, .addi), .{});
}

fn selector(b: *tcl.Builder, indices: builder.Tensor, axis: tcl.Axis, count: i64) tcl.FinishError!builder.Tensor {
    const source = try b.arange(count, .i32, &.{axis}, .{});
    var op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
    const distance = try op.commit(op.fetch(indices, .{}).subi(op.fetch(source, .{})), .{});
    op = b.tensorOperation(.{});
    return op.commit(op.where(distance, .eq, 0, 1, 0), .{});
}

// Keep metadata exact in I32. Scalar gathers and four-byte-offset slices do
// not have native tactics for these shapes in TCC 2026.5.
fn selectMetadata(b: *tcl.Builder, values: builder.Tensor, selected: builder.Tensor, axis: tcl.Axis) tcl.FinishError!builder.Tensor {
    const op = b.tensorOperation(.{ .tactic = if (values.axes.len > 1) .einsum_by_ve else .reduce_by_ve });
    return op.commit(op.fetch(selected, .{}).muli(op.fetch(values, .{})).reduce(&.{axis}, .addi), .{});
}

/// Keep adjacent I32 reads inside the native unit. Exposing offsets[1:] as a
/// separate argument would require alignment staging at every invocation.
fn prepareQueries(b: *tcl.Builder, cfg: Cfg, a: Axes, t: NativeInputs) tcl.FinishError!Prepared {
    const start_indices = try b.arange(cfg.sequences, .i32, &.{a.S}, .{});
    const end_indices = try b.arange(cfg.sequences + 1, .i32, &.{a.S}, .{ .start = 1 });
    const starts = try selectMetadata(b, t.query_offsets, try selector(b, start_indices, a.O, cfg.sequences + 1), a.O);
    const ends = try selectMetadata(b, t.query_offsets, try selector(b, end_indices, a.O, cfg.sequences + 1), a.O);
    var op = b.tensorOperation(.{});
    const qlen = try op.commit(op.fetch(ends, .{}).subi(op.fetch(starts, .{})), .{});
    const tokens = try b.arange(cfg.tokens, .i32, &.{a.T}, .{});
    const seq_of_token = try countBefore(b, ends, tokens, a.S);
    // All sequence ends precede a padding token. Reuse this vector instead of
    // bringing a symbolic scalar reduction back into a vector operation.
    op = b.tensorOperation(.{});
    const token_tail = try op.commit(op.fetch(seq_of_token, .{}).subi(cfg.sequences), .{});
    op = b.tensorOperation(.{});
    const active = try op.commit(op.where(token_tail, .lt, 0, 1, 0), .{});
    op = b.tensorOperation(.{});
    const token_seq = try op.commit(op.fetch(seq_of_token, .{}).binary(.mini, cfg.sequences - 1), .{});
    if (cfg.decode) {
        op = b.tensorOperation(.{});
        const live = try op.commit(op.where(qlen, .gt, 0, 1, 0), .{});
        op = b.tensorOperation(.{});
        const lengths = try op.commit(op.fetch(t.seq_lens, .{}).muli(op.fetch(live, .{})), .{});
        op = b.tensorOperation(.{});
        const qpos = try op.commit(op.fetch(lengths, .{}).subi(1), .{});
        op = b.tensorOperation(.{});
        const safe_start = try op.commit(op.fetch(starts, .{}).binary(.mini, cfg.tokens - 1), .{});
        const q = if (cfg.tokens == 1) t.q else try b.gather(t.q, safe_start, a.T, &.{ a.S, a.Hkv, a.G, a.D }, .{});
        return .{
            .q = try b.reshape(q, &.{ a.B, a.I, a.Hkv, a.G, a.D }, .{}),
            .pages = try b.reshape(t.pages, &.{ a.B, a.AP }, .{}),
            .qpos = try b.reshape(qpos, &.{ a.B, a.I }, .{}),
            .kvlen = try b.reshape(lengths, &.{a.B}, .{}),
            .row = token_seq,
            .active = active,
        };
    }

    const shift = std.math.log2_int(u64, prefill_block);
    op = b.tensorOperation(.{});
    const blocks = try op.commit(op.fetch(qlen, .{}).addi(prefill_block - 1).binary(.shr_arith, shift), .{});
    op = b.tensorOperation(.{});
    const block_end = try op.commit(op.fetch(blocks, .{}).cumsum(&.{a.S}), .{});
    op = b.tensorOperation(.{});
    const block_start = try op.commit(op.fetch(block_end, .{}).subi(op.fetch(blocks, .{})), .{});
    const block_ids = try b.arange(cfg.b, .i32, &.{a.B}, .{});
    const seq_of_block = try countBefore(b, block_end, block_ids, a.S);
    op = b.tensorOperation(.{});
    const beyond = try op.commit(op.fetch(seq_of_block, .{}).subi(cfg.sequences), .{});
    op = b.tensorOperation(.{});
    const live_block = try op.commit(op.where(beyond, .lt, 0, 1, 0), .{});
    op = b.tensorOperation(.{});
    const seq = try op.commit(op.fetch(seq_of_block, .{}).binary(.mini, cfg.sequences - 1), .{});
    const pick = try selector(b, seq, a.S, cfg.sequences);
    const block_begin = try selectMetadata(b, block_start, pick, a.S);
    const begin = try selectMetadata(b, starts, pick, a.S);
    const query_length = try selectMetadata(b, qlen, pick, a.S);
    const key_length = try selectMetadata(b, t.seq_lens, pick, a.S);
    const local_rows = try b.arange(cfg.i, .i32, &.{a.I}, .{});
    op = b.tensorOperation(.{});
    const first_row = try op.commit(op.fetch(block_ids, .{}).subi(op.fetch(block_begin, .{})).muli(prefill_block), .{});
    op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
    const qi = try op.commit(op.fetch(first_row, .{}).addi(op.fetch(local_rows, .{})), .{});
    op = b.tensorOperation(.{});
    const remaining = try op.commit(op.fetch(qi, .{}).subi(op.fetch(query_length, .{})), .{});
    op = b.tensorOperation(.{});
    const inside = try op.commit(op.where(remaining, .lt, 0, 1, 0), .{});
    op = b.tensorOperation(.{});
    const live = try op.commit(op.fetch(inside, .{}).muli(op.fetch(live_block, .{})), .{});
    op = b.tensorOperation(.{});
    const qidx = try op.commit(op.fetch(qi, .{}).addi(op.fetch(begin, .{})).muli(op.fetch(live, .{})), .{});
    const q = try b.gather(t.q, qidx, a.T, &.{ a.B, a.I, a.Hkv, a.G, a.D }, .{});
    op = b.tensorOperation(.{});
    const pos = try op.commit(op.fetch(qi, .{}).addi(op.fetch(key_length, .{})).subi(op.fetch(query_length, .{})), .{});
    op = b.tensorOperation(.{});
    const qpos = try op.commit(op.where(live, .ne, 0, op.fetch(pos, .{}), -1), .{});
    op = b.tensorOperation(.{});
    const kvlen = try op.commit(op.fetch(key_length, .{}).muli(op.fetch(live_block, .{})), .{});
    const table = try selectMetadata(b, t.pages, pick, a.S);

    const token_pick = try selector(b, token_seq, a.S, cfg.sequences);
    const token_begin = try selectMetadata(b, starts, token_pick, a.S);
    const token_block = try selectMetadata(b, block_start, token_pick, a.S);
    op = b.tensorOperation(.{});
    const row = try op.commit(op.fetch(token_block, .{}).muli(prefill_block).addi(op.fetch(tokens, .{})).subi(op.fetch(token_begin, .{})).binary(.maxi, 0).binary(.mini, cfg.b * cfg.i - 1), .{});
    return .{ .q = q, .pages = table, .qpos = qpos, .kvlen = kvlen, .row = row, .active = active };
}

fn restoreQueries(b: *tcl.Builder, cfg: Cfg, a: Axes, prepared: Prepared, output: builder.Tensor) tcl.FinishError!builder.Tensor {
    if (cfg.decode and cfg.tokens == 1) return b.reshape(output, &.{ a.T, a.Hkv, a.G, a.D }, .{});
    const R = b.axis("R", cfg.b * cfg.i);
    const rows = try b.reshape(output, &.{ R, a.Hkv, a.G, a.D }, .{});
    const ordered = try b.gather(rows, prepared.row, R, &.{ a.T, a.Hkv, a.G, a.D }, .{});
    const op = b.tensorOperation(.{ .tactic = .elementwise });
    // Establish the full data layout before broadcasting the I32 row mask.
    const values = op.fetch(ordered, .{});
    return op.commit(op.where(op.fetch(prepared.active, .{ .broadcast_to = ordered.axes }), .ne, 0, values, 0.0), .{ .dtype = .bf16 });
}

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
    .inputs = &.{ "q", "pages", "seq_lens", "query_offsets", "k_cache", "v_cache", "sink" },
    .outputs = &.{"out"},
    .run = struct {
        fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
            configure(b, cfg);
            const a: Axes = .init(b, cfg);
            const A = b.axis("A", cfg.a);
            const Nb = b.axis("Nb", cfg.pages * cfg.pg);
            const t = try declareInputs(b, a);
            const prepared = try prepareQueries(b, cfg, a, t);
            const kc = try b.reshape(t.k_cache, &.{ Nb, a.Hkv, a.D }, .{});
            const vc = try b.reshape(t.v_cache, &.{ Nb, a.Hkv, a.D }, .{});
            const page_indices = try validIndices(b, prepared.pages, cfg.pages);
            const within_page = try b.arange(cfg.pg, .i32, &.{a.PG}, .{});
            var slot_op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
            const flat_slots = try slot_op.commit(slot_op.fetch(page_indices.safe, .{}).muli(cfg.pg).addi(slot_op.fetch(within_page, .{})), .{});
            const expanded_slots = try b.reshape(flat_slots, &.{ a.B, A }, .{});
            slot_op = b.tensorOperation(.{});
            const valid = try slot_op.commit(slot_op.fetch(page_indices.valid, .{ .broadcast_to = &.{ a.B, a.AP, a.PG } }), .{});
            const valid_keys = try b.reshape(valid, &.{ a.B, A }, .{});
            // Match the original slot sanitization before reading invalid pages.
            slot_op = b.tensorOperation(.{});
            const slots = try slot_op.commit(slot_op.where(valid_keys, .ne, 0, slot_op.fetch(expanded_slots, .{}), 0), .{});
            const valid_length = if (cfg.decode) prepared.kvlen else null;
            const k = try b.gather(kc, slots, Nb, &.{ a.B, A, a.Hkv, a.D }, .{ .valid_length = valid_length });
            const v = try b.gather(vc, slots, Nb, &.{ a.B, A, a.Hkv, a.D }, .{ .valid_length = valid_length });
            const mask = try attentionMask(b, cfg, prepared.qpos, prepared.kvlen, try b.arange(cfg.a, .i32, &.{A}, .{}), valid_keys);
            const masked = try maskedScores(b, a, cfg, prepared.q, k, mask, A);
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
            const output = try op.commit(op.contract(v, p, &.{ a.B, a.I, a.Hkv, a.G, a.D }), .{ .dtype = .bf16 });
            b.ret(&.{try restoreQueries(b, cfg, a, prepared, output)});
        }
    }.run,
});

const Flash = tcl.Kernel(Cfg, .{
    .name = "flash_attention",
    .inputs = &.{ "q", "pages", "seq_lens", "query_offsets", "k_cache", "v_cache", "sink" },
    .outputs = &.{"out"},
    .run = struct {
        fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
            configure(b, cfg);
            const a: Axes = .init(b, cfg);
            const C = b.axis("C", @divExact(cfg.a, cfg.chunk));
            const Ck = b.axis("Ck", cfg.chunk);
            const Cp = b.axis("Cp", @divExact(cfg.chunk, cfg.pg));
            const t = try declareInputs(b, a);
            const prepared = try prepareQueries(b, cfg, a, t);
            const by_batch = try b.reshape(prepared.pages, &.{ a.B, C, Cp }, .{});
            const page_op = b.tensorOperation(.{});
            const pages = try page_op.commit(page_op.fetch(by_batch, .{}), .{ .axes = &.{ C, a.B, Cp } });
            const stats = &.{ a.Hkv, a.G, a.B, a.I };
            const out_axes = &.{ a.B, a.I, a.Hkv, a.G, a.D };
            var op = b.tensorOperation(.{});
            const n = try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(prepared.kvlen), cfg.chunk - 1), cfg.chunk);
            var loop = b.openFor(n, .{ try b.full(0, .f32, stats), try b.full(0, .f32, out_axes), try b.full(score_min, .f32, stats) });
            const sum_acc, const qkv_acc, const max_acc = loop.carried;
            const indices = try validIndices(b, try b.indexRead(pages, loop.iv), cfg.pages);
            const idx = indices.safe;
            op = b.tensorOperation(.{});
            const valid_pages = try op.commit(op.fetch(indices.valid, .{ .broadcast_to = &.{ a.B, Cp, a.PG } }), .{});
            const valid_keys = try b.reshape(valid_pages, &.{ a.B, Ck }, .{});
            const offset = try b.symExpr(.mul, loop.iv, cfg.chunk);
            const local_pos = try b.arange(cfg.chunk, .i32, &.{Ck}, .{});
            op = b.tensorOperation(.{});
            const position = try op.commit(op.fetch(local_pos, .{}).addi(op.fetch(offset, .{})), .{});
            const local_pages = try b.arange(@divExact(cfg.chunk, cfg.pg), .i32, &.{Cp}, .{});
            op = b.tensorOperation(.{});
            const first_keys = try op.commit(op.fetch(local_pages, .{}).muli(cfg.pg).addi(op.fetch(offset, .{})), .{});
            op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
            const remaining = try op.commit(op.fetch(prepared.kvlen, .{}).subi(op.fetch(first_keys, .{})), .{});
            op = b.tensorOperation(.{});
            const live_pages = try op.commit(op.where(remaining, .gt, 0, 1, 0), .{});
            op = b.tensorOperation(.{});
            const cvl = try op.commit(op.fetch(live_pages, .{}).reduce(&.{Cp}, .addi), .{});
            const mask = try attentionMask(b, cfg, prepared.qpos, prepared.kvlen, position, valid_keys);
            const k = try b.reshape(try b.gather(t.k_cache, idx, a.P, &.{ a.B, Cp, a.PG, a.Hkv, a.D }, .{ .valid_length = cvl }), &.{ a.B, Ck, a.Hkv, a.D }, .{});
            const v = try b.reshape(try b.gather(t.v_cache, idx, a.P, &.{ a.B, Cp, a.PG, a.Hkv, a.D }, .{ .valid_length = cvl }), &.{ a.B, Ck, a.Hkv, a.D }, .{});
            const masked = try maskedScores(b, a, cfg, prepared.q, k, mask, Ck);
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
            const output = try op.commit(op.fetch(numerator, .{}).divf(op.fetch(denominator, .{})), .{ .dtype = .bf16 });
            b.ret(&.{try restoreQueries(b, cfg, a, prepared, output)});
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
    seqs: []const [2]i32,
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
    try runCases(16, 32, &.{.{ .prefill = true, .seqs = &@as([4][2]i32, @splat(.{ 32, 256 })), .opts = .{ .scale = 1.0 } }}, .{});
    inline for (.{ .f16, .f32 }) |dtype| {
        try runCases(2, 8, &.{.{ .prefill = false, .seqs = &.{ .{ 1, 17 }, .{ 1, 1 }, .{ 1, 3 }, .{ 0, 0 } }, .padding = 1, .opts = .{ .sink = &.{ 0, -std.math.inf(f32) } } }}, .{ .hkv = 1, .g = 2, .d = 32, .dtype = dtype });
    }
}

test "tcl paged attention single decode row needs no rearrangement" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    try runCases(16, 32, &.{
        .{ .prefill = false, .seqs = &.{ .{ 1, 251 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } } },
        .{ .prefill = false, .seqs = &@as([4][2]i32, @splat(.{ 0, 0 })), .padding = 1 },
    }, .{ .batch = 1 });
}

test "tcl paged attention packs inactive leading sequences and odd batches" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    try runCases(16, 32, &.{
        .{ .prefill = false, .seqs = &.{ .{ 0, 17 }, .{ 1, 63 }, .{ 0, 23 }, .{ 1, 65 } }, .padding = 2 },
        .{ .prefill = true, .seqs = &.{ .{ 0, 17 }, .{ 33, 64 }, .{ 0, 23 }, .{ 2, 3 } }, .padding = 3 },
    }, .{});
    try runCases(2, 8, &.{
        .{ .prefill = false, .seqs = &.{ .{ 0, 9 }, .{ 1, 17 }, .{ 1, 1 }, .{ 0, 0 } }, .padding = 1 },
    }, .{ .batch = 3, .hkv = 1, .g = 2, .d = 32 });
}

test "tcl paged attention masks inactive decode rows across all heads" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    try runCases(32, 64, &.{.{
        .prefill = false,
        .seqs = &.{ .{ 1, 511 }, .{ 1, 504 }, .{ 1, 497 }, .{ 1, 490 }, .{ 0, 483 }, .{ 1, 476 }, .{ 1, 469 }, .{ 1, 462 } },
        .padding = 1,
    }}, .{ .batch = 8 });
}

test "tcl paged attention invalid pages cannot wrap into slot zero" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const cases = [_]TestCase{
        .{ .prefill = false, .seqs = &@as([4][2]i32, @splat(.{ 1, 17 })), .mask_all = true, .padding_page = 1 << 28 },
        .{ .prefill = false, .seqs = &@as([4][2]i32, @splat(.{ 1, 17 })), .mask_all = true, .padding_page = std.math.maxInt(i32) },
    };
    try runCases(2, 8, &cases, .{ .hkv = 1, .g = 1, .d = 32 });
}

test "tcl paged attention emits decode and prefill kernels with and without sinks" {
    inline for (.{ false, true }) |decode| {
        inline for (.{ false, true }) |use_sink| {
            const cfg: Cfg = .{ .tokens = if (decode) 2 else 8, .sequences = 2, .b = if (decode) 2 else 3, .i = if (decode) 1 else prefill_block, .a = 32, .chunk = 16, .hkv = 2, .g = 4, .d = 32, .pages = 8, .pg = 16, .scale = 0.125, .decode = decode, .use_sink = use_sink };
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
            .{ .prefill = true, .seqs = &.{ .{ 37, 200 }, .{ 1, 77 }, .{ 20, 20 }, .{ 0, 0 } } },
            .{ .prefill = false, .seqs = &.{ .{ 1, 250 }, .{ 1, 3 }, .{ 1, 64 }, .{ 1, 129 } } },
        } else [_]TestCase{
            .{ .prefill = true, .seqs = &.{ .{ 37, 4500 }, .{ 1, 77 }, .{ 70, 70 }, .{ 0, 0 } } },
            .{ .prefill = false, .seqs = &.{ .{ 1, 5000 }, .{ 1, 3 }, .{ 1, 4097 }, .{ 1, 129 } } },
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
        .{ .prefill = true, .seqs = &mixed, .padding = 5, .opts = .{ .sliding_window = 17, .scale = 0.2, .sink = &sinks } },
        .{ .prefill = true, .seqs = &mixed, .padding = 5, .opts = .{ .is_causal = false, .sliding_window = 17, .scale = 0.2, .sink = &sinks } },
        .{ .prefill = true, .seqs = &mixed, .padding = 5, .opts = .{ .is_causal = false, .sliding_window = 0 } },
        .{ .prefill = true, .seqs = &mixed, .padding = 5, .opts = .{ .sliding_window = 1, .sink = &disabled } },
        .{ .prefill = false, .seqs = &.{ .{ 1, 251 }, .{ 0, 73 }, .{ 1, 1 }, .{ 0, 17 } }, .padding = 2, .opts = .{ .sink = &sinks } },
        .{ .prefill = false, .seqs = &@as([4][2]i32, @splat(.{ 0, 0 })), .padding = 4 },
        .{ .prefill = false, .seqs = &@as([4][2]i32, @splat(.{ 1, 17 })), .mask_all = true },
        .{ .prefill = false, .seqs = &@as([4][2]i32, @splat(.{ 1, 17 })), .mask_all = true, .opts = .{ .sink = &sinks } },
        .{ .prefill = false, .seqs = &.{ .{ 1, 251 }, .{ 0, 73 }, .{ 1, 20 }, .{ 0, 17 } }, .padding = 2, .opts = .{ .is_causal = false, .sliding_window = 17, .scale = 0.2, .sink = &sinks } },
        .{ .prefill = false, .seqs = &.{ .{ 1, 251 }, .{ 0, 73 }, .{ 1, 20 }, .{ 0, 17 } }, .padding = 2, .opts = .{ .sliding_window = 1, .sink = &disabled } },
    };
    try runCases(16, 32, &cases, .{});
}

test "tcl paged attention chunk boundaries on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const cases = [_]TestCase{
        .{ .prefill = true, .seqs = &.{ .{ 37, 4097 }, .{ 0, 0 }, .{ 1, 4096 }, .{ 2, 4095 } }, .padding = 7, .opts = .{ .is_causal = false, .sliding_window = 33, .scale = 0.2, .sink = &.{ 0, 10 } } },
        .{ .prefill = false, .seqs = &.{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 1, 4097 }, .{ 0, 0 } }, .padding = 1, .opts = .{ .sliding_window = 17, .sink = &.{ -std.math.inf(f32), 0 } } },
        .{ .prefill = false, .seqs = &@as([4][2]i32, @splat(.{ 0, 0 })), .padding = 4 },
        .{ .prefill = false, .seqs = &@as([4][2]i32, @splat(.{ 1, 4097 })), .mask_all = true, .opts = .{ .sink = &.{ -std.math.inf(f32), 0 } } },
    };
    try runCases(257, 514, &cases, .{ .hkv = 1, .g = 2, .d = 32 });
    try runCases(256, 512, &.{.{ .prefill = false, .seqs = &.{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 0, 0 }, .{ 0, 0 } }, .padding = 2 }}, .{ .hkv = 1, .g = 1, .d = 32 });
    try runCases(86, 172, &.{.{ .prefill = false, .seqs = &.{ .{ 1, 4079 }, .{ 1, 4080 }, .{ 1, 4081 }, .{ 0, 0 } }, .padding = 1, .opts = .{ .sliding_window = 49 } }}, .{ .hkv = 1, .g = 1, .d = 32, .page = 48 });
}

test "tcl paged attention preserves constant values across chunks" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const cases = [_]TestCase{
        .{ .prefill = false, .seqs = &.{ .{ 1, 64 }, .{ 1, 65 }, .{ 1, 66 }, .{ 0, 0 } }, .padding = 1 },
        .{ .prefill = false, .seqs = &.{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 1, 4097 }, .{ 0, 0 } }, .padding = 1 },
        .{ .prefill = false, .seqs = &.{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 1, 4097 }, .{ 0, 0 } }, .padding = 1, .opts = .{ .sliding_window = 1, .sink = &.{0} } },
        .{ .prefill = true, .seqs = &.{ .{ 37, 4097 }, .{ 0, 0 }, .{ 1, 77 }, .{ 2, 2 } }, .padding = 3 },
    };
    try runCases(257, 514, &cases, .{ .hkv = 1, .g = 1, .d = 32, .uniform = true });
}

test "tcl paged attention chunked decode on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const cases = [_]TestCase{
        .{ .prefill = false, .seqs = &.{ .{ 1, 64 }, .{ 1, 65 }, .{ 1, 66 }, .{ 0, 0 } }, .padding = 1 },
        .{ .prefill = false, .seqs = &.{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 1, 4097 }, .{ 0, 0 } }, .padding = 1 },
    };
    try runCases(257, 514, &cases, .{ .hkv = 1, .g = 1, .d = 32 });
}

test "tcl paged attention uniform probabilities on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    try runCases(257, 514, &.{.{ .prefill = false, .seqs = &.{ .{ 1, 64 }, .{ 1, 65 }, .{ 1, 66 }, .{ 0, 0 } }, .padding = 1 }}, .{ .hkv = 1, .g = 1, .d = 32, .uniform_scores = true });
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
