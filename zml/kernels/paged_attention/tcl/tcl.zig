//! Furiosa RNGD paged attention in TCL, using the common/flash attention
//! algorithms from Furiosa's SDK (`furiosa-kernels 2026.4.0` reference).
//! This implementation retains FP32 normalization/accumulation and BF16
//! contractions. The exact common SDK operation materializes scores in BF16;
//! it fails the required unit-scale tolerance and is not a selectable backend.
//!
//! Queries are cut in blocks of rows of one sequence (one token per block for
//! decode). A block table that fits one chunk runs Furiosa's one-shot
//! attention: gather the keys by token slot, contract, mask, one softmax,
//! contract. Wider tables run Furiosa's flash attention: a device loop over
//! chunks (at most `prefill_chunk_keys` keys for prefill, `chunk_keys` for
//! decode) up to the longest sequence, gathering whole pages and accumulating
//! the softmax statistics in f32.
//!
//! Decode over BF16 caches instead runs fusible kernels in the layer, with
//! the token metadata every layer shares computed once per forward pass: one
//! straight-line kernel over the whole pages each token sees, or, for one
//! token without sink or window past `straight_decode_keys` keys, a
//! straight-line head over its first keys and a device loop over the chunks
//! holding its other keys, so that the work follows the sequence's length
//! instead of the block table's width. `chunked_decode_tokens` tokens past
//! `straight_batch_decode_keys` keys, and windows past `window_flash_chunks`
//! chunks, run flash attention for the same reason; a window's loop starts at
//! the chunk holding its first key.
const std = @import("std");

const stdx = @import("stdx");
const builder = @import("dsl/tcl/builder");

const zml = @import("../../../zml.zig");
const Tensor = zml.Tensor;
const tcl = zml.kernel.tcl;
const paged_attention = @import("../paged_attention.zig");
const triton = @import("../triton/triton.zig");

const chunk_keys = 4096;
const flash_chunk_pairs = 512 * 2048;
/// Keys a prefill chunk holds at most: its loop runs only the chunks under
/// the longest sequence, so finer chunks skip more of the table.
const prefill_chunk_keys = 2048;
const prefill_block = 32;

/// Query blocks a prefill step needs: each sequence's queries start a block,
/// so its last block may be partial. Regular (power of two) capacities keep
/// TCC's layout search short; 12 blocks of 32 rows panic its allocator. Extra
/// blocks keep zero valid lengths and invalid query positions.
fn prefillBlocks(tokens: i64, sequences: i64, block: i64) i64 {
    const blocks = @divTrunc(tokens + sequences * (block - 1), block);
    return @intCast(std.math.ceilPowerOfTwo(u64, @intCast(blocks)) catch unreachable);
}

/// Rows of a prefill query block: the larger one of the fewest rows in all.
/// Many sequences fill short blocks better: 16 sequences of a 256-token step
/// take 32 blocks of 16 rows instead of 32 of 32, prefilling Llama 3.1 8B at
/// 2406 instead of 1963 tokens/s; 4 sequences keep 32 rows (2785 vs 1946).
fn prefillBlock(tokens: i64, sequences: i64) i64 {
    const short = prefill_block / 2;
    return if (short * prefillBlocks(tokens, sequences, short) < prefill_block * prefillBlocks(tokens, sequences, prefill_block)) short else prefill_block;
}

/// How a call attends: a straight-line head over each token's first `head`
/// keys, a device loop over chunks of `chunk` keys past the head (none when
/// zero), and tokens in groups of `group`. Every route is a plan: straight
/// (the whole table as head, no loop), hybrid (head and loop, one token) and
/// flash (no head, all tokens or query blocks at once).
pub const Plan = struct {
    head: i64,
    chunk: i64,
    group: i64,
};

/// Forces every call's plan, for calibration.
pub var calibration_plan: ?Plan = null;

pub const paged = struct {
    pub const Options = triton.paged.Options;
    pub const Parameters = triton.paged.Parameters;

    /// `q` is `{.b = tokens, .hkv, .hg, .hd}`, the caches
    /// `{.page, .k_chunk, .hkv, .hd}`. Sinks are one logit per query head.
    /// Unwritten cache entries are excluded before the contractions; callers
    /// need only populate the tokens covered by their sequence lengths.
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
        stdx.debug.assert(compiler.platform.target == .furiosa, "TCL attention requires Furiosa chips", .{});
        const attributes = compiler.platform.devices[0].pjrt_device.attributes(compiler.platform.pjrt_api) catch
            std.debug.panic("TCL attention could not verify the Furiosa device's PE range", .{});
        defer attributes.deinit();
        const first_pe = attributes.get("pe_start");
        const last_pe = attributes.get("pe_end");
        stdx.debug.assert(first_pe != null and first_pe.? == .int64 and first_pe.?.int64 == 0 and
            last_pe != null and last_pe.? == .int64 and last_pe.?.int64 == 7, "TCL attention requires a complete 8-PE RNGD chip", .{});
        // Over several chips, each attends with its own heads.
        if (compiler.meshes[0].numDevices() > 1) return zml.ops.manualComputation(Shard.body, .{ .parameters = parameters, .q = q, .k_cache = k_cache, .v_cache = v_cache, .opts = opts }, q.shape());
        return attend(parameters, q, k_cache, v_cache, opts);
    }

    const Shard = struct {
        parameters: Parameters,
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        opts: paged_attention.AttentionOptions,

        fn body(self: Shard, _: zml.Shape) Tensor {
            return attend(self.parameters, self.q, self.k_cache, self.v_cache, self.opts);
        }
    };

    fn attend(parameters: Parameters, q: Tensor, k_cache: Tensor, v_cache: Tensor, opts: paged_attention.AttentionOptions) Tensor {
        if (q.dim(.b) == 0) return Tensor.zeroes(q.shape());
        const page = k_cache.dim(.k_chunk);
        // TCC 2026.5 has no DPE tactic for 32-key attention. Pad to
        // at least 64 keys; the extra pages remain invalid and contribute zero.
        const keys = @max(parameters.block_table.dim(.p), std.math.divCeil(i64, 64, page) catch unreachable) * page;
        // Chunks hold whole pages and a multiple of 64 keys: TCC 2026.5 fails
        // to allocate a chunk of 4080 keys (85 pages of 48).
        const key_alignment = @divExact(page, @as(i64, @intCast(std.math.gcd(@as(u64, @intCast(page)), 64)))) * 64;
        const alignment = if (key_alignment <= chunk_keys) key_alignment else page;
        const decode = !parameters.options_.is_prefill and q.dim(.b) == parameters.seq_lens.dim(.b);
        const block = if (decode) 1 else prefillBlock(q.dim(.b), parameters.seq_lens.dim(.b));
        const block_count = if (decode) q.dim(.b) else prefillBlocks(q.dim(.b), parameters.seq_lens.dim(.b), block);
        const rows = block_count * block;
        // Bound a flash chunk's query-key pairs: TCC 2026.5 takes 80 s on 512
        // query rows by 2048 keys and 430 s on 4096 keys for the same speed,
        // and does not finish 2048 rows by 4096 keys in 50 minutes.
        const chunk_limit = @max(1, @divTrunc(@min(@as(i64, if (decode) chunk_keys else prefill_chunk_keys), @divTrunc(flash_chunk_pairs, rows)), alignment)) * alignment;
        const chunk = @min(keys, chunk_limit);
        var cfg: Cfg = .{
            .tokens = q.dim(.b),
            .sequences = parameters.seq_lens.dim(.b),
            .b = block_count,
            .i = block,
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
        const plain_decode = decode and !cfg.use_sink and cfg.window == 0;
        // Converting the cache inside the unit stops TCC 2026.5's sparse
        // propagation ("Not implemented: Transmute"): straight decode reads
        // BF16 caches. TCC also fails on the metadata of several tokens over
        // a page count that is not a power of two ("Fail to cover valid
        // range" for 16 tokens over 80 to 160 pages but not 128), so their
        // straight decode pads the table.
        const straight = decode and k_cache.dtype() == .bf16 and v_cache.dtype() == .bf16;
        const straight_keys = if (straight and cfg.tokens > 1) @as(i64, @intCast(std.math.ceilPowerOfTwo(u64, @intCast(@divExact(keys, page))) catch unreachable)) * page else keys;
        const plan = calibration_plan orelse defaultPlan(cfg, straight_keys, parameters.block_table.dim(.p), plain_decode and straight, straight, chunk);
        const sink = if (opts.sink) |s| s.convert(.f32).reshape(.{ .hkv = cfg.hkv, .hg = cfg.g }) else Tensor.zeroes(.init(.{ .hkv = cfg.hkv, .hg = cfg.g }, .f32));
        if (plan.head > 0 and plan.chunk > 0) {
            stdx.debug.assert(plain_decode and cfg.tokens == 1, "TCL hybrid attention takes one plain decode token", .{});
            return hybridDecodeAttention(parameters, q, k_cache, v_cache, cfg.scale, cfg.is_causal, plan).convert(q.dtype());
        }
        if (plan.chunk == 0 and straight) {
            return decodeAttention(parameters, q, k_cache, v_cache, sink, .{
                .t = cfg.tokens,
                .a = straight_keys,
                .group = plan.group,
                .hkv = cfg.hkv,
                .g = cfg.g,
                .d = cfg.d,
                .pages = cfg.pages,
                .pg = page,
                .scale = cfg.scale,
                .is_causal = cfg.is_causal,
                .window = cfg.window,
                .use_sink = cfg.use_sink,
            }).convert(q.dtype());
        }
        // Without a loop, one chunk holds the table.
        cfg.chunk = if (plan.chunk > 0) plan.chunk else keys;
        cfg.a = (std.math.divCeil(i64, keys, cfg.chunk) catch unreachable) * cfg.chunk;
        const table = parameters.block_table.pad(-1, .{ .p = Tensor.Pad{ .high = @divExact(cfg.a, page) - parameters.block_table.dim(.p) } });
        const inputs: OneShot.Inputs = .{
            .q = q.convert(.bf16),
            .pages = table,
            .seq_lens = parameters.seq_lens,
            .query_offsets = parameters.query_start_len,
            .k_cache = k_cache.convert(.bf16),
            .v_cache = v_cache.convert(.bf16),
            .sink = sink,
        };
        const outputs: OneShot.Outputs = .{ .out = q.shape().withDtype(.bf16) };
        const out = if (cfg.a == cfg.chunk)
            OneShot.call(inputs, outputs, .{ .cfg = cfg }).out
        else
            Flash.call(inputs, outputs, .{ .cfg = cfg }).out;
        return out.convert(q.dtype());
    }
};

/// Today's routes: hybrid for one plain token past `straight_decode_keys`,
/// flash over a window's chunks past `window_flash_chunks` of them, straight
/// decode up to `decode_keys` (flash past `straight_batch_decode_keys` from
/// `chunked_decode_tokens`), else flash, in one chunk when it holds the table.
fn defaultPlan(cfg: Cfg, keys: i64, columns: i64, plain_decode: bool, straight: bool, chunk: i64) Plan {
    // The hybrid kernel needs tail pages past its head's. TCC 2026.5 does not
    // compile its tail for 2 or 4 KV heads wider than 128 (Qwen3.5's 4 of 256
    // ran out of 25 minutes), which run flash instead; 1, 4 of 128 and 8 of up
    // to 512 compile.
    if (plain_decode and cfg.tokens == 1 and columns > std.math.divCeil(i64, straight_decode_keys, cfg.pg) catch unreachable) {
        if (cfg.d <= 128 or cfg.hkv == 1 or cfg.hkv >= 8) return .{ .head = head_keys, .chunk = tail_chunk_keys, .group = 1 };
        return .{ .head = 0, .chunk = chunk, .group = cfg.b * cfg.i };
    }
    // Flash bounds a window's loop with power-of-two chunks.
    if (straight and cfg.window > 0) {
        const window_chunk = @max(window_chunk_keys, @as(i64, @intCast(std.math.ceilPowerOfTwo(u64, @intCast(cfg.window)) catch unreachable)));
        const chunks: i64 = if (cfg.tokens < chunked_decode_tokens) window_flash_chunks else batch_window_flash_chunks;
        if (window_chunk <= chunk and keys >= chunks * window_chunk)
            return .{ .head = 0, .chunk = window_chunk, .group = cfg.tokens };
    }
    if (straight and keys <= decode_keys and (cfg.tokens < chunked_decode_tokens or keys <= straight_batch_decode_keys))
        return .{ .head = keys, .chunk = 0, .group = @divExact(cfg.tokens, decodeGroups(cfg.tokens, keys)) };
    if (chunk == keys) return .{ .head = keys, .chunk = 0, .group = cfg.tokens };
    return .{ .head = 0, .chunk = chunk, .group = cfg.b * cfg.i };
}

/// Decode over a window from `window_flash_chunks` chunks of at least
/// `window_chunk_keys` keys, `batch_window_flash_chunks` from
/// `chunked_decode_tokens` tokens, runs flash attention, whose loop runs only
/// the window's chunks. Gemma 4 12B sliding layer (window 1024), straight vs
/// flash in 1024-key chunks, per layer at the shortest, a quarter and all of
/// the table: one token over 8K keys 449, 459, 499 vs 477, 476, 500 us; over
/// 16K 446, 469, 548 vs 477, 478, 502; over 32K 469, 519, 666 vs 480, 480,
/// 507. 16 tokens over 4K keys 584, 657, 927 vs 655, 723, 848, compiled in
/// 366 vs 55 s.
const window_chunk_keys = 1024;
const window_flash_chunks = 16;
const batch_window_flash_chunks = 4;

/// Widest block table the straight-line decode kernel takes for one token.
/// Its work follows the table's width, while the hybrid kernel's loop costs
/// about 15 us per layer whether it runs or not, and about 50 us more per
/// chunk it runs. Llama 3.1 8B decode layer on RNGD over a 32K table, straight
/// vs hybrid: 436 vs 437 us at 300 keys, 482 vs 554 at 16K, 536 vs 651 at 32K.
/// Several tokens do not run it: over a token axis TCC cannot propagate the
/// bounds of gathers outside a loop, and without them a 4-token layer took 11
/// minutes and 75 GB to compile.
const straight_decode_keys = 32768;

/// Keys a straight-line decode kernel attends to in one pass: TCC 2026.5 fails
/// on the decode metadata for 131072 keys in pages of 16.
const decode_keys = 65536;

/// Tokens from which decode over more than `straight_batch_decode_keys` keys
/// runs flash attention, whose loop computes only the chunks under the longest
/// sequence. Llama 3.1 8B layer, 16 tokens, straight vs flash per layer at the
/// shortest, a quarter and all of the table: over 4K keys 497, 522, 706 vs
/// 728, 732, 793 us; over 16K 711, 940, 1510 vs 657, 778, 1620, compiled in
/// 246 vs 48 s; over 32K 932, 1244, 2575 vs 656, 1062, 2745.
const chunked_decode_tokens = 16;
const straight_batch_decode_keys = 8192;

/// Decode contracts the gathered cache as is: keys past the valid ones hold
/// anything, NaN included, but the scores' select drops them and the tensor
/// units multiply their zero weights to zero, which a cache filled with NaN
/// and infinities confirms on RNGD. Masking them would cost a vector pass over
/// every key. TCC 2026.5 panics ("slice out of bounds") populating the score
/// contraction of more tokens times keys than this.
const unmasked_decode_keys = 1 << 17;

/// Keys up to which one token's decode masks the gathered cache anyway: 2 us
/// per layer faster over 4K keys. Several tokens never mask: unmasked, 4
/// tokens over 4K keys decode at 241 tok/s instead of 224, 16 at 861 instead
/// of 705.
const masked_decode_keys = 4096;

const DecodeCfg = struct {
    t: i64,
    a: i64,
    /// Tokens attended together.
    group: i64,
    hkv: i64,
    g: i64,
    d: i64,
    pages: i64,
    pg: i64,
    scale: f32,
    is_causal: bool,
    window: i64,
    use_sink: bool,
};

/// Decode attention in Furiosa's form: the kernel gathers whole pages, then
/// runs one contraction, a masked softmax and a second contraction. It reads
/// past each sequence's end, selecting zero for those unused cache entries.
/// The token metadata depends only on the attention parameters, which every
/// layer shares: one kernel computes it once per forward pass.
fn decodeAttention(parameters: paged.Parameters, q: Tensor, k_cache: Tensor, v_cache: Tensor, sink: Tensor, cfg: DecodeCfg) Tensor {
    const columns = @divExact(cfg.a, cfg.pg);
    const table = parameters.block_table.pad(0, .{ .p = Tensor.Pad{ .high = columns - parameters.block_table.dim(.p) } });
    const meta = DecodeMeta.call(.{ .seq_lens = parameters.seq_lens, .query_offsets = parameters.query_start_len, .table = table }, .{
        .pages = zml.Shape.init(.{ .t = cfg.t, .p = columns }, .i32),
        .bounds = zml.Shape.init(.{ .t = cfg.t }, .i32),
        .mask = zml.Shape.init(.{ .t = cfg.t, .a = cfg.a }, .i32),
        .keyed = zml.Shape.init(.{ .t = cfg.t }, .f32),
    }, .{ .cfg = .{ .t = cfg.t, .s = parameters.seq_lens.dim(.b), .ap = columns, .pg = cfg.pg, .pages = cfg.pages, .is_causal = cfg.is_causal, .window = cfg.window } });
    return Decode.call(.{
        .q = q.convert(.bf16),
        .pages = meta.pages,
        .bounds = meta.bounds,
        .mask = meta.mask,
        .keyed = meta.keyed,
        .k_cache = k_cache.convert(.bf16),
        .v_cache = v_cache.convert(.bf16),
        .sink = sink,
    }, .{ .out = q.shape().withDtype(.bf16) }, .{ .cfg = cfg }).out;
}

const MetaCfg = struct { t: i64, s: i64, ap: i64, pg: i64, pages: i64, is_causal: bool, window: i64 };

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
            var mask = try op.commit(op.fetch(position, .{}).muli(-1).addi(op.fetch(limit, .{})).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(try b.reshape(valid, &.{ T, A }, .{}), .{})), .{});
            if (cfg.window > 0) {
                // A window drops the keys more than its size before the limit.
                op = b.tensorOperation(.{ .tactic = .elementwise });
                mask = try op.commit(op.fetch(position, .{}).subi(op.fetch(limit, .{})).addi(cfg.window + 1).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(mask, .{})), .{});
            }
            b.ret(&.{ read, bounds, mask, keyed });
        }
    }.run,
});

/// `decodeTokens` over every token, in groups when they are many and the table
/// wide. Fusible: it joins the layer around it, one launch per layer.
const Decode = tcl.Kernel(DecodeCfg, .{
    .name = "paged_decode_attention",
    .inputs = &.{ "q", "pages", "bounds", "mask", "keyed", "k_cache", "v_cache", "sink" },
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
                .sink = .{ .dtype = .f32, .axes = &.{ Hkv, G } },
            });
            const sink = if (cfg.use_sink) t.sink else null;
            const groups = @divExact(cfg.t, cfg.group);
            if (groups == 1) return b.ret(&.{try decodeTokens(b, cfg.scale, t.q, t.pages, t.bounds, t.mask, t.keyed, t.k_cache, t.v_cache, sink)});
            // Several tokens over a wide table: a static loop over groups of
            // them, which costs nothing at runtime.
            const NG = b.axis("NG", groups);
            const TG = b.axis("TG", @divExact(cfg.t, groups));
            var loop = b.openFor(groups, .{try b.scratchpad(.bf16, &.{ NG, TG, Hkv, G, D })});
            {
                const q = try b.indexRead(try b.reshape(t.q, &.{ NG, TG, Hkv, G, D }, .{}), loop.iv);
                const pages = try b.indexRead(try b.reshape(t.pages, &.{ NG, TG, AP }, .{}), loop.iv);
                const bounds = try b.indexRead(try b.reshape(t.bounds, &.{ NG, TG }, .{}), loop.iv);
                const mask = try b.indexRead(try b.reshape(t.mask, &.{ NG, TG, A }, .{}), loop.iv);
                const keyed = try b.indexRead(try b.reshape(t.keyed, &.{ NG, TG }, .{}), loop.iv);
                const out = try decodeTokens(b, cfg.scale, q, pages, bounds, mask, keyed, t.k_cache, t.v_cache, sink);
                try loop.yield(.{try b.indexWrite(loop.carried[0], loop.iv, out)});
            }
            b.ret(&.{try b.reshape(loop.results[0], &.{ T, Hkv, G, D }, .{})});
        }
    }.run,
});

/// Tokens one unmasked decode group contracts at most: in llmd Llama 8B, 16
/// tokens in one group decode at 122 tok/s over 8K keys, 4 groups of 4 at 515
/// over 32K.
const unmasked_group_tokens = 4;

/// Groups of tokens a decode step attends in turn: their tokens times keys
/// within `unmasked_decode_keys`, and at most `unmasked_group_tokens` over an
/// unmasked cache.
fn decodeGroups(tokens: i64, keys: i64) i64 {
    var groups: i64 = 1;
    while ((@divExact(tokens, groups) > unmasked_group_tokens or @divExact(tokens, groups) * keys > unmasked_decode_keys) and groups < tokens) {
        groups += 1;
        while (@rem(tokens, groups) != 0) groups += 1;
    }
    return groups;
}

/// Decode attention of `q`'s tokens over the pages each one sees: gathers
/// their keys and values up to their bound, contracts with the queries, masks
/// the scores of keys past the sequence, normalizes in one softmax times
/// `keyed` (zero for tokens without keys, whose softmax is uniform) and
/// contracts with the values, gathered as is (see `unmasked_decode_keys`)
/// unless one token sees at most `masked_decode_keys`. A sink adds one zero
/// value to every head's softmax.
fn decodeTokens(b: *tcl.Builder, scale: f32, q: builder.Tensor, pages: builder.Tensor, bounds: builder.Tensor, mask: builder.Tensor, keyed: builder.Tensor, k_cache: builder.Tensor, v_cache: builder.Tensor, sink: ?builder.Tensor) tcl.FinishError!builder.Tensor {
    const T = q.axes[0];
    const Hkv = q.axes[1];
    const G = q.axes[2];
    const D = q.axes[3];
    const AP = pages.axes[1];
    const P = k_cache.axes[0];
    const PG = k_cache.axes[1];
    const keys: []const tcl.Axis = &.{ AP, PG };
    const key_mask = try b.reshape(mask, &.{ T, AP, PG }, .{});
    var k = try b.gather(k_cache, pages, P, &.{ T, AP, PG, Hkv, D }, .{ .valid_length = bounds });
    var v = try b.gather(v_cache, pages, P, &.{ T, AP, PG, Hkv, D }, .{ .valid_length = bounds });
    if (T.size == 1 and AP.size * PG.size <= masked_decode_keys) {
        k = try maskCacheValues(b, k, key_mask);
        v = try maskCacheValues(b, v, key_mask);
    }
    var op = b.tensorOperation(.{});
    const scores = try op.commit(op.contract(q, k, &.{ Hkv, G, T, AP, PG }).mulf(scale), .{});
    op = b.tensorOperation(.{});
    const s = op.fetch(scores, .{});
    const masked = try op.commit(op.where(op.fetch(key_mask, .{}), .ne, 0, s, score_min), .{});
    op = b.tensorOperation(.{});
    const p = if (sink) |logits| blk: {
        const maximum = try op.commit(op.fetch(masked, .{}).reduce(keys, .maxf).maxf(op.fetch(logits, .{ .broadcast_to = &.{ Hkv, G, T } })), .{});
        op = b.tensorOperation(.{});
        const e = try op.commit(op.fetch(masked, .{}).subf(op.fetch(maximum, .{})).exp(), .{});
        op = b.tensorOperation(.{});
        const sink_exp = try op.commit(op.fetch(maximum, .{}).mulf(-1).addf(op.fetch(logits, .{})).exp(), .{});
        op = b.tensorOperation(.{});
        const denominator = try op.commit(op.fetch(e, .{}).reduce(keys, .addf).addf(op.fetch(sink_exp, .{})), .{});
        op = b.tensorOperation(.{});
        break :blk try op.commit(op.fetch(e, .{}).divf(op.fetch(denominator, .{})).mulf(op.fetch(keyed, .{})), .{ .dtype = .bf16 });
    } else blk: {
        const x = op.fetch(masked, .{});
        const e = x.subf(x.reduce(keys, .maxf)).exp();
        break :blk try op.commit(e.divf(e.reduce(keys, .addf)).mulf(op.fetch(keyed, .{})), .{ .dtype = .bf16 });
    };
    op = b.tensorOperation(.{});
    return op.commit(op.contract(v, p, &.{ T, Hkv, G, D }), .{ .dtype = .bf16 });
}

/// The hybrid kernel runs a token's first `head_keys` keys straight line and
/// the rest in a loop over `tail_chunk_keys`-key chunks, so that its work
/// follows each step's longest sequence instead of the block table's width.
const head_keys = 4096;
const tail_chunk_keys = 8192;

const HybridCfg = struct {
    t: i64,
    s: i64,
    /// Head pages, tail chunks and pages per tail chunk.
    hp: i64,
    nc: i64,
    cp: i64,
    pg: i64,
    hkv: i64,
    g: i64,
    d: i64,
    pages: i64,
    scale: f32,
    is_causal: bool,
};

/// Decode attention whose work follows the longest sequence: a straight-line
/// head over every token's first keys, which the layer's schedule overlaps
/// like any straight-line kernel, and a device loop over the chunks holding
/// keys past it. Short contexts, which skip the loop, still pay for it: the
/// work it needs ends the layer's first stretch (the query), and nothing past
/// it overlaps that stretch.
fn hybridDecodeAttention(parameters: paged.Parameters, q: Tensor, k_cache: Tensor, v_cache: Tensor, scale: f32, is_causal: bool, plan: Plan) Tensor {
    const pg = k_cache.dim(.k_chunk);
    const t = q.dim(.b);
    const s = parameters.seq_lens.dim(.b);
    const hp = std.math.divCeil(i64, plan.head, pg) catch unreachable;
    const cp = std.math.divCeil(i64, plan.chunk, pg) catch unreachable;
    const nc = std.math.divCeil(i64, parameters.block_table.dim(.p) - hp, cp) catch unreachable;
    const cfg: HybridCfg = .{ .t = t, .s = s, .hp = hp, .nc = nc, .cp = cp, .pg = pg, .hkv = q.dim(.hkv), .g = q.dim(.hg), .d = q.dim(.hd), .pages = k_cache.dim(.page), .scale = scale, .is_causal = is_causal };
    const table = parameters.block_table.pad(-1, .{ .p = Tensor.Pad{ .high = hp + nc * cp - parameters.block_table.dim(.p) } });
    const meta = HybridMeta.call(.{ .seq_lens = parameters.seq_lens, .query_offsets = parameters.query_start_len, .table = table }, .{
        .hpages = zml.Shape.init(.{ .t = t, .p = hp }, .i32),
        .hmask = zml.Shape.init(.{ .t = t, .a = hp * pg }, .i32),
        .hbounds = zml.Shape.init(.{ .t = t }, .i32),
        .tpages = zml.Shape.init(.{ .c = nc, .t = t, .p = cp }, .i32),
        .tmask = zml.Shape.init(.{ .c = nc, .t = t, .a = cp * pg }, .i32),
        .tbounds = zml.Shape.init(.{ .c = nc, .t = t }, .i32),
        .limits = zml.Shape.init(.{ .t = t }, .i32),
        .keyed = zml.Shape.init(.{ .t = t }, .f32),
    }, .{ .cfg = cfg });
    return HybridDecode.call(.{
        .q = q.convert(.bf16),
        .hpages = meta.hpages,
        .hmask = meta.hmask,
        .hbounds = meta.hbounds,
        .tpages = meta.tpages,
        .tmask = meta.tmask,
        .tbounds = meta.tbounds,
        .limits = meta.limits,
        .keyed = meta.keyed,
        .k_cache = k_cache.convert(.bf16),
        .v_cache = v_cache.convert(.bf16),
    }, .{ .out = q.shape().withDtype(.bf16) }, .{ .cfg = cfg }).out;
}

/// Per token, once per forward pass: the head's pages, key mask and valid
/// page count, the same per tail chunk (chunk-major, for the chunks holding
/// keys of the longest token only), the visible key count and whether the
/// token sees any key.
const HybridMeta = tcl.Kernel(HybridCfg, .{
    .name = "paged_hybrid_metadata",
    .inputs = &.{ "seq_lens", "query_offsets", "table" },
    .outputs = &.{ "hpages", "hmask", "hbounds", "tpages", "tmask", "tbounds", "limits", "keyed" },
    .run = struct {
        fn run(b: *tcl.Builder, cfg: HybridCfg) tcl.FinishError!void {
            const total = (cfg.hp + cfg.nc * cfg.cp) * cfg.pg;
            const T = b.axis("T", cfg.t);
            const S = b.axis("S", cfg.s);
            const O = b.axis("O", cfg.s + 1);
            const AP = b.axis("AP", cfg.hp + cfg.nc * cfg.cp);
            const HP = b.axis("HP", cfg.hp);
            const RP = b.axis("RP", cfg.nc * cfg.cp);
            const NC = b.axis("NC", cfg.nc);
            const CP = b.axis("CP", cfg.cp);
            const HA = b.axis("HA", cfg.hp * cfg.pg);
            const CK = b.axis("CK", cfg.cp * cfg.pg);
            const t = try b.declareArgs(.{
                .seq_lens = .{ .dtype = .i32, .axes = &.{S} },
                .query_offsets = .{ .dtype = .i32, .axes = &.{O} },
                .table = .{ .dtype = .i32, .axes = &.{ S, AP } },
            });
            const ends = try selectMetadata(b, t.query_offsets, try selector(b, try b.arange(cfg.s + 1, .i32, &.{S}, .{ .start = 1 }), O, cfg.s + 1), O);
            const tokens = try b.arange(cfg.t, .i32, &.{T}, .{});
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
            const limit = try op.commit(op.fetch(visible, .{}).binary(.maxi, 0).binary(.mini, total).muli(op.fetch(active, .{})), .{});

            // Tail chunks first: a runtime loop at the start of the unit costs
            // nothing. A 2-D selection per chunk; a 3-D one-hot product over
            // [chunks, tokens, sequences, pages] computes garbage.
            const rows1 = try b.reshape(try b.slice(t.table, AP, cfg.hp, &.{ S, RP }, .{}), &.{ S, NC, CP }, .{});
            op = b.tensorOperation(.{});
            const rows = try op.commit(op.fetch(rows1, .{}), .{ .axes = &.{ NC, S, CP } });
            // Trips never exceed nc: `limit` is at most the table's keys.
            const trips = try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(limit), cfg.cp * cfg.pg - 1 - cfg.hp * cfg.pg), cfg.cp * cfg.pg);
            var loop = b.openFor(trips, .{ try b.scratchpad(.i32, &.{ NC, T, CP }), try b.scratchpad(.i32, &.{ NC, T, CK }), try b.scratchpad(.i32, &.{ NC, T }), try b.full(0, .i32, &.{T}) });
            {
                const reads, const masks, const counts, const seen = loop.carried;
                const first = try b.symExpr(.add, try b.symExpr(.mul, loop.iv, cfg.cp * cfg.pg), cfg.hp * cfg.pg);
                const chunk = try hybridColumns(b, cfg, own, try b.indexRead(rows, loop.iv), limit, first, CP, CK);
                op = b.tensorOperation(.{});
                const any = try op.commit(op.fetch(seen, .{}).binary(.maxi, op.fetch(chunk.any, .{})), .{});
                try loop.yield(.{
                    try b.indexWrite(reads, loop.iv, chunk.pages),
                    try b.indexWrite(masks, loop.iv, chunk.mask),
                    try b.indexWrite(counts, loop.iv, chunk.bounds),
                    any,
                });
            }
            const head = try hybridColumns(b, cfg, own, try b.slice(t.table, AP, 0, &.{ S, HP }, .{}), limit, 0, HP, HA);
            // Like the straight kernel's: tokens whose visible pages are all
            // outside the cache see no key.
            op = b.tensorOperation(.{});
            const keyed = try op.commit(op.fetch(head.any, .{}).binary(.maxi, op.fetch(loop.results[3], .{})).toFloat(), .{});
            b.ret(&.{ head.pages, head.mask, head.bounds, loop.results[0], loop.results[1], loop.results[2], limit, keyed });
        }
    }.run,
});

/// The cache page of each column of `table` (sequence rows) for every token,
/// its key mask (below the token's limit, page in the cache), how many
/// columns hold keys the token sees and whether one of them is in the cache.
/// Columns start at key `first`. Sparse gathers never read the columns past a
/// token's count.
fn hybridColumns(b: *tcl.Builder, cfg: HybridCfg, own: builder.Tensor, table: builder.Tensor, limit: builder.Tensor, first: anytype, cols: tcl.Axis, keys: tcl.Axis) tcl.FinishError!struct { pages: builder.Tensor, mask: builder.Tensor, bounds: builder.Tensor, any: builder.Tensor } {
    const T = own.axes[0];
    const S = own.axes[1];
    const PG = b.axis("PG", cfg.pg);
    const pages = try validIndices(b, try selectMetadata(b, table, own, S), cfg.pages);
    var op = b.tensorOperation(.{});
    const firsts = try op.commit(op.fetch(try b.arange(cols.size * cfg.pg, .i32, &.{cols}, .{ .step = cfg.pg }), .{}).addi(if (@TypeOf(first) == builder.Tensor) op.fetch(first, .{}) else first), .{});
    op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
    const used = try op.commit(op.fetch(limit, .{}).subi(op.fetch(firsts, .{})).binary(.maxi, 0).binary(.mini, 1), .{});
    op = b.tensorOperation(.{});
    const bounds = try op.commit(op.fetch(used, .{}).reduce(&.{cols}, .addi), .{});
    op = b.tensorOperation(.{});
    const any = try op.commit(op.fetch(used, .{}).muli(op.fetch(pages.valid, .{})).reduce(&.{cols}, .maxi), .{});
    op = b.tensorOperation(.{});
    const key_ids = try op.commit(op.fetch(try b.arange(keys.size, .i32, &.{keys}, .{}), .{}).addi(if (@TypeOf(first) == builder.Tensor) op.fetch(first, .{}) else first), .{});
    const to: []const tcl.Axis = &.{ T, cols, PG };
    op = b.tensorOperation(.{});
    const position = try op.commit(op.fetch(try b.reshape(key_ids, &.{ cols, PG }, .{}), .{ .broadcast_to = to }), .{});
    op = b.tensorOperation(.{});
    const valid = try op.commit(op.fetch(pages.valid, .{ .broadcast_to = to }), .{});
    op = b.tensorOperation(.{ .tactic = .elementwise });
    const mask = try op.commit(op.fetch(position, .{}).muli(-1).addi(op.fetch(limit, .{})).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(valid, .{})), .{});
    return .{ .pages = pages.safe, .mask = try b.reshape(mask, &.{ T, keys }, .{}), .bounds = bounds, .any = any };
}

const SoftmaxState = struct { max: builder.Tensor, sum: builder.Tensor, out: builder.Tensor };

/// The tail loop first, then the head with the tail's state folded in.
/// Fusible: in its own unit the loop costs about 100 us more per layer.
const HybridDecode = tcl.Kernel(HybridCfg, .{
    .name = "paged_hybrid_decode_attention",
    .inputs = &.{ "q", "hpages", "hmask", "hbounds", "tpages", "tmask", "tbounds", "limits", "keyed", "k_cache", "v_cache" },
    .outputs = &.{"out"},
    .run = struct {
        fn run(b: *tcl.Builder, cfg: HybridCfg) tcl.FinishError!void {
            b.fuse();
            const a = hybridAxes(b, cfg);
            const t = try b.declareArgs(.{
                .q = .{ .dtype = .bf16, .axes = &.{ a.T, a.Hkv, a.G, a.D } },
                .hpages = .{ .dtype = .i32, .axes = &.{ a.T, a.HP } },
                .hmask = .{ .dtype = .i32, .axes = &.{ a.T, a.HA } },
                .hbounds = .{ .dtype = .i32, .axes = &.{a.T} },
                .tpages = .{ .dtype = .i32, .axes = &.{ a.NC, a.T, a.CP } },
                .tmask = .{ .dtype = .i32, .axes = &.{ a.NC, a.T, a.CK } },
                .tbounds = .{ .dtype = .i32, .axes = &.{ a.NC, a.T } },
                .limits = .{ .dtype = .i32, .axes = &.{a.T} },
                .keyed = .{ .dtype = .f32, .axes = &.{a.T} },
                .k_cache = .{ .dtype = .bf16, .axes = &.{ a.P, a.PG, a.Hkv, a.D } },
                .v_cache = .{ .dtype = .bf16, .axes = &.{ a.P, a.PG, a.Hkv, a.D } },
            });
            const tail = try hybridTail(b, cfg, a, t.q, t.tpages, t.tmask, t.tbounds, t.limits, t.k_cache, t.v_cache);
            b.ret(&.{try hybridHead(b, cfg, a, t.q, t.hpages, t.hmask, t.hbounds, t.keyed, t.k_cache, t.v_cache, tail)});
        }
    }.run,
});

const HybridAxes = struct { T: tcl.Axis, HP: tcl.Axis, HA: tcl.Axis, NC: tcl.Axis, CP: tcl.Axis, CK: tcl.Axis, Hkv: tcl.Axis, G: tcl.Axis, D: tcl.Axis, P: tcl.Axis, PG: tcl.Axis };

fn hybridAxes(b: *tcl.Builder, cfg: HybridCfg) HybridAxes {
    return .{
        .T = b.axis("T", cfg.t),
        .HP = b.axis("HP", cfg.hp),
        .HA = b.axis("HA", cfg.hp * cfg.pg),
        .NC = b.axis("NC", cfg.nc),
        .CP = b.axis("CP", cfg.cp),
        .CK = b.axis("CK", cfg.cp * cfg.pg),
        .Hkv = b.axis("Hkv", cfg.hkv),
        .G = b.axis("G", cfg.g),
        .D = b.axis("D", cfg.d),
        .P = b.axis("P", cfg.pages),
        .PG = b.axis("PG", cfg.pg),
    };
}

/// The softmax state of the keys past the head: a loop over the chunks
/// holding the longest token's keys, none for short contexts.
fn hybridTail(b: *tcl.Builder, cfg: HybridCfg, a: HybridAxes, q: builder.Tensor, tpages: builder.Tensor, tmask: builder.Tensor, tbounds: builder.Tensor, limits: builder.Tensor, k_cache: builder.Tensor, v_cache: builder.Tensor) tcl.FinishError!SoftmaxState {
    const stats: []const tcl.Axis = &.{ a.Hkv, a.G, a.T };
    const trips = try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(limits), cfg.cp * cfg.pg - 1 - cfg.hp * cfg.pg), cfg.cp * cfg.pg);
    var loop = b.openFor(trips, .{ try b.full(score_min, .f32, stats), try b.full(0, .f32, stats), try b.full(0, .f32, &.{ a.T, a.Hkv, a.G, a.D }) });
    {
        const max, const sum, const out = loop.carried;
        const mask = try b.reshape(try b.indexRead(tmask, loop.iv), &.{ a.T, a.CP, a.PG }, .{});
        const bounds = try b.indexRead(tbounds, loop.iv);
        const pages = try b.indexRead(tpages, loop.iv);
        const k = try b.gather(k_cache, pages, a.P, &.{ a.T, a.CP, a.PG, a.Hkv, a.D }, .{ .valid_length = bounds });
        const v = try b.gather(v_cache, pages, a.P, &.{ a.T, a.CP, a.PG, a.Hkv, a.D }, .{ .valid_length = bounds });
        const merged = try mergeSoftmax(b, .{ .max = max, .sum = sum, .out = out }, try chunkSoftmax(b, cfg, q, k, v, mask));
        try loop.yield(.{ merged.max, merged.sum, merged.out });
    }
    return .{ .max = loop.results[0], .sum = loop.results[1], .out = loop.results[2] };
}

/// The head's keys with the tail's state folded into their softmax: one
/// maximum over both, the tail's rescaled sum in the head's denominator, the
/// tail's output rescaled by the same factor.
fn hybridHead(b: *tcl.Builder, cfg: HybridCfg, a: HybridAxes, q: builder.Tensor, hpages: builder.Tensor, hmask: builder.Tensor, hbounds: builder.Tensor, keyed: builder.Tensor, k_cache: builder.Tensor, v_cache: builder.Tensor, tail: SoftmaxState) tcl.FinishError!builder.Tensor {
    const head_mask = try b.reshape(hmask, &.{ a.T, a.HP, a.PG }, .{});
    const head_k = try b.gather(k_cache, hpages, a.P, &.{ a.T, a.HP, a.PG, a.Hkv, a.D }, .{ .valid_length = hbounds });
    const head_v = try b.gather(v_cache, hpages, a.P, &.{ a.T, a.HP, a.PG, a.Hkv, a.D }, .{ .valid_length = hbounds });
    const keys: []const tcl.Axis = &.{ a.HP, a.PG };
    var op = b.tensorOperation(.{});
    const scores = try op.commit(op.contract(q, head_k, &.{ a.Hkv, a.G, a.T, a.HP, a.PG }).mulf(cfg.scale), .{});
    op = b.tensorOperation(.{});
    const sv = op.fetch(scores, .{});
    const masked = try op.commit(op.where(op.fetch(head_mask, .{}), .ne, 0, sv, score_min), .{});
    op = b.tensorOperation(.{});
    const max = try op.commit(op.fetch(masked, .{}).reduce(keys, .maxf).maxf(op.fetch(tail.max, .{})), .{});
    const alpha = try expDifference(b, tail.max, max);
    op = b.tensorOperation(.{});
    const carried = try op.commit(op.fetch(tail.sum, .{}).mulf(op.fetch(alpha, .{})), .{});
    op = b.tensorOperation(.{});
    const denominator = try op.commit(op.fetch(masked, .{}).subf(op.fetch(max, .{})).exp().reduce(keys, .addf).addf(op.fetch(carried, .{})), .{});
    op = b.tensorOperation(.{});
    // Tokens without keys weigh their garbage values zero, not uniformly.
    const p = try op.commit(op.fetch(masked, .{}).subf(op.fetch(max, .{})).exp().divf(op.fetch(denominator, .{})).mulf(op.fetch(keyed, .{})), .{ .dtype = .bf16 });
    op = b.tensorOperation(.{});
    const head = try op.commit(op.contract(head_v, p, &.{ a.T, a.Hkv, a.G, a.D }), .{});
    op = b.tensorOperation(.{});
    const factor = try op.commit(op.fetch(alpha, .{}).divf(op.fetch(denominator, .{})), .{});
    // Tokens without keys have no defined attention: `keyed` zeroes them.
    op = b.tensorOperation(.{});
    return op.commit(op.fetch(tail.out, .{}).mulf(op.fetch(factor, .{})).addf(op.fetch(head, .{})).mulf(op.fetch(keyed, .{})), .{ .dtype = .bf16 });
}

/// Softmax state of every token over one chunk of masked keys: max and sum of
/// the scores and the unnormalized output, all f32.
fn chunkSoftmax(b: *tcl.Builder, cfg: HybridCfg, q: builder.Tensor, k: builder.Tensor, v: builder.Tensor, mask: builder.Tensor) tcl.FinishError!SoftmaxState {
    const T = q.axes[0];
    const Hkv = q.axes[1];
    const G = q.axes[2];
    const D = q.axes[3];
    const keys = mask.axes[1..];
    var op = b.tensorOperation(.{});
    const scores = try op.commit(op.contract(q, k, &.{ Hkv, G, T, keys[0], keys[1] }).mulf(cfg.scale), .{});
    op = b.tensorOperation(.{});
    const sv = op.fetch(scores, .{});
    const masked = try op.commit(op.where(op.fetch(mask, .{}), .ne, 0, sv, score_min), .{});
    op = b.tensorOperation(.{});
    const max = try op.commit(op.fetch(masked, .{}).reduce(keys, .maxf), .{});
    const exps = try maskedExp(b, masked, max, mask);
    op = b.tensorOperation(.{});
    const sum = try op.commit(op.fetch(exps, .{}).reduce(keys, .addf), .{});
    op = b.tensorOperation(.{});
    const p = try op.commit(op.fetch(exps, .{}).mulf(1.0), .{ .dtype = .bf16 });
    op = b.tensorOperation(.{});
    const out = try op.commit(op.contract(v, p, &.{ T, Hkv, G, D }), .{});
    return .{ .max = max, .sum = sum, .out = out };
}

fn mergeSoftmax(b: *tcl.Builder, x: SoftmaxState, y: SoftmaxState) tcl.FinishError!SoftmaxState {
    var op = b.tensorOperation(.{ .tactic = .elementwise });
    const max = try op.commit(op.fetch(x.max, .{}).maxf(op.fetch(y.max, .{})), .{});
    const alpha = try expDifference(b, x.max, max);
    const beta = try expDifference(b, y.max, max);
    op = b.tensorOperation(.{});
    const xs = try op.commit(op.fetch(x.sum, .{}).mulf(op.fetch(alpha, .{})), .{});
    op = b.tensorOperation(.{});
    const sum = try op.commit(op.fetch(y.sum, .{}).mulf(op.fetch(beta, .{})).addf(op.fetch(xs, .{})), .{});
    op = b.tensorOperation(.{});
    const xo = try op.commit(op.fetch(x.out, .{}).mulf(op.fetch(alpha, .{})), .{});
    op = b.tensorOperation(.{});
    const out = try op.commit(op.fetch(y.out, .{}).mulf(op.fetch(beta, .{})).addf(op.fetch(xo, .{})), .{});
    return .{ .max = max, .sum = sum, .out = out };
}

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
/// removed in TCC 2026.5. Prefill searches a bounded number of tactic shapes,
/// with TCC's pruning: Llama 3.1 8B's flash kernel compiles in 20 to 50 s
/// instead of 2 to 15 minutes under the compute-bound hint, and runs a 16K
/// prompt in 3.5 s instead of 4.1. Narrow heads search fewer shapes: sixteen
/// took 11 minutes on one kv head of 32 dimensions, eight 40 s; eight fail to
/// schedule Llama's kernel for 4 sequences at 128K max context.
fn configure(b: *tcl.Builder, cfg: Cfg) void {
    const shapes: i64 = if (cfg.hkv * cfg.g * cfg.d >= 1024) 16 else 8;
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
        .scheduler_beam_search = true,
        .allow_reduce_by_ve_cluster_chip_reduce = false,
        .allow_reduce_by_ve_cluster_chip_reduce_base_population = false,
        .tactic_hint = .{ .DefaultWithMaxNumUniqueShapes = shapes },
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
    /// Page table rows: one per block for decode, one per sequence for
    /// prefill, whose blocks select their sequence's row with `pick`.
    pages: builder.Tensor,
    pick: ?builder.Tensor,
    qpos: builder.Tensor,
    kvlen: builder.Tensor,
    row: builder.Tensor,
    active: builder.Tensor,
    /// Per page row: the first query position and the keys.
    first: builder.Tensor,
    keys: builder.Tensor,

    /// The pages of each block. TCC 2026.5 has no tactic for selecting
    /// whole rows of 8192 pages: flash attention selects them per chunk.
    fn blockPages(self: Prepared, b: *tcl.Builder, a: Axes, pages: builder.Tensor) tcl.FinishError!builder.Tensor {
        const pick = self.pick orelse return pages;
        return selectMetadata(b, pages, pick, a.S);
    }
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
            .pick = null,
            .qpos = try b.reshape(qpos, &.{ a.B, a.I }, .{}),
            .kvlen = try b.reshape(lengths, &.{a.B}, .{}),
            .row = token_seq,
            .active = active,
            .first = try b.reshape(qpos, &.{a.B}, .{}),
            .keys = try b.reshape(lengths, &.{a.B}, .{}),
        };
    }

    const shift = std.math.log2_int(u64, @intCast(cfg.i));
    op = b.tensorOperation(.{});
    const blocks = try op.commit(op.fetch(qlen, .{}).addi(cfg.i - 1).binary(.shr_arith, shift), .{});
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
    const first_row = try op.commit(op.fetch(block_ids, .{}).subi(op.fetch(block_begin, .{})).muli(cfg.i), .{});
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

    const token_pick = try selector(b, token_seq, a.S, cfg.sequences);
    const token_begin = try selectMetadata(b, starts, token_pick, a.S);
    const token_block = try selectMetadata(b, block_start, token_pick, a.S);
    op = b.tensorOperation(.{});
    const row = try op.commit(op.fetch(token_block, .{}).muli(cfg.i).addi(op.fetch(tokens, .{})).subi(op.fetch(token_begin, .{})).binary(.maxi, 0).binary(.mini, cfg.b * cfg.i - 1), .{});
    op = b.tensorOperation(.{});
    const first = try op.commit(op.fetch(t.seq_lens, .{}).subi(op.fetch(qlen, .{})), .{});
    return .{ .q = q, .pages = t.pages, .pick = pick, .qpos = qpos, .kvlen = kvlen, .row = row, .active = active, .first = first, .keys = t.seq_lens };
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

/// Cache validity is per sequence, not per query: causal/window masking stays
/// on scores, without replicating K/V for every row of a prefill block.
fn cacheMask(b: *tcl.Builder, valid_pages: builder.Tensor, kvlen: builder.Tensor, position: builder.Tensor) tcl.FinishError!builder.Tensor {
    var op = b.tensorOperation(.{});
    const pos = try op.commit(op.fetch(position, .{ .broadcast_to = valid_pages.axes }), .{});
    op = b.tensorOperation(.{ .tactic = .elementwise });
    return op.commit(op.fetch(pos, .{}).muli(-1).addi(op.fetch(kvlen, .{})).binary(.maxi, 0).binary(.mini, 1).muli(op.fetch(valid_pages, .{})), .{});
}

/// A true select excludes NaN/Inf in unwritten slots; multiplying by zero
/// would still propagate them into the contraction. Materialize only BF16.
fn maskCacheValues(b: *tcl.Builder, values: builder.Tensor, valid: builder.Tensor) tcl.FinishError!builder.Tensor {
    const op = b.tensorOperation(.{ .tactic = .elementwise });
    const data = op.fetch(values, .{});
    return op.commit(op.where(op.fetch(valid, .{ .broadcast_to = values.axes }), .ne, 0, data, 0.0), .{ .dtype = .bf16 });
}

/// Build the sequence, query, page and window mask for the current keys.
fn attentionMask(b: *tcl.Builder, cfg: Cfg, qpos: builder.Tensor, kvlen: builder.Tensor, position: builder.Tensor, valid_keys: builder.Tensor) tcl.FinishError!builder.Tensor {
    const axes = &.{ qpos.axes[0], qpos.axes[1], position.axes[position.axes.len - 1] };
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
            const page_indices = try validIndices(b, try prepared.blockPages(b, a, prepared.pages), cfg.pages);
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
            const position = try b.arange(cfg.a, .i32, &.{A}, .{});
            const valid_cache = try cacheMask(b, valid_keys, prepared.kvlen, position);
            const k = try maskCacheValues(b, try b.gather(kc, slots, Nb, &.{ a.B, A, a.Hkv, a.D }, .{ .valid_length = valid_length }), valid_cache);
            const v = try maskCacheValues(b, try b.gather(vc, slots, Nb, &.{ a.B, A, a.Hkv, a.D }, .{ .valid_length = valid_length }), valid_cache);
            const mask = try attentionMask(b, cfg, prepared.qpos, prepared.kvlen, position, valid_keys);
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
            const rows = prepared.pages.axes[0];
            const by_row = try b.reshape(prepared.pages, &.{ rows, C, Cp }, .{});
            const page_op = b.tensorOperation(.{});
            const pages = try page_op.commit(page_op.fetch(by_row, .{}), .{ .axes = &.{ C, rows, Cp } });
            const stats = &.{ a.Hkv, a.G, a.B, a.I };
            const out_axes = &.{ a.B, a.I, a.Hkv, a.G, a.D };
            var op = b.tensorOperation(.{});
            const shift = std.math.log2_int(u64, @intCast(cfg.chunk));
            // With a window, each page row's loop starts at the chunk holding
            // its first query's first visible key: the loop runs the window's
            // chunks, not the sequence's. TCL divides by shifts, so other
            // chunks run every chunk and the mask applies the window.
            const window = if (cfg.window > 0 and std.math.isPowerOfTwo(cfg.chunk)) blk: {
                op = b.tensorOperation(.{});
                const start = try op.commit(op.fetch(prepared.first, .{}).subi(cfg.window - 1).binary(.maxi, 0).binary(.shr_arith, shift), .{});
                op = b.tensorOperation(.{});
                const count = try op.commit(op.fetch(prepared.keys, .{}).addi(cfg.chunk - 1).binary(.shr_arith, shift).subi(op.fetch(start, .{})), .{});
                // Each trip's chunk per row: TCC reads the loop index only
                // as an index, so the table is built before the loop.
                const trip = try b.arange(C.size, .i32, &.{C}, .{});
                op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
                const chunks = try op.commit(op.fetch(trip, .{}).addi(op.fetch(start, .{})), .{ .axes = &.{ C, rows } });
                break :blk .{ .chunks = chunks, .trips = try b.reduceMaxI32(count) };
            } else null;
            const n = if (window) |w| w.trips else try b.symExpr(.div, try b.symExpr(.add, try b.reduceMaxI32(prepared.kvlen), cfg.chunk - 1), cfg.chunk);
            var loop = b.openFor(n, .{ try b.full(0, .f32, stats), try b.full(0, .f32, out_axes), try b.full(score_min, .f32, stats) });
            {
                const sum_acc, const qkv_acc, const max_acc = loop.carried;
                const local_pos = try b.arange(cfg.chunk, .i32, &.{Ck}, .{});
                const local_pages = try b.arange(@divExact(cfg.chunk, cfg.pg), .i32, &.{Cp}, .{});
                // This trip's chunk: the same for every row, or each row's own.
                var row_pages: builder.Tensor = undefined;
                var position: builder.Tensor = undefined;
                var first_keys: builder.Tensor = undefined;
                if (window) |w| {
                    const chunk_index = try b.indexRead(w.chunks, loop.iv);
                    // Rows with fewer chunks reread the table's last one, past
                    // their keys, which the mask drops.
                    op = b.tensorOperation(.{});
                    const read_index = try op.commit(op.fetch(chunk_index, .{}).binary(.mini, @divExact(cfg.a, cfg.chunk) - 1), .{});
                    // A batched gather of the chunk does not lower for every
                    // shape in TCC 2026.5; a one-hot product does.
                    row_pages = try selectMetadata(b, by_row, try selector(b, read_index, C, C.size), C);
                    op = b.tensorOperation(.{});
                    const row_offset = try op.commit(op.fetch(chunk_index, .{}).muli(cfg.chunk), .{});
                    const offset = if (prepared.pick) |pick| try selectMetadata(b, row_offset, pick, a.S) else row_offset;
                    op = b.tensorOperation(.{});
                    position = try op.commit(op.fetch(local_pos, .{ .broadcast_to = &.{ a.B, Ck } }).addi(op.fetch(offset, .{})), .{});
                    op = b.tensorOperation(.{});
                    first_keys = try op.commit(op.fetch(local_pages, .{ .broadcast_to = &.{ a.B, Cp } }).muli(cfg.pg).addi(op.fetch(offset, .{})), .{});
                } else {
                    row_pages = try b.indexRead(pages, loop.iv);
                    const offset = try b.symExpr(.mul, loop.iv, cfg.chunk);
                    op = b.tensorOperation(.{});
                    position = try op.commit(op.fetch(local_pos, .{}).addi(op.fetch(offset, .{})), .{});
                    op = b.tensorOperation(.{});
                    first_keys = try op.commit(op.fetch(local_pages, .{}).muli(cfg.pg).addi(op.fetch(offset, .{})), .{});
                }
                const indices = try validIndices(b, try prepared.blockPages(b, a, row_pages), cfg.pages);
                const idx = indices.safe;
                op = b.tensorOperation(.{});
                const valid_pages = try op.commit(op.fetch(indices.valid, .{ .broadcast_to = &.{ a.B, Cp, a.PG } }), .{});
                const valid_keys = try b.reshape(valid_pages, &.{ a.B, Ck }, .{});
                op = b.tensorOperation(.{ .tactic = .einsum_by_ve });
                const remaining = try op.commit(op.fetch(prepared.kvlen, .{}).subi(op.fetch(first_keys, .{})), .{});
                op = b.tensorOperation(.{});
                const live_pages = try op.commit(op.where(remaining, .gt, 0, 1, 0), .{});
                op = b.tensorOperation(.{});
                const cvl = try op.commit(op.fetch(live_pages, .{}).reduce(&.{Cp}, .addi), .{});
                const mask = try attentionMask(b, cfg, prepared.qpos, prepared.kvlen, position, valid_keys);
                // The cache is contracted as is (see `unmasked_decode_keys`):
                // masked scores and zero weights drop the keys past the bound.
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
            }

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
    poison: enum { none, k, v, both } = .none,
    poison_value: f32 = std.math.nan(f32),
};

const TestDims = struct {
    batch: usize = 4,
    hkv: usize = 2,
    g: usize = 4,
    d: usize = 64,
    page: usize = 16,
    uniform: bool = false,
    uniform_scores: bool = false,
    /// Keys of every sequence valued 32768, the others 0: with equal scores
    /// each one moves the output by 32768 over the sequence's length.
    spikes: []const usize = &.{},
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
    inline for (.{ 257, 320 }) |max_pages| {
        try runCases(max_pages, 2 * max_pages, &.{.{
            .prefill = false,
            .seqs = &.{ .{ 1, @intCast(max_pages * 16 - 1) }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } },
        }}, .{ .batch = 1, .hkv = 8, .g = 4, .d = 128 });
    }
    inline for (.{ 5, 1025 }) |max_pages| {
        try runCases(max_pages, 2 * max_pages, &.{.{
            .prefill = false,
            .seqs = &.{ .{ 1, @intCast(max_pages * 48 - 1) }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } },
        }}, .{ .batch = 1, .hkv = 1, .g = 1, .d = 32, .page = 48 });
    }
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

test "tcl paged attention cache masking emits a BF16 select without query replication" {
    const Mask = tcl.Kernel(usize, .{
        .name = "cache_mask",
        .inputs = &.{ "values", "valid" },
        .outputs = &.{"out"},
        .run = struct {
            fn run(b: *tcl.Builder, _: usize) tcl.FinishError!void {
                const B = b.axis("B", 2);
                const A = b.axis("A", 64);
                const Hkv = b.axis("Hkv", 2);
                const D = b.axis("D", 32);
                const t = try b.declareArgs(.{
                    .values = .{ .dtype = .bf16, .axes = &.{ B, A, Hkv, D } },
                    .valid = .{ .dtype = .i32, .axes = &.{ B, A } },
                });
                b.ret(&.{try maskCacheValues(b, t.values, t.valid)});
            }
        }.run,
    });
    const ir = try Mask.emit(std.testing.allocator, 0);
    defer std.testing.allocator.free(ir);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, ir, "\"tcl.ve_select\""));
    try std.testing.expect(std.mem.indexOf(u8, ir, "!tcl.logical<bf16") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "!tcl.logical<i32") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "#tcl.ve_opcode<\"mulf\">") == null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "#tcl.symbol<\"I\">") == null);

    const decode = try Decode.emit(std.testing.allocator, .{ .t = 2, .a = 64, .group = 2, .hkv = 2, .g = 4, .d = 32, .pages = 8, .pg = 16, .scale = 0.125, .is_causal = true, .window = 0, .use_sink = false });
    defer std.testing.allocator.free(decode);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, decode, "\"tcl.dpe\""));
    try std.testing.expect(std.mem.indexOf(u8, decode, "tcl.fusible") != null);
    try std.testing.expect(std.mem.indexOf(u8, decode, "propagate_sparse_axis_from_op = \"Gather\"") != null);
}

test "tcl paged attention excludes poisoned unused cache entries on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    const seqs = [_][2]i32{ .{ 1, 17 }, .{ 1, 31 }, .{ 0, 0 }, .{ 0, 23 } };
    try runCases(8, 32, &.{
        .{ .prefill = false, .seqs = &seqs, .padding = 2, .poison = .k },
        .{ .prefill = false, .seqs = &seqs, .padding = 2, .poison = .v },
        .{ .prefill = false, .seqs = &seqs, .padding = 2, .poison = .both },
        .{ .prefill = false, .seqs = &seqs, .padding = 2, .poison = .both, .poison_value = std.math.inf(f32) },
        .{ .prefill = false, .seqs = &seqs, .padding = 2, .poison = .both, .poison_value = -std.math.inf(f32) },
        .{ .prefill = false, .seqs = &seqs, .padding = 2, .poison = .both, .poison_value = 1e30 },
        // A sink selects OneShot's compact decode softmax instead of Decode.
        .{ .prefill = false, .seqs = &seqs, .padding = 2, .poison = .both, .opts = .{ .scale = 0.2, .sink = &.{ 0, -std.math.inf(f32) } } },
        .{ .prefill = false, .seqs = &seqs, .padding = 2, .poison = .both, .mask_all = true, .padding_page = std.math.maxInt(i32) },
        .{ .prefill = false, .seqs = &.{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } }, .padding = 4, .poison = .both },
    }, .{ .hkv = 1, .g = 2, .d = 32 });

    const mixed = [_][2]i32{ .{ 37, 65 }, .{ 0, 19 }, .{ 1, 17 }, .{ 4, 2 } };
    try runCases(8, 32, &.{
        .{ .prefill = true, .seqs = &mixed, .padding = 3, .poison = .k },
        .{ .prefill = true, .seqs = &mixed, .padding = 3, .poison = .v },
        .{ .prefill = true, .seqs = &mixed, .padding = 3, .poison = .both, .opts = .{ .is_causal = false, .sliding_window = 17, .scale = 0.2, .sink = &.{ 0, -std.math.inf(f32) } } },
    }, .{ .hkv = 1, .g = 2, .d = 32 });

    const boundary = [_][2]i32{ .{ 37, 4097 }, .{ 0, 0 }, .{ 1, 4096 }, .{ 2, 17 } };
    try runCases(257, 1028, &.{
        .{ .prefill = true, .seqs = &boundary, .padding = 3, .poison = .both },
        .{ .prefill = true, .seqs = &boundary, .padding = 3, .poison = .both, .poison_value = std.math.inf(f32), .opts = .{ .is_causal = false, .sliding_window = 33, .scale = 0.2, .sink = &.{ 0, -std.math.inf(f32) } } },
        .{ .prefill = false, .seqs = &.{ .{ 1, 4095 }, .{ 1, 4096 }, .{ 1, 4097 }, .{ 0, 0 } }, .padding = 1, .poison = .both, .opts = .{ .sliding_window = 17, .sink = &.{ 0, -std.math.inf(f32) } } },
    }, .{ .hkv = 1, .g = 2, .d = 32 });
    try runCases(86, 344, &.{.{
        .prefill = true,
        .seqs = &.{ .{ 3, 4033 }, .{ 1, 4032 }, .{ 0, 0 }, .{ 1, 4031 } },
        .padding = 1,
        .poison = .both,
        .poison_value = -std.math.inf(f32),
        .opts = .{ .sliding_window = 49 },
    }}, .{ .hkv = 1, .g = 1, .d = 32, .page = 48 });
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

test "tcl paged attention hybrid decode on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    // One decode token over a table wider than the straight-line kernel's,
    // unused cache slots poisoned: head only, the first tail key, every tail
    // chunk, no keys, no page in the cache.
    try runCases(2049, 2101, &.{
        .{ .prefill = false, .seqs = &.{.{ 1, 300 }}, .poison = .both },
        .{ .prefill = false, .seqs = &.{.{ 1, 4097 }}, .poison = .both },
        .{ .prefill = false, .seqs = &.{.{ 1, 32784 }}, .poison = .both, .poison_value = std.math.inf(f32) },
        .{ .prefill = false, .seqs = &.{.{ 0, 0 }}, .padding = 1, .poison = .both },
        .{ .prefill = false, .seqs = &.{.{ 1, 17 }}, .mask_all = true, .poison = .both },
    }, .{ .batch = 1, .hkv = 1, .g = 2, .d = 32 });
    // Equal scores and spikes at the head's last key, the tail's first, a
    // chunk boundary and the last key: each one lost or counted twice moves
    // the output by about 1.
    try runCases(2049, 2101, &.{.{ .prefill = false, .seqs = &.{.{ 1, 32784 }}, .poison = .both }}, .{ .batch = 1, .hkv = 1, .g = 2, .d = 32, .uniform_scores = true, .spikes = &.{ 4095, 4096, 12287, 12288, 32783 } });
}

test "tcl paged attention flash decode of many tokens on furiosa" {
    if (zml.testing.env().target != .furiosa) return error.SkipZigTest;
    // 16 decode tokens over more than 32K keys, unused cache slots poisoned;
    // with a window, each token's loop starts at its own window.
    const seqs = [_][2]i32{ .{ 1, 32784 }, .{ 1, 4097 }, .{ 1, 1 }, .{ 0, 0 } } ++ @as([12][2]i32, @splat(.{ 1, 300 }));
    try runCases(2049, 2600, &.{
        .{ .prefill = false, .seqs = &seqs, .padding = 1, .poison = .both },
        .{ .prefill = false, .seqs = &seqs, .padding = 1, .poison = .both, .opts = .{ .sliding_window = 1024 } },
        .{ .prefill = false, .seqs = &seqs, .padding = 1, .poison = .both, .opts = .{ .sliding_window = 5000, .sink = &.{ 0, -std.math.inf(f32) } } },
    }, .{ .batch = 16, .hkv = 1, .g = 2, .d = 32 });
    // Prefill rows whose windows start deep in long sequences.
    try runCases(2049, 2700, &.{
        .{ .prefill = true, .seqs = &.{ .{ 37, 32784 }, .{ 5, 9000 } }, .padding = 2, .poison = .both, .opts = .{ .sliding_window = 1024 } },
        .{ .prefill = true, .seqs = &.{ .{ 70, 32784 }, .{ 0, 0 } }, .padding = 1, .poison = .both, .opts = .{ .sliding_window = 3000 } },
    }, .{ .batch = 2, .hkv = 1, .g = 2, .d = 32 });
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
    const k_t: Tensor = .init(.{ .page = num_pages, .k_chunk = page, .hkv = hkv, .hd = d }, dims.dtype);

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

        for (kc) |*v| v.* = .fromF32(if (dims.uniform or dims.uniform_scores) 0 else r.floatNorm(f32));
        for (vc, 0..) |*v, i| v.* = .fromF32(if (dims.uniform) @as(f32, @floatFromInt(i % d)) / d else r.floatNorm(f32));
        if (dims.spikes.len > 0) {
            for (case.seqs[0..batch], 0..) |seq, row| {
                for (0..@intCast(seq[1])) |key| {
                    const slot: usize = @intCast(table[row][key / page] * @as(i32, page) + @as(i32, @intCast(key % page)));
                    const spike = std.mem.indexOfScalar(usize, dims.spikes, key) != null;
                    @memset(vc[slot * hkv * d ..][0 .. hkv * d], bf16.fromF32(if (spike) 32768 else 0));
                }
            }
        }
        if (case.poison != .none) {
            // Mark the union of live physical slots before poisoning: reordered
            // or shared pages must retain every token a live sequence can read.
            const live = try allocator.alloc(bool, num_pages * page);
            defer allocator.free(live);
            @memset(live, false);
            for (case.seqs[0..batch], 0..) |s, seq| {
                if (s[0] == 0) continue;
                for (0..@intCast(s[1])) |key| {
                    const physical = table[seq][key / page];
                    if (physical < 0 or physical >= num_pages) continue;
                    live[@as(usize, @intCast(physical)) * page + key % page] = true;
                }
            }
            const poison = bf16.fromF32(case.poison_value);
            for (live, 0..) |valid, slot| {
                if (valid) continue;
                const start = slot * hkv * d;
                if (case.poison == .k or case.poison == .both) @memset(kc[start..][0 .. hkv * d], poison);
                if (case.poison == .v or case.poison == .both) @memset(vc[start..][0 .. hkv * d], poison);
            }
        }
        var bk = try testBuffer(platform, k_t.shape(), kc);
        defer bk.deinit();
        var bv = try testBuffer(platform, k_t.shape(), vc);
        defer bv.deinit();
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
