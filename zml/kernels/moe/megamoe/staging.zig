//! Kernel #0 of the MegaMoE Program: stages one rank's inputs into its symmetric buffer, what
//! vLLM's `prepare_megamoe_inputs` (a Triton kernel) and DeepGEMM's `tests/test_mega_moe.py`
//! do before `fp8_fp4_mega_moe`. DeepGEMM itself has no staging kernel; its kernel reads:
//!
//!   x            `[T][H]` e4m3              (`buffer.x`)
//!   x_sf         `[T][H/128]` i32, K-major  (`buffer.x_sf`, 4 packed UE8M0 bytes per word)
//!   topk_idx     `[T][K]` i64, -1 = no route
//!   topk_weights `[T][K]` f32
//!   shared_l1_sf MN-major `[H/128][num_max_shared_sf_tokens]` i32 (`buffer.shared_l1_acts_sf`):
//!                token `t`'s `x_sf` word `g` at row `(t / BLOCK_M) * align(BLOCK_M, 128) +
//!                transposed(t % BLOCK_M)` (DeepGEMM's `_to_shared_mega_moe_sf_layout`,
//!                vLLM's `_shared_sf_row`), for the BLOCK_M of this call's kernel.
//!
//! Two input kinds:
//! - `.staged`: `x`/`x_sf`/`topk_idx`/`topk_weights` already in DeepGEMM's format: copies, and
//!   the shared-expert SF pages built from `x_sf`.
//! - `.bf16`: `x` bf16, `topk_idx` i32: DeepGEMM's `per_token_cast_to_fp8(x, use_ue8m0=True,
//!   gran_k=32, use_packed_ue8m0=True)` per 32-element group (the reference of vLLM's staging
//!   test), i32 -> i64 ids:
//!     amax = max(max_i |x_i|, 1e-4f);  sf = amax / 448 (IEEE)
//!     e = clamp(((bits(sf) >> 23) & 0xFF) + ((bits(sf) & 0x7FFFFF) != 0), 1, 254)
//!     x_fp8_i = e4m3_rn(x_i * (1 / 2^(e - 127)));  sf byte = e (byte j of the word = group j)
//!   Deviations (not bit-exact, non-finite inputs only, like vLLM's Triton staging): `max.f32`
//!   drops NaN where torch's `amax` propagates it, and `cvt.rn.satfinite` gives 448 for an
//!   infinite product where torch gives NaN.
//!
//! Layout (bandwidth-bound): each thread owns 16 consecutive columns (bf16: two 16-byte loads;
//! FP8: one 16-byte load; one 16-byte store either way), so lanes `2j, 2j+1` own one 32-element
//! group (amax with one `shfl.xor`) and lanes `8i..8i+7` one 128-column SF word: lane `8i`
//! gathers its 4 SF bytes with shuffles and stores the word once into `x_sf` and once into the
//! shared page (`.staged`: copies the input word). A CTA has `alignUp(H / 16, 32)` threads (at
//! most 1024; wider rows take several column chunks, `grid.y`) and stages `tokens_per_cta`
//! consecutive tokens, every load issued before the first store. Threads 0..K-1 of chunk 0 copy
//! the tokens' top-k ids and weights. Columns past `H` and tokens past `T` read an in-bounds
//! address and store nothing.
const std = @import("std");

const cute = @import("kernels/cute/builder");

const layout = @import("layout.zig");
const ptx = @import("ptx.zig");

const B = cute.Builder;
const V = cute.Value;

pub const Inputs = enum { staged, bf16 };

pub const cols_per_thread: u32 = 16;
pub const max_threads: u32 = 1024;
/// Tokens per CTA (loads in flight per thread).
pub const tokens_per_cta: u32 = 2;

pub const Config = struct {
    inputs: Inputs,
    num_tokens: u32,
    hidden: u32,
    num_topk: u32,
    /// BLOCK_M of the MegaMoE kernel this call runs (the shared-expert SF page size); 0 without
    /// a shared expert (no pages written).
    shared_block_m: u32,
    /// Rows of the MN-major shared SF region (`get_num_max_shared_sf_tokens`).
    num_shared_sf_tokens: u32,
    // Region offsets in the symmetric buffer.
    x_offset: u64,
    x_sf_offset: u64,
    topk_idx_offset: u64,
    topk_weights_offset: u64,
    shared_sf_offset: u64,

    pub fn init(k: layout.Kernel, buf: layout.MegaMoEBuffer, inputs: Inputs, num_tokens: u32) Config {
        const s = k.shape;
        std.debug.assert(num_tokens >= 1 and num_tokens <= s.num_max_tokens_per_rank and num_tokens <= 65535);
        std.debug.assert(s.hidden % 128 == 0 and s.num_topk <= 32);
        return .{
            .inputs = inputs,
            .num_tokens = num_tokens,
            .hidden = s.hidden,
            .num_topk = s.num_topk,
            .shared_block_m = if (k.hasShared()) k.cfg.block_m else 0,
            .num_shared_sf_tokens = layout.numMaxSharedSfTokens(s.num_max_tokens_per_rank),
            .x_offset = buf.input_token.offset,
            .x_sf_offset = buf.input_sf.offset,
            .topk_idx_offset = buf.input_topk_idx.offset,
            .topk_weights_offset = buf.input_topk_weights.offset,
            .shared_sf_offset = buf.shared_l1_sf.offset,
        };
    }

    /// Threads per CTA: one per 16 columns, whole warps, at most `max_threads`.
    pub fn numThreads(self: Config) u32 {
        return @min(max_threads, layout.alignUp(self.hidden / cols_per_thread, 32));
    }

    pub fn colsPerCta(self: Config) u32 {
        return self.numThreads() * cols_per_thread;
    }

    pub fn numChunks(self: Config) u32 {
        return layout.ceilDiv(self.hidden, self.colsPerCta());
    }

    pub fn numTokenBlocks(self: Config) u32 {
        return layout.ceilDiv(self.num_tokens, tokens_per_cta);
    }

    /// Host twin of the device row computation: the shared SF row of token `t`.
    pub fn sharedSfRow(self: Config, t: u32) u32 {
        const bm = self.shared_block_m;
        const m = t % bm;
        return t / bm * layout.alignUp(bm, 128) + (m / 128) * 128 + (m % 32) * 4 + (m % 128) / 32;
    }
};

/// Kernel parameters, all `i64` addresses: `x`, `x_sf` (`.staged` only, else unused),
/// `topk_idx`, `topk_weights`, `sym_base` (this rank's symmetric buffer).
pub const Args = struct {
    x: V,
    x_sf: V,
    topk_idx: V,
    topk_weights: V,
    sym_base: V,

    const names = [_][]const u8{ "x", "x_sf", "topk_idx", "topk_weights", "sym_base" };
};

/// Emits the staging kernel as the `cuda.kernel` `name` of the open program.
pub fn emit(b: *B, cfg: Config, name: []const u8) cute.FinishError!void {
    b.beginFunction(name, .cuda_kernel);
    const i64t = cute.DType.i64.toMlir(b.ctx);
    const specs = b.arena.allocator().alloc(cute.ArgSpec, Args.names.len) catch @panic("OOM");
    for (specs, Args.names) |*s, n| s.* = .{ .name = n, .kind = .{ .mlir_type = .{ .type_ = i64t } } };
    try b.declareArgList(specs);
    body(b, cfg, .{ .x = b.arg(0), .x_sf = b.arg(1), .topk_idx = b.arg(2), .topk_weights = b.arg(3), .sym_base = b.arg(4) });
    b.endFunction(.{ @as(i32, @intCast(cfg.numThreads())), 1, 1 });
}

/// `div.rn.f32` (IEEE division, what torch's `/` on CUDA tensors is).
fn divRn(b: *B, x: V, y: V) V {
    return ptx.asmValue(b, "div.rn.f32 $0, $1, $2;", &.{ x, y }, .f32, "=f,f,f", false);
}

/// `mul.rn.f32` (no flush to zero, no contraction).
fn mulRn(b: *B, x: V, y: V) V {
    return ptx.asmValue(b, "mul.rn.f32 $0, $1, $2;", &.{ x, y }, .f32, "=f,f,f", false);
}

fn f32Const(b: *B, comptime x: f32) V {
    return b.cst(.i32, @as(i32, @bitCast(x))).bitCast(.f32);
}

/// One token's loaded inputs: `.staged`: 4 words of FP8 `x` and (lane `8i`) the SF word;
/// `.bf16`: 8 words of bf16 `x`.
const Loaded = struct { x: [8]V, sf_word: V };

fn body(b: *B, cfg: Config, a: Args) void {
    const H: i64 = cfg.hidden;
    const K: i64 = cfg.num_topk;
    const U = tokens_per_cta;
    const tid = b.threadIdx().x;
    const lane = tid.bitAnd(@as(i32, 31));
    const chunk = b.blockIdx().y;
    const first_token = b.blockIdx().x.mul(@as(i32, @intCast(U)));

    const col = chunk.mul(@as(i32, @intCast(cfg.colsPerCta()))).add(tid.mul(@as(i32, @intCast(cols_per_thread))));
    const valid_col = col.lt(@as(i32, @intCast(cfg.hidden)));
    // Idle columns read the token's last 16 columns (in bounds) and store nothing. Lanes
    // `8i..8i+7` (128 columns, `H % 128 == 0`) are all valid or all idle.
    const col64 = b.select(valid_col, col, @as(i32, @intCast(cfg.hidden - cols_per_thread))).to(.i64);
    const sf_word_idx = col64.shrLogical(7);
    const sf_leader = valid_col.bitAnd(lane.bitAnd(@as(i32, 7)).eq(0));

    var tokens: [U]V = undefined;
    var valid_token: [U]V = undefined;
    for (0..U) |u| {
        const t = first_token.add(@as(i32, @intCast(u)));
        valid_token[u] = t.lt(@as(i32, @intCast(cfg.num_tokens)));
        tokens[u] = b.select(valid_token[u], t, @as(i32, @intCast(cfg.num_tokens - 1))).to(.i64);
    }

    // Every load first.
    var loaded: [U]Loaded = undefined;
    for (0..U) |u| {
        const elem = tokens[u].mul(H).add(col64);
        switch (cfg.inputs) {
            .staged => {
                const w = ptx.ldGlobal(b, 4, a.x.add(elem));
                for (0..4) |i| loaded[u].x[i] = w[i];
                loaded[u].sf_word = ptx.ldGlobal(b, 1, a.x_sf.add(tokens[u].mul(@divExact(H, 128)).add(sf_word_idx).mul(4)))[0];
            },
            .bf16 => {
                const lo = ptx.ldGlobal(b, 4, a.x.add(elem.mul(2)));
                const hi = ptx.ldGlobal(b, 4, a.x.add(elem.mul(2)).add(16));
                for (0..4) |i| {
                    loaded[u].x[i] = lo[i];
                    loaded[u].x[4 + i] = hi[i];
                }
            },
        }
    }

    for (0..U) |u| {
        const token = tokens[u];
        const x_dst = a.sym_base.add(@as(i64, @intCast(cfg.x_offset))).add(token.mul(H).add(col64));
        var out: [4]V = undefined;
        var sf_word: V = undefined;
        switch (cfg.inputs) {
            .staged => {
                for (0..4) |i| out[i] = loaded[u].x[i];
                sf_word = loaded[u].sf_word;
            },
            .bf16 => {
                var f: [16]V = undefined;
                for (0..8) |i| f[2 * i], f[2 * i + 1] = ptx.bfloat1622float2(b, loaded[u].x[i]);
                // `x_view.abs().float().amax(dim=2).clamp(1e-4)`
                var amax = ptx.fabsf(b, f[0]);
                for (f[1..]) |v| amax = ptx.fmaxf(b, amax, ptx.fabsf(b, v));
                amax = ptx.fmaxf(b, amax, ptx.shflXorSync(b, amax, @as(i32, 1)));
                amax = ptx.fmaxf(b, amax, f32Const(b, 1e-4));
                // `sf = ceil_to_ue8m0(amax / 448.0)`
                const bits = divRn(b, amax, f32Const(b, 448.0)).bitCast(.i32);
                var e = bits.shrLogical(23).bitAnd(@as(i32, 0xFF)).add(ptx.zext(b, bits.bitAnd(@as(i32, 0x7FFFFF)).ne(0), .i32));
                e = ptx.minU(b, ptx.maxU(b, e, @as(i32, 1)), @as(i32, 254));
                // `x_view * (1.0 / sf)`: `reciprocal(sf)`, then an exact power-of-two scaling.
                const inv = divRn(b, f32Const(b, 1.0), e.shl(23).bitCast(.f32));
                var q: [16]V = undefined;
                for (&q, f) |*o, v| o.* = mulRn(b, v, inv);
                for (0..4) |i| out[i] = ptx.fp8x4E4m3(b, q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
                // The SF word of lanes `8i..8i+7`: byte j = group j (lanes `8i + 2j`).
                sf_word = e;
                for (1..4) |j| sf_word = sf_word.bitOr(ptx.shflSync(b, e, lane.add(@as(i32, @intCast(2 * j)))).shl(@as(i32, @intCast(8 * j))));
            },
        }
        var store = b.openIf(valid_col.bitAnd(valid_token[u]));
        ptx.stGlobal(b, 4, x_dst, out);
        store.yieldThen(.{});

        // Lane 8i: the SF word, K-major `x_sf` and (with a shared expert) the shared page.
        var leader = b.openIf(sf_leader.bitAnd(valid_token[u]));
        ptx.stGlobal(b, 1, a.sym_base.add(@as(i64, @intCast(cfg.x_sf_offset))).add(token.mul(@divExact(H, 128)).add(sf_word_idx).mul(4)), .{sf_word});
        if (cfg.shared_block_m != 0) {
            const bm: i64 = cfg.shared_block_m;
            const m = token.rem(bm);
            const row = token.div(bm).mul(@as(i64, layout.alignUp(cfg.shared_block_m, 128)))
                .add(m.div(128).mul(128))
                .add(m.rem(32).mul(4))
                .add(m.rem(128).div(32));
            const word = sf_word_idx.mul(@as(i64, cfg.num_shared_sf_tokens)).add(row);
            ptx.stGlobal(b, 1, a.sym_base.add(@as(i64, @intCast(cfg.shared_sf_offset))).add(word.mul(4)), .{sf_word});
        }
        leader.yieldThen(.{});

        // Chunk 0, threads 0..K-1: top-k ids (i64; `.bf16`: sign-extended i32) and weights.
        var topk = b.openIf(chunk.eq(0).bitAnd(tid.lt(@as(i32, @intCast(cfg.num_topk)))).bitAnd(valid_token[u]));
        const i = token.mul(K).add(tid.to(.i64));
        const idx = switch (cfg.inputs) {
            .staged => ptx.ldGlobalU64(b, a.topk_idx.add(i.mul(8))),
            .bf16 => ptx.ldGlobal(b, 1, a.topk_idx.add(i.mul(4)))[0].to(.i64),
        };
        ptx.stGlobalU64(b, a.sym_base.add(@as(i64, @intCast(cfg.topk_idx_offset))).add(i.mul(8)), idx);
        const w = ptx.ldGlobal(b, 1, a.topk_weights.add(i.mul(4)));
        ptx.stGlobal(b, 1, a.sym_base.add(@as(i64, @intCast(cfg.topk_weights_offset))).add(i.mul(4)), w);
        topk.yieldThen(.{});
    }
}

/// Launches the staging kernel `name` (emitted by `emit(b, cfg, name)`) from the current host
/// function: grid (token blocks, chunks), `numThreads()` threads, no cluster, no PDL (it overwrites input regions the
/// previous MegaMoE call of this rank has finished reading, by stream order). Returns the
/// launch status (i32).
pub fn emitLaunch(b: *B, cfg: Config, name: []const u8, args: Args) V {
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.cst(.i32, @as(i32, @intCast(cfg.numTokenBlocks()))), b.cst(.i32, @as(i32, @intCast(cfg.numChunks()))), one },
        .block = .{ b.cst(.i32, @as(i32, @intCast(cfg.numThreads()))), one, one },
        .dynamic_smem = b.kernelSmemSize(name),
        .stream = b.cudaStream(),
    });
    return b.cudaResultStatus(b.launchEx(name, config, .{ args.x, args.x_sf, args.topk_idx, args.topk_weights, args.sym_base }));
}

test "shared SF rows: DeepGEMM's `_to_shared_mega_moe_sf_layout`" {
    var cfg: Config = undefined;
    cfg.shared_block_m = 192;
    // Page 0 (192 tokens in a 256-row page): m -> (m/128)*128 + (m%32)*4 + (m%128)/32.
    try std.testing.expectEqual(@as(u32, 0), cfg.sharedSfRow(0));
    try std.testing.expectEqual(@as(u32, 4), cfg.sharedSfRow(1));
    try std.testing.expectEqual(@as(u32, 1), cfg.sharedSfRow(32));
    try std.testing.expectEqual(@as(u32, 128 + 4 * 31 + 1), cfg.sharedSfRow(128 + 32 + 31));
    // Token 192 starts page 1 at row 256.
    try std.testing.expectEqual(@as(u32, 256), cfg.sharedSfRow(192));
}
