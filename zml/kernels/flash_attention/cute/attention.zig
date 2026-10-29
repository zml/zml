const std = @import("std");
const mlir = @import("mlir");
const llvm = @import("mlir/dialects/llvm");
const vector = @import("mlir/dialects/vector");

const zml = @import("../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const Value = cute.Value;

pub const Config = struct {
    batch: i64,
    query_length: i64,
    key_length: i64,
    query_heads: i64,
    kv_heads: i64,
    head_dim: i64,
    dtype: cute.DType,
    splits: i64 = 1,
};

/// Planning uses local (post-sharding) dimensions. XLA owns the compiled CUDA
/// function, so this is an estimated resource budget, not an occupancy query.
pub const DecodeResources = struct {
    multiprocessors: i64,
    shared_bytes: i64,
};

pub fn decodeSplits(q: zml.Shape, k: zml.Shape, resources: DecodeResources) i64 {
    const query_length = q.dim(.q);
    const key_length = k.dim(.k);
    const num_heads = q.dim(.h);
    const num_kv_heads = k.dim(.h);
    const head_dim = q.dim(.hd);
    const batch = if (q.hasTag(.b) != null) q.dim(.b) else 1;
    const dtype = q.dtype();

    if (query_length != 1 or key_length <= 64) return 1;
    if (resources.multiprocessors <= 0 or resources.shared_bytes <= 0) return 1;
    const group = @divExact(num_heads, num_kv_heads);
    const blocks_per_sequence = num_kv_heads * @divTrunc(group + 3, 4);
    const base_blocks = batch * blocks_per_sequence;
    var padded_dim: i64 = 32;
    while (padded_dim < head_dim) padded_dim *= 2;
    const bytes_per_element: i64 = if (dtype == .f32) 4 else 2;
    const shared_per_block = 2 * tile_keys * padded_dim * bytes_per_element;
    const blocks_per_sm = @max(1, @min(12, @divTrunc(resources.shared_bytes, shared_per_block)));
    // Multiple scheduling waves keep the serial tile loop supplied with work.
    // The measured tradeoff and distinction from CUDA occupancy are in README.
    const grid_budget = 4 * resources.multiprocessors * blocks_per_sm;
    if (base_blocks >= grid_budget) return 1;
    // Round up a partial wave: flooring can leave SMs idle and add a whole
    // extra KV tile to every block (e.g. 31 partitions versus 32).
    const split_budget = @divTrunc(grid_budget + base_blocks - 1, base_blocks);
    // One tile for short caches, two otherwise. FlashInfer's larger chunk
    // floor is tuned for a different per-block KV processing rate.
    // With static dense shapes, capacity bounds the scratch space and grid;
    // each invocation balances its live prefix over these partitions on device.
    const min_chunk: i64 = if (key_length <= 256) tile_keys else 2 * tile_keys;
    const chunk = @max(min_chunk, @divTrunc(key_length + split_budget - 1, split_budget));
    return @divTrunc(key_length + chunk - 1, chunk);
}

test "CuTe decode planner respects workload and resource limits" {
    const resources: DecodeResources = .{ .multiprocessors = 170, .shared_bytes = 101376 };
    var q = zml.Shape.init(.{ .b = 1, .q = 1, .h = 16, .hd = 128 }, .bf16);
    var k = zml.Shape.init(.{ .b = 1, .k = 2048, .h = 4, .hd = 128 }, .bf16);
    try std.testing.expectEqual(64, decodeSplits(q, k, resources));
    k = k.setDim(.k, 64);
    try std.testing.expectEqual(1, decodeSplits(q, k, resources));
    k = k.setDim(.k, 65);
    try std.testing.expectEqual(5, decodeSplits(q, k, resources));
    k = k.setDim(.k, 2048);
    var previous: i64 = 64;
    for ([_]i64{ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 }) |batch| {
        q = q.setDim(.b, batch);
        k = k.setDim(.b, batch);
        const splits = decodeSplits(q, k, resources);
        try std.testing.expect(splits <= previous);
        try std.testing.expect(splits >= 1);
        try std.testing.expect(splits == 1 or (splits - 1) * batch * 4 < 8160);
        previous = splits;
    }
    try std.testing.expectEqual(1, previous);
    q = q.setDim(.b, 16);
    k = k.setDim(.b, 16);
    q = q.setDim(.hd, 512);
    k = k.setDim(.hd, 512);
    q = q.withDtype(.f32);
    k = k.withDtype(.f32);
    try std.testing.expectEqual(11, decodeSplits(q, k, resources));
    q = q.setDim(.q, 7);
    try std.testing.expectEqual(1, decodeSplits(q, k, resources));
    q = q.setDim(.q, 1);
    try std.testing.expectEqual(1, decodeSplits(q, k, .{ .multiprocessors = 0, .shared_bytes = 0 }));
    // More query-head groups already create more independent CTAs.
    q = q.setDim(.b, 1);
    k = k.setDim(.b, 1);
    q = q.setDim(.hd, 128);
    k = k.setDim(.hd, 128);
    q = q.withDtype(.bf16);
    k = k.withDtype(.bf16);
    q = q.setDim(.h, 1024);
    try std.testing.expectEqual(32, decodeSplits(q, k, resources));
}

pub const block: [3]i32 = .{ 128, 1, 1 };
const tile_keys = 16;

pub const Kernel = cute.Kernel(Config, .{
    .name = "flash_attention_cute",
    .inputs = &.{ "q", "k", "v", "positions" },
    .outputs = &.{"out"},
    .run = run,
});

fn run(b: *B, cfg: Config) cute.FinishError!void {
    std.debug.assert(cfg.head_dim > 0 and cfg.head_dim <= 512);
    inline for (.{ 1, 2, 4, 8, 16 }) |fragments| {
        if (cfg.head_dim <= fragments * 32) return runSized(b, cfg, fragments);
    }
    unreachable;
}

/// Four warps own four (query position, query head) rows sharing one KV head.
/// Only a KV tile lives in shared memory; maxima, sums, and outputs stay in
/// registers. All lanes participate in shuffles, including padded rows/features.
fn runSized(b: *B, cfg: Config, comptime fragments: usize) cute.FinishError!void {
    const padded_dim = fragments * 32;
    const split = cfg.splits > 1;
    const out_shape = if (split) &[_]i64{ cfg.batch, cfg.splits, cfg.query_length, cfg.query_heads, cfg.head_dim + 2 } else &[_]i64{ cfg.batch, cfg.query_length, cfg.query_heads, cfg.head_dim };
    const a = try b.declareArgs(.{
        .q = .{ .tensor = .{ .dtype = cfg.dtype, .shape = &.{ cfg.batch, cfg.query_length, cfg.query_heads, cfg.head_dim } } },
        .k = .{ .tensor = .{ .dtype = cfg.dtype, .shape = &.{ cfg.batch, cfg.key_length, cfg.kv_heads, cfg.head_dim } } },
        .v = .{ .tensor = .{ .dtype = cfg.dtype, .shape = &.{ cfg.batch, cfg.key_length, cfg.kv_heads, cfg.head_dim } } },
        .positions = .{ .tensor = .{ .dtype = .i32, .shape = &.{cfg.batch} } },
        .out = .{ .tensor = .{ .dtype = if (split) .f32 else cfg.dtype, .shape = out_shape } },
    });
    const tid = b.threadIdx().x;
    const lane = tid.rem(32);
    const group = @divExact(cfg.query_heads, cfg.kv_heads);
    const batch = b.blockIdx().y.div(cfg.kv_heads);
    const kv_head = b.blockIdx().y.rem(cfg.kv_heads);
    const first_row = b.blockIdx().x.mul(4);
    const row = first_row.add(tid.div(32));
    // Padded warps compute a valid dummy row and suppress their final stores.
    const query = row.div(group).minimum(cfg.query_length - 1);
    const head = kv_head.mul(group).add(row.rem(group));
    const start = a.positions.get(.{batch});
    const position = start.add(query);
    const causal_end = start.add(first_row.add(3).div(group).minimum(cfg.query_length - 1)).add(1);
    // Partition the live cache, rather than its capacity, so decode uses all
    // blocks even when only a prefix of the cache is populated.
    const chunk = causal_end.add(cfg.splits - 1).div(cfg.splits);
    const begin = if (split) b.blockIdx().z.mul(chunk) else b.constant(.i32, 0);
    const end = if (split) begin.add(chunk).minimum(causal_end) else causal_end;
    const zero = b.constant(.f32, 0);
    const neg_inf = b.constant(.f32, -std.math.inf(f32));

    var queries: [fragments]Value = undefined;
    inline for (0..fragments) |f| {
        const d = lane.add(f * 32);
        var valid = b.openIfElse(d.lt(cfg.head_dim), .{.f32});
        {
            valid.yieldThen(.{a.q.get(.{ batch, query, head, d }).to(.f32)});
        }
        {
            valid.yieldElse(.{zero});
        }
        queries[f] = valid.results[0];
    }

    const layout = b.makeLayout(&.{ tile_keys, padded_dim }, .{ .stride = &.{ padded_dim, 1 } });
    const keys = b.makeTensor(b.allocSmem(cfg.dtype, tile_keys * padded_dim, 16), layout);
    const values = b.makeTensor(b.allocSmem(cfg.dtype, tile_keys * padded_dim, 16), layout);
    var initial: [fragments + 2]Value = @splat(zero);
    initial[0] = neg_inf;
    var tiles = b.openFor(begin, end, tile_keys, initial);
    {

        // Zero-fill partial tiles and feature padding before the CTA-wide barrier.
        var copy = b.openFor(tid, tile_keys * padded_dim, 128, .{});
        {
            const tile_key = copy.iv.div(padded_dim);
            const d = copy.iv.rem(padded_dim);
            const key = tiles.iv.add(tile_key);
            var valid_load = b.openIfElse(key.lt(end).bitAnd(d.lt(cfg.head_dim)), .{ cfg.dtype, cfg.dtype });
            {
                valid_load.yieldThen(.{ a.k.get(.{ batch, key, kv_head, d }), a.v.get(.{ batch, key, kv_head, d }) });
            }
            {
                valid_load.yieldElse(.{ zero.to(cfg.dtype), zero.to(cfg.dtype) });
            }
            keys.set(.{ tile_key, d }, valid_load.results[0]);
            values.set(.{ tile_key, d }, valid_load.results[1]);
            copy.close();
        }
        b.syncThreads();

        // Each lane keeps one score; dot products reduce across the feature lanes.
        var scores = b.openFor(0, tile_keys, 1, .{neg_inf});
        {
            var dot = zero;
            inline for (0..fragments) |f| {
                dot = dot.add(queries[f].mul(keys.get(.{ scores.iv, lane.add(f * 32) }).to(.f32)));
            }
            inline for (.{ 16, 8, 4, 2, 1 }) |offset| dot = dot.add(b.shuffleXor(dot, offset));
            const scale = std.math.log2e / std.math.sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
            const score = b.select(tiles.iv.add(scores.iv).le(position).bitAnd(tiles.iv.add(scores.iv).lt(end)), dot.mul(scale), neg_inf);
            scores.yield(.{b.select(lane.eq(scores.iv), score, scores.carried[0])});
        }
        var maximum = scores.results[0];
        inline for (.{ 16, 8, 4, 2, 1 }) |offset| maximum = maximum.maximum(b.shuffleXor(maximum, offset));
        maximum = maximum.maximum(tiles.carried[0]);
        const rescale = b.exp2Approx(tiles.carried[0].sub(maximum));
        const probability = b.exp2Approx(scores.results[0].sub(maximum));
        var sum = probability;
        inline for (.{ 16, 8, 4, 2, 1 }) |offset| sum = sum.add(b.shuffleXor(sum, offset));
        sum = sum.add(tiles.carried[1].mul(rescale));

        var accumulators: [fragments]Value = undefined;
        inline for (0..fragments) |f| accumulators[f] = tiles.carried[f + 2].mul(rescale);
        var weighted = b.openFor(0, tile_keys, 1, accumulators);
        {
            const weight = b.shuffleIdx(probability, weighted.iv);
            inline for (0..fragments) |f| {
                accumulators[f] = weighted.carried[f].add(weight.mul(values.get(.{ weighted.iv, lane.add(f * 32) }).to(.f32)));
            }
            weighted.yield(accumulators);
        }
        // No warp may overwrite the tile before every warp has consumed it.
        b.syncThreads();
        var next: [fragments + 2]Value = undefined;
        next[0] = maximum;
        next[1] = sum;
        inline for (0..fragments) |f| next[f + 2] = weighted.results[f];
        tiles.yield(next);
    }

    inline for (0..fragments) |f| {
        const col = lane.add(f * 32);
        var valid_store = b.openIf(row.lt(cfg.query_length * group).bitAnd(col.lt(cfg.head_dim)));
        {
            if (split) {
                a.out.set(.{ batch, b.blockIdx().z, query, head, col }, tiles.results[f + 2]);
            } else {
                a.out.set(.{ batch, query, head, col }, tiles.results[f + 2].div(tiles.results[1]));
            }
            valid_store.close();
        }
    }
    if (split) {
        var stats = b.openIf(row.lt(cfg.query_length * group).bitAnd(lane.eq(0)));
        {
            a.out.set(.{ batch, b.blockIdx().z, query, head, cfg.head_dim }, tiles.results[0]);
            a.out.set(.{ batch, b.blockIdx().z, query, head, cfg.head_dim + 1 }, tiles.results[1]);
            stats.close();
        }
    }
}

/// Merge unnormalized partial attention states in FP32. Empty partitions have
/// m=-inf, l=0, u=0 and contribute zero, including at the first decode token.
pub const Merge = cute.Kernel(Config, .{
    .name = "flash_attention_cute_merge",
    .inputs = &.{"partials"},
    .outputs = &.{"out"},
    .run = merge,
});

fn merge(b: *B, cfg: Config) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .partials = .{ .tensor = .{ .dtype = .f32, .shape = &.{ cfg.batch, cfg.splits, cfg.query_heads, cfg.head_dim + 2 } } },
        .out = .{ .tensor = .{ .dtype = cfg.dtype, .shape = &.{ cfg.batch, cfg.query_heads, cfg.head_dim } } },
    });
    const batch = b.blockIdx().y;
    const head = b.blockIdx().x;
    const lane = b.threadIdx().x.rem(32);
    const zero = b.constant(.f32, 0);
    const neg_inf = b.constant(.f32, -std.math.inf(f32));
    // Every warp reduces the partition statistics, then independently handles
    // a contiguous range of output features.
    var maxima = b.openFor(lane, cfg.splits, 32, .{neg_inf});
    {
        maxima.yield(.{maxima.carried[0].maximum(a.partials.get(.{ batch, maxima.iv, head, cfg.head_dim }))});
    }
    var maximum = maxima.results[0];
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| maximum = maximum.maximum(b.shuffleXor(maximum, offset));
    const weights = b.makeTensor(b.allocSmem(.f32, cfg.splits, 16), b.makeLayout(&.{cfg.splits}, .{}));
    var stats = b.openFor(b.threadIdx().x, cfg.splits, 128, .{});
    {
        weights.set(.{stats.iv}, b.exp2Approx(a.partials.get(.{ batch, stats.iv, head, cfg.head_dim }).sub(maximum)));
        stats.close();
    }
    b.syncThreads();
    var features = b.openFor(b.threadIdx().x, cfg.head_dim, 128, .{});
    {
        var sum = b.openFor(0, cfg.splits, 1, .{ zero, zero });
        {
            const weight = weights.get(.{sum.iv});
            sum.yield(.{
                sum.carried[0].add(weight.mul(a.partials.get(.{ batch, sum.iv, head, features.iv }))),
                sum.carried[1].add(weight.mul(a.partials.get(.{ batch, sum.iv, head, cfg.head_dim + 1 }))),
            });
        }
        a.out.set(.{ batch, head, features.iv }, sum.results[0].div(sum.results[1]));
        features.close();
    }
}

// ==================== Prefill ====================
//
// FlashAttention-2 forward for `query_length > 1`, with the tiling FA2's sm80 kernel uses on sm_120
// (`Flash_fwd_kernel_traits<128, 128, 64, 4>`): a 128-thread block owns `prefillRows` queries of one
// head and every warp owns whole rows (all the keys of a tile, the whole head dimension). Scores and
// output stay in `mma.sync` accumulators, so P is reused from registers as the A operand of P @ V,
// without cross-warp reductions. Q, K and V tiles go through swizzled shared memory: V_j is copied
// while S_j = Q K_j^T is computed, K_{j+1} while O += P_j V_j is.
//
// It is a program, not a bare kernel: its host function launches the kernel with the dynamic shared
// memory the kernel allocates (up to 96 KiB). A bare kernel's shared memory is static, which ptxas
// caps at 48 KiB before sm_120.

pub const Prefill = cute.Program(Config, .{
    .name = "flash_attention_cute_prefill",
    .inputs = &.{ "q", "k", "v", "positions" },
    .outputs = &.{"out"},
    .run = prefillProgram,
});

const prefill_kernel_name = "flash_attention_cute_prefill_kernel";

/// Keys per tile (FA2's kBlockN).
const prefill_keys = 64;

/// Whether `Prefill` handles `cfg`; the other configurations use `Kernel`.
/// Same as FA2 we only support 16-bit inputs
pub fn prefillSupported(cfg: Config) bool {
    const dtype_ok = cfg.dtype == .bf16 or cfg.dtype == .f16;
    const head_dim_ok = cfg.head_dim == 32 or cfg.head_dim == 64 or cfg.head_dim == 128 or cfg.head_dim == 256;
    return cfg.query_length > 1 and cfg.splits == 1 and dtype_ok and head_dim_ok;
}

/// m16 row tiles per warp: two up to head dimension 128, one above so the output accumulators fit in registers.
fn prefillRowTiles(head_dim: i64) i64 {
    return if (head_dim <= 128) 2 else 1;
}

/// Queries per block (FA2's kBlockM).
fn prefillRows(cfg: Config) i64 {
    return 4 * 16 * prefillRowTiles(cfg.head_dim);
}

/// Query tiles on x, so the blocks resident at the same time share the K/V of one head in L2.
fn prefillGrid(cfg: Config) [3]i32 {
    return .{ @intCast(@divTrunc(cfg.query_length + prefillRows(cfg) - 1, prefillRows(cfg))), @intCast(cfg.query_heads), @intCast(cfg.batch) };
}

fn prefillProgram(b: *B, cfg: Config) cute.FinishError!void {
    std.debug.assert(prefillSupported(cfg));
    switch (cfg.head_dim) {
        inline 32, 64, 128, 256 => |d| switch (cfg.dtype) {
            inline .bf16, .f16 => |dt| {
                b.beginFunction(prefill_kernel_name, .cuda_kernel);
                try prefillSized(b, cfg, d, dt);
                b.endFunction(block);

                b.beginFunction(Prefill.name, .host);
                const host = try b.declareArgs(.{
                    .q = .{ .ptr = dt },
                    .k = .{ .ptr = dt },
                    .v = .{ .ptr = dt },
                    .positions = .{ .ptr = cute.DType.i32 },
                    .out = .{ .ptr = dt },
                });
                const grid = prefillGrid(cfg);
                const config = b.makeLaunchConfig(.{
                    .grid = .{ b.constant(.i32, grid[0]), b.constant(.i32, grid[1]), b.constant(.i32, grid[2]) },
                    .block = .{ b.constant(.i32, block[0]), b.constant(.i32, block[1]), b.constant(.i32, block[2]) },
                    .dynamic_smem = b.kernelSmemSize(prefill_kernel_name),
                    .stream = b.cudaStream(),
                });
                const launch = b.launchEx(prefill_kernel_name, config, .{ host.q, host.k, host.v, host.positions, host.out });
                b.returnHostStatus(b.cudaResultStatus(launch));
                b.endFunction(null);
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

/// PTX `max.f32` (FMNMX). `Value.maximum` and `Value.maxnum` both lower to NaN-handling compare and
/// select sequences (three instructions); prefill scores are never NaN, masked ones are -inf.
fn fmax(b: *B, x: Value, y: Value) Value {
    return b.callIntrinsic("llvm.nvvm.fmax.f", &.{ x, y }, .float(b.ctx, .f32)).?;
}

fn unpackStruct(b: *B, comptime n: usize, value: Value, dtype: cute.DType) [n]Value {
    var result: [n]Value = undefined;
    for (&result, 0..) |*r, i| {
        r.* = b.emit(llvm.extractvalue(b.ctx, value.inner, dtype.toMlir(b.ctx), &.{@intCast(i)}, b.loc()));
    }
    return result;
}

/// `ldmatrix.x4`: lanes 8i..8i+7 address the rows of matrix i, the result holds the four 8x8 fragments.
fn matrixLoad(b: *B, comptime transpose: bool, address: Value) [4]Value {
    const assembly = "ldmatrix.sync.aligned.m8n8.x4" ++ (if (transpose) ".trans" else "") ++ ".shared.b16 {$0, $1, $2, $3}, [$4];";
    const result = b.emit(llvm.inline_asm(b.ctx, assembly, &.{address.inner}, .{
        .result_type = llvm.structType(b.ctx, &@as([4]*const mlir.Type, @splat(.int(b.ctx, .i32)))),
        .constraints = "=r,=r,=r,=r,r",
        .has_side_effects = true,
    }, b.loc()));
    return unpackStruct(b, 4, result, .i32);
}

/// `mma.sync.m16n8k16` with f32 accumulators: acc += a (16x16, row) @ b (16x8, col).
fn mma(b: *B, comptime dtype: cute.DType, a: [4]Value, rhs: [2]Value, acc: [4]Value) [4]Value {
    const assembly = "mma.sync.aligned.m16n8k16.row.col.f32." ++ @tagName(dtype) ++ "." ++ @tagName(dtype) ++ ".f32 " ++
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};";
    const result = b.emit(llvm.inline_asm(b.ctx, assembly, &.{ a[0].inner, a[1].inner, a[2].inner, a[3].inner, rhs[0].inner, rhs[1].inner, acc[0].inner, acc[1].inner, acc[2].inner, acc[3].inner }, .{
        .result_type = llvm.structType(b.ctx, &@as([4]*const mlir.Type, @splat(.float(b.ctx, .f32)))),
        .constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
    }, b.loc()));
    return unpackStruct(b, 4, result, .f32);
}

/// The four f32 accumulators of one mma.sync, carried across loop iterations as one value (the loop
/// carries at most 64 values).
fn packAcc(b: *B, values: [4]Value) Value {
    return b.emit(vector.from_elements(b.ctx, &.{ values[0].inner, values[1].inner, values[2].inner, values[3].inner }, .vector(&.{4}, .float(b.ctx, .f32)), b.loc()));
}

fn unpackAcc(b: *B, value: Value) [4]Value {
    var result: [4]Value = undefined;
    for (&result, 0..) |*r, i| {
        r.* = b.vectorExtract(value, @intCast(i), .f32);
    }
    return result;
}

/// Comptime layout of one prefill block, shared by the setup and the per-tile steps.
///
/// Tiles are row major with the 16-byte vector index of every row XORed with row % swizzle_rows, so
/// 16-byte cp.async stores and 8-row ldmatrix reads are free of bank conflicts. Shared memory addresses
/// are folded the way CuTe's static layouts let nvcc fold them: the XORed row bits of every address a
/// thread uses are a per-thread constant, and the k-step part of a column offset is bit-disjoint from
/// the per-thread part, so each ldmatrix / cp.async address is a per-thread base plus a constant
/// (XORed into the 16-byte vector index).
fn PrefillTile(comptime d: i64, comptime dt: cute.DType) type {
    return struct {
        const row_tiles = prefillRowTiles(d);
        const rows = 4 * 16 * row_tiles;
        const n = prefill_keys;
        const score_tiles = n / 8;
        const out_tiles = d / 8;
        const row_bytes = d * 2;
        /// The XORed vector index must stay in its row: rows of d=32 only have 4 vectors.
        const swizzle_rows = @min(8, d / 8);
        const q_bytes = rows * row_bytes;
        const kv_bytes = n * row_bytes;
        /// Rows covered by one 16-byte cp.async per thread of the block.
        const copy_rows = @divExact(128, d / 8);
        /// Running max and per-thread partial sum of the two rows of every row tile.
        const stats = 2 * row_tiles;
        /// Carried by the key loop: maxima, sums, then the packed output accumulators of every
        /// (row tile, 8 output features).
        const carried = 2 * stats + row_tiles * out_tiles;
        const scale: f32 = std.math.log2e / std.math.sqrt(@as(f32, @floatFromInt(d)));

        comptime {
            // copy_rows rows starting at a multiple of copy_rows never carry into the XORed bits.
            std.debug.assert(std.math.isPowerOfTwo(copy_rows));
            std.debug.assert(std.math.isPowerOfTwo(swizzle_rows));
        }

        b: *B,
        cfg: Config,
        lane: Value,
        warp: Value,
        first_query: Value,
        start: Value,
        q_smem: Value,
        k_smem: Value,
        v_smem: Value,
        k_ptr: Value,
        v_ptr: Value,
        kv_base: Value,
        tiles: Value,
        /// cp.async: this thread copies the 16-byte vector `copy_col` of rows `copy_row + i * copy_rows`.
        copy_row: Value,
        copy_col: Value,
        /// Byte offset in a tile of this thread's first cp.async destination, without the XORed part.
        copy_row_offset: Value,
        copy_vector_offset: Value,
        /// ldmatrix rows of this thread, as byte offsets, and their XORed 16-byte vector offsets.
        q_frag_row: Value,
        k_frag_row: Value,
        v_frag_row: Value,
        qv_frag_vector: Value,
        k_frag_vector: Value,

        const Self = @This();

        fn init(b: *B, cfg: Config, tid: Value, first_query: Value, start: Value, shared: Value, k_ptr: Value, v_ptr: Value, kv_base: Value, tiles: Value) Self {
            const lane = tid.rem(32);
            const warp = tid.div(32);
            const copy_row = tid.div(d / 8);
            const copy_col = tid.rem(d / 8);
            const lane8 = lane.rem(8);
            // The fragment rows a thread reads are lane % 8 plus multiples of 8.
            const lane_xor = lane.rem(swizzle_rows);
            return .{
                .b = b,
                .cfg = cfg,
                .lane = lane,
                .warp = warp,
                .first_query = first_query,
                .start = start,
                .q_smem = shared,
                .k_smem = shared.add(q_bytes),
                .v_smem = shared.add(q_bytes + kv_bytes),
                .k_ptr = k_ptr,
                .v_ptr = v_ptr,
                .kv_base = kv_base,
                .tiles = tiles,
                .copy_row = copy_row,
                .copy_col = copy_col,
                .copy_row_offset = copy_row.mul(row_bytes),
                .copy_vector_offset = copy_col.bitXor(copy_row.rem(swizzle_rows)).mul(16),
                // Q (A operand): rows lane % 16, vectors lane / 16 (+ 2 per k-step).
                .q_frag_row = shared.add(warp.mul(16 * row_tiles).add(lane.rem(16)).mul(row_bytes)),
                // K (B operand): keys lane % 8 + 8 * (lane / 16), vectors (lane / 8) % 2 (+ 2 per k-step).
                .k_frag_row = shared.add(q_bytes).add(lane8.add(lane.div(16).mul(8)).mul(row_bytes)),
                // V (B operand, transposed): keys lane % 8 + 8 * ((lane / 8) % 2), vectors lane / 16 (+ 2 per 16 features).
                .v_frag_row = shared.add(q_bytes + kv_bytes).add(lane8.add(lane.div(8).rem(2).mul(8)).mul(row_bytes)),
                .qv_frag_vector = lane.div(16).bitXor(lane_xor).mul(16),
                .k_frag_vector = lane.div(8).rem(2).bitXor(lane_xor).mul(16),
            };
        }

        /// Destination of the i-th cp.async of this thread in a tile starting at `base`.
        fn copyDst(t: Self, base: Value, comptime i: usize) Value {
            const xor = ((copy_rows * i) % swizzle_rows) * 16;
            const vector_offset = if (xor == 0) t.copy_vector_offset else t.copy_vector_offset.bitXor(xor);
            return base.add(t.copy_row_offset).add(vector_offset).add(copy_rows * i * row_bytes);
        }

        /// Copy keys `key_base..key_base + n` of `src` into `dst`; keys past the end are zero-filled.
        fn loadTile(t: Self, src: Value, dst: Value, key_base: Value) void {
            const b = t.b;
            const stride = t.cfg.kv_heads * d;
            const first_key = key_base.add(t.copy_row);
            const safe = src.add(t.kv_base.mul(2));
            const first = safe.add(first_key.to(.i64).mul(stride).add(t.copy_col.mul(8).to(.i64)).mul(2));
            inline for (0..@divExact(n, copy_rows)) |i| {
                const valid = first_key.add(copy_rows * i).lt(t.cfg.key_length);
                const source = b.select(valid, first.add(copy_rows * i * stride * 2), safe);
                b.cpAsync16(t.copyDst(dst, i), source, valid);
            }
        }

        fn queryOf(t: Self, mi: usize, h: usize) Value {
            return t.first_query.add(t.warp.mul(16 * row_tiles)).add(mi * 16).add(t.lane.div(4)).add(h * 8);
        }

        fn accIndex(comptime mi: usize, comptime di: usize) usize {
            return @intCast(2 * stats + @as(i64, mi) * out_tiles + @as(i64, di));
        }

        /// One key tile. `masked` tiles apply the causal and key length masks; the other ones are
        /// fully visible to every query of the block.
        fn step(t: Self, comptime masked: bool, tile: Value, carried_in: [carried]Value) [carried]Value {
            const b = t.b;
            const lane = t.lane;
            const zero = b.constant(.f32, 0);
            const neg_inf = b.constant(.f32, -std.math.inf(f32));
            const key_base = tile.mul(n);
            b.cpAsyncWaitAll();
            b.syncThreads();
            t.loadTile(t.v_ptr, t.v_smem, key_base);

            // S = Q K^T, k-steps of 16 features.
            var s: [row_tiles][score_tiles][4]Value = @splat(@splat(@splat(zero)));
            inline for (0..d / 16) |ks| {
                var qa: [row_tiles][4]Value = undefined;
                inline for (0..row_tiles) |mi| {
                    qa[mi] = matrixLoad(b, false, t.q_frag_row.add(mi * 16 * row_bytes).add(t.qv_frag_vector.bitXor(ks * 32)));
                }
                const k_vector = t.k_frag_vector.bitXor(ks * 32);
                inline for (0..score_tiles / 2) |np| {
                    const kb = matrixLoad(b, false, t.k_frag_row.add(np * 16 * row_bytes).add(k_vector));
                    inline for (0..row_tiles) |mi| {
                        s[mi][2 * np] = mma(b, dt, qa[mi], .{ kb[0], kb[1] }, s[mi][2 * np]);
                        s[mi][2 * np + 1] = mma(b, dt, qa[mi], .{ kb[2], kb[3] }, s[mi][2 * np + 1]);
                    }
                }
            }

            // Online softmax in base 2 on the raw scores, the scale is folded into the exponent:
            // p = exp2(s * scale - m * scale). Thread rows: lane / 4 (accumulators 0, 1) and lane / 4 + 8 (2, 3).
            var next: [carried]Value = undefined;
            var alpha: [row_tiles][2]Value = undefined;
            inline for (0..row_tiles) |mi| {
                if (masked) {
                    inline for (0..score_tiles) |ni| {
                        inline for (0..4) |j| {
                            const key = key_base.add(ni * 8).add(lane.rem(4).mul(2)).add(j % 2);
                            const visible = key.lt(t.cfg.key_length).bitAnd(key.le(t.start.add(t.queryOf(mi, j / 2))));
                            s[mi][ni][j] = b.select(visible, s[mi][ni][j], neg_inf);
                        }
                    }
                }
                inline for (0..2) |h| {
                    var row_max = s[mi][0][h * 2];
                    inline for (0..score_tiles) |ni| {
                        inline for (0..2) |j| {
                            row_max = fmax(b, row_max, s[mi][ni][h * 2 + j]);
                        }
                    }
                    inline for (.{ 2, 1 }) |offset| {
                        row_max = fmax(b, row_max, b.shuffleXor(row_max, offset));
                    }
                    const old_max = carried_in[mi * 2 + h];
                    const new_max = fmax(b, old_max, row_max);
                    // A row without any visible key so far keeps m = -inf: exponentiate against a finite value.
                    const neg_scaled_max = fmax(b, new_max, b.constant(.f32, -std.math.floatMax(f32))).mul(-scale);
                    alpha[mi][h] = b.exp2Approx(old_max.mul(scale).add(neg_scaled_max));
                    next[mi * 2 + h] = new_max;
                    var sum = carried_in[stats + mi * 2 + h].mul(alpha[mi][h]);
                    inline for (0..score_tiles) |ni| {
                        inline for (0..2) |j| {
                            s[mi][ni][h * 2 + j] = b.exp2Approx(s[mi][ni][h * 2 + j].mul(scale).add(neg_scaled_max));
                            sum = sum.add(s[mi][ni][h * 2 + j]);
                        }
                    }
                    next[stats + mi * 2 + h] = sum;
                }
            }

            // All warps are done with K_j once V_j has landed: start copying K_{j+1}.
            b.cpAsyncWaitAll();
            b.syncThreads();
            var prefetch = b.openIf(tile.add(1).lt(t.tiles));
            {
                t.loadTile(t.k_ptr, t.k_smem, key_base.add(n));
                prefetch.close();
            }

            // O = O * alpha + P V. The accumulators of two adjacent score tiles are the A operand of one k-step.
            var o: [row_tiles][out_tiles][4]Value = undefined;
            inline for (0..row_tiles) |mi| {
                inline for (0..out_tiles) |di| {
                    const prev = unpackAcc(b, carried_in[accIndex(mi, di)]);
                    inline for (0..4) |j| {
                        o[mi][di][j] = prev[j].mul(alpha[mi][j / 2]);
                    }
                }
            }
            inline for (0..n / 16) |ks| {
                var pa: [row_tiles][4]Value = undefined;
                inline for (0..row_tiles) |mi| {
                    pa[mi] = .{
                        b.packHalf2(s[mi][2 * ks][0], s[mi][2 * ks][1], dt),
                        b.packHalf2(s[mi][2 * ks][2], s[mi][2 * ks][3], dt),
                        b.packHalf2(s[mi][2 * ks + 1][0], s[mi][2 * ks + 1][1], dt),
                        b.packHalf2(s[mi][2 * ks + 1][2], s[mi][2 * ks + 1][3], dt),
                    };
                }
                const v_row = t.v_frag_row.add(ks * 16 * row_bytes);
                inline for (0..out_tiles / 2) |dp| {
                    const vb = matrixLoad(b, true, v_row.add(t.qv_frag_vector.bitXor(dp * 32)));
                    inline for (0..row_tiles) |mi| {
                        o[mi][2 * dp] = mma(b, dt, pa[mi], .{ vb[0], vb[1] }, o[mi][2 * dp]);
                        o[mi][2 * dp + 1] = mma(b, dt, pa[mi], .{ vb[2], vb[3] }, o[mi][2 * dp + 1]);
                    }
                }
            }
            inline for (0..row_tiles) |mi| {
                inline for (0..out_tiles) |di| {
                    next[accIndex(mi, di)] = packAcc(b, o[mi][di]);
                }
            }
            return next;
        }
    };
}

fn prefillSized(b: *B, cfg: Config, comptime d: i64, comptime dt: cute.DType) cute.FinishError!void {
    const T = PrefillTile(d, dt);
    const a = try b.declareArgs(.{
        .q = .{ .ptr = dt },
        .k = .{ .ptr = dt },
        .v = .{ .ptr = dt },
        .positions = .{ .ptr = cute.DType.i32 },
        .out = .{ .ptr = dt },
    });
    b.setFunctionAttribute(.named(b.ctx, "cu_attrs", .dict(b.ctx, &.{.named(b.ctx, "max_dynamic_shared_size_bytes", b.devMaxSharedMemoryOptin())})));

    const tid = b.threadIdx().x;
    const first_query = b.blockIdx().x.mul(T.rows);
    const head = b.blockIdx().y;
    const batch = b.blockIdx().z;
    const kv_head = head.div(@divExact(cfg.query_heads, cfg.kv_heads));
    // One shared storage object (like a Python CuTe `SharedStorage`): the host compiler sizes the
    // launch's dynamic shared memory from it, which `kernelSmemSize` returns. `allocSmem` and
    // `allocSmemView` both become static shared memory instead.
    b.setFunctionAttribute(.named(b.ctx, "smem.partition_num", .int(b.ctx, .i32, 1)));
    const fields = [_][]const u8{
        std.fmt.comptimePrint("sQ:{d}:0", .{T.q_bytes}),
        std.fmt.comptimePrint("sK:{d}:{d}", .{ T.kv_bytes, T.q_bytes }),
        std.fmt.comptimePrint("sV:{d}:{d}", .{ T.kv_bytes, T.q_bytes + T.kv_bytes }),
    };
    const smem = b.allocSmemStorage(.i8, 128, b.layoutType(b.layoutSpec(.{T.q_bytes + 2 * T.kv_bytes}, .{1})), 0, &fields);
    const shared = b.ptrToInt(b.getIter(smem, b.ptrTy(.i8, .smem, 128) catch @panic("bad shared memory pointer")).value(), .i32);
    const q_ptr = b.ptrToInt(a.q, .i64);
    const out_ptr = b.ptrToInt(a.out, .i64);
    const start = b.loadWords(1, b.ptrToInt(a.positions, .i64).add(batch.to(.i64).mul(4)), .global, .{})[0];

    // Element offsets of this (batch, head) in q/out and (batch, kv head) in k/v, and the q/out row stride.
    const q_stride = cfg.query_heads * d;
    const q_base = batch.to(.i64).mul(cfg.query_length * q_stride).add(head.to(.i64).mul(d));
    const kv_base = batch.to(.i64).mul(cfg.key_length * cfg.kv_heads * d).add(kv_head.to(.i64).mul(d));

    // Causal: the last query of the block sees keys up to start + last_query, and every query of the
    // block sees the keys up to start + first_query.
    const last_query = first_query.add(T.rows).minimum(cfg.query_length).sub(1);
    const tiles = start.add(last_query).add(1).minimum(cfg.key_length).add(T.n - 1).div(T.n);
    const full_tiles = start.add(first_query).add(1).minimum(cfg.key_length).div(T.n).minimum(tiles);

    const t: T = .init(b, cfg, tid, first_query, start, shared, b.ptrToInt(a.k, .i64), b.ptrToInt(a.v, .i64), kv_base, tiles);

    // Out of range rows are zero-filled and never read from global memory.
    {
        const first_row = first_query.add(t.copy_row);
        const safe = q_ptr.add(q_base.mul(2));
        const first = safe.add(first_row.to(.i64).mul(q_stride).add(t.copy_col.mul(8).to(.i64)).mul(2));
        inline for (0..@divExact(T.rows, T.copy_rows)) |i| {
            const valid = first_row.add(T.copy_rows * i).lt(cfg.query_length);
            b.cpAsync16(t.copyDst(t.q_smem, i), b.select(valid, first.add(T.copy_rows * i * q_stride * 2), safe), valid);
        }
    }
    t.loadTile(t.k_ptr, t.k_smem, b.constant(.i32, 0));

    var initial: [T.carried]Value = undefined;
    const zero = b.constant(.f32, 0);
    for (initial[0..T.stats]) |*v| {
        v.* = b.constant(.f32, -std.math.inf(f32));
    }
    for (initial[T.stats .. 2 * T.stats]) |*v| {
        v.* = zero;
    }
    for (initial[2 * T.stats ..]) |*v| {
        v.* = packAcc(b, .{ zero, zero, zero, zero });
    }

    var full = b.openFor(0, full_tiles, 1, initial);
    {
        full.yield(t.step(false, full.iv, full.carried[0..T.carried].*));
    }
    var diagonal = b.openFor(full_tiles, tiles, 1, full.results[0..T.carried].*);
    {
        diagonal.yield(t.step(true, diagonal.iv, diagonal.carried[0..T.carried].*));
    }
    const results = diagonal.results[0..T.carried].*;

    // Normalize by the row sums (reduced over the quad) and store pairs of features as one 32-bit word.
    inline for (0..T.row_tiles) |mi| {
        inline for (0..2) |h| {
            var sum = results[T.stats + mi * 2 + h];
            inline for (.{ 2, 1 }) |offset| {
                sum = sum.add(b.shuffleXor(sum, offset));
            }
            const norm = b.select(sum.gt(0), b.rcpApprox(sum), zero);
            const query = t.queryOf(mi, h);
            var valid = b.openIf(query.lt(cfg.query_length));
            {
                const row_offset = q_base.add(query.to(.i64).mul(q_stride)).add(t.lane.rem(4).mul(2).to(.i64));
                inline for (0..T.out_tiles) |di| {
                    const acc = unpackAcc(b, results[T.accIndex(mi, di)]);
                    const word = b.packHalf2(acc[h * 2].mul(norm), acc[h * 2 + 1].mul(norm), dt);
                    b.storeWords(1, out_ptr.add(row_offset.add(di * 8).mul(2)), .global, .{word});
                }
                valid.close();
            }
        }
    }
}

test "CuTe attention emits tiled online softmax through head dimension 512" {
    for ([_]i64{ 64, 80, 128, 256, 512 }) |head_dim| {
        const ir = try Kernel.emit(std.testing.allocator, .{
            .batch = 2,
            .query_length = 7,
            .key_length = 37,
            .query_heads = 6,
            .kv_heads = 2,
            .head_dim = head_dim,
            .dtype = .bf16,
        }, block);
        defer std.testing.allocator.free(ir);
        try std.testing.expect(std.mem.indexOf(u8, ir, "scf.for") != null);
    }
}
