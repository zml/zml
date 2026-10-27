const std = @import("std");
const mlir = @import("mlir");
const llvm = @import("mlir/dialects/llvm");
const vector = @import("mlir/dialects/vector");

const zml = @import("../../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;

pub const Config = struct {
    head_dim: i64,
    tokens: i64,
    batch: i64,
    kv_heads: i64,
    group: i64,
    pages_per_seq: i64,
    page_size: i64,
    max_query: i64,
    dtype: cute.DType,
    warps_per_query: i64,
    kv_tile: i64,
    causal: bool,
    window: i64,
    scale: f32,
    sink: bool,
    splits: i64 = 1,

    pub fn rows(c: Config) i64 {
        return @divExact(64, c.warps_per_query);
    }
    pub fn queryTiles(c: Config) i64 {
        return @divTrunc(c.max_query * c.group + c.rows() - 1, c.rows());
    }
    fn kvBase(c: Config) i64 {
        return c.rows() * c.head_dim * 2;
    }
    fn pBase(c: Config) i64 {
        return c.kvBase() + c.kv_tile * c.head_dim * 2;
    }
    fn redBase(c: Config) i64 {
        return c.pBase() + c.rows() * c.kv_tile * 2;
    }
    fn sharedBytes(c: Config) i64 {
        return c.redBase() + 2 * 64 * 4;
    }
};

pub const Kernel = cute.Kernel(Config, .{
    .name = "paged_attention_cute_sm120",
    .inputs = &.{ "q", "k", "v", "table", "lengths", "starts", "sink" },
    .outputs = &.{"out"},
    .run = run,
});

fn flatten(comptime n: usize, values: [n][4]V) [n * 4]V {
    var result: [n * 4]V = undefined;
    for (0..n) |i| @memcpy(result[i * 4 ..][0..4], &values[i]);
    return result;
}

fn packAcc(b: *B, values: [4]V) V {
    return b.emit(vector.from_elements(b.ctx, &.{ values[0].inner, values[1].inner, values[2].inner, values[3].inner }, .vector(&.{4}, .float(b.ctx, .f32)), b.loc()));
}

fn loadI32(b: *B, ptr: V, index: V) V {
    return b.loadWords(1, ptr.add(index.to(.i64).mul(4)), .global, .{})[0];
}

// XOR whole 16-byte vectors so cp.async and ldmatrix agree on bank layout.
fn swizzle(row: V, col: V, width: i64) V {
    return row.mul(width).add(col.bitXor(row.rem(@min(8, @divExact(width, 8))).mul(8))).mul(2);
}

fn unpackStruct(b: *B, comptime n: usize, value: V, dtype: cute.DType) [n]V {
    var result: [n]V = undefined;
    for (&result, 0..) |*r, i| r.* = b.emit(llvm.extractvalue(b.ctx, value.inner, dtype.toMlir(b.ctx), &.{@intCast(i)}, b.loc()));
    return result;
}

fn matrixLoad(b: *B, comptime n: usize, comptime transpose: bool, address: V) [n]V {
    const assembly = std.fmt.comptimePrint("ldmatrix.sync.aligned.m8n8.x{d}{s}.shared.b16 {{", .{ n, if (transpose) ".trans" else "" }) ++
        (if (n == 4) "$0, $1, $2, $3}, [$4];" else "$0, $1}, [$2];");
    const result = b.emit(llvm.inline_asm(b.ctx, assembly, &.{address.inner}, .{ .result_type = llvm.structType(b.ctx, &@as([n]*const mlir.Type, @splat(.int(b.ctx, .i32)))), .constraints = if (n == 4) "=r,=r,=r,=r,r" else "=r,=r,r", .has_side_effects = true }, b.loc()));
    return unpackStruct(b, n, result, .i32);
}

fn mma(b: *B, comptime dtype: cute.DType, a: [4]V, k: [2]V, acc: [4]V) [4]V {
    const assembly = "mma.sync.aligned.m16n8k16.row.col.f32." ++ @tagName(dtype) ++ "." ++ @tagName(dtype) ++ ".f32 " ++
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};";
    const result = b.emit(llvm.inline_asm(b.ctx, assembly, &.{ a[0].inner, a[1].inner, a[2].inner, a[3].inner, k[0].inner, k[1].inner, acc[0].inner, acc[1].inner, acc[2].inner, acc[3].inner }, .{ .result_type = llvm.structType(b.ctx, &@as([4]*const mlir.Type, @splat(.float(b.ctx, .f32)))), .constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f" }, b.loc()));
    return unpackStruct(b, 4, result, .f32);
}

fn run(b: *B, c: Config) cute.FinishError!void {
    switch (c.head_dim) {
        inline 32, 64, 128, 256, 512 => |d| switch (c.warps_per_query) {
            inline 2, 4 => |w| switch (c.kv_tile) {
                inline 32, 64 => |n| switch (c.dtype) {
                    inline .bf16, .f16 => |dt| try runTile(b, c, d, w, n, dt),
                    else => unreachable,
                },
                else => unreachable,
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

const TileContext = struct {
    b: *B,
    c: Config,
    q: V,
    k: V,
    v: V,
    table: V,
    sink: V,
    out: V,
    shared: V,
    tid: V,
    lane: V,
    warp: V,
    seq: V,
    head: V,
    first_row: V,
    start: V,
    qlen: V,
    klen: V,
    split: V,

    // Every load resolves its logical page on device. Tail copies are zeroed
    // before either MMA, including V tails whose backing allocation holds NaN.
    fn loadKV(t: TileContext, source: V, key_base: V) void {
        const b = t.b;
        const c = t.c;
        if (c.page_size == 16 and c.head_dim >= 64) {
            // A page ID is shared by all feature vectors in its 16 rows.
            // Resolve it once, then issue independent vector copies instead
            // of serializing a page-table load before every cp.async.
            var pages = b.openFor(0, c.kv_tile, 16, .{});
            {
                const first_key = key_base.add(pages.iv);
                const page_valid = first_key.lt(t.klen);
                const logical_page = b.select(page_valid, first_key.div(16), b.constant(.i32, 0));
                const page = loadI32(b, t.table, t.seq.mul(c.pages_per_seq).add(logical_page));
                const base = page.to(.i64).mul(16 * c.kv_heads * c.head_dim).add(t.head.to(.i64).mul(c.head_dim));
                for (0..@as(usize, @intCast(@divExact(c.head_dim, 64)))) |i| {
                    const vector_index = t.tid.add(i * 128);
                    const row = vector_index.div(@divExact(c.head_dim, 8));
                    const col = vector_index.rem(@divExact(c.head_dim, 8)).mul(8);
                    const offset = base.add(row.to(.i64).mul(c.kv_heads * c.head_dim)).add(col.to(.i64));
                    b.cpAsync16(t.shared.add(c.kvBase()).add(swizzle(pages.iv.add(row), col, c.head_dim)), source.add(offset.mul(2)), first_key.add(row).lt(t.klen));
                }
                pages.close();
            }
            return;
        }
        var copy = b.openFor(t.tid, c.kv_tile * @divExact(c.head_dim, 8), 128, .{});
        {
            const row = copy.iv.div(@divExact(c.head_dim, 8));
            const col = copy.iv.rem(@divExact(c.head_dim, 8)).mul(8);
            const key = key_base.add(row);
            const valid = key.lt(t.klen);
            const safe_key = b.select(valid, key, b.constant(.i32, 0));
            const page = loadI32(b, t.table, t.seq.mul(c.pages_per_seq).add(safe_key.div(c.page_size)));
            const offset = page.to(.i64).mul(c.page_size).add(safe_key.rem(c.page_size).to(.i64)).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.head_dim).add(col.to(.i64));
            b.cpAsync16(t.shared.add(c.kvBase()).add(swizzle(row, col, c.head_dim)), source.add(offset.mul(2)), valid);
            copy.close();
        }
    }

    fn publishReduction(t: TileContext, row: V, value: V, comptime maximum: bool) void {
        const b = t.b;
        var v = value;
        inline for (.{ 2, 1 }) |step| v = if (maximum) v.maximum(b.shuffleXor(v, step)) else v.add(b.shuffleXor(v, step));
        const address = t.shared.add(t.c.redBase() + (if (maximum) @as(i64, 0) else 256)).add(t.warp.mul(16).add(row).mul(4));
        var leader = b.openIf(t.lane.rem(4).eq(0));
        {
            b.storeWords(1, address, .shared, .{v.bitCast(.i32)});
            leader.close();
        }
        // Callers synchronize once both row halves have been published.
    }

    fn readReduction(t: TileContext, comptime w: usize, row: V, comptime maximum: bool) V {
        const b = t.b;
        const base = t.shared.add(t.c.redBase() + (if (maximum) @as(i64, 0) else 256)).add(t.warp.div(w).mul(w * 16).add(row).mul(4));
        var result = b.loadWords(1, base, .shared, .{})[0].bitCast(.f32);
        for (1..w) |i| {
            const v = b.loadWords(1, base.add(i * 16 * 4), .shared, .{})[0].bitCast(.f32);
            result = if (maximum) result.maximum(v) else result.add(v);
        }
        return result;
    }

    fn store(t: TileContext, row: V, dim: V, value: V) void {
        const c = t.c;
        const token = t.start.add(row.div(c.group));
        var offset = token.to(.i64).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.group).add(row.rem(c.group).to(.i64));
        if (c.splits > 1) {
            offset = t.split.to(.i64).mul(c.tokens * c.kv_heads * c.group).add(offset).mul(c.head_dim + 2).add(dim.to(.i64));
            t.b.storeWords(1, t.out.add(offset.mul(4)), .global, .{value.bitCast(.i32)});
        } else {
            offset = offset.mul(c.head_dim).add(dim.to(.i64));
            t.b.storeHalf(t.out.add(offset.mul(2)), .global, value.to(c.dtype).bitCast(.i16).to(.i32));
        }
    }
};

fn runTile(b: *B, c: Config, comptime d: usize, comptime w: usize, comptime n: usize, comptime dt: cute.DType) cute.FinishError!void {
    std.debug.assert(c.sharedBytes() <= 99 * 1024);
    const a = try b.declareArgs(.{
        .q = .{ .ptr = dt },
        .k = .{ .ptr = dt },
        .v = .{ .ptr = dt },
        .table = .{ .ptr = cute.DType.i32 },
        .lengths = .{ .ptr = cute.DType.i32 },
        .starts = .{ .ptr = cute.DType.i32 },
        .sink = .{ .ptr = cute.DType.f32 },
        .out = .{ .ptr = if (c.splits > 1) cute.DType.f32 else dt },
    });
    b.setFunctionAttribute(.named(b.ctx, "cu_attrs", .dict(b.ctx, &.{.named(b.ctx, "max_dynamic_shared_size_bytes", b.devMaxSharedMemoryOptin())})));
    const seq = b.blockIdx().x.div(c.queryTiles());
    const first_row = b.blockIdx().x.rem(c.queryTiles()).mul(c.rows());
    const starts = b.ptrToInt(a.starts, .i64);
    const start = loadI32(b, starts, seq);
    const qlen = loadI32(b, starts, seq.add(1)).sub(start);
    var active = b.openIf(first_row.lt(qlen.mul(c.group)));
    {
        const tid = b.threadIdx().x;
        const t: TileContext = .{
            .b = b,
            .c = c,
            .q = b.ptrToInt(a.q, .i64),
            .k = b.ptrToInt(a.k, .i64),
            .v = b.ptrToInt(a.v, .i64),
            .table = b.ptrToInt(a.table, .i64),
            .sink = b.ptrToInt(a.sink, .i64),
            .out = b.ptrToInt(a.out, .i64),
            .shared = b.ptrToInt(b.allocSmem(.i8, c.sharedBytes(), 128), .i32),
            .tid = tid,
            .lane = tid.rem(32),
            .warp = tid.div(32),
            .seq = seq,
            .head = b.blockIdx().y,
            .first_row = first_row,
            .start = start,
            .qlen = qlen,
            .klen = loadI32(b, b.ptrToInt(a.lengths, .i64), seq),
            .split = b.blockIdx().z,
        };
        var copy_q = b.openFor(tid, c.rows() * (d / 8), 128, .{});
        {
            const row = copy_q.iv.div(d / 8);
            const col = copy_q.iv.rem(d / 8).mul(8);
            const packed_row = first_row.add(row);
            const valid = packed_row.lt(qlen.mul(c.group));
            const token = start.add(b.select(valid, packed_row.div(c.group), b.constant(.i32, 0)));
            const offset = token.to(.i64).mul(c.kv_heads).add(t.head.to(.i64)).mul(c.group).add(packed_row.rem(c.group).to(.i64)).mul(d).add(col.to(.i64));
            b.cpAsync16(t.shared.add(swizzle(row, col, d)), t.q.add(offset.mul(2)), valid);
            copy_q.close();
        }
        const first_pos = t.klen.sub(qlen).add(first_row.div(c.group));
        const last_pos = t.klen.sub(qlen).add(first_row.add(c.rows() - 1).div(c.group).minimum(qlen.sub(1)));
        const begin = if (c.window > 0) first_pos.sub(c.window - 1).maximum(0) else b.constant(.i32, 0);
        const end = if (c.causal) last_pos.add(1).minimum(t.klen) else t.klen;
        const first_tile = begin.div(n);
        const tiles = end.add(n - 1).div(n).sub(first_tile).maximum(0);
        const lo = first_tile.add(tiles.mul(t.split).div(c.splits));
        const hi = first_tile.add(tiles.mul(t.split.add(1)).div(c.splits));
        const row_base = t.warp.div(w).mul(16);
        const key_part = t.warp.rem(w).mul(n / w);
        const dim_part = t.warp.rem(w).mul(d / w);
        const row0 = t.lane.div(4);
        const col0 = t.lane.rem(4).mul(2);
        const zero = b.constant(.f32, 0);
        const neg_inf = b.constant(.f32, -std.math.inf(f32));
        const nf = n / w / 8;
        const of = d / w / 8;
        const count = 4 + of;
        var initial: [count]V = @splat(zero);
        initial[0] = neg_inf;
        initial[1] = neg_inf;
        for (initial[4..]) |*v| v.* = packAcc(b, .{ zero, zero, zero, zero });
        b.cpAsyncWaitAll();
        b.syncThreads();
        var loop = b.openFor(lo, hi, 1, initial);
        {
            const key_base = loop.iv.mul(n);
            t.loadKV(t.k, key_base);
            b.cpAsyncWaitAll();
            b.syncThreads();
            var scores: [nf][4]V = @splat(@splat(zero));
            // Keep only MMA operands live in registers; Q remains in shared
            // memory so D512 does not consume the entire per-thread register file.
            var qk = b.openFor(0, d, 16, @as([nf * 4]V, @splat(zero)));
            {
                const qa = matrixLoad(b, 4, false, t.shared.add(swizzle(row_base.add(t.lane.rem(16)), qk.iv.add(t.lane.div(16).mul(8)), d)));
                for (0..nf) |i| {
                    const kb = matrixLoad(b, 2, false, t.shared.add(c.kvBase()).add(swizzle(key_part.add(i * 8).add(t.lane.rem(8)), qk.iv.add(t.lane.div(8).rem(2).mul(8)), d)));
                    scores[i] = mma(b, dt, qa, kb, qk.carried[i * 4 ..][0..4].*);
                }
                qk.yield(flatten(nf, scores));
            }
            for (0..nf) |i| scores[i] = qk.results[i * 4 ..][0..4].*;
            var local_max: [2]V = @splat(neg_inf);
            for (0..nf) |i| {
                for (0..4) |j| {
                    const row = first_row.add(row_base).add(row0).add((j / 2) * 8);
                    const position = t.klen.sub(qlen).add(row.div(c.group));
                    const key = key_base.add(key_part).add(i * 8).add(col0).add(j % 2);
                    var valid = row.lt(qlen.mul(c.group)).bitAnd(key.lt(t.klen));
                    if (c.causal) valid = valid.bitAnd(key.le(position));
                    if (c.window > 0) valid = valid.bitAnd(key.gt(position.sub(c.window)));
                    scores[i][j] = b.select(valid, scores[i][j].mul(c.scale * std.math.log2e), neg_inf);
                    local_max[j / 2] = local_max[j / 2].maximum(scores[i][j]);
                }
            }
            for (0..2) |h| t.publishReduction(row0.add(h * 8), local_max[h], true);
            b.syncThreads();
            // All K reads have completed. Reuse its allocation for V, and
            // overlap these global loads with softmax and probability stores.
            t.loadKV(t.v, key_base);
            var m: [2]V = undefined;
            var alpha: [2]V = undefined;
            var local_sum: [2]V = @splat(zero);
            for (0..2) |h| {
                m[h] = loop.carried[h].maximum(t.readReduction(w, row0.add(h * 8), true));
                // Empty/fully masked row tiles have m=-inf and denominator 0.
                alpha[h] = b.exp2Approx(loop.carried[h].sub(m[h].maximum(-std.math.floatMax(f32))));
            }
            for (0..nf) |i| {
                for (0..4) |j| {
                    scores[i][j] = b.exp2Approx(scores[i][j].sub(m[j / 2].maximum(-std.math.floatMax(f32))));
                    local_sum[j / 2] = local_sum[j / 2].add(scores[i][j]);
                }
                for (0..2) |h| {
                    const row = row_base.add(row0).add(h * 8);
                    const col = key_part.add(i * 8).add(col0);
                    b.storeWords(1, t.shared.add(c.pBase()).add(swizzle(row, col, n)), .shared, .{b.packHalf2(scores[i][h * 2], scores[i][h * 2 + 1], dt)});
                }
            }
            for (0..2) |h| t.publishReduction(row0.add(h * 8), local_sum[h], false);
            b.cpAsyncWaitAll();
            b.syncThreads();
            var next: [count]V = undefined;
            for (0..2) |h| {
                next[h] = m[h];
                next[h + 2] = loop.carried[h + 2].mul(alpha[h]).add(t.readReduction(w, row0.add(h * 8), false));
            }
            var acc: [of][4]V = undefined;
            for (0..of) |i| for (0..4) |j| {
                acc[i][j] = b.vectorExtract(loop.carried[4 + i], @intCast(j), .f32).mul(alpha[j / 2]);
            };
            var packed_acc: [of]V = undefined;
            for (0..of) |i| packed_acc[i] = packAcc(b, acc[i]);
            var pv = b.openFor(0, n, 16, packed_acc);
            {
                const pa = matrixLoad(b, 4, false, t.shared.add(c.pBase()).add(swizzle(row_base.add(t.lane.rem(16)), pv.iv.add(t.lane.div(16).mul(8)), n)));
                for (0..of) |i| {
                    const vb = matrixLoad(b, 2, true, t.shared.add(c.kvBase()).add(swizzle(pv.iv.add(t.lane.rem(16)), dim_part.add(i * 8), d)));
                    var prev: [4]V = undefined;
                    for (0..4) |j| prev[j] = b.vectorExtract(pv.carried[i], @intCast(j), .f32);
                    acc[i] = mma(b, dt, pa, vb, prev);
                    packed_acc[i] = packAcc(b, acc[i]);
                }
                pv.yield(packed_acc);
            }
            @memcpy(next[4..], &pv.results);
            b.syncThreads();
            loop.yield(next);
        }
        for (0..2) |h| {
            const row = first_row.add(row_base).add(row0).add(h * 8);
            var valid = b.openIf(row.lt(qlen.mul(c.group)));
            {
                const m = loop.results[h];
                const l = loop.results[h + 2];
                var norm = b.select(l.gt(0), b.rcpApprox(l), zero);
                if (c.splits == 1 and c.sink) {
                    const sink = loadI32(b, t.sink, t.head.mul(c.group).add(row.rem(c.group))).bitCast(.f32).mul(std.math.log2e);
                    const maximum = m.maximum(sink);
                    const weight = b.exp2Approx(m.sub(maximum));
                    norm = weight.div(l.mul(weight).add(b.exp2Approx(sink.sub(maximum))));
                }
                for (0..of) |i| for (0..2) |j| {
                    const value = b.vectorExtract(loop.results[4 + i], @intCast(h * 2 + j), .f32);
                    t.store(row, dim_part.add(i * 8).add(col0).add(j), if (c.splits > 1) value else value.mul(norm));
                };
                if (c.splits > 1) {
                    var leader = b.openIf(t.warp.rem(w).eq(0).bitAnd(t.lane.rem(4).eq(0)));
                    {
                        t.store(row, b.constant(.i32, d), m);
                        t.store(row, b.constant(.i32, d + 1), l);
                        leader.close();
                    }
                }
                valid.close();
            }
        }
        active.close();
    }
}

pub const Merge = cute.Kernel(Config, .{
    .name = "paged_attention_cute_sm120_merge",
    .inputs = &.{ "partials", "starts", "sink" },
    .outputs = &.{"out"},
    .run = merge,
});

fn merge(b: *B, c: Config) cute.FinishError!void {
    const a = try b.declareArgs(.{
        .partials = .{ .tensor = .{ .dtype = .f32, .shape = &.{ c.splits, c.tokens, c.kv_heads * c.group, c.head_dim + 2 } } },
        .starts = .{ .tensor = .{ .dtype = .i32, .shape = &.{c.batch + 1} } },
        .sink = .{ .tensor = .{ .dtype = .f32, .shape = &.{c.kv_heads * c.group} } },
        .out = .{ .tensor = .{ .dtype = c.dtype, .shape = &.{ c.tokens, c.kv_heads * c.group, c.head_dim } } },
    });
    const token = b.blockIdx().x;
    const head = b.blockIdx().y;
    var valid = b.openIf(token.lt(a.starts.get(.{c.batch})));
    {
        const zero = b.constant(.f32, 0);
        const sink = if (c.sink) a.sink.get(.{head}).mul(std.math.log2e) else b.constant(.f32, -std.math.inf(f32));
        var maximum = b.openFor(0, c.splits, 1, .{sink});
        {
            maximum.yield(.{maximum.carried[0].maximum(a.partials.get(.{ maximum.iv, token, head, c.head_dim }))});
        }
        const m = maximum.results[0].maximum(-std.math.floatMax(f32));
        var dims = b.openFor(b.threadIdx().x, c.head_dim, 128, .{});
        {
            var sum = b.openFor(0, c.splits, 1, .{ zero, if (c.sink) b.exp2Approx(sink.sub(m)) else zero });
            {
                const weight = b.exp2Approx(a.partials.get(.{ sum.iv, token, head, c.head_dim }).sub(m));
                sum.yield(.{
                    sum.carried[0].add(weight.mul(a.partials.get(.{ sum.iv, token, head, dims.iv }))),
                    sum.carried[1].add(weight.mul(a.partials.get(.{ sum.iv, token, head, c.head_dim + 1 }))),
                });
            }
            a.out.set(.{ token, head, dims.iv }, sum.results[0].mul(b.select(sum.results[1].gt(0), b.rcpApprox(sum.results[1]), zero)).to(c.dtype));
            dims.close();
        }
        valid.close();
    }
}
