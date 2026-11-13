const std = @import("std");

const zml = @import("zml/core");
const nn = @import("zml/nn");
const flash_attn = @import("flash_attn.zig");

pub const Metadata = struct {
    /// Live queries of a single-sequence prefill; 0 means all.
    num_tokens: zml.Tensor,

    pub fn init() Metadata {
        return .{ .num_tokens = .fromShape(.scalar(.u32)) };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        _ = self;
        return .{ .num_tokens = try zml.Buffer.scalar(io, platform, 0, .u32) };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        self.num_tokens.deinit();
    }
};

/// Causal attention: fa_vec for bf16 decode, fa_tiled otherwise, sdpa for
/// unsupported dtypes and head sizes.
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata) zml.Tensor {
    const dtype = q.dtype();
    const head_dim = q.dim(.hd);
    if (!((dtype == .bf16 or dtype == .f16) and k.dtype() == dtype and v.dtype() == dtype and flash_attn.FaTiled.tileConfig(head_dim) != null)) {
        const seq_len = k.dim(.k);
        const mask = nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, dtype, null)
            .gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, dtype), token_index.appendAxes(.{.coord}), .{});
        return nn.sdpa(q, k, v, .{ .attn_mask = mask });
    }

    const batched = q.shape().hasTag(.b) != null;
    const qb = if (batched) q.transpose(.{ .b, .q, .h, .hd }) else q.transpose(.{ .q, .h, .hd }).insertAxes(0, .{.b});
    const kb = if (batched) k.transpose(.{ .b, .k, .h, .hd }) else k.transpose(.{ .k, .h, .hd }).insertAxes(0, .{.b});
    const vb = if (batched) v.transpose(.{ .b, .k, .h, .hd }) else v.transpose(.{ .k, .h, .hd }).insertAxes(0, .{.b});
    const positions = token_index.convert(.i32).broad(.init(.{ .b = qb.dim(.b) }, .i32));
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const vec_head_dim = head_dim == 32 or head_dim == 64 or head_dim == 128 or head_dim == 256 or head_dim == 512;
    const out = if (qb.dim(.q) == 1 and dtype == .bf16 and vec_head_dim)
        decode(qb, kb, vb, positions, scale)
    else
        prefill(qb, kb, vb, positions, liveQueries(qb, metadata), scale);
    return (if (batched) out else out.squeeze(.b)).transpose(q.shape());
}

fn liveQueries(q: zml.Tensor, metadata: Metadata) zml.Tensor {
    const all: zml.Tensor = .scalar(q.dim(.q), .i32);
    if (q.dim(.b) != 1 or q.dim(.q) == 1) return all;
    const num_tokens = metadata.num_tokens.convert(.i32);
    return num_tokens.cmp(.GE, .scalar(1, .i32)).select(num_tokens.minimum(all), all);
}

/// `q` is `[b, 1, h, hd]`, `k` and `v` `[b, k, kv_h, hd]`.
fn decode(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, positions: zml.Tensor, scale: f32) zml.Tensor {
    const batch: i32 = @intCast(q.dim(.b));
    const num_heads: i32 = @intCast(q.dim(.h));
    const head_dim: i32 = @intCast(q.dim(.hd));
    const num_kv_heads: i32 = @intCast(k.dim(.h));
    const keys: i32 = @intCast(k.dim(.k));
    const nrows = batch * num_heads;
    const launch = vecLaunch(nrows, @divExact(num_heads, num_kv_heads), keys, coreCount());
    const args: flash_attn.FaVec.HostArgs = .{ .num_heads = num_heads, .n_kv = num_kv_heads, .k_len = keys, .scale = scale };

    const dst = if (launch.splits == 1) q.shape() else zml.Shape.init(.{nrows * (head_dim * launch.splits + 2 * launch.splits)}, .f32);
    const out = flash_attn.FaVec.Kernel.call(.{ .q = q, .k = k, .v = v, .token_index = positions }, .{ .dst = dst }, .{
        .cfg = .{ .head_dim = head_dim, .heads = launch.heads, .splits = launch.splits },
        .grid = .{ @intCast(@divExact(launch.splits, launch.splits_per_threadgroup)), @intCast(@divExact(num_heads, launch.heads)), @intCast(batch) },
        .threadgroup = .{ 32, @intCast(launch.heads * launch.splits_per_threadgroup), 1 },
        .constants = .{ .args = .of(args) },
    }).dst;
    if (launch.splits == 1) return out;

    return flash_attn.FaVecReduce.Kernel.call(.{ .htmp = out, .token_index = positions }, .{ .dst = q.shape() }, .{
        .cfg = .{ .splits = launch.splits, .head_dim = head_dim },
        .grid = .{ @intCast(nrows), @intCast(@divExact(head_dim, 4 * flash_attn.FaVecReduce.simdgroups)), 1 },
        .threadgroup = .{ 32, flash_attn.FaVecReduce.simdgroups, 1 },
        .constants = .{ .args = .of(args) },
    }).dst;
}

/// `q` is `[b, q, h, hd]`, `k` and `v` `[b, k, kv_h, hd]`.
fn prefill(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, positions: zml.Tensor, live: zml.Tensor, scale: f32) zml.Tensor {
    const queries: i32 = @intCast(q.dim(.q));
    const tile = flash_attn.FaTiled.tileConfig(q.dim(.hd)).?;
    const cfg: flash_attn.FaTiled.Cfg = .{
        .dtype = zml.kernel.metal.from(q.dtype()),
        .head_size = @intCast(q.dim(.hd)),
        .bq = tile[0],
        .tile_kv = tile[1],
        .num_threads = tile[2],
    };
    return flash_attn.FaTiled.Kernel.call(.{ .q = q, .k = k, .v = v, .token_index = positions, .live = live }, .{ .out = q.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(q.dim(.h)), @intCast(@divFloor(queries + tile[0] - 1, tile[0])), @intCast(q.dim(.b)) },
        .threadgroup = .{ @intCast(tile[2]), 1, 1 },
        .threadgroup_memory = flash_attn.FaTiled.threadgroupMemory(cfg),
        .constants = .{
            .num_kv_heads = .of(@as(i32, @intCast(k.dim(.h)))),
            .scale = .of(scale),
            .q_len = .of(queries),
            .k_len = .of(@as(i32, @intCast(k.dim(.k)))),
        },
    }).out;
}

const VecLaunch = struct {
    heads: i32,
    splits: i32,
    splits_per_threadgroup: i32,
};

/// At most 256 threads, which every Apple GPU pipeline accepts. Splits past a
/// row's length exit early, so `splits` follows the core and row counts.
fn vecLaunch(rows: i32, group: i32, span: i32, cores: i32) VecLaunch {
    var heads: i32 = @min(group, 4);
    while (@rem(group, heads) != 0) heads -= 1;
    const want: i32 = if (rows <= 2 * cores) 128 else 64;
    const chunks = @divFloor(span + 31, 32);
    var splits: i32 = 1;
    while (splits * 2 <= @min(want, chunks)) splits *= 2;
    return .{ .heads = heads, .splits = splits, .splits_per_threadgroup = @min(splits, @divFloor(8, heads)) };
}

fn coreCount() i32 {
    const platform = zml.Compiler.current().platform;
    const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse
        std.debug.panic("Metal device has no core_count attribute", .{});
    return @intCast(attribute.int64);
}

test vecLaunch {
    try std.testing.expectEqual(VecLaunch{ .heads = 4, .splits = 128, .splits_per_threadgroup = 2 }, vecLaunch(16, 4, 128000, 40));
    try std.testing.expectEqual(VecLaunch{ .heads = 1, .splits = 1, .splits_per_threadgroup = 1 }, vecLaunch(8, 1, 20, 40));
}

test {
    std.testing.refAllDecls(flash_attn);
}
