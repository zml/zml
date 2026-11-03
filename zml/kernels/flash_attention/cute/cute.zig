const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const kernel = @import("attention.zig");

test {
    _ = kernel;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major >= 8;
}

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    stdx.debug.assert(q.shape().hasTags(.{ .q, .h, .hd }) and k.shape().hasTags(.{ .k, .h, .hd }) and v.shape().hasTags(.{ .k, .h, .hd }), "CuTe attention requires q/h/hd and k/h/hd tags", .{});
    const batched = q.shape().hasTag(.b) != null;
    const rank: usize = if (batched) 4 else 3;
    stdx.debug.assert(q.rank() == rank and k.rank() == rank and v.rank() == rank, "CuTe attention expects rank-3 tensors or rank-4 tensors with a batch axis", .{});
    stdx.debug.assert(q.dtype() == k.dtype() and q.dtype() == v.dtype(), "CuTe attention requires matching Q/K/V dtypes", .{});
    stdx.debug.assert(q.dtype() == .bf16 or q.dtype() == .f16 or q.dtype() == .f32, "CuTe attention supports bf16, f16, and f32", .{});
    stdx.debug.assert(q.dim(.hd) > 0 and q.dim(.hd) <= 512 and q.dim(.hd) == k.dim(.hd) and k.dim(.hd) == v.dim(.hd), "CuTe attention requires matching head dimensions in [1, 512]", .{});
    stdx.debug.assert(k.dim(.h) > 0 and q.dim(.h) > 0 and @rem(q.dim(.h), k.dim(.h)) == 0 and k.dim(.h) == v.dim(.h), "CuTe attention requires Hq divisible by Hkv and matching K/V heads", .{});
    stdx.debug.assert(q.dim(.q) > 0 and q.dim(.q) <= k.dim(.k) and k.dim(.k) == v.dim(.k), "CuTe attention requires 0 < Lq <= Lk and matching K/V lengths", .{});
    stdx.debug.assert(token_index.dtype().isInteger(), "CuTe attention requires integer token positions", .{});
    if (batched) {
        stdx.debug.assert(k.shape().hasTag(.b) != null and v.shape().hasTag(.b) != null and q.dim(.b) == k.dim(.b) and q.dim(.b) == v.dim(.b), "CuTe attention requires matching batch dimensions", .{});
        stdx.debug.assert(token_index.rank() == 0 or (token_index.rank() == 1 and token_index.shape().hasTag(.b) != null and token_index.dim(.b) == q.dim(.b)), "CuTe attention requires a scalar position or one position per batch element", .{});
    } else {
        stdx.debug.assert(token_index.rank() == 0, "Unbatched CuTe attention requires a scalar token position", .{});
    }

    const positions = if (batched)
        token_index.convert(.i32).broad(zml.Shape.init(.{ .b = q.dim(.b) }, .i32))
    else
        token_index.convert(.i32);
    const q_ = if (batched) q.transpose(.{ .b, .q, .h, .hd }) else q.transpose(.{ .q, .h, .hd });
    const k_ = if (batched) k.transpose(.{ .b, .k, .h, .hd }) else k.transpose(.{ .k, .h, .hd });
    const v_ = if (batched) v.transpose(.{ .b, .k, .h, .hd }) else v.transpose(.{ .k, .h, .hd });

    const splits = kernel.decodeSplits(q_.shape(), k_.shape(), decodeResources(zml.Compiler.current().platform));

    const cfg: kernel.Config = .{
        .batch = if (batched) q_.dim(.b) else 1,
        .query_length = q_.dim(.q),
        .key_length = k_.dim(.k),
        .query_heads = q_.dim(.h),
        .kv_heads = k_.dim(.h),
        .head_dim = q_.dim(.hd),
        .dtype = zml.kernel.cute.from(q_.dtype()),
        .splits = splits,
    };

    const out = if (cfg.splits > 1) b: {
        const partials = kernel.Kernel.call(.{ .q = q_, .k = k_, .v = v_, .positions = positions }, .{
            .out = zml.Shape.init(.{ cfg.batch, cfg.splits, cfg.query_length, cfg.query_heads, cfg.head_dim + 2 }, .f32),
        }, .{
            .cfg = cfg,
            .grid = .{ @intCast(@divTrunc(@divExact(cfg.query_heads, cfg.kv_heads) + 3, 4)), @intCast(cfg.kv_heads * cfg.batch), @intCast(cfg.splits) },
            .block = kernel.block,
        }).out;
        break :b kernel.Merge.call(.{ .partials = partials }, .{ .out = q_.shape() }, .{
            .cfg = cfg,
            .grid = .{ @intCast(cfg.query_heads), @intCast(cfg.batch), 1 },
            .block = kernel.block,
        }).out;
    } else kernel.Kernel.call(.{ .q = q_, .k = k_, .v = v_, .positions = positions }, .{ .out = q_.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(@divTrunc(cfg.query_length * @divExact(cfg.query_heads, cfg.kv_heads) + 3, 4)), @intCast(cfg.kv_heads * cfg.batch), 1 },
        .block = kernel.block,
    }).out;

    return out.transpose(q.shape());
}

/// Returns the minimum number of multiprocessors and shared memory per block across all devices in the platform.
/// If there are no devices, returns 0 for both values.
/// If any device does not have the required attributes, returns 0 for both values.
fn decodeResources(platform: *const zml.Platform) kernel.DecodeResources {
    var result: kernel.DecodeResources = .{ .multiprocessors = std.math.maxInt(i64), .shared_bytes = std.math.maxInt(i64) };
    if (platform.devices.len == 0) return .{ .multiprocessors = 0, .shared_bytes = 0 };
    for (platform.devices) |device| {
        const sms = device.pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return .{ .multiprocessors = 0, .shared_bytes = 0 };
        const shared = device.pjrt_desc.attribute(platform.pjrt_api, "shared_memory_per_block_optin") orelse return .{ .multiprocessors = 0, .shared_bytes = 0 };
        result.multiprocessors = @min(result.multiprocessors, sms.int64);
        result.shared_bytes = @min(result.shared_bytes, shared.int64);
    }
    return result;
}
