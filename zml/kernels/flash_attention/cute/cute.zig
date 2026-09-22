const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const kernel = @import("attention.zig");

test {
    _ = kernel;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    return switch (platform.capabilities) {
        .cuda => |cuda| switch ((cuda orelse return false).compute) {
            .sm80, .sm86, .sm87, .sm89, .sm90, .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => true,
            .sm70, .sm72, .sm75 => false,
        },
        else => false,
    };
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

    const mesh = zml.Compiler.current().resolveMesh(.{.model});
    const q_sharded = q.withPartitioning(mesh, .{ .h = .model });
    const k_sharded = k.withPartitioning(mesh, .{ .h = .model });
    const v_sharded = v.withPartitioning(mesh, .{ .h = .model });
    const positions = if (batched)
        token_index.convert(.i32).broad(zml.Shape.init(.{ .b = q.dim(.b) }, .i32))
    else
        token_index.convert(.i32);
    return zml.ops.manualComputation(Context.body, Context{ .q = q_sharded, .k = k_sharded, .v = v_sharded, .positions = positions }, q_sharded.shape());
}

const Context = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    positions: zml.Tensor,

    fn body(self: Context, output_shape: zml.Shape) zml.Tensor {
        const batched = self.q.shape().hasTag(.b) != null;
        const q = if (batched) self.q.transpose(.{ .b, .q, .h, .hd }) else self.q.transpose(.{ .q, .h, .hd });
        const k = if (batched) self.k.transpose(.{ .b, .k, .h, .hd }) else self.k.transpose(.{ .k, .h, .hd });
        const v = if (batched) self.v.transpose(.{ .b, .k, .h, .hd }) else self.v.transpose(.{ .k, .h, .hd });

        const splits = kernel.decodeSplits(q.shape(), k.shape(), decodeResources(zml.Compiler.current().platform));

        const cfg: kernel.Config = .{
            .batch = if (batched) q.dim(.b) else 1,
            .query_length = q.dim(.q),
            .key_length = k.dim(.k),
            .query_heads = q.dim(.h),
            .kv_heads = k.dim(.h),
            .head_dim = q.dim(.hd),
            .dtype = zml.kernel.cute.from(q.dtype()),
            .splits = splits,
        };

        const out = if (cfg.splits > 1) b: {
            const partials = kernel.Kernel.call(.{ .q = q, .k = k, .v = v, .positions = self.positions }, .{
                .out = zml.Shape.init(.{ cfg.batch, cfg.splits, cfg.query_length, cfg.query_heads, cfg.head_dim + 2 }, .f32),
            }, .{
                .cfg = cfg,
                .grid = .{ @intCast(@divTrunc(@divExact(cfg.query_heads, cfg.kv_heads) + 3, 4)), @intCast(cfg.kv_heads * cfg.batch), @intCast(cfg.splits) },
                .block = kernel.block,
            }).out;
            break :b kernel.Merge.call(.{ .partials = partials }, .{ .out = q.shape() }, .{
                .cfg = cfg,
                .grid = .{ @intCast(cfg.query_heads), @intCast(cfg.batch), 1 },
                .block = kernel.block,
            }).out;
        } else kernel.Kernel.call(.{ .q = q, .k = k, .v = v, .positions = self.positions }, .{ .out = q.shape() }, .{
            .cfg = cfg,
            .grid = .{ @intCast(@divTrunc(cfg.query_length * @divExact(cfg.query_heads, cfg.kv_heads) + 3, 4)), @intCast(cfg.kv_heads * cfg.batch), 1 },
            .block = kernel.block,
        }).out;

        return out.transpose(output_shape);
    }
};

fn decodeResources(platform: *const zml.Platform) kernel.DecodeResources {
    const cuda = platform.capabilities.cuda orelse return .{ .multiprocessors = 0, .shared_bytes = 0 };
    const hardware = cuda.hardware;
    return .{
        .multiprocessors = hardware.properties.multiprocessorCount orelse 0,
        .shared_bytes = hardware.limits.sharedMemoryPerBlockOptin,
    };
}
