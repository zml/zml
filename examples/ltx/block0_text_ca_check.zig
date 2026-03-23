/// M2 parity checker: block-0 video text cross-attention residual.
///
/// Two check modes depending on fixture contents:
///   1. Attn2-only (required):
///        block0_text_ca.attn2_x
///        block0_text_ca.context
///        block0_text_ca.attn2_out
///      optional: block0_text_ca.context_mask
///
///   2. Full residual (optional additional keys):
///        block0_text_ca.vx_in
///        block0_text_ca.text_ca_out
///        block0_text_ca.vx_out
///
/// Generate fixture with:
///   python examples/ltx/export_block0_text_ca_fixture.py <trace.pt> <fixture.safetensors>
///
/// Run:
///   bazel run //examples/ltx:block0_text_ca_check -- \
///       <stage2_checkpoint.safetensors> <fixture.safetensors> [token_limit]

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

comptime {
    @setEvalBranchQuota(20000);
}

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("M2 block0 video text cross-attn residual parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const stage2_checkpoint_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:block0_text_ca_check -- " ++
                "<stage2_checkpoint.safetensors> <fixture.safetensors> [token_limit]",
            .{},
        );
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:block0_text_ca_check -- " ++
                "<stage2_checkpoint.safetensors> <fixture.safetensors> [token_limit]",
            .{},
        );
        return error.InvalidArgs;
    };
    const token_limit: ?usize = if (it.next()) |v|
        std.fmt.parseInt(usize, v, 10) catch {
            std.log.err("Invalid token_limit: {s}", .{v});
            return error.InvalidArgs;
        }
    else
        null;

    var ckpt_registry: zml.safetensors.TensorRegistry =
        zml.safetensors.TensorRegistry.fromPath(allocator, io, stage2_checkpoint_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{stage2_checkpoint_path});
        return err;
    };
    defer ckpt_registry.deinit();

    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_registry);
    defer ckpt_store.deinit();

    var fix_registry: zml.safetensors.TensorRegistry =
        zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open fixture: {s}", .{fixture_path});
        return err;
    };
    defer fix_registry.deinit();

    var fix_store: zml.io.TensorStore = .fromRegistry(allocator, &fix_registry);
    defer fix_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const sharding = try zml.sharding.replicatedSharding(platform);

    var attn2_x_buf = loadFixtureBuf(allocator, io, platform, &fix_store, "block0_text_ca.attn2_x", sharding) catch |err| {
        std.log.err("Fixture missing key: block0_text_ca.attn2_x", .{});
        return err;
    };
    defer attn2_x_buf.deinit();

    var context_buf = loadFixtureBuf(allocator, io, platform, &fix_store, "block0_text_ca.context", sharding) catch |err| {
        std.log.err("Fixture missing key: block0_text_ca.context", .{});
        return err;
    };
    defer context_buf.deinit();

    var attn2_out_ref_buf = loadFixtureBuf(allocator, io, platform, &fix_store, "block0_text_ca.attn2_out", sharding) catch |err| {
        std.log.err("Fixture missing key: block0_text_ca.attn2_out", .{});
        return err;
    };
    defer attn2_out_ref_buf.deinit();

    const has_context_mask = fix_store.view().hasKey("block0_text_ca.context_mask");
    var context_mask_buf: ?zml.Buffer = null;
    if (has_context_mask) {
        context_mask_buf = try loadFixtureBuf(allocator, io, platform, &fix_store, "block0_text_ca.context_mask", sharding);
    }
    defer if (context_mask_buf) |*b| b.deinit();

    if (token_limit) |lim| {
        attn2_x_buf = try check_utils.sliceTokenPrefix(io, platform, attn2_x_buf, sharding, lim);
        context_buf = try check_utils.sliceTokenPrefix(io, platform, context_buf, sharding, lim);
        attn2_out_ref_buf = try check_utils.sliceTokenPrefix(io, platform, attn2_out_ref_buf, sharding, lim);

        if (context_mask_buf) |mask| {
            context_mask_buf = try sliceContextMaskPrefix(io, platform, mask, sharding, lim);
        }

        std.log.info("Using token_limit={d}; sliced fixture tensors", .{lim});
    }

    var attn2_params_shape = model.initBlock0AttentionParams(ckpt_store.view(), .attn2);

    std.log.info("Compiling block0 video text cross-attn graph...", .{});
    var attn2_exe = if (context_mask_buf) |mask|
        try platform.compileFn(
            allocator,
            io,
            model.forwardBlock0Attn2WithContextMask,
            .{
                zml.Tensor.fromShape(attn2_x_buf.shape()),
                zml.Tensor.fromShape(context_buf.shape()),
                zml.Tensor.fromShape(mask.shape()),
                attn2_params_shape,
            },
            .{ .shardings = &.{sharding} },
        )
    else
        try platform.compileFn(
            allocator,
            io,
            model.forwardBlock0Attn2,
            .{
                zml.Tensor.fromShape(attn2_x_buf.shape()),
                zml.Tensor.fromShape(context_buf.shape()),
                attn2_params_shape,
            },
            .{ .shardings = &.{sharding} },
        );
    defer attn2_exe.deinit();

    std.log.info("Compile completed", .{});

    std.log.info("Loading attn2 parameters from checkpoint...", .{});
    var attn2_params_buffers = try zml.io.load(model.Attention.Params, &attn2_params_shape, allocator, io, platform, .{
        .store = &ckpt_store,
        .shardings = &.{sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0AttentionBuffers(&attn2_params_buffers);

    std.log.info("Parameter load completed", .{});

    var attn2_args = try attn2_exe.args(allocator);
    defer attn2_args.deinit(allocator);

    var attn2_results = try attn2_exe.results(allocator);
    defer attn2_results.deinit(allocator);

    if (context_mask_buf) |mask| {
        attn2_args.set(.{ attn2_x_buf, context_buf, mask, attn2_params_buffers });
    } else {
        attn2_args.set(.{ attn2_x_buf, context_buf, attn2_params_buffers });
    }

    std.log.info("Executing block0 attn2 forward...", .{});
    attn2_exe.call(attn2_args, &attn2_results);
    std.log.info("Execution completed", .{});

    var attn2_out_computed = attn2_results.get(zml.Buffer);
    defer attn2_out_computed.deinit();

    try zml.testing.expectClose(io, attn2_out_computed, attn2_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M2 attn2 parity PASSED", .{});

    const has_residual_keys =
        fix_store.view().hasKey("block0_text_ca.vx_in") and
        fix_store.view().hasKey("block0_text_ca.text_ca_out") and
        fix_store.view().hasKey("block0_text_ca.vx_out");

    if (!has_residual_keys) {
        std.log.info(
            "Skipping residual test (keys block0_text_ca.vx_in/.text_ca_out/.vx_out absent).",
            .{},
        );
        std.log.info("M2 block0 video text cross-attn parity PASSED (attn2-only mode)", .{});
        return;
    }

    var vx_in_buf = try loadFixtureBuf(allocator, io, platform, &fix_store, "block0_text_ca.vx_in", sharding);
    defer vx_in_buf.deinit();

    var text_ca_out_buf = try loadFixtureBuf(allocator, io, platform, &fix_store, "block0_text_ca.text_ca_out", sharding);
    defer text_ca_out_buf.deinit();

    var vx_out_ref_buf = try loadFixtureBuf(allocator, io, platform, &fix_store, "block0_text_ca.vx_out", sharding);
    defer vx_out_ref_buf.deinit();

    if (token_limit) |lim| {
        vx_in_buf = try check_utils.sliceTokenPrefix(io, platform, vx_in_buf, sharding, lim);
        text_ca_out_buf = try check_utils.sliceTokenPrefix(io, platform, text_ca_out_buf, sharding, lim);
        vx_out_ref_buf = try check_utils.sliceTokenPrefix(io, platform, vx_out_ref_buf, sharding, lim);
    }

    std.log.info("Compiling block0 text cross-attn residual graph...", .{});
    var residual_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0VideoTextCaResidualFromDelta,
        .{
            zml.Tensor.fromShape(vx_in_buf.shape()),
            zml.Tensor.fromShape(text_ca_out_buf.shape()),
        },
        .{ .shardings = &.{sharding} },
    );
    defer residual_exe.deinit();

    var residual_args = try residual_exe.args(allocator);
    defer residual_args.deinit(allocator);

    var residual_results = try residual_exe.results(allocator);
    defer residual_results.deinit(allocator);

    residual_args.set(.{ vx_in_buf, text_ca_out_buf });
    std.log.info("Executing block0 text cross-attn residual...", .{});
    residual_exe.call(residual_args, &residual_results);

    var vx_out_computed = residual_results.get(zml.Buffer);
    defer vx_out_computed.deinit();

    try zml.testing.expectClose(io, vx_out_computed, vx_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });

    std.log.info("✓ M2 vx + text_ca_out residual parity PASSED", .{});
    std.log.info("M2 block0 video text cross-attn parity PASSED (full residual mode)", .{});
}

fn loadFixtureBuf(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    fix_store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    return check_utils.loadBufferFromStore(allocator, io, platform, fix_store, key, sharding);
}

/// Slice a context mask [B, 1, Q, K] (or [B, H, Q, K]) to token prefix on Q and K.
fn sliceContextMaskPrefix(
    io: std.Io,
    platform: *zml.Platform,
    src: zml.Buffer,
    sharding: zml.sharding.Sharding,
    token_limit: usize,
) !zml.Buffer {
    const shape = src.shape();
    if (shape.rank() != 4) return error.UnexpectedRank;

    const dims = shape.dims();
    const b: usize = @intCast(dims[0]);
    const h: usize = @intCast(dims[1]);
    const q: usize = @intCast(dims[2]);
    const k: usize = @intCast(dims[3]);

    const out_q = @min(token_limit, q);
    const out_k = @min(token_limit, k);
    const elem_size: usize = shape.dtype().sizeOf();

    const out_shape = zml.Shape.init(
        .{ dims[0], dims[1], @as(i64, @intCast(out_q)), @as(i64, @intCast(out_k)) },
        shape.dtype(),
    );

    var src_slice = try src.toSliceAlloc(std.heap.smp_allocator, io);
    defer src_slice.free(std.heap.smp_allocator);

    var out_slice = try zml.Slice.alloc(std.heap.smp_allocator, out_shape);
    defer out_slice.free(std.heap.smp_allocator);

    const src_bytes = src_slice.constData();
    const out_bytes = out_slice.data();

    const src_b_stride = h * q * k * elem_size;
    const src_h_stride = q * k * elem_size;
    const src_q_stride = k * elem_size;

    const dst_b_stride = h * out_q * out_k * elem_size;
    const dst_h_stride = out_q * out_k * elem_size;
    const dst_q_stride = out_k * elem_size;

    var bi: usize = 0;
    while (bi < b) : (bi += 1) {
        var hi: usize = 0;
        while (hi < h) : (hi += 1) {
            var qi: usize = 0;
            while (qi < out_q) : (qi += 1) {
                const src_off = bi * src_b_stride + hi * src_h_stride + qi * src_q_stride;
                const dst_off = bi * dst_b_stride + hi * dst_h_stride + qi * dst_q_stride;
                std.mem.copyForwards(
                    u8,
                    out_bytes[dst_off .. dst_off + out_k * elem_size],
                    src_bytes[src_off .. src_off + out_k * elem_size],
                );
            }
        }
    }

    return zml.Buffer.fromSlice(io, platform, out_slice, sharding);
}
