/// M5-A parity checker: block-0 AV cross-attn A->V branch.
///
/// Verifies:
///   1) audio_to_video_attn(x, context, pe, k_pe) == attn_out
///   2) attn_out * gate * mask == delta

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

fn loadBuf(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    return check_utils.loadBufferFromStore(allocator, io, platform, store, key, sharding) catch |err| {
        std.log.err("Fixture missing key: {s}", .{key});
        return err;
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("M5-A block0 AV A->V parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: block0_av_a2v_check <checkpoint.safetensors> <fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err("Usage: block0_av_a2v_check <checkpoint.safetensors> <fixture.safetensors> [token_limit]", .{});
        return error.InvalidArgs;
    };
    const token_limit: ?usize = if (it.next()) |v|
        std.fmt.parseInt(usize, v, 10) catch {
            std.log.err("Invalid token_limit: {s}", .{v});
            return error.InvalidArgs;
        }
    else
        null;

    var ckpt_reg: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    var fix_reg: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open fixture: {s}", .{fixture_path});
        return err;
    };
    defer fix_reg.deinit();
    var fix_store: zml.io.TensorStore = .fromRegistry(allocator, &fix_reg);
    defer fix_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    var x_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.x", sharding);
    defer x_buf.deinit();
    var context_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.context", sharding);
    defer context_buf.deinit();
    var pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.pe_cos", sharding);
    defer pe_cos_buf.deinit();
    var pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.pe_sin", sharding);
    defer pe_sin_buf.deinit();
    var k_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.k_pe_cos", sharding);
    defer k_pe_cos_buf.deinit();
    var k_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.k_pe_sin", sharding);
    defer k_pe_sin_buf.deinit();
    var attn_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.attn_out", sharding);
    defer attn_ref_buf.deinit();
    var gate_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.gate", sharding);
    defer gate_buf.deinit();
    var mask_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.mask", sharding);
    defer mask_buf.deinit();
    var delta_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block0_av_a2v.delta", sharding);
    defer delta_ref_buf.deinit();

    if (token_limit) |lim| {
        x_buf = try check_utils.sliceTokenPrefix(io, platform, x_buf, sharding, lim);
        context_buf = try check_utils.sliceTokenPrefix(io, platform, context_buf, sharding, lim);
        pe_cos_buf = try check_utils.sliceTokenPrefixBHTD(io, platform, pe_cos_buf, sharding, lim);
        pe_sin_buf = try check_utils.sliceTokenPrefixBHTD(io, platform, pe_sin_buf, sharding, lim);
        k_pe_cos_buf = try check_utils.sliceTokenPrefixBHTD(io, platform, k_pe_cos_buf, sharding, lim);
        k_pe_sin_buf = try check_utils.sliceTokenPrefixBHTD(io, platform, k_pe_sin_buf, sharding, lim);
        attn_ref_buf = try check_utils.sliceTokenPrefix(io, platform, attn_ref_buf, sharding, lim);
        gate_buf = try check_utils.sliceTokenPrefix(io, platform, gate_buf, sharding, lim);
        mask_buf = try check_utils.sliceTokenPrefix(io, platform, mask_buf, sharding, lim);
        delta_ref_buf = try check_utils.sliceTokenPrefix(io, platform, delta_ref_buf, sharding, lim);
        std.log.info("Using token_limit={d}", .{lim});
    }

    var attn_shape = model.initBlock0AttentionParams(ckpt_store.view(), .audio_to_video_attn);

    std.log.info("Compiling A->V attention graph...", .{});
    var attn_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0AudioToVideoAttnWithContextPeKPe,
        .{
            zml.Tensor.fromShape(x_buf.shape()),
            zml.Tensor.fromShape(context_buf.shape()),
            zml.Tensor.fromShape(pe_cos_buf.shape()),
            zml.Tensor.fromShape(pe_sin_buf.shape()),
            zml.Tensor.fromShape(k_pe_cos_buf.shape()),
            zml.Tensor.fromShape(k_pe_sin_buf.shape()),
            attn_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer attn_exe.deinit();

    var attn_params = try zml.io.load(model.Attention.Params, &attn_shape, allocator, io, platform, .{
        .store = &ckpt_store,
        .shardings = &.{sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0AttentionBuffers(&attn_params);

    var attn_args = try attn_exe.args(allocator);
    defer attn_args.deinit(allocator);
    var attn_res = try attn_exe.results(allocator);
    defer attn_res.deinit(allocator);

    attn_args.set(.{ x_buf, context_buf, pe_cos_buf, pe_sin_buf, k_pe_cos_buf, k_pe_sin_buf, attn_params });
    std.log.info("Executing A->V attention...", .{});
    attn_exe.call(attn_args, &attn_res);

    var attn_computed = attn_res.get(zml.Buffer);
    defer attn_computed.deinit();

    try zml.testing.expectClose(io, attn_computed, attn_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M5-A audio_to_video_attn parity PASSED", .{});

    std.log.info("Compiling A->V gated delta graph...", .{});
    var delta_exe = try platform.compileFn(
        allocator,
        io,
        model.forwardBlock0A2VDeltaFromAttnOut,
        .{
            zml.Tensor.fromShape(attn_ref_buf.shape()),
            zml.Tensor.fromShape(gate_buf.shape()),
            zml.Tensor.fromShape(mask_buf.shape()),
        },
        .{ .shardings = &.{sharding} },
    );
    defer delta_exe.deinit();

    var delta_args = try delta_exe.args(allocator);
    defer delta_args.deinit(allocator);
    var delta_res = try delta_exe.results(allocator);
    defer delta_res.deinit(allocator);

    delta_args.set(.{ attn_ref_buf, gate_buf, mask_buf });
    std.log.info("Executing A->V gated delta algebra...", .{});
    delta_exe.call(delta_args, &delta_res);

    var delta_computed = delta_res.get(zml.Buffer);
    defer delta_computed.deinit();

    try zml.testing.expectClose(io, delta_computed, delta_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("✓ M5-A gated delta parity PASSED", .{});
    std.log.info("M5-A block0 AV A->V parity PASSED", .{});
}
