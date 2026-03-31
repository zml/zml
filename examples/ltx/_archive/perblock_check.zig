/// Per-block parity checker: compares vx/ax after each sampled block
/// against Python reference fixture from export_perblock_fixture.py.
///
/// Usage:
///   bazel run //examples/ltx:perblock_check -- \
///       /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
///       /root/repos/LTX-2/trace_run/step2_fixture_step_000_t512.safetensors \
///       /root/repos/LTX-2/trace_run/perblock_fixture_step_000_t512.safetensors

const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

comptime {
    @setEvalBranchQuota(200000);
}

pub const std_options: std.Options = .{ .log_level = .info };

/// Block indices for which we have reference fixtures.
/// Dense sampling 23-47 to pinpoint error accumulation in later blocks.
const sampled_blocks = [_]usize{ 0, 1, 2, 3, 7, 15, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("Per-block parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const ckpt_path = it.next() orelse {
        std.log.err("Usage: perblock_check <checkpoint.safetensors> <step2_fixture.safetensors> <perblock_fixture.safetensors>", .{});
        return error.InvalidArgs;
    };
    const step2_path = it.next() orelse {
        std.log.err("Usage: perblock_check <checkpoint.safetensors> <step2_fixture.safetensors> <perblock_fixture.safetensors>", .{});
        return error.InvalidArgs;
    };
    const perblock_path = it.next() orelse {
        std.log.err("Usage: perblock_check <checkpoint.safetensors> <step2_fixture.safetensors> <perblock_fixture.safetensors>", .{});
        return error.InvalidArgs;
    };

    // Open checkpoint
    var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
        std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
        return err;
    };
    defer ckpt_reg.deinit();
    var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
    defer ckpt_store.deinit();

    // Open step2 fixture (for raw inputs + preprocessing)
    var step2_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, step2_path) catch |err| {
        std.log.err("Failed to open step2 fixture: {s}", .{step2_path});
        return err;
    };
    defer step2_reg.deinit();
    var step2_store: zml.io.TensorStore = .fromRegistry(allocator, &step2_reg);
    defer step2_store.deinit();

    // Open perblock fixture
    var pb_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, perblock_path) catch |err| {
        std.log.err("Failed to open perblock fixture: {s}", .{perblock_path});
        return err;
    };
    defer pb_reg.deinit();
    var pb_store: zml.io.TensorStore = .fromRegistry(allocator, &pb_reg);
    defer pb_store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // ========================================================================
    // Load raw fixture inputs (same as step2_check)
    // ========================================================================
    std.log.info("Loading raw fixture inputs...", .{});
    var v_latent_buf = try loadBuf(allocator, io, platform, &step2_store, "raw.video_latent", sharding);
    defer v_latent_buf.deinit();
    var a_latent_buf = try loadBuf(allocator, io, platform, &step2_store, "raw.audio_latent", sharding);
    defer a_latent_buf.deinit();
    var v_mask_buf = try loadBuf(allocator, io, platform, &step2_store, "raw.video_denoise_mask", sharding);
    defer v_mask_buf.deinit();
    var a_mask_buf = try loadBuf(allocator, io, platform, &step2_store, "raw.audio_denoise_mask", sharding);
    defer a_mask_buf.deinit();
    var sigma_buf = try loadBuf(allocator, io, platform, &step2_store, "raw.sigma", sharding);
    defer sigma_buf.deinit();
    var v_positions_buf = try loadBuf(allocator, io, platform, &step2_store, "raw.video_positions", sharding);
    defer v_positions_buf.deinit();
    var a_positions_buf = try loadBuf(allocator, io, platform, &step2_store, "raw.audio_positions", sharding);
    defer a_positions_buf.deinit();
    var v_context_buf = try loadBuf(allocator, io, platform, &step2_store, "raw.v_context", sharding);
    defer v_context_buf.deinit();
    var a_context_buf = try loadBuf(allocator, io, platform, &step2_store, "raw.a_context", sharding);
    defer a_context_buf.deinit();

    // ========================================================================
    // Run preprocessing (identical to step2_check)
    // ========================================================================
    std.log.info("Initializing preprocessing params...", .{});
    const preprocess_shape = model.initPreprocessParams(ckpt_store.view());

    std.log.info("Compiling preprocessing exe...", .{});
    const sigma_shape_1d = zml.Shape.init(.{ .b = 1 }, .f32);

    var preprocess_exe = try platform.compileFn(
        allocator, io,
        model.forwardPreprocess,
        .{
            zml.Tensor.fromShape(v_latent_buf.shape()),
            zml.Tensor.fromShape(a_latent_buf.shape()),
            zml.Tensor.fromShape(v_mask_buf.shape()),
            zml.Tensor.fromShape(a_mask_buf.shape()),
            zml.Tensor.fromShape(sigma_shape_1d),
            zml.Tensor.fromShape(sigma_shape_1d),
            zml.Tensor.fromShape(v_positions_buf.shape()),
            zml.Tensor.fromShape(a_positions_buf.shape()),
            zml.Tensor.fromShape(v_context_buf.shape()),
            zml.Tensor.fromShape(a_context_buf.shape()),
            preprocess_shape,
        },
        .{ .shardings = &.{sharding} },
    );
    defer preprocess_exe.deinit();

    std.log.info("Loading preprocessing weights...", .{});
    var preprocess_bufs = try zml.io.load(
        model.PreprocessParams,
        &preprocess_shape,
        allocator, io, platform,
        .{
            .store = &ckpt_store,
            .shardings = &.{sharding},
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        },
    );
    defer model.PreprocessParams.unloadBuffers(&preprocess_bufs);

    std.log.info("Running preprocessing...", .{});
    var sigma_1d_buf = try reshapeScalarTo1d(io, platform, sigma_buf, sharding);
    defer sigma_1d_buf.deinit();

    var pre_args = try preprocess_exe.args(allocator);
    defer pre_args.deinit(allocator);
    var pre_results = try preprocess_exe.results(allocator);
    defer pre_results.deinit(allocator);

    pre_args.set(.{
        v_latent_buf,
        a_latent_buf,
        v_mask_buf,
        a_mask_buf,
        sigma_1d_buf,
        sigma_1d_buf,
        v_positions_buf,
        a_positions_buf,
        v_context_buf,
        a_context_buf,
        preprocess_bufs,
    });
    preprocess_exe.call(pre_args, &pre_results);

    const pre_out = pre_results.get(zml.Bufferized(model.PreprocessOutput));
    std.log.info("Preprocessing complete.", .{});
    std.log.info("  vx: {any}  ax: {any}", .{ pre_out.vx.shape(), pre_out.ax.shape() });

    // ========================================================================
    // Validate block input (pre-chain) against perblock fixture
    // ========================================================================
    std.log.info("Comparing block inputs (pre-chain)...", .{});
    try comparePerblock(io, allocator, platform, &pb_store, sharding, "block_input", pre_out.vx, pre_out.ax);

    // ========================================================================
    // Compile + load block chain
    // ========================================================================
    std.log.info("Initializing block params...", .{});
    var block_params_shape = try allocator.create(model.FullStepParams);
    defer allocator.destroy(block_params_shape);
    block_params_shape.* = model.initFullStepParams(ckpt_store.view());

    std.log.info("Compiling single-block exe...", .{});
    var block_exe = try platform.compileFn(
        allocator, io,
        model.forwardBlock0Native,
        .{
            zml.Tensor.fromShape(pre_out.vx.shape()),
            zml.Tensor.fromShape(pre_out.ax.shape()),
            zml.Tensor.fromShape(pre_out.video_timesteps.shape()),
            zml.Tensor.fromShape(pre_out.audio_timesteps.shape()),
            zml.Tensor.fromShape(pre_out.v_prompt_timestep.shape()),
            zml.Tensor.fromShape(pre_out.a_prompt_timestep.shape()),
            zml.Tensor.fromShape(pre_out.v_pe_cos.shape()),
            zml.Tensor.fromShape(pre_out.v_pe_sin.shape()),
            zml.Tensor.fromShape(pre_out.a_pe_cos.shape()),
            zml.Tensor.fromShape(pre_out.a_pe_sin.shape()),
            zml.Tensor.fromShape(pre_out.v_text_ctx.shape()),
            zml.Tensor.fromShape(pre_out.a_text_ctx.shape()),
            zml.Tensor.fromShape(pre_out.v_cross_ss_ts.shape()),
            zml.Tensor.fromShape(pre_out.v_cross_gate_ts.shape()),
            zml.Tensor.fromShape(pre_out.a_cross_ss_ts.shape()),
            zml.Tensor.fromShape(pre_out.a_cross_gate_ts.shape()),
            zml.Tensor.fromShape(pre_out.a2v_pe_cos.shape()),
            zml.Tensor.fromShape(pre_out.a2v_pe_sin.shape()),
            zml.Tensor.fromShape(pre_out.a2v_k_pe_cos.shape()),
            zml.Tensor.fromShape(pre_out.a2v_k_pe_sin.shape()),
            zml.Tensor.fromShape(pre_out.v2a_pe_cos.shape()),
            zml.Tensor.fromShape(pre_out.v2a_pe_sin.shape()),
            zml.Tensor.fromShape(pre_out.v2a_k_pe_cos.shape()),
            zml.Tensor.fromShape(pre_out.v2a_k_pe_sin.shape()),
            block_params_shape.blocks[0],
        },
        .{ .shardings = &.{sharding} },
    );
    defer block_exe.deinit();
    std.log.info("Single-block exe compiled.", .{});

    std.log.info("Loading 48 block weights...", .{});
    var block_params_bufs = try allocator.create([48]zml.Bufferized(model.Block0FullParams));
    defer allocator.destroy(block_params_bufs);
    for (0..48) |i| {
        block_params_bufs[i] = try zml.io.load(
            model.Block0FullParams,
            &block_params_shape.blocks[i],
            allocator, io, platform,
            .{
                .store = &ckpt_store,
                .shardings = &.{sharding},
                .parallelism = 4,
                .dma_chunks = 4,
                .dma_chunk_size = 16 * zml.MiB,
            },
        );
    }
    defer for (&block_params_bufs.*) |*bp| model.unloadBlock0FullBuffers(bp);
    std.log.info("Block weights loaded.", .{});

    // ========================================================================
    // Run 48-block chain with per-block comparison
    // ========================================================================
    std.log.info("Running 48-block chain with per-block comparison...", .{});
    std.log.info("", .{});
    std.log.info("block | vx cos_sim  vx close   vx max_abs | ax cos_sim  ax close   ax max_abs", .{});
    std.log.info("------+-----------+----------+------------+-----------+----------+-----------", .{});

    var h_v = pre_out.vx;
    var h_a = pre_out.ax;

    for (0..48) |i| {
        var args = try block_exe.args(allocator);
        defer args.deinit(allocator);
        var results = try block_exe.results(allocator);
        defer results.deinit(allocator);

        args.set(.{
            h_v, h_a,
            pre_out.video_timesteps, pre_out.audio_timesteps,
            pre_out.v_prompt_timestep, pre_out.a_prompt_timestep,
            pre_out.v_pe_cos, pre_out.v_pe_sin,
            pre_out.a_pe_cos, pre_out.a_pe_sin,
            pre_out.v_text_ctx, pre_out.a_text_ctx,
            pre_out.v_cross_ss_ts, pre_out.v_cross_gate_ts,
            pre_out.a_cross_ss_ts, pre_out.a_cross_gate_ts,
            pre_out.a2v_pe_cos, pre_out.a2v_pe_sin,
            pre_out.a2v_k_pe_cos, pre_out.a2v_k_pe_sin,
            pre_out.v2a_pe_cos, pre_out.v2a_pe_sin,
            pre_out.v2a_k_pe_cos, pre_out.v2a_k_pe_sin,
            block_params_bufs[i],
        });
        block_exe.call(args, &results);

        const out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));

        if (i > 0) {
            h_v.deinit();
            h_a.deinit();
        }
        h_v = out.vx_out;
        h_a = out.ax_out;

        // Check if this block is in our sampled set
        if (isSampledBlock(i)) {
            var key_buf: [64]u8 = undefined;
            const block_key = std.fmt.bufPrint(&key_buf, "block_{d:0>3}", .{i}) catch unreachable;
            try comparePerblock(io, allocator, platform, &pb_store, sharding, block_key, h_v, h_a);
        }
    }

    std.log.info("", .{});
    std.log.info("48-block chain (uncorrected) complete.", .{});

    // Clean up block outputs
    h_v.deinit();
    h_a.deinit();

    // ========================================================================
    // Pass 2: Corrected chain — reset to Python reference at each checkpoint
    //
    // This tells us the per-segment error when starting from correct inputs.
    // Segments: [input→0], [0→1], [1→2], [2→3], [3→7], [7→15], [15→23],
    //           [23→31], [31→39], [39→47]
    // ========================================================================
    std.log.info("", .{});
    std.log.info("=== Pass 2: Corrected chain (reset at each checkpoint) ===", .{});
    std.log.info("segment      | vx cos_sim  vx close   vx max_abs | ax cos_sim  ax close   ax max_abs", .{});
    std.log.info("-------------+-----------+----------+------------+-----------+----------+-----------", .{});

    // Load block_input as the initial reference
    var corr_v = try loadBuf(allocator, io, platform, &pb_store, "block_input.vx", sharding);
    var corr_a = try loadBuf(allocator, io, platform, &pb_store, "block_input.ax", sharding);

    for (0..sampled_blocks.len) |si| {
        const end_block = sampled_blocks[si];
        // Determine start block for this segment
        const start_block: usize = if (si == 0) 0 else sampled_blocks[si - 1] + 1;

        // Run blocks [start_block .. end_block] inclusive
        for (start_block..end_block + 1) |i| {
            var args = try block_exe.args(allocator);
            defer args.deinit(allocator);
            var results = try block_exe.results(allocator);
            defer results.deinit(allocator);

            args.set(.{
                corr_v, corr_a,
                pre_out.video_timesteps, pre_out.audio_timesteps,
                pre_out.v_prompt_timestep, pre_out.a_prompt_timestep,
                pre_out.v_pe_cos, pre_out.v_pe_sin,
                pre_out.a_pe_cos, pre_out.a_pe_sin,
                pre_out.v_text_ctx, pre_out.a_text_ctx,
                pre_out.v_cross_ss_ts, pre_out.v_cross_gate_ts,
                pre_out.a_cross_ss_ts, pre_out.a_cross_gate_ts,
                pre_out.a2v_pe_cos, pre_out.a2v_pe_sin,
                pre_out.a2v_k_pe_cos, pre_out.a2v_k_pe_sin,
                pre_out.v2a_pe_cos, pre_out.v2a_pe_sin,
                pre_out.v2a_k_pe_cos, pre_out.v2a_k_pe_sin,
                block_params_bufs[i],
            });
            block_exe.call(args, &results);

            const out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
            corr_v.deinit();
            corr_a.deinit();
            corr_v = out.vx_out;
            corr_a = out.ax_out;
        }

        // Compare against reference at this checkpoint
        var key_buf: [64]u8 = undefined;
        const block_key = std.fmt.bufPrint(&key_buf, "block_{d:0>3}", .{end_block}) catch unreachable;

        // Format segment label
        var seg_buf: [32]u8 = undefined;
        const seg_label = if (si == 0)
            std.fmt.bufPrint(&seg_buf, "input->{d}", .{end_block}) catch unreachable
        else
            std.fmt.bufPrint(&seg_buf, "{d}->{d}", .{ sampled_blocks[si - 1], end_block }) catch unreachable;

        try comparePerblockLabeled(io, allocator, platform, &pb_store, sharding, block_key, seg_label, corr_v, corr_a);

        // Reset to Python reference for next segment
        corr_v.deinit();
        corr_a.deinit();
        var vx_load_buf: [64]u8 = undefined;
        var ax_load_buf: [64]u8 = undefined;
        corr_v = try loadBuf(allocator, io, platform, &pb_store, std.fmt.bufPrint(&vx_load_buf, "block_{d:0>3}.vx", .{end_block}) catch unreachable, sharding);
        corr_a = try loadBuf(allocator, io, platform, &pb_store, std.fmt.bufPrint(&ax_load_buf, "block_{d:0>3}.ax", .{end_block}) catch unreachable, sharding);
    }

    corr_v.deinit();
    corr_a.deinit();

    std.log.info("", .{});
    std.log.info("Corrected chain complete.", .{});
}

// ============================================================================
// Helpers
// ============================================================================

fn isSampledBlock(idx: usize) bool {
    for (sampled_blocks) |s| {
        if (s == idx) return true;
    }
    return false;
}

fn comparePerblock(
    io: std.Io,
    allocator: std.mem.Allocator,
    platform: *zml.Platform,
    pb_store: *zml.io.TensorStore,
    sharding: zml.sharding.Sharding,
    block_key: []const u8,
    actual_vx: zml.Buffer,
    actual_ax: zml.Buffer,
) !void {
    return comparePerblockLabeled(io, allocator, platform, pb_store, sharding, block_key, block_key, actual_vx, actual_ax);
}

fn comparePerblockLabeled(
    io: std.Io,
    allocator: std.mem.Allocator,
    platform: *zml.Platform,
    pb_store: *zml.io.TensorStore,
    sharding: zml.sharding.Sharding,
    block_key: []const u8,
    label: []const u8,
    actual_vx: zml.Buffer,
    actual_ax: zml.Buffer,
) !void {
    // Load reference vx
    var vx_key_buf: [80]u8 = undefined;
    const vx_key = std.fmt.bufPrint(&vx_key_buf, "{s}.vx", .{block_key}) catch unreachable;
    var ref_vx = loadBuf(allocator, io, platform, pb_store, vx_key, sharding) catch |err| {
        std.log.warn("  {s}: vx ref missing ({any})", .{ block_key, err });
        return;
    };
    defer ref_vx.deinit();

    // Load reference ax
    var ax_key_buf: [80]u8 = undefined;
    const ax_key = std.fmt.bufPrint(&ax_key_buf, "{s}.ax", .{block_key}) catch unreachable;
    var ref_ax = loadBuf(allocator, io, platform, pb_store, ax_key, sharding) catch |err| {
        std.log.warn("  {s}: ax ref missing ({any})", .{ block_key, err });
        return;
    };
    defer ref_ax.deinit();

    // Compare vx
    const vm = check_utils.compareBuffersExtended(io, actual_vx, ref_vx, 0.25, 0.02) catch |err| {
        std.log.err("  {s} vx: comparison error: {}", .{ block_key, err });
        return;
    };

    // Compare ax
    const am = check_utils.compareBuffersExtended(io, actual_ax, ref_ax, 0.25, 0.02) catch |err| {
        std.log.err("  {s} ax: comparison error: {}", .{ block_key, err });
        return;
    };

    std.log.info("{s:>13} | {d:.6}  {d:.4}  {d:>10.5} | {d:.6}  {d:.4}  {d:>10.5}", .{
        label,
        vm.cosine_similarity,
        vm.close_fraction,
        vm.max_abs_error,
        am.cosine_similarity,
        am.close_fraction,
        am.max_abs_error,
    });
}

fn loadBuf(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    return check_utils.loadBufferFromStore(allocator, io, platform, store, key, sharding) catch |err| {
        std.log.err("Missing fixture key: {s}", .{key});
        return err;
    };
}

fn reshapeScalarTo1d(
    io: std.Io,
    platform: *zml.Platform,
    scalar_buf: zml.Buffer,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const val = try scalar_buf.getValue(f32, io);
    const new_shape = zml.Shape.init(.{ .b = 1 }, .f32);
    return zml.Buffer.fromBytes(io, platform, new_shape, sharding, std.mem.asBytes(&val));
}
