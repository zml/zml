const std = @import("std");
const zml = @import("zml");
const model = @import("model.zig");
const check_utils = @import("check_utils.zig");

comptime {
    // The PJRT call wrappers introspect large generated type names at comptime.
    // Raise the default quota so this checker can compile with attention param structs.
    @setEvalBranchQuota(20000);
}

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Mode = enum {
    attn1,
    attn2,
    audio_attn1,
    audio_attn2,
    audio_to_video_attn,
    video_to_audio_attn,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var it = init.minimal.args.iterate();
    _ = it.next(); // exe

    const stage2_checkpoint_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:attention_forward_check -- <stage2_checkpoint.safetensors> <attention_fixture.safetensors> <mode> [token_limit] [diagnostic_reference.safetensors] [--token-limited-reference]",
            .{},
        );
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err(
            "Usage: bazel run //examples/ltx:attention_forward_check -- <stage2_checkpoint.safetensors> <attention_fixture.safetensors> <mode> [token_limit] [diagnostic_reference.safetensors] [--token-limited-reference]",
            .{},
        );
        return error.InvalidArgs;
    };
    const mode_txt = it.next() orelse {
        std.log.err("Missing mode. Expected one of: attn1, attn2, audio_attn1, audio_attn2, audio_to_video_attn, video_to_audio_attn", .{});
        return error.InvalidArgs;
    };

    const mode = try parseMode(mode_txt);

    var token_limit: ?usize = null;
    var diagnostic_ref_path: ?[]const u8 = null;
    var token_limited_reference = false;

    while (it.next()) |v| {
        if (std.mem.eql(u8, v, "--token-limited-reference")) {
            token_limited_reference = true;
            continue;
        }

        if (token_limit == null) {
            const parsed_limit = std.fmt.parseInt(usize, v, 10) catch null;
            if (parsed_limit) |limit| {
                token_limit = limit;
                continue;
            }
        }

        if (diagnostic_ref_path == null) {
            diagnostic_ref_path = v;
            continue;
        }

        std.log.err("Too many arguments", .{});
        return error.InvalidArgs;
    }

    if (token_limited_reference and token_limit == null) {
        std.log.warn("Ignoring --token-limited-reference because no token_limit was provided", .{});
        token_limited_reference = false;
    }

    const references_are_token_comparable = token_limit == null or token_limited_reference;

    var stage2_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, stage2_checkpoint_path) catch |err| {
        std.log.err("Failed to open stage-2 checkpoint: {s}", .{stage2_checkpoint_path});
        return err;
    };
    defer stage2_registry.deinit();

    var stage2_store: zml.io.TensorStore = .fromRegistry(allocator, &stage2_registry);
    defer stage2_store.deinit();

    var fixture_registry: zml.safetensors.TensorRegistry = zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
        std.log.err("Failed to open attention fixture: {s}", .{fixture_path});
        return err;
    };
    defer fixture_registry.deinit();

    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();

    // Attempt to load optional diagnostic reference
    var diagnostic_registry_opt: ?zml.safetensors.TensorRegistry = null;
    if (diagnostic_ref_path) |ref_path| {
        if (zml.safetensors.TensorRegistry.fromPath(allocator, io, ref_path)) |reg| {
            diagnostic_registry_opt = reg;
        } else |_| {
            std.log.warn("Failed to open diagnostic reference: {s}, proceeding without diagnostics", .{ref_path});
        }
    }
    defer if (diagnostic_registry_opt) |*reg| reg.deinit();

    var diagnostic_store: ?zml.io.TensorStore = null;
    if (diagnostic_registry_opt) |*reg| {
        diagnostic_store = .fromRegistry(allocator, reg);
    }
    defer if (diagnostic_store) |*store| store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    const input_key = try std.fmt.allocPrint(allocator, "{s}.input0", .{@tagName(mode)});
    defer allocator.free(input_key);
    const output_key = try std.fmt.allocPrint(allocator, "{s}.output0", .{@tagName(mode)});
    defer allocator.free(output_key);
    const pe_cos_key = try std.fmt.allocPrint(allocator, "{s}.pe_cos0", .{@tagName(mode)});
    defer allocator.free(pe_cos_key);
    const pe_sin_key = try std.fmt.allocPrint(allocator, "{s}.pe_sin0", .{@tagName(mode)});
    defer allocator.free(pe_sin_key);
    const k_pe_cos_key = try std.fmt.allocPrint(allocator, "{s}.k_pe_cos0", .{@tagName(mode)});
    defer allocator.free(k_pe_cos_key);
    const k_pe_sin_key = try std.fmt.allocPrint(allocator, "{s}.k_pe_sin0", .{@tagName(mode)});
    defer allocator.free(k_pe_sin_key);
    const mask_key = try std.fmt.allocPrint(allocator, "{s}.mask0", .{@tagName(mode)});
    defer allocator.free(mask_key);
    const context_key = try std.fmt.allocPrint(allocator, "{s}.context0", .{@tagName(mode)});
    defer allocator.free(context_key);

    var attn_input = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, input_key, replicated_sharding);
    defer attn_input.deinit();

    var attn_expected = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, output_key, replicated_sharding);
    defer attn_expected.deinit();

    var attn_pe_cos = try loadOptionalFixtureBuffer(allocator, io, platform, &fixture_store, pe_cos_key, replicated_sharding);
    defer if (attn_pe_cos) |*b| b.deinit();
    var attn_pe_sin = try loadOptionalFixtureBuffer(allocator, io, platform, &fixture_store, pe_sin_key, replicated_sharding);
    defer if (attn_pe_sin) |*b| b.deinit();
    var attn_k_pe_cos = try loadOptionalFixtureBuffer(allocator, io, platform, &fixture_store, k_pe_cos_key, replicated_sharding);
    defer if (attn_k_pe_cos) |*b| b.deinit();
    var attn_k_pe_sin = try loadOptionalFixtureBuffer(allocator, io, platform, &fixture_store, k_pe_sin_key, replicated_sharding);
    defer if (attn_k_pe_sin) |*b| b.deinit();
    var attn_mask = try loadOptionalFixtureBuffer(allocator, io, platform, &fixture_store, mask_key, replicated_sharding);
    defer if (attn_mask) |*b| b.deinit();
    var attn_context = try loadOptionalFixtureBuffer(allocator, io, platform, &fixture_store, context_key, replicated_sharding);
    defer if (attn_context) |*b| b.deinit();

    // Load diagnostic reference if provided
    var diagnostic_ref = check_utils.DiagnosticReference{};
    if (diagnostic_store) |*store| {
        diagnostic_ref = try check_utils.loadDiagnosticReference(allocator, io, platform, store, @tagName(mode), replicated_sharding);
        if (diagnostic_ref.q_head_split != null or diagnostic_ref.q_rotated != null) {
            std.log.info("Diagnostic reference loaded successfully", .{});
        }
    }
    defer diagnostic_ref.deinit();

    if (token_limit) |limit| {
        attn_input = try check_utils.sliceTokenPrefix(io, platform, attn_input, replicated_sharding, limit);
        attn_expected = try check_utils.sliceTokenPrefix(io, platform, attn_expected, replicated_sharding, limit);
        if (attn_pe_cos) |pe_cos| {
            attn_pe_cos = try check_utils.sliceTokenPrefixBHTD(io, platform, pe_cos, replicated_sharding, limit);
        }
        if (attn_pe_sin) |pe_sin| {
            attn_pe_sin = try check_utils.sliceTokenPrefixBHTD(io, platform, pe_sin, replicated_sharding, limit);
        }
        // Slice diagnostic reference tensors too
        if (diagnostic_ref.q_head_split) |q_hs| {
            diagnostic_ref.q_head_split = try check_utils.sliceTokenPrefixBTHDorBHTD(io, platform, q_hs, replicated_sharding, limit);
        }
        if (diagnostic_ref.k_head_split) |k_hs| {
            diagnostic_ref.k_head_split = try check_utils.sliceTokenPrefixBTHDorBHTD(io, platform, k_hs, replicated_sharding, limit);
        }
        if (diagnostic_ref.v_head_split) |v_hs| {
            diagnostic_ref.v_head_split = try check_utils.sliceTokenPrefixBTHDorBHTD(io, platform, v_hs, replicated_sharding, limit);
        }
        if (diagnostic_ref.q_rotated) |q_rot| {
            diagnostic_ref.q_rotated = try check_utils.sliceTokenPrefixBTHDorBHTD(io, platform, q_rot, replicated_sharding, limit);
        }
        if (diagnostic_ref.k_rotated) |k_rot| {
            diagnostic_ref.k_rotated = try check_utils.sliceTokenPrefixBTHDorBHTD(io, platform, k_rot, replicated_sharding, limit);
        }
        if (token_limited_reference) {
            std.log.info("Using token_limit={d}; sliced fixture and diagnostic tensors; token-limited references enabled", .{limit});
        } else {
            std.log.info("Using token_limit={d}; sliced fixture and diagnostic tensors", .{limit});
        }
    }

    const kind = modeToKind(mode);
    var attn_params_shape = model.initBlock0AttentionParams(stage2_store.view(), kind);

    const input_tensor = zml.Tensor.fromShape(attn_input.shape());
    std.log.info("Compiling attention graph for mode={s}...", .{@tagName(mode)});
    if (mode == .attn1) {
        std.log.info("attn1 kwargs presence: pe_cos={any} pe_sin={any} mask={any}", .{ attn_pe_cos != null, attn_pe_sin != null, attn_mask != null });
    }

    if (mode == .attn2 and attn_context == null) {
        std.log.err("Fixture missing required tensor: {s}.context0 for mode=attn2", .{@tagName(mode)});
        return error.InvalidArgs;
    }

    var exe = switch (mode) {
        .attn1 => blk: {
            if (attn_pe_cos) |pe_cos| {
                if (attn_pe_sin == null) {
                    std.log.err("Fixture has {s}.pe_cos0 but missing {s}.pe_sin0", .{ @tagName(mode), @tagName(mode) });
                    return error.InvalidArgs;
                }
                const pe_cos_tensor = zml.Tensor.fromShape(pe_cos.shape());
                const pe_sin_tensor = zml.Tensor.fromShape(attn_pe_sin.?.shape());
                if (attn_mask) |mask| {
                    std.log.info("Compiling masked attn1 path (forwardBlock0Attn1WithPeCosSinMask)", .{});
                    const mask_tensor = zml.Tensor.fromShape(mask.shape());
                    break :blk try platform.compileFn(allocator, io, model.forwardBlock0Attn1WithPeCosSinMask, .{ input_tensor, pe_cos_tensor, pe_sin_tensor, mask_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                }
                std.log.info("Compiling unmasked attn1 path (forwardBlock0Attn1WithPeCosSin)", .{});
                break :blk try platform.compileFn(allocator, io, model.forwardBlock0Attn1WithPeCosSin, .{ input_tensor, pe_cos_tensor, pe_sin_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
            }
            std.log.info("Compiling no-rope attn1 path (forwardBlock0Attn1)", .{});
            break :blk try platform.compileFn(allocator, io, model.forwardBlock0Attn1, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
        },
        .attn2 => blk: {
            const context_tensor = zml.Tensor.fromShape(attn_context.?.shape());
            break :blk try platform.compileFn(allocator, io, model.forwardBlock0Attn2, .{ input_tensor, context_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
        },
        .audio_attn1 => blk: {
            if (attn_pe_cos) |pe_cos| {
                if (attn_pe_sin == null) {
                    std.log.err("Fixture has {s}.pe_cos0 but missing {s}.pe_sin0", .{ @tagName(mode), @tagName(mode) });
                    return error.InvalidArgs;
                }
                const pe_cos_tensor = zml.Tensor.fromShape(pe_cos.shape());
                const pe_sin_tensor = zml.Tensor.fromShape(attn_pe_sin.?.shape());
                break :blk try platform.compileFn(allocator, io, model.forwardBlock0AudioAttn1WithPeCosSin, .{ input_tensor, pe_cos_tensor, pe_sin_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
            }
            break :blk try platform.compileFn(allocator, io, model.forwardBlock0AudioAttn1, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
        },
        .audio_attn2 => blk: {
            if (attn_context == null) {
                std.log.err("Fixture missing required tensor: {s}.context0 for mode={s}", .{ @tagName(mode), @tagName(mode) });
                return error.InvalidArgs;
            }
            const context_tensor = zml.Tensor.fromShape(attn_context.?.shape());
            break :blk try platform.compileFn(allocator, io, model.forwardBlock0AudioAttn2WithContext, .{ input_tensor, context_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
        },
        .audio_to_video_attn => blk: {
            if (attn_context == null or attn_pe_cos == null or attn_pe_sin == null or attn_k_pe_cos == null or attn_k_pe_sin == null) {
                std.log.err("Fixture missing required tensors for mode={s}: need context0, pe_cos0, pe_sin0, k_pe_cos0, k_pe_sin0", .{@tagName(mode)});
                return error.InvalidArgs;
            }
            const context_tensor = zml.Tensor.fromShape(attn_context.?.shape());
            const pe_cos_tensor = zml.Tensor.fromShape(attn_pe_cos.?.shape());
            const pe_sin_tensor = zml.Tensor.fromShape(attn_pe_sin.?.shape());
            const k_pe_cos_tensor = zml.Tensor.fromShape(attn_k_pe_cos.?.shape());
            const k_pe_sin_tensor = zml.Tensor.fromShape(attn_k_pe_sin.?.shape());
            break :blk try platform.compileFn(allocator, io, model.forwardBlock0AudioToVideoAttnWithContextPeKPe, .{ input_tensor, context_tensor, pe_cos_tensor, pe_sin_tensor, k_pe_cos_tensor, k_pe_sin_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
        },
        .video_to_audio_attn => blk: {
            if (attn_context == null or attn_pe_cos == null or attn_pe_sin == null or attn_k_pe_cos == null or attn_k_pe_sin == null) {
                std.log.err("Fixture missing required tensors for mode={s}: need context0, pe_cos0, pe_sin0, k_pe_cos0, k_pe_sin0", .{@tagName(mode)});
                return error.InvalidArgs;
            }
            const context_tensor = zml.Tensor.fromShape(attn_context.?.shape());
            const pe_cos_tensor = zml.Tensor.fromShape(attn_pe_cos.?.shape());
            const pe_sin_tensor = zml.Tensor.fromShape(attn_pe_sin.?.shape());
            const k_pe_cos_tensor = zml.Tensor.fromShape(attn_k_pe_cos.?.shape());
            const k_pe_sin_tensor = zml.Tensor.fromShape(attn_k_pe_sin.?.shape());
            break :blk try platform.compileFn(allocator, io, model.forwardBlock0VideoToAudioAttnWithContextPeKPe, .{ input_tensor, context_tensor, pe_cos_tensor, pe_sin_tensor, k_pe_cos_tensor, k_pe_sin_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
        },
    };
    defer exe.deinit();
    std.log.info("Compile completed", .{});

    std.log.info("Loading attention parameters from checkpoint...", .{});
    var attn_params_buffers = try zml.io.load(model.Attention.Params, &attn_params_shape, allocator, io, platform, .{
        .store = &stage2_store,
        .shardings = &.{replicated_sharding},
        .parallelism = 16,
        .dma_chunks = 8,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer model.unloadBlock0AttentionBuffers(&attn_params_buffers);
    std.log.info("Parameter load completed", .{});

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    switch (mode) {
        .attn1 => {
            if (attn_pe_cos) |pe_cos| {
                const pe_sin = attn_pe_sin.?;
                if (attn_mask) |mask| {
                    args.set(.{ attn_input, pe_cos, pe_sin, mask, attn_params_buffers });
                } else {
                    args.set(.{ attn_input, pe_cos, pe_sin, attn_params_buffers });
                }
            } else {
                args.set(.{ attn_input, attn_params_buffers });
            }
        },
        .attn2 => args.set(.{ attn_input, attn_context.?, attn_params_buffers }),
        .audio_attn1 => {
            if (attn_pe_cos) |pe_cos| {
                args.set(.{ attn_input, pe_cos, attn_pe_sin.?, attn_params_buffers });
            } else {
                args.set(.{ attn_input, attn_params_buffers });
            }
        },
        .audio_attn2 => args.set(.{ attn_input, attn_context.?, attn_params_buffers }),
        .audio_to_video_attn => args.set(.{ attn_input, attn_context.?, attn_pe_cos.?, attn_pe_sin.?, attn_k_pe_cos.?, attn_k_pe_sin.?, attn_params_buffers }),
        .video_to_audio_attn => args.set(.{ attn_input, attn_context.?, attn_pe_cos.?, attn_pe_sin.?, attn_k_pe_cos.?, attn_k_pe_sin.?, attn_params_buffers }),
    }
    std.log.info("Executing attention forward for mode={s}...", .{@tagName(mode)});
    exe.call(args, &results);
    std.log.info("Execution completed", .{});

    var attn_output = results.get(zml.Buffer);
    defer attn_output.deinit();

    // Stage-by-stage diagnostic comparison if reference is available
    // Stage-by-stage diagnostic comparison (attn1 with pe_cos/pe_sin only)
    if (mode == .attn1) {
        if (attn_pe_cos) |pe_cos_buf| {
            const pe_sin_buf = attn_pe_sin.?;
            const pe_cos_t = zml.Tensor.fromShape(pe_cos_buf.shape());
            const pe_sin_t = zml.Tensor.fromShape(pe_sin_buf.shape());

            std.log.info("=== DIAGNOSTIC STAGE-BY-STAGE COMPARISON ===", .{});

            // --- Stage 1: q_norm — validate to_q projection + RMSNorm ---
            const q_norm_key = try std.fmt.allocPrint(allocator, "{s}.q_norm_diag0", .{@tagName(mode)});
            defer allocator.free(q_norm_key);

            if (fixture_store.view().hasKey(q_norm_key)) {
                var q_norm_ref = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, q_norm_key, replicated_sharding);
                defer q_norm_ref.deinit();

                if (token_limit) |limit| {
                    q_norm_ref = try check_utils.sliceTokenPrefix(io, platform, q_norm_ref, replicated_sharding, limit);
                }

                std.log.info("Compiling forwardBlock0Attn1DiagQNorm...", .{});
                var diag_qnorm_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagQNorm, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer diag_qnorm_exe.deinit();

                var diag_qnorm_args = try diag_qnorm_exe.args(allocator);
                defer diag_qnorm_args.deinit(allocator);
                var diag_qnorm_results = try diag_qnorm_exe.results(allocator);
                defer diag_qnorm_results.deinit(allocator);

                diag_qnorm_args.set(.{ attn_input, attn_params_buffers });
                diag_qnorm_exe.call(diag_qnorm_args, &diag_qnorm_results);

                var qnorm_output = diag_qnorm_results.get(zml.Buffer);
                defer qnorm_output.deinit();

                const qnorm_metrics = try check_utils.compareBuffers(io, qnorm_output, q_norm_ref, 0.1, 0.01);
                _ = check_utils.reportStageMetrics(.q_norm, qnorm_metrics, 0.99);
            } else {
                std.log.info("Skipping stage 1: {s} not found in fixture", .{q_norm_key});
            }

            // --- Stage 2: q_rotated — validate head-split + RoPE for Q ---
            if (diagnostic_ref.q_rotated) |q_rotated_ref| {
                std.log.info("Compiling forwardBlock0Attn1DiagQRot...", .{});
                var diag_qrot_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagQRot, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer diag_qrot_exe.deinit();

                var diag_qrot_args = try diag_qrot_exe.args(allocator);
                defer diag_qrot_args.deinit(allocator);
                var diag_qrot_results = try diag_qrot_exe.results(allocator);
                defer diag_qrot_results.deinit(allocator);

                diag_qrot_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                diag_qrot_exe.call(diag_qrot_args, &diag_qrot_results);

                var qrot_output = diag_qrot_results.get(zml.Buffer);
                defer qrot_output.deinit();

                const qrot_metrics: ?check_utils.CompareMetrics = check_utils.compareBuffersBTHDCompatible(io, platform, qrot_output, q_rotated_ref, replicated_sharding, 0.1, 0.01) catch |err| switch (err) {
                    error.ShapeMismatch => blk: {
                        std.log.warn("Skipping q_rotated diagnostic due to incompatible shapes: computed={f} expected={f}", .{ qrot_output.shape(), q_rotated_ref.shape() });
                        break :blk null;
                    },
                    else => return err,
                };
                if (qrot_metrics) |metrics| {
                    _ = check_utils.reportStageMetrics(.q_rotated, metrics, 0.99);
                }
            } else {
                std.log.info("Skipping stage 2: q_rotated not found in diagnostic reference", .{});
            }

            // --- Stage 3: k_rotated — validate head-split + RoPE for K ---
            if (diagnostic_ref.k_rotated) |k_rotated_ref| {
                std.log.info("Compiling forwardBlock0Attn1DiagKRot...", .{});
                var diag_krot_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagKRot, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer diag_krot_exe.deinit();

                var diag_krot_args = try diag_krot_exe.args(allocator);
                defer diag_krot_args.deinit(allocator);
                var diag_krot_results = try diag_krot_exe.results(allocator);
                defer diag_krot_results.deinit(allocator);

                diag_krot_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                diag_krot_exe.call(diag_krot_args, &diag_krot_results);

                var krot_output = diag_krot_results.get(zml.Buffer);
                defer krot_output.deinit();

                const krot_metrics: ?check_utils.CompareMetrics = check_utils.compareBuffersBTHDCompatible(io, platform, krot_output, k_rotated_ref, replicated_sharding, 0.1, 0.01) catch |err| switch (err) {
                    error.ShapeMismatch => blk: {
                        std.log.warn("Skipping k_rotated diagnostic due to incompatible shapes: computed={f} expected={f}", .{ krot_output.shape(), k_rotated_ref.shape() });
                        break :blk null;
                    },
                    else => return err,
                };
                if (krot_metrics) |metrics| {
                    _ = check_utils.reportStageMetrics(.k_rotated, metrics, 0.99);
                }
            } else {
                std.log.info("Skipping stage 3: k_rotated not found in diagnostic reference", .{});
            }

            // --- Stage 3.5: to_v projection (pre head-split) ---
            const to_v_key = try std.fmt.allocPrint(allocator, "{s}.to_v_diag0", .{@tagName(mode)});
            defer allocator.free(to_v_key);
            if (fixture_store.view().hasKey(to_v_key)) {
                var to_v_ref = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, to_v_key, replicated_sharding);
                defer to_v_ref.deinit();
                if (token_limit) |limit| {
                    to_v_ref = try check_utils.sliceTokenPrefix(io, platform, to_v_ref, replicated_sharding, limit);
                }

                std.log.info("Compiling forwardBlock0Attn1DiagVProj...", .{});
                var diag_vproj_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagVProj, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer diag_vproj_exe.deinit();

                var diag_vproj_args = try diag_vproj_exe.args(allocator);
                defer diag_vproj_args.deinit(allocator);
                var diag_vproj_results = try diag_vproj_exe.results(allocator);
                defer diag_vproj_results.deinit(allocator);

                diag_vproj_args.set(.{ attn_input, attn_params_buffers });
                diag_vproj_exe.call(diag_vproj_args, &diag_vproj_results);

                var vproj_output = diag_vproj_results.get(zml.Buffer);
                defer vproj_output.deinit();

                const vproj_metrics = try check_utils.compareBuffers(io, vproj_output, to_v_ref, 0.1, 0.01);
                std.log.info(
                    "to_v pre-split stage: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}",
                    .{ vproj_metrics.max_abs_error, vproj_metrics.mean_abs_error, vproj_metrics.close_fraction },
                );
            } else {
                std.log.info("Skipping stage 3.5: {s} not found in fixture", .{to_v_key});
            }

            // --- Stage 4: v_head_split — validate to_v projection + head-split (no RoPE) ---
            if (token_limit != null and fixture_store.view().hasKey(to_v_key)) {
                var to_v_ref_for_split = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, to_v_key, replicated_sharding);
                defer to_v_ref_for_split.deinit();
                if (token_limit) |limit| {
                    to_v_ref_for_split = try check_utils.sliceTokenPrefix(io, platform, to_v_ref_for_split, replicated_sharding, limit);
                }

                std.log.info("Compiling forwardBlock0Attn1DiagVHead...", .{});
                var diag_vhead_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagVHead, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer diag_vhead_exe.deinit();

                var diag_vhead_args = try diag_vhead_exe.args(allocator);
                defer diag_vhead_args.deinit(allocator);
                var diag_vhead_results = try diag_vhead_exe.results(allocator);
                defer diag_vhead_results.deinit(allocator);

                diag_vhead_args.set(.{ attn_input, attn_params_buffers });
                diag_vhead_exe.call(diag_vhead_args, &diag_vhead_results);

                var vhead_output = diag_vhead_results.get(zml.Buffer);
                defer vhead_output.deinit();

                const to_v_ref_tensor = zml.Tensor.fromShape(to_v_ref_for_split.shape());
                var split_ref_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagVHeadFromProj, .{to_v_ref_tensor}, .{ .shardings = &.{replicated_sharding} });
                defer split_ref_exe.deinit();

                var split_ref_args = try split_ref_exe.args(allocator);
                defer split_ref_args.deinit(allocator);
                var split_ref_results = try split_ref_exe.results(allocator);
                defer split_ref_results.deinit(allocator);

                split_ref_args.set(.{to_v_ref_for_split});
                split_ref_exe.call(split_ref_args, &split_ref_results);

                var vhead_ref_from_vproj = split_ref_results.get(zml.Buffer);
                defer vhead_ref_from_vproj.deinit();

                const vhead_metrics = try check_utils.compareBuffers(io, vhead_output, vhead_ref_from_vproj, 0.1, 0.01);
                _ = check_utils.reportStageMetrics(.v_head_split, vhead_metrics, 0.99);
            } else if (diagnostic_ref.v_head_split) |v_head_ref| {
                std.log.info("Compiling forwardBlock0Attn1DiagVHead...", .{});
                var diag_vhead_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagVHead, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer diag_vhead_exe.deinit();

                var diag_vhead_args = try diag_vhead_exe.args(allocator);
                defer diag_vhead_args.deinit(allocator);
                var diag_vhead_results = try diag_vhead_exe.results(allocator);
                defer diag_vhead_results.deinit(allocator);

                diag_vhead_args.set(.{ attn_input, attn_params_buffers });
                diag_vhead_exe.call(diag_vhead_args, &diag_vhead_results);

                var vhead_output = diag_vhead_results.get(zml.Buffer);
                defer vhead_output.deinit();

                const vhead_metrics: ?check_utils.CompareMetrics = check_utils.compareBuffersBTHDCompatible(io, platform, vhead_output, v_head_ref, replicated_sharding, 0.1, 0.01) catch |err| switch (err) {
                    error.ShapeMismatch => blk: {
                        std.log.warn("Skipping v_head_split diagnostic due to incompatible shapes: computed={f} expected={f}", .{ vhead_output.shape(), v_head_ref.shape() });
                        break :blk null;
                    },
                    else => return err,
                };
                if (vhead_metrics) |metrics| {
                    _ = check_utils.reportStageMetrics(.v_head_split, metrics, 0.99);
                }
            } else {
                std.log.info("Skipping stage 4: v_head_split not found in diagnostic reference", .{});
            }

            // --- Stage 4b: direct SDPA q/k/v captures from replay ---
            // These come from monkeypatched torch SDPA call and are the strongest ground truth.
            if (references_are_token_comparable) {
                const sdpa_q_key = try std.fmt.allocPrint(allocator, "{s}.sdpa_q_diag0", .{@tagName(mode)});
                defer allocator.free(sdpa_q_key);
                if (fixture_store.view().hasKey(sdpa_q_key)) {
                    var sdpa_q_ref = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, sdpa_q_key, replicated_sharding);
                    defer sdpa_q_ref.deinit();

                    std.log.info("Compiling sdpa_q direct compare (forwardBlock0Attn1DiagQRot)...", .{});
                    var sdpa_q_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagQRot, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                    defer sdpa_q_exe.deinit();
                    var sdpa_q_args = try sdpa_q_exe.args(allocator);
                    defer sdpa_q_args.deinit(allocator);
                    var sdpa_q_results = try sdpa_q_exe.results(allocator);
                    defer sdpa_q_results.deinit(allocator);
                    sdpa_q_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                    sdpa_q_exe.call(sdpa_q_args, &sdpa_q_results);
                    var sdpa_q_out = sdpa_q_results.get(zml.Buffer);
                    defer sdpa_q_out.deinit();
                    const m = try check_utils.compareBuffers(io, sdpa_q_out, sdpa_q_ref, 0.1, 0.01);
                    std.log.info("sdpa_q direct: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}", .{ m.max_abs_error, m.mean_abs_error, m.close_fraction });
                }

                const sdpa_k_key = try std.fmt.allocPrint(allocator, "{s}.sdpa_k_diag0", .{@tagName(mode)});
                defer allocator.free(sdpa_k_key);
                if (fixture_store.view().hasKey(sdpa_k_key)) {
                    var sdpa_k_ref = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, sdpa_k_key, replicated_sharding);
                    defer sdpa_k_ref.deinit();

                    std.log.info("Compiling sdpa_k direct compare (forwardBlock0Attn1DiagKRot)...", .{});
                    var sdpa_k_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagKRot, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                    defer sdpa_k_exe.deinit();
                    var sdpa_k_args = try sdpa_k_exe.args(allocator);
                    defer sdpa_k_args.deinit(allocator);
                    var sdpa_k_results = try sdpa_k_exe.results(allocator);
                    defer sdpa_k_results.deinit(allocator);
                    sdpa_k_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                    sdpa_k_exe.call(sdpa_k_args, &sdpa_k_results);
                    var sdpa_k_out = sdpa_k_results.get(zml.Buffer);
                    defer sdpa_k_out.deinit();
                    const m = try check_utils.compareBuffers(io, sdpa_k_out, sdpa_k_ref, 0.1, 0.01);
                    std.log.info("sdpa_k direct: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}", .{ m.max_abs_error, m.mean_abs_error, m.close_fraction });
                }

                const sdpa_v_key = try std.fmt.allocPrint(allocator, "{s}.sdpa_v_diag0", .{@tagName(mode)});
                defer allocator.free(sdpa_v_key);
                if (fixture_store.view().hasKey(sdpa_v_key)) {
                    var sdpa_v_ref = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, sdpa_v_key, replicated_sharding);
                    defer sdpa_v_ref.deinit();

                    std.log.info("Compiling sdpa_v direct compare (forwardBlock0Attn1DiagVHead)...", .{});
                    var sdpa_v_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagVHead, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                    defer sdpa_v_exe.deinit();
                    var sdpa_v_args = try sdpa_v_exe.args(allocator);
                    defer sdpa_v_args.deinit(allocator);
                    var sdpa_v_results = try sdpa_v_exe.results(allocator);
                    defer sdpa_v_results.deinit(allocator);
                    sdpa_v_args.set(.{ attn_input, attn_params_buffers });
                    sdpa_v_exe.call(sdpa_v_args, &sdpa_v_results);
                    var sdpa_v_out = sdpa_v_results.get(zml.Buffer);
                    defer sdpa_v_out.deinit();
                    const m = try check_utils.compareBuffers(io, sdpa_v_out, sdpa_v_ref, 0.1, 0.01);
                    std.log.info("sdpa_v direct: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}", .{ m.max_abs_error, m.mean_abs_error, m.close_fraction });
                }

                const sdpa_out_key = try std.fmt.allocPrint(allocator, "{s}.sdpa_out_diag0", .{@tagName(mode)});
                defer allocator.free(sdpa_out_key);
                if (fixture_store.view().hasKey(sdpa_out_key)) {
                    var sdpa_out_ref = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, sdpa_out_key, replicated_sharding);
                    defer sdpa_out_ref.deinit();

                    std.log.info("Compiling sdpa_out direct compare (forwardBlock0Attn1DiagSdpaOut)...", .{});
                    var sdpa_out_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagSdpaOut, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                    defer sdpa_out_exe.deinit();
                    var sdpa_out_args = try sdpa_out_exe.args(allocator);
                    defer sdpa_out_args.deinit(allocator);
                    var sdpa_out_results = try sdpa_out_exe.results(allocator);
                    defer sdpa_out_results.deinit(allocator);
                    sdpa_out_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                    sdpa_out_exe.call(sdpa_out_args, &sdpa_out_results);
                    var sdpa_out_buf = sdpa_out_results.get(zml.Buffer);
                    defer sdpa_out_buf.deinit();
                    const m = try check_utils.compareBuffers(io, sdpa_out_buf, sdpa_out_ref, 0.1, 0.01);
                    std.log.info("sdpa_out direct: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}", .{ m.max_abs_error, m.mean_abs_error, m.close_fraction });

                    std.log.info("Compiling sdpa_out manual f32 compare (forwardBlock0Attn1DiagSdpaOutManualF32)...", .{});
                    var sdpa_out_manual_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagSdpaOutManualF32, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                    defer sdpa_out_manual_exe.deinit();
                    var sdpa_out_manual_args = try sdpa_out_manual_exe.args(allocator);
                    defer sdpa_out_manual_args.deinit(allocator);
                    var sdpa_out_manual_results = try sdpa_out_manual_exe.results(allocator);
                    defer sdpa_out_manual_results.deinit(allocator);
                    sdpa_out_manual_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                    sdpa_out_manual_exe.call(sdpa_out_manual_args, &sdpa_out_manual_results);
                    var sdpa_out_manual_buf = sdpa_out_manual_results.get(zml.Buffer);
                    defer sdpa_out_manual_buf.deinit();
                    const mm = try check_utils.compareBuffers(io, sdpa_out_manual_buf, sdpa_out_ref, 0.1, 0.01);
                    std.log.info("sdpa_out_manual_f32 direct: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}", .{ mm.max_abs_error, mm.mean_abs_error, mm.close_fraction });
                }
            } else {
                std.log.info("Skipping context-dependent SDPA direct stages for token-limited run (references were captured with full context)", .{});
            }

            // --- Stage 5: gate_logits — validate to_gate_logits projection ---
            const gate_key = try std.fmt.allocPrint(allocator, "{s}.to_gate_logits_diag0", .{@tagName(mode)});
            defer allocator.free(gate_key);
            if (fixture_store.view().hasKey(gate_key)) {
                var gate_ref = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, gate_key, replicated_sharding);
                defer gate_ref.deinit();
                if (token_limit) |limit| {
                    gate_ref = try check_utils.sliceTokenPrefix(io, platform, gate_ref, replicated_sharding, limit);
                }

                std.log.info("Compiling forwardBlock0Attn1DiagGateLogits...", .{});
                var gate_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagGateLogits, .{ input_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer gate_exe.deinit();

                var gate_args = try gate_exe.args(allocator);
                defer gate_args.deinit(allocator);
                var gate_results = try gate_exe.results(allocator);
                defer gate_results.deinit(allocator);

                gate_args.set(.{ attn_input, attn_params_buffers });
                gate_exe.call(gate_args, &gate_results);

                var gate_output = gate_results.get(zml.Buffer);
                defer gate_output.deinit();

                const gate_metrics = try check_utils.compareBuffers(io, gate_output, gate_ref, 0.1, 0.01);
                _ = check_utils.reportStageMetrics(.gate_logits, gate_metrics, 0.99);
            } else {
                std.log.info("Skipping stage 5: {s} not found in fixture", .{gate_key});
            }

            // --- Stage 6: pre_to_out — validate SDPA + gate merge before to_out linear ---
            const pre_to_out_key = try std.fmt.allocPrint(allocator, "{s}.to_out_input_diag0", .{@tagName(mode)});
            defer allocator.free(pre_to_out_key);
            if (references_are_token_comparable and fixture_store.view().hasKey(pre_to_out_key)) {
                var pre_to_out_ref = try check_utils.loadBufferFromStore(allocator, io, platform, &fixture_store, pre_to_out_key, replicated_sharding);
                defer pre_to_out_ref.deinit();
                if (token_limit) |limit| {
                    pre_to_out_ref = try check_utils.sliceTokenPrefix(io, platform, pre_to_out_ref, replicated_sharding, limit);
                }

                std.log.info("Compiling forwardBlock0Attn1DiagPreToOut...", .{});
                var pre_to_out_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagPreToOut, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer pre_to_out_exe.deinit();

                var pre_to_out_args = try pre_to_out_exe.args(allocator);
                defer pre_to_out_args.deinit(allocator);
                var pre_to_out_results = try pre_to_out_exe.results(allocator);
                defer pre_to_out_results.deinit(allocator);

                pre_to_out_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                pre_to_out_exe.call(pre_to_out_args, &pre_to_out_results);

                var pre_to_out_output = pre_to_out_results.get(zml.Buffer);
                defer pre_to_out_output.deinit();

                const pre_to_out_metrics = try check_utils.compareBuffers(io, pre_to_out_output, pre_to_out_ref, 0.1, 0.01);
                _ = check_utils.reportStageMetrics(.pre_to_out, pre_to_out_metrics, 0.99);

                // Isolate to_out: apply to_out on fixture pre_to_out and compare to fixture output.
                std.log.info("Compiling forwardBlock0Attn1DiagToOutOnly...", .{});
                const pre_to_out_tensor = zml.Tensor.fromShape(pre_to_out_ref.shape());
                var to_out_only_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagToOutOnly, .{ pre_to_out_tensor, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer to_out_only_exe.deinit();

                var to_out_only_args = try to_out_only_exe.args(allocator);
                defer to_out_only_args.deinit(allocator);
                var to_out_only_results = try to_out_only_exe.results(allocator);
                defer to_out_only_results.deinit(allocator);

                to_out_only_args.set(.{ pre_to_out_ref, attn_params_buffers });
                to_out_only_exe.call(to_out_only_args, &to_out_only_results);

                var to_out_only_output = to_out_only_results.get(zml.Buffer);
                defer to_out_only_output.deinit();

                const to_out_only_metrics = try check_utils.compareBuffers(io, to_out_only_output, attn_expected, 0.2, 0.01);
                std.log.info(
                    "to_out_only from fixture pre_to_out: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}",
                    .{ to_out_only_metrics.max_abs_error, to_out_only_metrics.mean_abs_error, to_out_only_metrics.close_fraction },
                );

                // F32 SDPA ablation: same computation but SDPA core in f32.
                std.log.info("Compiling forwardBlock0Attn1DiagPreToOutF32Sdpa...", .{});
                var pre_to_out_f32_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagPreToOutF32Sdpa, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer pre_to_out_f32_exe.deinit();

                var pre_to_out_f32_args = try pre_to_out_f32_exe.args(allocator);
                defer pre_to_out_f32_args.deinit(allocator);
                var pre_to_out_f32_results = try pre_to_out_f32_exe.results(allocator);
                defer pre_to_out_f32_results.deinit(allocator);

                pre_to_out_f32_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                pre_to_out_f32_exe.call(pre_to_out_f32_args, &pre_to_out_f32_results);

                var pre_to_out_f32_output = pre_to_out_f32_results.get(zml.Buffer);
                defer pre_to_out_f32_output.deinit();

                const pre_to_out_f32_metrics = try check_utils.compareBuffers(io, pre_to_out_f32_output, pre_to_out_ref, 0.1, 0.01);
                std.log.info(
                    "pre_to_out_f32_sdpa ablation: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}",
                    .{ pre_to_out_f32_metrics.max_abs_error, pre_to_out_f32_metrics.mean_abs_error, pre_to_out_f32_metrics.close_fraction },
                );

                // Gate ablation 1: no gate at all
                std.log.info("Compiling forwardBlock0Attn1DiagPreToOutNoGate...", .{});
                var pre_to_out_nogate_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagPreToOutNoGate, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer pre_to_out_nogate_exe.deinit();

                var pre_to_out_nogate_args = try pre_to_out_nogate_exe.args(allocator);
                defer pre_to_out_nogate_args.deinit(allocator);
                var pre_to_out_nogate_results = try pre_to_out_nogate_exe.results(allocator);
                defer pre_to_out_nogate_results.deinit(allocator);

                pre_to_out_nogate_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                pre_to_out_nogate_exe.call(pre_to_out_nogate_args, &pre_to_out_nogate_results);

                var pre_to_out_nogate_output = pre_to_out_nogate_results.get(zml.Buffer);
                defer pre_to_out_nogate_output.deinit();

                const pre_to_out_nogate_metrics = try check_utils.compareBuffers(io, pre_to_out_nogate_output, pre_to_out_ref, 0.1, 0.01);
                std.log.info(
                    "pre_to_out_no_gate ablation: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}",
                    .{ pre_to_out_nogate_metrics.max_abs_error, pre_to_out_nogate_metrics.mean_abs_error, pre_to_out_nogate_metrics.close_fraction },
                );

                // Gate ablation 2: sigmoid gate without *2
                std.log.info("Compiling forwardBlock0Attn1DiagPreToOutSigmoidGate...", .{});
                var pre_to_out_sig_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagPreToOutSigmoidGate, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer pre_to_out_sig_exe.deinit();

                var pre_to_out_sig_args = try pre_to_out_sig_exe.args(allocator);
                defer pre_to_out_sig_args.deinit(allocator);
                var pre_to_out_sig_results = try pre_to_out_sig_exe.results(allocator);
                defer pre_to_out_sig_results.deinit(allocator);

                pre_to_out_sig_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                pre_to_out_sig_exe.call(pre_to_out_sig_args, &pre_to_out_sig_results);

                var pre_to_out_sig_output = pre_to_out_sig_results.get(zml.Buffer);
                defer pre_to_out_sig_output.deinit();

                const pre_to_out_sig_metrics = try check_utils.compareBuffers(io, pre_to_out_sig_output, pre_to_out_ref, 0.1, 0.01);
                std.log.info(
                    "pre_to_out_sigmoid_gate ablation: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}",
                    .{ pre_to_out_sig_metrics.max_abs_error, pre_to_out_sig_metrics.mean_abs_error, pre_to_out_sig_metrics.close_fraction },
                );

                // Manual SDPA ablation: explicit qk^T softmax v in f32 (no zml.nn.sdpa call).
                std.log.info("Compiling forwardBlock0Attn1DiagPreToOutManualSdpaF32...", .{});
                var pre_to_out_manual_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagPreToOutManualSdpaF32, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer pre_to_out_manual_exe.deinit();

                var pre_to_out_manual_args = try pre_to_out_manual_exe.args(allocator);
                defer pre_to_out_manual_args.deinit(allocator);
                var pre_to_out_manual_results = try pre_to_out_manual_exe.results(allocator);
                defer pre_to_out_manual_results.deinit(allocator);

                pre_to_out_manual_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                pre_to_out_manual_exe.call(pre_to_out_manual_args, &pre_to_out_manual_results);

                var pre_to_out_manual_output = pre_to_out_manual_results.get(zml.Buffer);
                defer pre_to_out_manual_output.deinit();

                const pre_to_out_manual_metrics = try check_utils.compareBuffers(io, pre_to_out_manual_output, pre_to_out_ref, 0.1, 0.01);
                std.log.info(
                    "pre_to_out_manual_sdpa_f32 ablation: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}",
                    .{ pre_to_out_manual_metrics.max_abs_error, pre_to_out_manual_metrics.mean_abs_error, pre_to_out_manual_metrics.close_fraction },
                );

                std.log.info("Compiling forwardBlock0Attn1DiagPreToOutAltMergeTranspose...", .{});
                var pre_to_out_alt_merge_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagPreToOutAltMergeTranspose, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer pre_to_out_alt_merge_exe.deinit();

                var pre_to_out_alt_merge_args = try pre_to_out_alt_merge_exe.args(allocator);
                defer pre_to_out_alt_merge_args.deinit(allocator);
                var pre_to_out_alt_merge_results = try pre_to_out_alt_merge_exe.results(allocator);
                defer pre_to_out_alt_merge_results.deinit(allocator);

                pre_to_out_alt_merge_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                pre_to_out_alt_merge_exe.call(pre_to_out_alt_merge_args, &pre_to_out_alt_merge_results);

                var pre_to_out_alt_merge_output = pre_to_out_alt_merge_results.get(zml.Buffer);
                defer pre_to_out_alt_merge_output.deinit();

                const pre_to_out_alt_merge_metrics = try check_utils.compareBuffers(io, pre_to_out_alt_merge_output, pre_to_out_ref, 0.1, 0.01);
                std.log.info(
                    "pre_to_out_alt_merge_transpose ablation: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}",
                    .{ pre_to_out_alt_merge_metrics.max_abs_error, pre_to_out_alt_merge_metrics.mean_abs_error, pre_to_out_alt_merge_metrics.close_fraction },
                );

                // Head-first SDPA ablation: [b,h,q,hd] layout around sdpa, then swap back.
                std.log.info("Compiling forwardBlock0Attn1DiagPreToOutHeadFirstSdpa...", .{});
                var pre_to_out_head_first_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1DiagPreToOutHeadFirstSdpa, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer pre_to_out_head_first_exe.deinit();

                var pre_to_out_head_first_args = try pre_to_out_head_first_exe.args(allocator);
                defer pre_to_out_head_first_args.deinit(allocator);
                var pre_to_out_head_first_results = try pre_to_out_head_first_exe.results(allocator);
                defer pre_to_out_head_first_results.deinit(allocator);

                pre_to_out_head_first_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                pre_to_out_head_first_exe.call(pre_to_out_head_first_args, &pre_to_out_head_first_results);

                var pre_to_out_head_first_output = pre_to_out_head_first_results.get(zml.Buffer);
                defer pre_to_out_head_first_output.deinit();

                const pre_to_out_head_first_metrics = try check_utils.compareBuffers(io, pre_to_out_head_first_output, pre_to_out_ref, 0.1, 0.01);
                std.log.info(
                    "pre_to_out_head_first_sdpa ablation: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}",
                    .{ pre_to_out_head_first_metrics.max_abs_error, pre_to_out_head_first_metrics.mean_abs_error, pre_to_out_head_first_metrics.close_fraction },
                );
            } else {
                if (token_limit != null and !token_limited_reference) {
                    std.log.info("Skipping stage 6/pre_to_out ablations for token-limited run (reference captured with full context)", .{});
                } else {
                    std.log.info("Skipping stage 6: {s} not found in fixture", .{pre_to_out_key});
                }
            }

            // Ablation: run attention without mask and compare against the same expected output.
            // If this is materially closer than the masked run, mask semantics are likely wrong.
            if (references_are_token_comparable and attn_pe_cos != null and attn_pe_sin != null) {
                std.log.info("Compiling no-mask ablation: forwardBlock0Attn1WithPeCosSin...", .{});
                var ablation_exe = try platform.compileFn(allocator, io, model.forwardBlock0Attn1WithPeCosSin, .{ input_tensor, pe_cos_t, pe_sin_t, attn_params_shape }, .{ .shardings = &.{replicated_sharding} });
                defer ablation_exe.deinit();

                var ablation_args = try ablation_exe.args(allocator);
                defer ablation_args.deinit(allocator);
                var ablation_results = try ablation_exe.results(allocator);
                defer ablation_results.deinit(allocator);

                ablation_args.set(.{ attn_input, pe_cos_buf, pe_sin_buf, attn_params_buffers });
                ablation_exe.call(ablation_args, &ablation_results);

                var ablation_output = ablation_results.get(zml.Buffer);
                defer ablation_output.deinit();

                const ablation_metrics = try check_utils.compareBuffers(io, ablation_output, attn_expected, 0.2, 0.01);
                std.log.info(
                    "no_mask ablation: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4}",
                    .{ ablation_metrics.max_abs_error, ablation_metrics.mean_abs_error, ablation_metrics.close_fraction },
                );
            } else {
                if (token_limit != null and !token_limited_reference) {
                    std.log.info("Skipping no-mask ablation for token-limited run (output reference is full-context)", .{});
                } else {
                    std.log.info("Skipping no-mask ablation: pe_cos/pe_sin are required", .{});
                }
            }
        }
    }

    if (references_are_token_comparable) {
        try zml.testing.expectClose(io, attn_output, attn_expected, .{
            .absolute_tolerance = 0.2,
            .relative_tolerance = 0.01,
            .minimum_close_fraction = 0.999,
        });
    } else {
        std.log.info("Skipping final output parity assert for token-limited run: full-context reference is not directly comparable", .{});
    }

    std.log.info("attention parity PASSED for mode={s}", .{@tagName(mode)});
}

fn parseMode(v: []const u8) !Mode {
    inline for (std.meta.fields(Mode)) |field| {
        if (std.mem.eql(u8, v, field.name)) {
            return @enumFromInt(field.value);
        }
    }

    std.log.err("Invalid mode: {s}. Expected one of: attn1, attn2, audio_attn1, audio_attn2, audio_to_video_attn, video_to_audio_attn", .{v});
    return error.InvalidArgs;
}

fn modeToKind(mode: Mode) model.AttentionKind {
    return switch (mode) {
        .attn1 => .attn1,
        .attn2 => .attn2,
        .audio_attn1 => .audio_attn1,
        .audio_attn2 => .audio_attn2,
        .audio_to_video_attn => .audio_to_video_attn,
        .video_to_audio_attn => .video_to_audio_attn,
    };
}

fn loadOptionalFixtureBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !?zml.Buffer {
    if (!store.view().hasKey(key)) return null;
    return try check_utils.loadBufferFromStore(allocator, io, platform, store, key, sharding);
}
