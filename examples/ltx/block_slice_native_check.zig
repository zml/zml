/// Native block-slice parity checker (8 contiguous blocks): inline-AdaLN full stream.
///
/// Verifies:
///   1) forwardBlockSlice8NativeVideo(...) == block_slice_native.vx_out
///   2) forwardBlockSlice8NativeAudio(...) == block_slice_native.ax_out
///
/// Checkpoint expectation:
///   transformer blocks are re-indexed locally as transformer_blocks.0..7.

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

/// Context holding all fixture and checkpoint buffers.
/// Separated from main() to reduce stack pressure during tracing.
const CheckContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    params_shape: model.BlockSlice8FullParams,
    params_bufs: zml.Bufferized(model.BlockSlice8FullParams),

    vx_in_buf: zml.Buffer,
    ax_in_buf: zml.Buffer,
    video_timesteps_buf: zml.Buffer,
    audio_timesteps_buf: zml.Buffer,
    v_prompt_timestep_buf: zml.Buffer,
    a_prompt_timestep_buf: zml.Buffer,
    v_pe_cos_buf: zml.Buffer,
    v_pe_sin_buf: zml.Buffer,
    a_pe_cos_buf: zml.Buffer,
    a_pe_sin_buf: zml.Buffer,
    v_text_ctx_buf: zml.Buffer,
    a_text_ctx_buf: zml.Buffer,
    v_text_ctx_mask_buf: ?zml.Buffer,
    a_text_ctx_mask_buf: ?zml.Buffer,
    v_cross_ss_ts_buf: zml.Buffer,
    v_cross_gate_ts_buf: zml.Buffer,
    a_cross_ss_ts_buf: zml.Buffer,
    a_cross_gate_ts_buf: zml.Buffer,
    a2v_pe_cos_buf: zml.Buffer,
    a2v_pe_sin_buf: zml.Buffer,
    a2v_k_pe_cos_buf: zml.Buffer,
    a2v_k_pe_sin_buf: zml.Buffer,
    a2v_masks_buf: ?zml.Buffer,
    a2v_mask_block_bufs: [8]?zml.Buffer,
    v2a_pe_cos_buf: zml.Buffer,
    v2a_pe_sin_buf: zml.Buffer,
    v2a_k_pe_cos_buf: zml.Buffer,
    v2a_k_pe_sin_buf: zml.Buffer,
    v2a_masks_buf: ?zml.Buffer,
    v2a_mask_block_bufs: [8]?zml.Buffer,
    vx_out_ref_buf: zml.Buffer,
    ax_out_ref_buf: zml.Buffer,
    vx_block_out_ref_bufs: [8]?zml.Buffer,
    ax_block_out_ref_bufs: [8]?zml.Buffer,
    ax_in_block_bufs: [8]?zml.Buffer,
    norm_ax_block_bufs: [8]?zml.Buffer,
    a_text_x_block_bufs: [8]?zml.Buffer,
    a_text_ctx_block_bufs: [8]?zml.Buffer,
    v2a_x_block_bufs: [8]?zml.Buffer,
    v2a_ctx_block_bufs: [8]?zml.Buffer,
    v2a_gate_block_bufs: [8]?zml.Buffer,
    ax_scaled_block_bufs: [8]?zml.Buffer,
    agate_msa_block_bufs: [8]?zml.Buffer,
    agate_mlp_block_bufs: [8]?zml.Buffer,
    agate_text_ca_block_bufs: [8]?zml.Buffer,
    audio_ff_out_block_bufs: [8]?zml.Buffer,
    audio_ff_net0_proj_out_block_bufs: [8]?zml.Buffer,
    audio_ff_net0_out_block_bufs: [8]?zml.Buffer,
    ax_after_msa_block0_ref_buf: ?zml.Buffer,
    ax_after_text_ca_block0_ref_buf: ?zml.Buffer,
    ax_after_v2a_block0_ref_buf: ?zml.Buffer,
    ax_after_ff_block0_ref_buf: ?zml.Buffer,

    fn deinit(self: *CheckContext) void {
        self.vx_in_buf.deinit();
        self.ax_in_buf.deinit();
        self.video_timesteps_buf.deinit();
        self.audio_timesteps_buf.deinit();
        self.v_prompt_timestep_buf.deinit();
        self.a_prompt_timestep_buf.deinit();
        self.v_pe_cos_buf.deinit();
        self.v_pe_sin_buf.deinit();
        self.a_pe_cos_buf.deinit();
        self.a_pe_sin_buf.deinit();
        self.v_text_ctx_buf.deinit();
        self.a_text_ctx_buf.deinit();
        if (self.v_text_ctx_mask_buf) |*b| b.deinit();
        if (self.a_text_ctx_mask_buf) |*b| b.deinit();
        self.v_cross_ss_ts_buf.deinit();
        self.v_cross_gate_ts_buf.deinit();
        self.a_cross_ss_ts_buf.deinit();
        self.a_cross_gate_ts_buf.deinit();
        self.a2v_pe_cos_buf.deinit();
        self.a2v_pe_sin_buf.deinit();
        self.a2v_k_pe_cos_buf.deinit();
        self.a2v_k_pe_sin_buf.deinit();
        if (self.a2v_masks_buf) |*b| b.deinit();
        for (&self.a2v_mask_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        self.v2a_pe_cos_buf.deinit();
        self.v2a_pe_sin_buf.deinit();
        self.v2a_k_pe_cos_buf.deinit();
        self.v2a_k_pe_sin_buf.deinit();
        if (self.v2a_masks_buf) |*b| b.deinit();
        for (&self.v2a_mask_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        self.vx_out_ref_buf.deinit();
        self.ax_out_ref_buf.deinit();
        for (&self.vx_block_out_ref_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.ax_block_out_ref_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.ax_in_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.norm_ax_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.a_text_x_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.a_text_ctx_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.v2a_x_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.v2a_ctx_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.v2a_gate_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.ax_scaled_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.agate_msa_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.agate_mlp_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.agate_text_ca_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.audio_ff_out_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.audio_ff_net0_proj_out_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        for (&self.audio_ff_net0_out_block_bufs) |*buf| {
            if (buf.*) |*b| b.deinit();
        }
        if (self.ax_after_msa_block0_ref_buf) |*b| b.deinit();
        if (self.ax_after_text_ca_block0_ref_buf) |*b| b.deinit();
        if (self.ax_after_v2a_block0_ref_buf) |*b| b.deinit();
        if (self.ax_after_ff_block0_ref_buf) |*b| b.deinit();
        model.unloadBlockSlice8FullBuffers(&self.params_bufs);
        self.platform.deinit(self.allocator);
    }

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        ckpt_path: []const u8,
        fixture_path: []const u8,
    ) !CheckContext {
        var ckpt_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, ckpt_path) catch |err| {
            std.log.err("Failed to open checkpoint: {s}", .{ckpt_path});
            return err;
        };
        defer ckpt_reg.deinit();
        var ckpt_store: zml.io.TensorStore = .fromRegistry(allocator, &ckpt_reg);
        defer ckpt_store.deinit();

        var fix_reg = zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path) catch |err| {
            std.log.err("Failed to open fixture: {s}", .{fixture_path});
            return err;
        };
        defer fix_reg.deinit();
        var fix_store: zml.io.TensorStore = .fromRegistry(allocator, &fix_reg);
        defer fix_store.deinit();

        const platform: *zml.Platform = try .auto(allocator, io, .{});
        errdefer platform.deinit(allocator);
        const sharding = try zml.sharding.replicatedSharding(platform);

        var vx_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.vx_in", sharding);
        errdefer vx_in_buf.deinit();
        var ax_in_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.ax_in", sharding);
        errdefer ax_in_buf.deinit();

        var video_timesteps_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.video_timesteps", sharding);
        errdefer video_timesteps_buf.deinit();
        var audio_timesteps_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.audio_timesteps", sharding);
        errdefer audio_timesteps_buf.deinit();
        var v_prompt_timestep_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_prompt_timestep", sharding);
        errdefer v_prompt_timestep_buf.deinit();
        var a_prompt_timestep_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_prompt_timestep", sharding);
        errdefer a_prompt_timestep_buf.deinit();

        var v_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_pe_cos", sharding);
        errdefer v_pe_cos_buf.deinit();
        var v_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_pe_sin", sharding);
        errdefer v_pe_sin_buf.deinit();
        var a_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_pe_cos", sharding);
        errdefer a_pe_cos_buf.deinit();
        var a_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_pe_sin", sharding);
        errdefer a_pe_sin_buf.deinit();

        var v_text_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_text_ctx", sharding);
        errdefer v_text_ctx_buf.deinit();
        var a_text_ctx_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_text_ctx", sharding);
        errdefer a_text_ctx_buf.deinit();
        const has_v_text_ctx_mask = fix_store.view().hasKey("block_slice_native.v_text_ctx_mask");
        const has_a_text_ctx_mask = fix_store.view().hasKey("block_slice_native.a_text_ctx_mask");
        if (has_v_text_ctx_mask != has_a_text_ctx_mask) {
            std.log.err("Fixture must provide both text context masks or neither", .{});
            return error.InvalidArgs;
        }
        var v_text_ctx_mask_buf: ?zml.Buffer = null;
        errdefer if (v_text_ctx_mask_buf) |*b| b.deinit();
        var a_text_ctx_mask_buf: ?zml.Buffer = null;
        errdefer if (a_text_ctx_mask_buf) |*b| b.deinit();
        if (has_v_text_ctx_mask) {
            v_text_ctx_mask_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_text_ctx_mask", sharding);
            a_text_ctx_mask_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_text_ctx_mask", sharding);
        }

        var v_cross_ss_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_cross_ss_ts", sharding);
        errdefer v_cross_ss_ts_buf.deinit();
        var v_cross_gate_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v_cross_gate_ts", sharding);
        errdefer v_cross_gate_ts_buf.deinit();
        var a_cross_ss_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_cross_ss_ts", sharding);
        errdefer a_cross_ss_ts_buf.deinit();
        var a_cross_gate_ts_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a_cross_gate_ts", sharding);
        errdefer a_cross_gate_ts_buf.deinit();

        var a2v_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a2v_pe_cos", sharding);
        errdefer a2v_pe_cos_buf.deinit();
        var a2v_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a2v_pe_sin", sharding);
        errdefer a2v_pe_sin_buf.deinit();
        var a2v_k_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a2v_k_pe_cos", sharding);
        errdefer a2v_k_pe_cos_buf.deinit();
        var a2v_k_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a2v_k_pe_sin", sharding);
        errdefer a2v_k_pe_sin_buf.deinit();

        const has_a2v_masks = fix_store.view().hasKey("block_slice_native.a2v_masks");
        const has_v2a_masks = fix_store.view().hasKey("block_slice_native.v2a_masks");
        if (has_a2v_masks != has_v2a_masks) {
            std.log.err("Fixture must provide both AV masks or neither", .{});
            return error.InvalidArgs;
        }
        var a2v_masks_buf: ?zml.Buffer = null;
        errdefer if (a2v_masks_buf) |*b| b.deinit();
        var v2a_masks_buf: ?zml.Buffer = null;
        errdefer if (v2a_masks_buf) |*b| b.deinit();
        if (has_a2v_masks) {
            a2v_masks_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.a2v_masks", sharding);
            v2a_masks_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v2a_masks", sharding);
        }
        var a2v_mask_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer {
            for (&a2v_mask_block_bufs) |*buf| {
                if (buf.*) |*b| b.deinit();
            }
        }
        var v2a_mask_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer {
            for (&v2a_mask_block_bufs) |*buf| {
                if (buf.*) |*b| b.deinit();
            }
        }
        inline for (0..8) |i| {
            const a2v_key = try std.fmt.allocPrint(allocator, "block_slice_native.a2v_mask_block_{d}", .{i});
            defer allocator.free(a2v_key);
            const v2a_key = try std.fmt.allocPrint(allocator, "block_slice_native.v2a_mask_block_{d}", .{i});
            defer allocator.free(v2a_key);
            if (fix_store.view().hasKey(a2v_key)) {
                a2v_mask_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, a2v_key, sharding);
            }
            if (fix_store.view().hasKey(v2a_key)) {
                v2a_mask_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, v2a_key, sharding);
            }
        }

        var v2a_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v2a_pe_cos", sharding);
        errdefer v2a_pe_cos_buf.deinit();
        var v2a_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v2a_pe_sin", sharding);
        errdefer v2a_pe_sin_buf.deinit();
        var v2a_k_pe_cos_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v2a_k_pe_cos", sharding);
        errdefer v2a_k_pe_cos_buf.deinit();
        var v2a_k_pe_sin_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.v2a_k_pe_sin", sharding);
        errdefer v2a_k_pe_sin_buf.deinit();

        var vx_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.vx_out", sharding);
        errdefer vx_out_ref_buf.deinit();
        var ax_out_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.ax_out", sharding);
        errdefer ax_out_ref_buf.deinit();

        var vx_block_out_ref_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer {
            for (&vx_block_out_ref_bufs) |*buf| {
                if (buf.*) |*b| b.deinit();
            }
        }
        var ax_block_out_ref_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer {
            for (&ax_block_out_ref_bufs) |*buf| {
                if (buf.*) |*b| b.deinit();
            }
        }
        var ax_in_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&ax_in_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var norm_ax_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&norm_ax_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var a_text_x_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&a_text_x_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var a_text_ctx_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&a_text_ctx_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var v2a_x_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&v2a_x_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var v2a_ctx_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&v2a_ctx_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var v2a_gate_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&v2a_gate_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var ax_scaled_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&ax_scaled_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var agate_msa_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&agate_msa_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var agate_mlp_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&agate_mlp_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var agate_text_ca_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&agate_text_ca_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var audio_ff_out_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&audio_ff_out_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var audio_ff_net0_proj_out_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&audio_ff_net0_proj_out_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        var audio_ff_net0_out_block_bufs: [8]?zml.Buffer = .{null, null, null, null, null, null, null, null};
        errdefer for (&audio_ff_net0_out_block_bufs) |*buf| if (buf.*) |*b| b.deinit();
        inline for (0..8) |i| {
            const vx_key = try std.fmt.allocPrint(allocator, "block_slice_native.vx_out_block_{d}", .{i});
            defer allocator.free(vx_key);
            if (fix_store.view().hasKey(vx_key)) {
                vx_block_out_ref_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, vx_key, sharding);
            }

            const key = try std.fmt.allocPrint(allocator, "block_slice_native.ax_out_block_{d}", .{i});
            defer allocator.free(key);
            if (fix_store.view().hasKey(key)) {
                ax_block_out_ref_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, key, sharding);
            }

            const ax_in_key = try std.fmt.allocPrint(allocator, "block_slice_native.ax_in_block_{d}", .{i});
            defer allocator.free(ax_in_key);
            if (fix_store.view().hasKey(ax_in_key)) {
                ax_in_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, ax_in_key, sharding);
            }

            const norm_ax_key = try std.fmt.allocPrint(allocator, "block_slice_native.norm_ax_block_{d}", .{i});
            defer allocator.free(norm_ax_key);
            if (fix_store.view().hasKey(norm_ax_key)) {
                norm_ax_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, norm_ax_key, sharding);
            }

            const a_text_x_key = try std.fmt.allocPrint(allocator, "block_slice_native.a_text_x_block_{d}", .{i});
            defer allocator.free(a_text_x_key);
            if (fix_store.view().hasKey(a_text_x_key)) {
                a_text_x_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, a_text_x_key, sharding);
            }

            const a_text_ctx_key = try std.fmt.allocPrint(allocator, "block_slice_native.a_text_ctx_block_{d}", .{i});
            defer allocator.free(a_text_ctx_key);
            if (fix_store.view().hasKey(a_text_ctx_key)) {
                a_text_ctx_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, a_text_ctx_key, sharding);
            }

            const v2a_x_key = try std.fmt.allocPrint(allocator, "block_slice_native.v2a_x_block_{d}", .{i});
            defer allocator.free(v2a_x_key);
            if (fix_store.view().hasKey(v2a_x_key)) {
                v2a_x_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, v2a_x_key, sharding);
            }

            const v2a_ctx_key = try std.fmt.allocPrint(allocator, "block_slice_native.v2a_ctx_block_{d}", .{i});
            defer allocator.free(v2a_ctx_key);
            if (fix_store.view().hasKey(v2a_ctx_key)) {
                v2a_ctx_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, v2a_ctx_key, sharding);
            }

            const v2a_gate_key = try std.fmt.allocPrint(allocator, "block_slice_native.v2a_gate_block_{d}", .{i});
            defer allocator.free(v2a_gate_key);
            if (fix_store.view().hasKey(v2a_gate_key)) {
                v2a_gate_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, v2a_gate_key, sharding);
            }

            const ax_scaled_key = try std.fmt.allocPrint(allocator, "block_slice_native.ax_scaled_block_{d}", .{i});
            defer allocator.free(ax_scaled_key);
            if (fix_store.view().hasKey(ax_scaled_key)) {
                ax_scaled_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, ax_scaled_key, sharding);
            }

            const audio_ff_out_key = try std.fmt.allocPrint(allocator, "block_slice_native.audio_ff_out_block_{d}", .{i});
            defer allocator.free(audio_ff_out_key);
            if (fix_store.view().hasKey(audio_ff_out_key)) {
                audio_ff_out_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, audio_ff_out_key, sharding);
            }

            const audio_ff_net0_proj_out_key = try std.fmt.allocPrint(allocator, "block_slice_native.audio_ff_net0_proj_out_block_{d}", .{i});
            defer allocator.free(audio_ff_net0_proj_out_key);
            if (fix_store.view().hasKey(audio_ff_net0_proj_out_key)) {
                audio_ff_net0_proj_out_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, audio_ff_net0_proj_out_key, sharding);
            }

            const audio_ff_net0_out_key = try std.fmt.allocPrint(allocator, "block_slice_native.audio_ff_net0_out_block_{d}", .{i});
            defer allocator.free(audio_ff_net0_out_key);
            if (fix_store.view().hasKey(audio_ff_net0_out_key)) {
                audio_ff_net0_out_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, audio_ff_net0_out_key, sharding);
            }

            const agate_msa_key = try std.fmt.allocPrint(allocator, "block_slice_native.agate_msa_block_{d}", .{i});
            defer allocator.free(agate_msa_key);
            if (fix_store.view().hasKey(agate_msa_key)) {
                agate_msa_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, agate_msa_key, sharding);
            }

            const agate_mlp_key = try std.fmt.allocPrint(allocator, "block_slice_native.agate_mlp_block_{d}", .{i});
            defer allocator.free(agate_mlp_key);
            if (fix_store.view().hasKey(agate_mlp_key)) {
                agate_mlp_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, agate_mlp_key, sharding);
            }

            const agate_text_ca_key = try std.fmt.allocPrint(allocator, "block_slice_native.agate_text_ca_block_{d}", .{i});
            defer allocator.free(agate_text_ca_key);
            if (fix_store.view().hasKey(agate_text_ca_key)) {
                agate_text_ca_block_bufs[i] = try loadBuf(allocator, io, platform, &fix_store, agate_text_ca_key, sharding);
            }
        }

        var ax_after_msa_block0_ref_buf: ?zml.Buffer = null;
        errdefer if (ax_after_msa_block0_ref_buf) |*b| b.deinit();
        if (fix_store.view().hasKey("block_slice_native.ax_after_msa_block_0")) {
            ax_after_msa_block0_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.ax_after_msa_block_0", sharding);
        }

        var ax_after_text_ca_block0_ref_buf: ?zml.Buffer = null;
        errdefer if (ax_after_text_ca_block0_ref_buf) |*b| b.deinit();
        if (fix_store.view().hasKey("block_slice_native.ax_after_text_ca_block_0")) {
            ax_after_text_ca_block0_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.ax_after_text_ca_block_0", sharding);
        }

        var ax_after_v2a_block0_ref_buf: ?zml.Buffer = null;
        errdefer if (ax_after_v2a_block0_ref_buf) |*b| b.deinit();
        if (fix_store.view().hasKey("block_slice_native.ax_after_v2a_block_0")) {
            ax_after_v2a_block0_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.ax_after_v2a_block_0", sharding);
        }

        var ax_after_ff_block0_ref_buf: ?zml.Buffer = null;
        errdefer if (ax_after_ff_block0_ref_buf) |*b| b.deinit();
        if (fix_store.view().hasKey("block_slice_native.ax_after_ff_block_0")) {
            ax_after_ff_block0_ref_buf = try loadBuf(allocator, io, platform, &fix_store, "block_slice_native.ax_after_ff_block_0", sharding);
        }

        var params_shape = model.initBlockSlice8FullParams(ckpt_store.view());
        
        // Load each block's parameters individually to avoid bufferize issues with arrays of complex structs.
        var params_bufs: zml.Bufferized(model.BlockSlice8FullParams) = undefined;
        inline for (0..8) |i| {
            const block_params = try zml.io.load(model.Block0FullParams, &params_shape.blocks[i], allocator, io, platform, .{
                .store = &ckpt_store,
                .shardings = &.{sharding},
                .parallelism = 16,
                .dma_chunks = 8,
                .dma_chunk_size = 64 * zml.MiB,
            });
            params_bufs.blocks[i] = block_params;
        }
        errdefer {
            inline for (0..8) |i| {
                model.unloadBlock0FullBuffers(&params_bufs.blocks[i]);
            }
        }

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .sharding = sharding,
            .params_shape = params_shape,
            .params_bufs = params_bufs,
            .vx_in_buf = vx_in_buf,
            .ax_in_buf = ax_in_buf,
            .video_timesteps_buf = video_timesteps_buf,
            .audio_timesteps_buf = audio_timesteps_buf,
            .v_prompt_timestep_buf = v_prompt_timestep_buf,
            .a_prompt_timestep_buf = a_prompt_timestep_buf,
            .v_pe_cos_buf = v_pe_cos_buf,
            .v_pe_sin_buf = v_pe_sin_buf,
            .a_pe_cos_buf = a_pe_cos_buf,
            .a_pe_sin_buf = a_pe_sin_buf,
            .v_text_ctx_buf = v_text_ctx_buf,
            .a_text_ctx_buf = a_text_ctx_buf,
            .v_text_ctx_mask_buf = v_text_ctx_mask_buf,
            .a_text_ctx_mask_buf = a_text_ctx_mask_buf,
            .v_cross_ss_ts_buf = v_cross_ss_ts_buf,
            .v_cross_gate_ts_buf = v_cross_gate_ts_buf,
            .a_cross_ss_ts_buf = a_cross_ss_ts_buf,
            .a_cross_gate_ts_buf = a_cross_gate_ts_buf,
            .a2v_pe_cos_buf = a2v_pe_cos_buf,
            .a2v_pe_sin_buf = a2v_pe_sin_buf,
            .a2v_k_pe_cos_buf = a2v_k_pe_cos_buf,
            .a2v_k_pe_sin_buf = a2v_k_pe_sin_buf,
            .a2v_masks_buf = a2v_masks_buf,
            .a2v_mask_block_bufs = a2v_mask_block_bufs,
            .v2a_pe_cos_buf = v2a_pe_cos_buf,
            .v2a_pe_sin_buf = v2a_pe_sin_buf,
            .v2a_k_pe_cos_buf = v2a_k_pe_cos_buf,
            .v2a_k_pe_sin_buf = v2a_k_pe_sin_buf,
            .v2a_masks_buf = v2a_masks_buf,
            .v2a_mask_block_bufs = v2a_mask_block_bufs,
            .vx_out_ref_buf = vx_out_ref_buf,
            .ax_out_ref_buf = ax_out_ref_buf,
            .vx_block_out_ref_bufs = vx_block_out_ref_bufs,
            .ax_block_out_ref_bufs = ax_block_out_ref_bufs,
            .ax_in_block_bufs = ax_in_block_bufs,
            .norm_ax_block_bufs = norm_ax_block_bufs,
            .a_text_x_block_bufs = a_text_x_block_bufs,
            .a_text_ctx_block_bufs = a_text_ctx_block_bufs,
            .v2a_x_block_bufs = v2a_x_block_bufs,
            .v2a_ctx_block_bufs = v2a_ctx_block_bufs,
            .v2a_gate_block_bufs = v2a_gate_block_bufs,
            .ax_scaled_block_bufs = ax_scaled_block_bufs,
            .agate_msa_block_bufs = agate_msa_block_bufs,
            .agate_mlp_block_bufs = agate_mlp_block_bufs,
            .agate_text_ca_block_bufs = agate_text_ca_block_bufs,
            .audio_ff_out_block_bufs = audio_ff_out_block_bufs,
            .audio_ff_net0_proj_out_block_bufs = audio_ff_net0_proj_out_block_bufs,
            .audio_ff_net0_out_block_bufs = audio_ff_net0_out_block_bufs,
            .ax_after_msa_block0_ref_buf = ax_after_msa_block0_ref_buf,
            .ax_after_text_ca_block0_ref_buf = ax_after_text_ca_block0_ref_buf,
            .ax_after_v2a_block0_ref_buf = ax_after_v2a_block0_ref_buf,
            .ax_after_ff_block0_ref_buf = ax_after_ff_block0_ref_buf,
        };
    }
};

fn logAudioBlock0StageDiagnostics(ctx: *CheckContext) !void {
    if (ctx.ax_after_msa_block0_ref_buf == null or
        ctx.ax_after_text_ca_block0_ref_buf == null or
        ctx.ax_after_v2a_block0_ref_buf == null or
        ctx.ax_in_block_bufs[0] == null or
        ctx.norm_ax_block_bufs[0] == null or
        ctx.a_text_x_block_bufs[0] == null or
        ctx.a_text_ctx_block_bufs[0] == null or
        ctx.v2a_x_block_bufs[0] == null or
        ctx.v2a_ctx_block_bufs[0] == null or
        ctx.v2a_gate_block_bufs[0] == null or
        ctx.v2a_mask_block_bufs[0] == null or
        ctx.ax_scaled_block_bufs[0] == null or
        ctx.agate_msa_block_bufs[0] == null or
        ctx.agate_mlp_block_bufs[0] == null or
        ctx.agate_text_ca_block_bufs[0] == null)
    {
        std.log.info("Block-0 stage diagnostics unavailable (missing fixture keys)", .{});
        return;
    }

    std.log.info("Running block-0 audio stage diagnostics...", .{});
    var exe = try ctx.platform.compileFn(
        ctx.allocator,
        ctx.io,
        model.forwardBlock0AudioStreamStages,
        .{
            zml.Tensor.fromShape(ctx.ax_in_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.norm_ax_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.a_text_x_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.agate_msa_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.agate_text_ca_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.a_text_ctx_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.v2a_x_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.v2a_ctx_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_gate_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.v2a_mask_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.ax_scaled_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.agate_mlp_block_bufs[0].?.shape()),
            ctx.params_shape.blocks[0],
        },
        .{ .shardings = &.{ctx.sharding} },
    );
    defer exe.deinit();

    var args = try exe.args(ctx.allocator);
    defer args.deinit(ctx.allocator);
    var results = try exe.results(ctx.allocator);
    defer results.deinit(ctx.allocator);

    args.set(.{
        ctx.ax_in_block_bufs[0].?,
        ctx.norm_ax_block_bufs[0].?,
        ctx.a_text_x_block_bufs[0].?,
        ctx.a_pe_cos_buf,
        ctx.a_pe_sin_buf,
        ctx.agate_msa_block_bufs[0].?,
        ctx.agate_text_ca_block_bufs[0].?,
        ctx.a_text_ctx_block_bufs[0].?,
        ctx.v2a_x_block_bufs[0].?,
        ctx.v2a_ctx_block_bufs[0].?,
        ctx.v2a_pe_cos_buf,
        ctx.v2a_pe_sin_buf,
        ctx.v2a_k_pe_cos_buf,
        ctx.v2a_k_pe_sin_buf,
        ctx.v2a_gate_block_bufs[0].?,
        ctx.v2a_mask_block_bufs[0].?,
        ctx.ax_scaled_block_bufs[0].?,
        ctx.agate_mlp_block_bufs[0].?,
        ctx.params_bufs.blocks[0],
    });
    exe.call(args, &results);

    const out = results.get(zml.Bufferized(model.Block0AudioStageOutputs));

    const msa_metrics = try check_utils.compareBuffers(ctx.io, out.ax_after_msa, ctx.ax_after_msa_block0_ref_buf.?, 0.2, 0.01);
    std.log.info(
        "Audio block 0 stage after_msa: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
        .{ msa_metrics.close_fraction, msa_metrics.max_abs_error, msa_metrics.mean_abs_error },
    );

    const text_ca_metrics = try check_utils.compareBuffers(ctx.io, out.ax_after_text_ca, ctx.ax_after_text_ca_block0_ref_buf.?, 0.2, 0.01);
    std.log.info(
        "Audio block 0 stage after_text_ca: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
        .{ text_ca_metrics.close_fraction, text_ca_metrics.max_abs_error, text_ca_metrics.mean_abs_error },
    );

    const v2a_metrics = try check_utils.compareBuffers(ctx.io, out.ax_after_v2a, ctx.ax_after_v2a_block0_ref_buf.?, 0.2, 0.01);
    std.log.info(
        "Audio block 0 stage after_v2a: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
        .{ v2a_metrics.close_fraction, v2a_metrics.max_abs_error, v2a_metrics.mean_abs_error },
    );

    if (ctx.ax_after_ff_block0_ref_buf) |ax_after_ff_ref| {
        const ff_metrics = try check_utils.compareBuffers(ctx.io, out.ax_out, ax_after_ff_ref, 0.2, 0.01);
        std.log.info(
            "Audio block 0 stage after_ff: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ ff_metrics.close_fraction, ff_metrics.max_abs_error, ff_metrics.mean_abs_error },
        );
    }

    if (ctx.audio_ff_out_block_bufs[0]) |ff_out_ref| {
        const ff_out_metrics = try check_utils.compareBuffers(ctx.io, out.ff_out, ff_out_ref, 0.2, 0.01);
        std.log.info(
            "Audio block 0 stage ff_out: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ ff_out_metrics.close_fraction, ff_out_metrics.max_abs_error, ff_out_metrics.mean_abs_error },
        );
    }

    if (ctx.audio_ff_net0_proj_out_block_bufs[0]) |ff_proj_ref| {
        const ff_proj_metrics = try check_utils.compareBuffers(ctx.io, out.ff_proj_out, ff_proj_ref, 0.2, 0.01);
        std.log.info(
            "Audio block 0 stage ff_proj_out: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ ff_proj_metrics.close_fraction, ff_proj_metrics.max_abs_error, ff_proj_metrics.mean_abs_error },
        );
    }

    if (ctx.audio_ff_net0_out_block_bufs[0]) |ff_gelu_ref| {
        const ff_gelu_metrics = try check_utils.compareBuffers(ctx.io, out.ff_gelu_out, ff_gelu_ref, 0.2, 0.01);
        std.log.info(
            "Audio block 0 stage ff_gelu_out: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ ff_gelu_metrics.close_fraction, ff_gelu_metrics.max_abs_error, ff_gelu_metrics.mean_abs_error },
        );
    }
}

fn logAudioBlockLocalization(ctx: *CheckContext, use_extended_error_stats: bool) !void {
    if (ctx.ax_block_out_ref_bufs[0] == null) return;

    std.log.info("Running per-block audio localization...", .{});
    var single_block_exe = if (ctx.a2v_mask_block_bufs[0]) |a2v_mask|
        try ctx.platform.compileFn(
            ctx.allocator,
            ctx.io,
            model.forwardBlock0NativeWithAVMasks,
            .{
                zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(a2v_mask.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_mask_block_bufs[0].?.shape()),
                ctx.params_shape.blocks[0],
            },
            .{ .shardings = &.{ctx.sharding} },
        )
    else
        try ctx.platform.compileFn(
            ctx.allocator,
            ctx.io,
            model.forwardBlock0Native,
            .{
                zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                ctx.params_shape.blocks[0],
            },
            .{ .shardings = &.{ctx.sharding} },
        );
    defer single_block_exe.deinit();

    var h_v = try check_utils.copyBuffer(ctx.io, ctx.platform, ctx.vx_in_buf, ctx.sharding);
    defer h_v.deinit();
    var h_a = try check_utils.copyBuffer(ctx.io, ctx.platform, ctx.ax_in_buf, ctx.sharding);
    defer h_a.deinit();

    var first_failing_block: ?usize = null;
    inline for (0..8) |i| {
        if (i > 0 and ctx.vx_block_out_ref_bufs[i - 1] == null) {
            std.log.info("Video block refs unavailable; skipping extended localization at block {d}", .{i});
            break;
        }
        const ref_vx_in = if (i == 0) ctx.vx_in_buf else ctx.vx_block_out_ref_bufs[i - 1].?;
        const ref_ax_in = if (i == 0) ctx.ax_in_buf else ctx.ax_block_out_ref_bufs[i - 1].?;

        const v_input_metrics = try check_utils.compareBuffers(ctx.io, h_v, ref_vx_in, 0.2, 0.01);
        std.log.info(
            "Video block {d} input drift: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, v_input_metrics.close_fraction, v_input_metrics.max_abs_error, v_input_metrics.mean_abs_error },
        );

        const input_metrics = try check_utils.compareBuffers(ctx.io, h_a, ref_ax_in, 0.2, 0.01);
        std.log.info(
            "Audio block {d} input drift: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, input_metrics.close_fraction, input_metrics.max_abs_error, input_metrics.mean_abs_error },
        );

        var args = try single_block_exe.args(ctx.allocator);
        defer args.deinit(ctx.allocator);
        var results = try single_block_exe.results(ctx.allocator);
        defer results.deinit(ctx.allocator);

        if (ctx.a2v_mask_block_bufs[i]) |a2v_mask| {
            args.set(.{
                h_v, h_a,
                ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf, a2v_mask,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf, ctx.v2a_mask_block_bufs[i].?,
                ctx.params_bufs.blocks[i],
            });
        } else {
            args.set(.{
                h_v, h_a,
                ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf,
                ctx.params_bufs.blocks[i],
            });
        }
        single_block_exe.call(args, &results);

        var out = results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
        const metrics = try check_utils.compareBuffers(ctx.io, out.ax_out, ctx.ax_block_out_ref_bufs[i].?, 0.2, 0.01);
        std.log.info(
            "Audio block {d}: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, metrics.close_fraction, metrics.max_abs_error, metrics.mean_abs_error },
        );
        if (use_extended_error_stats) {
            const ext = try check_utils.compareBuffersExtended(ctx.io, out.ax_out, ctx.ax_block_out_ref_bufs[i].?, 0.2, 0.01);
            std.log.info(
                "Audio block {d} chain ext: p50={d:.6} p90={d:.6} p99={d:.6} p99.9={d:.6} rel_l2={d:.6} cos={d:.8}",
                .{ i, ext.abs_err_p50, ext.abs_err_p90, ext.abs_err_p99, ext.abs_err_p999, ext.rel_l2_error, ext.cosine_similarity },
            );
            std.log.info(
                "Audio block {d} chain ext: frac(|err|>1e-2)={d:.6} frac(|err|>1e-1)={d:.6}",
                .{ i, ext.frac_abs_err_gt_1e2, ext.frac_abs_err_gt_1e1 },
            );
        }

        // Diagnostic: keep computed video chain, but feed reference audio input
        // for this block to distinguish propagated drift from intrinsic block error.
        var tf_args = try single_block_exe.args(ctx.allocator);
        defer tf_args.deinit(ctx.allocator);
        var tf_results = try single_block_exe.results(ctx.allocator);
        defer tf_results.deinit(ctx.allocator);

        if (ctx.a2v_mask_block_bufs[i]) |a2v_mask| {
            tf_args.set(.{
                h_v, ref_ax_in,
                ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf, a2v_mask,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf, ctx.v2a_mask_block_bufs[i].?,
                ctx.params_bufs.blocks[i],
            });
        } else {
            tf_args.set(.{
                h_v, ref_ax_in,
                ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf,
                ctx.params_bufs.blocks[i],
            });
        }
        single_block_exe.call(tf_args, &tf_results);

        var tf_out = tf_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
        const tf_metrics = try check_utils.compareBuffers(ctx.io, tf_out.ax_out, ctx.ax_block_out_ref_bufs[i].?, 0.2, 0.01);
        std.log.info(
            "Audio block {d} (teacher-forced a_in): close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, tf_metrics.close_fraction, tf_metrics.max_abs_error, tf_metrics.mean_abs_error },
        );
        if (use_extended_error_stats) {
            const tf_ext = try check_utils.compareBuffersExtended(ctx.io, tf_out.ax_out, ctx.ax_block_out_ref_bufs[i].?, 0.2, 0.01);
            std.log.info(
                "Audio block {d} tf-a_in ext: p50={d:.6} p90={d:.6} p99={d:.6} p99.9={d:.6} rel_l2={d:.6} cos={d:.8}",
                .{ i, tf_ext.abs_err_p50, tf_ext.abs_err_p90, tf_ext.abs_err_p99, tf_ext.abs_err_p999, tf_ext.rel_l2_error, tf_ext.cosine_similarity },
            );
        }

        var tf_both_args = try single_block_exe.args(ctx.allocator);
        defer tf_both_args.deinit(ctx.allocator);
        var tf_both_results = try single_block_exe.results(ctx.allocator);
        defer tf_both_results.deinit(ctx.allocator);

        if (ctx.a2v_mask_block_bufs[i]) |a2v_mask| {
            tf_both_args.set(.{
                ref_vx_in, ref_ax_in,
                ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf, a2v_mask,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf, ctx.v2a_mask_block_bufs[i].?,
                ctx.params_bufs.blocks[i],
            });
        } else {
            tf_both_args.set(.{
                ref_vx_in, ref_ax_in,
                ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf,
                ctx.params_bufs.blocks[i],
            });
        }
        single_block_exe.call(tf_both_args, &tf_both_results);

        var tf_both_out = tf_both_results.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));
        const tf_both_metrics = try check_utils.compareBuffers(ctx.io, tf_both_out.ax_out, ctx.ax_block_out_ref_bufs[i].?, 0.2, 0.01);
        std.log.info(
            "Audio block {d} (teacher-forced v_in+a_in): close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, tf_both_metrics.close_fraction, tf_both_metrics.max_abs_error, tf_both_metrics.mean_abs_error },
        );

        if (first_failing_block == null and metrics.close_fraction < 0.999) {
            first_failing_block = i;
        }

        const next_v = out.vx_out;
        const next_a = out.ax_out;
        h_v.deinit();
        h_a.deinit();
        h_v = next_v;
        h_a = next_a;
    }

    if (first_failing_block) |i| {
        std.log.err("First failing audio block: {d}", .{i});
    } else {
        std.log.info("All per-block audio references passed localization.", .{});
    }
}

fn logAudioBlockExactInputs(ctx: *CheckContext, use_extended_error_stats: bool) !void {
    if (ctx.ax_in_block_bufs[0] == null or ctx.agate_text_ca_block_bufs[0] == null) return;

    std.log.info("Running per-block audio exact-input diagnostics...", .{});
    var audio_exe = try ctx.platform.compileFn(
        ctx.allocator,
        ctx.io,
        model.forwardBlock0AudioStream,
        .{
            zml.Tensor.fromShape(ctx.ax_in_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.norm_ax_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.a_text_x_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.agate_msa_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.agate_text_ca_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.a_text_ctx_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.v2a_x_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.v2a_ctx_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_gate_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.v2a_mask_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.ax_scaled_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.agate_mlp_block_bufs[0].?.shape()),
            ctx.params_shape.blocks[0],
        },
        .{ .shardings = &.{ctx.sharding} },
    );
    defer audio_exe.deinit();

    for (0..8) |i| {
        if (ctx.ax_in_block_bufs[i] == null or
            ctx.norm_ax_block_bufs[i] == null or
            ctx.a_text_x_block_bufs[i] == null or
            ctx.a_text_ctx_block_bufs[i] == null or
            ctx.v2a_x_block_bufs[i] == null or
            ctx.v2a_ctx_block_bufs[i] == null or
            ctx.v2a_gate_block_bufs[i] == null or
            ctx.v2a_mask_block_bufs[i] == null or
            ctx.ax_scaled_block_bufs[i] == null or
            ctx.agate_msa_block_bufs[i] == null or
            ctx.agate_mlp_block_bufs[i] == null or
            ctx.agate_text_ca_block_bufs[i] == null or
            ctx.ax_block_out_ref_bufs[i] == null)
        {
            std.log.info("Audio block {d} exact-input diagnostics unavailable", .{i});
            continue;
        }

        var args = try audio_exe.args(ctx.allocator);
        defer args.deinit(ctx.allocator);
        var results = try audio_exe.results(ctx.allocator);
        defer results.deinit(ctx.allocator);

        args.set(.{
            ctx.ax_in_block_bufs[i].?,
            ctx.norm_ax_block_bufs[i].?,
            ctx.a_text_x_block_bufs[i].?,
            ctx.a_pe_cos_buf,
            ctx.a_pe_sin_buf,
            ctx.agate_msa_block_bufs[i].?,
            ctx.agate_text_ca_block_bufs[i].?,
            ctx.a_text_ctx_block_bufs[i].?,
            ctx.v2a_x_block_bufs[i].?,
            ctx.v2a_ctx_block_bufs[i].?,
            ctx.v2a_pe_cos_buf,
            ctx.v2a_pe_sin_buf,
            ctx.v2a_k_pe_cos_buf,
            ctx.v2a_k_pe_sin_buf,
            ctx.v2a_gate_block_bufs[i].?,
            ctx.v2a_mask_block_bufs[i].?,
            ctx.ax_scaled_block_bufs[i].?,
            ctx.agate_mlp_block_bufs[i].?,
            ctx.params_bufs.blocks[i],
        });
        audio_exe.call(args, &results);

        const out = results.get(zml.Buffer);
        const metrics = try check_utils.compareBuffers(ctx.io, out, ctx.ax_block_out_ref_bufs[i].?, 0.2, 0.01);
        std.log.info(
            "Audio block {d} (exact inputs): close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, metrics.close_fraction, metrics.max_abs_error, metrics.mean_abs_error },
        );
        if (use_extended_error_stats) {
            const ext = try check_utils.compareBuffersExtended(ctx.io, out, ctx.ax_block_out_ref_bufs[i].?, 0.2, 0.01);
            std.log.info(
                "Audio block {d} exact ext: p50={d:.6} p90={d:.6} p99={d:.6} p99.9={d:.6} rel_l2={d:.6} cos={d:.8}",
                .{ i, ext.abs_err_p50, ext.abs_err_p90, ext.abs_err_p99, ext.abs_err_p999, ext.rel_l2_error, ext.cosine_similarity },
            );
        }
    }
}

fn logAudioNativeRecomputedIntermediates(ctx: *CheckContext) !void {
    if (ctx.a2v_mask_block_bufs[0] == null or ctx.v2a_mask_block_bufs[0] == null) return;
    if (ctx.norm_ax_block_bufs[0] == null or ctx.a_text_x_block_bufs[0] == null or ctx.v2a_x_block_bufs[0] == null or ctx.ax_scaled_block_bufs[0] == null) return;

    std.log.info("Running native recomputed audio-intermediate diagnostics...", .{});
    var exe = try ctx.platform.compileFn(
        ctx.allocator,
        ctx.io,
        model.forwardBlock0NativeAudioIntermediatesWithAVMasks,
        .{
            zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
            zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
            zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
            zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
            zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
            zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
            zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
            zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
            zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
            zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
            zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
            zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
            zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.a2v_mask_block_bufs[0].?.shape()),
            zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
            zml.Tensor.fromShape(ctx.v2a_mask_block_bufs[0].?.shape()),
            ctx.params_shape.blocks[0],
        },
        .{ .shardings = &.{ctx.sharding} },
    );
    defer exe.deinit();

    for (0..8) |i| {
        if (ctx.ax_block_out_ref_bufs[i] == null or ctx.vx_block_out_ref_bufs[i] == null or
            ctx.norm_ax_block_bufs[i] == null or ctx.a_text_x_block_bufs[i] == null or ctx.v2a_x_block_bufs[i] == null or ctx.ax_scaled_block_bufs[i] == null or
            ctx.a2v_mask_block_bufs[i] == null or ctx.v2a_mask_block_bufs[i] == null)
        {
            std.log.info("Audio block {d} native-intermediate diagnostics unavailable", .{i});
            continue;
        }

        const ref_vx_in = if (i == 0) ctx.vx_in_buf else ctx.vx_block_out_ref_bufs[i - 1].?;
        const ref_ax_in = if (i == 0) ctx.ax_in_buf else ctx.ax_block_out_ref_bufs[i - 1].?;

        var args = try exe.args(ctx.allocator);
        defer args.deinit(ctx.allocator);
        var results = try exe.results(ctx.allocator);
        defer results.deinit(ctx.allocator);

        args.set(.{
            ref_vx_in,
            ref_ax_in,
            ctx.video_timesteps_buf,
            ctx.audio_timesteps_buf,
            ctx.v_prompt_timestep_buf,
            ctx.a_prompt_timestep_buf,
            ctx.v_pe_cos_buf,
            ctx.v_pe_sin_buf,
            ctx.a_pe_cos_buf,
            ctx.a_pe_sin_buf,
            ctx.v_text_ctx_buf,
            ctx.a_text_ctx_buf,
            ctx.v_cross_ss_ts_buf,
            ctx.v_cross_gate_ts_buf,
            ctx.a_cross_ss_ts_buf,
            ctx.a_cross_gate_ts_buf,
            ctx.a2v_pe_cos_buf,
            ctx.a2v_pe_sin_buf,
            ctx.a2v_k_pe_cos_buf,
            ctx.a2v_k_pe_sin_buf,
            ctx.a2v_mask_block_bufs[i].?,
            ctx.v2a_pe_cos_buf,
            ctx.v2a_pe_sin_buf,
            ctx.v2a_k_pe_cos_buf,
            ctx.v2a_k_pe_sin_buf,
            ctx.v2a_mask_block_bufs[i].?,
            ctx.params_bufs.blocks[i],
        });
        exe.call(args, &results);

        const out = results.get(zml.Bufferized(model.Block0NativeAudioIntermediates));

        const norm_metrics = try check_utils.compareBuffers(ctx.io, out.norm_ax, ctx.norm_ax_block_bufs[i].?, 0.2, 0.01);
        std.log.info(
            "Audio block {d} native norm_ax: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, norm_metrics.close_fraction, norm_metrics.max_abs_error, norm_metrics.mean_abs_error },
        );

        const text_x_metrics = try check_utils.compareBuffers(ctx.io, out.a_text_x, ctx.a_text_x_block_bufs[i].?, 0.2, 0.01);
        std.log.info(
            "Audio block {d} native a_text_x: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, text_x_metrics.close_fraction, text_x_metrics.max_abs_error, text_x_metrics.mean_abs_error },
        );

        const v2a_x_metrics = try check_utils.compareBuffers(ctx.io, out.v2a_x, ctx.v2a_x_block_bufs[i].?, 0.2, 0.01);
        std.log.info(
            "Audio block {d} native v2a_x: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, v2a_x_metrics.close_fraction, v2a_x_metrics.max_abs_error, v2a_x_metrics.mean_abs_error },
        );

        const ax_scaled_metrics = try check_utils.compareBuffers(ctx.io, out.ax_scaled_ff, ctx.ax_scaled_block_bufs[i].?, 0.2, 0.01);
        std.log.info(
            "Audio block {d} native ax_scaled_ff: close_fraction={d:.8} max_abs={d:.6} mean_abs={d:.6}",
            .{ i, ax_scaled_metrics.close_fraction, ax_scaled_metrics.max_abs_error, ax_scaled_metrics.mean_abs_error },
        );
    }
}

fn logExtendedStats(io: std.Io, label: []const u8, computed: zml.Buffer, expected: zml.Buffer) !void {
    const m = try check_utils.compareBuffersExtended(io, computed, expected, 0.2, 0.01);
    std.log.info(
        "{s}: close={d:.8} max_abs={d:.6} mean_abs={d:.6} rmse={d:.6} rel_l2={d:.6} cos={d:.8}",
        .{ label, m.close_fraction, m.max_abs_error, m.mean_abs_error, m.rmse_error, m.rel_l2_error, m.cosine_similarity },
    );
    std.log.info(
        "{s}: abs_err quantiles p50={d:.6} p90={d:.6} p99={d:.6} p99.9={d:.6}",
        .{ label, m.abs_err_p50, m.abs_err_p90, m.abs_err_p99, m.abs_err_p999 },
    );
    std.log.info(
        "{s}: sign fractions pos={d:.6} neg={d:.6} zero={d:.6}",
        .{ label, m.positive_diff_fraction, m.negative_diff_fraction, m.zero_diff_fraction },
    );
    std.log.info(
        "{s}: frac(|err|>1e-3)={d:.6} frac(|err|>1e-2)={d:.6} frac(|err|>1e-1)={d:.6}",
        .{ label, m.frac_abs_err_gt_1e3, m.frac_abs_err_gt_1e2, m.frac_abs_err_gt_1e1 },
    );
}

fn runCheck(
    ctx: *CheckContext,
    use_audio_ff_residual_f32_experiment: bool,
    use_audio_all_residuals_f32_experiment: bool,
    use_extended_error_stats: bool,
) !void {
    std.log.info("Compiling native 8-block full graph...", .{});
    if (use_audio_all_residuals_f32_experiment) {
        std.log.info("Checker experiment enabled: all audio residual adds in f32", .{});
    } else if (use_audio_ff_residual_f32_experiment) {
        std.log.info("Checker experiment enabled: audio FF residual add in f32", .{});
    }
    var full_exe = if (ctx.v_text_ctx_mask_buf) |v_mask|
        if (ctx.a2v_masks_buf) |a2v_masks|
            if (use_audio_all_residuals_f32_experiment)
                try ctx.platform.compileFn(
                    ctx.allocator,
                    ctx.io,
                    model.forwardBlockSlice8NativeWithAllMasksAudioAllResidualsF32,
                    .{
                        zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                        zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                        zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                        zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                        zml.Tensor.fromShape(v_mask.shape()),
                        zml.Tensor.fromShape(ctx.a_text_ctx_mask_buf.?.shape()),
                        zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(a2v_masks.shape()),
                        zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_masks_buf.?.shape()),
                        ctx.params_shape,
                    },
                    .{ .shardings = &.{ctx.sharding} },
                )
            else if (use_audio_ff_residual_f32_experiment)
                try ctx.platform.compileFn(
                    ctx.allocator,
                    ctx.io,
                    model.forwardBlockSlice8NativeWithAllMasksAudioFFResidualF32,
                    .{
                        zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                        zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                        zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                        zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                        zml.Tensor.fromShape(v_mask.shape()),
                        zml.Tensor.fromShape(ctx.a_text_ctx_mask_buf.?.shape()),
                        zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(a2v_masks.shape()),
                        zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_masks_buf.?.shape()),
                        ctx.params_shape,
                    },
                    .{ .shardings = &.{ctx.sharding} },
                )
            else
                try ctx.platform.compileFn(
                    ctx.allocator,
                    ctx.io,
                    model.forwardBlockSlice8NativeWithAllMasks,
                    .{
                        zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                        zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                        zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                        zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                        zml.Tensor.fromShape(v_mask.shape()),
                        zml.Tensor.fromShape(ctx.a_text_ctx_mask_buf.?.shape()),
                        zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(a2v_masks.shape()),
                        zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                        zml.Tensor.fromShape(ctx.v2a_masks_buf.?.shape()),
                        ctx.params_shape,
                    },
                    .{ .shardings = &.{ctx.sharding} },
                )
        else
            try ctx.platform.compileFn(
                ctx.allocator,
                ctx.io,
                model.forwardBlockSlice8NativeWithTextMasks,
                .{
                    zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                    zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                    zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                    zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                    zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                    zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                    zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                    zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                    zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                    zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                    zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                    zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                    zml.Tensor.fromShape(v_mask.shape()),
                    zml.Tensor.fromShape(ctx.a_text_ctx_mask_buf.?.shape()),
                    zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                    zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                    zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                    zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                    zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                    zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                    zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                    zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                    zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                    zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                    zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                    zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                    ctx.params_shape,
                },
                .{ .shardings = &.{ctx.sharding} },
            )
    else if (ctx.a2v_masks_buf) |a2v_masks|
        try ctx.platform.compileFn(
            ctx.allocator,
            ctx.io,
            model.forwardBlockSlice8NativeWithAVMasks,
            .{
                zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(a2v_masks.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_masks_buf.?.shape()),
                ctx.params_shape,
            },
            .{ .shardings = &.{ctx.sharding} },
        )
    else
        try ctx.platform.compileFn(
            ctx.allocator,
            ctx.io,
            model.forwardBlockSlice8Native,
            .{
                zml.Tensor.fromShape(ctx.vx_in_buf.shape()),
                zml.Tensor.fromShape(ctx.ax_in_buf.shape()),
                zml.Tensor.fromShape(ctx.video_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.audio_timesteps_buf.shape()),
                zml.Tensor.fromShape(ctx.v_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.a_prompt_timestep_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.a_text_ctx_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.v_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_ss_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a_cross_gate_ts_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.a2v_k_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_pe_sin_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_cos_buf.shape()),
                zml.Tensor.fromShape(ctx.v2a_k_pe_sin_buf.shape()),
                ctx.params_shape,
            },
            .{ .shardings = &.{ctx.sharding} },
        );
    defer full_exe.deinit();

    var full_args = try full_exe.args(ctx.allocator);
    defer full_args.deinit(ctx.allocator);
    var full_res = try full_exe.results(ctx.allocator);
    defer full_res.deinit(ctx.allocator);

    if (ctx.v_text_ctx_mask_buf) |v_mask| {
        if (ctx.a2v_masks_buf) |a2v_masks| {
            full_args.set(.{
                ctx.vx_in_buf, ctx.ax_in_buf, ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                v_mask, ctx.a_text_ctx_mask_buf.?,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf, a2v_masks,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf, ctx.v2a_masks_buf.?,
                ctx.params_bufs,
            });
        } else {
            full_args.set(.{
                ctx.vx_in_buf, ctx.ax_in_buf, ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
                ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
                ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
                ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
                v_mask, ctx.a_text_ctx_mask_buf.?,
                ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
                ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf,
                ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf,
                ctx.params_bufs,
            });
        }
    } else if (ctx.a2v_masks_buf) |a2v_masks| {
        full_args.set(.{
            ctx.vx_in_buf, ctx.ax_in_buf, ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
            ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
            ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
            ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
            ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
            ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf, a2v_masks,
            ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf, ctx.v2a_masks_buf.?,
            ctx.params_bufs,
        });
    } else {
        full_args.set(.{
            ctx.vx_in_buf, ctx.ax_in_buf, ctx.video_timesteps_buf, ctx.audio_timesteps_buf,
            ctx.v_prompt_timestep_buf, ctx.a_prompt_timestep_buf,
            ctx.v_pe_cos_buf, ctx.v_pe_sin_buf, ctx.a_pe_cos_buf, ctx.a_pe_sin_buf,
            ctx.v_text_ctx_buf, ctx.a_text_ctx_buf,
            ctx.v_cross_ss_ts_buf, ctx.v_cross_gate_ts_buf, ctx.a_cross_ss_ts_buf, ctx.a_cross_gate_ts_buf,
            ctx.a2v_pe_cos_buf, ctx.a2v_pe_sin_buf, ctx.a2v_k_pe_cos_buf, ctx.a2v_k_pe_sin_buf,
            ctx.v2a_pe_cos_buf, ctx.v2a_pe_sin_buf, ctx.v2a_k_pe_cos_buf, ctx.v2a_k_pe_sin_buf,
            ctx.params_bufs,
        });
    }
    full_exe.call(full_args, &full_res);

    var out = full_res.get(zml.Bufferized(model.BasicAVTransformerBlock.FullOutputs));

    const vx_out_buf = out.vx_out;
    if (use_extended_error_stats) {
        try logExtendedStats(ctx.io, "Video final", vx_out_buf, ctx.vx_out_ref_buf);
    }
    try zml.testing.expectClose(ctx.io, vx_out_buf, ctx.vx_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("Native 8-block video parity PASSED", .{});

    try logAudioBlockLocalization(ctx, use_extended_error_stats);
    try logAudioBlockExactInputs(ctx, use_extended_error_stats);
    try logAudioBlock0StageDiagnostics(ctx);
    try logAudioNativeRecomputedIntermediates(ctx);

    const ax_out_buf = out.ax_out;
    if (use_extended_error_stats) {
        try logExtendedStats(ctx.io, "Audio final", ax_out_buf, ctx.ax_out_ref_buf);
    }
    try zml.testing.expectClose(ctx.io, ax_out_buf, ctx.ax_out_ref_buf, .{
        .absolute_tolerance = 0.2,
        .relative_tolerance = 0.01,
        .minimum_close_fraction = 0.999,
    });
    std.log.info("Native 8-block audio parity PASSED", .{});

    std.log.info("Native 8-block full stream parity PASSED", .{});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    std.log.info("Native block-slice (8) full stream parity checker", .{});

    var it = init.minimal.args.iterate();
    _ = it.next();

    const usage = "Usage: block_slice_native_check <checkpoint.safetensors> <fixture.safetensors> [--audio-ff-residual-f32] [--audio-all-residuals-f32] [--extended-error-stats]";

    const ckpt_path = it.next() orelse {
        std.log.err(usage, .{});
        return error.InvalidArgs;
    };
    const fixture_path = it.next() orelse {
        std.log.err(usage, .{});
        return error.InvalidArgs;
    };

    var use_audio_ff_residual_f32_experiment = false;
    var use_audio_all_residuals_f32_experiment = false;
    var use_extended_error_stats = false;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--audio-ff-residual-f32")) {
            use_audio_ff_residual_f32_experiment = true;
        } else if (std.mem.eql(u8, arg, "--audio-all-residuals-f32")) {
            use_audio_all_residuals_f32_experiment = true;
        } else if (std.mem.eql(u8, arg, "--extended-error-stats")) {
            use_extended_error_stats = true;
        } else {
            std.log.err("Unknown arg: {s}", .{arg});
            std.log.err(usage, .{});
            return error.InvalidArgs;
        }
    }

    if (use_audio_ff_residual_f32_experiment and use_audio_all_residuals_f32_experiment) {
        std.log.err("Use only one experiment flag at a time.", .{});
        return error.InvalidArgs;
    }

    var ctx = try CheckContext.init(allocator, io, ckpt_path, fixture_path);
    defer ctx.deinit();

    try runCheck(&ctx, use_audio_ff_residual_f32_experiment, use_audio_all_residuals_f32_experiment, use_extended_error_stats);
}

