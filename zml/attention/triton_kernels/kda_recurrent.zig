const std = @import("std");

const zml = @import("../../zml.zig");
const tri = zml.kernel.triton;

/// Fused channel-wise Kimi KDA recurrence over a static sequence.
///
/// One program owns a value-row tile for one batch/head pair. The recurrent
/// state stays in FP32 for the complete sequence and is written once at the
/// end; each output row is reduced over the key channel in FP32. This layout
/// is race-free because value rows are independent in the delta update.
pub const Config = struct {
    batch: i32,
    sequence: i32,
    heads: i32,
    value_dim: i32,
    key_dim: i32,
    block_v: i32,
    block_k: i32,
    input_dtype: tri.DType,
    state_dtype: tri.DType = .f32,
    output_dtype: tri.DType = .f32,
};

pub const Kernel = tri.Kernel(Config, .{
    .name = "kimi_k3_kda_recurrent",
    .inputs = &.{ "q_ptr", "k_ptr", "v_ptr", "alpha_ptr", "beta_ptr", "state_ptr" },
    .outputs = &.{ "recurrent_output", "state_output" },
    .run = run,
});

fn run(b: *tri.Builder, cfg: Config) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .q_ptr = .{ .ptr = cfg.input_dtype },
        .k_ptr = .{ .ptr = cfg.input_dtype },
        .v_ptr = .{ .ptr = cfg.input_dtype },
        .alpha_ptr = .{ .ptr = cfg.input_dtype },
        .beta_ptr = .{ .ptr = cfg.input_dtype },
        .state_ptr = .{ .ptr = cfg.state_dtype },
        .recurrent_output = .{ .ptr = cfg.output_dtype },
        .state_output = .{ .ptr = cfg.state_dtype },
    });

    std.debug.assert(cfg.batch > 0 and cfg.sequence > 0 and cfg.heads > 0);
    std.debug.assert(cfg.value_dim > 0 and cfg.key_dim > 0);
    std.debug.assert(cfg.block_k >= cfg.key_dim);

    const value_tiles: i32 = @divTrunc(cfg.value_dim + cfg.block_v - 1, cfg.block_v);
    const pid = b.programId(.x);
    const value_tile = pid.rem(value_tiles);
    const batch_head = pid.div(value_tiles);
    const head = batch_head.rem(cfg.heads);
    const batch = batch_head.div(cfg.heads);

    const value_offsets = value_tile.mul(cfg.block_v).add(b.arange(0, cfg.block_v, .i32));
    const key_offsets = b.arange(0, cfg.block_k, .i32);
    const value_mask = value_offsets.lt(cfg.value_dim);
    const key_mask = key_offsets.lt(cfg.key_dim);
    const state_mask = b.mask2d(value_mask, key_mask, cfg.block_v, cfg.block_k);

    const state_base = batch.mul(cfg.heads).add(head).mul(cfg.value_dim).mul(cfg.key_dim);
    const state_offsets = state_base
        .add(value_offsets.expandDims(1).mul(cfg.key_dim))
        .add(key_offsets.expandDims(0));
    const initial_state = b.loadOpts(a.state_ptr.addPtr(state_offsets), .{
        .mask = state_mask,
        .other = b.zeros(&.{ cfg.block_v, cfg.block_k }, cfg.state_dtype),
    }).to(.f32);

    var loop = b.openFor(0, cfg.sequence, 1, .{initial_state});
    {
        const token = loop.iv;
        const token_head = batch.mul(cfg.sequence).add(token).mul(cfg.heads).add(head);
        const key_base = token_head.mul(cfg.key_dim);
        const value_base = token_head.mul(cfg.value_dim);

        const query = b.loadOpts(a.q_ptr.addPtr(key_base.add(key_offsets)), .{
            .mask = key_mask,
            .other = b.zeros(&.{cfg.block_k}, cfg.input_dtype),
        }).to(.f32);
        const key = b.loadOpts(a.k_ptr.addPtr(key_base.add(key_offsets)), .{
            .mask = key_mask,
            .other = b.zeros(&.{cfg.block_k}, cfg.input_dtype),
        }).to(.f32);
        const alpha = b.loadOpts(a.alpha_ptr.addPtr(key_base.add(key_offsets)), .{
            .mask = key_mask,
            .other = b.zeros(&.{cfg.block_k}, cfg.input_dtype),
        }).to(.f32);
        const value = b.loadOpts(a.v_ptr.addPtr(value_base.add(value_offsets)), .{
            .mask = value_mask,
            .other = b.zeros(&.{cfg.block_v}, cfg.input_dtype),
        }).to(.f32);
        const beta = b.load(a.beta_ptr.addPtr(token_head)).to(.f32);

        const decayed = loop.carried[0].mul(alpha.expandDims(0));
        const prediction = decayed.mul(key.expandDims(0)).sumOpts(.{ .axis = 1 });
        const error_value = value.sub(prediction).mul(beta);
        const next_state = decayed.add(error_value.expandDims(1).mul(key.expandDims(0)));
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(cfg.key_dim)));
        const output = next_state.mul(query.expandDims(0)).sumOpts(.{ .axis = 1 }).mul(scale);

        b.storeOpts(a.recurrent_output.addPtr(value_base.add(value_offsets)), output.to(cfg.output_dtype), .{
            .mask = value_mask,
        });
        loop.yield(.{next_state});
    }

    b.storeOpts(a.state_output.addPtr(state_offsets), loop.results[0].to(cfg.state_dtype), .{
        .mask = state_mask,
    });
}
