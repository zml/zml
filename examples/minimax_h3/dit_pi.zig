//! Pi-H3-specific DiT math. Based on LakonLab v0.3 (Hansheng Chen),
//! `minimax_h3.py`, `lora.py`, and `flow_euler_ode.py`; the pruned
//! conditioning is adapted upstream from MiniMax/Hugging Face (Apache-2.0).

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");
const ops = @import("ops.zig");

pub const first_replacement_block = 46;

pub const LoRA = struct {
    a: zml.nn.Linear,
    b: zml.nn.Linear,
    column_parallel: bool,

    pub fn init(store: zml.io.TensorStore.View, comptime column_parallel: bool) LoRA {
        const a_part = if (column_parallel) .replicated else .{ .d = .model, .dout = .replicated };
        const b_part = if (column_parallel) .{ .d = .replicated, .dout = .model } else .replicated;
        return .{
            .a = ops.linear(store, "lora_A.weight", null, a_part, .replicated),
            .b = ops.linear(store, "lora_B.weight", null, b_part, .replicated),
            .column_parallel = column_parallel,
        };
    }

    pub fn apply(self: LoRA, base: zml.Tensor, x: zml.Tensor) zml.Tensor {
        // InplaceLoraLinear: BF16 A(x), then base.addmm_(A(x), B.T).
        // For row-parallel inputs A contracts the sharded feature axis.
        // Its rank activation is replicated, never explicitly all-reduced
        // again after the B projection. Rank is not a tensor-parallel axis.
        const h = ops.bf16Linear(self.a, x).rename(.{ .dout = .d }).withPartitioning(.model, .{ .d = .replicated });
        const update = h.convert(.f32).dot(self.b.weight.convert(.f32), .d);
        const y = base.convert(.f32).add(update).convert(.bf16);
        return if (self.column_parallel)
            y.withPartitioning(.model, .{ .dout = .model })
        else
            y.withPartitioning(.model, .{ .dout = .replicated });
    }
};

/// Pi-H3 only: DWD decoder input, computed once after the final DiT block.
/// Replaces the video_out projection and the original video VAE for video.
///
///   temb[step, video_slot] ─► BF16 ─► SiLU ─► dwd_norm_out.linear ─► (scale, shift)
///   hidden[video rows] ─► normalize ─► ×(1+scale) + shift ─► [1, video_tokens, 5376]
pub const Features = struct {
    projection: zml.nn.Linear,
    first_video: i64,
    eps: f32,

    pub const Input = struct {
        model: Features,
        hidden: zml.Tensor,
        temb: zml.Tensor,
        step: zml.Tensor,
        timestep_indices: zml.Tensor,
    };
    pub const Result = struct { features: zml.Tensor };

    pub fn forward(input: Input) Result {
        const temb = input.temb.splitAxis(.n, .{ .t = .auto, .n = config.timestep_slot_count });
        // All video tokens use the same timestep slot, so read it at the first video row.
        const video_slot = input.timestep_indices.slice(.s, .single(input.model.first_video));
        // The FP32 embedder result is type_as(hidden_states) before ANY
        // modulation, including before the DWD-specific SiLU.
        const coordinates = temb.gather(.{ .t = input.step, .n = video_slot }, .{}).convert(.f32);
        const activated = coordinates.silu().convert(.bf16);
        const scale, const shift = ops.bf16Linear(input.model.projection, activated)
            .rename(.{ .dout = .d }).chunkExact(.d, 2);
        const video = input.hidden.slice(.s, .{ .start = input.model.first_video });
        const norm = zml.nn.normalizeVariance(video, input.model.eps);
        return .{ .features = norm.mul(scale.addConstant(1).broad(norm.shape())).add(shift.broad(norm.shape())).replicate() };
    }
};
