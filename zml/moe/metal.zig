const std = @import("std");

const zml = @import("../zml.zig");
const Tensor = zml.Tensor;

pub const ActivationMode = enum {
    silu,
    relu,
    gelu,
};

pub const Options = struct {
    activation: ActivationMode = .silu,
    global_num_experts: i64 = -1,
    expert_map: ?Tensor = null,
    w1_scale: ?Tensor = null,
    w2_scale: ?Tensor = null,
    w1_global_scale: ?Tensor = null,
    w2_global_scale: ?Tensor = null,
    w1_bias: ?Tensor = null,
    w2_bias: ?Tensor = null,
};

pub const Parameters = struct {
    num_experts_per_tok: u32,
    activation: ActivationMode,

    pub const InitOptions = struct {
        num_experts_per_tok: u32,
        activation: ActivationMode = .silu,
    };

    pub fn init(opts: InitOptions) Parameters {
        return .{
            .num_experts_per_tok = opts.num_experts_per_tok,
            .activation = opts.activation,
        };
    }
};

pub const Metadata = struct {
    pub const InitOptions = struct {};

    pub fn init(opts: InitOptions) Metadata {
        _ = opts;
        return .{};
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        _ = self;
        _ = io;
        _ = platform;
        return {};
    }
};

pub fn deinitBuffer(bufferized: *zml.Bufferized(Metadata)) void {
    _ = bufferized;
}

const QuantMode = enum { none, fp8, nvfp4 };

fn quantMode(gate_up: Tensor, down: Tensor) !QuantMode {
    if (gate_up.dtype() != down.dtype()) return error.UnsupportedType;
    return switch (gate_up.dtype()) {
        .bf16, .f16, .f32 => .none,
        .f8e4m3fn => .fp8,
        .u8, .f4e2m1 => .nvfp4,
        else => error.UnsupportedType,
    };
}

fn requireScalePair(opts: Options, dtype: zml.DataType) !void {
    const s1 = opts.w1_scale orelse return error.UnsupportedQuantization;
    const s2 = opts.w2_scale orelse return error.UnsupportedQuantization;
    if (s1.dtype() != dtype or s2.dtype() != dtype) return error.UnsupportedType;
}

fn applyActivation(x: Tensor, mode: ActivationMode) Tensor {
    const mid = @divFloor(x.dim(.out), 2);
    const gate = x.slice1d(.out, .{ .end = mid });
    const up = x.slice1d(.out, .{ .start = mid });

    const act = switch (mode) {
        .silu => gate.silu().mul(up),
        .gelu => gate.gelu().mul(up),
        .relu => gate.relu().mul(up),
    };

    return act.rename(.{ .out = .mid });
}

fn moeGemm(
    x_rows: Tensor,
    w: Tensor,
    scale: ?Tensor,
    expert_ids: Tensor,
    global_scale: ?Tensor,
    out_shape: zml.Shape,
    mode: QuantMode,
) Tensor {
    const side: zml.ops.CustomCallOptions = .{ .has_side_effect = false };
    return switch (mode) {
        .none => zml.ops.customCall("__metal$moe_gemm", .{ x_rows, w, expert_ids }, .{out_shape}, .{}, side),
        .fp8 => zml.ops.customCall("__metal$moe_gemm$f8", .{ x_rows, w, scale.?, expert_ids }, .{out_shape}, .{}, side),
        .nvfp4 => if (global_scale) |g|
            zml.ops.customCall("__metal$moe_gemm$f4", .{ x_rows, w, scale.?, expert_ids, g }, .{out_shape}, .{}, side)
        else
            zml.ops.customCall("__metal$moe_gemm$f4", .{ x_rows, w, scale.?, expert_ids }, .{out_shape}, .{}, side),
    };
}

pub fn fusedExpertsImpl(
    hidden_states: Tensor,
    gate_up: Tensor,
    down: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    metadata: Metadata,
    opts: Options,
) !Tensor {
    _ = metadata;
    if (opts.expert_map != null) return error.InvalidShape;
    if (opts.w1_bias != null or opts.w2_bias != null) return error.UnsupportedBias;

    const mode = try quantMode(gate_up, down);
    switch (mode) {
        .none => {
            if (opts.w1_scale != null or opts.w2_scale != null) return error.UnsupportedQuantization;
            if (opts.w1_global_scale != null or opts.w2_global_scale != null) return error.UnsupportedQuantization;
            switch (hidden_states.dtype()) {
                .bf16, .f16, .f32 => {},
                else => return error.UnsupportedType,
            }
        },
        .fp8 => {
            try requireScalePair(opts, .bf16);
            if (opts.w1_global_scale != null or opts.w2_global_scale != null) return error.UnsupportedQuantization;
            if (hidden_states.dtype() != .bf16) return error.UnsupportedType;
        },
        .nvfp4 => {
            try requireScalePair(opts, .f8e4m3fn);
            if (hidden_states.dtype() != .bf16) return error.UnsupportedType;
        },
    }

    switch (topk_weights.dtype()) {
        .bf16, .f16, .f32 => {},
        else => return error.UnsupportedType,
    }
    if (topk_ids.dtype() != .i32) return error.UnsupportedType;

    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);
    const d = hidden_states.dim(.d);
    const num_tokens = b * s;
    const top_expert = topk_ids.dim(.top_expert);
    const num_routes = num_tokens * top_expert;

    const act_dtype: zml.DataType = switch (mode) {
        .none => hidden_states.dtype(),
        .fp8, .nvfp4 => .bf16,
    };

    const hidden = hidden_states.reshape(.{ .token = num_tokens, .d = d }).withTags(.{ .token, .d });
    const x_rows = hidden.insertAxes(.d, .{.top_expert})
        .broad(zml.Shape.init(.{ .token = num_tokens, .top_expert = top_expert, .d = d }, act_dtype))
        .merge(.{ .r = .{ .token, .top_expert } });
    const expert_ids = topk_ids.reshape(.{ .token = num_tokens, .top_expert = top_expert }).withTags(.{ .token, .top_expert })
        .merge(.{ .r = .{ .token, .top_expert } }).convert(.i32);

    const gate_up_w = if (gate_up.dtype() == .u8) zml.ops.unpackNvfp4(gate_up, .d) else gate_up;
    const down_w = if (down.dtype() == .u8) zml.ops.unpackNvfp4(down, .dout) else down;

    const gate_up_out = moeGemm(
        x_rows,
        gate_up_w,
        opts.w1_scale,
        expert_ids,
        opts.w1_global_scale,
        zml.Shape.init(.{ .r = num_routes, .out = gate_up_w.dim(.dout) }, act_dtype),
        mode,
    );
    const activated = applyActivation(gate_up_out, opts.activation);

    const down_out = moeGemm(
        activated,
        down_w,
        opts.w2_scale,
        expert_ids,
        opts.w2_global_scale,
        zml.Shape.init(.{ .r = num_routes, .d = down_w.dim(.d) }, act_dtype),
        mode,
    );

    const weights = topk_weights.reshape(.{ .token = num_tokens, .top_expert = top_expert }).withTags(.{ .token, .top_expert })
        .merge(.{ .r = .{ .token, .top_expert } });
    const weighted = down_out.mul(weights.convert(down_out.dtype()).broad(down_out.shape()));
    const combined = weighted.splitAxis(.r, .{ .token = num_tokens, .top_expert = top_expert })
        .sum(.top_expert).squeeze(.top_expert);

    return combined.reshape(.{ .b = b, .s = s, .d = down.dim(.d) });
}
