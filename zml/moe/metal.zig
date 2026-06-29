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

fn validateOptions(opts: Options) !void {
    if (opts.expert_map != null) return error.InvalidShape;
    if (opts.w1_scale != null or opts.w2_scale != null) return error.UnsupportedQuantization;
    if (opts.w1_bias != null or opts.w2_bias != null) return error.UnsupportedBias;
}

fn validateInputs(hidden: Tensor, gate_up: Tensor, down: Tensor, weights: Tensor, ids: Tensor) !void {
    for ([_]Tensor{ hidden, gate_up, down, weights }) |t| {
        switch (t.dtype()) {
            .bf16, .f16, .f32 => {},
            else => return error.UnsupportedType,
        }
    }

    if (ids.dtype() != .i32) return error.UnsupportedType;
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
    try validateOptions(opts);
    try validateInputs(hidden_states, gate_up, down, topk_weights, topk_ids);

    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);
    const d = hidden_states.dim(.d);
    const num_tokens = b * s;
    const top_expert = topk_ids.dim(.top_expert);
    const num_routes = num_tokens * top_expert;

    const hidden = hidden_states.reshape(.{ .token = num_tokens, .d = d }).withTags(.{ .token, .d });
    const x_rows = hidden.insertAxes(.d, .{.top_expert})
        .broad(zml.Shape.init(.{ .token = num_tokens, .top_expert = top_expert, .d = d }, hidden.dtype()))
        .merge(.{ .r = .{ .token, .top_expert } });
    const expert_ids = topk_ids.reshape(.{ .token = num_tokens, .top_expert = top_expert }).withTags(.{ .token, .top_expert })
        .merge(.{ .r = .{ .token, .top_expert } }).convert(.i32);

    const gate_up_out = zml.ops.customCall(
        "__metal$moe_gemm",
        .{ x_rows, gate_up, expert_ids },
        .{zml.Shape.init(.{ .r = num_routes, .out = gate_up.dim(.dout) }, x_rows.dtype())},
        .{},
        .{ .has_side_effect = false },
    );
    const activated = applyActivation(gate_up_out, opts.activation);

    const down_out = zml.ops.customCall(
        "__metal$moe_gemm",
        .{ activated, down, expert_ids },
        .{zml.Shape.init(.{ .r = num_routes, .d = down.dim(.d) }, activated.dtype())},
        .{},
        .{ .has_side_effect = false },
    );

    const weights = topk_weights.reshape(.{ .token = num_tokens, .top_expert = top_expert }).withTags(.{ .token, .top_expert })
        .merge(.{ .r = .{ .token, .top_expert } });
    const weighted = down_out.mul(weights.convert(down_out.dtype()).broad(down_out.shape()));
    const combined = weighted.splitAxis(.r, .{ .token = num_tokens, .top_expert = top_expert })
        .sum(.top_expert).squeeze(.top_expert);

    return combined.reshape(.{ .b = b, .s = s, .d = down.dim(.d) });
}
