//! qaic: the graph plugin's MoE custom call `qaic$moe`. Like the GPU backends' grouped GEMMs it is sparse: given each
//! token's top-k expert ids and routing weights, its kernels read and run the routed experts' weights only. The experts
//! are split on their intermediate axis over the devices (each holds a slice of every expert, which balances the load
//! whatever the routing: `ExpertsLayout.sharding = .intermediate`, gate and up rows interleaved so that a device's slice
//! pairs them), or sharded whole (expert parallel, unquantized only); the devices' partial sums meet in one all-reduce.
const stdx = @import("stdx");

const zml = @import("../zml.zig");

const Tensor = zml.Tensor;

pub fn validateOptions(opts: zml.moe.Options) void {
    stdx.debug.assert(!opts.quantize_input, "qaic MoE backend does not quantize its input", .{});
    stdx.debug.assert(opts.activation == .swiglu, "qaic MoE backend supports SwiGLU only, got {}", .{opts.activation});
    stdx.debug.assert(opts.activation.swiglu.scale == null and opts.activation.swiglu.bias == null, "qaic MoE backend supports SwiGLU with a limit only", .{});
}

/// input [b, s, d], ids and weights [b, s, topk]; gate_up [expert, out (gate and up rows interleaved), in] (the
/// `ExpertsLayout` of `.qaic`) or the halves of a fused checkpoint tensor [expert, 2, out, in], down [expert, out, in];
/// unquantized or MXFP4 (u8 pairs, e8m0 scales per 32). The routing weights may be applied before down or after it (it
/// is linear).
pub fn fusedExperts(input: Tensor, ids: Tensor, weights: Tensor, gate_up: zml.nn.Linear, down: zml.nn.Linear, opts: zml.moe.Options) Tensor {
    validateOptions(opts);
    stdx.debug.assert(gate_up.bias == null and down.bias == null, "qaic MoE backend has no expert biases", .{});
    const limit = opts.activation.swiglu.limit;
    const expert_parallel = gate_up.weight.shape().partition(0).isSharded();
    const halves = gate_up.weight.rank() == 4;
    // (the activations replicated on the experts' mesh, the result back on the input's)
    const mesh = gate_up.weight.shape()._sharding.mesh;
    const context: Context = .{
        .input = replicatedOn(input, mesh),
        .ids = replicatedOn(ids, mesh),
        .weights = replicatedOn(weights, mesh),
        .gate_up = gate_up.weight,
        .gate_up_scales = if (gate_up.quantization) |q| q.scales else null,
        .down = down.weight,
        .down_scales = if (down.quantization) |q| q.scales else null,
        .expert_parallel = expert_parallel,
        .limit_milli = if (limit) |l| @intFromFloat(@round(l * 1000)) else 0,
        .sharded = expert_parallel or gate_up.weight.shape().partition(@as(u3, if (halves) 2 else 1)).isSharded(),
    };
    if (context.gate_up_scales != null) {
        const scheme = gate_up.quantization.?.scheme;
        stdx.debug.assert(scheme == .mxfp4 and down.quantization.?.scheme == .mxfp4, "qaic MoE backend supports MXFP4 experts only, got {}", .{scheme});
        stdx.debug.assert(!expert_parallel, "qaic MXFP4 experts are split on their intermediate axis, not sharded whole", .{});
    } else {
        stdx.debug.assert(limit == null, "qaic MoE backend clamps SwiGLU for MXFP4 experts only", .{});
    }
    const out = if (context.sharded) zml.ops.manualComputation(Context.body, context, context.input.shape()) else context.body(input.shape());
    return if (input.shape()._sharding.mesh) |m| replicatedOn(out, m) else out;
}

fn replicatedOn(x: Tensor, mesh: ?*const zml.Sharding.Mesh) Tensor {
    return x.withPartitioning(mesh orelse return x, zml.Sharding.Partitioning.replicated(x.rank()));
}

const Context = struct {
    input: Tensor,
    ids: Tensor,
    weights: Tensor,
    gate_up: Tensor,
    gate_up_scales: ?Tensor,
    down: Tensor,
    down_scales: ?Tensor,
    expert_parallel: bool,
    limit_milli: i64,
    sharded: bool,

    fn body(self: Context, _: zml.Shape) Tensor {
        const tokens = self.input.dim(.b) * self.input.dim(.s);
        const topk = self.ids.dim(.topk);
        const x = self.input.reshape(.{ .t = tokens, .d = self.input.dim(.d) });
        const ids = self.ids.convert(.i32).reshape(.{ .t = tokens, .topk = topk });
        const w = self.weights.convert(.f32).reshape(.{ .t = tokens, .topk = topk });
        // (the halves: this device's part of each, as halves of its own)
        const s = self.gate_up.shape();
        const gate_up = if (s.rank() == 4) self.gate_up.reshape(.{ s.dim(0), s.dim(1) * s.dim(2), s.dim(3) }) else self.gate_up;
        const config = .{
            .gate_up_interleaved = @as(i64, @intFromBool(s.rank() == 3)),
            .expert_parallel = @as(i64, @intFromBool(self.expert_parallel)),
            .swiglu_limit_milli = self.limit_milli,
        };
        const y = if (self.gate_up_scales) |gate_up_scales|
            zml.ops.customCall(
                "qaic$moe",
                .{ x, ids, w, gate_up.bitCast(.u8), gate_up_scales.bitCast(.u8), self.down.bitCast(.u8), self.down_scales.?.bitCast(.u8) },
                x.shape(),
                config,
                .{ .has_side_effect = false },
            )
        else
            zml.ops.customCall(
                "qaic$moe",
                .{ x, ids, w, gate_up.convert(x.dtype()), self.down.convert(x.dtype()) },
                x.shape(),
                config,
                .{ .has_side_effect = false },
            );
        const out = y.reshape(self.input.shape());
        return if (self.sharded) zml.ops.allReduce(out, Tensor.add) else out;
    }
};
