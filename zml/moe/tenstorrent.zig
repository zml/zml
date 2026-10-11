//! The Tenstorrent MoE backend: the experts on tt-metal's sparse matmul
//! (`tt_kernels/sparse_matmul.zig`), as Tenstorrent's gpt-oss and Gemma 4
//! run them. Tokens go in groups of 32, a group through the experts any of
//! its tokens is routed to; the routing weights scale the experts' outputs
//! after `down`.

const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const Tensor = zml.Tensor;
pub const sparse_matmul = @import("tt_kernels/sparse_matmul.zig");

/// Tokens per group: a tile of rows.
const group = 32;

/// Dense routing `[t, expert]` (each token's weight per expert, padded to
/// whole groups) and the experts each group of tokens runs (`[group,
/// expert]`, 1 or 0).
pub fn routing(topk_ids: Tensor, topk_weights: Tensor, experts: i64) struct { Tensor, Tensor } {
    const tokens = topk_ids.dim(.b) * topk_ids.dim(.s);
    const groups = std.math.divCeil(i64, tokens, group) catch unreachable;
    const pad: Tensor.Pad = .{ .high = groups * group - tokens };
    const ids = topk_ids.reshape(.{ .t = tokens, .topk = topk_ids.dim(.topk) });
    const routes: zml.Shape = .init(.{ .t = tokens, .topk = ids.dim(.topk), .expert = experts }, .i32);
    const routed = ids.convert(.i32).insertAxes(.last, .{.expert}).broad(routes).cmp(.EQ, Tensor.iota(routes, .expert))
        .convert(.f32);
    const weights = topk_weights.reshape(.{ .t = tokens, .topk = ids.dim(.topk) }).convert(.f32);
    const dense = routed.mul(weights.insertAxes(.last, .{.expert}).broad(routes.withDtype(.f32))).sum(.topk).squeeze(.topk)
        .pad(0, .{ .t = pad });
    const sparsity = routed.max(.topk).squeeze(.topk).pad(0, .{ .t = pad })
        .reshape(.{ .g = groups, .tok = group, .expert = experts }).max(.tok).squeeze(.tok);
    return .{ dense, sparsity };
}

pub fn fusedExperts(input: Tensor, topk_ids: Tensor, topk_weights: Tensor, gate_up: zml.nn.Linear, down: zml.nn.Linear, opts: zml.moe.Options) Tensor {
    stdx.debug.assert(gate_up.quantization == null and down.quantization == null, "tenstorrent moe backend expects unquantized expert weights", .{});
    stdx.debug.assert(gate_up.bias == null and down.bias == null, "tenstorrent moe backend doesn't support expert biases", .{});
    stdx.debug.assert(opts.routing_weight_placement == .after_down, "tenstorrent moe backend applies the routing weights after down", .{});

    const b = input.dim(.b);
    const s = input.dim(.s);
    const d = input.dim(.d);
    const experts = gate_up.weight.dim(.expert);
    const tokens = b * s;
    const groups = std.math.divCeil(i64, tokens, group) catch unreachable;
    const padded = groups * group;
    const pad: Tensor.Pad = .{ .high = padded - tokens };
    const dense, const sparsity = routing(topk_ids, topk_weights, experts);

    const x = input.reshape(.{ .t = tokens, .d = d }).convert(.bf16).pad(0, .{ .t = pad })
        .reshape(.{ .g = groups, .tok = group, .d = d });
    // The experts' `[out, in]` weights, read transposed.
    const w1 = gate_up.weight.transpose(.{ .expert, .out, gate_up.tag }).convert(.bf16);
    const w2 = down.weight.transpose(.{ .expert, .out, down.tag }).convert(.bf16);

    const h = sparse_matmul.sparseB(x, w1, sparsity, .{}).withTags(.{ .g, .expert, .tok, .out });
    const act = zml.moe.applyActivation(h, opts.activation, .concatenated).convert(.bf16);
    const y = sparse_matmul.sparseA(act, w2, sparsity, .{}).withTags(.{ .g, .expert, .tok, .d });

    const scale = dense.reshape(.{ .g = groups, .tok = group, .expert = experts }).transpose(.{ .g, .expert, .tok });
    const out = y.convert(.f32).mul(scale.insertAxes(.last, .{.d}).broad(y.shape().withDtype(.f32))).sum(.expert).squeeze(.expert);
    return out.reshape(.{ .t = padded, .d = d }).slice(.t, .{ .end = tokens }).reshape(.{ .b = b, .s = s, .d = d }).convert(input.dtype());
}
