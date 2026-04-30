const zml = @import("../zml.zig");
const stdx = zml.stdx;

const source = @embedFile("neuron_attention.py");

pub const Parameters = struct {
    fn entrypoint(_: Parameters) []const u8 {
        return "decode_tkg";
    }
};

/// Decode-only attention lowered to NKI custom calls.
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, parameters: Parameters) zml.Tensor {
    assertDecodeContract(q, k, v, token_index);

    const token_index_2d = token_index.broad(zml.Shape.init(.{ .row = 1, .col = 1 }, token_index.dtype()));

    return zml.ops.neuronNki(
        .{ q, k, v, token_index_2d },
        .{q.shape()},
        .{
            .name = "attention_decode",
            .entrypoint = parameters.entrypoint(),
            .source = source,
        },
    )[0].convert(q.dtype());
}

fn assertDecodeContract(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) void {
    stdx.debug.assert(q.rank() == 3 and k.rank() == 3 and v.rank() == 3, "neuron attention expects q/k/v rank-3 tensors, got {f}, {f}, {f}", .{ q, k, v });
    stdx.debug.assert(q.dtype() == .bf16 and k.dtype() == .bf16 and v.dtype() == .bf16, "neuron attention expects bf16 q/k/v, got {}, {}, {}", .{ q.dtype(), k.dtype(), v.dtype() });
    stdx.debug.assert(q.dim(.q) == 1, "neuron attention only supports decode q=1, got {f}", .{q});
    stdx.debug.assert(k.dim(.h) == v.dim(.h), "neuron attention expects k/v head counts to match, got {f} and {f}", .{ k, v });
    stdx.debug.assert(q.dim(.hd) == k.dim(.hd) and k.dim(.hd) == v.dim(.hd), "neuron attention expects q/k/v head dimensions to match, got {f}, {f}, {f}", .{ q, k, v });
    stdx.debug.assert(k.dim(.k) == v.dim(.k), "neuron attention expects k/v sequence lengths to match, got {f} and {f}", .{ k, v });
    stdx.debug.assert(@mod(q.dim(.h), k.dim(.h)) == 0, "neuron attention expects query heads to be divisible by KV heads, got {f} and {f}", .{ q, k });
    stdx.debug.assert(token_index.rank() == 0 and token_index.dtype() == .u32, "neuron attention expects scalar u32 token_index, got {f}", .{token_index});
}
