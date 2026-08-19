const zml = @import("../zml.zig");
const metal = @import("metal_attention.zig");

pub const Metadata = metal.Metadata;

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata) zml.Tensor {
    const qc = q.transpose(.{ .h, .q, .hd });
    const kc = k.transpose(.{ .h, .k, .hd });
    const vc = v.transpose(.{ .h, .k, .hd });
    const tok_i32 = token_index.convert(.i32);

    const attn = zml.ops.customCall("zml$flash_attn", .{ qc, kc, vc, tok_i32, metadata.num_tokens }, qc.shape(), .{}, .{ .has_side_effect = false });
    return attn.transpose(q.shape());
}
