const zml = @import("../../../zml.zig");
const kernel = @import("attention.zig");
const Tensor = zml.Tensor;

test {
    _ = @import("tests.zig");
}

/// Dense noncausal BF16 attention with head widths 32, 64 or 128.
/// Other shared-API cases retain SDPA semantics.
pub fn attention(q: Tensor, k: Tensor, v: Tensor, token_index: Tensor, causal: bool) Tensor {
    const supportedWidth = switch (q.dim(.hd)) {
        32, 64, 128 => true,
        else => false,
    };
    const supportedDtype = q.dtype() == .bf16;
    if (!supportedWidth or !supportedDtype or causal or k.dtype() != q.dtype() or v.dtype() != q.dtype() or q.dim(.h) != k.dim(.h)) {
        const mask: ?Tensor = if (causal)
            zml.nn.causalAttnMask(.{ .q = k.dim(.k), .k = k.dim(.k) }, q.dtype(), null)
                .gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, q.dtype()), token_index.appendAxes(.{.coord}), .{})
        else
            null;
        return zml.nn.sdpa(q, k, v, .{ .attn_mask = mask });
    }
    const batched = q.shape().hasTag(.b) != null;
    const rank: usize = if (batched) 4 else 3; // Why ?
    zml.stdx.debug.assert(q.rank() == rank and k.rank() == rank and v.rank() == rank and
        q.dim(.hd) == k.dim(.hd) and k.shape().eqlWithTags(v.shape()) and
        q.dim(.q) > 0 and k.dim(.k) > 0 and q.dim(.hd) > 0, "TCL dense attention requires matching nonempty Q/K/V shapes", .{});
    if (batched) zml.stdx.debug.assert(q.dim(.b) == k.dim(.b), "TCL attention batch sizes must match", .{});

    const qb = if (batched) q.transpose(.{ .b, .q, .h, .hd }) else q.transpose(.{ .q, .h, .hd }).insertAxes(0, .{.b});
    const kb = if (batched) k.transpose(.{ .b, .k, .h, .hd }) else k.transpose(.{ .k, .h, .hd }).insertAxes(0, .{.b});
    const vb = if (batched) v.transpose(.{ .b, .k, .h, .hd }) else v.transpose(.{ .k, .h, .hd }).insertAxes(0, .{.b});
    const out = kernel.attention(qb, kb, vb);
    return (if (batched) out else out.squeeze(.b)).transpose(q.shape());
}
