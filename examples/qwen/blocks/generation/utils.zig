const zml = @import("zml");

pub fn select_modulation_rows(x: zml.Tensor, mask: zml.Tensor) zml.Tensor {
    const p = x.withTags(.{ .bs, .d });
    const m = mask.withTags(.{.seq_len});
    const batch_size = p.dim(.bs) - 1;

    const out_shape: zml.Shape = .init(.{ .bs = batch_size, .seq_len = m.dim(.seq_len), .d = p.dim(.d) }, p.dtype());
    const real = p.slice(.bs, .{ .end = batch_size });
    const zero = p.slice(.bs, .{ .start = batch_size }).squeeze(.bs);

    return m.broad(out_shape.withDtype(.bool)).select(real.broad(out_shape), zero.broad(out_shape));
}
