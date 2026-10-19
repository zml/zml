const zml = @import("zml");

pub fn select_modulation_rows(x: zml.Tensor, mask: zml.Tensor) zml.Tensor {
    const broad = x.withTags(.{ .mod, .d }).broad(.init(.{ .x = 1, .mod = 2, .d = x.dim(1) }, x.dtype()));
    const real, const zero = broad.chunkExact(.mod, 2);

    const tagged = mask.withTags(.{.d});
    const mask_ = tagged.broad(.init(.{ .x = 1, .d = tagged.dim(.d), .mod = 1 }, .bool))
        .broad(.init(.{ .x = 1, .d = tagged.dim(.d), .mod = real.dim(.d) }, .bool));

    // Condition (mask) : (1, 16415,    1)  -> Expands to (1, 16415, 4096)
    // Input 1 (real)   : (1,     1, 4096)  -> Expands to (1, 16415, 4096)
    // Input 2 (zero)   : (1,     1, 4096)  -> Expands to (1, 16415, 4096)
    return mask_.select(
        real.broad(.init(.{ .x = 1, .d = mask_.dim(.d), .mod = real.dim(.d) }, real.dtype())),
        zero.broad(.init(.{ .x = 1, .d = mask_.dim(.d), .mod = real.dim(.d) }, real.dtype())),
    );
}
