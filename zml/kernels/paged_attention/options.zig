const Tensor = @import("zml/core").Tensor;

pub const AttentionMask = union(enum) {
    standard: struct {
        /// Restrict attention to keys at or before the query position.
        causal: bool = true,
        /// Optional left window size, including the current token.
        /// Non-causal attention also includes every later valid key.
        sliding_window: ?u32 = null,
    },
    /// Attend to keys at specific positions.
    indices: Tensor,

    /// Attend to every valid key in the sequence.
    pub const none: AttentionMask = .{ .standard = .{ .causal = false } };

    pub fn isCausal(mask: AttentionMask) bool {
        return switch (mask) {
            .standard => |options| options.causal,
            .indices => false,
        };
    }

    pub fn slidingWindow(mask: AttentionMask) ?u32 {
        return switch (mask) {
            .standard => |options| options.sliding_window,
            .indices => null,
        };
    }
};

pub const AttentionOptions = struct {
    mask: AttentionMask = .{ .standard = .{} },
    scale: ?f32 = null,
    sink: ?Tensor = null,
};
