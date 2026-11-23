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
    /// One logit per query head, `[h]`, or per query token and head, `[b, h]`
    /// (per-token sinks: Triton only).
    sink: ?Tensor = null,
    /// `[b, h, distance]` logit bias added to the key at `distance` positions
    /// before the query, for distances below the last axis (Triton only).
    rel_bias: ?Tensor = null,
};
