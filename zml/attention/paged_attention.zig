const std = @import("std");

const zml = @import("../zml.zig");
const flashattn = @import("flashattn.zig");

const PagedAttention = @This();

pub const Backend = enum {
    cuda_fa2,
    cuda_fa3,
};

pub const Options = union(Backend) {
    cuda_fa2: flashattn.paged_fa2.Options,
    cuda_fa3: flashattn.paged_fa3.Options,

    pub fn isPrefill(self: Options) bool {
        return switch (self) {
            inline else => |v| v.isPrefill(),
        };
    }
};

pub const Parameters = union(Backend) {
    cuda_fa2: flashattn.paged_fa2.Parameters,
    cuda_fa3: flashattn.paged_fa3.Parameters,

    pub fn init(options_: Options) Parameters {
        return switch (options_) {
            .cuda_fa2 => |cuda_fa2_options| .{ .cuda_fa2 = flashattn.paged_fa2.Parameters.init(cuda_fa2_options) },
            .cuda_fa3 => |cuda_fa3_options| .{ .cuda_fa3 = flashattn.paged_fa3.Parameters.init(cuda_fa3_options) },
        };
    }

    pub fn options(self: Parameters) Options {
        return switch (self) {
            .cuda_fa2 => |v| .{ .cuda_fa2 = v.options() },
            .cuda_fa3 => |v| .{ .cuda_fa3 = v.options() },
        };
    }
};

/// Internal state that can be used inside the model code. It's derived from Parameters and Option.
pub const Context = union(Backend) {
    cuda_fa2: flashattn.paged_fa2.Context,
    cuda_fa3: flashattn.paged_fa3.Context,

    pub fn init(parameters: Parameters, num_heads: i64, num_kv_heads: i64, head_dim: i64, page_size: i64) Context {
        return switch (parameters) {
            .cuda_fa2 => |cuda_fa2_parameters| .{ .cuda_fa2 = flashattn.paged_fa2.Context.init(cuda_fa2_parameters, num_heads, num_kv_heads, head_dim, page_size) },
            .cuda_fa3 => |cuda_fa3_parameters| .{ .cuda_fa3 = flashattn.paged_fa3.Context.init(cuda_fa3_parameters, num_heads, num_kv_heads, head_dim, page_size) },
        };
    }
};

pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index: zml.Tensor) zml.Tensor {
    _ = k; // autofix
    _ = v; // autofix
    return switch (parameters) {
        .cuda_fa2 => |cuda_fa2_parameters| flashattn.paged_fa2.pagedAttention(cuda_fa2_parameters, context.cuda_fa2, q, k_cache, v_cache, layer_index),
        .cuda_fa3 => |cuda_fa3_parameters| flashattn.paged_fa3.pagedAttention(cuda_fa3_parameters, context.cuda_fa3, q, k_cache, v_cache, layer_index),
    };
}
