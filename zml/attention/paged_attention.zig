const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const flashattn = @import("flashattn.zig");

const PagedAttention = @This();

pub const Backend = enum {
    cuda_fa2,
    cuda_fa3,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .cuda => b: {
                const first_device = platform.devices[0].pjrt_device;

                if (zml.platform.cuda.tryGetComputeCapabilities(platform, first_device)) |cc| {
                    if (std.mem.eql(u8, cc, "9.0")) {
                        break :b .cuda_fa3;
                    }
                }

                break :b .cuda_fa2;
            },
            else => stdx.debug.panic("Paged attention is not supported on {s} yet", .{@tagName(platform.target)}),
        };
    }
};

pub const Options = union(Backend) {
    cuda_fa2: flashattn.paged_fa2.Options,
    cuda_fa3: flashattn.paged_fa3.Options,

    pub fn fromBackend(backend: Backend, is_prefill: bool, batch_size: u32, seq_len: u32, page_chunk_size: u32, max_token_count: u32, num_attention_heads: u32, head_dim: u32) Options {
        return switch (backend) {
            .cuda_fa2 => if (is_prefill) .{
                .cuda_fa2 = .{
                    .mixed = .{
                        .batch_size_decode = batch_size,
                        .batch_size_prefill = batch_size,
                        .max_num_pages = seq_len / page_chunk_size,
                        .max_seqlen_k = seq_len,
                        .max_token_count = max_token_count,
                        .num_heads = num_attention_heads,
                        .head_dim = head_dim,
                    },
                },
            } else .{
                .cuda_fa2 = .{
                    .decode = .{
                        .batch_size = batch_size,
                        .max_num_pages = seq_len / page_chunk_size,
                        .max_seqlen_k = seq_len,
                        .max_token_count = max_token_count,
                        .num_heads = num_attention_heads,
                        .head_dim = head_dim,
                    },
                },
            },
            .cuda_fa3 => if (is_prefill) .{
                .cuda_fa3 = .{
                    .mixed = .{
                        .batch_size_decode = batch_size,
                        .batch_size_prefill = batch_size,
                        .max_num_pages = seq_len / page_chunk_size,
                        .max_seqlen_k = seq_len,
                        .max_token_count = max_token_count,
                        .num_heads = num_attention_heads,
                        .head_dim = head_dim,
                    },
                },
            } else .{
                .cuda_fa3 = .{
                    .decode = .{
                        .batch_size = batch_size,
                        .max_num_pages = seq_len / page_chunk_size,
                        .max_seqlen_k = seq_len,
                        .max_token_count = max_token_count,
                        .num_heads = num_attention_heads,
                        .head_dim = head_dim,
                    },
                },
            },
        };
    }

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

    pub fn allocationSize(self: Parameters) usize {
        return switch (self) {
            inline else => |v| v.allocationSize(),
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

pub const AttentionOptions = struct {
    is_causal: bool = true,
    sliding_window: i32 = -1,
};

pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    _ = k; // autofix
    _ = v; // autofix
    return switch (parameters) {
        .cuda_fa2 => |cuda_fa2_parameters| flashattn.paged_fa2.pagedAttention(cuda_fa2_parameters, context.cuda_fa2, q, k_cache, v_cache, layer_index, opts),
        .cuda_fa3 => |cuda_fa3_parameters| flashattn.paged_fa3.pagedAttention(cuda_fa3_parameters, context.cuda_fa3, q, k_cache, v_cache, layer_index, opts),
    };
}
