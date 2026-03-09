const std = @import("std");

const platforms = @import("platforms");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const flashattn = @import("flashattn.zig");

const PagedAttention = @This();

pub const Backend = enum {
    cuda_fa2,
    cuda_fa3,
    mosaic_tpu,

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
            .tpu => .mosaic_tpu,
            else => stdx.debug.panic("Paged attention is not supported on {s} yet", .{@tagName(platform.target)}),
        };
    }
};

pub const Options = union(Backend) {
    cuda_fa2: flashattn.paged_fa2.Options,
    cuda_fa3: flashattn.paged_fa3.Options,
    mosaic_tpu: flashattn.mosaic_tpu.Options,

    pub fn fromBackend(backend: Backend, is_prefill: bool, batch_size: u32, seq_len: u32, page_chunk_size: u32, max_token_count: u32, num_attention_heads: u32, head_dim: u32) Options {
        const max_num_pages = @divFloor(seq_len + page_chunk_size - 1, page_chunk_size); // @divCeil
        return switch (backend) {
            .cuda_fa2 => if (is_prefill) .{
                .cuda_fa2 = .{
                    .mixed = .{
                        .batch_size_decode = batch_size,
                        .batch_size_prefill = batch_size,
                        .max_num_pages = max_num_pages,
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
                        .max_num_pages = max_num_pages,
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
                        .max_num_pages = max_num_pages,
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
                        .max_num_pages = max_num_pages,
                        .max_seqlen_k = seq_len,
                        .max_token_count = max_token_count,
                        .num_heads = num_attention_heads,
                        .head_dim = head_dim,
                    },
                },
            },
            .mosaic_tpu => .{
                .mosaic_tpu = .{
                    .is_prefill = is_prefill,
                    .batch_size = batch_size,
                    .max_num_pages = seq_len / page_chunk_size,
                    .max_seqlen_k = seq_len,
                    .max_token_count = max_token_count,
                    .num_heads = num_attention_heads,
                    .head_dim = head_dim,
                },
            },
        };
    }

    pub fn isPrefill(self: Options) bool {
        return switch (self) {
            inline else => |v| v.isPrefill(),
        };
    }

    pub fn maxNumPages(self: Options) usize {
        return switch (self) {
            inline else => |v| v.maxNumPages(),
        };
    }
};

pub const Parameters = union(Backend) {
    cuda_fa2: flashattn.paged_fa2.Parameters,
    cuda_fa3: flashattn.paged_fa3.Parameters,
    mosaic_tpu: flashattn.mosaic_tpu.Parameters,

    pub fn init(options_: Options) Parameters {
        return switch (options_) {
            .cuda_fa2 => |cuda_fa2_options| .{ .cuda_fa2 = flashattn.paged_fa2.Parameters.init(cuda_fa2_options) },
            .cuda_fa3 => |cuda_fa3_options| .{ .cuda_fa3 = flashattn.paged_fa3.Parameters.init(cuda_fa3_options) },
            .mosaic_tpu => |tpu_options| .{ .mosaic_tpu = flashattn.mosaic_tpu.Parameters.init(tpu_options) },
        };
    }

    pub fn options(self: Parameters) Options {
        return switch (self) {
            .cuda_fa2 => |v| .{ .cuda_fa2 = v.options() },
            .cuda_fa3 => |v| .{ .cuda_fa3 = v.options() },
            .mosaic_tpu => |v| .{ .mosaic_tpu = v.options() },
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
    mosaic_tpu: flashattn.mosaic_tpu.Context,

    pub fn init(parameters: Parameters, num_heads: i64, num_kv_heads: i64, head_dim: i64, page_size: i64) Context {
        return switch (parameters) {
            .cuda_fa2 => |cuda_fa2_parameters| .{ .cuda_fa2 = flashattn.paged_fa2.Context.init(cuda_fa2_parameters, num_heads, num_kv_heads, head_dim, page_size) },
            .cuda_fa3 => |cuda_fa3_parameters| .{ .cuda_fa3 = flashattn.paged_fa3.Context.init(cuda_fa3_parameters, num_heads, num_kv_heads, head_dim, page_size) },
            .mosaic_tpu => |tpu_parameters| .{ .mosaic_tpu = flashattn.mosaic_tpu.Context.init(tpu_parameters, num_heads, num_kv_heads, head_dim, page_size) },
        };
    }
};

pub const AttentionOptions = struct {
    is_causal: bool = true,
    sliding_window: i32 = -1,
    token_pos: ?zml.Tensor = null,
};

pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    var threaded: std.Io.Threaded = .init_single_threaded;
    threaded.allocator = zml.module.CompilationContext.current().allocator;

    return switch (parameters) {
        .cuda_fa2 => |cuda_fa2_parameters| flashattn.paged_fa2.pagedAttention(cuda_fa2_parameters, context.cuda_fa2, q, k_cache, v_cache, layer_index, opts),
        .cuda_fa3 => |cuda_fa3_parameters| flashattn.paged_fa3.pagedAttention(cuda_fa3_parameters, context.cuda_fa3, q, k_cache, v_cache, layer_index, opts),
        .mosaic_tpu => |tpu_params| if (tpu_params.opts.is_prefill)
            flashattn.mosaic_tpu.prefillAttention(
                threaded.allocator,
                threaded.io(),
                q.rename(.{ .b = .q }),
                k.rename(.{ .b = .k }),
                v.rename(.{ .b = .k }),
            ).rename(.{ .q = .b })
        else
            flashattn.mosaic_tpu.pagedAttention(tpu_params, context.mosaic_tpu, threaded.allocator, threaded.io(), q, k_cache, v_cache, layer_index, opts.token_pos),
    };
}
