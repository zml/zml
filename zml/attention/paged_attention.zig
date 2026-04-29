const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const flashattn = @import("flashattn.zig");
const tpu = @import("tpu_attention.zig");
const triton = @import("triton.zig");

const PagedAttention = @This();

pub const Backend = enum {
    cuda_fa2,
    cuda_fa3,
    triton,
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
            .rocm => .triton,
            .tpu => .mosaic_tpu,
            else => stdx.debug.panic("Paged attention is not supported on {s} yet", .{@tagName(platform.target)}),
        };
    }
};

pub const Options = union(Backend) {
    cuda_fa2: flashattn.paged_fa2.Options,
    cuda_fa3: flashattn.paged_fa3.Options,
    triton: triton.paged.Options,
    mosaic_tpu: tpu.mosaic_tpu.Options,

    const Args = struct {
        backend: Backend,
        is_prefill: bool,
        batch_size: u32,
        seq_len: u32,
        page_chunk_size: u32,
        max_token_count: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        max_seqlen_q: u32,
    };

    pub fn fromBackend(args: Args) Options {
        const max_num_pages = std.math.divCeil(u32, args.seq_len, args.page_chunk_size) catch unreachable;
        return switch (args.backend) {
            .cuda_fa2 => if (args.is_prefill) .{
                .cuda_fa2 = .{
                    .mixed = .{
                        .batch_size_decode = args.batch_size,
                        .batch_size_prefill = args.batch_size,
                        .max_num_pages = max_num_pages,
                        .max_seqlen_k = args.seq_len,
                        .max_token_count = args.max_token_count,
                        .num_heads = args.num_heads,
                        .num_kv_heads = args.num_kv_heads,
                        .head_dim = args.head_dim,
                    },
                },
            } else .{
                .cuda_fa2 = .{
                    .decode = .{
                        .batch_size = args.batch_size,
                        .max_num_pages = max_num_pages,
                        .max_seqlen_k = args.seq_len,
                        .max_token_count = args.max_token_count,
                        .num_heads = args.num_heads,
                        .num_kv_heads = args.num_kv_heads,
                        .head_dim = args.head_dim,
                    },
                },
            },
            .cuda_fa3 => if (args.is_prefill) .{
                .cuda_fa3 = .{
                    .mixed = .{
                        .batch_size_decode = args.batch_size,
                        .batch_size_prefill = args.batch_size,
                        .max_num_pages = max_num_pages,
                        .max_seqlen_k = args.seq_len,
                        .max_token_count = args.max_token_count,
                        .num_heads = args.num_heads,
                        .num_kv_heads = args.num_kv_heads,
                        .head_dim = args.head_dim,
                    },
                },
            } else .{
                .cuda_fa3 = .{
                    .decode = .{
                        .batch_size = args.batch_size,
                        .max_num_pages = max_num_pages,
                        .max_seqlen_k = args.seq_len,
                        .max_token_count = args.max_token_count,
                        .num_heads = args.num_heads,
                        .num_kv_heads = args.num_kv_heads,
                        .head_dim = args.head_dim,
                    },
                },
            },
            .triton => .{
                .triton = .{
                    .batch_size = args.batch_size,
                    .max_num_pages = max_num_pages,
                    .max_seqlen_q = args.max_seqlen_q,
                    .is_prefill = args.is_prefill,
                },
            },
            .mosaic_tpu => .{
                .mosaic_tpu = .{
                    .is_prefill = args.is_prefill,
                    .batch_size = args.batch_size,
                    .max_num_pages = max_num_pages,
                    .max_seqlen_k = args.seq_len,
                    .max_token_count = args.max_token_count,
                    .num_heads = args.num_heads,
                    .num_kv_heads = args.num_kv_heads,
                    .head_dim = args.head_dim,
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
    triton: triton.paged.Parameters,
    mosaic_tpu: tpu.mosaic_tpu.Parameters,

    pub fn init(options_: Options) Parameters {
        return switch (options_) {
            .cuda_fa2 => |cuda_fa2_options| .{ .cuda_fa2 = flashattn.paged_fa2.Parameters.init(cuda_fa2_options) },
            .cuda_fa3 => |cuda_fa3_options| .{ .cuda_fa3 = flashattn.paged_fa3.Parameters.init(cuda_fa3_options) },
            .triton => |triton_options| .{ .triton = triton.paged.Parameters.init(triton_options) },
            .mosaic_tpu => |mosaic_tpu_options| .{ .mosaic_tpu = tpu.mosaic_tpu.Parameters.init(mosaic_tpu_options) },
        };
    }

    pub fn options(self: Parameters) Options {
        return switch (self) {
            .cuda_fa2 => |v| .{ .cuda_fa2 = v.options() },
            .cuda_fa3 => |v| .{ .cuda_fa3 = v.options() },
            .triton => |v| .{ .triton = v.options() },
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
    triton: triton.paged.Context,
    mosaic_tpu: tpu.mosaic_tpu.Context,

    pub fn init(parameters: Parameters, num_heads: i64, num_kv_heads: i64, head_dim: i64, page_size: i64) Context {
        return switch (parameters) {
            .cuda_fa2 => |cuda_fa2_parameters| .{ .cuda_fa2 = flashattn.paged_fa2.Context.init(cuda_fa2_parameters, num_heads, num_kv_heads, head_dim, page_size) },
            .cuda_fa3 => |cuda_fa3_parameters| .{ .cuda_fa3 = flashattn.paged_fa3.Context.init(cuda_fa3_parameters, num_heads, num_kv_heads, head_dim, page_size) },
            .triton => |triton_parameters| .{ .triton = triton.paged.Context.init(triton_parameters, num_heads, num_kv_heads, head_dim, page_size) },
            .mosaic_tpu => |mosaic_tpu_parameters| .{ .mosaic_tpu = tpu.mosaic_tpu.Context.init(mosaic_tpu_parameters, num_heads, num_kv_heads, head_dim, page_size) },
        };
    }
};

pub const AttentionOptions = struct {
    is_causal: bool = true,
    sliding_window: i32 = -1,
    scale: ?f32 = null,
};

pub const KvCache = union(enum) {
    split: struct {
        k: zml.Tensor,
        v: zml.Tensor,
    },
    dense: zml.Tensor,
};

pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, kv_cache: KvCache, opts: AttentionOptions) zml.Tensor {
    _ = k;
    _ = v;
    return switch (parameters) {
        .cuda_fa2 => |cuda_fa2_parameters| switch (kv_cache) {
            .split => |split| flashattn.paged_fa2.pagedAttention(cuda_fa2_parameters, context.cuda_fa2, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
        },
        .cuda_fa3 => |cuda_fa3_parameters| switch (kv_cache) {
            .split => |split| flashattn.paged_fa3.pagedAttention(cuda_fa3_parameters, context.cuda_fa3, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
        },
        .triton => |triton_parameters| switch (kv_cache) {
            .split => |split| triton.paged.pagedAttention(triton_parameters, context.triton, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
        },
        .mosaic_tpu => |mosaic_tpu_parameters| tpu.mosaic_tpu.pagedAttention(mosaic_tpu_parameters, context.mosaic_tpu, q, kv_cache.dense, opts),
    };
}

test "Options.fromBackend configures mosaic_tpu ragged backend" {
    const options = Options.fromBackend(.{
        .backend = .mosaic_tpu,
        .is_prefill = false,
        .batch_size = 4,
        .seq_len = 512,
        .page_chunk_size = 16,
        .max_token_count = 4,
        .num_heads = 16,
        .num_kv_heads = 4,
        .head_dim = 128,
        .max_seqlen_q = 1,
    });

    switch (options) {
        .mosaic_tpu => |tpu_option| {
            try std.testing.expect(!tpu_option.is_prefill);
            try std.testing.expectEqual(@as(usize, 4), tpu_option.batch_size);
            try std.testing.expectEqual(@as(usize, 32), tpu_option.max_num_pages);
            try std.testing.expectEqual(@as(usize, 512), tpu_option.max_seqlen_k);
            try std.testing.expectEqual(@as(usize, 4), tpu_option.max_token_count);
            try std.testing.expectEqual(@as(usize, 16), tpu_option.num_heads);
            try std.testing.expectEqual(@as(usize, 4), tpu_option.num_kv_heads);
            try std.testing.expectEqual(@as(usize, 128), tpu_option.head_dim);
        },
        else => return error.UnexpectedBackendVariant,
    }
}

test "Backend.auto selects mosaic_tpu on TPU" {
    const platform: zml.Platform = .{
        .arena_state = .{},
        .target = .tpu,
        .pjrt_api = undefined,
        .pjrt_client = undefined,
        .devices = &[_]zml.platform.Device{},
        .memories = &[_]zml.platform.Memory{},
        .physical_mesh = undefined,
        .triton_runtime = null,
        .tpu_ir_runtime = null,
    };

    try std.testing.expectEqual(Backend.mosaic_tpu, Backend.auto(&platform));
}
