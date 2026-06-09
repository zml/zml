const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const flashattn = @import("flashattn.zig");
const tpu = @import("tpu_attention.zig");
const triton = @import("triton_attention.zig");

const PagedAttention = @This();

pub const Backend = enum {
    cuda_fa2,
    cuda_fa3,
    triton,
    mosaic_tpu,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .cuda => .triton,
            .rocm => .triton,
            .oneapi => .triton,
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
        max_num_pages: u32,
        max_token_count: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        max_seqlen_q: u32,
    };

    pub fn fromBackend(args: Args) Options {
        return switch (args.backend) {
            .cuda_fa2 => if (args.is_prefill) .{
                .cuda_fa2 = .{
                    .mixed = .{
                        .batch_size_decode = args.batch_size,
                        .batch_size_prefill = args.batch_size,
                        .max_num_pages = args.max_num_pages,
                        .max_seqlen_k = args.seq_len,
                        .max_seqlen_q = args.max_seqlen_q,
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
                        .max_num_pages = args.max_num_pages,
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
                        .max_num_pages = args.max_num_pages,
                        .max_seqlen_k = args.seq_len,
                        .max_seqlen_q = args.max_seqlen_q,
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
                        .max_num_pages = args.max_num_pages,
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
                    .max_num_pages = args.max_num_pages,
                    .max_seqlen_q = args.max_seqlen_q,
                    .is_prefill = args.is_prefill,
                },
            },
            .mosaic_tpu => .{
                .mosaic_tpu = .{
                    .is_prefill = args.is_prefill,
                    .batch_size = args.batch_size,
                    .max_num_pages = args.max_num_pages,
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

    pub fn onMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
        return switch (self) {
            .cuda_fa2 => |v| .{ .cuda_fa2 = v.onMemory(memory) },
            .cuda_fa3 => |v| .{ .cuda_fa3 = v.onMemory(memory) },
            .triton => |v| .{ .triton = v.onMemory(memory) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = v.onMemory(memory) },
        };
    }

    pub fn toMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
        return switch (self) {
            .cuda_fa2 => |v| .{ .cuda_fa2 = v.toMemory(memory) },
            .cuda_fa3 => |v| .{ .cuda_fa3 = v.toMemory(memory) },
            .triton => |v| .{ .triton = v.toMemory(memory) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = v.toMemory(memory) },
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

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, kv_cache: KvCache, opts: AttentionOptions) zml.Tensor {
    _ = k;
    _ = v;
    return switch (parameters) {
        .cuda_fa2 => |cuda_fa2_parameters| switch (kv_cache) {
            .split => |split| flashattn.paged_fa2.pagedAttention(cuda_fa2_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
        },
        .cuda_fa3 => |cuda_fa3_parameters| switch (kv_cache) {
            .split => |split| flashattn.paged_fa3.pagedAttention(cuda_fa3_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
        },
        .triton => |triton_parameters| switch (kv_cache) {
            .split => |split| triton.paged.pagedAttention(triton_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
        },
        .mosaic_tpu => |mosaic_tpu_parameters| tpu.mosaic_tpu.pagedAttention(mosaic_tpu_parameters, q, kv_cache.dense, opts),
    };
}

test "Backend.auto selects mosaic_tpu on TPU" {
    const platform: zml.Platform = .{
        .arena = undefined,
        .target = .tpu,
        .pjrt_api = undefined,
        .pjrt_client = undefined,
        .devices = &.{},
        .memories = &.{},
        .physical_mesh = undefined,
        .replicated_sharding = undefined,
        .shardings = .empty,
    };

    try std.testing.expectEqual(Backend.mosaic_tpu, Backend.auto(&platform));
}

test "Backend.auto selects triton on oneAPI" {
    const platform: zml.Platform = .{
        .arena = undefined,
        .target = .oneapi,
        .pjrt_api = undefined,
        .pjrt_client = undefined,
        .devices = &.{},
        .memories = &.{},
        .physical_mesh = undefined,
        .replicated_sharding = undefined,
        .shardings = .empty,
    };

    try std.testing.expectEqual(Backend.triton, Backend.auto(&platform));
}
