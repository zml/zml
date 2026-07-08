const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const flashattn = @import("flashattn.zig");
const metal = @import("metal_attention.zig");
const tpu = @import("tpu_attention.zig");
const triton = @import("triton_attention.zig");

const PagedAttention = @This();

pub const Backend = enum {
    cuda_fa2,
    cuda_fa3,
    triton,
    mosaic_tpu,
    metal,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .cuda => .triton,
            .rocm => .triton,
            .oneapi => .triton,
            .tpu => .mosaic_tpu,
            .metal => .metal,
            else => stdx.debug.panic("Paged attention is not supported on {s} yet", .{@tagName(platform.target)}),
        };
    }
};

pub const Options = union(Backend) {
    cuda_fa2: flashattn.paged_fa2.Options,
    cuda_fa3: flashattn.paged_fa3.Options,
    triton: triton.paged.Options,
    mosaic_tpu: tpu.mosaic_tpu.Options,
    metal: metal.paged.Options,

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
            .metal => .{
                .metal = .{
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
    metal: metal.paged.Parameters,

    pub fn init(options_: Options) Parameters {
        return switch (options_) {
            .cuda_fa2 => |cuda_fa2_options| .{ .cuda_fa2 = flashattn.paged_fa2.Parameters.init(cuda_fa2_options) },
            .cuda_fa3 => |cuda_fa3_options| .{ .cuda_fa3 = flashattn.paged_fa3.Parameters.init(cuda_fa3_options) },
            .triton => |triton_options| .{ .triton = triton.paged.Parameters.init(triton_options) },
            .mosaic_tpu => |mosaic_tpu_options| .{ .mosaic_tpu = tpu.mosaic_tpu.Parameters.init(mosaic_tpu_options) },
            .metal => |metal_options| .{ .metal = metal.paged.Parameters.init(metal_options) },
        };
    }

    pub fn options(self: Parameters) Options {
        return switch (self) {
            .cuda_fa2 => |v| .{ .cuda_fa2 = v.options() },
            .cuda_fa3 => |v| .{ .cuda_fa3 = v.options() },
            .triton => |v| .{ .triton = v.options() },
            .mosaic_tpu => |v| .{ .mosaic_tpu = v.options() },
            .metal => |v| .{ .metal = v.options() },
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
            .metal => |v| .{ .metal = v.onMemory(memory) },
        };
    }

    pub fn toMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
        return switch (self) {
            .cuda_fa2 => |v| .{ .cuda_fa2 = v.toMemory(memory) },
            .cuda_fa3 => |v| .{ .cuda_fa3 = v.toMemory(memory) },
            .triton => |v| .{ .triton = v.toMemory(memory) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = v.toMemory(memory) },
            .metal => |v| .{ .metal = v.toMemory(memory) },
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
        .metal => |metal_parameters| switch (kv_cache) {
            .split => |split| metal.paged.pagedAttention(metal_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
        },
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

test pagedAttention {
    const platform = zml.testing.env();
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const batch_size = 8;
    const num_pages = 64;
    const page_size = 16;
    const tensors: struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, kv_cache: KvCache, token_index: zml.Tensor } = .{
        .q = .init(.{ .b = 8, .hkv = 2, .hg = 4, .hd = 32 }, .f32),
        .k = .init(.{ .b = 8, .hkv = 2, .k = 16, .hd = 32 }, .f32),
        .v = .init(.{ .b = 8, .hkv = 2, .k = 16, .hd = 32 }, .f32),
        .kv_cache = .{
            .split = .{
                .k = .init(.{ .page = num_pages, .hkv = 2, .k_chunk = page_size, .hd = 32 }, .f32),
                .v = .init(.{ .page = num_pages, .hkv = 2, .k_chunk = page_size, .hd = 32 }, .f32),
            },
        },
        .token_index = .init(.{}, .u32),
    };

    const rng_q = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.q.shape(), .{} }, .{});
    defer rng_q.deinit();
    const rng_kv_cache = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.kv_cache.split.k.shape(), .{} }, .{});
    defer rng_kv_cache.deinit();

    const triton_options: Options.Args = .{
        .backend = .triton,
        .is_prefill = true,
        .batch_size = batch_size,
        .seq_len = 64,
        .max_num_pages = 16,
        .max_token_count = 16 * page_size,
        .num_heads = 8,
        .num_kv_heads = 2,
        .head_dim = 32,
        .max_seqlen_q = 16 * 2,
    };
    const triton_parameters: Parameters = .init(.fromBackend(triton_options));
    const attn_opts: AttentionOptions = .{ .is_causal = true };
    var q = try zml.testing.autoCall(allocator, io, &rng_q, zml.Tensor.Rng.normal, {});
    defer q.deinit();

    var kv_cache_k = try zml.testing.autoCall(allocator, io, &rng_kv_cache, zml.Tensor.Rng.normal, {});
    defer kv_cache_k.deinit();
    var kv_cache_v = try zml.testing.autoCall(allocator, io, &rng_kv_cache, zml.Tensor.Rng.normal, {});
    defer kv_cache_v.deinit();
    const kv_cache_d: zml.Bufferized(KvCache) = .{ .split = .{ .k = kv_cache_k, .v = kv_cache_v } };

    var token_index = try zml.Buffer.scalar(io, platform, 64, .u32);
    defer token_index.deinit();

    const triton_exe = try platform.compileFn(allocator, io, pagedAttention, .{ triton_parameters, tensors.q, tensors.k, tensors.v, tensors.kv_cache, attn_opts }, .{ .program_name = "paged_attention_triton" });
    defer triton_exe.deinit();

    const triton_parameters_d: zml.Bufferized(Parameters) = .{ .triton = .{
        .block_table = try .fromBytes(io, platform, triton_parameters.triton.block_table.shape(), .replicated, @ptrCast(&[batch_size][16]i32{
            .{ 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
            .{ 0, 1, 2, 3, 4, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1 },
            .{ 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 },
            .{ 0, 1, 2, 3, 4, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
            .{ 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
            .{ 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
            .{ 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
            .{ 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        })),
        .seq_lens = try .fromBytes(io, platform, triton_parameters.triton.seq_lens.shape(), .replicated, @ptrCast(&[batch_size]i32{
            4 * page_size + 3,
            7 * page_size + 11,
            15 * page_size + 9,
            4 * page_size,
            4 * page_size + 3,
            4 * page_size + 3,
            4 * page_size + 3,
            4 * page_size + 3,
        })),
        .query_start_len = try .fromBytes(io, platform, triton_parameters.triton.query_start_len.shape(), .replicated, @ptrCast(&[batch_size]i32{
            1,
            1,
            1,
            32,
            1,
            1,
            1,
            1,
        })),
    } };

    const triton_d = try zml.testing.autoCall(allocator, io, &triton_exe, pagedAttention, .{ triton_parameters_d, q, undefined, undefined, kv_cache_d });
    const triton_h: zml.Slice = try triton_d.toSliceAlloc(allocator, io);
    defer triton_h.free(allocator);

    for (std.enums.values(Backend)) |backend| {
        switch (backend) {
            .triton => continue,
            else => {},
        }

        var backend_options = triton_options;
        backend_options.backend = backend;
        const parameters: Parameters = .init(.fromBackend(backend_options));

        const exe = try platform.compileFn(
            allocator,
            io,
            pagedAttention,
            .{ parameters, tensors.q, tensors.k, tensors.v, tensors.kv_cache, attn_opts },
            .{ .program_name = try std.fmt.allocPrint(arena, "paged_attention_{t}", .{backend}) },
        );
        defer exe.deinit();

        const parameters_d: zml.Bufferized(Parameters) = switch (backend) {
            // No materializer implemented for cuda fa2/fa3
            .cuda_fa2, .cuda_fa3 => return error.SkipZigTest,
            .triton => unreachable,
            .metal => .{ .metal = .{
                .block_table = triton_parameters_d.triton.block_table,
                .seq_lens = triton_parameters_d.triton.seq_lens,
                .query_start_len = triton_parameters_d.triton.query_start_len,
            } },
            .mosaic_tpu => .{ .mosaic_tpu = .{
                .block_table = triton_parameters_d.triton.block_table,
                .seq_lens = triton_parameters_d.triton.seq_lens,
                .query_start_len = triton_parameters_d.triton.query_start_len,
            } },
        };
        // defer Parameters.deinitBuffer(&parameters_d);

        const output_d = try zml.testing.autoCall(allocator, io, &exe, pagedAttention, .{ parameters_d, q, undefined, undefined, kv_cache_d });
        try zml.testing.expectClose(io, triton_h, output_d, .{
            .absolute_tolerance = 1e-3,
            .relative_tolerance = 1e-2,
            .epsilon_relative = 1e-6,
        });
    }
}
