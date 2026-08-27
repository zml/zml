const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const flashattn = @import("flashattn.zig");
const metal = @import("metal_attention.zig");
const tpu = @import("tpu_attention.zig");
const triton = @import("triton_attention.zig");

const PagedAttention = @This();

test {
    std.testing.refAllDecls(Backend);
    std.testing.refAllDecls(Options);
    std.testing.refAllDecls(Parameters);
    std.testing.refAllDecls(KvCache);
}

pub const Backend = enum {
    cuda_fa2,
    cuda_fa3,
    triton,
    mosaic_tpu,
    metal,
    stablehlo,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .cuda => .triton,
            .rocm => .triton,
            .oneapi => .triton,
            .tpu => .mosaic_tpu,
            .metal => .metal,
            .cpu => .stablehlo,
            .neuron => stdx.debug.panic("Paged attention is not supported on {s} yet", .{@tagName(platform.target)}),
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .stablehlo => true,
            .triton => platform.target != .cpu,
            .metal => platform.target == .metal,
            .mosaic_tpu => platform.target == .tpu,
            .cuda_fa2 => platform.target == .cuda,
            .cuda_fa3 => {
                if (platform.target != .cuda) return false;
                const first_device = platform.pjrt_client.devices(platform.pjrt_api)[0];
                const cc = zml.platform.cuda.tryGetComputeCapabilities(platform, first_device) orelse return false;
                return std.mem.eql(u8, cc, "9.0");
            },
        };
    }
};

pub const Options = union(Backend) {
    cuda_fa2: flashattn.paged_fa2.Options,
    cuda_fa3: flashattn.paged_fa3.Options,
    triton: triton.paged.Options,
    mosaic_tpu: tpu.mosaic_tpu.Options,
    metal: metal.paged.Options,
    stablehlo: triton.paged.Options,

    const Args = struct {
        backend: Backend,
        is_prefill: bool,
        batch_size: u32,
        batch_size_prefill: ?u32 = null,
        batch_size_decode: ?u32 = null,
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
                        .batch_size_decode = args.batch_size_decode orelse args.batch_size,
                        .batch_size_prefill = args.batch_size_prefill orelse args.batch_size,
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
                        .batch_size_decode = args.batch_size_decode orelse args.batch_size,
                        .batch_size_prefill = args.batch_size_prefill orelse args.batch_size,
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
            inline .triton, .metal, .stablehlo => |t| @unionInit(Options, @tagName(t), .{
                .batch_size = args.batch_size,
                .max_num_pages = args.max_num_pages,
                .max_seqlen_q = args.max_seqlen_q,
                .is_prefill = args.is_prefill,
            }),
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
    stablehlo: triton.paged.Parameters,

    pub fn init(options_: Options) Parameters {
        return switch (options_) {
            inline else => |opts, backend| @unionInit(Parameters, @tagName(backend), .init(opts)),
        };
    }

    pub fn options(self: Parameters) Options {
        return switch (self) {
            inline else => |v, backend| @unionInit(Options, @tagName(backend), v.options()),
        };
    }

    pub fn allocationSize(self: Parameters) usize {
        return switch (self) {
            inline else => |v| v.allocationSize(),
        };
    }

    pub fn onMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
        zml.Tensor.onMemoryAll(&self, memory);
        return self;
    }

    pub fn toMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
        return zml.Tensor.toMemoryAll(self, memory);
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
    // TODO: rename `dense` to `fused`.
    dense: zml.Tensor,
    latent: zml.Tensor,

    pub fn update(
        self: KvCache,
        new_k: zml.Tensor,
        new_v: zml.Tensor,
        slot_mapping: zml.Tensor,
        chunk_size: u32,
        backend: Backend,
    ) KvCache {
        const page_index, const offset = getPageAndOffsetFromSlotMapping(slot_mapping, chunk_size);

        const kv: KvCache = switch (self) {
            .split => |split| switch (backend) {
                .cuda_fa2, .cuda_fa3, .triton, .mosaic_tpu, .metal, .stablehlo => .{
                    .split = .{
                        .k = split.k.scatterSlices(
                            .{ .page = page_index, .k_chunk = offset },
                            new_k,
                            .{ .update_fn = zml.Tensor.ScatterOpts.override, .indices_are_unique = false, .indices_are_sorted = false },
                        ).reuseBuffer(self.split.k),
                        .v = split.v.scatterSlices(
                            .{ .page = page_index, .k_chunk = offset },
                            new_v,
                            .{ .update_fn = zml.Tensor.ScatterOpts.override, .indices_are_unique = false, .indices_are_sorted = false },
                        ).reuseBuffer(self.split.v),
                    },
                },
            },
            .dense => @panic("TODO"),
            .latent => |latent_kv| .{
                .latent = latent_kv.scatterSlices(
                    .{ .page = page_index, .k_chunk = offset },
                    new_k,
                    .{ .update_fn = zml.Tensor.ScatterOpts.override, .indices_are_unique = false, .indices_are_sorted = false },
                ).reuseBuffer(self.latent),
            },
        };

        return kv;
    }

    pub fn getPageAndOffsetFromSlotMapping(slot_mapping: zml.Tensor, page_size: u32) [2]zml.Tensor {
        const page_index = slot_mapping.divByConst(page_size);
        const offset = slot_mapping.remainder(.scalar(page_size, slot_mapping.dtype()));
        return .{ page_index, offset };
    }

    pub fn getPage(
        self: KvCache,
        page_index: zml.Tensor,
    ) [2]zml.Tensor {
        return switch (self) {
            .split => |split| .{
                split.k.slice(split.k.axis(.page), .dynSingle(page_index)),
                split.v.slice(split.k.axis(.page), .dynSingle(page_index)),
            },
            .dense, .latent => @panic("TODO"),
        };
    }
};

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, kv_cache: KvCache, opts: AttentionOptions) zml.Tensor {
    _ = k;
    _ = v;
    return switch (parameters) {
        .cuda_fa2 => |cuda_fa2_parameters| switch (kv_cache) {
            .split => |split| flashattn.paged_fa2.pagedAttention(cuda_fa2_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
            .latent => std.debug.panic("latent KV pages are only supported with Multi-Latent Attention", .{}),
        },
        .cuda_fa3 => |cuda_fa3_parameters| switch (kv_cache) {
            .split => |split| flashattn.paged_fa3.pagedAttention(cuda_fa3_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
            .latent => std.debug.panic("latent KV pages are only supported with Multi-Latent Attention", .{}),
        },
        .triton => |triton_parameters| switch (kv_cache) {
            .split => |split| triton.paged.pagedAttention(triton_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
            .latent => std.debug.panic("latent KV pages are only supported with Multi-Latent Attention", .{}),
        },
        .mosaic_tpu => |mosaic_tpu_parameters| switch (kv_cache) {
            .split => std.debug.panic("split KV pages is not supported with the mosaic_tpu backend", .{}),
            .latent => std.debug.panic("latent KV pages are only supported with Multi-Latent Attention", .{}),
            .dense => tpu.mosaic_tpu.pagedAttention(mosaic_tpu_parameters, q, kv_cache.dense, opts),
        },
        .metal => |metal_parameters| switch (kv_cache) {
            .split => |split| metal.paged.pagedAttention(metal_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
            .latent => std.debug.panic("latent KV pages are only supported with Multi-Latent Attention", .{}),
        },
        .stablehlo => |params| switch (kv_cache) {
            .latent => std.debug.panic("latent KV pages are only supported with Multi-Latent Attention", .{}),
            else => stablehlo_pagedAttention(params, q, kv_cache, opts),
        },
    };
}

test "Backend.auto selects mosaic_tpu on TPU" {
    const platform: zml.Platform = .{
        .arena = undefined,
        .target = .tpu,
        .pjrt_api = undefined,
        .pjrt_client = undefined,
        .state = zml.platform.State.init(.tpu),
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
        .state = zml.platform.State.init(.oneapi),
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

    const num_pages = 64;
    const page_size = 16;
    const max_num_pages = 16;
    const prefill_token_count = 32;
    const num_prefill = 1;
    const num_decode = 6;
    const active_batch_size = num_prefill + num_decode;
    const batch_size = active_batch_size + 1;
    const query_token_count = prefill_token_count + num_decode;
    const dt: zml.DataType = .bf16;
    const partition = .{ .hkv = .model };
    const tensors: struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, kv_cache: KvCache } = .{
        .q = .withPartitioning(.init(.{ .b = query_token_count, .hkv = 4, .hg = 4, .hd = 32 }, dt), partition),
        .k = .withPartitioning(.init(.{ .b = query_token_count, .hkv = 4, .hd = 32 }, dt), partition),
        .v = .withPartitioning(.init(.{ .b = query_token_count, .hkv = 4, .hd = 32 }, dt), partition),
        .kv_cache = .{
            .split = .{
                .k = .withPartitioning(.init(.{ .page = num_pages, .k_chunk = page_size, .hkv = 4, .hd = 32 }, dt), partition),
                .v = .withPartitioning(.init(.{ .page = num_pages, .k_chunk = page_size, .hkv = 4, .hd = 32 }, dt), partition),
            },
        },
    };

    const shardings: []const zml.Sharding = &.{ platform.replicated_sharding, platform.shardings.get("model").? };
    const rng_q = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.q.shape(), .{} }, .{ .shardings = shardings });
    defer rng_q.deinit();
    const rng_k = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.k.shape(), .{} }, .{ .shardings = shardings });
    defer rng_k.deinit();
    const rng_kv_cache = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.kv_cache.split.k.shape(), .{} }, .{ .shardings = shardings });
    defer rng_kv_cache.deinit();

    const triton_options_args: Options.Args = .{
        .backend = .triton,
        .is_prefill = true,
        .batch_size = batch_size,
        .batch_size_prefill = num_prefill,
        .batch_size_decode = num_decode,
        .seq_len = max_num_pages * page_size,
        .max_num_pages = max_num_pages,
        .max_token_count = 16 * page_size,
        .num_heads = @intCast(tensors.q.dim(.hg) * tensors.q.dim(.hkv)),
        .num_kv_heads = @intCast(tensors.q.dim(.hkv)),
        .head_dim = @intCast(tensors.q.dim(.hd)),
        .max_seqlen_q = 16 * 2,
    };
    const triton_parameters: Parameters = .init(.fromBackend(triton_options_args));
    const attn_opts: AttentionOptions = .{ .is_causal = true };
    var q = try zml.testing.autoCall(allocator, io, &rng_q, zml.Tensor.Rng.normal, {});
    defer q.deinit();
    var new_k = try zml.testing.autoCall(allocator, io, &rng_k, zml.Tensor.Rng.normal, {});
    defer new_k.deinit();
    var new_v = try zml.testing.autoCall(allocator, io, &rng_k, zml.Tensor.Rng.normal, {});
    defer new_v.deinit();

    var kv_cache_d: zml.Bufferized(KvCache) = .{ .split = .{
        .k = try zml.testing.autoCall(allocator, io, &rng_kv_cache, zml.Tensor.Rng.normal, {}),
        .v = try zml.testing.autoCall(allocator, io, &rng_kv_cache, zml.Tensor.Rng.normal, {}),
    } };
    defer zml.Buffer.deinitAll(KvCache, &kv_cache_d);

    const block_table: [batch_size][max_num_pages]i32 = .{
        // prefilling pages 9-10
        .{ 0, 1, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        // generation
        .{ 0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        .{ 0, 1, 2, 3, 4, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1 },
        .{ 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 },
        .{ 0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        .{ 0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        .{ 0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        .{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    };

    const seq_lens: [batch_size]i32 = .{
        4 * page_size,
        4 * page_size + 3,
        7 * page_size + 11,
        15 * page_size + 9,
        4 * page_size + 3,
        4 * page_size + 3,
        4 * page_size + 3,
        0,
    };

    const query_start_len: [batch_size + 1]i32 = .{ 0, 32, 33, 34, 35, 36, 37, 38, 38 };
    var triton_parameters_d: zml.Bufferized(Parameters) = .{ .triton = .{
        .block_table = try .fromBytes(io, platform, triton_parameters.triton.block_table.shape(), .replicated, @ptrCast(&block_table)),
        .seq_lens = try .fromBytes(io, platform, triton_parameters.triton.seq_lens.shape(), .replicated, @ptrCast(&seq_lens)),
        .query_start_len = try .fromBytes(io, platform, triton_parameters.triton.query_start_len.shape(), .replicated, @ptrCast(&query_start_len)),
    } };
    defer zml.Buffer.deinitAll(Parameters, &triton_parameters_d);

    var results_per_backend: std.enums.EnumArray(Backend, ?zml.Slice) = .initFill(null);
    defer {
        for (std.enums.values(Backend)) |backend| {
            if (results_per_backend.get(backend)) |slice| slice.free(allocator);
        }
    }

    for (std.enums.values(Backend)) |backend| {
        if (!backend.isAvailable(platform)) {
            std.log.warn("paged_attention backend {t} not available", .{backend});
            continue;
        }
        std.log.warn("Testing paged_attention with backend {t}", .{backend});

        var backend_options_args = triton_options_args;
        backend_options_args.backend = backend;
        const parameters: Parameters = .init(.fromBackend(backend_options_args));

        const exe = try platform.compileFn(
            allocator,
            io,
            pagedAttention,
            .{ parameters, tensors.q, tensors.k, tensors.v, tensors.kv_cache, attn_opts },
            .{ .program_name = try std.fmt.allocPrint(arena, "paged_attention_{t}", .{backend}), .shardings = shardings },
        );
        defer exe.deinit();

        var parameters_d: zml.Bufferized(Parameters) = switch (parameters) {
            // No materializer implemented for cuda fa3
            .cuda_fa3 => continue,
            .cuda_fa2 => |params| cuda_fa2: {
                var block_table_prefill: [num_prefill][max_num_pages]i32 = undefined;
                @memcpy(&block_table_prefill, block_table[0..num_prefill]);
                var block_table_decode: [num_decode][max_num_pages]i32 = undefined;
                @memcpy(&block_table_decode, block_table[num_prefill .. num_prefill + num_decode]);

                var cu_seqlens_q_prefill: [num_prefill + 1]i32 = undefined;
                @memcpy(&cu_seqlens_q_prefill, query_start_len[0 .. num_prefill + 1]);
                var cu_seqlens_q_decode: [num_decode + 1]i32 = undefined;
                for (&cu_seqlens_q_decode, query_start_len[num_prefill .. num_prefill + num_decode + 1]) |*decode_len, query_start| {
                    decode_len.* = query_start - prefill_token_count;
                }

                var seqused_k_prefill: [num_prefill]i32 = undefined;
                @memcpy(&seqused_k_prefill, seq_lens[0..num_prefill]);
                var seqused_k_decode: [num_decode]i32 = undefined;
                @memcpy(&seqused_k_decode, seq_lens[num_prefill .. num_prefill + num_decode]);

                break :cuda_fa2 .{ .cuda_fa2 = .{ .mixed = .{
                    .block_table_prefill = try .fromBytes(io, platform, params.mixed.block_table_prefill.shape(), .replicated, @ptrCast(&block_table_prefill)),
                    .cu_seqlens_q_prefill = try .fromBytes(io, platform, params.mixed.cu_seqlens_q_prefill.shape(), .replicated, @ptrCast(&cu_seqlens_q_prefill)),
                    .seqused_k_prefill = try .fromBytes(io, platform, params.mixed.seqused_k_prefill.shape(), .replicated, @ptrCast(&seqused_k_prefill)),

                    .block_table_decode = try .fromBytes(io, platform, params.mixed.block_table_decode.shape(), .replicated, @ptrCast(&block_table_decode)),
                    .cu_seqlens_q_decode = try .fromBytes(io, platform, params.mixed.cu_seqlens_q_decode.shape(), .replicated, @ptrCast(&cu_seqlens_q_decode)),
                    .seqused_k_decode = try .fromBytes(io, platform, params.mixed.seqused_k_decode.shape(), .replicated, @ptrCast(&seqused_k_decode)),

                    .metadata = .{ .decode_offset = try .scalar(io, platform, prefill_token_count, .i32) },
                } } };
            },
            .triton => triton_parameters_d,
            inline .metal, .mosaic_tpu, .stablehlo => |_, t| @unionInit(zml.Bufferized(Parameters), @tagName(t), .{
                .block_table = triton_parameters_d.triton.block_table,
                .seq_lens = triton_parameters_d.triton.seq_lens,
                .query_start_len = triton_parameters_d.triton.query_start_len,
            }),
        };
        // cu_fa2 creates new buffers while other reuse triton buffers.
        defer if (backend == .cuda_fa2) zml.Buffer.deinitAll(Parameters, &parameters_d);

        var output_d = try zml.testing.autoCall(allocator, io, &exe, pagedAttention, .{ parameters_d, q, new_k, new_v, kv_cache_d });
        defer output_d.deinit();
        results_per_backend.set(backend, try output_d.toSliceAlloc(allocator, io));
    }

    const reference_backend: Backend = .triton;
    const reference = results_per_backend.get(reference_backend) orelse return error.SkipZigTest;

    var num_failed: u32 = 0;
    for (std.enums.values(Backend)) |backend| {
        if (backend == reference_backend) continue;
        const output_h = results_per_backend.get(backend) orelse continue;

        const tolerance: zml.testing.CompareOpts = .{
            .absolute_tolerance = 1e-2,
            .relative_tolerance = 1e-2,
            .epsilon_relative = 1e-6,
        };

        zml.testing.expectClose(io, reference, output_h, tolerance) catch |err| switch (err) {
            error.TestUnexpectedResult => {
                num_failed += 1;
                std.log.err("test pagedAttention failed on backend {0t}\n--> reference ({1t}): {2d:32.3}\n--> pagedAttention({0t}): {3d:32.3}", .{
                    backend,
                    reference_backend,
                    reference.subSlice(1, 0, 1).squeeze(1).subSlice(1, 0, 1).squeeze(1),
                    output_h.subSlice(1, 0, 1).squeeze(1).subSlice(1, 0, 1).squeeze(1),
                });
            },
            else => |e| return e,
        };
    }

    if (num_failed > 0) return error.TestUnexpectedResult;
}

fn stablehlo_pagedAttention(
    parameters: triton.paged.Parameters,
    q: zml.Tensor,
    kv_cache: KvCache,
    opts: AttentionOptions,
) zml.Tensor {
    const page_size = kv_cache.split.k.dim(.k_chunk);

    const final_state = zml.ops.@"while"(
        AttentionLoop,
        .{ .q = q, .kv_cache = kv_cache, .parameters = parameters, .opts = opts },
        .{
            // i32 to match triton conventions
            .slot_id = .scalar(0, .i32),
            .seq_id = .scalar(0, .i32),
            .q_page_idx = .scalar(0, .i32),
            .k_page_idx = .scalar(0, .i32),
            // Note: we do all partial softmax in f32
            .partial_softmax_prefill = .zeroes(q.shape().setDim(.b, page_size).withDtype(.f32)),
            .partial_softmax_decode = .zeroes(q.shape().setDim(.b, 1).withDtype(.f32)),
            .out = .zeroes(q.shape()),
        },
    );

    return final_state.out;
}

const AttentionLoop = struct {
    q: zml.Tensor,
    kv_cache: KvCache,
    parameters: triton.paged.Parameters,
    opts: AttentionOptions,

    const While = @This();

    pub const State = struct {
        slot_id: zml.Tensor,
        seq_id: zml.Tensor,
        q_page_idx: zml.Tensor,
        k_page_idx: zml.Tensor,
        partial_softmax_prefill: PartialSoftmax,
        partial_softmax_decode: PartialSoftmax,
        out: zml.Tensor,
    };

    pub fn cond(self: While, state: State) zml.Tensor {
        const has_more_sequences = state.seq_id.cmp(.LT, .scalar(self.parameters.seq_lens.count(), .i32));
        const query_start = self.parameters.query_start_len.slice(0, .dynSingle(state.seq_id));
        const query_end = self.parameters.query_start_len.slice(0, .dynSingle(state.seq_id.addConstant(1)));
        const has_queries = query_start.cmp(.LT, query_end).asScalar();
        return has_more_sequences.logical(.AND, has_queries);
    }

    pub fn body(self: While, state_: State) State {
        const query_start = self.parameters.query_start_len.slice(0, .dynSingle(state_.seq_id));
        const query_end = self.parameters.query_start_len.slice(0, .dynSingle(state_.seq_id.addConstant(1)));
        const seq_num_queries_: zml.Tensor = .asScalar(.sub(query_end, query_start));
        const page_size_: u32 = @intCast(self.kv_cache.split.k.dim(.k_chunk));

        return zml.ops.@"if"(
            struct {
                const IfPrefill = @This();
                while_body: While,
                state: State,
                seq_num_queries: zml.Tensor,
                page_size: u32,

                /// Prefill version: num_slots == page_size, we multiply a full page of queries with a page of keys.
                pub fn onTrue(if_ctx: IfPrefill) State {
                    const while_body = if_ctx.while_body;
                    const state = if_ctx.state;
                    const page_size = if_ctx.page_size;
                    const num_slots: u32 = page_size;

                    // seq_lens includes the current queries; subtracting their count gives the context length.
                    const seq_len = while_body.parameters.seq_lens.slice(0, .dynSingle(state.seq_id));
                    const q_offset = seq_len.sub(if_ctx.seq_num_queries).add(state.q_page_idx.scale(page_size));
                    const next_k_offset = state.k_page_idx.addConstant(1).scale(page_size);

                    // The current K page is processed in this iteration. Continue if the next page contains
                    // any key visible to the current query chunk, whose exclusive end is q_offset + num_slots.
                    const query_has_more_keys: zml.Tensor = .cmp(next_k_offset, .LT, q_offset.addConstant(num_slots));
                    const sequence_has_more_queries: zml.Tensor = .cmp(state.q_page_idx.addConstant(1).scale(num_slots), .LT, if_ctx.seq_num_queries);

                    const partial_softmax = while_body.attentionOnePage(state, q_offset, num_slots, page_size);
                    return updateState(state, num_slots, query_has_more_keys, sequence_has_more_queries, partial_softmax);
                }

                /// Decode version: num_slots == 1, we multiply one query with a page of keys.
                pub fn onFalse(if_ctx: IfPrefill) State {
                    const while_body = if_ctx.while_body;
                    const state = if_ctx.state;
                    const page_size = if_ctx.page_size;
                    const num_slots: u32 = 1;

                    // A decode query is the final token in seq_len, so its zero-based offset is seq_len - 1.
                    const seq_len = while_body.parameters.seq_lens.slice(0, .dynSingle(state.seq_id)).asScalar();
                    const q_offset = seq_len.sub(if_ctx.seq_num_queries);
                    const next_k_offset = state.k_page_idx.addConstant(1).scale(page_size);

                    const query_has_more_keys: zml.Tensor = .cmp(next_k_offset, .LT, q_offset.addConstant(num_slots));
                    const sequence_has_more_queries: zml.Tensor = .scalar(false, .bool);

                    const partial_softmax = while_body.attentionOnePage(state, q_offset, num_slots, page_size);
                    return updateState(state, num_slots, query_has_more_keys, sequence_has_more_queries, partial_softmax);
                }
            },
            .{
                .while_body = self,
                .state = state_,
                .seq_num_queries = seq_num_queries_,
                .page_size = page_size_,
            },
            .cmp(seq_num_queries_, .GT, .scalar(1, .i32)),
        );
    }

    pub fn attentionOnePage(self: While, state: State, q_offset: zml.Tensor, q_chunk: u32, page_size: u32) PartialSoftmax {
        const k_offset = state.k_page_idx.scale(page_size);
        const active_q = self.q.slice(self.q.axis(.b), .dyn(state.slot_id, q_chunk));

        const active_k_page_id = self.parameters.block_table.slices(
            .{ .b, .p },
            &.{ .dynSingle(state.seq_id), .dynSingle(state.k_page_idx) },
        );
        const active_k, const active_v = self.kv_cache.getPage(active_k_page_id);

        const dtype = self.q.dtype();
        const attn_mask: zml.Tensor = attn_mask: {
            const mask_shape: zml.Shape = .init(.{ .b = q_chunk, .k_chunk = page_size }, dtype);

            const q_idx = zml.Tensor.iota(mask_shape, .b).add(q_offset);
            const k_idx = zml.Tensor.iota(mask_shape, .k_chunk).add(k_offset);

            const mask = zml.Tensor.cmp(k_idx, .LE, q_idx);
            const minus_inf = zml.Tensor.constant(dtype.minValue());
            break :attn_mask .select(mask, .scalar(0, dtype), minus_inf);
        };

        const scale: f32 = self.opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(active_q.dim(.hd)))));
        var attn_weights = active_q.dot(active_k, .hd).scale(scale);
        attn_weights = attn_weights.add(attn_mask.broad(attn_weights.shape()));

        const og_softmax = if (q_chunk == 1) state.partial_softmax_decode else state.partial_softmax_prefill;
        // Note: we compute q * k using q and k dtype, but the partial softmax is in f32.
        const precise_dt = og_softmax.values.dtype();
        var partial = partialSoftmax(attn_weights.convert(precise_dt), .k_chunk);
        partial.values = partial.values.dot(active_v.convert(precise_dt), .k_chunk).transpose(og_softmax.values.shape());
        partial.max_value = partial.max_value.transpose(og_softmax.max_value.shape());
        partial.exp_sum = partial.exp_sum.transpose(og_softmax.exp_sum.shape());

        return og_softmax.merge(partial);
    }

    pub fn updateState(state: State, num_slots: u32, query_has_more_keys: zml.Tensor, sequence_has_more_queries: zml.Tensor, softmax: PartialSoftmax) State {
        return zml.ops.if2(
            query_has_more_keys,
            // If the query has more pages: update partial softmax with attention on current page, delay updating output.
            State{
                .slot_id = state.slot_id,
                .seq_id = state.seq_id,
                .q_page_idx = state.q_page_idx,
                .k_page_idx = state.k_page_idx.addConstant(1),
                .partial_softmax_prefill = if (num_slots == 1) state.partial_softmax_prefill else softmax,
                .partial_softmax_decode = if (num_slots == 1) softmax else state.partial_softmax_decode,
                .out = state.out,
            },
            // Query has no more pages, we commit softmax to output and increment slot
            State{
                .k_page_idx = .scalar(0, .i32),
                .partial_softmax_prefill = if (num_slots == 1) state.partial_softmax_prefill else .reset(softmax),
                .partial_softmax_decode = if (num_slots == 1) .reset(softmax) else state.partial_softmax_decode,
                .out = state.out.dynamicUpdateSlice(.{ .b = state.slot_id }, softmax.finalize().convert(state.out.dtype())),
                .slot_id = state.slot_id.addConstant(num_slots),
                // If the sequence has more query chunks: increment q_page_idx, otherwise start over at 0.
                .q_page_idx = .select(sequence_has_more_queries, state.q_page_idx.addConstant(1), .scalar(0, .i32)),
                // If the sequence has more query chunks: keep seq_id, otherwise increment seq_id.
                .seq_id = .select(sequence_has_more_queries, state.seq_id, state.seq_id.addConstant(1)),
            },
        );
    }
};

pub const PartialSoftmax = struct {
    values: zml.Tensor,
    exp_sum: zml.Tensor,
    max_value: zml.Tensor,

    pub fn zeroes(q_shape: zml.Shape) PartialSoftmax {
        return .{
            .values = .zeroes(q_shape),
            .exp_sum = .zeroes(q_shape.drop(.hd)),
            .max_value = .broad(.constant(q_shape.dtype().minValue()), q_shape.drop(.hd)),
        };
    }

    pub fn reset(self: PartialSoftmax) PartialSoftmax {
        return .zeroes(self.values.shape());
    }

    pub fn merge(self: PartialSoftmax, other: PartialSoftmax) PartialSoftmax {
        // Rescale self and other using the new global_max.
        const global_max = self.max_value.maximum(other.max_value.convert(self.max_value.dtype()));
        const new_self = self.rescale(global_max);
        const new_other = other.rescale(global_max);

        // Now that self and other are using the same scale, we can just add them:
        return .{
            .max_value = global_max,
            .values = new_self.values.add(new_other.values),
            .exp_sum = new_self.exp_sum.add(new_other.exp_sum),
        };
    }

    /// Update max_value and rescale attn and exp_sum accordingly.
    pub fn rescale(self: PartialSoftmax, max_value: zml.Tensor) PartialSoftmax {
        const max_diff_exp: zml.Tensor = .exp(self.max_value.sub(max_value));
        const sum_dtype = self.exp_sum.dtype();
        return .{
            .max_value = max_value,
            .values = self.values.mul(max_diff_exp.broad(self.values.shape())),
            .exp_sum = self.exp_sum.mul(max_diff_exp.convert(sum_dtype)),
        };
    }

    /// Divides the intermediary results by the exp_sum to get the proper attention values.
    pub fn finalize(self: PartialSoftmax) zml.Tensor {
        return self.values.div(self.exp_sum.broad(self.values.shape()).convert(self.values.dtype()));
    }
};

/// Compute softmax over a chunk.
/// Returns intermediary results to allow aggregating later.
pub fn partialSoftmax(self: zml.Tensor, axis: anytype) PartialSoftmax {
    const a = self.axis(axis);
    const max_val = self.max(a).squeeze(a).maximum(.scalar(-1e16, self.dtype()));
    const out = self.sub(max_val.broad(self.shape())).exp();
    return .{
        .values = out,
        .exp_sum = out.convert(.f32).sum(a).squeeze(a),
        .max_value = max_val,
    };
}

pub const Mla = struct {
    pub const Options = struct {
        rope_rank: i64,
        scale: ?f32 = null,
        /// null selects automatically; 1 forces the 2D kernel; other values must be powers of two up to 16.
        num_kv_splits: ?u8 = null,
    };

    fn stablehlo_pagedSparseAttention(q: zml.Tensor, kv: zml.Tensor, sink: ?zml.Tensor, topk: zml.Tensor, opts: Mla.Options) zml.Tensor {
        stdx.debug.assert(kv.dim(.hkv) == 1, "StableHLO MLA expects one latent KV head, got {}", .{kv.dim(.hkv)});
        const kv_flat = kv.squeeze(.hkv).reshape(.{
            .kv = kv.dim(.page) * kv.dim(.k_chunk),
            .hd = kv.dim(.hd),
        });
        const valid_topk = topk.cmp(.GE, zml.Tensor.zeroes(topk.shape()));
        const safe_topk = zml.Tensor.select(valid_topk, topk, zml.Tensor.zeroes(topk.shape()));
        const mask = valid_topk.insertAxes(.topk, .{.h});
        const selected_kv = kv_flat.gather(.{ .kv = safe_topk.rename(.{ .q = .b }) }, .{}).rename(.{ .b = .q, .topk = .kv }).convert(.f32);

        const dims = zml.nn.collectDims(.{ .h, .q, .kv, .hd }, &.{ q, kv_flat }, .strict) catch {
            stdx.debug.panic("Inputs have incompatible shapes (q: {f}, kv: {f}).", .{ q, kv_flat });
        };

        const sqrt_head_dim = opts.scale orelse 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.hd)));
        const q_32 = q.convert(.f32);
        var scores = q_32.dot(selected_kv, .hd).scale(sqrt_head_dim);
        scores = zml.Tensor.select(mask.broad(scores.shape()), scores, zml.Tensor.constant(scores.dtype().minValue()));

        const sink_shape = q.shape().set(.hd, 1);
        const attn_sink = sink orelse stdx.debug.panic("ragged MLA attention requires an attention sink", .{});
        const sink_ = attn_sink.insertAxes(0, .{.q}).insertAxes(.last, .{.hd}).broad(sink_shape);
        const scores_sink = zml.Tensor.concatenate(&.{ scores, sink_.convert(scores.dtype()) }, .kv);

        const attn_weights = scores_sink.softmax(.kv);
        const attn_weights_non_sink = attn_weights.slice(-1, .{ .end = topk.dim(.topk) });
        return attn_weights_non_sink.dot(selected_kv, .kv).convert(q.dtype());
    }

    pub fn pagedSparseAttention(parameters: Parameters, q: zml.Tensor, kv_cache: KvCache, sink: ?zml.Tensor, topk: zml.Tensor, tokens_pos: zml.Tensor, opts: Mla.Options) zml.Tensor {
        const latent_kv = switch (kv_cache) {
            .latent => |latent_kv| latent_kv,
            else => std.debug.panic("Sparse Multi-Latent Attention support only latent KV pages, got: {}", .{std.meta.activeTag(kv_cache)}),
        };

        stdx.debug.assert(q.shape().hasTags(.{ .q, .h, .hd }), "expected q to have tags .q, .h, .hd after flattening, got {f}", .{q.shape()});
        stdx.debug.assert(q.dim(.hd) > opts.rope_rank, "expected q head dim ({}) to include a rope tail of {}", .{ q.dim(.hd), opts.rope_rank });
        stdx.debug.assert(latent_kv.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "expected paged latent KV cache to have tags .page, .k_chunk, .hkv, .hd, got {f}", .{latent_kv.shape()});
        stdx.debug.assert(latent_kv.dim(.hd) == q.dim(.hd), "expected q and kv cache head dims to match, got q={} kv={}", .{ q.dim(.hd), latent_kv.dim(.hd) });

        return switch (parameters) {
            .triton => |triton_parameters| triton.paged.pagedSparseMla(triton_parameters, q, latent_kv, sink, topk, tokens_pos, opts),
            .stablehlo => |stablehlo_parameters| stablehlo_pagedSparseAttention(
                q,
                latent_kv,
                sink,
                triton.paged.topkToPhysical(stablehlo_parameters, topk, tokens_pos, latent_kv.dim(.k_chunk)),
                opts,
            ),
            else => @panic("NOPE"),
        };
    }
};

test "use mla kernel" {
    const platform = zml.testing.env();
    if (!Backend.triton.isAvailable(platform)) return error.SkipZigTest;

    const parameters = Parameters.init(.fromBackend(.{
        .backend = .triton,
        .is_prefill = true,
        .batch_size = 1,
        .seq_len = 32,
        .max_num_pages = 2,
        .max_token_count = 1,
        .num_heads = 16,
        .num_kv_heads = 1,
        .head_dim = 128,
        .max_seqlen_q = 1,
    }));
    const q_shape = zml.Shape.init(.{ .q = 1, .h = 16, .hd = 128 }, .f32);
    const kv_shape = zml.Shape.init(.{ .page = 2, .k_chunk = 16, .hkv = 1, .hd = 128 }, .f32);
    const sink_shape = zml.Shape.init(.{ .h = 16 }, .f32);
    const topk_shape = zml.Shape.init(.{ .q = 1, .topk = 32 }, .i32);
    const tokens_pos_shape = zml.Shape.init(.{ .q = 1 }, .i32);

    var q_data: [1][16][128]f32 = undefined;
    @memset(std.mem.sliceAsBytes(&q_data), 0);
    var kv_data: [2][16][1][128]f32 = undefined;
    @memset(std.mem.sliceAsBytes(&kv_data), 0);
    for (&kv_data[1][0][0]) |*value| value.* = 10;
    for (&kv_data[1][1][0]) |*value| value.* = 11;
    var sink_data: [16]f32 = undefined;
    for (&sink_data) |*value| value.* = -std.math.inf(f32);
    var topk_data: [1][32]i32 = undefined;
    @memset(&topk_data[0], -1);
    topk_data[0][0] = 0;
    topk_data[0][1] = 1;
    const tokens_pos_data: [1]i32 = .{31};
    const block_table: [1][2]i32 = .{.{ 1, 0 }};
    const seq_lens: [1]i32 = .{32};
    const query_start_len: [2]i32 = .{ 0, 1 };

    const q = zml.Tensor.init(q_shape, .f32);
    const kv: KvCache = .{ .latent = zml.Tensor.init(kv_shape, .f32) };
    const sink = zml.Tensor.init(sink_shape, .f32);
    const topk = zml.Tensor.init(topk_shape, .i32);
    const tokens_pos = zml.Tensor.init(.{ .q = 1 }, .i32);

    var parameters_d: zml.Bufferized(Parameters) = .{ .triton = .{
        .block_table = try .fromBytes(std.testing.io, platform, parameters.triton.block_table.shape(), .replicated, std.mem.sliceAsBytes(&block_table)),
        .seq_lens = try .fromBytes(std.testing.io, platform, parameters.triton.seq_lens.shape(), .replicated, std.mem.sliceAsBytes(&seq_lens)),
        .query_start_len = try .fromBytes(std.testing.io, platform, parameters.triton.query_start_len.shape(), .replicated, std.mem.sliceAsBytes(&query_start_len)),
    } };
    defer zml.Buffer.deinitAll(Parameters, &parameters_d);
    var q_d = try zml.Buffer.fromBytes(std.testing.io, platform, q_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_d.deinit();
    var kv_d: zml.Bufferized(KvCache) = .{ .latent = try .fromBytes(std.testing.io, platform, kv_shape, .replicated, std.mem.sliceAsBytes(&kv_data)) };
    defer zml.Buffer.deinitAll(KvCache, &kv_d);
    var sink_d = try zml.Buffer.fromBytes(std.testing.io, platform, sink_shape, .replicated, std.mem.sliceAsBytes(&sink_data));
    defer sink_d.deinit();
    var topk_d = try zml.Buffer.fromBytes(std.testing.io, platform, topk_shape, .replicated, std.mem.sliceAsBytes(&topk_data));
    defer topk_d.deinit();
    var tokens_pos_d = try zml.Buffer.fromBytes(std.testing.io, platform, tokens_pos_shape, .replicated, std.mem.sliceAsBytes(&tokens_pos_data));
    defer tokens_pos_d.deinit();

    const exe = try platform.compileFn(
        std.testing.allocator,
        std.testing.io,
        Mla.pagedSparseAttention,
        .{ parameters, q, kv, sink, topk, tokens_pos, .{ .rope_rank = 64, .num_kv_splits = 2 } },
        .{},
    );
    defer exe.deinit();

    var output_d = try zml.testing.autoCall(
        std.testing.allocator,
        std.testing.io,
        &exe,
        Mla.pagedSparseAttention,
        .{ parameters_d, q_d, kv_d, sink_d, topk_d, tokens_pos_d },
    );
    defer zml.Buffer.deinitAll(zml.Tensor, &output_d);
}

test "execute stablehlo mla kernel" {
    const platform = zml.testing.env();
    const parameters = Parameters.init(.fromBackend(.{
        .backend = .stablehlo,
        .is_prefill = true,
        .batch_size = 1,
        .seq_len = 32,
        .max_num_pages = 2,
        .max_token_count = 1,
        .num_heads = 16,
        .num_kv_heads = 1,
        .head_dim = 128,
        .max_seqlen_q = 1,
    }));

    const q_shape = zml.Shape.init(.{ .q = 1, .h = 16, .hd = 128 }, .f32);
    const kv_shape = zml.Shape.init(.{ .page = 2, .k_chunk = 16, .hkv = 1, .hd = 128 }, .f32);
    const sink_shape = zml.Shape.init(.{ .h = 16 }, .f32);
    const topk_shape = zml.Shape.init(.{ .q = 1, .topk = 2 }, .i32);
    const tokens_pos_shape = zml.Shape.init(.{ .q = 1 }, .i32);

    var q_data: [1][16][128]f32 = undefined;
    @memset(std.mem.sliceAsBytes(&q_data), 0);
    var kv_data: [2][16][1][128]f32 = undefined;
    @memset(std.mem.sliceAsBytes(&kv_data), 0);
    for (&kv_data[1][0][0]) |*value| value.* = 10;
    for (&kv_data[1][1][0]) |*value| value.* = 11;
    var sink_data: [16]f32 = undefined;
    for (&sink_data) |*value| value.* = -std.math.inf(f32);
    const topk_data: [1][2]i32 = .{.{ 0, 1 }};
    const tokens_pos_data: [1]i32 = .{31};
    const block_table: [1][2]i32 = .{.{ 1, 0 }};
    const seq_lens: [1]i32 = .{32};
    const query_start_len: [2]i32 = .{ 0, 1 };

    const q = zml.Tensor.init(q_shape, .f32);
    const kv: KvCache = .{ .latent = zml.Tensor.init(kv_shape, .f32) };
    const sink = zml.Tensor.init(sink_shape, .f32);
    const topk = zml.Tensor.init(topk_shape, .i32);
    const tokens_pos = zml.Tensor.init(tokens_pos_shape, .i32);

    var parameters_d: zml.Bufferized(Parameters) = .{ .stablehlo = .{
        .block_table = try .fromBytes(std.testing.io, platform, parameters.stablehlo.block_table.shape(), .replicated, std.mem.sliceAsBytes(&block_table)),
        .seq_lens = try .fromBytes(std.testing.io, platform, parameters.stablehlo.seq_lens.shape(), .replicated, std.mem.sliceAsBytes(&seq_lens)),
        .query_start_len = try .fromBytes(std.testing.io, platform, parameters.stablehlo.query_start_len.shape(), .replicated, std.mem.sliceAsBytes(&query_start_len)),
    } };
    defer zml.Buffer.deinitAll(Parameters, &parameters_d);
    var q_d = try zml.Buffer.fromBytes(std.testing.io, platform, q_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_d.deinit();
    var kv_d: zml.Bufferized(KvCache) = .{ .latent = try .fromBytes(std.testing.io, platform, kv_shape, .replicated, std.mem.sliceAsBytes(&kv_data)) };
    defer zml.Buffer.deinitAll(KvCache, &kv_d);
    var sink_d = try zml.Buffer.fromBytes(std.testing.io, platform, sink_shape, .replicated, std.mem.sliceAsBytes(&sink_data));
    defer sink_d.deinit();
    var topk_d = try zml.Buffer.fromBytes(std.testing.io, platform, topk_shape, .replicated, std.mem.sliceAsBytes(&topk_data));
    defer topk_d.deinit();
    var tokens_pos_d = try zml.Buffer.fromBytes(std.testing.io, platform, tokens_pos_shape, .replicated, std.mem.sliceAsBytes(&tokens_pos_data));
    defer tokens_pos_d.deinit();

    const exe = try platform.compileFn(
        std.testing.allocator,
        std.testing.io,
        Mla.pagedSparseAttention,
        .{ parameters, q, kv, sink, topk, tokens_pos, .{ .rope_rank = 64 } },
        .{},
    );
    defer exe.deinit();

    var output_d = try zml.testing.autoCall(
        std.testing.allocator,
        std.testing.io,
        &exe,
        Mla.pagedSparseAttention,
        .{ parameters_d, q_d, kv_d, sink_d, topk_d, tokens_pos_d },
    );
    defer zml.Buffer.deinitAll(zml.Tensor, &output_d);
}
