const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const flashattn = @import("flashattn.zig");
const metal = @import("metal_attention.zig");
const tpu = @import("tpu_attention.zig");
const triton = @import("triton_attention.zig");
const triton2 = @import("triton2_attention.zig");

const PagedAttention = @This();

pub const Backend = enum {
    cuda_fa2,
    cuda_fa3,
    triton,
    triton2,
    mosaic_tpu,
    metal,
    // vanilla,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .cuda => .triton,
            .rocm => .triton,
            .oneapi => .triton,
            .tpu => .mosaic_tpu,
            .metal => .metal,
            // .cpu => .vanilla
            .cpu, .neuron => stdx.debug.panic("Paged attention is not supported on {s} yet", .{@tagName(platform.target)}),
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            // .vanilla => true,
            .triton, .triton2 => true,
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
    triton2: triton2.paged.Options,
    mosaic_tpu: tpu.mosaic_tpu.Options,
    metal: metal.paged.Options,

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
            .triton => .{
                .triton = .{
                    .batch_size = args.batch_size,
                    .max_num_pages = args.max_num_pages,
                    .max_seqlen_q = args.max_seqlen_q,
                    .is_prefill = args.is_prefill,
                },
            },
            .triton2 => .{
                .triton2 = .{
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
    triton2: triton2.paged.Parameters,
    mosaic_tpu: tpu.mosaic_tpu.Parameters,
    metal: metal.paged.Parameters,

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
        return switch (self) {
            .cuda_fa2 => |v| .{ .cuda_fa2 = v.onMemory(memory) },
            .cuda_fa3 => |v| .{ .cuda_fa3 = v.onMemory(memory) },
            .triton => |v| .{ .triton = v.onMemory(memory) },
            .triton2 => |v| .{ .triton2 = v.onMemory(memory) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = v.onMemory(memory) },
            .metal => |v| .{ .metal = v.onMemory(memory) },
        };
    }

    pub fn toMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
        return switch (self) {
            .cuda_fa2 => |v| .{ .cuda_fa2 = v.toMemory(memory) },
            .cuda_fa3 => |v| .{ .cuda_fa3 = v.toMemory(memory) },
            .triton => |v| .{ .triton = v.toMemory(memory) },
            .triton2 => |v| .{ .triton2 = v.toMemory(memory) },
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

    pub fn update(
        self: KvCache,
        new_k: zml.Tensor,
        new_v: zml.Tensor,
        slot_mapping: zml.Tensor,
        chunk_size: u32,
        backend: Backend,
    ) KvCache {
        const active_page, const k_chunk = getPageAndOffsetFromSlotMapping(slot_mapping, chunk_size);

        var kv: KvCache = switch (self) {
            .split => |split| switch (backend) {
                .cuda_fa2, .cuda_fa3, .triton, .triton2, .mosaic_tpu, .metal => .{ .split = .{
                    .k = split.k.scatterSlices(
                        .{ .page = active_page, .k_chunk = k_chunk },
                        new_k,
                        .{ .update_fn = zml.Tensor.ScatterOpts.override, .indices_are_unique = false, .indices_are_sorted = false },
                    ),
                    .v = split.v.scatterSlices(
                        .{ .page = active_page, .k_chunk = k_chunk },
                        new_v,
                        .{ .update_fn = zml.Tensor.ScatterOpts.override, .indices_are_unique = false, .indices_are_sorted = false },
                    ),
                } },
            },
            .dense => @panic("TODO"),
        };
        kv.split.k = kv.split.k.reuseBuffer(self.split.k);
        kv.split.v = kv.split.v.reuseBuffer(self.split.v);
        return kv;
    }

    pub fn getPageAndOffsetFromSlotMapping(slot_mapping: zml.Tensor, page_size: u32) [2]zml.Tensor {
        const page_index = slot_mapping.divByConst(page_size);
        const offset = slot_mapping.remainder(.scalar(page_size, slot_mapping.dtype()));
        return .{ page_index, offset };
    }
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
        .triton2 => |triton2_parameters| switch (kv_cache) {
            .split => |split| triton2.paged.pagedAttention(triton2_parameters, q, split.k, split.v, opts),
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
    // Check the reference implem is available
    if (!Backend.triton.isAvailable(platform)) return error.SkipZigTest;
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const batch_size = 8;
    const num_pages = 64;
    const page_size = 16;
    const max_num_pages = 16;
    const num_prefill = 1;
    const prefill_token_count = 32;
    const num_decode = 5;
    const query_token_count = prefill_token_count + num_decode;
    const dt: zml.DataType = .bf16;
    const tensors: struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, kv_cache: KvCache, token_index: zml.Tensor, slot_mapping: zml.Tensor } = .{
        .q = .init(.{ .b = query_token_count, .hkv = 2, .hg = 4, .hd = 32 }, dt),
        .k = .init(.{ .b = 8, .hkv = 2, .hd = 32 }, dt),
        .v = .init(.{ .b = 8, .hkv = 2, .hd = 32 }, dt),
        .kv_cache = .{
            .split = .{
                .k = .init(.{ .page = num_pages, .k_chunk = page_size, .hkv = 2, .hd = 32 }, dt),
                .v = .init(.{ .page = num_pages, .k_chunk = page_size, .hkv = 2, .hd = 32 }, dt),
            },
        },
        .token_index = .init(.{}, .u32),
        .slot_mapping = .init(.{ .b = 8 }, .u32),
    };

    const rng_q = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.q.shape(), .{} }, .{});
    defer rng_q.deinit();
    const rng_k = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.k.shape(), .{} }, .{});
    defer rng_k.deinit();
    const rng_kv_cache = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.kv_cache.split.k.shape(), .{} }, .{});
    defer rng_kv_cache.deinit();

    const update_exe = try platform.compileFn(allocator, io, KvCache.update, .{ tensors.kv_cache, tensors.k, tensors.v, tensors.slot_mapping, page_size, .triton }, .{});
    defer update_exe.deinit();

    const triton_options_args: Options.Args = .{
        .backend = .triton,
        .is_prefill = true,
        .batch_size = batch_size,
        .batch_size_prefill = num_prefill,
        .batch_size_decode = num_decode,
        .seq_len = 64,
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

    var kv_cache_k = try zml.testing.autoCall(allocator, io, &rng_kv_cache, zml.Tensor.Rng.normal, {});
    defer kv_cache_k.deinit();
    var kv_cache_v = try zml.testing.autoCall(allocator, io, &rng_kv_cache, zml.Tensor.Rng.normal, {});
    defer kv_cache_v.deinit();
    const kv_cache_d: zml.Bufferized(KvCache) = .{ .split = .{ .k = kv_cache_k, .v = kv_cache_v } };

    const shardings: []const zml.Sharding = &.{ platform.replicated_sharding, platform.shardings.get("model").? };
    const triton_exe = try platform.compileFn(
        allocator,
        io,
        pagedAttention,
        .{ triton_parameters, tensors.q, tensors.k, tensors.v, tensors.kv_cache, attn_opts },
        .{ .program_name = "paged_attention_triton", .shardings = shardings },
    );
    defer triton_exe.deinit();

    const block_table: [batch_size][max_num_pages]i32 = .{
        // prefilling pages 9-10
        .{ 0, 1, 2, 3, 4, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        // generation
        .{ 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        .{ 0, 1, 2, 3, 4, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1 },
        .{ 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 },
        .{ 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        .{ 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        .{ 0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        @splat(-1),
    };

    const seq_lens: [batch_size]i32 = .{
        4 * page_size,
        4 * page_size + 3,
        7 * page_size + 11,
        15 * page_size + 9,
        4 * page_size + 3,
        4 * page_size + 3,
        4 * page_size + 3,
        4 * page_size + 3,
    };

    // TODO: fix me. Currently this work cause all implementation ready k and v directly from the kv cache.
    // update kv cache
    // {
    //     const slot_mapping: [batch_size]u32 = .{
    //         10 * page_size,
    //         5 * page_size + 3,
    //         8 * page_size + 11,
    //         63 * page_size + 9,
    //         5 * page_size + 3,
    //         5 * page_size + 3,
    //         5 * page_size + 3,
    //         5 * page_size + 3,
    //     };
    //     const slot_mapping_d: zml.Buffer = try .fromBytes(io, platform, tensors.slot_mapping.shape(), .replicated, @ptrCast(&slot_mapping));
    //     kv_cache_d = try zml.testing.autoCall(allocator, io, &update_exe, KvCache.update, .{ kv_cache_d, new_k, new_v, slot_mapping_d });
    // }
    const query_start_len: [batch_size + 1]i32 = .{ 0, 32, 33, 34, 35, 36, 37, 37, 37 };

    const triton_parameters_d: zml.Bufferized(Parameters) = .{ .triton = .{
        .block_table = try .fromBytes(io, platform, triton_parameters.triton.block_table.shape(), .replicated, @ptrCast(&block_table)),
        .seq_lens = try .fromBytes(io, platform, triton_parameters.triton.seq_lens.shape(), .replicated, @ptrCast(&seq_lens)),
        .query_start_len = try .fromBytes(io, platform, triton_parameters.triton.query_start_len.shape(), .replicated, @ptrCast(&query_start_len)),
    } };

    const Benchmark = struct {
        const warmup_iterations = 5;
        const timed_iterations = 50;

        fn run(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            exe: *const zml.Exe,
            parameters: zml.Bufferized(Parameters),
            q_: zml.Buffer,
            k_: zml.Buffer,
            v_: zml.Buffer,
            kv_cache: zml.Bufferized(KvCache),
        ) !u64 {
            for (0..warmup_iterations) |_| {
                var output = try zml.testing.autoCall(allocator_, io_, exe, pagedAttention, .{ parameters, q_, k_, v_, kv_cache });
                output.deinit();
            }

            const start: std.Io.Timestamp = .now(io_, .awake);
            for (0..timed_iterations) |_| {
                var output = try zml.testing.autoCall(allocator_, io_, exe, pagedAttention, .{ parameters, q_, k_, v_, kv_cache });
                output.deinit();
            }
            return @intCast(@divTrunc(start.untilNow(io_, .awake).toNanoseconds(), timed_iterations));
        }
    };

    const triton_d = try zml.testing.autoCall(allocator, io, &triton_exe, pagedAttention, .{ triton_parameters_d, q, new_k, new_v, kv_cache_d });
    const triton_h: zml.Slice = try triton_d.toSliceAlloc(allocator, io);
    defer triton_h.free(allocator);
    const triton_avg_ns = try Benchmark.run(allocator, io, &triton_exe, triton_parameters_d, q, new_k, new_v, kv_cache_d);
    std.log.warn("paged attention triton: {d:.2} us average over {} runs", .{
        @as(f64, @floatFromInt(triton_avg_ns)) / std.time.ns_per_us,
        Benchmark.timed_iterations,
    });

    var triton2_3d_options_args = triton_options_args;
    triton2_3d_options_args.backend = .triton2;
    triton2_3d_options_args.is_prefill = false;
    const triton2_3d_parameters: Parameters = .init(.fromBackend(triton2_3d_options_args));
    const triton2_3d_exe = try platform.compileFn(
        allocator,
        io,
        pagedAttention,
        .{ triton2_3d_parameters, tensors.q, tensors.k, tensors.v, tensors.kv_cache, attn_opts },
        .{ .program_name = "paged_attention_triton2_3d", .shardings = shardings },
    );
    defer triton2_3d_exe.deinit();
    const triton2_3d_parameters_d: zml.Bufferized(Parameters) = .{ .triton2 = .{
        .block_table = triton_parameters_d.triton.block_table,
        .seq_lens = triton_parameters_d.triton.seq_lens,
        .query_start_len = triton_parameters_d.triton.query_start_len,
    } };
    var triton2_3d_d = try zml.testing.autoCall(allocator, io, &triton2_3d_exe, pagedAttention, .{ triton2_3d_parameters_d, q, new_k, new_v, kv_cache_d });
    defer triton2_3d_d.deinit();
    try zml.testing.expectClose(io, triton_h, triton2_3d_d, .{
        .absolute_tolerance = 1e-3,
        .relative_tolerance = 1e-2,
        .epsilon_relative = 1e-6,
    });
    const triton2_3d_avg_ns = try Benchmark.run(allocator, io, &triton2_3d_exe, triton2_3d_parameters_d, q, new_k, new_v, kv_cache_d);
    std.log.warn("paged attention triton2_3d: {d:.2} us average over {} runs", .{
        @as(f64, @floatFromInt(triton2_3d_avg_ns)) / std.time.ns_per_us,
        Benchmark.timed_iterations,
    });

    for (std.enums.values(Backend)) |backend| {
        if (!backend.isAvailable(platform)) continue;
        if (backend == .triton) continue;

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

        const parameters_d: zml.Bufferized(Parameters) = switch (backend) {
            .triton => unreachable,
            .triton2 => .{ .triton2 = .{
                .block_table = triton_parameters_d.triton.block_table,
                .seq_lens = triton_parameters_d.triton.seq_lens,
                .query_start_len = triton_parameters_d.triton.query_start_len,
            } },
            // No materializer implemented for cuda fa3
            .cuda_fa3 => return error.SkipZigTest,
            .cuda_fa2 => cuda_fa2: {
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
                    .block_table_prefill = try .fromBytes(io, platform, parameters.cuda_fa2.mixed.block_table_prefill.shape(), .replicated, @ptrCast(&block_table_prefill)),
                    .cu_seqlens_q_prefill = try .fromBytes(io, platform, parameters.cuda_fa2.mixed.cu_seqlens_q_prefill.shape(), .replicated, @ptrCast(&cu_seqlens_q_prefill)),
                    .seqused_k_prefill = try .fromBytes(io, platform, parameters.cuda_fa2.mixed.seqused_k_prefill.shape(), .replicated, @ptrCast(&seqused_k_prefill)),

                    .block_table_decode = try .fromBytes(io, platform, parameters.cuda_fa2.mixed.block_table_decode.shape(), .replicated, @ptrCast(&block_table_decode)),
                    .cu_seqlens_q_decode = try .fromBytes(io, platform, parameters.cuda_fa2.mixed.cu_seqlens_q_decode.shape(), .replicated, @ptrCast(&cu_seqlens_q_decode)),
                    .seqused_k_decode = try .fromBytes(io, platform, parameters.cuda_fa2.mixed.seqused_k_decode.shape(), .replicated, @ptrCast(&seqused_k_decode)),

                    .metadata = .{ .decode_offset = try .scalar(io, platform, prefill_token_count, .i32) },
                } } };
            },
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

        const output_d = try zml.testing.autoCall(allocator, io, &exe, pagedAttention, .{ parameters_d, q, new_k, new_v, kv_cache_d });
        try zml.testing.expectClose(io, triton_h, output_d, .{
            .absolute_tolerance = 1e-3,
            .relative_tolerance = 1e-2,
            .epsilon_relative = 1e-6,
        });

        const backend_avg_ns = try Benchmark.run(allocator, io, &exe, parameters_d, q, new_k, new_v, kv_cache_d);
        const speedup = @as(f64, @floatFromInt(triton_avg_ns)) / @as(f64, @floatFromInt(backend_avg_ns));
        const improvement = (1.0 - @as(f64, @floatFromInt(backend_avg_ns)) / @as(f64, @floatFromInt(triton_avg_ns))) * 100.0;
        std.log.warn("paged attention {t}: {d:.2} us average over {} runs ({d:.2}x, {d:.1}% vs triton)", .{
            backend,
            @as(f64, @floatFromInt(backend_avg_ns)) / std.time.ns_per_us,
            Benchmark.timed_iterations,
            speedup,
            improvement,
        });
    }
}
