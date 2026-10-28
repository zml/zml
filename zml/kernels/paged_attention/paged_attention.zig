const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../zml.zig");
const cute = @import("cute/cute.zig");
const metal = @import("metal/metal.zig");
const tpu = @import("mosaic_tpu/mosaic_tpu.zig");
const triton = @import("triton/triton.zig");
pub const tcl = @import("tcl/tcl.zig");
const stablehlo = @import("stablehlo/stablehlo.zig");

const PagedAttention = @This();

test {
    _ = tcl;
    std.testing.refAllDecls(Backend);
    std.testing.refAllDecls(Options);
    std.testing.refAllDecls(Parameters);
    std.testing.refAllDecls(KvCache);
}

pub const Backend = enum {
    cute,
    triton,
    mosaic_tpu,
    metal,
    stablehlo,
    tcl,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .cuda => .triton,
            .rocm => .triton,
            .oneapi => .triton,
            .tpu => .mosaic_tpu,
            .metal => .metal,
            .cpu => .stablehlo,
            .furiosa => .tcl,
            .neuron => stdx.debug.panic("Paged attention is not supported on {s} yet", .{@tagName(platform.target)}),
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .cute => cute.isAvailable(platform),
            .stablehlo => true,
            .tcl => platform.target == .furiosa,
            .triton => platform.target != .cpu and platform.target != .furiosa,
            .metal => platform.target == .metal,
            .mosaic_tpu => platform.target == .tpu,
        };
    }
};

pub const Options = union(Backend) {
    cute: cute.Options,
    triton: triton.paged.Options,
    mosaic_tpu: tpu.mosaic_tpu.Options,
    metal: metal.Options,
    stablehlo: triton.paged.Options,
    tcl: tcl.paged.Options,

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
            inline .cute, .triton, .metal, .stablehlo, .tcl => |t| @unionInit(Options, @tagName(t), .{
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
    cute: cute.Parameters,
    triton: triton.paged.Parameters,
    mosaic_tpu: tpu.mosaic_tpu.Parameters,
    metal: metal.Parameters,
    stablehlo: triton.paged.Parameters,
    tcl: tcl.paged.Parameters,

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

pub const AttentionMask = union(enum) {
    standard: struct {
        /// Restrict attention to keys at or before the query position.
        causal: bool = true,
        /// Optional left window size, including the current token.
        /// Non-causal attention also includes every later valid key.
        sliding_window: ?u32 = null,
    },
    /// Attend to keys at specific positions.
    indices: zml.Tensor,

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
    sink: ?zml.Tensor = null,
};

pub const KvCache = union(enum) {
    split: struct {
        k: zml.Tensor,
        v: zml.Tensor,
    },
    // TODO: rename `dense` to `fused`.
    dense: zml.Tensor,

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
                .cute, .triton, .mosaic_tpu, .metal, .stablehlo, .tcl => .{
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
            .dense => @panic("TODO"),
        };
    }
};

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, kv_cache: KvCache, opts: AttentionOptions) zml.Tensor {
    _ = k;
    _ = v;
    return switch (parameters) {
        .cute => |p| switch (kv_cache) {
            .split => |split| cute.pagedAttention(p, q, split.k, split.v, opts),
            .dense => @panic("CuTe paged attention requires split KV pages"),
        },
        .triton => |triton_parameters| switch (kv_cache) {
            .split => |split| triton.paged.pagedAttention(triton_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
        },
        .mosaic_tpu => |mosaic_tpu_parameters| switch (kv_cache) {
            .split => std.debug.panic("split KV pages is not supported with the mosaic_tpu backend", .{}),
            .dense => tpu.mosaic_tpu.pagedAttention(mosaic_tpu_parameters, q, kv_cache.dense, opts),
        },
        .metal => |metal_parameters| switch (kv_cache) {
            .split => |split| metal.pagedAttention(metal_parameters, q, split.k, split.v, opts),
            .dense => std.debug.panic("fused KV pages are only supported with the mosaic_tpu backend", .{}),
        },
        .stablehlo => |params| stablehlo.pagedAttention(params, q, kv_cache, opts),
        .tcl => |params| switch (kv_cache) {
            .split => |split| tcl.paged.pagedAttention(params, q, split.k, split.v, opts),
            .dense => std.debug.panic("TCL paged attention takes separate K and V pages", .{}),
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
        .replicated_mesh = undefined,
        .meshes = .empty,
        .io_impl = .threaded,
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
        .replicated_mesh = undefined,
        .meshes = .empty,
        .io_impl = .threaded,
    };

    try std.testing.expectEqual(Backend.triton, Backend.auto(&platform));
}

test pagedAttention {
    const platform = zml.testing.env();
    if (!Backend.triton.isAvailable(platform)) return error.SkipZigTest;
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
    const model_meshe = platform.meshes.get("model").?;
    const partition = .{ .hkv = .model };
    const tensors: struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, kv_cache: KvCache } = .{
        .q = zml.Tensor.fromShape(.init(.{ .b = query_token_count, .hkv = 4, .hg = 4, .hd = 32 }, dt)).withPartitioning(model_meshe, partition),
        .k = zml.Tensor.fromShape(.init(.{ .b = query_token_count, .hkv = 4, .hd = 32 }, dt)).withPartitioning(model_meshe, partition),
        .v = zml.Tensor.fromShape(.init(.{ .b = query_token_count, .hkv = 4, .hd = 32 }, dt)).withPartitioning(model_meshe, partition),
        .kv_cache = .{
            .split = .{
                .k = zml.Tensor.fromShape(.init(.{ .page = num_pages, .k_chunk = page_size, .hkv = 4, .hd = 32 }, dt)).withPartitioning(model_meshe, partition),
                .v = zml.Tensor.fromShape(.init(.{ .page = num_pages, .k_chunk = page_size, .hkv = 4, .hd = 32 }, dt)).withPartitioning(model_meshe, partition),
            },
        },
    };

    const meshes: []const *const zml.Mesh = &.{ platform.replicated_mesh, platform.meshes.get("model").? };
    const rng_q = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.q.shape(), .{} }, .{ .meshes = meshes });
    defer rng_q.deinit();
    const rng_k = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.k.shape(), .{} }, .{ .meshes = meshes });
    defer rng_k.deinit();
    const rng_kv_cache = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.kv_cache.split.k.shape(), .{} }, .{ .meshes = meshes });
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
    var q = try rng_q.eval(allocator, io, {});
    defer q.deinit();
    var new_k = try rng_k.eval(allocator, io, {});
    defer new_k.deinit();
    var new_v = try rng_k.eval(allocator, io, {});
    defer new_v.deinit();

    var kv_cache_d: zml.Bufferized(KvCache) = .{ .split = undefined };
    kv_cache_d.split.k = try rng_kv_cache.eval(allocator, io, .{});
    defer kv_cache_d.split.k.deinit();
    kv_cache_d.split.v = try rng_kv_cache.eval(allocator, io, .{});
    defer kv_cache_d.split.v.deinit();

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
        .block_table = try .fromBytes(io, platform, triton_parameters.triton.block_table.shape(), @ptrCast(&block_table)),
        .seq_lens = try .fromBytes(io, platform, triton_parameters.triton.seq_lens.shape(), @ptrCast(&seq_lens)),
        .query_start_len = try .fromBytes(io, platform, triton_parameters.triton.query_start_len.shape(), @ptrCast(&query_start_len)),
    } };
    defer zml.Buffer.deinitAll(&triton_parameters_d);

    const all_backends = std.enums.values(Backend);
    // stablehlo ignores these options, mosaic_tpu requires a dense
    // KV cache instead of the split cache used by this test.
    const option_sensitive_backends = [_]Backend{ .triton, .cute, .metal, .tcl };
    const test_cases = [_]struct {
        name: []const u8,
        attention_options: AttentionOptions,
        backends: []const Backend,
    }{
        .{
            .name = "unbounded",
            .attention_options = .{ .mask = .{ .standard = .{} } },
            .backends = all_backends,
        },
        .{
            .name = "non_causal",
            .attention_options = .{ .mask = .{ .standard = .{ .causal = false } } },
            .backends = &option_sensitive_backends,
        },
        .{
            .name = "sliding_window",
            .attention_options = .{ .mask = .{ .standard = .{ .sliding_window = page_size } } },
            .backends = &option_sensitive_backends,
        },
    };

    for (test_cases) |test_case| {
        var results_per_backend: std.enums.EnumArray(Backend, ?zml.Slice) = .initFill(null);
        defer {
            for (std.enums.values(Backend)) |backend| {
                if (results_per_backend.get(backend)) |slice| slice.free(allocator);
            }
        }

        for (test_case.backends) |backend| {
            if (!backend.isAvailable(platform)) {
                std.log.warn("paged_attention backend {t} not available", .{backend});
                continue;
            }
            std.log.warn("Testing paged_attention {s} with backend {t}", .{ test_case.name, backend });

            var backend_options_args = triton_options_args;
            backend_options_args.backend = backend;
            const parameters: Parameters = .init(.fromBackend(backend_options_args));

            const exe = try platform.compileFn(
                allocator,
                io,
                pagedAttention,
                .{ parameters, tensors.q, tensors.k, tensors.v, tensors.kv_cache, test_case.attention_options },
                .{ .program_name = try std.fmt.allocPrint(arena, "paged_attention_{s}_{t}", .{ test_case.name, backend }), .meshes = meshes },
            );
            defer exe.deinit();

            const parameters_d: zml.Bufferized(Parameters) = switch (parameters) {
                .triton => triton_parameters_d,
                inline .cute, .metal, .mosaic_tpu, .stablehlo, .tcl => |_, t| @unionInit(zml.Bufferized(Parameters), @tagName(t), .{
                    .block_table = triton_parameters_d.triton.block_table,
                    .seq_lens = triton_parameters_d.triton.seq_lens,
                    .query_start_len = triton_parameters_d.triton.query_start_len,
                }),
            };

            var output_d = try exe.eval(allocator, io, .{ parameters_d, q, new_k, new_v, kv_cache_d, .{ .mask = .{ .standard = {} }, .sink = null } });
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
                    std.log.err("test pagedAttention {s} failed on backend {t} against reference {t}", .{
                        test_case.name,
                        backend,
                        reference_backend,
                    });
                },
                else => |e| return e,
            };
        }

        if (num_failed > 0) return error.TestUnexpectedResult;
    }
}
