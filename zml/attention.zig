const std = @import("std");

pub const attnd = @import("attention/attnd.zig");
pub const flashattn = @import("attention/flashattn.zig");
pub const metal = @import("attention/metal_attention.zig");
pub const nki = @import("attention/nki/attention.zig");
pub const paged_attention = @import("attention/paged_attention.zig");
pub const tpu = @import("attention/tpu_attention.zig");
pub const triton = @import("attention/triton_attention.zig");
pub const triton_kernels = @import("attention/triton_kernels/unified_attention.zig");
const zml = @import("zml.zig");

test {
    std.testing.refAllDecls(@This());
}

pub const Backend = enum {
    vanilla,
    attnd,
    nki,
    cuda_fa2,
    cuda_fa3,
    metal_fa,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .cuda => b: {
                const first_device = platform.pjrt_client.devices(platform.pjrt_api)[0];

                if (zml.platform.cuda.tryGetComputeCapabilities(platform, first_device)) |cc| {
                    break :b if (std.mem.eql(u8, cc, "9.0"))
                        .cuda_fa3
                    else
                        .cuda_fa2;
                }

                break :b .vanilla;
            },
            .neuron => .nki,
            .metal => .metal_fa,
            .cpu, .rocm, .tpu, .oneapi => .vanilla,
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .vanilla => true,
            .attnd => true, // attnd runs over network
            .nki => platform.target == .neuron,
            .metal_fa => platform.target == .metal,
            .cuda_fa2 => platform.target == .cuda,
            .cuda_fa3 => {
                if (platform.target != .cuda) return false;
                const first_device = platform.pjrt_client.devices(platform.pjrt_api)[0];
                const cc = zml.platform.cuda.tryGetComputeCapabilities(platform, first_device) orelse return false;
                return std.mem.eql(u8, cc, "9.0");
            },
        };
    }

    pub fn supportsHeadDim(backend: Backend, head_dim: i64) bool {
        return switch (backend) {
            .vanilla, .attnd, .nki, .metal_fa => head_dim > 0,
            .cuda_fa2 => head_dim >= 8 and head_dim <= 256 and @rem(head_dim, 8) == 0,
            .cuda_fa3 => head_dim == 64 or head_dim == 96 or head_dim == 128 or head_dim == 192 or head_dim == 256,
        };
    }

    /// Dense FA workspace needs head dim multiple of 32. `supportsHeadDim` is the broader ABI.
    pub fn supportsDenseHeadDim(backend: Backend, head_dim: i64) bool {
        return switch (backend) {
            .cuda_fa2 => head_dim >= 32 and head_dim <= 256 and @rem(head_dim, 32) == 0,
            .cuda_fa3 => backend.supportsHeadDim(head_dim),
            else => backend.supportsHeadDim(head_dim),
        };
    }
};

pub const Parameters = union(Backend) {
    vanilla: void,
    attnd: attnd.Parameters,
    nki: nki.Parameters,
    cuda_fa2: flashattn.fa2.Parameters,
    cuda_fa3: flashattn.fa3.Parameters,
    metal_fa: void,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        attnd: void,
        nki: nki.Parameters,
        cuda_fa2: flashattn.fa2.Parameters.InitOptions,
        cuda_fa3: flashattn.fa3.Parameters.InitOptions,
        metal_fa: void,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .vanilla => .{ .vanilla = {} },
                .attnd => @panic("Must be initialized manually"),
                .nki => .{ .nki = .init() },
                .cuda_fa2 => .{ .cuda_fa2 = .{} },
                .cuda_fa3 => .{ .cuda_fa3 = .{} },
                .metal_fa => .{ .metal_fa = {} },
            };
        }
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .vanilla => .{ .vanilla = {} },
            .attnd => @panic("Must be initialized manually"),
            .nki => |v| .{ .nki = v },
            .cuda_fa2 => |v| .{ .cuda_fa2 = .init(v) },
            .cuda_fa3 => |v| .{ .cuda_fa3 = .init(v) },
            .metal_fa => .{ .metal_fa = {} },
        };
    }
};

pub const Metadata = union(Backend) {
    vanilla: void,
    attnd: attnd.Metadata,
    nki: void,
    cuda_fa2: flashattn.fa2.Metadata,
    cuda_fa3: flashattn.fa3.Metadata,
    metal_fa: metal.Metadata,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        attnd: void,
        nki: void,
        cuda_fa2: flashattn.fa2.Metadata.InitOptions,
        cuda_fa3: flashattn.fa3.Metadata.InitOptions,
        metal_fa: void,

        pub fn fromBackend(backend: Backend, seqlen: i64, num_heads: i64) InitOptions {
            return switch (backend) {
                .vanilla => .{ .vanilla = {} },
                .attnd => .{ .attnd = {} },
                .nki => .{ .nki = {} },
                .cuda_fa2 => .{ .cuda_fa2 = .{ .seqlen = seqlen, .num_heads = num_heads } },
                .cuda_fa3 => .{ .cuda_fa3 = .{ .seqlen = seqlen, .num_heads = num_heads } },
                .metal_fa => .{ .metal_fa = {} },
            };
        }
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .vanilla => .{ .vanilla = {} },
            .attnd => @panic("Must be initialized manually"),
            .nki => .{ .nki = {} },
            .cuda_fa2 => |o| .{ .cuda_fa2 = flashattn.fa2.Metadata.init(o) },
            .cuda_fa3 => |o| .{ .cuda_fa3 = flashattn.fa3.Metadata.init(o) },
            .metal_fa => .{ .metal_fa = .init() },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(Metadata) {
        return switch (self) {
            .vanilla => .{ .vanilla = {} },
            .nki => .{ .nki = {} },
            inline else => |v, tag| @unionInit(zml.Bufferized(Metadata), @tagName(tag), try v.initBuffer(io, platform, sharding)),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .vanilla => {},
            .attnd => |*v| attnd.Metadata.deinitBuffer(v),
            .nki => {},
            .cuda_fa2 => |*v| flashattn.fa2.Metadata.deinitBuffer(v),
            .cuda_fa3 => |*v| flashattn.fa3.Metadata.deinitBuffer(v),
            .metal_fa => |*v| metal.Metadata.deinitBuffer(v),
        }
    }
};

/// Causal attention as used in transformers.
///
/// **Shapes**:
///   - q, result: .{ .q, .h, .hd }
///   - k, v:      .{ .k, .h, .hd }
///
/// Where:
///   - .h is the number of head
///   - .q is the number of queries
///   - .k is the number of keys
///   - .hd is the head dimension
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters) zml.Tensor {
    return switch (parameters) {
        .vanilla => b: {
            // Generate the attention mask.
            const seq_len = k.dim(.k);
            var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, q.dtype(), null);

            // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
            // then slice into it, but XLA is able to optimize this correctly.
            attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()), token_index.appendAxes(.{.coord}), .{});
            const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask });
            break :b attn_output;
        },
        .attnd => attnd.causalAttention(q, k, v, token_index, metadata.attnd, parameters.attnd),
        .nki => |params| nki.attention(q, k, v, token_index, params),
        .cuda_fa2 => flashattn.fa2.attention(q, k, v, token_index, metadata.cuda_fa2, parameters.cuda_fa2),
        .cuda_fa3 => flashattn.fa3.attention(q, k, v, token_index, metadata.cuda_fa3, parameters.cuda_fa3),
        .metal_fa => metal.attention(q, k, v, token_index, metadata.metal_fa),
    };
}

pub const DenseOpts = struct {
    is_causal: bool = false,
};

/// Dense attention (causal or bidirectional).
/// CUDA FA2 / FA3 when available and legal; otherwise `zml.nn.sdpa`.
pub fn dense(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, backend: Backend, opts: DenseOpts) zml.Tensor {
    if (denseCanUseFlash(q, k, v, backend)) {
        return switch (backend) {
            .cuda_fa2 => flashattn.fa2.dense(q, k, v, .{ .is_causal = opts.is_causal }),
            .cuda_fa3 => flashattn.fa3.dense(q, k, v, .{ .is_causal = opts.is_causal }),
            else => unreachable,
        };
    }
    const mask = if (opts.is_causal)
        zml.nn.causalAttnMask(.{ .q = q.dim(.q), .k = k.dim(.k) }, q.dtype(), null)
    else
        null;
    return zml.nn.sdpa(q, k, v, .{ .attn_mask = mask });
}

fn denseCanUseFlash(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, backend: Backend) bool {
    switch (backend) {
        .cuda_fa2, .cuda_fa3 => {},
        else => return false,
    }
    if (!q.shape().hasTags(.{ .q, .h, .hd })) return false;
    if (!k.shape().hasTags(.{ .k, .h, .hd })) return false;
    if (!v.shape().hasTags(.{ .k, .h, .hd })) return false;

    const q_b = q.shape().hasTag(.b) != null;
    const k_b = k.shape().hasTag(.b) != null;
    const v_b = v.shape().hasTag(.b) != null;
    if (q_b != k_b or q_b != v_b) return false;
    if (q_b and (q.dim(.b) != k.dim(.b) or q.dim(.b) != v.dim(.b) or q.dim(.b) <= 0)) return false;

    if (q.dim(.q) <= 0 or k.dim(.k) <= 0 or v.dim(.k) != k.dim(.k)) return false;
    // Dense FA2 varlen is only numerically validated for Q==K. Unequal
    // lengths disagree with SDPA on CUDA; use vanilla until that path is fixed.
    if (q.dim(.q) != k.dim(.k)) return false;
    if (q.dim(.h) <= 0 or k.dim(.h) <= 0 or v.dim(.h) != k.dim(.h)) return false;
    if (@rem(q.dim(.h), k.dim(.h)) != 0) return false;
    if (q.dim(.hd) != k.dim(.hd) or q.dim(.hd) != v.dim(.hd)) return false;
    if (!backend.supportsDenseHeadDim(q.dim(.hd))) return false;

    if (q.dtype() != k.dtype() or q.dtype() != v.dtype()) return false;
    if (q.dtype() != .f16 and q.dtype() != .bf16) return false;

    const compiler = zml.Compiler.currentOrNull() orelse return false;
    return backend.isAvailable(compiler.platform);
}

test "attention: q=1,qh=64,kh=8" {
    try testAttention(
        .init(.{ .q = 1, .h = 64, .hd = 64 }, .bf16),
        .init(.{ .k = 64, .h = 8, .hd = 64 }, .bf16),
        &.{63},
    );

    try testAttention(
        .init(.{ .q = 1, .h = 64, .hd = 64 }, .bf16),
        .init(.{ .k = 64, .h = 8, .hd = 64 }, .bf16),
        &.{36},
    );
}

test "attention: b=4,q=1,qh=64,kh=8" {
    // Full attention
    try testAttention(
        .init(.{ .b = 4, .q = 1, .h = 64, .hd = 64 }, .bf16),
        .init(.{ .b = 4, .k = 64, .h = 8, .hd = 64 }, .bf16),
        &.{ 63, 63, 63, 63 },
    );

    // Partial attention
    try testAttention(
        .init(.{ .b = 4, .q = 1, .h = 64, .hd = 64 }, .bf16),
        .init(.{ .b = 4, .k = 64, .h = 8, .hd = 64 }, .bf16),
        &.{ 61, 57, 23, 63 },
    );
}

test "attention: q=1,qh=8,kh=8" {
    try testAttention(
        .init(.{ .q = 1, .h = 8, .hd = 64 }, .bf16),
        .init(.{ .k = 64, .h = 8, .hd = 64 }, .bf16),
        &.{62},
    );
    try testAttention(
        .init(.{ .q = 1, .h = 8, .hd = 64 }, .bf16),
        .init(.{ .k = 64, .h = 8, .hd = 64 }, .bf16),
        &.{63},
    );
}

test "attention: q=8,qh=64,kh=8" {
    try testAttention(
        .init(.{ .q = 8, .h = 64, .hd = 64 }, .bf16),
        .init(.{ .k = 64, .h = 8, .hd = 64 }, .bf16),
        &.{56},
    );
}

test "dense attention: non-causal hd=128" {
    try testDense(
        .init(.{ .b = 1, .q = 16, .h = 8, .hd = 128 }, .bf16),
        .init(.{ .b = 1, .k = 16, .h = 8, .hd = 128 }, .bf16),
        false,
    );
}

test "dense attention: causal hd=128" {
    try testDense(
        .init(.{ .b = 1, .q = 16, .h = 8, .hd = 128 }, .bf16),
        .init(.{ .b = 1, .k = 16, .h = 8, .hd = 128 }, .bf16),
        true,
    );
}

test "dense attention: short and long buckets" {
    try testDense(
        .init(.{ .q = 8, .h = 8, .hd = 128 }, .bf16),
        .init(.{ .k = 8, .h = 8, .hd = 128 }, .bf16),
        false,
    );
    try testDense(
        .init(.{ .q = 64, .h = 8, .hd = 128 }, .bf16),
        .init(.{ .k = 64, .h = 8, .hd = 128 }, .bf16),
        false,
    );
}

test "dense attention: f16 hd=128" {
    try testDense(
        .init(.{ .b = 1, .q = 16, .h = 8, .hd = 128 }, .f16),
        .init(.{ .b = 1, .k = 16, .h = 8, .hd = 128 }, .f16),
        false,
    );
}

test "dense attention: supported head dimensions" {
    inline for (.{ 64, 96, 128, 256 }) |head_dim| {
        try testDense(
            .init(.{ .b = 1, .q = 16, .h = 8, .hd = head_dim }, .bf16),
            .init(.{ .b = 1, .k = 16, .h = 8, .hd = head_dim }, .bf16),
            false,
        );
    }
}

test "dense attention: hd=70 falls back to sdpa" {
    try std.testing.expect(!Backend.supportsHeadDim(.cuda_fa2, 70));
    try std.testing.expect(!Backend.supportsDenseHeadDim(.cuda_fa2, 70));
    try std.testing.expect(!Backend.supportsHeadDim(.cuda_fa3, 70));
    try testDense(
        .init(.{ .b = 1, .q = 16, .h = 8, .hd = 70 }, .bf16),
        .init(.{ .b = 1, .k = 16, .h = 8, .hd = 70 }, .bf16),
        false,
    );
}

test "dense attention: batch size > 1" {
    try testDense(
        .init(.{ .b = 2, .q = 8, .h = 4, .hd = 64 }, .bf16),
        .init(.{ .b = 2, .k = 8, .h = 4, .hd = 64 }, .bf16),
        false,
    );
    try testDense(
        .init(.{ .b = 2, .q = 8, .h = 4, .hd = 64 }, .bf16),
        .init(.{ .b = 2, .k = 8, .h = 4, .hd = 64 }, .bf16),
        true,
    );
}

test "dense attention: f32 falls back to sdpa" {
    try testDense(
        .init(.{ .b = 1, .q = 8, .h = 4, .hd = 64 }, .f32),
        .init(.{ .b = 1, .k = 8, .h = 4, .hd = 64 }, .f32),
        false,
    );
}

test "dense attention: rounded FA2 head dims fall back to sdpa" {
    try std.testing.expect(Backend.supportsHeadDim(.cuda_fa2, 48));
    try std.testing.expect(Backend.supportsHeadDim(.cuda_fa2, 80));
    try std.testing.expect(Backend.supportsHeadDim(.cuda_fa2, 144));
    try std.testing.expect(!Backend.supportsDenseHeadDim(.cuda_fa2, 48));
    try std.testing.expect(!Backend.supportsDenseHeadDim(.cuda_fa2, 80));
    try std.testing.expect(!Backend.supportsDenseHeadDim(.cuda_fa2, 144));
    try std.testing.expect(!Backend.supportsHeadDim(.cuda_fa3, 48));
    try std.testing.expect(!Backend.supportsHeadDim(.cuda_fa3, 80));
    try std.testing.expect(!Backend.supportsHeadDim(.cuda_fa3, 144));
    inline for (.{ 48, 80, 144 }) |head_dim| {
        try testDense(
            .init(.{ .b = 1, .q = 8, .h = 4, .hd = head_dim }, .bf16),
            .init(.{ .b = 1, .k = 8, .h = 4, .hd = head_dim }, .bf16),
            false,
        );
    }
}

test "Backend.supportsHeadDim matches FA C ABI" {
    try std.testing.expect(Backend.supportsHeadDim(.cuda_fa2, 8));
    try std.testing.expect(Backend.supportsHeadDim(.cuda_fa2, 24));
    try std.testing.expect(Backend.supportsHeadDim(.cuda_fa2, 72));
    try std.testing.expect(!Backend.supportsHeadDim(.cuda_fa2, 7));
    try std.testing.expect(!Backend.supportsHeadDim(.cuda_fa2, 70));
    try std.testing.expect(!Backend.supportsHeadDim(.cuda_fa2, 264));
    try std.testing.expect(Backend.supportsHeadDim(.cuda_fa3, 64));
    try std.testing.expect(!Backend.supportsHeadDim(.cuda_fa3, 48));
}

test "dense attention: gqa causal hd=128" {
    try testDense(
        .init(.{ .q = 16, .h = 16, .hd = 128 }, .bf16),
        .init(.{ .k = 16, .h = 4, .hd = 128 }, .bf16),
        true,
    );
}

test "dense attention: Q != K" {
    try testDense(
        .init(.{ .b = 1, .q = 16, .h = 8, .hd = 64 }, .bf16),
        .init(.{ .b = 1, .k = 32, .h = 8, .hd = 64 }, .bf16),
        false,
    );
    try testDense(
        .init(.{ .b = 2, .q = 8, .h = 16, .hd = 64 }, .bf16),
        .init(.{ .b = 2, .k = 16, .h = 4, .hd = 64 }, .bf16),
        true,
    );
}

test "dense attention: f16 causal gqa" {
    try testDense(
        .init(.{ .b = 1, .q = 16, .h = 8, .hd = 64 }, .f16),
        .init(.{ .b = 1, .k = 16, .h = 4, .hd = 64 }, .f16),
        true,
    );
}

pub fn testDense(q_shape: zml.Shape, k_shape: zml.Shape, is_causal: bool) !void {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    const platform = zml.testing.env();

    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const tensors: struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor } = .{
        .q = .fromShape(q_shape),
        .k = .fromShape(k_shape),
        .v = .fromShape(k_shape),
    };

    const rng_q = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.q.shape(), .{ .mean = 0, .stddev = 1 } }, .{});
    defer rng_q.deinit();
    const rng_k = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.k.shape(), .{ .mean = 0, .stddev = 1 } }, .{});
    defer rng_k.deinit();

    var q = try zml.testing.autoCall(allocator, io, &rng_q, zml.Tensor.Rng.normal, {});
    defer q.deinit();
    var k = try zml.testing.autoCall(allocator, io, &rng_k, zml.Tensor.Rng.normal, {});
    defer k.deinit();
    var v = try zml.testing.autoCall(allocator, io, &rng_k, zml.Tensor.Rng.normal, {});
    defer v.deinit();

    const shardings = platform.shardings.values();
    const vanilla_exe = try platform.compileFn(
        allocator,
        io,
        dense,
        .{ tensors.q, tensors.k, tensors.v, .vanilla, DenseOpts{ .is_causal = is_causal } },
        .{
            .program_name = "dense_attention_vanilla",
            .shardings = shardings,
        },
    );
    defer vanilla_exe.deinit();

    var vanilla_d = try zml.testing.autoCall(allocator, io, &vanilla_exe, dense, .{ q, k, v });
    defer vanilla_d.deinit();
    try vanilla_d.await(io);
    const vanilla_h: zml.Slice = try vanilla_d.toSliceAlloc(allocator, io);
    defer vanilla_h.free(allocator);

    const backends = [_]Backend{ .cuda_fa2, .cuda_fa3 };
    for (backends) |backend| {
        const exe = try platform.compileFn(
            allocator,
            io,
            dense,
            .{ tensors.q, tensors.k, tensors.v, backend, DenseOpts{ .is_causal = is_causal } },
            .{
                .program_name = try std.fmt.allocPrint(arena, "dense_attention_{t}", .{backend}),
                .shardings = shardings,
            },
        );
        defer exe.deinit();

        var output_d = try zml.testing.autoCall(allocator, io, &exe, dense, .{ q, k, v });
        defer output_d.deinit();
        try output_d.await(io);
        const output_h = try output_d.toSliceAlloc(allocator, io);
        defer output_h.free(allocator);

        try zml.testing.expectClose(io, vanilla_h, output_h, .{
            .absolute_tolerance = 5e-3,
            .relative_tolerance = 1e-2,
            .epsilon_relative = 1e-3,
            .minimum_close_fraction = 0.99,
        });
    }
}

pub fn testAttention(q_shape: zml.Shape, k_shape: zml.Shape, token_index_h: []const u32) !void {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    const platform = zml.testing.env();

    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const token_index_shape: zml.Shape = if (q_shape.hasTag(.b)) |_| b: {
        std.debug.assert(token_index_h.len == q_shape.dim(.b));
        break :b .init(.{ .b = q_shape.dim(.b) }, .u32);
    } else .init(.{}, .u32);

    const max_k_idx: i64 = k_shape.dim(.k) - 1;
    const max_q_idx: i64 = q_shape.dim(.q) - 1;
    for (token_index_h) |index| {
        // Check for out of bound reads
        std.debug.assert(index + max_q_idx <= max_k_idx);
    }

    const tensors: struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor } = .{
        .q = .fromShape(q_shape),
        .k = .fromShape(k_shape),
        .v = .fromShape(k_shape),
        .token_index = .fromShape(token_index_shape),
    };

    const rng_q = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.q.shape(), .{ .mean = 0, .stddev = 1 } }, .{});
    defer rng_q.deinit();
    const rng_k = try platform.compileFn(allocator, io, zml.Tensor.Rng.normal, .{ tensors.k.shape(), .{ .mean = 0, .stddev = 1 } }, .{});
    defer rng_k.deinit();

    const q = try zml.testing.autoCall(allocator, io, &rng_q, zml.Tensor.Rng.normal, {});
    const k = try zml.testing.autoCall(allocator, io, &rng_k, zml.Tensor.Rng.normal, {});
    const v = try zml.testing.autoCall(allocator, io, &rng_k, zml.Tensor.Rng.normal, {});
    const token_index = try zml.Buffer.fromBytes(io, platform, token_index_shape, .replicated, @ptrCast(token_index_h));

    const shardings = platform.shardings.values();
    const vanilla_exe = try platform.compileFn(allocator, io, attention, .{ tensors.q, tensors.k, tensors.v, tensors.token_index, .vanilla, .vanilla }, .{
        .program_name = "attention_vanilla",
        .shardings = shardings,
    });
    defer vanilla_exe.deinit();

    const vanilla_d = try zml.testing.autoCall(allocator, io, &vanilla_exe, attention, .{ q, k, v, token_index, .vanilla });
    try vanilla_d.await(io);
    const vanilla_h: zml.Slice = try vanilla_d.toSliceAlloc(allocator, io);
    defer vanilla_h.free(allocator);

    for (std.enums.values(Backend)) |backend| {
        switch (backend) {
            .attnd, .vanilla => continue,
            else => if (!backend.isAvailable(platform)) continue,
        }

        const metadata: Metadata = .init(switch (backend) {
            .cuda_fa2 => .{ .cuda_fa2 = .{
                .seqlen = tensors.k.dim(.k),
                .num_heads = tensors.q.dim(.h),
                .head_dim = tensors.q.dim(.hd),
            } },
            .cuda_fa3 => .{ .cuda_fa3 = .{
                .seqlen = tensors.k.dim(.k),
                .num_heads = tensors.q.dim(.h),
                .head_dim = tensors.q.dim(.hd),
            } },
            else => .fromBackend(backend, tensors.k.dim(.k), tensors.q.dim(.h)),
        });
        const parameters: Parameters = .init(.fromBackend(backend));
        const exe = try platform.compileFn(
            allocator,
            io,
            attention,
            .{ tensors.q, tensors.k, tensors.v, tensors.token_index, metadata, parameters },
            .{
                .program_name = try std.fmt.allocPrint(arena, "attention_{t}", .{backend}),
                .shardings = shardings,
            },
        );
        defer exe.deinit();

        var metadata_d = try metadata.initBuffer(io, platform, platform.shardings.get("model").?);
        defer Metadata.deinitBuffer(&metadata_d);

        var output_d = try zml.testing.autoCall(allocator, io, &exe, attention, .{ q, k, v, token_index, metadata_d });
        defer output_d.deinit();
        try output_d.await(io);
        const output_h = try output_d.toSliceAlloc(allocator, io);
        defer output_h.free(allocator);

        errdefer std.log.err(
            \\ Attention test failed, {0t} output doesn't match reference.
            \\ - reference: {1d}
            \\ - {0t}: {2d}
        , .{ backend, vanilla_h, output_h });
        try zml.testing.expectClose(io, vanilla_h, output_h, .{
            .absolute_tolerance = 5e-3,
            .relative_tolerance = 1e-2,
            .epsilon_relative = 1e-3,
            .minimum_close_fraction = 0.99,
        });
    }
}
