const std = @import("std");

pub const attnd = @import("attention/attnd.zig");
pub const flashattn = @import("attention/flashattn.zig");
pub const metal = @import("attention/metal_attention.zig");
pub const nki = @import("attention/nki/attention.zig");
pub const mla = @import("attention/multi_latent_attention.zig");
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

        const metadata: Metadata = .init(.fromBackend(backend, tensors.k.dim(.k), tensors.q.dim(.h)));
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
