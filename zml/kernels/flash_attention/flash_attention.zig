const std = @import("std");

pub const cute = @import("cute/cute.zig");
pub const metal = @import("metal/metal.zig");
pub const nki = @import("nki/nki.zig");
const zml = @import("../../zml.zig");

test {
    std.testing.refAllDecls(@This());
}

pub const Backend = enum {
    vanilla,
    cute,
    nki,
    metal_fa,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .neuron => .nki,
            .metal => .metal_fa,
            .cuda => if (cute.isAvailable(platform)) .cute else .vanilla,
            .cpu, .rocm, .tpu, .oneapi, .furiosa => .vanilla,
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .vanilla => true,
            .cute => cute.isAvailable(platform),
            .nki => platform.target == .neuron,
            .metal_fa => platform.target == .metal,
        };
    }
};

pub const Parameters = union(Backend) {
    vanilla: void,
    cute: void,
    nki: nki.Parameters,
    metal_fa: void,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        cute: void,
        nki: nki.Parameters,
        metal_fa: void,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .vanilla => .{ .vanilla = {} },
                .cute => .{ .cute = {} },
                .nki => .{ .nki = .init() },
                .metal_fa => .{ .metal_fa = {} },
            };
        }
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .vanilla => .{ .vanilla = {} },
            .cute => .{ .cute = {} },
            .nki => |v| .{ .nki = v },
            .metal_fa => .{ .metal_fa = {} },
        };
    }
};

pub const Metadata = union(Backend) {
    vanilla: void,
    cute: void,
    nki: void,
    metal_fa: metal.Metadata,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        cute: void,
        nki: void,
        metal_fa: void,

        pub fn fromBackend(backend: Backend, seqlen: i64, num_heads: i64) InitOptions {
            _ = seqlen; // autofix
            _ = num_heads; // autofix
            return switch (backend) {
                .vanilla => .{ .vanilla = {} },
                .cute => .{ .cute = {} },
                .nki => .{ .nki = {} },
                .metal_fa => .{ .metal_fa = {} },
            };
        }
    };

    pub fn init(opts: InitOptions, mesh: *const zml.Sharding.Mesh) Metadata {
        _ = mesh; // autofix
        return switch (opts) {
            .vanilla => .{ .vanilla = {} },
            .cute => .{ .cute = {} },
            .nki => .{ .nki = {} },
            .metal_fa => .{ .metal_fa = .init() },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        return switch (self) {
            .vanilla => .{ .vanilla = {} },
            .cute => .{ .cute = {} },
            .nki => .{ .nki = {} },
            inline else => |v, tag| @unionInit(zml.Bufferized(Metadata), @tagName(tag), try v.initBuffer(io, platform)),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .vanilla => {},
            .cute => {},
            .nki => {},
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
        .cute => cute.attention(q, k, v, token_index),
        .nki => nki.attention(q, k, v, token_index, parameters.nki),
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
    return testAttentionBackends(q_shape, k_shape, token_index_h, std.enums.values(Backend));
}

fn testAttentionBackends(q_shape: zml.Shape, k_shape: zml.Shape, token_index_h: []const u32, backends: []const Backend) !void {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    const platform = zml.testing.env();

    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const token_index_shape: zml.Shape = if (token_index_h.len > 1) b: {
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

    var q = try attentionTestInput(q_shape, 17);
    defer q.deinit();
    var k = try attentionTestInput(k_shape, 29);
    defer k.deinit();
    var v = try attentionTestInput(k_shape, 43);
    defer v.deinit();
    var token_index = try zml.Buffer.fromBytes(io, platform, token_index_shape, @ptrCast(token_index_h));
    defer token_index.deinit();

    const meshes = platform.meshes.values();
    const vanilla_exe = try platform.compileFn(allocator, io, attention, .{ tensors.q, tensors.k, tensors.v, tensors.token_index, .vanilla, .vanilla }, .{
        .program_name = "attention_vanilla",
        .meshes = meshes,
    });
    defer vanilla_exe.deinit();

    var vanilla_d = try vanilla_exe.eval(allocator, io, .{ q, k, v, token_index, .vanilla });
    defer vanilla_d.deinit();
    try vanilla_d.await(io);
    const vanilla_h: zml.Slice = try vanilla_d.toSliceAlloc(allocator, io);
    defer vanilla_h.free(allocator);

    for (backends) |backend| {
        switch (backend) {
            .vanilla => continue,
            else => if (!backend.isAvailable(platform)) continue,
        }

        const metadata: Metadata = .init(.fromBackend(backend, tensors.k.dim(.k), tensors.q.dim(.h)), platform.meshes.get("model").?);
        const parameters: Parameters = .init(.fromBackend(backend));
        const exe = try platform.compileFn(
            allocator,
            io,
            attention,
            .{ tensors.q, tensors.k, tensors.v, tensors.token_index, metadata, parameters },
            .{
                .program_name = try std.fmt.allocPrint(arena, "attention_{t}", .{backend}),
                .meshes = meshes,
            },
        );
        defer exe.deinit();

        var metadata_d = try metadata.initBuffer(io, platform);
        defer Metadata.deinitBuffer(&metadata_d);

        var output_d = try exe.eval(allocator, io, .{ q, k, v, token_index, metadata_d });
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

fn attentionTestInput(shape: zml.Shape, seed: u64) !zml.Buffer {
    const host = try zml.Slice.alloc(std.testing.allocator, shape);
    defer host.free(std.testing.allocator);
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    switch (shape.dtype()) {
        inline .bf16, .f16, .f32 => |dtype| {
            for (host.items(dtype.toZigType())) |*value| {
                const x = random.float(f32) * 4 - 2;
                value.* = if (dtype == .bf16) .fromF32(x) else @floatCast(x);
            }
        },
        else => unreachable,
    }
    return zml.Buffer.fromSlice(std.testing.io, zml.testing.env(), host);
}

/// Independent f64 oracle: the SDPA reference uses fast dot precision, so its
/// f32 result alone cannot establish the accuracy of the kernel's f32 arithmetic.
fn checkAttentionF32(q: zml.Slice, k: zml.Slice, v: zml.Slice, positions: []const u32, output: zml.Slice) !void {
    const Layout = struct {
        b: usize,
        s: usize,
        h: usize,
        d: usize,

        fn stride(shape: zml.Shape, tag: anytype) usize {
            const axis = shape.hasTag(tag) orelse return 0;
            var result: usize = 1;
            for (shape.dims()[axis + 1 ..]) |dim| result *= @intCast(dim);
            return result;
        }

        fn init(shape: zml.Shape) @This() {
            return .{ .b = stride(shape, .b), .s = if (shape.hasTag(.q) != null) stride(shape, .q) else stride(shape, .k), .h = stride(shape, .h), .d = stride(shape, .hd) };
        }

        fn offset(self: @This(), b: usize, s: usize, h: usize, d: usize) usize {
            return b * self.b + s * self.s + h * self.h + d * self.d;
        }
    };
    const ql = Layout.init(q.shape);
    const kl = Layout.init(k.shape);
    const vl = Layout.init(v.shape);
    const batch: usize = if (q.shape.hasTag(.b) != null) @intCast(q.shape.dim(.b)) else 1;
    const queries: usize = @intCast(q.shape.dim(.q));
    const heads: usize = @intCast(q.shape.dim(.h));
    const dim: usize = @intCast(q.shape.dim(.hd));
    const group: usize = @intCast(@divExact(q.shape.dim(.h), k.shape.dim(.h)));
    const scores = try std.testing.allocator.alloc(f64, @intCast(k.shape.dim(.k)));
    defer std.testing.allocator.free(scores);
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(dim)));
    for (0..batch) |b| {
        const start: usize = positions[if (positions.len == 1) 0 else b];
        for (0..queries) |i| {
            for (0..heads) |h| {
                var maximum: f64 = -std.math.inf(f64);
                const active = scores[0 .. start + i + 1];
                for (active, 0..) |*score, j| {
                    var dot: f64 = 0;
                    for (0..dim) |d| dot += @as(f64, q.items(f32)[ql.offset(b, i, h, d)]) * @as(f64, k.items(f32)[kl.offset(b, j, h / group, d)]);
                    score.* = dot * scale;
                    maximum = @max(maximum, score.*);
                }
                var sum: f64 = 0;
                for (active) |*score| {
                    score.* = @exp(score.* - maximum);
                    sum += score.*;
                }
                for (0..dim) |d| {
                    var expected: f64 = 0;
                    for (active, 0..) |weight, j| expected += weight * @as(f64, v.items(f32)[vl.offset(b, j, h / group, d)]);
                    expected /= sum;
                    const actual = output.items(f32)[ql.offset(b, i, h, d)];
                    try std.testing.expectApproxEqAbs(expected, @as(f64, actual), 2e-5 + 2e-5 * @abs(expected));
                }
            }
        }
    }
}
