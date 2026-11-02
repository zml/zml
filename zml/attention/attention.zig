const std = @import("std");

const zml = @import("../zml.zig");
const flash_attention = @import("../kernels/flash_attention/flash_attention.zig");

test {
    std.testing.refAllDecls(@This());
}

pub const BackendType = enum {
    stablehlo,
    flash_attention,
};

pub const Backend = union(BackendType) {
    stablehlo: void,
    flash_attention: flash_attention.Backend,

    pub fn auto(platform: *const zml.Platform) Backend {
        return if (flash_attention.isAvailable(platform))
            .{ .flash_attention = flash_attention.Backend.auto(platform) }
        else
            .{ .stablehlo = {} };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .stablehlo => true,
            .flash_attention => |b| b.isAvailable(platform),
        };
    }
};

pub const Parameters = union(BackendType) {
    stablehlo: void,
    flash_attention: flash_attention.Parameters,

    pub const InitOptions = union(BackendType) {
        stablehlo: void,
        flash_attention: flash_attention.Parameters.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .stablehlo => .{ .stablehlo = {} },
                .flash_attention => |b| .{ .flash_attention = .fromBackend(b) },
            };
        }
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .stablehlo => .{ .stablehlo = {} },
            .flash_attention => |v| .{ .flash_attention = .init(v) },
        };
    }
};

pub const Metadata = union(BackendType) {
    stablehlo: void,
    flash_attention: flash_attention.Metadata,

    pub const InitOptions = union(BackendType) {
        stablehlo: void,
        flash_attention: flash_attention.Metadata.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .stablehlo => .{ .stablehlo = {} },
                .flash_attention => |b| .{ .flash_attention = .fromBackend(b) },
            };
        }
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .stablehlo => .{ .stablehlo = {} },
            .flash_attention => |v| .{ .flash_attention = .init(v) },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        return switch (self) {
            .stablehlo => .{ .stablehlo = {} },
            .flash_attention => |v| .{ .flash_attention = try v.initBuffer(io, platform) },
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .stablehlo => {},
            .flash_attention => |*v| flash_attention.Metadata.deinitBuffer(v),
        }
    }
};

/// Caller-selected logical mesh and query/KV head axes. `null` explicitly replicates.
pub const Sharding = struct {
    mesh: *const zml.Mesh,
    heads: ?zml.Shape.Tag,
    kv_heads: ?zml.Shape.Tag,

    pub fn kvPartition(self: Sharding, shape: zml.Shape, head_dim: anytype) zml.Sharding.Partitioning {
        var kv = self;
        kv.heads = self.kv_heads;
        return kv.partition(shape, head_dim);
    }

    pub fn partition(self: Sharding, shape: zml.Shape, head_dim: anytype) zml.Sharding.Partitioning {
        const replicated: zml.Sharding.Partitioning = .replicated(shape.rank());
        const axis = self.heads orelse return replicated;
        const mesh_axis = self.mesh.resolveLogicalAxis(axis) orelse
            std.debug.panic("Attention mesh '{s}' has no logical axis '{s}'", .{ self.mesh.name, axis });
        return replicated.set(shape.axis(head_dim), .sharded(@intCast(mesh_axis)));
    }
};

/// Causal attention as used in transformers.
/// The result uses `sharding.heads` on the query head dimension on every backend.
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
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters, sharding: Sharding) zml.Tensor {
    return switch (parameters) {
        .stablehlo => b: {
            // Generate the attention mask.
            const seq_len = k.dim(.k);
            var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, q.dtype(), null);

            // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
            // then slice into it, but XLA is able to optimize this correctly.
            attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()), token_index.appendAxes(.{.coord}), .{});
            const attn_output = zml.nn.sdpa(q.withPartitioning(sharding.mesh, sharding.partition(q.shape(), .h)), k.withPartitioning(sharding.mesh, sharding.kvPartition(k.shape(), .h)), v.withPartitioning(sharding.mesh, sharding.kvPartition(v.shape(), .h)), .{ .attn_mask = attn_mask });
            break :b attn_output.withPartitioning(sharding.mesh, sharding.partition(q.shape(), .h));
        },
        .flash_attention => b: {
            zml.stdx.debug.assert(sharding.heads == sharding.kv_heads, "Attention kernels require matching query and KV head sharding", .{});
            const Call = struct {
                q: zml.Tensor,
                k: zml.Tensor,
                v: zml.Tensor,
                token_index: zml.Tensor,
                metadata: flash_attention.Metadata,
                parameters: flash_attention.Parameters,

                fn local(self: @This(), _: zml.Shape) zml.Tensor {
                    return flash_attention.attention(self.q, self.k, self.v, self.token_index, self.metadata, self.parameters);
                }
            };
            break :b zml.ops.manualComputation(Call.local, .{
                .q = q,
                .k = k,
                .v = v,
                .token_index = token_index,
                .metadata = metadata.flash_attention,
                .parameters = parameters.flash_attention,
            }, q.shape(), .{
                .mesh = sharding.mesh,
                .in_specs = .{
                    .q = sharding.partition(q.shape(), .h),
                    .k = sharding.kvPartition(k.shape(), .h),
                    .v = sharding.kvPartition(v.shape(), .h),
                    .token_index = .replicated,
                    .metadata = .replicated,
                    .parameters = .replicated,
                },
                .out_specs = sharding.partition(q.shape(), .h),
            });
        },
    };
}

test "attention backends preserve the same output sharding" {
    const mlir = @import("mlir");
    const platform = zml.testing.env();
    const mesh = zml.testing.tpMesh();
    const partitions = mesh.numPartitionsForLogicalAxis(.model);
    var compiler: zml.Compiler = .init(std.testing.allocator, std.testing.io, platform, .{ .meshes = &.{mesh} });
    defer compiler.deinit();
    compiler.activate();
    defer compiler.deactivate();
    const block = mlir.Block.init(&.{}, &.{});
    defer block.deinit();
    const scope = compiler.pushBlock(block);
    defer scope.pop();

    // Trace CuTe on any platform: no device kernel is compiled or executed here.
    for ([_]bool{ false, true }) |batched| {
        const q_shape: zml.Shape = if (batched)
            .init(.{ .b = 2, .q = 1, .h = 4 * partitions, .hd = 64 }, .bf16)
        else
            .init(.{ .q = 1, .h = 4 * partitions, .hd = 64 }, .bf16);
        const k_shape: zml.Shape = if (batched)
            .init(.{ .b = 2, .k = 8, .h = 2 * partitions, .hd = 64 }, .bf16)
        else
            .init(.{ .k = 8, .h = 2 * partitions, .hd = 64 }, .bf16);
        for ([_]?zml.Shape.Tag{ null, "model" }) |axis| {
            const sharding: Sharding = .{ .mesh = mesh, .heads = axis, .kv_heads = axis };
            const expected = q_shape.withPartitioning(mesh, sharding.partition(q_shape, .h));
            for ([_]Backend{ .{ .stablehlo = {} }, .{ .flash_attention = .cute } }) |backend| {
                const output = attention(.zeroes(q_shape), .zeroes(k_shape), .zeroes(k_shape), .scalar(0, .u32), .init(.fromBackend(backend)), .init(.fromBackend(backend)), sharding);
                try std.testing.expect(output.shape().eql(q_shape));
                try std.testing.expectEqualDeep(expected._sharding, output.shape()._sharding);
            }
        }
    }
}
