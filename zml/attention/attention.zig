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

/// Causal attention as used in transformers.
/// Every backend runs on local head shards inside a manual computation.
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
    const Call = struct {
        q: zml.Tensor,
        k: zml.Tensor,
        v: zml.Tensor,
        token_index: zml.Tensor,
        metadata: Metadata,
        parameters: Parameters,

        fn forward(args: @This(), _: zml.Shape) zml.Tensor {
            return switch (args.parameters) {
                .stablehlo => b: {
                    const seq_len = args.k.dim(.k);
                    var mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, args.q.dtype(), null);
                    mask = mask.gatherSlices(zml.Shape.init(.{ .q = args.q.dim(.q) }, mask.dtype()), args.token_index.appendAxes(.{.coord}), .{});
                    break :b zml.nn.sdpa(args.q, args.k, args.v, .{ .attn_mask = mask });
                },
                .flash_attention => flash_attention.attention(args.q, args.k, args.v, args.token_index, args.metadata.flash_attention, args.parameters.flash_attention),
            };
        }
    };
    const mesh = zml.Compiler.current().resolveMesh(.{.model});
    const args: Call = .{
        .q = q.withPartitioning(mesh, q.shape().withPartitioning(mesh, .{ .h = .model })._sharding.partition),
        .k = k.withPartitioning(mesh, k.shape().withPartitioning(mesh, .{ .h = .model })._sharding.partition),
        .v = v.withPartitioning(mesh, v.shape().withPartitioning(mesh, .{ .h = .model })._sharding.partition),
        .token_index = token_index,
        .metadata = metadata,
        .parameters = parameters,
    };
    return zml.ops.manualComputation(Call.forward, args, q.shape().withPartitioning(mesh, .{ .h = .model }));
}

test "attention runs every backend inside the shared manual region" {
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

    for ([_]bool{ false, true }) |batched| {
        const q_shape: zml.Shape = if (batched)
            .init(.{ .b = 2, .q = 1, .h = 4 * partitions, .hd = 64 }, .bf16)
        else
            .init(.{ .q = 1, .h = 4 * partitions, .hd = 64 }, .bf16);
        const k_shape: zml.Shape = if (batched)
            .init(.{ .b = 2, .k = 8, .h = 2 * partitions, .hd = 64 }, .bf16)
        else
            .init(.{ .k = 8, .h = 2 * partitions, .hd = 64 }, .bf16);
        const q: zml.Tensor = .zeroes(q_shape);
        const k: zml.Tensor = .zeroes(k_shape);
        const v: zml.Tensor = .zeroes(k_shape);

        // Trace CuTe on any platform; no device kernel is compiled or executed.
        for ([_]Backend{ .stablehlo, .{ .flash_attention = .cute } }) |backend| {
            const output = attention(q, k, v, .scalar(0, .u32), .init(.fromBackend(backend)), .init(.fromBackend(backend)));
            try std.testing.expect(output.shape().eql(q_shape));
            try std.testing.expectEqualDeep(q_shape.withPartitioning(mesh, .{ .h = .model })._sharding, output.shape()._sharding);
            try std.testing.expectEqualStrings("sdy.manual_computation", compiler.currentScope().value(output).owner().name());
        }
    }
}
