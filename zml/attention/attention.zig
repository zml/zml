const std = @import("std");

const flash_attention = @import("zml/kernels/flash_attention");
pub const Options = @import("zml/attention/options");

const zml = @import("../zml.zig");

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

pub const Partitioning = struct {
    const Logical = zml.Sharding.Partitioning.Logical;

    in_specs: struct {
        q: Logical,
        k: Logical,
        v: Logical,
        token_index: Logical = .replicated,
        /// Omit to replicate backend metadata tensors.
        metadata: ?zml.ops.PartitionSpecs(Metadata) = null,
    },
    out_specs: Logical,

    /// Shards the `.h` dimension of Q, K, V and the output over `mesh_axis`.
    /// Other dimensions, token indices and backend metadata are replicated.
    pub fn head(comptime mesh_axis: @EnumLiteral()) Partitioning {
        const heads: Logical = .parse(.{ .h = mesh_axis });
        return .{ .in_specs = .{ .q = heads, .k = heads, .v = heads }, .out_specs = heads };
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
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters, mesh: *const zml.Mesh, partitioning: Partitioning) zml.Tensor {
    return attentionOpts(q, k, v, token_index, metadata, parameters, .{}, mesh, partitioning);
}

/// `attention`, causal or not depending on `opts`.
pub fn attentionOpts(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters, opts: Options, mesh: *const zml.Mesh, partitioning: Partitioning) zml.Tensor {
    const args: Attention = .{ .q = q, .k = k, .v = v, .token_index = token_index, .metadata = metadata, .parameters = parameters, .opts = opts };
    return zml.ops.manualComputation(Attention.forward, args, q.shape(), mesh, .{
        .in_specs = .{
            .q = partitioning.in_specs.q,
            .k = partitioning.in_specs.k,
            .v = partitioning.in_specs.v,
            .token_index = partitioning.in_specs.token_index,
            .metadata = partitioning.in_specs.metadata orelse zml.ops.partitionSpecs(metadata, .replicated),
        },
        .out_specs = partitioning.out_specs,
    });
}

const Attention = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    token_index: zml.Tensor,
    metadata: Metadata,
    parameters: Parameters,
    opts: Options,

    fn forward(args: @This(), _: zml.Shape) zml.Tensor {
        return switch (args.parameters) {
            .stablehlo => b: {
                const seq_len = args.k.dim(.k);
                const mask: ?zml.Tensor = if (args.opts.causal)
                    zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, args.q.dtype(), null)
                        .gatherSlices(zml.Shape.init(.{ .q = args.q.dim(.q) }, args.q.dtype()), args.token_index.appendAxes(.{.coord}), .{})
                else
                    null;
                break :b zml.nn.sdpa(args.q, args.k, args.v, .{ .attn_mask = mask });
            },
            .flash_attention => flash_attention.attention(args.q, args.k, args.v, args.token_index, args.metadata.flash_attention, args.parameters.flash_attention, args.opts),
        };
    }
};
