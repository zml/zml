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
        .stablehlo => b: {
            // Generate the attention mask.
            const seq_len = k.dim(.k);
            var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, q.dtype(), null);

            // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
            // then slice into it, but XLA is able to optimize this correctly.
            attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()), token_index.appendAxes(.{.coord}), .{});
            const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask });
            break :b attn_output;
        },
        .flash_attention => flash_attention.attention(q, k, v, token_index, metadata.flash_attention, parameters.flash_attention),
    };
}
