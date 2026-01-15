const std = @import("std");

const zml = @import("../zml.zig");
pub const flashattn = @import("flashattn/flashattn.zig");

const Attention = @This();

pub const Backend = enum {
    vanilla,
    flashattn,
};

pub const Parameters = union(Backend) {
    vanilla: void,
    flashattn: flashattn.Parameters,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        flashattn: flashattn.Parameters.InitOptions,
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .vanilla => .{ .vanilla = {} },
            inline else => |v, tag| @unionInit(Parameters, @tagName(tag), @field(Attention, @tagName(tag)).Parameters.init(v)),
        };
    }
};

pub const Metadata = union(Backend) {
    vanilla: void,
    flashattn: flashattn.Metadata,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        flashattn: flashattn.Metadata.InitOptions,
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .vanilla => .{ .vanilla = {} },
            inline else => |v, tag| @unionInit(Metadata, @tagName(tag), @field(Attention, @tagName(tag)).Metadata.init(v)),
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: zml.Platform) !zml.Bufferized(Metadata) {
        return switch (self) {
            .vanilla => .{ .vanilla = {} },
            inline else => |v, tag| @unionInit(zml.Bufferized(Metadata), @tagName(tag), try v.initBuffer(io, platform)),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .vanilla => {},
            inline else => |*v, tag| @field(Attention, @tagName(tag)).Metadata.deinitBuffer(v),
        }
    }
};

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters) zml.Tensor {
    return switch (parameters) {
        .vanilla => b: {
            // Generate the attention mask.
            const seq_len = k.dim(.k);
            var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, q.dtype(), null);

            // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
            // then slice into it, but XLA is able to optimize this correctly.
            attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});
            const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
            break :b attn_output;
        },
        .flashattn => flashattn.attention(q, k, v, token_index, metadata.flashattn, parameters.flashattn),
    };
}
