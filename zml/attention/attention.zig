const std = @import("std");

const zml = @import("../zml.zig");
const attnd = @import("attnd.zig");
const flashattn = @import("flashattn.zig");
const nki = @import("nki/attention.zig");
const triton = @import("triton_attention.zig");

const Attention = @This();

pub const Backend = enum {
    vanilla,
    attnd,
    nki,
    cuda_fa2,
    cuda_fa3,
    triton,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            //.cuda => b: {
            //    const first_device = platform.pjrt_client.devices(platform.pjrt_api)[0];

            //    if (zml.platform.cuda.tryGetComputeCapabilities(platform, first_device)) |cc| {
            //        break :b if (std.mem.eql(u8, cc, "9.0"))
            //            .cuda_fa3
            //        else
            //            .cuda_fa2;
            //    }

            //    break :b .vanilla;
            //},
            .neuron => .nki,
            .cpu, .tpu, .oneapi => .vanilla,
            .cuda, .rocm => .triton,
        };
    }
};

pub const Parameters = union(Backend) {
    vanilla: void,
    attnd: attnd.Parameters,
    nki: nki.Parameters,
    cuda_fa2: flashattn.fa2.Parameters,
    cuda_fa3: flashattn.fa3.Parameters,
    triton: triton.flashattn.Parameters,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        attnd: void,
        nki: nki.Parameters,
        cuda_fa2: flashattn.fa2.Parameters.InitOptions,
        cuda_fa3: flashattn.fa3.Parameters.InitOptions,
        triton: triton.flashattn.Parameters.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .vanilla => .{ .vanilla = {} },
                .attnd => @panic("Must be initialized manually"),
                .nki => .{ .nki = .init() },
                .cuda_fa2 => .{ .cuda_fa2 = .{} },
                .cuda_fa3 => .{ .cuda_fa3 = .{} },
                .triton => .{ .triton = .{} },
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
            .triton => |v| .{ .triton = .init(v) },
        };
    }
};

pub const Metadata = union(Backend) {
    vanilla: void,
    attnd: attnd.Metadata,
    nki: void,
    cuda_fa2: flashattn.fa2.Metadata,
    cuda_fa3: flashattn.fa3.Metadata,
    triton: triton.flashattn.Metadata,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        attnd: void,
        nki: void,
        cuda_fa2: flashattn.fa2.Metadata.InitOptions,
        cuda_fa3: flashattn.fa3.Metadata.InitOptions,
        triton: triton.flashattn.Metadata.InitOptions,

        pub fn fromBackend(backend: Backend, seqlen: i64, num_heads: i64) InitOptions {
            return switch (backend) {
                .vanilla => .{ .vanilla = {} },
                .attnd => .{ .attnd = {} },
                .nki => .{ .nki = {} },
                .cuda_fa2 => .{ .cuda_fa2 = .{ .seqlen = seqlen, .num_heads = num_heads } },
                .cuda_fa3 => .{ .cuda_fa3 = .{ .seqlen = seqlen, .num_heads = num_heads } },
                .triton => .{ .triton = .{} },
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
            .triton => |o| .{ .triton = triton.flashattn.Metadata.init(o) },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(Metadata) {
        return switch (self) {
            .vanilla => .{ .vanilla = {} },
            .attnd => |v| .{ .attnd = try v.initBuffer(io, platform, sharding) },
            .nki => .{ .nki = {} },
            .cuda_fa2 => |v| .{ .cuda_fa2 = try v.initBuffer(io, platform, sharding) },
            .cuda_fa3 => |v| .{ .cuda_fa3 = try v.initBuffer(io, platform, sharding) },
            .triton => .{ .triton = {} },
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .vanilla => {},
            .attnd => |*v| attnd.Metadata.deinitBuffer(v),
            .nki => {},
            .cuda_fa2 => |*v| flashattn.fa2.Metadata.deinitBuffer(v),
            .cuda_fa3 => |*v| flashattn.fa3.Metadata.deinitBuffer(v),
            .triton => {},
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
        .attnd => attnd.causalAttention(q, k, v, token_index, metadata.attnd, parameters.attnd),
        .nki => |params| nki.attention(q, k, v, token_index, params),
        .cuda_fa2 => flashattn.fa2.attention(q, k, v, token_index, metadata.cuda_fa2, parameters.cuda_fa2),
        .cuda_fa3 => flashattn.fa3.attention(q, k, v, token_index, metadata.cuda_fa3, parameters.cuda_fa3),
        .triton => triton.flashattn.attention(q, k, v, token_index, metadata.triton, parameters.triton),
    };
}
