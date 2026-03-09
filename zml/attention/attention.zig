const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../zml.zig");
const flashattn = @import("flashattn.zig");

const Attention = @This();

pub const Backend = enum {
    vanilla,
    tpu,
    cuda_fa2,
    cuda_fa3,

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
            .tpu => .tpu,
            else => .vanilla,
        };
    }
};

pub const Parameters = union(Backend) {
    vanilla: void,
    tpu: void,
    cuda_fa2: flashattn.fa2.Parameters,
    cuda_fa3: flashattn.fa3.Parameters,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        tpu: void,
        cuda_fa2: flashattn.fa2.Parameters.InitOptions,
        cuda_fa3: flashattn.fa3.Parameters.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .vanilla => .{ .vanilla = {} },
                .tpu => .{ .tpu = {} },
                .cuda_fa2 => .{ .cuda_fa2 = .{} },
                .cuda_fa3 => .{ .cuda_fa3 = .{} },
            };
        }
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .vanilla => .{ .vanilla = {} },
            .tpu => .{ .tpu = {} },
            .cuda_fa2 => |v| .{ .cuda_fa2 = .init(v) },
            .cuda_fa3 => |v| .{ .cuda_fa3 = .init(v) },
        };
    }
};

pub const Metadata = union(Backend) {
    vanilla: void,
    tpu: void,
    cuda_fa2: flashattn.fa2.Metadata,
    cuda_fa3: flashattn.fa3.Metadata,

    pub const InitOptions = union(Backend) {
        vanilla: void,
        tpu: void,
        cuda_fa2: flashattn.fa2.Metadata.InitOptions,
        cuda_fa3: flashattn.fa3.Metadata.InitOptions,

        pub fn fromBackend(backend: Backend, seqlen: i64) InitOptions {
            return switch (backend) {
                .vanilla => .{ .vanilla = {} },
                .tpu => .{ .tpu = {} },
                .cuda_fa2 => .{ .cuda_fa2 = .{ .seqlen = seqlen } },
                .cuda_fa3 => .{ .cuda_fa3 = .{ .seqlen = seqlen } },
            };
        }
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .vanilla => .{ .vanilla = {} },
            .tpu => .{ .tpu = {} },
            .cuda_fa2 => |o| .{ .cuda_fa2 = flashattn.fa2.Metadata.init(o) },
            .cuda_fa3 => |o| .{ .cuda_fa3 = flashattn.fa3.Metadata.init(o) },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        return switch (self) {
            .vanilla => .{ .vanilla = {} },
            .tpu => .{ .tpu = {} },
            .cuda_fa2 => |v| .{ .cuda_fa2 = try v.initBuffer(io, platform) },
            .cuda_fa3 => |v| .{ .cuda_fa3 = try v.initBuffer(io, platform) },
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .vanilla => {},
            .tpu => {},
            .cuda_fa2 => |*v| flashattn.fa2.Metadata.deinitBuffer(v),
            .cuda_fa3 => |*v| flashattn.fa3.Metadata.deinitBuffer(v),
        }
    }
};

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters) zml.Tensor {
    return switch (parameters) {
        .vanilla => vanillaAttention(q, k, v, token_index),
        .tpu => tpuAttention(q, k, v, token_index),
        .cuda_fa2 => flashattn.fa2.attention(q, k, v, token_index, metadata.cuda_fa2, parameters.cuda_fa2),
        .cuda_fa3 => flashattn.fa3.attention(q, k, v, token_index, metadata.cuda_fa3, parameters.cuda_fa3),
    };
}

fn tpuAttention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    var threaded: std.Io.Threaded = .init_single_threaded;
    threaded.allocator = zml.module.CompilationContext.current().allocator;

    if (q.dim(.q) == 1) {
        const has_batch = q.shape().hasTag(.b) != null;
        const head_dim: i64 = q.dim(.hd);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const q_f32 = q.convert(.f32).scale(scale);
        const q_nd = if (has_batch)
            q_f32.squeeze(.q)
        else
            q_f32.reshape(.{ .h = q.dim(.h), .hd = q.dim(.hd) });

        const lengths = token_index.addConstant(1).convert(.i32).reshape(.{1});
        const out = flashattn.mosaic_tpu.decodeAttention(
            threaded.allocator,
            threaded.io(),
            q_nd,
            k.convert(.f32),
            v.convert(.f32),
            lengths,
        );
        return out.reshape(q.shape()).convert(q.dtype());
    }

    return flashattn.mosaic_tpu.prefillAttention(threaded.allocator, threaded.io(), q, k, v);
}

fn vanillaAttention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const seq_len = k.dim(.k);
    const allow_cudnn = zml.module.CompilationContext.current().platform.target == .cuda;
    var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, q.dtype(), null);

    // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
    // then slice into it, but XLA is able to optimize this correctly.
    attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});
    return zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = allow_cudnn });
}
