const std = @import("std");
const zml = @import("../zml.zig");
const triton = @import("triton.zig");

pub const Backend = enum {
    triton,

    pub fn auto(platform: *const zml.Platform, weights_dtype: zml.DataType) !Backend {
        return switch (platform.target) {
            .cuda => b: {
                const first_device = platform.pjrt_client.devices(platform.pjrt_api)[0];
                if (zml.platform.cuda.tryGetComputeCapabilities(platform, first_device)) |cc| {
                    if (std.mem.eql(u8, cc, "9.0")) {
                        break :b switch (weights_dtype) {
                            .bf16, .f16 => .triton,
                            else => error.UnsupportedDataType,
                        };
                    }
                    break :b error.UnsupportedComputeCapability;
                }
                break :b error.UnsupportedComputeCapability;
            },
            else => error.UnimplementedMoEBackend,
        };
    }

    pub fn load(backend: Backend, allocator: std.mem.Allocator) !void {
        _ = allocator; // autofix
        return switch (backend) {
            .triton => {},
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        _ = platform; // autofix
        return switch (backend) {
            .triton => {},
        };
    }
};

pub const Parameters = union(Backend) {
    triton: triton.Parameters,

    pub const InitOptions = union(Backend) {
        triton: triton.Parameters.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .triton => .{ .triton = .{} },
            };
        }
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .triton => |v| .{ .triton = triton.Parameters.init(v) },
        };
    }
};

pub const Metadata = union(Backend) {
    triton: triton.Metadata,

    pub const InitOptions = union(Backend) {
        triton: triton.Metadata.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .triton => .{ .triton = .{} },
            };
        }
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .triton => |v| .{ .triton = triton.Metadata.init(v) },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *zml.Platform) !zml.Bufferized(Metadata) {
        return switch (self) {
            .triton => |metadata| .{ .triton = try metadata.initBuffer(io, platform) },
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .triton => |*metadata| triton.deinitBuffer(metadata),
        }
    }
};

pub fn moe(
    input: zml.Tensor,
    tokens_mask: ?zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    weights_gate_up: zml.Tensor,
    scales_gate_up: ?zml.Tensor,
    bias_gate_up: ?zml.Tensor,
    weights_down: zml.Tensor,
    scales_down: ?zml.Tensor,
    bias_down: ?zml.Tensor,
    num_experts_per_tok: u32,
    metadata: Metadata,
    parameters: Parameters,
) !zml.Tensor {
    _ = tokens_mask; // autofix
    _ = num_experts_per_tok; // autofix
    return switch (parameters) {
        .triton => b: {
            const triton_metadata = switch (metadata) {
                .triton => |v| v,
            };

            break :b try triton.fusedExpertsImpl(
                input,
                weights_gate_up,
                weights_down,
                topk_weights,
                topk_ids,
                triton_metadata,
                .{
                    .w1_scale = scales_gate_up,
                    .w2_scale = scales_down,
                    .w1_bias = bias_gate_up,
                    .w2_bias = bias_down,
                },
            );
        },
    };
}
