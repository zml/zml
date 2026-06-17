const std = @import("std");
const zml = @import("../zml.zig");
const stdx = zml.stdx;
pub const triton = @import("triton.zig");
pub const mosaic_tpu = @import("mosaic_tpu.zig");

pub const triton_kernels = @import("triton_kernels/triton_kernels.zig");

pub const ActivationMode = enum {
    silu,
    relu,
    gelu,
};

pub const Backend = enum {
    // Could select a more specific name like "triton_sm90_bf16"
    triton,
    mosaic_tpu,

    pub fn auto(platform: *const zml.Platform, weights_dtype: zml.DataType) !Backend {
        return switch (platform.target) {
            .cuda => b: {
                const first_device = platform.pjrt_client.devices(platform.pjrt_api)[0];
                if (zml.platform.cuda.tryGetComputeCapabilities(platform, first_device)) |cc| {
                    if (std.mem.eql(u8, cc, "9.0") or
                        std.mem.eql(u8, cc, "10.0"))
                    {
                        break :b switch (weights_dtype) {
                            .bf16, .f16, .f8e4m3fn => .triton,
                            else => error.UnsupportedDataType,
                        };
                    }
                    break :b error.UnsupportedComputeCapability;
                }
                break :b error.UnsupportedComputeCapability;
            },
            .tpu => switch (weights_dtype) {
                .bf16, .f16, .f32 => .mosaic_tpu,
                else => error.UnsupportedDataType,
            },
            else => error.UnimplementedMoEBackend,
        };
    }

    pub fn load(backend: Backend, allocator: std.mem.Allocator) !void {
        _ = allocator;
        return switch (backend) {
            .triton => {},
            .mosaic_tpu => {},
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        _ = platform;
        return switch (backend) {
            .triton => {},
            .mosaic_tpu => {},
        };
    }
};

pub const Parameters = union(Backend) {
    triton: triton.Parameters,
    mosaic_tpu: mosaic_tpu.Parameters,

    pub const InitOptions = union(Backend) {
        triton: triton.Parameters.InitOptions,
        mosaic_tpu: mosaic_tpu.Parameters.InitOptions,

        pub fn fromBackend(backend: Backend, num_experts_per_tok: ?u32, activation: ActivationMode) InitOptions {
            return switch (backend) {
                .triton => .{ .triton = .{
                    .num_experts_per_tok = num_experts_per_tok.?,
                    .activation = switch (activation) {
                        .silu => .silu,
                        .relu => .relu,
                        .gelu => .gelu,
                    },
                } },
                .mosaic_tpu => .{ .mosaic_tpu = .{
                    .num_experts_per_tok = num_experts_per_tok.?,
                    .activation = switch (activation) {
                        .silu => .silu,
                        .relu => .relu,
                        .gelu => .gelu,
                    },
                } },
            };
        }
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .triton => |v| .{ .triton = triton.Parameters.init(v) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = mosaic_tpu.Parameters.init(v) },
        };
    }
};

pub const Metadata = union(Backend) {
    triton: triton.Metadata,
    mosaic_tpu: mosaic_tpu.Metadata,

    pub const InitOptions = union(Backend) {
        triton: triton.Metadata.InitOptions,
        mosaic_tpu: mosaic_tpu.Metadata.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .triton => .{ .triton = .{} },
                .mosaic_tpu => .{ .mosaic_tpu = .{} },
            };
        }
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .triton => |v| .{ .triton = triton.Metadata.init(v) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = mosaic_tpu.Metadata.init(v) },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        return switch (self) {
            .triton => |metadata| .{ .triton = try metadata.initBuffer(io, platform) },
            .mosaic_tpu => |metadata| .{ .mosaic_tpu = try metadata.initBuffer(io, platform) },
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .triton => |*metadata| triton.deinitBuffer(metadata),
            .mosaic_tpu => |*metadata| mosaic_tpu.deinitBuffer(metadata),
        }
    }
};

pub fn forwardMoe(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    weights_gate_up: zml.Tensor,
    scales_gate_up: ?zml.Tensor,
    bias_gate_up: ?zml.Tensor,
    weights_down: zml.Tensor,
    scales_down: ?zml.Tensor,
    bias_down: ?zml.Tensor,
    metadata: Metadata,
    parameters: Parameters,
) !zml.Tensor {
    return switch (parameters) {
        .triton => b: {
            const triton_metadata = switch (metadata) {
                .triton => |v| v,
                else => return error.InvalidMetadata,
            };

            const expert_partition = weights_gate_up.shape().partition(.expert);

            if (expert_partition.eql(.init(.experts))) {
                const global_num_experts = weights_down.dim(.expert);

                break :b zml.ops.manualComputation(
                    .{ input, topk_ids, topk_weights, weights_gate_up, weights_down },
                    input.shape(),
                    .{
                        .activation = parameters.triton.activation,
                        .global_num_experts = global_num_experts,
                        .scales_gate_up = scales_gate_up,
                        .bias_gate_up = bias_gate_up,
                        .scales_down = scales_down,
                        .bias_down = bias_down,
                    },
                    (struct {
                        fn body(ctx: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                            const local_num_experts = sharded_inputs[3].dim(.expert);
                            const partition_id = zml.ops.partitionId().convert(.i32);
                            const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                            // List of global expert ids
                            const global_expert_ids = zml.Tensor.arange(.{ .end = ctx.global_num_experts }, .i32).withTags(.{.expert});

                            // Mapping of local experts to global expert ids, -1 if the global expert is not present in the local partition
                            const local_expert_mask = global_expert_ids.cmp(.GE, expert_start)
                                .logical(.AND, global_expert_ids.cmp(.LT, expert_start.addConstant(local_num_experts)));
                            const expert_map = local_expert_mask.select(
                                global_expert_ids.sub(expert_start),
                                zml.Tensor.scalar(-1, .i32),
                            );
                            const local_output = triton.fusedExpertsImpl(
                                sharded_inputs[0],
                                sharded_inputs[3],
                                sharded_inputs[4],
                                sharded_inputs[2],
                                sharded_inputs[1],
                                .{},
                                .{
                                    .activation = ctx.activation,
                                    .global_num_experts = ctx.global_num_experts,
                                    .expert_map = expert_map,
                                    .w1_scale = ctx.scales_gate_up,
                                    .w2_scale = ctx.scales_down,
                                    .w1_bias = ctx.bias_gate_up,
                                    .w2_bias = ctx.bias_down,
                                },
                            ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});
                            return zml.ops.allReduce(
                                local_output.reshape(sharded_inputs[0].shape().dims()).withTags(.{ .b, .s, .d }),
                                zml.Tensor.add,
                            );
                        }
                    }).body,
                );
            }

            break :b try triton.fusedExpertsImpl(
                input,
                weights_gate_up,
                weights_down,
                topk_weights,
                topk_ids,
                triton_metadata,
                .{
                    .activation = parameters.triton.activation,
                    .global_num_experts = weights_gate_up.dim(.expert),
                    .w1_scale = scales_gate_up,
                    .w2_scale = scales_down,
                    .w1_bias = bias_gate_up,
                    .w2_bias = bias_down,
                },
            );
        },
        .mosaic_tpu => b: {
            const tpu_metadata = switch (metadata) {
                .mosaic_tpu => |v| v,
                else => return error.InvalidMetadata,
            };

            const expert_partition = weights_gate_up.shape().partition(.expert);

            if (expert_partition.eql(.init(.experts))) {
                const global_num_experts = weights_down.dim(.expert);
                break :b zml.ops.manualComputation(
                    .{ input, topk_ids, topk_weights, weights_gate_up, weights_down },
                    input.shape(),
                    .{
                        .activation = parameters.mosaic_tpu.activation,
                        .global_num_experts = global_num_experts,
                        .scales_gate_up = scales_gate_up,
                        .bias_gate_up = bias_gate_up,
                        .scales_down = scales_down,
                        .bias_down = bias_down,
                    },
                    (struct {
                        fn body(ctx: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                            const local_num_experts = sharded_inputs[3].dim(.expert);
                            const partition_id = zml.ops.partitionId().convert(.i32);
                            const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                            const global_expert_ids = zml.Tensor.arange(.{ .end = ctx.global_num_experts }, .i32).withTags(.{.expert});

                            const local_expert_mask = global_expert_ids.cmp(.GE, expert_start)
                                .logical(.AND, global_expert_ids.cmp(.LT, expert_start.addConstant(local_num_experts)));
                            const expert_map = local_expert_mask.select(
                                global_expert_ids.sub(expert_start),
                                zml.Tensor.scalar(-1, .i32),
                            );
                            const local_output = mosaic_tpu.fusedExpertsImpl(
                                sharded_inputs[0],
                                sharded_inputs[3],
                                sharded_inputs[4],
                                sharded_inputs[2],
                                sharded_inputs[1],
                                .{},
                                .{
                                    .activation = ctx.activation,
                                    .global_num_experts = ctx.global_num_experts,
                                    .expert_map = expert_map,
                                    .w1_scale = ctx.scales_gate_up,
                                    .w2_scale = ctx.scales_down,
                                    .w1_bias = ctx.bias_gate_up,
                                    .w2_bias = ctx.bias_down,
                                },
                            ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});
                            return zml.ops.allReduce(
                                local_output.reshape(sharded_inputs[0].shape().dims()).withTags(.{ .b, .s, .d }),
                                zml.Tensor.add,
                            );
                        }
                    }).body,
                );
            }

            break :b try mosaic_tpu.fusedExpertsImpl(
                input,
                weights_gate_up,
                weights_down,
                topk_weights,
                topk_ids,
                tpu_metadata,
                .{
                    .activation = parameters.mosaic_tpu.activation,
                    .global_num_experts = weights_gate_up.dim(.expert),
                    .w1_scale = scales_gate_up,
                    .w2_scale = scales_down,
                    .w1_bias = bias_gate_up,
                    .w2_bias = bias_down,
                },
            );
        },
    };
}
