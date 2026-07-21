const std = @import("std");

const zml = @import("../zml.zig");
const stdx = zml.stdx;
pub const metal = @import("metal.zig");
pub const mosaic_tpu = @import("mosaic_tpu.zig");
pub const triton = @import("triton.zig");
pub const triton_kernels = @import("triton_kernels/triton_kernels.zig");
pub const triton_a16w4_kernel = @import("triton_kernels/a16w4_kernel.zig");

pub const ActivationMode = enum {
    silu,
    relu,
    gelu,
};

pub const Backend = enum {
    // Could select a more specific name like "triton_sm90_bf16"
    triton,
    mosaic_tpu,
    metal,

    pub fn auto(platform: *const zml.Platform, weights_dtype: zml.DataType) !Backend {
        return switch (platform.target) {
            .cuda, .rocm, .oneapi => switch (weights_dtype) {
                .bf16, .f16, .f32 => .triton,
                else => error.UnsupportedDataType,
            },
            .tpu => switch (weights_dtype) {
                .bf16, .f16, .f32 => .mosaic_tpu,
                else => error.UnsupportedDataType,
            },
            .metal => switch (weights_dtype) {
                .bf16, .f16, .f32 => .metal,
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
            .metal => {},
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        _ = platform;
        return switch (backend) {
            .triton => {},
            .mosaic_tpu => {},
            .metal => {},
        };
    }
};

pub const Parameters = union(Backend) {
    triton: triton.Parameters,
    mosaic_tpu: mosaic_tpu.Parameters,
    metal: metal.Parameters,

    pub const InitOptions = union(Backend) {
        triton: triton.Parameters.InitOptions,
        mosaic_tpu: mosaic_tpu.Parameters.InitOptions,
        metal: metal.Parameters.InitOptions,

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
                .metal => .{ .metal = .{
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
            .metal => |v| .{ .metal = metal.Parameters.init(v) },
        };
    }
};

pub const Metadata = union(Backend) {
    triton: triton.Metadata,
    mosaic_tpu: mosaic_tpu.Metadata,
    metal: metal.Metadata,

    pub const InitOptions = union(Backend) {
        triton: triton.Metadata.InitOptions,
        mosaic_tpu: mosaic_tpu.Metadata.InitOptions,
        metal: metal.Metadata.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .triton => .{ .triton = .{} },
                .mosaic_tpu => .{ .mosaic_tpu = .{} },
                .metal => .{ .metal = .{} },
            };
        }
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .triton => |v| .{ .triton = triton.Metadata.init(v) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = mosaic_tpu.Metadata.init(v) },
            .metal => |v| .{ .metal = metal.Metadata.init(v) },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        return switch (self) {
            .triton => |metadata| .{ .triton = try metadata.initBuffer(io, platform) },
            .mosaic_tpu => |metadata| .{ .mosaic_tpu = try metadata.initBuffer(io, platform) },
            .metal => |metadata| .{ .metal = try metadata.initBuffer(io, platform) },
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .triton => |*metadata| triton.deinitBuffer(metadata),
            .mosaic_tpu => |*metadata| mosaic_tpu.deinitBuffer(metadata),
            .metal => |*metadata| metal.deinitBuffer(metadata),
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
                            const local_reshaped = local_output.reshape(sharded_inputs[0].shape().dims()).withTags(.{ .b, .s, .d });
                            return zml.ops.allReduce(local_reshaped, zml.Tensor.add);
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
                const partial_output = zml.ops.manualComputation(
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
                            return local_output.reshape(sharded_inputs[0].shape().dims()).withTags(.{ .b, .s, .d });
                        }
                    }).body,
                );
                break :b zml.ops.allReduce(partial_output, zml.Tensor.add);
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
        .metal => b: {
            const metal_metadata = switch (metadata) {
                .metal => |v| v,
                else => return error.InvalidMetadata,
            };

            break :b try metal.fusedExpertsImpl(
                input,
                weights_gate_up,
                weights_down,
                topk_weights,
                topk_ids,
                metal_metadata,
                .{
                    .activation = parameters.metal.activation,
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

// ================================

const Routing = struct {
    num_tokens: i64,
    num_routes: i64,
    topk: i64,
    gather_divisor: i64,
    grid_m: i64,
    sorted_route_ids: zml.Tensor,
    sorted_weights: zml.Tensor,
    active_routes: zml.Tensor,
    hist: zml.Tensor,
    offsets: zml.Tensor,
    expt_data: zml.Tensor,
};

const CompactLocalRoutes = struct {
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    token_ids: zml.Tensor,
    overflow: zml.Tensor,
};

const GemmOpts = struct {
    routing: Routing,
    weight_contract_tag: zml.Shape.Tag,
    weight_output_tag: zml.Shape.Tag,
    output_shape: zml.Shape,
    gather: ?zml.Tensor = null,
    gammas: ?zml.Tensor = null,
    bias: ?zml.Tensor = null,
    apply_swiglu: bool = false,
    activation_limit: f32 = 1.0,
    block_m: u32,
    block_n: u32,
    block_k: u32,
    group_m: u32,
    num_warps: u32,
    num_stages: u32,
};

const Vanilla = struct {
    fn fastRoundScale(scale: zml.Tensor) zml.Tensor {
        // Matches kernel.py fast_round_scale(): 2 ** ceil(log2(scale)).
        // DeepSeek V4 uses MXFP/UE8M0 activation scales for FP8 GEMMs.
        const exponent = scale.log().divByConst(@log(2.0)).ceil();
        return zml.Tensor.constant(scale.dtype().one()).broad(scale.shape()).scale(2.0).pow(exponent);
    }

    fn to_fp8(x: zml.Tensor, block_size: u32, round_scale: bool) struct { zml.Tensor, zml.Tensor } {
        // x = [seq, d=4096]
        // block = 128
        stdx.debug.assert(@mod(x.dim(-1), block_size) == 0, "last dimension {} must be divisible by block_size={}", .{ x.dim(-1), block_size });

        // x_blocked = [seq, 32, 128]
        const x_blocked = x.splitAxis(-1, .{ .n = .auto, .m = block_size }).convert(.f32);

        const min_value = zml.Tensor.scalar(1e-4, x_blocked.dtype());

        var max_block = x_blocked.abs().max(-1);
        max_block = zml.Tensor.select(max_block.cmp(.LT, min_value), min_value, max_block);

        const fp8_min = zml.Tensor.constant(zml.DataType.f8e4m3fn.minValue()).convert(x_blocked.dtype());
        const fp8_max = zml.Tensor.constant(zml.DataType.f8e4m3fn.maxValue()).convert(x_blocked.dtype());
        const fp8_max_inv = zml.Tensor.constant(fp8_max.dtype().one()).broad(fp8_max.shape()).div(fp8_max);

        var scale = max_block.mul(fp8_max_inv);
        if (round_scale) {
            scale = fastRoundScale(scale);
        }

        const x_fp8 = x_blocked.div(scale).clamp(fp8_min, fp8_max).reshape(x.shape());

        return .{ x_fp8.convert(.f8e4m3fn), scale.convert(.f8e8m0) };
    }
    fn dequantizeScaledBlocks(weight: zml.Tensor, scale: zml.Tensor, block_size: usize, dtype: zml.DataType) zml.Tensor {
        // Reshape scale from [8, 32] -> [8, 1, 32, 1] to inject dummy block dimensions
        const scale_4d_shape = scale.shape().insert(1, .{1}).insert(.last, .{1});
        const scale_4d = scale.convert(dtype).reshape(scale_4d_shape);

        // Broadcast along the block dimensions (block size = 128)
        // [8, 1, 32, 1] -> [8, 128, 32, 128]
        const broad_shape = scale.shape().insert(1, .{block_size}).insert(.last, .{block_size});
        const scale_broad = scale_4d.broad(broad_shape);

        // Flatten the 4D expanded scale back to 2D matrix matching the weights: [1024, 4096]
        const target_shape = weight.shape();
        const scale_expanded = scale_broad.reshape(target_shape);

        // Element-wise multiplication completes the dequantization pipeline
        return weight.convert(dtype).mul(scale_expanded);
    }
    const FP8Linear = struct {
        scale: zml.Tensor,
        weight: zml.Tensor,
        block_size: usize,
        tag: zml.Shape.Tag,

        pub fn init(store: zml.io.TensorStore.View, tagz: anytype, block_size: usize, proj_tag: anytype) FP8Linear {
            return .{
                .scale = store.createTensor("scale", null, .replicated),
                .weight = store.createTensor("weight", tagz, .replicated),
                .block_size = block_size,
                .tag = zml.Shape.toTag(proj_tag),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(FP8Linear)) void {
            self.scale.deinit();
            self.weight.deinit();
        }

        fn dequantizeActivationF32(x_fp8: zml.Tensor, scale: zml.Tensor, block_size: usize, contract_tag: zml.Shape.Tag) zml.Tensor {
            var x = x_fp8.splitAxis(contract_tag, .{ .kb = .auto, .bk = block_size }).convert(.f32);
            x = x.mul(scale.convert(.f32).broad(x.shape()));
            return x.reshape(x_fp8.shape().withDtype(.f32));
        }

        pub fn forward(self: FP8Linear, x: zml.Tensor) zml.Tensor {
            const x_fp8, const x_scale = to_fp8(x, @intCast(self.block_size), true);
            const x_dequant = dequantizeActivationF32(x_fp8, x_scale, self.block_size, self.tag);
            const weight_dequant = dequantizeScaledBlocks(self.weight, self.scale, self.block_size, .f32);
            return x_dequant.dot(weight_dequant, self.tag).convert(.bf16);
        }
    };
    const FP4Linear = struct {
        scale: zml.Tensor,
        weight: zml.Tensor,
        act_block_size: usize,
        weight_block_size: usize,
        tag: zml.Shape.Tag,

        pub fn init(store: zml.io.TensorStore.View, tagz: anytype, scale_tagz: anytype, act_block_size: usize, proj_tag: anytype) FP4Linear {
            return .{
                .scale = store.createTensor("scale", null, .replicated).withTags(scale_tagz),
                .weight = store.createTensor("weight", tagz, .replicated),
                .act_block_size = act_block_size,
                .tag = zml.Shape.toTag(proj_tag),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(FP4Linear)) void {
            self.scale.deinit();
            self.weight.deinit();
        }

        fn dequantizePackedWeightF32(weight_: zml.Tensor, scale: zml.Tensor, weight_block_size: usize) zml.Tensor {
            const logical_weight_shape = weight_.shape().set(-1, weight_.dim(-1) * 2);
            const weight = weight_.bitCast(.f4e2m1).reshape(logical_weight_shape).convert(.f32);

            const scale_shape = scale.shape();
            const scale_3d = scale.convert(.f32).reshape(scale_shape.insert(.last, .{1}));
            const scale_broad = scale_3d.broad(scale_shape.insert(.last, .{weight_block_size}));
            const scale_expanded = scale_broad.reshape(logical_weight_shape);

            return weight.mul(scale_expanded);
        }

        fn forwardWith(weight: zml.Tensor, scale: zml.Tensor, x: zml.Tensor, act_block_size: usize, weight_block_size: usize, contract_tag: zml.Shape.Tag) zml.Tensor {
            const x_fp8, const x_scale = to_fp8(x, @intCast(act_block_size), true);
            const x_dequant = FP8Linear.dequantizeActivationF32(x_fp8, x_scale, act_block_size, contract_tag);
            const weight_dequant = dequantizePackedWeightF32(weight, scale, weight_block_size);
            return x_dequant.dot(weight_dequant, contract_tag).convert(.bf16);
        }

        pub fn forward(self: FP4Linear, x: zml.Tensor) zml.Tensor {
            return forwardWith(self.weight, self.scale, x, self.act_block_size, self.weight_block_size, self.tag);
        }
    };
    fn forwardRoutedExpert(
        x: zml.Tensor,
        weight: zml.Tensor,
        gate_up_weight: zml.Tensor,
        gate_up_scale: zml.Tensor,
        w2_weight: zml.Tensor,
        w2_scale: zml.Tensor,
        block_size: usize,
        activation_threshold: f32,
    ) zml.Tensor {
        const threshold = zml.Tensor.scalar(activation_threshold, .f32);

        const gate_up = FP4Linear.forwardWith(gate_up_weight, gate_up_scale, x, block_size, 32, zml.Shape.toTag(.d)).convert(.f32);
        var gate = gate_up.slice1d(.dout, .{ .start = 0, .step = 2 });
        gate = zml.Tensor.select(gate.cmp(.GT, threshold), threshold, gate);
        const up = gate_up.slice1d(.dout, .{ .start = 1, .step = 2 }).clamp(threshold.negate(), threshold);

        var hidden = gate.silu().mul(up);
        hidden = hidden.mul(weight.appendAxes(.{.dout}).broad(hidden.shape()));
        return FP4Linear.forwardWith(w2_weight, w2_scale, hidden.convert(x.dtype()), block_size, 32, zml.Shape.toTag(.dout));
    }

    fn forwardMoe_fp4(
        x: zml.Tensor,
        gate_up: zml.Tensor,
        gate_up_scale: zml.Tensor,
        topk_ids: zml.Tensor,
        topk_weight: zml.Tensor,
        down: zml.Tensor,
        down_scale: zml.Tensor,
        block_size: u32,
        activation_threshold: f32,
    ) zml.Tensor {
        var y = zml.Tensor.zeroes(x.shape().withDtype(.f32));

        const weight_dtype = gate_up.dtype();
        const scale_dtype = gate_up_scale.dtype();

        const num_active_experts = topk_ids.dim(.eid);

        for (0..@as(usize, @intCast(num_active_experts))) |route_idx| {
            const expert_ids = topk_ids.choose1d(.eid, @intCast(route_idx));
            const weight = topk_weight.choose1d(.eid, @intCast(route_idx));
            const routed = forwardRoutedExpert(
                x,
                weight,
                gate_up.convert(.bf16).gather(.{ .expert = expert_ids }, .{}).convert(weight_dtype),
                gate_up_scale.convert(.bf16).gather(.{ .expert = expert_ids }, .{}).convert(scale_dtype),
                down.convert(.bf16).gather(.{ .expert = expert_ids }, .{}).convert(weight_dtype),
                down_scale.convert(.bf16).gather(.{ .expert = expert_ids }, .{}).convert(scale_dtype),
                block_size,
                activation_threshold,
            ).convert(.f32);

            y = y.add(routed);
        }

        return y;
    }
};

const KernelConf = struct {
    block_m: u32,
    block_n: u32,
    block_k: u32,
    group_m: u32,
    num_warps: u32,
    num_stages: u32,
};

const kernel_config_token_buckets = [_]u32{
    1,  2,   4,   8,   16,   24,   32,   48,   64,
    96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096,
};

fn configForTokenBucket(num_tokens: u32) KernelConf {
    return switch (num_tokens) {
        1 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 4,
        },
        2 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 4,
        },
        4 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 3,
        },
        8 => .{
            .block_m = 16,
            .block_n = 128,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 3,
        },
        16 => .{
            .block_m = 16,
            .block_n = 64,
            .block_k = 64,
            .group_m = 64,
            .num_warps = 4,
            .num_stages = 5,
        },
        24 => .{
            .block_m = 16,
            .block_n = 64,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 2,
        },
        32 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 2,
        },
        48 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 128,
            .group_m = 64,
            .num_warps = 4,
            .num_stages = 2,
        },
        64 => .{
            .block_m = 16,
            .block_n = 64,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 2,
        },
        96 => .{
            .block_m = 16,
            .block_n = 128,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 3,
        },
        128 => .{
            .block_m = 16,
            .block_n = 256,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 2,
        },
        256 => .{
            .block_m = 16,
            .block_n = 256,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 2,
        },
        512 => .{
            .block_m = 32,
            .block_n = 128,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 3,
        },
        1024 => .{
            .block_m = 64,
            .block_n = 128,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 3,
        },
        1536 => .{
            .block_m = 64,
            .block_n = 128,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 3,
        },
        2048 => .{
            .block_m = 128,
            .block_n = 128,
            .block_k = 64,
            .group_m = 16,
            .num_warps = 8,
            .num_stages = 3,
        },
        3072 => .{
            .block_m = 128,
            .block_n = 256,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 4,
        },
        4096 => .{
            .block_m = 128,
            .block_n = 256,
            .block_k = 64,
            .group_m = 16,
            .num_warps = 8,
            .num_stages = 4,
        },
        else => unreachable,
    };
}

fn getBestConfig(num_tokens: u32, topk: u32, num_experts: u32) KernelConf {
    const num_routes = std.math.mul(u32, num_tokens, topk) catch std.math.maxInt(u32);
    var config = getBestTokenBucketConfig(num_routes);

    if (num_routes <= 256 and num_experts <= 64) {
        config.block_m = 16;
        config.block_n = 256;
        config.block_k = 128;
        config.group_m = 1;
        config.num_warps = 4;
        config.num_stages = 2;
    } else if (num_routes <= 512 and num_experts <= 64) {
        config.block_m = 16;
        config.block_n = 128;
        config.block_k = 128;
        config.group_m = 1;
        config.num_warps = 4;
        config.num_stages = 2;
    }

    return config;
}

fn getBestTokenBucketConfig(num_tokens: u32) KernelConf {
    var best_num_tokens = kernel_config_token_buckets[0];
    var best_distance = tokenDistance(num_tokens, best_num_tokens);

    for (kernel_config_token_buckets[1..]) |candidate| {
        const distance = tokenDistance(num_tokens, candidate);
        if (distance < best_distance or (distance == best_distance and candidate < best_num_tokens)) {
            best_num_tokens = candidate;
            best_distance = distance;
        }
    }

    return configForTokenBucket(best_num_tokens);
}

fn tokenDistance(a: u32, b: u32) u32 {
    return if (a >= b) a - b else b - a;
}

fn compactLocalRouteCapacity(num_tokens: i64, topk: i64, global_num_experts: i64, local_num_experts: i64, block_m: i64) i64 {
    const num_routes = num_tokens * topk;
    if (num_routes > 512 or global_num_experts <= local_num_experts) return num_routes;

    // Decode-time EP routing is expected to be close to balanced; keep slack so
    // normal skew still fits while shrinking the local GEMM launch grid.
    const expected_local_routes = std.math.divCeil(i64, num_routes * local_num_experts, global_num_experts) catch unreachable;
    const slack = @max(expected_local_routes, 2 * block_m);
    const unrounded_capacity = expected_local_routes + slack;
    const rounded_capacity = (std.math.divCeil(i64, unrounded_capacity, block_m) catch unreachable) * block_m;
    return @min(num_routes, rounded_capacity);
}

const Triton = struct {
    pub fn forwardMoe_fp4(
        input: zml.Tensor,
        topk_ids: zml.Tensor,
        topk_weights: zml.Tensor,
        weights_gate_up: zml.Tensor,
        scales_gate_up: zml.Tensor,
        bias_gate_up: ?zml.Tensor,
        weights_down: zml.Tensor,
        scales_down: zml.Tensor,
        bias_down: ?zml.Tensor,
        activation_limit: f32,
    ) !zml.Tensor {
        const expert_partition = weights_gate_up.shape().partition(.expert);
        if (!expert_partition.eql(.init(.experts))) {
            return forwardMoeLocal_fp4(
                input,
                topk_ids,
                topk_weights,
                weights_gate_up,
                scales_gate_up,
                bias_gate_up,
                weights_down,
                scales_down,
                bias_down,
                null,
                activation_limit,
            );
        }

        const output = zml.ops.manualComputation(
            .{ input, topk_ids, topk_weights, weights_gate_up, scales_gate_up, weights_down, scales_down },
            input.shape(),
            .{
                .activation_limit = activation_limit,
                .global_num_experts = weights_gate_up.dim(.expert),
            },
            (struct {
                fn body(ctx: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                    const local_num_experts = sharded_inputs[3].dim(.expert);
                    const partition_id = zml.ops.partitionId().convert(.i32);
                    const expert_start = partition_id.scale(local_num_experts).convert(.i32);

                    const local_topk_ids, const local_topk_weights = blk: {
                        const local_ids = sharded_inputs[1].convert(.i32).sub(expert_start);
                        const in_range = local_ids.cmp(.GE, zml.Tensor.scalar(0, .i32))
                            .logical(.AND, local_ids.cmp(.LT, zml.Tensor.scalar(local_num_experts, .i32)));

                        break :blk .{
                            // Keep the static route shape, but use one-past-the-end as
                            // a sentinel so local routing excludes non-local entries.
                            in_range.select(local_ids, zml.Tensor.scalar(local_num_experts, .i32)),
                            in_range.select(sharded_inputs[2], zml.Tensor.scalar(0, sharded_inputs[2].dtype())),
                        };
                    };

                    const output = if (compactLocalRoutes(
                        local_topk_ids,
                        local_topk_weights,
                        local_num_experts,
                        ctx.global_num_experts,
                        16,
                    )) |routes| blk: {
                        const compact_output = forwardMoeLocal_fp4(
                            sharded_inputs[0],
                            routes.topk_ids,
                            routes.topk_weights,
                            sharded_inputs[3],
                            sharded_inputs[4],
                            null,
                            sharded_inputs[5],
                            sharded_inputs[6],
                            null,
                            routes.token_ids,
                            ctx.activation_limit,
                        );

                        const fallback_ctx = .{
                            .input = sharded_inputs[0],
                            .topk_ids = local_topk_ids,
                            .topk_weights = local_topk_weights,
                            .weights_gate_up = sharded_inputs[3],
                            .scales_gate_up = sharded_inputs[4],
                            .weights_down = sharded_inputs[5],
                            .scales_down = sharded_inputs[6],
                            .activation_limit = ctx.activation_limit,
                        };
                        const OverflowFallback = struct {
                            fn cond(overflow: zml.Tensor, _: zml.Tensor, _: anytype) zml.Tensor {
                                return overflow;
                            }

                            fn body(_: zml.Tensor, _: zml.Tensor, fallback: anytype) struct { zml.Tensor, zml.Tensor } {
                                const dense_output = forwardMoeLocal_fp4(
                                    fallback.input,
                                    fallback.topk_ids,
                                    fallback.topk_weights,
                                    fallback.weights_gate_up,
                                    fallback.scales_gate_up,
                                    null,
                                    fallback.weights_down,
                                    fallback.scales_down,
                                    null,
                                    null,
                                    fallback.activation_limit,
                                );
                                return .{ zml.Tensor.scalar(false, .bool), dense_output };
                            }
                        };
                        const loop_state = zml.ops.@"while"(
                            .{ routes.overflow, compact_output },
                            OverflowFallback.cond,
                            OverflowFallback.body,
                            .{fallback_ctx},
                        );
                        break :blk loop_state[1];
                    } else forwardMoeLocal_fp4(
                        sharded_inputs[0],
                        local_topk_ids,
                        local_topk_weights,
                        sharded_inputs[3],
                        sharded_inputs[4],
                        null,
                        sharded_inputs[5],
                        sharded_inputs[6],
                        null,
                        null,
                        ctx.activation_limit,
                    );

                    const out_reshaped = output.reshape(sharded_inputs[0].shape().dims()).withTags(.{ .b, .d });
                    return zml.ops.allReduce(out_reshaped, zml.Tensor.add);
                }
            }).body,
        );

        return output;
    }

    fn forwardMoeLocal_fp4(
        input: zml.Tensor,
        topk_ids: zml.Tensor,
        topk_weights: zml.Tensor,
        weights_gate_up: zml.Tensor,
        scales_gate_up: zml.Tensor,
        bias_gate_up: ?zml.Tensor,
        weights_down: zml.Tensor,
        scales_down: zml.Tensor,
        bias_down: ?zml.Tensor,
        route_token_ids: ?zml.Tensor,
        activation_limit: f32,
    ) zml.Tensor {
        const x = input.rename(.{ .b = .token });
        const kernel_cfg = getBestConfig(
            @intCast(topk_ids.dim(.b)),
            @intCast(topk_ids.dim(.eid)),
            @intCast(weights_gate_up.dim(.expert)),
        );
        const routing = prepareRouting(
            topk_ids,
            topk_weights,
            route_token_ids,
            x.dim(.token),
            weights_gate_up.dim(.expert),
            @intCast(kernel_cfg.block_m),
        );

        const hidden_shape: zml.Shape = .init(.{
            .route = routing.num_routes,
            .dout = @divExact(weights_gate_up.dim(.dout), 2),
        }, .bf16);

        const hidden = runGemm(
            x,
            weights_gate_up,
            scales_gate_up,
            .{
                .routing = routing,
                .weight_contract_tag = zml.Shape.toTag(.d),
                .weight_output_tag = zml.Shape.toTag(.dout),
                .output_shape = hidden_shape,
                .gather = routing.sorted_route_ids,
                .gammas = routing.sorted_weights,
                .bias = bias_gate_up,
                .apply_swiglu = true,
                .activation_limit = activation_limit,
                .block_m = kernel_cfg.block_m,
                .block_n = kernel_cfg.block_n,
                .block_k = kernel_cfg.block_k,
                .group_m = kernel_cfg.group_m,
                .num_warps = kernel_cfg.num_warps,
                .num_stages = kernel_cfg.num_stages,
            },
        );

        const routed_shape: zml.Shape = .init(.{
            .route = routing.num_routes,
            .d = weights_down.dim(.d),
        }, .bf16);

        const routed = runGemm(
            hidden,
            weights_down,
            scales_down,
            .{
                .routing = routing,
                .weight_contract_tag = zml.Shape.toTag(.dout),
                .weight_output_tag = zml.Shape.toTag(.d),
                .output_shape = routed_shape,
                .bias = bias_down,
                .apply_swiglu = false,
                .activation_limit = 1.0,
                .block_m = kernel_cfg.block_m,
                .block_n = kernel_cfg.block_n,
                .block_k = kernel_cfg.block_k,
                .group_m = kernel_cfg.group_m,
                .num_warps = kernel_cfg.num_warps,
                .num_stages = kernel_cfg.num_stages,
            },
        );

        const active_routed = blk: {
            const mask = routing.active_routes.broad(routed.shape().withDtype(.bool));
            break :blk mask.select(routed, zml.Tensor.zeroes(routed.shape()));
        };

        const token_ids = routing.sorted_route_ids.divByConst(routing.gather_divisor).withTags(.{.route});
        const output_flat_shape: zml.Shape = .init(.{ .token = routing.num_tokens, .d = input.dim(.d) }, .f32);
        const output_flat = zml.Tensor.zeroes(output_flat_shape).scatterSlices(
            .{ .token = token_ids },
            active_routed.convert(.f32),
            .{},
        );

        return output_flat.reshape(input.shape().withDtype(.f32)).convert(input.dtype());
    }

    fn compactLocalRoutes(
        local_topk_ids: zml.Tensor,
        local_topk_weights: zml.Tensor,
        local_num_experts: i64,
        global_num_experts: i64,
        block_m: i64,
    ) ?CompactLocalRoutes {
        const num_tokens = local_topk_ids.dim(.b);
        const topk = local_topk_ids.dim(.eid);
        const num_routes = num_tokens * topk;
        const route_capacity = compactLocalRouteCapacity(num_tokens, topk, global_num_experts, local_num_experts, block_m);
        if (route_capacity >= num_routes) return null;

        const compact_route_shape: zml.Shape = .init(.{ .route = route_capacity }, .i32);
        const flat_ids = local_topk_ids.flatten().withTags(.{.route}).convert(.i32);
        const valid = flat_ids.cmp(.GE, zml.Tensor.scalar(0, .i32))
            .logical(.AND, flat_ids.cmp(.LT, zml.Tensor.scalar(local_num_experts, .i32)));
        const valid_i32 = valid.convert(.i32);
        const local_rank = valid_i32.cumulativeSum(.route).sub(valid_i32).withTags(.{.route});
        const local_route_count = valid_i32.sum(.route).squeeze(.route);
        const overflow = local_route_count.cmp(.GT, zml.Tensor.scalar(route_capacity, .i32));
        const in_capacity = valid.logical(.AND, local_rank.cmp(.LT, zml.Tensor.scalar(route_capacity, .i32)));
        const compact_index = in_capacity.select(local_rank, zml.Tensor.scalar(0, .i32).broad(local_rank.shape()));

        const invalid_expert = zml.Tensor.scalar(-1, .i32);
        const compact_ids_flat = invalid_expert.broad(compact_route_shape).scatterSlices(
            .{ .route = compact_index },
            in_capacity.select(flat_ids, invalid_expert.broad(flat_ids.shape())),
            .{ .update_fn = scatterMax },
        );

        const flat_weights = local_topk_weights.flatten().withTags(.{.route});
        const compact_weights_flat = zml.Tensor.zeroes(compact_route_shape.withDtype(local_topk_weights.dtype())).scatterSlices(
            .{ .route = compact_index },
            in_capacity.select(flat_weights, zml.Tensor.zeroes(flat_weights.shape())),
            .{},
        );

        const route_ids = zml.Tensor.arange(.{ .end = num_routes }, .i32).withTags(.{.route});
        const token_ids = route_ids.divByConst(topk).withTags(.{.route});
        const compact_token_ids_flat = zml.Tensor.zeroes(compact_route_shape).scatterSlices(
            .{ .route = compact_index },
            in_capacity.select(token_ids, zml.Tensor.zeroes(token_ids.shape())),
            .{},
        );

        return .{
            .topk_ids = compact_ids_flat.reshape(.{ .b = route_capacity, .eid = 1 }),
            .topk_weights = compact_weights_flat.reshape(.{ .b = route_capacity, .eid = 1 }),
            .token_ids = compact_token_ids_flat.reshape(.{ .b = route_capacity, .eid = 1 }),
            .overflow = overflow,
        };
    }

    fn prepareRouting(topk_ids: zml.Tensor, topk_weights: zml.Tensor, route_token_ids: ?zml.Tensor, output_num_tokens: i64, num_experts: i64, block_m: i64) Routing {
        const topk = topk_ids.dim(.eid);
        const num_tokens = topk_ids.dim(.b);
        const num_routes = num_tokens * topk;
        const gather_divisor: i64 = if (route_token_ids != null) 1 else topk;

        const grid_m = blk: {
            if (num_routes <= num_experts) break :blk num_routes;
            break :blk (std.math.divCeil(i64, num_routes - num_experts + 1, block_m) catch unreachable) + num_experts - 1;
        };

        const route_ids = topk_ids.convert(.i32);
        const valid_route_ids = route_ids.cmp(.GE, zml.Tensor.scalar(0, .i32))
            .logical(.AND, route_ids.cmp(.LT, zml.Tensor.scalar(num_experts, .i32)));
        const routable_ids = valid_route_ids.select(route_ids, zml.Tensor.scalar(num_experts, .i32));
        const sorted = routable_ids.flatten().withTags(.{.route}).sort(.route, .{});
        const sorted_ids = sorted.values.withTags(.{.route}).convert(.i32);
        const sorted_route_indices = sorted.indices.withTags(.{.route}).convert(.i32);
        const sorted_route_ids = if (route_token_ids) |token_ids|
            token_ids.flatten().withTags(.{.route})
                .gather(.{ .route = sorted_route_indices.rename(.{ .route = .sorted_route }) }, .{})
                .rename(.{ .sorted_route = .route })
                .convert(.i32)
        else
            sorted_route_indices;
        const active_routes = sorted_ids.cmp(.LT, zml.Tensor.scalar(num_experts, .i32));

        const sorted_weights = topk_weights.flatten().withTags(.{.route})
            .gather(.{ .route = sorted_route_indices.rename(.{ .route = .sorted_route }) }, .{})
            .rename(.{ .sorted_route = .route })
            .convert(.f32);

        const experts = zml.Tensor.arange(.{ .end = num_experts }, .i32).withTags(.{.expert});
        const route_expert_shape: zml.Shape = .init(.{ .route = num_routes, .expert = num_experts }, .i32);
        const ids_by_expert = sorted_ids.insertAxes(.last, .{.expert}).broad(route_expert_shape);
        const expert_ids = experts.insertAxes(0, .{.route}).broad(route_expert_shape);

        const hist = ids_by_expert.cmp(.EQ, expert_ids)
            .convert(.i32)
            .sum(.route)
            .squeeze(.route)
            .withTags(.{.expert});

        const offsets = hist.cumulativeSum(.expert).sub(hist).withTags(.{.expert});

        const expert_data = buildExpertBlockMap(hist, num_routes, grid_m, block_m);

        return .{
            .num_tokens = output_num_tokens,
            .num_routes = num_routes,
            .topk = topk,
            .gather_divisor = gather_divisor,
            .grid_m = grid_m,
            .sorted_route_ids = sorted_route_ids,
            .sorted_weights = sorted_weights,
            .active_routes = active_routes,
            .hist = hist,
            .offsets = offsets,
            .expt_data = expert_data,
        };
    }

    fn buildExpertBlockMap(hist: zml.Tensor, num_routes: i64, grid_m: i64, block_m: i64) zml.Tensor {
        const num_experts = hist.dim(.expert);
        const max_blocks_per_expert = std.math.divCeil(i64, num_routes, block_m) catch unreachable;

        const tiles_per_expert = hist.addConstant(block_m - 1).divByConst(block_m).withTags(.{.expert});
        const tile_offsets = tiles_per_expert.cumulativeSum(.expert).sub(tiles_per_expert).withTags(.{.expert});

        const expert_ids = zml.Tensor.arange(.{ .end = num_experts }, .i32).withTags(.{.expert});
        const block_ids = zml.Tensor.arange(.{ .end = max_blocks_per_expert }, .i32).withTags(.{.block});
        const grid_shape: zml.Shape = .init(.{ .expert = num_experts, .block = max_blocks_per_expert }, .i32);

        const block_grid = block_ids.insertAxes(0, .{.expert}).broad(grid_shape);
        const valid = block_grid.cmp(.LT, tiles_per_expert.insertAxes(.last, .{.block}).broad(grid_shape));
        const target_idx = valid.select(
            tile_offsets.insertAxes(.last, .{.block}).broad(grid_shape).add(block_grid),
            zml.Tensor.scalar(0, .i32).broad(grid_shape),
        );
        const packed_data = block_grid.scale(65536).add(expert_ids.insertAxes(.last, .{.block}).broad(grid_shape));
        const updates = valid.select(packed_data, zml.Tensor.scalar(-1, .i32).broad(grid_shape));

        return zml.Tensor.scalar(-1, .i32)
            .broad(zml.Shape.init(.{ .tile = grid_m }, .i32))
            .scatterSlices(.{ .tile = target_idx }, updates, .{ .update_fn = scatterMax });
    }

    fn scatterMax(values: zml.ops.ScatterArgs) struct { zml.Tensor } {
        return .{values.input.maximum(values.update)};
    }

    fn runGemm(
        input: zml.Tensor,
        weights: zml.Tensor,
        scales: zml.Tensor,
        opts: GemmOpts,
    ) zml.Tensor {
        const input_matrix = input.withTags(.{ .row, .k });
        const contract_k = input_matrix.dim(.k);
        const packed_k = weights.dim(opts.weight_contract_tag);
        const scale_k = scales.dim(opts.weight_contract_tag);
        const n = weights.dim(opts.weight_output_tag);

        stdx.debug.assert(packed_k * 2 == contract_k, "expected packed int4 weight K {} to match activation K {}", .{ packed_k, contract_k });
        stdx.debug.assert(scale_k * 32 == contract_k, "expected MX scale K {} to match activation K {}", .{ scale_k, contract_k });
        const activation_reduction_n: i64 = if (opts.apply_swiglu) 2 else 1;
        stdx.debug.assert(@mod(n, activation_reduction_n) == 0, "invalid GEMM output width {}", .{n});
        stdx.debug.assert(opts.output_shape.dim(-1) == @divExact(n, activation_reduction_n), "output shape {f} does not match GEMM N {}", .{ opts.output_shape, n });

        const block_m: i32 = @intCast(opts.block_m);
        const block_n: i32 = @intCast(opts.block_n);
        const block_k: i32 = @intCast(opts.block_k);
        const grid_n = std.math.divCeil(i64, n, block_n) catch unreachable;
        const has_bias = opts.bias != null;
        const has_gather = opts.gather != null;
        const has_gammas = opts.gammas != null;

        const cfg: triton_a16w4_kernel.Cfg = .{
            .x_dtype = zml.kernel.triton.from(input_matrix.dtype()),
            .w_dtype = packedByteDtype(weights.dtype()),
            .w_mx_scale_dtype = packedByteDtype(scales.dtype()),
            .b_dtype = zml.kernel.triton.from((opts.bias orelse input_matrix).dtype()),
            .gammas_dtype = zml.kernel.triton.from((opts.gammas orelse zml.Tensor.scalar(1.0, .f32)).dtype()),
            .y_dtype = zml.kernel.triton.from(opts.output_shape.dtype()),
            .HAS_B = has_bias,
            .HAS_GAMMAS = has_gammas,
            .HAS_GATHER_INDX = has_gather,
            .APPLY_SWIGLU = opts.apply_swiglu,
            .ACTIVATION_REDUCTION_N = @intCast(activation_reduction_n),
            .SWIGLU_ADD_RESIDUAL = false,
            .N_EXPTS_ACT = @intCast(opts.routing.topk),
            .BLOCK_M = block_m,
            .BLOCK_N = block_n,
            .BLOCK_K = block_k,
            .GROUP_M = @intCast(opts.group_m),
            .XCD_SWIZZLE = 1,
            .EVEN_K = @mod(contract_k, block_k) == 0,
            .MASK_K_LIMIT = @intCast(if (@mod(contract_k, block_k) == 0) block_k else @mod(contract_k, block_k)),
            .W_CACHE_MODIFIER = if (block_m <= 32) .cg else .none,
        };

        const y = triton_a16w4_kernel.Kernel.call(
            .{
                .stride_y_k = scalarI64(0),
                .stride_y_m = scalarI64(opts.output_shape.dim(-1)),
                .stride_y_n = scalarI64(1),
                .X = input_matrix,
                .stride_x_m = scalarI64(contract_k),
                .stride_x_k = scalarI64(1),
                .W = weights,
                .stride_w_e = scalarI64(n * packed_k),
                .stride_w_k = scalarI64(1),
                .stride_w_n = scalarI64(packed_k),
                .WMxScale = scales,
                .stride_w_mx_e = scalarI64(n * scale_k),
                .stride_w_mx_k = scalarI64(1),
                .stride_w_mx_n = scalarI64(scale_k),
                .B = opts.bias orelse input_matrix,
                .stride_b_e = scalarI64(if (has_bias) n else 0),
                .Gammas = opts.gammas orelse zml.Tensor.scalar(1.0, .f32),
                .N = scalarI64(n),
                .K = scalarI64(contract_k),
                .GatherIndx = opts.gather orelse opts.routing.sorted_route_ids,
                .ExptHist = opts.routing.hist,
                .ExptOffs = opts.routing.offsets,
                .ExptOffsSum = zml.Tensor.scalar(0, .i32),
                .ExptData = opts.routing.expt_data,
                .grid_m = scalarI64(opts.routing.grid_m),
                .grid_n = scalarI64(grid_n),
                .alpha = scalarF32(1.0),
                .limit = scalarF32(opts.activation_limit),
            },
            .{ .Y = opts.output_shape },
            .{
                .cfg = cfg,
                .grid = .{ @intCast(opts.routing.grid_m * grid_n), 1, 1 },
                .num_warps = @intCast(opts.num_warps),
                .num_stages = @intCast(opts.num_stages),
            },
        ).Y;

        return y;
    }

    fn packedByteDtype(dt: zml.DataType) zml.kernel.triton.DType {
        return switch (dt) {
            .i8, .u8, .f4e2m1, .f8e8m0 => .i8,
            else => zml.kernel.triton.from(dt),
        };
    }

    fn scalarI64(v: i64) zml.Tensor {
        return zml.Tensor.constant(.{ .i64 = v }).reshape(.{1});
    }

    fn scalarF32(v: f32) zml.Tensor {
        return zml.Tensor.constant(.{ .f32 = v }).reshape(.{1});
    }
};

pub fn forwardMoe_fp4(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    weights_gate_up: zml.Tensor,
    scales_gate_up: ?zml.Tensor,
    bias_gate_up: ?zml.Tensor,
    weights_down: zml.Tensor,
    scales_down: ?zml.Tensor,
    bias_down: ?zml.Tensor,
    activation_limit: f32,
    metadata: Metadata,
    parameters: zml.moe.Parameters,
) !zml.Tensor {
    _ = metadata; // autofix
    stdx.debug.assert(input.shape().hasTags(.{ .b, .d }), "expected MoE input tags (.b, .d), got {f}", .{input.shape()});
    stdx.debug.assert(topk_ids.shape().hasTags(.{ .b, .eid }), "expected topk id tags (.b, .eid), got {f}", .{topk_ids.shape()});
    stdx.debug.assert(topk_weights.shape().hasTags(.{ .b, .eid }), "expected topk weight tags (.b, .eid), got {f}", .{topk_weights.shape()});
    stdx.debug.assert(bias_gate_up == null and bias_down == null, "partitioned A16W4 MoE bias is not wired yet", .{});

    const scales_down_ = scales_down orelse return error.MissingWeightScale;
    const scales_gate_up_ = scales_gate_up orelse return error.MissingWeightScale;

    const block_size = 128;

    return switch (parameters) {
        .triton => Triton.forwardMoe_fp4(
            input,
            topk_ids,
            topk_weights,
            weights_gate_up,
            scales_gate_up_,
            bias_gate_up,
            weights_down,
            scales_down_,
            bias_down,
            activation_limit,
        ),
        else => Vanilla.forwardMoe_fp4(
            input,
            weights_gate_up,
            scales_gate_up_,
            topk_ids,
            topk_weights,
            weights_down,
            scales_down_,
            block_size,
            activation_limit,
        ),
    };
}
