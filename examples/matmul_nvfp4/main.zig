const std = @import("std");

const zml = @import("zml");
const BFloat16 = zml.floats.BFloat16;
const stdx = zml.stdx;
const block_config: i32 = 16;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.matmul_nvfp4);

const Args = struct {
    model: []const u8,
    batch: usize = 1,
    seed: u64 = 42,

    pub const help =
        \\Use matmul_nvfp4 --model=<path> [--batch=<n>] [--seed=<n>]
        \\
        \\Options:
        \\  --model=<path>      Path to the model repository (required)
        \\  --batch=<n>         Batch size for synthetic input (default: 1)
        \\  --seed=<n>          Random seed for synthetic input (default: 42)
        \\
    ;
};

// const GemmaLayer0Mlp = struct {
//     up_proj: zml.nn.Linear,
//     gate_proj: zml.nn.Linear,
//     down_proj: zml.nn.Linear,
//     quantization_tensors: QuantizationTensors,

//     const QuantizationTensors = struct {
//         up_proj_input_scale: zml.Tensor,
//         up_proj_scale: zml.Tensor,
//         up_proj_scale_2: zml.Tensor,

//         gate_proj_input_scale: zml.Tensor,
//         gate_proj_scale: zml.Tensor,
//         gate_proj_scale_2: zml.Tensor,

//         down_proj_input_scale: zml.Tensor,
//         down_proj_scale: zml.Tensor,
//         down_proj_scale_2: zml.Tensor,
//     };

//     pub fn init(store: zml.io.TensorStore.View) !GemmaLayer0Mlp {
//         const up_proj_store = store.withPrefix("up_proj");
//         const gate_proj_store = store.withPrefix("gate_proj");
//         const down_proj_store = store.withPrefix("down_proj");

//         const up_weight = up_proj_store.createTensor("weight", .{ .dout, .d_packed }, null);
//         const gate_weight = gate_proj_store.createTensor("weight", .{ .dout, .d_packed }, null);
//         const down_weight = down_proj_store.createTensor("weight", .{ .dout, .d_packed }, null);

//         const up_proj_input_scale = up_proj_store.maybeCreateTensor("input_scale", .{}, null) orelse return error.MissingQuantizationTensors;
//         const up_proj_scale = up_proj_store.maybeCreateTensor("weight_scale", .{ .dout, .d_group }, null) orelse return error.MissingQuantizationTensors;
//         const up_proj_scale_2 = up_proj_store.maybeCreateTensor("weight_scale_2", .{}, null) orelse return error.MissingQuantizationTensors;

//         const gate_proj_input_scale = gate_proj_store.maybeCreateTensor("input_scale", .{}, null) orelse return error.MissingQuantizationTensors;
//         const gate_proj_scale = gate_proj_store.maybeCreateTensor("weight_scale", .{ .dout, .d_group }, null) orelse return error.MissingQuantizationTensors;
//         const gate_proj_scale_2 = gate_proj_store.maybeCreateTensor("weight_scale_2", .{}, null) orelse return error.MissingQuantizationTensors;

//         const down_proj_input_scale = down_proj_store.maybeCreateTensor("input_scale", .{}, null) orelse return error.MissingQuantizationTensors;
//         const down_proj_scale = down_proj_store.maybeCreateTensor("weight_scale", .{ .dout, .d_group }, null) orelse return error.MissingQuantizationTensors;
//         const down_proj_scale_2 = down_proj_store.maybeCreateTensor("weight_scale_2", .{}, null) orelse return error.MissingQuantizationTensors;

//         return .{
//             .up_proj = .init(up_weight, null, .d),
//             .gate_proj = .init(gate_weight, null, .d),
//             .down_proj = .init(down_weight, null, .d),
//             .quantization_tensors = .{
//                 .up_proj_input_scale = up_proj_input_scale,
//                 .up_proj_scale = up_proj_scale,
//                 .up_proj_scale_2 = up_proj_scale_2,
//                 .gate_proj_input_scale = gate_proj_input_scale,
//                 .gate_proj_scale = gate_proj_scale,
//                 .gate_proj_scale_2 = gate_proj_scale_2,
//                 .down_proj_input_scale = down_proj_input_scale,
//                 .down_proj_scale = down_proj_scale,
//                 .down_proj_scale_2 = down_proj_scale_2,
//             },
//         };
//     }

//     pub fn load(
//         self: *const GemmaLayer0Mlp,
//         allocator: std.mem.Allocator,
//         io: std.Io,
//         platform: *const zml.Platform,
//         store: *const zml.io.TensorStore,
//         replicated: zml.sharding.Sharding,
//     ) !zml.Bufferized(GemmaLayer0Mlp) {
//         return zml.io.load(GemmaLayer0Mlp, self, allocator, io, platform, store, .{
//             .parallelism = 1,
//             .shardings = &.{replicated},
//             .dma_chunks = 2,
//             .dma_chunk_size = 16 * 1024 * 1024,
//         });
//     }

//     pub fn unloadBuffers(self: *zml.Bufferized(GemmaLayer0Mlp)) void {
//         self.up_proj.weight.deinit();
//         if (self.up_proj.bias) |*bias| bias.deinit();
//         self.gate_proj.weight.deinit();
//         if (self.gate_proj.bias) |*bias| bias.deinit();
//         self.down_proj.weight.deinit();
//         if (self.down_proj.bias) |*bias| bias.deinit();

//         self.quantization_tensors.up_proj_input_scale.deinit();
//         self.quantization_tensors.up_proj_scale.deinit();
//         self.quantization_tensors.up_proj_scale_2.deinit();
//         self.quantization_tensors.gate_proj_input_scale.deinit();
//         self.quantization_tensors.gate_proj_scale.deinit();
//         self.quantization_tensors.gate_proj_scale_2.deinit();
//         self.quantization_tensors.down_proj_input_scale.deinit();
//         self.quantization_tensors.down_proj_scale.deinit();
//         self.quantization_tensors.down_proj_scale_2.deinit();
//     }

//     pub fn forward(self: GemmaLayer0Mlp, x: zml.Tensor) zml.Tensor {
//         const up_proj_scale_bf16 = self.quantization_tensors.up_proj_scale_2.convert(.bf16);
//         const gate_proj_scale_bf16 = self.quantization_tensors.gate_proj_scale_2.convert(.bf16);
//         const down_proj_scale_bf16 = self.quantization_tensors.down_proj_scale_2.convert(.bf16);
//         const out_quantized = quantizeActivationNVFP4(x);
//         const proj = forwardLinearDequantizeNVFP4(
//             self.up_proj,
//             out_quantized.block,
//             out_quantized.scale,
//             self.quantization_tensors.up_proj_scale,
//             up_proj_scale_bf16,
//         );

//         var output = forwardLinearDequantizeNVFP4(
//             self.gate_proj,
//             out_quantized.block,
//             out_quantized.scale,
//             self.quantization_tensors.gate_proj_scale,
//             gate_proj_scale_bf16,
//         );
//         output = output.gelu().mul(proj).rename(.{ .dout = .d });

//         const activated_quant = quantizeActivationNVFP4(output);

//         return forwardLinearDequantizeNVFP4(
//             self.down_proj,
//             activated_quant.block,
//             activated_quant.scale,
//             self.quantization_tensors.down_proj_scale,
//             down_proj_scale_bf16,
//         );
//     }
// };

pub fn scale_dot_product(lhs_block: zml.Tensor, lhs_scale: zml.Tensor, rhs_block: zml.Tensor, rhs_scale: zml.Tensor, global_scale: zml.Tensor) zml.Tensor {
    const output_shape = zml.Shape.init(.{ .b = lhs_block.dim(.b), .dout = rhs_block.dim(.dout) }, .f32);
    // const dequant_type: []const u8 = "BF16";
    return zml.ops.customCall(
        "__op$block_scaled_dot",
        .{ lhs_block, rhs_block, lhs_scale, rhs_scale, global_scale },
        output_shape,
        .{
            // .dequantize_type = dequant_type,
        },
        .{ .has_side_effect = false },
    );
}

pub fn dequantize(block: zml.Tensor, scale: zml.Tensor, _type: zml.DataType) zml.Tensor {
    // Step 1: Convert both tensors
    const block_converted = block.convert(_type);
    const scale_converted = scale.convert(_type);

    // Step 2: Broadcast and reshape scale tensor.
    const scale_broadcast = scale_converted.broad(scale_converted.shape().appendDim(block_config, null));
    const scale_reshaped = scale_broadcast.reshape(block_converted.shape());

    const result = block_converted.mul(scale_reshaped);
    return result;
}

const QuantizeNVFP4Result = struct {
    block: zml.Tensor,
    scale: zml.Tensor,
};

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

/// Quantize activations into NVFP4 blocks and per-block scales.
/// Matches vLLM NVFP4 quant math (global scale aware).
pub fn quantizeActivationNVFP4(x: zml.Tensor, input_global_scale: zml.Tensor) QuantizeNVFP4Result {
    const block_size: i64 = block_config;
    const d_group = @divExact(x.dim(.d), block_size);

    // Reshape into blocks: [b, d_group, d_block]
    const x_blocks = x.reshape(.{ .b = x.dim(.b), .d_group = d_group, .d_block = block_size });

    // amax per block
    var amax = x_blocks.abs().max(.d_block);
    // Ensure scale has rank-2 shape: [b, d_group]
    amax = amax.reshape(.{ .b = x.dim(.b), .d_group = d_group });

    // scale = input_global_scale * (amax * (1/6))
    const one_sixth = zml.Tensor.scalar(@as(f32, 1.0 / 6.0), .f32);
    const input_scale_bc = input_global_scale.broad(amax.shape());
    var scale = amax.mul(one_sixth.broad(amax.shape())).mul(input_scale_bc);

    // Clamp scale to FP8 E4M3 range (max 448) to match vLLM behavior.
    const max_scale = zml.Tensor.scalar(@as(f32, 448.0), .f32).broad(scale.shape());
    scale = scale.minimum(max_scale);

    // output_scale = reciprocal(scale * reciprocal(input_global_scale))
    const zero = zml.Tensor.scalar(@as(f32, 0.0), .f32);
    const huge = zml.Tensor.scalar(@as(f32, 1e8), .f32);
    const input_scale_inv = zml.Tensor.scalar(@as(f32, 1.0), .f32).div(input_global_scale);
    const input_scale_inv_bc = input_scale_inv.broad(scale.shape());
    const denom = scale.mul(input_scale_inv_bc);
    const denom_safe = denom.add(denom.cmp(.EQ, zero.broad(denom.shape())).select(huge.broad(denom.shape()), zero.broad(denom.shape())));
    const output_scale = zml.Tensor.scalar(@as(f32, 1.0), .f32).div(denom_safe);

    // Apply output_scale to blocks and clamp to [-6, 6].
    const output_scale_bc = output_scale.broad(x_blocks.shape());
    var scaled_x = x_blocks.mul(output_scale_bc).reshape(.{ .b = x.dim(.b), .d = x.dim(.d) });
    const min_val = zml.Tensor.scalar(@as(f32, -6.0), .f32).broad(scaled_x.shape());
    const max_val = zml.Tensor.scalar(@as(f32, 6.0), .f32).broad(scaled_x.shape());
    scaled_x = scaled_x.maximum(min_val).minimum(max_val);

    return .{
        .block = scaled_x.convert(.f4e2m1),
        .scale = scale.convert(.f8e4m3fn),
    };
}

const MatmulNVFP4 = struct {
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,
    quantization_tensors: QuantizationTensors,

    const QuantizationTensors = struct {
        up_proj_input_scale: zml.Tensor,
        up_proj_scale: zml.Tensor,
        up_proj_scale_2: zml.Tensor,

        gate_proj_input_scale: zml.Tensor,
        gate_proj_scale: zml.Tensor,
        gate_proj_scale_2: zml.Tensor,

        down_proj_input_scale: zml.Tensor,
        down_proj_scale: zml.Tensor,
        down_proj_scale_2: zml.Tensor,
    };

    pub fn init(store: zml.io.TensorStore.View) !MatmulNVFP4 {
        const mlp_store = store.withPrefix("mlp");
        const up_proj_store = mlp_store.withPrefix("up_proj");
        const gate_proj_store = mlp_store.withPrefix("gate_proj");
        const down_proj_store = mlp_store.withPrefix("down_proj");

        const up_weight = up_proj_store.createTensor("weight", .{ .dout, .d_packed }, null);
        const gate_weight = gate_proj_store.createTensor("weight", .{ .dout, .d_packed }, null);
        const down_weight = down_proj_store.createTensor("weight", .{ .dout, .d_packed }, null);

        const up_proj_input_scale = up_proj_store.maybeCreateTensor("input_scale", .{}, null) orelse return error.MissingQuantizationTensors;
        const up_proj_scale = up_proj_store.maybeCreateTensor("weight_scale", .{ .dout, .d_group }, null) orelse return error.MissingQuantizationTensors;
        const up_proj_scale_2 = up_proj_store.maybeCreateTensor("weight_scale_2", .{}, null) orelse return error.MissingQuantizationTensors;

        const gate_proj_input_scale = gate_proj_store.maybeCreateTensor("input_scale", .{}, null) orelse return error.MissingQuantizationTensors;
        const gate_proj_scale = gate_proj_store.maybeCreateTensor("weight_scale", .{ .dout, .d_group }, null) orelse return error.MissingQuantizationTensors;
        const gate_proj_scale_2 = gate_proj_store.maybeCreateTensor("weight_scale_2", .{}, null) orelse return error.MissingQuantizationTensors;

        const down_proj_input_scale = down_proj_store.maybeCreateTensor("input_scale", .{}, null) orelse return error.MissingQuantizationTensors;
        const down_proj_scale = down_proj_store.maybeCreateTensor("weight_scale", .{ .dout, .d_group }, null) orelse return error.MissingQuantizationTensors;
        const down_proj_scale_2 = down_proj_store.maybeCreateTensor("weight_scale_2", .{}, null) orelse return error.MissingQuantizationTensors;

        return .{
            .pre_feedforward_layernorm = RmsNorm.init(store.withPrefix("pre_feedforward_layernorm"), 1e-6),
            .post_feedforward_layernorm = RmsNorm.init(store.withPrefix("post_feedforward_layernorm"), 1e-6),
            .up_proj = .init(up_weight, null, .d),
            .gate_proj = .init(gate_weight, null, .d),
            .down_proj = .init(down_weight, null, .d),
            .quantization_tensors = .{
                .up_proj_input_scale = up_proj_input_scale,
                .up_proj_scale = up_proj_scale,
                .up_proj_scale_2 = up_proj_scale_2,
                .gate_proj_input_scale = gate_proj_input_scale,
                .gate_proj_scale = gate_proj_scale,
                .gate_proj_scale_2 = gate_proj_scale_2,
                .down_proj_input_scale = down_proj_input_scale,
                .down_proj_scale = down_proj_scale,
                .down_proj_scale_2 = down_proj_scale_2,
            },
        };
    }

    pub fn forward(self: MatmulNVFP4, x: zml.Tensor) zml.Tensor {
        x.print("Input to MLP (before layernorm): ");
        const x_norm = self.pre_feedforward_layernorm.forward(x);
        const x_fp32 = x_norm.convert(.f32);

        const up_input_scale = self.quantization_tensors.up_proj_input_scale;
        const gate_input_scale = self.quantization_tensors.gate_proj_input_scale;
        const down_input_scale = self.quantization_tensors.down_proj_input_scale;
        const up_proj_scale_global = self.quantization_tensors.up_proj_scale_2.mul(up_input_scale);
        const gate_proj_scale_global = self.quantization_tensors.gate_proj_scale_2.mul(gate_input_scale);
        const down_proj_scale_global = self.quantization_tensors.down_proj_scale_2.mul(down_input_scale);

        const x_quantized_up = quantizeActivationNVFP4(x_fp32, up_input_scale);

        const up_out = forwardLinearDequantizeNVFP4(
            self.up_proj,
            x_quantized_up.block,
            x_quantized_up.scale,
            self.quantization_tensors.up_proj_scale,
            up_proj_scale_global,
        );

        const x_quantized_gate = quantizeActivationNVFP4(x_fp32, gate_input_scale);
        const gate_out = forwardLinearDequantizeNVFP4(
            self.gate_proj,
            x_quantized_gate.block,
            x_quantized_gate.scale,
            self.quantization_tensors.gate_proj_scale,
            gate_proj_scale_global,
        );

        const output = gate_out.gelu().mul(up_out).rename(.{ .dout = .d });

        const activated_quant = quantizeActivationNVFP4(output, down_input_scale);

        const out = forwardLinearDequantizeNVFP4(
            self.down_proj,
            activated_quant.block,
            activated_quant.scale,
            self.quantization_tensors.down_proj_scale,
            down_proj_scale_global,
        );

        const out_norm = self.post_feedforward_layernorm.forward(out.rename(.{ .dout = .d }));
        return out_norm;
    }

    pub fn load(
        self: *const MatmulNVFP4,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        replicated: zml.sharding.Sharding,
    ) !zml.Bufferized(MatmulNVFP4) {
        return zml.io.load(MatmulNVFP4, self, allocator, io, platform, store, .{
            .parallelism = 1,
            .shardings = &.{replicated},
            .dma_chunks = 2,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MatmulNVFP4)) void {
        RmsNorm.unloadBuffers(&self.pre_feedforward_layernorm);
        RmsNorm.unloadBuffers(&self.post_feedforward_layernorm);
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();

        self.quantization_tensors.up_proj_input_scale.deinit();
        self.quantization_tensors.up_proj_scale.deinit();
        self.quantization_tensors.up_proj_scale_2.deinit();
        self.quantization_tensors.gate_proj_input_scale.deinit();
        self.quantization_tensors.gate_proj_scale.deinit();
        self.quantization_tensors.gate_proj_scale_2.deinit();
        self.quantization_tensors.down_proj_input_scale.deinit();
        self.quantization_tensors.down_proj_scale.deinit();
        self.quantization_tensors.down_proj_scale_2.deinit();
    }
};

pub fn forwardLinearDequantizeNVFP4(
    linear: zml.nn.Linear,
    x: zml.Tensor,
    input_scale: zml.Tensor,
    weight_scale: zml.Tensor,
    weight_scale_2: zml.Tensor,
) zml.Tensor {
    const num_tokens = x.dim(.b);

    // Gemma stores two FP4 values per U8 weight element.
    var weight_f4 = linear.weight.bitCast(.f4e2m1);
    weight_f4 = weight_f4.merge(.{ .d = .{ .d_packed, .bitcast } });

    const flat_x = x.reshape(.{ .b = num_tokens, .d = x.dim(.d) });

    log.info("lhs block shape: {f}, rhs block shape: {f}, input scale shape: {f}, weight scale shape: {f}, weight scale 2 shape: {f}", .{ flat_x.shape(), weight_f4.shape(), input_scale.shape(), weight_scale.shape(), weight_scale_2.shape() });

    // Fused NVFP4 path: runtime custom call handles block-scale dequantization + matmul.
    var y = scale_dot_product(flat_x, input_scale, weight_f4, weight_scale, weight_scale_2)
        .rename(.{ .dout = .dout_w });

    // const group_size = @divExact(weight_f4.dim(.d), weight_scale.dim(.d_group));
    // const block_scale = weight_scale.convert(.f32)
    //     .reshape(.{ .dout = weight_scale.dim(.dout), .d_group = weight_scale.dim(.d_group), .d_block = 1 })
    //     .broad(zml.Shape.init(
    //         .{ .dout = weight_scale.dim(.dout), .d_group = weight_scale.dim(.d_group), .d_block = group_size },
    //         .f32,
    //     ))
    //     .merge(.{ .d = .{ .d_group, .d_block } });
    //
    // const x_dequant = x;
    // const flat_x_ref = x_dequant.reshape(.{ .token = num_tokens, .d = x.dim(.d) });
    //
    // const dequantized_weight = weight_f4.convert(.f32)
    //     .mul(block_scale)
    //     .mul(weight_scale_2)
    //     .rename(.{ .dout = .dout_w, .d = .d_w });
    //
    // var y = flat_x_ref.dotGeneral(
    //     dequantized_weight,
    //     &.{
    //         .{ @intCast(flat_x_ref.axis(.d)), @intCast(dequantized_weight.axis(.d_w)) },
    //     },
    //     &.{},
    // );

    if (linear.bias) |bias| {
        y = y.add(bias.withTags(.{.dout_w}).broad(y.shape()));
    }

    return y
        .rename(.{ .dout_w = .dout })
        .reshape(.{ .b = x.dim(.b), .dout = y.dim(.dout_w) });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, Args);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const mlp_prefix = "model.language_model.layers.0";
    const mlp = try MatmulNVFP4.init(store.view().withPrefix(mlp_prefix));

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated = try zml.sharding.replicatedSharding(platform);

    const input_dim = mlp.up_proj.weight.dim(.d_packed) * 2;
    const batch_dim: i64 = @intCast(args.batch);
    const input: zml.Tensor = .init(.{ .b = batch_dim, .d = input_dim }, .bf16);

    log.info("Compiling layer-0 NVFP4 MLP", .{});
    var exe = try platform.compile(allocator, io, mlp, .forward, .{input}, .{ .shardings = &.{replicated} });
    defer exe.deinit();

    log.info("Loading MLP buffers", .{});
    var mlp_buffers = try mlp.load(allocator, io, platform, &store, replicated);
    defer MatmulNVFP4.unloadBuffers(&mlp_buffers);

    var up_input_scale_slice = try mlp_buffers.quantization_tensors.up_proj_input_scale.toSliceAlloc(allocator, io);
    defer up_input_scale_slice.free(allocator);
    const up_input_scale_value = up_input_scale_slice.constItems(f32)[0];
    _ = up_input_scale_value; // autofix

    const input_len = batch_dim * input_dim;
    const input_data = try allocator.alloc(BFloat16, @intCast(input_len));
    defer allocator.free(input_data);

    var prng = std.Random.DefaultPrng.init(args.seed);
    const rng = prng.random();
    for (input_data) |*v| {
        const sample = (rng.float(f32) * 0.2) - 0.1;
        v.* = BFloat16.fromF32(sample);
    }

    var input_buffer = try zml.Buffer.fromBytes(io, platform, input.shape(), replicated, std.mem.sliceAsBytes(input_data));
    defer input_buffer.deinit();

    var call_args = try exe.args(allocator);
    defer call_args.deinit(allocator);
    call_args.set(.{ mlp_buffers, input_buffer });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.call(call_args, &results);

    var output = results.get(zml.Buffer);
    defer output.deinit();

    var out_slice = try output.toSliceAlloc(allocator, io);
    defer out_slice.free(allocator);

    const output_f32 = out_slice.constItems(f32);
    const preview_len = @min(output_f32.len, 20);
    var output_preview_f32: [20]f32 = undefined;
    for (0..preview_len) |i| {
        output_preview_f32[i] = output_f32[i];
    }

    log.info("NVFP4 MLP run complete", .{});
    log.info("Input shape: {f}", .{input.shape()});
    log.info("Output shape: {f}", .{output.shape()});
    log.info("Output preview (first {d} values, as f32): {any}", .{ preview_len, output_preview_f32[0..preview_len] });
}
