const std = @import("std");

const zml = @import("zml");
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

pub fn scale_dot_product(lhs_block: zml.Tensor, lhs_scale: zml.Tensor, rhs_block: zml.Tensor, rhs_scale: zml.Tensor) zml.Tensor {
    const output_shape = zml.Shape.init(.{ .token = lhs_block.dim(.token), .dout = rhs_block.dim(.dout) }, .bf16);
    return zml.ops.customCall(
        "__op$block_scaled_dot",
        .{ lhs_block, rhs_block, lhs_scale, rhs_scale },
        output_shape,
        .{
            .dequantize_type = "BF16",
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

const GemmaLayer0Mlp = struct {
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

    pub fn init(store: zml.io.TensorStore.View) !GemmaLayer0Mlp {
        const up_proj_store = store.withPrefix("up_proj");
        const gate_proj_store = store.withPrefix("gate_proj");
        const down_proj_store = store.withPrefix("down_proj");

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

    pub fn load(
        self: *const GemmaLayer0Mlp,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        replicated: zml.sharding.Sharding,
    ) !zml.Bufferized(GemmaLayer0Mlp) {
        return zml.io.load(GemmaLayer0Mlp, self, allocator, io, platform, store, .{
            .parallelism = 1,
            .shardings = &.{replicated},
            .dma_chunks = 2,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GemmaLayer0Mlp)) void {
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

    pub fn forward(self: GemmaLayer0Mlp, x: zml.Tensor) zml.Tensor {
        const x_f32 = x.convert(.f32);
        const proj = forwardLinearDequantizeNVFP4(
            self.up_proj,
            x_f32,
            self.quantization_tensors.up_proj_input_scale,
            self.quantization_tensors.up_proj_scale,
            self.quantization_tensors.up_proj_scale_2,
        );

        var output = forwardLinearDequantizeNVFP4(
            self.gate_proj,
            x_f32,
            self.quantization_tensors.gate_proj_input_scale,
            self.quantization_tensors.gate_proj_scale,
            self.quantization_tensors.gate_proj_scale_2,
        );
        output = output.gelu().mul(proj).rename(.{ .dout = .d });

        return forwardLinearDequantizeNVFP4(
            self.down_proj,
            output,
            self.quantization_tensors.down_proj_input_scale,
            self.quantization_tensors.down_proj_scale,
            self.quantization_tensors.down_proj_scale_2,
        );
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

    const flat_x = x.reshape(.{ .token = num_tokens, .d = x.dim(.d) });
    const fused_weight_scale = weight_scale.convert(.f32).mul(weight_scale_2.convert(.f32));

    // Fused NVFP4 path: runtime custom call handles block-scale dequantization + matmul.
    var y = scale_dot_product(flat_x, input_scale.convert(.f32), weight_f4, fused_weight_scale)
        .convert(.f32)
        .rename(.{ .dout = .dout_w });

    // Old reference path kept for debugging/comparison.
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

    const mlp_prefix = "model.language_model.layers.0.mlp";
    const mlp = try GemmaLayer0Mlp.init(store.view().withPrefix(mlp_prefix));

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated = try zml.sharding.replicatedSharding(platform);

    const input_dim = mlp.up_proj.weight.dim(.d_packed) * 2;
    const batch_dim: i64 = @intCast(args.batch);
    const input: zml.Tensor = .init(.{ .b = batch_dim, .d = input_dim }, .f32);

    log.info("Compiling layer-0 NVFP4 MLP", .{});
    var exe = try platform.compile(allocator, io, mlp, .forward, .{input}, .{ .shardings = &.{replicated} });
    defer exe.deinit();

    log.info("Loading MLP buffers", .{});
    var mlp_buffers = try mlp.load(allocator, io, platform, &store, replicated);
    defer GemmaLayer0Mlp.unloadBuffers(&mlp_buffers);

    const input_len = batch_dim * input_dim;
    const input_data = try allocator.alloc(f32, @intCast(input_len));
    defer allocator.free(input_data);

    var prng = std.Random.DefaultPrng.init(args.seed);
    const rng = prng.random();
    for (input_data) |*v| {
        v.* = (rng.float(f32) * 2.0) - 1.0;
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
    const preview_len = @min(output_f32.len, 8);

    log.info("NVFP4 MLP run complete", .{});
    log.info("Input shape: {f}", .{input.shape()});
    log.info("Output shape: {f}", .{output.shape()});
    log.info("Output preview (first {d} values): {any}", .{ preview_len, output_f32[0..preview_len] });
}
