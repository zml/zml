const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");

const hz_type = .f32;


pub const AceVae_handler = struct {
    model: AceVae,
    params: Params,
    config: Config,
    decode_exe: zml.Exe,
    encode_exe: zml.Exe,
    model_buffers: zml.Bufferized(AceVae),
    shardings: main.Shardings,

    pub fn initFromFile(zml_handler: main.Zml_handler, target_duration: u32) !AceVae_handler {
        const model_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//diffusion_pytorch_model.safetensors";
        const config_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//config.json";

        //try main.printSafetensors(zml_handler.allocator, zml_handler.io, model_path);
        
        var registry: zml.safetensors.TensorRegistry = try .fromPath(zml_handler.allocator, zml_handler.io, model_path);
        defer registry.deinit();

        std.log.info("VAE config and safetensors", .{});
        const parsed_config = try parseConfig(zml_handler, config_path);
        defer parsed_config.deinit();
        const config = try parsed_config.value.dupe(zml_handler.allocator);
        std.log.info("VAE parsed", .{});

        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();

        std.log.info("VAE initialize model", .{});
        const model: AceVae = try .init(zml_handler.allocator, store.view(), config);
        std.log.info("VAE initialized", .{});

        const t_25hz = target_duration * 25;
        var t_48khz = t_25hz;
        for (config.downsampling_ratios) |ratio| {
            t_48khz *= ratio;
        }
        const params: Params = .{
            .latents = .init(.{ .c = config.decoder_input_channels, .t = t_25hz }, .f32),
            .audio = .init(.{ .c = config.audio_channels, .t = t_48khz }, .f32),
        };
        
        const shardings: main.Shardings = try .init(zml_handler.platform);

        std.log.info("VAE compile decoder", .{});
        const decode_exe = try compileDecodeModel(zml_handler, model, params, shardings);
        std.log.info("VAE compile encoder", .{});
        const encode_exe = try compileEncodeModel(zml_handler, model, params, shardings);
        std.log.info("VAE compiled models", .{});
        
        std.log.info("VAE load buffers", .{});
        const model_buffers = try model.load(zml_handler.arena.allocator(), zml_handler.io, zml_handler.platform, &store, &shardings.all());
        std.log.info("VAE loaded buffers", .{});
        
        return .{
            .model = model,
            .params = params,
            .config = config,
            .decode_exe = decode_exe,
            .encode_exe = encode_exe,
            .model_buffers = model_buffers,
            .shardings = shardings,
        };
    }

    pub fn unloadBuffers(self: *AceVae_handler) void {
        AceVae.unloadBuffers(&self.model_buffers);
    }

    pub fn deinit(self: *AceVae_handler, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.decode_exe.deinit();
        self.encode_exe.deinit();
    }
};


pub const Params = struct {
    latents: zml.Tensor,
    audio: zml.Tensor,
};

pub const Config = struct {
    audio_channels: u32,
    channel_multiples: []const u32,
    decoder_channels: u32,
    decoder_input_channels: u32,
    downsampling_ratios: []const u32,
    encoder_hidden_size: u32,
    sampling_rate: u32,

    pub fn dupe(self: Config, allocator: std.mem.Allocator) !Config {
        return .{
            .audio_channels = self.audio_channels,
            .channel_multiples = try allocator.dupe(u32, self.channel_multiples),
            .decoder_channels = self.decoder_channels,
            .decoder_input_channels = self.decoder_input_channels,
            .downsampling_ratios = try allocator.dupe(u32, self.downsampling_ratios),
            .encoder_hidden_size = self.encoder_hidden_size,
            .sampling_rate = self.sampling_rate,
        };
    }

    pub fn deinit(self: Config, allocator: std.mem.Allocator) void {
        allocator.free(self.channel_multiples);
        allocator.free(self.downsampling_ratios);
    }
};

pub fn parseConfig(zml_handler: main.Zml_handler, path: []const u8) !std.json.Parsed(Config) {
    const parsed_config = blk: {
        const config_json_file = try std.Io.Dir.openFileAbsolute(zml_handler.io, path, .{});
        defer config_json_file.close(zml_handler.io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(zml_handler.io, &config_json_buffer);
        var reader = std.json.Reader.init(zml_handler.allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(Config, zml_handler.allocator, &reader, .{
            .ignore_unknown_fields = true,
        });
    };
    errdefer parsed_config.deinit();
    return parsed_config;
}

pub fn compileDecodeModel(zml_handler: main.Zml_handler, model: AceVae, parameters: Params, shardings: main.Shardings) !zml.Exe {
    const opts: zml.module.CompilationOptions = .{
        .shardings = &shardings.all(),
    };
    return zml_handler.platform.compile(
        zml_handler.allocator,
        zml_handler.io,
        model,
        .decode,
        .{ parameters.latents },
        opts,
    );
}

pub fn compileEncodeModel(zml_handler: main.Zml_handler, model: AceVae, parameters: Params, shardings: main.Shardings) !zml.Exe {
    const opts: zml.module.CompilationOptions = .{
        .shardings = &shardings.all(),
    };
    return zml_handler.platform.compile(
        zml_handler.allocator,
        zml_handler.io,
        model,
        .encode,
        .{ parameters.audio },
        opts,
    );
}


pub const AceVae = struct {
    encoder: OobleckEncoder,
    decoder: OobleckDecoder,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !AceVae {
        return .{
            .encoder = try .init(allocator, store.withPrefix("encoder"), config),
            .decoder = try .init(allocator, store.withPrefix("decoder"), config),
        };
    }

    pub fn load(self: *const AceVae, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(AceVae) {
        return zml.io.load(AceVae, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 128 * 1024 * 1024,
        });
    }

    pub fn deinit(self: *const AceVae, allocator: std.mem.Allocator) void {
        self.encoder.deinit(allocator);
        self.decoder.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AceVae)) void {
        OobleckEncoder.unloadBuffers(&self.encoder);
        OobleckDecoder.unloadBuffers(&self.decoder);
    }

    pub fn encode(self: AceVae, audio: zml.Tensor) zml.Tensor {
        return self.encoder.forward(audio);
    }

    pub fn decode(self: AceVae, latents: zml.Tensor) zml.Tensor {
        return self.decoder.forward(latents);
    }
};


pub const OobleckEncoder = struct {
    conv1: Conv1D,
    block: []EncoderBlock,
    snake1: Snake1D,
    conv2: Conv1D,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !OobleckEncoder {
        const num_stages = config.channel_multiples.len;
        const blocks = try allocator.alloc(EncoderBlock, num_stages);
        errdefer allocator.free(blocks);

        for (blocks, 0..) |*encoder_block, stride_index| {
            const channel_multiples_prev = if (stride_index == 0) 1 else config.channel_multiples[stride_index - 1];
            encoder_block.* = .init(
                store.withPrefix("block").withLayer(stride_index),
                config.encoder_hidden_size * channel_multiples_prev,
                config.encoder_hidden_size * config.channel_multiples[stride_index],
                config.downsampling_ratios[stride_index],
            );
        }

        return .{
            .conv1 = .init(
                store.withPrefix("conv1"),
                config.audio_channels,
                config.encoder_hidden_size,
                7,
                1,
                3,
                1,
            ),
            .block = blocks,
            .snake1 = .init(store.withPrefix("snake1")),
            .conv2 = .init(
                store.withPrefix("conv2"),
                config.encoder_hidden_size * config.channel_multiples[num_stages - 1],
                config.encoder_hidden_size,
                3,
                1,
                1,
                1,
            ),
        };
    }

    pub fn deinit(self: *const OobleckEncoder, allocator: std.mem.Allocator) void {
        allocator.free(self.block);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(OobleckEncoder)) void {
        Conv1D.unloadBuffers(&self.conv1);
        for (self.block) |*encoder_block| {
            EncoderBlock.unloadBuffers(encoder_block);
        }
        Snake1D.unloadBuffers(&self.snake1);
        Conv1D.unloadBuffers(&self.conv2);
    }

    pub fn forward(self: OobleckEncoder, hidden_state0: zml.Tensor) zml.Tensor {
        var hidden_state = self.conv1.forward(hidden_state0);
        for (self.block) |encoder_block| {
            hidden_state = encoder_block.forward(hidden_state);
        }
        hidden_state = self.snake1.forward(hidden_state);
        hidden_state = self.conv2.forward(hidden_state);
        return hidden_state;
    }
};

pub const EncoderBlock = struct {
    res_unit1: ResidualUnit,
    res_unit2: ResidualUnit,
    res_unit3: ResidualUnit,
    snake1: Snake1D,
    conv1: Conv1D,

    pub fn init(store: zml.io.TensorStore.View, input_dim: u32, output_dim: u32, stride: u32) EncoderBlock {
        return .{
            .res_unit1 = .init(store.withPrefix("res_unit1"), input_dim, 1),
            .res_unit2 = .init(store.withPrefix("res_unit2"), input_dim, 3),
            .res_unit3 = .init(store.withPrefix("res_unit3"), input_dim, 9),
            .snake1 = .init(store.withPrefix("snake1")),
            .conv1 = .init(
                store.withPrefix("conv1"),
                input_dim,
                output_dim,
                2 * stride,
                stride,
                @divFloor(stride + 1, 2),
                1,
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(EncoderBlock)) void {
        ResidualUnit.unloadBuffers(&self.res_unit1);
        ResidualUnit.unloadBuffers(&self.res_unit2);
        ResidualUnit.unloadBuffers(&self.res_unit3);
        Snake1D.unloadBuffers(&self.snake1);
        Conv1D.unloadBuffers(&self.conv1);
    }

    pub fn forward(self: EncoderBlock, hidden_state0: zml.Tensor) zml.Tensor {
        var hidden_state = self.res_unit1.forward(hidden_state0);
        hidden_state = self.res_unit2.forward(hidden_state);
        hidden_state = self.snake1.forward(self.res_unit3.forward(hidden_state));
        hidden_state = self.conv1.forward(hidden_state);
        return hidden_state;
    }
};


pub const OobleckDecoder = struct {
    conv1: Conv1D,
    block: []DecoderBlock,
    snake1: Snake1D,
    conv2: Conv1D,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !OobleckDecoder {
        const num_stages = config.channel_multiples.len;
        const block = try allocator.alloc(DecoderBlock, num_stages);
        errdefer allocator.free(block);

        for (block, 0..) |*decoder_block, stride_index| {
            const channel_multiples_prev = if (stride_index == num_stages - 1) 1 else config.channel_multiples[num_stages - stride_index - 2];
            const input_mult = config.channel_multiples[num_stages - stride_index - 1];
            const output_mult = channel_multiples_prev;
            decoder_block.* = .init(
                store.withPrefix("block").withLayer(stride_index),
                config.decoder_channels * input_mult,
                config.decoder_channels * output_mult,
                config.downsampling_ratios[num_stages - stride_index - 1],
            );
        }

        return .{
            .conv1 = .init(
                store.withPrefix("conv1"),
                config.decoder_input_channels,
                config.decoder_channels * config.channel_multiples[num_stages - 1],
                7,
                1,
                3,
                1,
            ),
            .block = block,
            .snake1 = .init(store.withPrefix("snake1")),
            .conv2 = .init(
                store.withPrefix("conv2"),
                config.decoder_channels,
                config.audio_channels,
                7,
                1,
                3,
                1,
            ),
        };
    }

    pub fn deinit(self: *const OobleckDecoder, allocator: std.mem.Allocator) void {
        allocator.free(self.block);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(OobleckDecoder)) void {
        Conv1D.unloadBuffers(&self.conv1);
        for (self.block) |*decoder_block| {
            DecoderBlock.unloadBuffers(decoder_block);
        }
        Snake1D.unloadBuffers(&self.snake1);
        Conv1D.unloadBuffers(&self.conv2);
    }

    pub fn forward(self: OobleckDecoder, hidden_state0: zml.Tensor) zml.Tensor {
        var hidden_state = self.conv1.forward(hidden_state0);
        for (self.block) |decoder_block| {
            hidden_state = decoder_block.forward(hidden_state);
        }
        hidden_state = self.snake1.forward(hidden_state);
        hidden_state = self.conv2.forward(hidden_state);
        return hidden_state;
    }
};

const OobleckDecoderWrapper = struct {
    decoder: OobleckDecoder,
    pub fn forward(wrapper: OobleckDecoderWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .a, .t }).squeeze(.b);
        const output = wrapper.decoder.forward(tagged_input).withTags(.{ .t, .a });
        return output.insertAxes(0, .{ .b });
    }
};

pub const DecoderBlock = struct {
    snake1: Snake1D,
    conv_t1: ConvTranspose1D,
    res_unit1: ResidualUnit,
    res_unit2: ResidualUnit,
    res_unit3: ResidualUnit,

    pub fn init(store: zml.io.TensorStore.View, input_dim: u32, output_dim: u32, stride: u32) DecoderBlock {
        return .{
            .snake1 = .init(store.withPrefix("snake1")),
            .conv_t1 = .init(
                store.withPrefix("conv_t1"),
                input_dim,
                output_dim,
                2 * stride,
                stride,
                @divFloor(stride + 1, 2),
            ),
            .res_unit1 = .init(store.withPrefix("res_unit1"), output_dim, 1),
            .res_unit2 = .init(store.withPrefix("res_unit2"), output_dim, 3),
            .res_unit3 = .init(store.withPrefix("res_unit3"), output_dim, 9),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DecoderBlock)) void {
        Snake1D.unloadBuffers(&self.snake1);
        ConvTranspose1D.unloadBuffers(&self.conv_t1);
        ResidualUnit.unloadBuffers(&self.res_unit1);
        ResidualUnit.unloadBuffers(&self.res_unit2);
        ResidualUnit.unloadBuffers(&self.res_unit3);
    }

    pub fn forward(self: DecoderBlock, hidden_state0: zml.Tensor) zml.Tensor {
        var hidden_state = self.snake1.forward(hidden_state0);
        hidden_state = self.conv_t1.forward(hidden_state);
        hidden_state = self.res_unit1.forward(hidden_state);
        hidden_state = self.res_unit2.forward(hidden_state);
        hidden_state = self.res_unit3.forward(hidden_state);
        return hidden_state;
    }
};


pub const ResidualUnit = struct {
    snake1: Snake1D,
    conv1: Conv1D,
    snake2: Snake1D,
    conv2: Conv1D,

    pub fn init(store: zml.io.TensorStore.View, dimension: u32, dilation: u32) ResidualUnit {
        const pad = 3 * dilation;
        return .{
            .snake1 = .init(store.withPrefix("snake1")),
            .conv1 = .init(store.withPrefix("conv1"), dimension, dimension, 7, 1, pad, dilation),
            .snake2 = .init(store.withPrefix("snake2")),
            .conv2 = .init(store.withPrefix("conv2"), dimension, dimension, 1, 1, 0, 1),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ResidualUnit)) void {
        Snake1D.unloadBuffers(&self.snake1);
        Conv1D.unloadBuffers(&self.conv1);
        Snake1D.unloadBuffers(&self.snake2);
        Conv1D.unloadBuffers(&self.conv2);
    }

    pub fn forward(self: ResidualUnit, hidden_state0: zml.Tensor) zml.Tensor {
        var output_tensor = hidden_state0;
        output_tensor = self.conv1.forward(self.snake1.forward(output_tensor));
        output_tensor = self.conv2.forward(self.snake2.forward(output_tensor));

        const hidden_len = hidden_state0.shape().dim(.t);
        const output_len = output_tensor.shape().dim(.t);
        const padding = @divFloor(hidden_len - output_len, 2);

        var hidden_state = hidden_state0;
        if (padding > 0) {
            const b_slice: zml.Tensor.Slice = .{ .start = 0, .end = hidden_state.dim(.b), };
            const c_slice: zml.Tensor.Slice = .{ .start = 0, .end = hidden_state.dim(.c), };
            const t_slice: zml.Tensor.Slice = .{ .start = padding, .end = hidden_len - padding, };
            hidden_state = hidden_state.slice(&.{ b_slice, c_slice, t_slice });
        }

        return hidden_state.add(output_tensor);
    }
};

pub const Snake1D = struct {
    alpha: zml.Tensor,
    beta: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Snake1D {
        return .{
            .alpha = store.createTensor("alpha", .{ .b, .c, .t }, null),
            .beta = store.createTensor("beta", .{ .b, .c, .t }, null),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Snake1D)) void {
        self.alpha.deinit();
        self.beta.deinit();
    }

    pub fn forward(self: Snake1D, x: zml.Tensor) zml.Tensor {
        const a = self.alpha.squeeze(.b).squeeze(.t).convert(hz_type);
        const b = self.beta.squeeze(.b).squeeze(.t).convert(hz_type);
        const alpha = a.exp().broad(x.shape());
        const beta = b.exp().addConstant(1e-9).broad(x.shape());
        const sinusoid = alpha.mul(x).sin().powByConst(2);
        return x.add(sinusoid.div(beta));
    }
};

pub const Conv1D = struct {
    weight_g: zml.Tensor,
    weight_v: zml.Tensor,
    bias: ?zml.Tensor,
    in_channels: u32,
    out_channels: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    dilation: u32,

    pub fn init(
        store: zml.io.TensorStore.View,
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: u32,
        padding: u32,
        dilation: u32,
    ) Conv1D {
        return .{
            .weight_g = store.createTensor("weight_g", .{ .c_out, .singleton, .singleton }, null),
            .weight_v = store.createTensor("weight_v", .{ .c_out, .c_in, .w }, null),
            .bias = store.maybeCreateTensor("bias", .{ .c }, null),
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
            .dilation = dilation,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Conv1D)) void {
        self.weight_g.deinit();
        self.weight_v.deinit();
        if (self.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Conv1D, x0: zml.Tensor) zml.Tensor {
        var x = x0.insertAxes(0, . { .b }).withTags(.{ .b, .c, .t });

        const weight_v = self.weight_v.convert(hz_type);
        const weight_g = self.weight_g.convert(hz_type);
        const v_norm = weight_v.powByConst(2).sum(.c_in).sum(.w).sqrt();
        const weight = weight_v.mul(weight_g.div(v_norm.broad(weight_g.shape())).broad(weight_v.shape()));
        
        var y = x.conv1d(
            weight,
            .{
                .window_strides = self.stride,
                .padding = &.{ self.padding, self.padding },
                .rhs_dilation = self.dilation,
            },
        );
        if (self.bias) |bias| {
            y = y.add(bias.convert(hz_type).broad(y.shape()));
        }
        return y.squeeze(.b);
    }
};

pub const ConvTranspose1D = struct {
    weight_g: zml.Tensor,
    weight_v: zml.Tensor,
    bias: ?zml.Tensor,
    in_channels: u32,
    out_channels: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,

    pub fn init(
        store: zml.io.TensorStore.View,
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: u32,
        padding: u32,
    ) ConvTranspose1D {
        return .{
            .weight_g = store.createTensor("weight_g", .{ .c_in, .singleton, .singleton }, null),
            .weight_v = store.createTensor("weight_v", .{ .c_in, .c_out, .w }, null),
            .bias = store.maybeCreateTensor("bias", .{ .c }, null),
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel_size = kernel_size,
            .stride = stride,
            .padding = padding,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ConvTranspose1D)) void {
        self.weight_g.deinit();
        self.weight_v.deinit();
        if (self.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: ConvTranspose1D, x0: zml.Tensor) zml.Tensor {
        var x = x0.insertAxes(0, . { .b }).withTags(.{ .b, .c, .t });
    
        const weight_v = self.weight_v.convert(hz_type);
        const weight_g = self.weight_g.convert(hz_type);
        const v_norm = weight_v.powByConst(2).sum(.c_out).sum(.w).sqrt();
        const weight = weight_v.mul(weight_g.div(v_norm.broad(weight_g.shape())).broad(weight_v.shape()));
            
        const flipped_weight = weight.reverse(.{ .w });
        const edge_pad = self.kernel_size - 1 - self.padding;
    
        var y = x.conv1d(
            flipped_weight,
            .{
                .padding = &.{ edge_pad, edge_pad },
                .lhs_dilation = self.stride,
                .kernel_output_feature_dimension = 1,
                .kernel_input_feature_dimension = 0,
            },
        );
    
        if (self.bias) |bias| {
            y = y.add(bias.convert(hz_type).broad(y.shape()));
        }
    
        return y.squeeze(.b);
    }

};


pub fn print(x: zml.Tensor, n: u8) void {
    const name = &.{ n };
    const sh = x._shape;
    const nb_dims = sh.rank();
    switch (nb_dims) {
        0 => std.log.info("{s} is scalar", .{ name }),
        1 => std.log.info("{s} is dim 1 : {s} = {d}", .{ name, sh.tag(0), sh.dim(0) }),
        2 => std.log.info("{s} is dim 2 : {s} x {s} = {d} x {d}", .{ name, sh.tag(0), sh.tag(1), sh.dim(0), sh.dim(1) }),
        3 => std.log.info("{s} is dim 3 : {s} x {s} x {s} = {d} x {d} x {d}", .{ name, sh.tag(0), sh.tag(1), sh.tag(2), sh.dim(0), sh.dim(1), sh.dim(2) }),
        4 => std.log.info("{s} is dim 4 : {s} x {s} x {s} x {s} = {d} x {d} x {d} x {d}", .{ name, sh.tag(0), sh.tag(1), sh.tag(2), sh.tag(3), sh.dim(0), sh.dim(1), sh.dim(2), sh.dim(3) }),
        else => std.log.info("{s} is rank >= 5", .{ name }),
    }
}

pub fn testModel(zml_handler: main.Zml_handler, acevae: AceVae_handler) !void {
    const activations_path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//activations.safetensors";
    
    try main.printSafetensors(zml_handler.allocator, zml_handler.io, activations_path);

    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(zml_handler.allocator, zml_handler.io, activations_path);
    defer activations_registry.deinit();

    var activations_store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &activations_registry);
    defer activations_store.deinit();

    std.log.info("Test activations : decoder layer", .{});
    const wrapper_decoder: OobleckDecoderWrapper = .{
        .decoder = acevae.model.decoder,
    };
    const wrapper_buffers_decoder: zml.Bufferized(OobleckDecoderWrapper) = .{
        .decoder = acevae.model_buffers.decoder,
    };
    const layer_embed = "model";
    try zml.testing.testLayer(zml_handler.allocator, zml_handler.io, zml_handler.platform, wrapper_decoder, .forward, activations_store.view(), layer_embed, wrapper_buffers_decoder, &acevae.shardings.all(), .{});
}