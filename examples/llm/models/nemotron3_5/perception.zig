const std = @import("std");

const zml = @import("zml");

const frontend = @import("frontend.zig");

pub const Config = struct {
    preprocessor: frontend.Config,
    encoder: EncoderConfig,
    modality_adapter: struct {
        _target_: []const u8,
        d_model: i64,
    },
    output_dim: i64,

    pub fn validate(self: Config) !void {
        try self.encoder.validate();

        if (self.preprocessor.features != self.encoder.feat_in or self.output_dim <= 0)
            return error.InvalidPerceptionConfig;

        if (!std.mem.eql(u8, self.modality_adapter._target_, "nemo.collections.speechlm2.modules.perception.IdentityConnector") or
            self.modality_adapter.d_model != self.encoder.d_model)
            return error.UnsupportedModalityAdapter;
    }
};

pub const EncoderConfig = struct {
    feat_in: i64,
    d_model: i64,
    n_heads: i64,
    n_layers: usize,
    qkv_bias: bool = false,
    qk_norm: bool = false,
    ff_expansion: f32 = 4,
    pre_block_norm: bool = true,
    subsampling_factor: i64,
    attn_mode: enum { full, causal } = .full,
    self_attention_model: enum { rope, rel_pos, abs_pos, no_pos } = .rope,
    rope_base: f32 = 10_000,
    rotary_fraction: f32 = 1,
    final_norm: bool = true,
    feat_out: i64 = -1,
    subsampling: enum { feature_stacking, stacking, stacking_norm } = .feature_stacking,
    xscaling: bool = false,
    causal_tail_len: i64 = 0,

    pub fn validate(self: EncoderConfig) !void {
        if (self.feat_in <= 0 or self.d_model <= 0 or self.n_heads <= 0 or self.n_layers == 0 or self.subsampling_factor <= 0)
            return error.InvalidPerceptionConfig;

        if (@rem(self.d_model, self.n_heads) != 0 or self.feat_in > @divTrunc(std.math.maxInt(i64), self.subsampling_factor))
            return error.InvalidPerceptionConfig;

        if (!std.math.isFinite(self.ff_expansion) or self.ff_expansion <= 0 or
            !std.math.isFinite(self.rope_base) or self.rope_base <= 0 or
            !std.math.isFinite(self.rotary_fraction) or self.rotary_fraction <= 0 or self.rotary_fraction > 1)
            return error.InvalidPerceptionConfig;

        if (self.attn_mode != .full or self.self_attention_model != .rope or self.subsampling != .feature_stacking or
            self.xscaling or self.causal_tail_len != 0 or (self.feat_out > 0 and self.feat_out != self.d_model))
            return error.UnsupportedPerceptionConfig;

        const head_dim = @divExact(self.d_model, self.n_heads);
        const rotary_dim: i64 = @intFromFloat(@floor(@as(f64, @floatFromInt(head_dim)) * self.rotary_fraction));
        if (rotary_dim < 2 or @rem(rotary_dim, 2) != 0)
            return error.InvalidPerceptionConfig;
    }
};

pub const ParallelConfig = struct {
    asr_encoder_type: enum { transformer, fastconformer } = .transformer,
    asr_encoder_cfg: EncoderConfig,
    diarization_model_cfg: ?DiarizationConfig = null,
    asr_normalize_type: ?[]const u8 = "per_feature",
    diar_normalize_type: ?[]const u8 = null,
    speaker_feature_config_version: u32 = 1,
    speaker_feature_mode: enum { thresholded, continuous } = .thresholded,
    speaker_activity_threshold: ?f32 = 0.5,
    spk_kernel_scale: f32 = 1,
};

pub const DiarizationConfig = struct {
    encoder: EncoderConfig,
    model_defaults: struct {
        fc_d_model: i64 = 512,
        tf_d_model: i64 = 192,
    } = .{},
    sortformer_modules: struct {
        num_spks: i64 = 8,
        fc_d_model: i64 = 512,
        tf_d_model: i64 = 192,
    } = .{},
    preprocessor: struct {
        normalize: ?[]const u8 = "NA",
    } = .{},
    max_num_of_spks: i64 = 8,
    high_resolution: bool = true,
    transformer_encoder: ?struct {} = null,

    fn validate(self: DiarizationConfig, asr: EncoderConfig) !void {
        try self.encoder.validate();

        if (@mod(asr.d_model, 2) != 0 or self.encoder.feat_in != asr.feat_in or self.encoder.subsampling_factor != asr.subsampling_factor or
            self.sortformer_modules.fc_d_model != self.encoder.d_model or
            self.model_defaults.fc_d_model != self.encoder.d_model or
            self.sortformer_modules.tf_d_model != self.model_defaults.tf_d_model or
            self.sortformer_modules.tf_d_model <= 0 or self.sortformer_modules.num_spks <= 0 or
            self.max_num_of_spks != self.sortformer_modules.num_spks)
            return error.InvalidPerceptionConfig;

        if (self.transformer_encoder != null)
            return error.UnsupportedPerceptionConfig;
    }
};

const Normalization = enum {
    none,
    per_feature,

    fn parse(value: ?[]const u8) !Normalization {
        const name = value orelse return .none;

        if (std.mem.eql(u8, name, "NA"))
            return .none;

        if (std.mem.eql(u8, name, "per_feature"))
            return .per_feature;

        return error.UnsupportedPerceptionNormalization;
    }

    fn forward(self: Normalization, input: Input) Input {
        if (self == .none)
            return input;

        const x = input.features;
        const valid = frameMask(x.dim(.b), x.dim(.s), input.lengths).broad(x.shape());
        const count = input.lengths.convert(.f32);
        const sum = valid.select(x, .zeroes(x.shape())).convert(.f32).sum(.s);

        const mean = sum.div(count.maximum(.scalar(1, .f32)).broad(sum.shape())).convert(x.dtype());
        const deviations = valid.select(x.sub(mean), .zeroes(x.shape()));
        const squared_sum = deviations.mul(deviations).convert(.f32).sum(.s).convert(x.dtype());

        const variance = squared_sum.convert(.f32)
            .div(count.addConstant(-1).maximum(.scalar(1, .f32)).broad(sum.shape()));
        const normalized = deviations.convert(.f32).div(variance.sqrt().addConstant(1e-5)).convert(x.dtype());

        return .{
            .features = valid.select(normalized, .zeroes(x.shape())),
            .lengths = input.lengths,
        };
    }
};

pub const Input = struct {
    features: zml.Tensor,
    lengths: zml.Tensor,

    pub const Buffers = zml.Bufferized(Input);

    pub fn initBuffers(io: std.Io, platform: *const zml.Platform, features: frontend.Features, config: EncoderConfig) !Buffers {
        try config.validate();

        if (features.frame_count == 0 or features.valid_frame_count == 0 or features.valid_frame_count > features.frame_count or
            features.frame_count > std.math.maxInt(i64) or features.valid_frame_count > std.math.maxInt(u32) or
            features.bin_count != config.feat_in)
            return error.InvalidPerceptionInput;

        const count = std.math.mul(usize, features.frame_count, features.bin_count) catch return error.InvalidPerceptionInput;
        if (features.data.len != count)
            return error.InvalidPerceptionInput;

        for (features.data) |value| {
            if (!std.math.isFinite(value))
                return error.InvalidPerceptionInput;
        }

        var feature_buffer = try zml.Buffer.fromBytes(io, platform, .init(.{ .b = 1, .s = features.frame_count, .mel = features.bin_count }, .f32), std.mem.sliceAsBytes(features.data));
        errdefer feature_buffer.deinit();

        const lengths = [_]u32{@intCast(features.valid_frame_count)};
        return .{
            .features = feature_buffer,
            .lengths = try zml.Buffer.fromBytes(io, platform, .init(.{ .b = 1 }, .u32), std.mem.asBytes(&lengths)),
        };
    }

    pub fn deinitBuffers(buffers: *Buffers) void {
        zml.Buffer.deinitAll(Input, buffers);
    }
};

pub const Output = struct {
    hidden: zml.Tensor,
    lengths: zml.Tensor,
};

pub const FeatureStacking = struct {
    proj: zml.nn.Linear,
    factor: i64,
    feat_in: i64,

    pub fn init(store: zml.io.TensorStore.View, config: EncoderConfig) !FeatureStacking {
        try config.validate();

        const projection = store.withPrefix("proj");
        try validateLinear(projection, config.feat_in * config.subsampling_factor, config.d_model, false);
        if (projection.getShape("weight").?.dtype() == .f8e4m3fn)
            return error.UnsupportedPerceptionQuantization;

        return .{
            .proj = .init(projection.createReplicatedTensor("weight", .{ .dout, .stacked }), null, .stacked),
            .factor = config.subsampling_factor,
            .feat_in = config.feat_in,
        };
    }

    pub fn forward(self: FeatureStacking, input: Input) Output {
        zml.stdx.debug.assert(input.features.rank() == 3 and input.features.shape().hasTags(.{ .b, .s, .mel }) and input.features.dim(.mel) == self.feat_in, "Perception features must have shape {{b, s, mel}}", .{});
        zml.stdx.debug.assert(input.lengths.rank() == 1 and input.lengths.shape().hasTags(.{.b}) and input.lengths.dim(.b) == input.features.dim(.b) and input.lengths.dtype() == .u32, "Perception lengths must have shape {{b}} and dtype u32", .{});

        const features = input.features.transpose(.{ .b, .s, .mel });
        const frames = features.dim(.s);
        zml.stdx.debug.assert(frames > 0, "Perception requires at least one feature frame", .{});

        const padding = @mod(-frames, self.factor);
        const padded = features.pad(0, .{ .s = zml.Tensor.Pad{ .high = padding } });
        const stacked = padded.splitAxis(.s, .{ .s = @divExact(frames + padding, self.factor), .frame = self.factor })
            .merge(.{ .stacked = .{ .frame, .mel } });

        const lengths = input.lengths.convert(.u64).addConstant(self.factor - 1)
            .div(.scalar(self.factor, .u64)).convert(.u32);

        return .{
            .hidden = self.proj.forward(stacked.convert(self.proj.weight.dtype()), self.proj.weight.dtype()).rename(.{ .dout = .d }),
            .lengths = lengths,
        };
    }
};

pub const EncoderAttention = struct {
    w_qkv: Linear,
    out_proj: Linear,
    q_norm: ?zml.nn.LayerNorm,
    k_norm: ?zml.nn.LayerNorm,
    config: EncoderConfig,
    rotary_dim: i64,

    /// `store` must point at a block's `attn` prefix.
    pub fn init(store: zml.io.TensorStore.View, config: EncoderConfig) !EncoderAttention {
        try config.validate();

        if (config.d_model > @divTrunc(std.math.maxInt(i64), 3))
            return error.InvalidPerceptionConfig;

        const head_dim = @divExact(config.d_model, config.n_heads);
        const rotary_dim: i64 = @intFromFloat(@floor(@as(f64, @floatFromInt(head_dim)) * config.rotary_fraction));
        if (rotary_dim < 2 or @rem(rotary_dim, 2) != 0)
            return error.InvalidPerceptionConfig;

        return .{
            .w_qkv = try initLinear(store.withPrefix("w_qkv"), config.d_model, 3 * config.d_model, config.qkv_bias),
            .out_proj = try initLinear(store.withPrefix("out_proj"), config.d_model, config.d_model, true),
            .q_norm = if (config.qk_norm) try initLayerNorm(store.withPrefix("q_norm"), head_dim) else null,
            .k_norm = if (config.qk_norm) try initLayerNorm(store.withPrefix("k_norm"), head_dim) else null,
            .config = config,
            .rotary_dim = rotary_dim,
        };
    }

    pub fn forward(self: EncoderAttention, hidden: zml.Tensor, lengths: zml.Tensor) zml.Tensor {
        zml.stdx.debug.assert(hidden.rank() == 3 and hidden.shape().hasTags(.{ .b, .s, .d }) and hidden.dim(.d) == self.config.d_model, "Perception attention expects {{b, s, d_model}} input", .{});
        zml.stdx.debug.assert(lengths.rank() == 1 and lengths.shape().hasTags(.{.b}) and lengths.dim(.b) == hidden.dim(.b) and lengths.dtype() == .u32, "Perception attention lengths must have shape {{b}} and dtype u32", .{});
        zml.stdx.debug.assert(hidden.dim(.s) > 0 and hidden.dim(.s) <= std.math.maxInt(u32), "Perception attention requires a nonempty sequence addressable by u32 lengths", .{});

        const x = hidden.transpose(.{ .b, .s, .d });
        const head_dim = @divExact(self.config.d_model, self.config.n_heads);
        const q_raw, const k_raw, const v_raw = self.w_qkv.forward(x, x.dtype()).chunkExact(.dout, 3);
        var q = q_raw.splitAxis(.dout, .{ .h = self.config.n_heads, .hd = head_dim });
        var k = k_raw.splitAxis(.dout, .{ .h = self.config.n_heads, .hd = head_dim });
        const v = v_raw.splitAxis(.dout, .{ .h = self.config.n_heads, .hd = head_dim }).rename(.{ .s = .k });

        if (self.q_norm) |norm|
            q = applyLayerNorm(norm, q);

        if (self.k_norm) |norm|
            k = applyLayerNorm(norm, k);

        q = self.applyRope(q).rename(.{ .s = .q });
        k = self.applyRope(k).rename(.{ .s = .k });

        const positions = zml.Tensor.iota(zml.Shape.init(.{ .b = x.dim(.b), .k = x.dim(.s) }, .u32), .k).convert(.u32);
        const valid_keys = positions.cmp(.LT, lengths.broad(positions.shape()));
        const has_keys = lengths.cmp(.GT, .scalar(0, .u32));

        const safe_keys = has_keys.broad(valid_keys.shape()).select(valid_keys, positions.cmp(.EQ, .scalar(0, .u32)));
        const mask = safe_keys.select(.scalar(0, .f32), .scalar(-std.math.inf(f32), .f32));

        const attended = zml.nn.sdpa(q.convert(.f32), k.convert(.f32), v.convert(.f32), .{ .attn_mask = mask }).convert(v.dtype());
        const nonempty_output = has_keys.broad(attended.shape()).select(attended, .zeroes(attended.shape()));
        const combined = nonempty_output.transpose(.{ .b, .q, .h, .hd })
            .merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });

        return self.out_proj.forward(combined, x.dtype()).rename(.{ .dout = .d });
    }

    fn applyRope(self: EncoderAttention, x: zml.Tensor) zml.Tensor {
        const rotated = zml.nn.rope(x.slice(.hd, .{ .end = self.rotary_dim }), null, .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = self.config.rope_base } },
        });

        if (self.rotary_dim == x.dim(.hd))
            return rotated;

        return .concatenate(&.{ rotated, x.slice(.hd, .{ .start = self.rotary_dim }) }, .hd);
    }
};

pub const EncoderFeedForward = struct {
    up_proj: Linear,
    down_proj: Linear,

    pub fn init(store: zml.io.TensorStore.View, config: EncoderConfig) !EncoderFeedForward {
        try config.validate();

        const expanded = @as(f64, @floatFromInt(config.d_model)) * @as(f64, config.ff_expansion);
        if (!std.math.isFinite(expanded) or expanded < 1 or expanded >= @as(f64, @floatFromInt(std.math.maxInt(i64))))
            return error.InvalidPerceptionConfig;

        const intermediate_dim: i64 = @intFromFloat(expanded);
        const net = store.withPrefix("net");

        return .{
            .up_proj = try initLinear(net.withLayer(0), config.d_model, intermediate_dim, true),
            .down_proj = try initLinear(net.withLayer(3), intermediate_dim, config.d_model, true),
        };
    }

    pub fn forward(self: EncoderFeedForward, hidden: zml.Tensor) zml.Tensor {
        zml.stdx.debug.assert(hidden.rank() == 3 and hidden.shape().hasTags(.{ .b, .s, .d }) and hidden.dim(.d) == self.up_proj.weight.dim(.d), "Perception FFN expects {{b, s, d_model}} input", .{});

        const x = hidden.transpose(.{ .b, .s, .d });
        const projected = self.up_proj.forward(x, x.dtype());

        const fp32 = projected.convert(.f32);
        const activated = fp32.mul(fp32.scale(-std.math.sqrt(@as(f32, 0.5))).erfc()).scale(0.5)
            .convert(projected.dtype()).rename(.{ .dout = .d });

        return self.down_proj.forward(activated, x.dtype()).rename(.{ .dout = .d });
    }
};

pub const EncoderBlock = struct {
    norm1: zml.nn.LayerNorm,
    attn: EncoderAttention,
    norm2: zml.nn.LayerNorm,
    ffn: EncoderFeedForward,

    pub fn init(store: zml.io.TensorStore.View, config: EncoderConfig) !EncoderBlock {
        try config.validate();

        return .{
            .norm1 = try initLayerNorm(store.withPrefix("norm1"), config.d_model),
            .attn = try .init(store.withPrefix("attn"), config),
            .norm2 = try initLayerNorm(store.withPrefix("norm2"), config.d_model),
            .ffn = try .init(store.withPrefix("ffn"), config),
        };
    }

    pub fn forward(self: EncoderBlock, input: Output) Output {
        zml.stdx.debug.assert(input.hidden.rank() == 3 and input.hidden.shape().hasTags(.{ .b, .s, .d }), "Perception block input must have shape {{b, s, d}}", .{});
        zml.stdx.debug.assert(input.hidden.dim(.d) == self.norm1.weight.dim(.d) and input.hidden.dim(.d) == self.norm2.weight.dim(.d), "Perception block hidden width does not match its LayerNorm weights", .{});
        zml.stdx.debug.assert(input.lengths.rank() == 1 and input.lengths.shape().hasTags(.{.b}) and input.lengths.dim(.b) == input.hidden.dim(.b) and input.lengths.dtype() == .u32, "Perception block lengths must have shape {{b}} and dtype u32", .{});

        const x = input.hidden.transpose(.{ .b, .s, .d });
        const attention = self.attn.forward(applyLayerNorm(self.norm1, x), input.lengths);
        const after_attention = x.add(attention);

        const feed_forward = self.ffn.forward(applyLayerNorm(self.norm2, after_attention));
        return .{
            .hidden = after_attention.add(feed_forward),
            .lengths = input.lengths,
        };
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(EncoderBlock)) void {
        zml.Buffer.deinitAll(EncoderBlock, buffers);
    }
};

pub const Encoder = struct {
    pre_encode: FeatureStacking,
    embed_norm: ?zml.nn.LayerNorm,
    layers: []EncoderBlock,
    final_norm: ?zml.nn.LayerNorm,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: EncoderConfig) !Encoder {
        try config.validate();

        const layers = try allocator.alloc(EncoderBlock, config.n_layers);
        errdefer allocator.free(layers);

        const layer_store = store.withPrefix("layers");
        for (layers, 0..) |*layer, i|
            layer.* = try .init(layer_store.withLayer(i), config);

        return .{
            .pre_encode = try .init(store.withPrefix("pre_encode"), config),
            .embed_norm = if (config.pre_block_norm) try initLayerNorm(store.withPrefix("embed_norm"), config.d_model) else null,
            .layers = layers,
            .final_norm = if (config.final_norm) try initLayerNorm(store.withPrefix("final_norm"), config.d_model) else null,
        };
    }

    pub fn deinit(self: Encoder, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn forwardInput(self: Encoder, input: Input) Output {
        var output = self.pre_encode.forward(input);
        if (self.embed_norm) |norm|
            output.hidden = applyLayerNorm(norm, output.hidden);

        return output;
    }

    pub fn forward(self: Encoder, input: Input) Output {
        var output = self.forwardInput(input);
        for (self.layers) |layer|
            output = layer.forward(output);

        if (self.final_norm) |norm|
            output.hidden = applyLayerNorm(norm, output.hidden);

        return output;
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(Encoder), allocator: std.mem.Allocator) void {
        zml.Buffer.deinitAll(FeatureStacking, &buffers.pre_encode);
        if (buffers.embed_norm) |*norm|
            zml.Buffer.deinitAll(zml.nn.LayerNorm, norm);

        for (buffers.layers) |*layer|
            EncoderBlock.unloadBuffers(layer);
        allocator.free(buffers.layers);

        if (buffers.final_norm) |*norm|
            zml.Buffer.deinitAll(zml.nn.LayerNorm, norm);
    }
};

const SubpixelUpsample = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    factor: i64,
    hidden_dim: i64,

    fn init(store: zml.io.TensorStore.View, factor: i64, hidden_dim: i64) !SubpixelUpsample {
        const weight = store.getShape("weight") orelse return error.MissingPerceptionTensor;
        const bias = store.getShape("bias") orelse return error.MissingPerceptionTensor;
        const output_dim = std.math.mul(i64, factor, hidden_dim) catch return error.InvalidPerceptionConfig;

        if (weight.rank() != 3 or weight.dim(0) != output_dim or weight.dim(1) != hidden_dim or weight.dim(2) != 3 or
            bias.rank() != 1 or bias.dim(0) != output_dim or bias.dtype() != weight.dtype())
            return error.InvalidPerceptionShape;

        switch (weight.dtype()) {
            .bf16, .f16, .f32 => {},
            else => return error.UnsupportedPerceptionQuantization,
        }

        return .{
            .weight = store.createReplicatedTensor("weight", .{ .dout, .d, .kernel }),
            .bias = store.createReplicatedTensor("bias", .{.dout}),
            .factor = factor,
            .hidden_dim = hidden_dim,
        };
    }

    fn forward(self: SubpixelUpsample, hidden: zml.Tensor) zml.Tensor {
        const convolved = hidden.convert(self.weight.dtype()).conv1d(self.weight, .{
            .padding = &.{ 1, 1 },
            .input_feature_dimension = 2,
            .input_spatial_dimensions = 1,
            .output_feature_dimension = 2,
            .output_spatial_dimensions = 1,
        }).withTags(.{ .b, .s, .dout });
        const projected = convolved.add(self.bias.broad(convolved.shape()));
        return projected.splitAxis(.dout, .{ .phase = self.factor, .d = self.hidden_dim })
            .merge(.{ .s = .{ .s, .phase } });
    }
};

pub const Diarization = struct {
    encoder: Encoder,
    encoder_proj: ?Linear,
    subpixel_upsample: ?SubpixelUpsample,
    first_hidden_to_hidden: Linear,
    single_hidden_to_spks: Linear,
    normalization: Normalization,
    factor: i64,

    fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: DiarizationConfig, normalization: Normalization) !Diarization {
        const encoder = try Encoder.init(allocator, store.withPrefix("encoder"), config.encoder);
        errdefer encoder.deinit(allocator);

        const modules = store.withPrefix("sortformer_modules");
        const hidden_dim = config.sortformer_modules.tf_d_model;

        return .{
            .encoder = encoder,
            .encoder_proj = if (config.encoder.d_model != hidden_dim) try initLinear(modules.withPrefix("encoder_proj"), config.encoder.d_model, hidden_dim, true) else null,
            .subpixel_upsample = if (config.high_resolution and config.encoder.subsampling_factor > 1)
                try .init(modules.withPrefix("subpixel_upsample"), config.encoder.subsampling_factor, hidden_dim)
            else
                null,
            .first_hidden_to_hidden = try initLinear(modules.withPrefix("first_hidden_to_hidden"), hidden_dim, hidden_dim, true),
            .single_hidden_to_spks = try initLinear(modules.withPrefix("single_hidden_to_spks"), hidden_dim, config.sortformer_modules.num_spks, true),
            .normalization = normalization,
            .factor = if (config.high_resolution) config.encoder.subsampling_factor else 1,
        };
    }

    fn deinit(self: Diarization, allocator: std.mem.Allocator) void {
        self.encoder.deinit(allocator);
    }

    fn forward(self: Diarization, input: Input) zml.Tensor {
        const encoded = self.encoder.forward(self.normalization.forward(input));
        var hidden = encoded.hidden;
        if (self.encoder_proj) |projection|
            hidden = projection.forward(hidden, hidden.dtype()).rename(.{ .dout = .d });

        if (self.subpixel_upsample) |upsample|
            hidden = upsample.forward(hidden);

        hidden = self.first_hidden_to_hidden.forward(hidden.relu(), hidden.dtype()).rename(.{ .dout = .d }).relu();
        const logits = self.single_hidden_to_spks.forward(hidden, hidden.dtype()).rename(.{ .dout = .spk });
        const native_lengths = encoded.lengths.convert(.u64).mul(.scalar(self.factor, .u64));
        const mask = frameMask(logits.dim(.b), logits.dim(.s), native_lengths).broad(logits.shape());
        const probabilities = mask.select(logits.sigmoid(), .zeroes(logits.shape()));

        if (self.factor == 1)
            return probabilities;

        return probabilities.splitAxis(.s, .{ .s = encoded.hidden.dim(.s), .phase = self.factor })
            .convert(.f32).mean(.phase).squeeze(.phase).convert(probabilities.dtype());
    }

    fn unloadBuffers(buffers: *zml.Bufferized(Diarization), allocator: std.mem.Allocator) void {
        Encoder.unloadBuffers(&buffers.encoder, allocator);

        if (buffers.encoder_proj) |*projection|
            zml.Buffer.deinitAll(Linear, projection);

        if (buffers.subpixel_upsample) |*upsample|
            zml.Buffer.deinitAll(SubpixelUpsample, upsample);

        zml.Buffer.deinitAll(Linear, &buffers.first_hidden_to_hidden);
        zml.Buffer.deinitAll(Linear, &buffers.single_hidden_to_spks);
    }
};

const ParallelExperts = struct {
    diarization: Diarization,
    asr_norm: zml.nn.LayerNorm,
    diar_norm: zml.nn.LayerNorm,
    asr_normalization: Normalization,
    speaker_threshold: ?f32,
    speaker_scale: f32,

    fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, pe: ParallelConfig, diar_config: ?DiarizationConfig) !ParallelExperts {
        if (pe.asr_encoder_type != .transformer or pe.speaker_feature_config_version != 1)
            return error.UnsupportedPerceptionConfig;

        if (config.preprocessor.normalize != null or !std.math.isFinite(pe.spk_kernel_scale))
            return error.InvalidPerceptionConfig;

        var threshold: ?f32 = null;
        if (pe.speaker_feature_mode == .thresholded) {
            threshold = pe.speaker_activity_threshold orelse return error.InvalidPerceptionConfig;
            if (!std.math.isFinite(threshold.?) or threshold.? < 0 or threshold.? > 1)
                return error.InvalidPerceptionConfig;
        } else if (pe.speaker_activity_threshold != null)
            return error.InvalidPerceptionConfig;

        const dc = pe.diarization_model_cfg orelse diar_config orelse return error.MissingDiarizationConfig;
        try dc.validate(pe.asr_encoder_cfg);

        const normalization = try Normalization.parse(pe.asr_normalize_type orelse "per_feature");
        const diarization = try Diarization.init(allocator, store.withPrefix("diarization_model"), dc, try Normalization.parse(pe.diar_normalize_type orelse dc.preprocessor.normalize));
        errdefer diarization.deinit(allocator);

        return .{
            .diarization = diarization,
            .asr_norm = try initLayerNorm(store.withPrefix("asr_norm"), pe.asr_encoder_cfg.d_model),
            .diar_norm = try initLayerNorm(store.withPrefix("diar_norm"), dc.sortformer_modules.num_spks),
            .asr_normalization = normalization,
            .speaker_threshold = threshold,
            .speaker_scale = pe.spk_kernel_scale,
        };
    }

    fn deinit(self: ParallelExperts, allocator: std.mem.Allocator) void {
        self.diarization.deinit(allocator);
    }

    fn forward(self: ParallelExperts, input: Input, hidden: zml.Tensor) zml.Tensor {
        const diar_input: Input = .{
            .features = input.features.convert(self.diarization.encoder.pre_encode.proj.weight.dtype()),
            .lengths = input.lengths,
        };
        const probabilities = self.diarization.forward(diar_input).convert(hidden.dtype());
        zml.stdx.debug.assert(probabilities.dim(.s) == hidden.dim(.s), "ASR and diarization frame grids must match", .{});

        const activity = if (self.speaker_threshold) |threshold|
            probabilities.cmp(.GT, .scalar(threshold, probabilities.dtype())).convert(hidden.dtype())
        else
            probabilities;
        const speakers = applyLayerNorm(self.diar_norm, activity);
        const kernel = speakerKernel(speakers.dim(.spk), hidden.dim(.d)).convert(speakers.dtype());
        const infusion = speakers.dot(kernel, .spk).convert(hidden.dtype());
        return applyLayerNorm(self.asr_norm, hidden).add(infusion.scale(self.speaker_scale));
    }

    fn unloadBuffers(buffers: *zml.Bufferized(ParallelExperts), allocator: std.mem.Allocator) void {
        Diarization.unloadBuffers(&buffers.diarization, allocator);
        zml.Buffer.deinitAll(zml.nn.LayerNorm, &buffers.asr_norm);
        zml.Buffer.deinitAll(zml.nn.LayerNorm, &buffers.diar_norm);
    }
};

pub const Perception = struct {
    encoder: Encoder,
    parallel: ?ParallelExperts,
    proj: Linear,

    pub const Buffers = zml.Bufferized(Perception);

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, pe_config: ?ParallelConfig, diar_config: ?DiarizationConfig) !Perception {
        try config.validate();

        const wrapper = store.withPrefix("encoder");
        const wrapped = wrapper.hasKey("asr_encoder.pre_encode.proj.weight");
        if (wrapped != (pe_config != null))
            return error.MissingParallelPerceptionConfig;

        if (!wrapped and (diar_config != null or wrapper.hasKey("diar_norm.weight") or wrapper.hasKey("diarization_model.encoder.pre_encode.proj.weight")))
            return error.InvalidPerceptionConfig;

        const asr_config = if (pe_config) |pe| pe.asr_encoder_cfg else config.encoder;
        if (asr_config.d_model != config.modality_adapter.d_model or asr_config.feat_in != config.preprocessor.features)
            return error.InvalidPerceptionConfig;

        const encoder = try Encoder.init(allocator, if (wrapped) wrapper.withPrefix("asr_encoder") else wrapper, asr_config);
        errdefer encoder.deinit(allocator);

        const parallel: ?ParallelExperts = if (pe_config) |pe| try ParallelExperts.init(allocator, wrapper, config, pe, diar_config) else null;
        errdefer if (parallel) |experts|
            experts.deinit(allocator);

        return .{
            .encoder = encoder,
            .parallel = parallel,
            .proj = try initLinear(store.withPrefix("proj"), asr_config.d_model, config.output_dim, true),
        };
    }

    pub fn deinit(self: Perception, allocator: std.mem.Allocator) void {
        self.encoder.deinit(allocator);
        if (self.parallel) |experts|
            experts.deinit(allocator);
    }

    pub fn forward(self: Perception, input: Input) Output {
        const asr_input: Input = .{
            .features = input.features.convert(self.encoder.pre_encode.proj.weight.dtype()),
            .lengths = input.lengths,
        };
        const normalization: Normalization = if (self.parallel) |experts| experts.asr_normalization else .none;
        const encoded = self.encoder.forward(normalization.forward(asr_input));
        const hidden = if (self.parallel) |experts| experts.forward(input, encoded.hidden) else encoded.hidden;

        return .{
            .hidden = self.proj.forward(hidden, hidden.dtype()).rename(.{ .dout = .d }),
            .lengths = encoded.lengths,
        };
    }

    pub fn unloadBuffers(buffers: *Buffers, allocator: std.mem.Allocator) void {
        Encoder.unloadBuffers(&buffers.encoder, allocator);

        if (buffers.parallel) |*experts|
            ParallelExperts.unloadBuffers(experts, allocator);

        zml.Buffer.deinitAll(Linear, &buffers.proj);
    }
};

fn frameMask(batch: i64, frames: i64, lengths: zml.Tensor) zml.Tensor {
    const positions = zml.Tensor.iota(zml.Shape.init(.{ .b = batch, .s = frames }, lengths.dtype()), .s).convert(lengths.dtype());
    return positions.cmp(.LT, lengths.broad(positions.shape()));
}

fn speakerKernel(speakers: i64, width: i64) zml.Tensor {
    const pairs = @divExact(width, 2);
    const shape = zml.Shape.init(.{ .spk = speakers, .pair = pairs }, .f32);
    const positions = zml.Tensor.iota(shape, .spk).convert(.f32);
    const frequencies = zml.Tensor.iota(shape, .pair).convert(.f32).scale(-2 * @log(@as(f32, 10_000)) / @as(f32, @floatFromInt(width))).exp();
    const angles = positions.mul(frequencies);
    return zml.Tensor.stack(&.{ angles.sin(), angles.cos() }, .last, .component)
        .merge(.{ .d = .{ .pair, .component } });
}

fn initLayerNorm(store: zml.io.TensorStore.View, dimension: i64) !zml.nn.LayerNorm {
    const weight = store.getShape("weight") orelse return error.MissingPerceptionTensor;
    const bias = store.getShape("bias") orelse return error.MissingPerceptionTensor;
    if (weight.rank() != 1 or bias.rank() != 1 or weight.dim(0) != dimension or !weight.eql(bias))
        return error.InvalidPerceptionShape;

    switch (weight.dtype()) {
        .bf16, .f16, .f32 => {},
        else => return error.UnsupportedPerceptionQuantization,
    }

    return .{
        .weight = store.createReplicatedTensor("weight", .{.d}),
        .bias = store.createReplicatedTensor("bias", .{.d}),
    };
}

fn applyLayerNorm(norm: zml.nn.LayerNorm, x: zml.Tensor) zml.Tensor {
    const fp32_norm: zml.nn.LayerNorm = .{
        .weight = norm.weight.convert(.f32),
        .bias = if (norm.bias) |bias| bias.convert(.f32) else null,
        .eps = norm.eps,
    };
    return fp32_norm.forward(x.convert(.f32)).convert(x.dtype());
}

const Linear = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor,
    fp8: ?struct {
        weight_scale: zml.Tensor,
        input_scale: zml.Tensor,
    },

    fn forward(self: Linear, x: zml.Tensor, out_dtype: zml.DataType) zml.Tensor {
        if (self.fp8) |fp8| {
            const input_scale = fp8.input_scale.asScalar().convert(.f32);
            const quantized = x.convert(.f32).div(input_scale)
                .clamp(.scalar(-448, .f32), .scalar(448, .f32)).convert(.f8e4m3fn);
            var result = quantized.convert(.f32).dot(self.weight.convert(.f32), .d)
                .mul(input_scale.mul(fp8.weight_scale.asScalar().convert(.f32)));
            if (self.bias) |bias|
                result = result.add(bias.convert(.f32).broad(result.shape()));

            return result.convert(out_dtype);
        }

        var result = x.convert(self.weight.dtype()).convert(.f32).dot(self.weight.convert(.f32), .d);
        if (self.bias) |bias|
            result = result.add(bias.convert(.f32).broad(result.shape()));

        return result.convert(out_dtype);
    }
};

fn initLinear(store: zml.io.TensorStore.View, input_dim: i64, output_dim: i64, has_bias: bool) !Linear {
    try validateLinear(store, input_dim, output_dim, has_bias);
    const fp8 = store.getShape("weight").?.dtype() == .f8e4m3fn;

    return .{
        .weight = store.createReplicatedTensor("weight", .{ .dout, .d }),
        .bias = if (has_bias) store.createReplicatedTensor("bias", .{.dout}) else null,
        .fp8 = if (fp8) .{
            .weight_scale = store.createReplicatedTensor("weight_scale", null),
            .input_scale = store.createReplicatedTensor("input_scale", null),
        } else null,
    };
}

fn validateLinear(store: zml.io.TensorStore.View, input_dim: i64, output_dim: i64, has_bias: bool) !void {
    const weight = store.getShape("weight") orelse return error.MissingPerceptionTensor;
    if (weight.rank() != 2 or weight.dim(0) != output_dim or weight.dim(1) != input_dim)
        return error.InvalidPerceptionShape;

    switch (weight.dtype()) {
        .bf16, .f16, .f32 => {
            if (store.hasKey("weight_scale") or store.hasKey("input_scale"))
                return error.UnsupportedPerceptionQuantization;
        },
        .f8e4m3fn => {
            inline for (.{ "weight_scale", "input_scale" }) |key| {
                const scale = store.getShape(key) orelse return error.MissingPerceptionTensor;
                if (scale.dtype() != .f32 or scale.count() != 1)
                    return error.UnsupportedPerceptionQuantization;
            }
        },
        else => return error.UnsupportedPerceptionQuantization,
    }

    if (has_bias) {
        const bias = store.getShape("bias") orelse return error.MissingPerceptionTensor;
        if (bias.rank() != 1 or bias.dim(0) != output_dim)
            return error.InvalidPerceptionShape;

        if (weight.dtype() == .f8e4m3fn) {
            switch (bias.dtype()) {
                .bf16, .f16, .f32 => {},
                else => return error.UnsupportedPerceptionQuantization,
            }
        } else if (bias.dtype() != weight.dtype())
            return error.InvalidPerceptionShape;
    } else if (store.hasKey("bias"))
        return error.InvalidPerceptionShape;
}
