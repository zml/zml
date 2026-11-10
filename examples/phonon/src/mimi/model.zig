const std = @import("std");
const zml = @import("zml");

// Graph construction runs on the host; concurrent compiler threads keep separate depths.
threadlocal var debug_depth: usize = 0;
threadlocal var debug_filter: ?[]const u8 = null;
threadlocal var debug_selected_depth: ?usize = null;

// --- PocketTTSAudioTokenizerModel ---

/// Denormalizes 32-coordinate Phonon latents and decodes them into 24 kHz mono audio.
/// Each call processes a complete sequence without retaining decoder state.
///
/// Reference:
/// ```
/// PocketTTSAudioTokenizerModel
/// ├── decoder: PocketTTSAudioTokenizerDecoderModel
/// │   ├── upsample: PocketTTSAudioTokenizerConvTranspose1d
/// │   │   └── conv: ConvTranspose1d
/// │   │       └── weight: shape=(512, 1, 32), dtype=torch.float32
/// │   │           └── loaded: decoder.upsample.conv.weight
/// │   ├── decoder_transformer: PocketTTSAudioTokenizerTransformerModel
/// │   │   ├── layers: ModuleList
/// │   │   │   └── 0..1: PocketTTSAudioTokenizerTransformerLayer
/// │   │   │       ├── self_attn: PocketTTSAudioTokenizerAttention
/// │   │   │       │   ├── q_proj: Linear
/// │   │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │       │   │       └── loaded: decoder.decoder_transformer.layers.1.self_attn.q_proj.weight
/// │   │   │       │   ├── k_proj: Linear
/// │   │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │       │   │       └── loaded: decoder.decoder_transformer.layers.1.self_attn.k_proj.weight
/// │   │   │       │   ├── v_proj: Linear
/// │   │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │       │   │       └── loaded: decoder.decoder_transformer.layers.1.self_attn.v_proj.weight
/// │   │   │       │   └── o_proj: Linear
/// │   │   │       │       └── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │       │           └── loaded: decoder.decoder_transformer.layers.1.self_attn.o_proj.weight
/// │   │   │       ├── mlp: PocketTTSAudioTokenizerMLP
/// │   │   │       │   ├── activation_fn: GELUActivation
/// │   │   │       │   ├── fc1: Linear
/// │   │   │       │   │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │   │   │       │   │       └── loaded: decoder.decoder_transformer.layers.1.mlp.fc1.weight
/// │   │   │       │   └── fc2: Linear
/// │   │   │       │       └── weight: shape=(512, 2048), dtype=torch.float32
/// │   │   │       │           └── loaded: decoder.decoder_transformer.layers.1.mlp.fc2.weight
/// │   │   │       ├── input_layernorm: LayerNorm
/// │   │   │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   │       │   │   └── loaded: decoder.decoder_transformer.layers.1.input_layernorm.weight
/// │   │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │   │       │       └── loaded: decoder.decoder_transformer.layers.1.input_layernorm.bias
/// │   │   │       ├── post_attention_layernorm: LayerNorm
/// │   │   │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   │       │   │   └── loaded: decoder.decoder_transformer.layers.1.post_attention_layernorm.weight
/// │   │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │   │       │       └── loaded: decoder.decoder_transformer.layers.1.post_attention_layernorm.bias
/// │   │   │       ├── self_attn_layer_scale: PocketTTSAudioTokenizerLayerScale
/// │   │   │       │   └── scale: shape=(512,), dtype=torch.float32
/// │   │   │       │       └── loaded: decoder.decoder_transformer.layers.1.self_attn_layer_scale.scale
/// │   │   │       └── mlp_layer_scale: PocketTTSAudioTokenizerLayerScale
/// │   │   │           └── scale: shape=(512,), dtype=torch.float32
/// │   │   │               └── loaded: decoder.decoder_transformer.layers.1.mlp_layer_scale.scale
/// │   │   └── rotary_emb: PocketTTSAudioTokenizerRotaryEmbedding
/// │   └── decoder_seanet: PocketTTSAudioTokenizerDecoder
/// │       └── layers: ModuleList
/// │           ├── 0: PocketTTSAudioTokenizerConv1d
/// │           │   └── conv: Conv1d
/// │           │       ├── weight: shape=(512, 512, 7), dtype=torch.float32
/// │           │       │   └── loaded: decoder.decoder_seanet.layers.0.conv.weight
/// │           │       └── bias: shape=(512,), dtype=torch.float32
/// │           │           └── loaded: decoder.decoder_seanet.layers.0.conv.bias
/// │           ├── 1: ELU
/// │           ├── 2: PocketTTSAudioTokenizerConvTranspose1d
/// │           │   └── conv: ConvTranspose1d
/// │           │       ├── weight: shape=(512, 256, 12), dtype=torch.float32
/// │           │       │   └── loaded: decoder.decoder_seanet.layers.2.conv.weight
/// │           │       └── bias: shape=(256,), dtype=torch.float32
/// │           │           └── loaded: decoder.decoder_seanet.layers.2.conv.bias
/// │           ├── 3: PocketTTSAudioTokenizerResnetBlock
/// │           │   ├── block: ModuleList
/// │           │   │   ├── 0: ELU
/// │           │   │   ├── 1: PocketTTSAudioTokenizerConv1d
/// │           │   │   │   └── conv: Conv1d
/// │           │   │   │       ├── weight: shape=(128, 256, 3), dtype=torch.float32
/// │           │   │   │       │   └── loaded: decoder.decoder_seanet.layers.3.block.1.conv.weight
/// │           │   │   │       └── bias: shape=(128,), dtype=torch.float32
/// │           │   │   │           └── loaded: decoder.decoder_seanet.layers.3.block.1.conv.bias
/// │           │   │   ├── 2: ELU
/// │           │   │   └── 3: PocketTTSAudioTokenizerConv1d
/// │           │   │       └── conv: Conv1d
/// │           │   │           ├── weight: shape=(256, 128, 1), dtype=torch.float32
/// │           │   │           │   └── loaded: decoder.decoder_seanet.layers.3.block.3.conv.weight
/// │           │   │           └── bias: shape=(256,), dtype=torch.float32
/// │           │   │               └── loaded: decoder.decoder_seanet.layers.3.block.3.conv.bias
/// │           │   └── shortcut: Identity
/// │           ├── 4: ELU
/// │           ├── 5: PocketTTSAudioTokenizerConvTranspose1d
/// │           │   └── conv: ConvTranspose1d
/// │           │       ├── weight: shape=(256, 128, 10), dtype=torch.float32
/// │           │       │   └── loaded: decoder.decoder_seanet.layers.5.conv.weight
/// │           │       └── bias: shape=(128,), dtype=torch.float32
/// │           │           └── loaded: decoder.decoder_seanet.layers.5.conv.bias
/// │           ├── 6: PocketTTSAudioTokenizerResnetBlock
/// │           │   ├── block: ModuleList
/// │           │   │   ├── 0: ELU
/// │           │   │   ├── 1: PocketTTSAudioTokenizerConv1d
/// │           │   │   │   └── conv: Conv1d
/// │           │   │   │       ├── weight: shape=(64, 128, 3), dtype=torch.float32
/// │           │   │   │       │   └── loaded: decoder.decoder_seanet.layers.6.block.1.conv.weight
/// │           │   │   │       └── bias: shape=(64,), dtype=torch.float32
/// │           │   │   │           └── loaded: decoder.decoder_seanet.layers.6.block.1.conv.bias
/// │           │   │   ├── 2: ELU
/// │           │   │   └── 3: PocketTTSAudioTokenizerConv1d
/// │           │   │       └── conv: Conv1d
/// │           │   │           ├── weight: shape=(128, 64, 1), dtype=torch.float32
/// │           │   │           │   └── loaded: decoder.decoder_seanet.layers.6.block.3.conv.weight
/// │           │   │           └── bias: shape=(128,), dtype=torch.float32
/// │           │   │               └── loaded: decoder.decoder_seanet.layers.6.block.3.conv.bias
/// │           │   └── shortcut: Identity
/// │           ├── 7: ELU
/// │           ├── 8: PocketTTSAudioTokenizerConvTranspose1d
/// │           │   └── conv: ConvTranspose1d
/// │           │       ├── weight: shape=(128, 64, 8), dtype=torch.float32
/// │           │       │   └── loaded: decoder.decoder_seanet.layers.8.conv.weight
/// │           │       └── bias: shape=(64,), dtype=torch.float32
/// │           │           └── loaded: decoder.decoder_seanet.layers.8.conv.bias
/// │           ├── 9: PocketTTSAudioTokenizerResnetBlock
/// │           │   ├── block: ModuleList
/// │           │   │   ├── 0: ELU
/// │           │   │   ├── 1: PocketTTSAudioTokenizerConv1d
/// │           │   │   │   └── conv: Conv1d
/// │           │   │   │       ├── weight: shape=(32, 64, 3), dtype=torch.float32
/// │           │   │   │       │   └── loaded: decoder.decoder_seanet.layers.9.block.1.conv.weight
/// │           │   │   │       └── bias: shape=(32,), dtype=torch.float32
/// │           │   │   │           └── loaded: decoder.decoder_seanet.layers.9.block.1.conv.bias
/// │           │   │   ├── 2: ELU
/// │           │   │   └── 3: PocketTTSAudioTokenizerConv1d
/// │           │   │       └── conv: Conv1d
/// │           │   │           ├── weight: shape=(64, 32, 1), dtype=torch.float32
/// │           │   │           │   └── loaded: decoder.decoder_seanet.layers.9.block.3.conv.weight
/// │           │   │           └── bias: shape=(64,), dtype=torch.float32
/// │           │   │               └── loaded: decoder.decoder_seanet.layers.9.block.3.conv.bias
/// │           │   └── shortcut: Identity
/// │           ├── 10: ELU
/// │           └── 11: PocketTTSAudioTokenizerConv1d
/// │               └── conv: Conv1d
/// │                   ├── weight: shape=(1, 64, 3), dtype=torch.float32
/// │                   │   └── loaded: decoder.decoder_seanet.layers.11.conv.weight
/// │                   └── bias: shape=(1,), dtype=torch.float32
/// │                       └── loaded: decoder.decoder_seanet.layers.11.conv.bias
/// └── latent_proj: Conv1d
///     └── weight: shape=(512, 32, 1), dtype=torch.float32
///         └── loaded: latent_proj.weight
/// ```
pub const Pocket_TTS_Audio_Tokenizer_Model = struct {
    latent_proj: Latent_Projection,
    decoder: Decoder,

    mean: zml.Tensor,
    stddev: zml.Tensor,

    /// Expects the root gradium/phonon-mimi checkpoint view.
    pub fn init(store_phonon: zml.io.TensorStore.View, store_mimi: zml.io.TensorStore.View) Pocket_TTS_Audio_Tokenizer_Model {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .latent_proj = .init(store_mimi.withPrefix("latent_proj")),
            .decoder = .init(store_mimi.withPrefix("decoder")),

            .mean = store_phonon.createReplicatedTensor("flow_lm.emb_mean", .{.ld}),
            .stddev = store_phonon.createReplicatedTensor("flow_lm.emb_std", .{.ld}),
        };
    }

    pub fn bufferize(self: Pocket_TTS_Audio_Tokenizer_Model, allocator: std.mem.Allocator) !zml.Bufferized(Pocket_TTS_Audio_Tokenizer_Model) {
        const here = @typeName(@This()) ++ ".bufferize";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .latent_proj = try zml.mem.bufferize(allocator, Latent_Projection, &self.latent_proj),
            .decoder = try self.decoder.bufferize(allocator),

            .mean = try zml.mem.bufferize(allocator, zml.Tensor, &self.mean),
            .stddev = try zml.mem.bufferize(allocator, zml.Tensor, &self.stddev),
        };
    }

    /// `active_lengths`: [2] i32, valid token positions and completed frames.
    /// Uses the post-step frame count in slot 1 to select the newest audio chunk.
    pub fn forward(self: Pocket_TTS_Audio_Tokenizer_Model, active_lengths: zml.Tensor, history34: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("active_lengths: {f}\n", .{active_lengths.shape()});
        debug_print("history: {f}\n", .{history34.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        // std.debug.assert(active_lengths.dtype() == .i32 and active_lengths.rank() == 1 and active_lengths.dim(0) == 2);
        // const active_frame_length = active_lengths.slice(0, .single(1)); // [2] i32 -> scalar i32 frame count

        var frame = history34.withPartialTags(.{.ld});
        debug_print("frame: {f}\n", .{frame.shape()});
        // const winsz = @min(frame.dim(.a), 34);
        // frame = frame.slice(.a, .dyn(active_frame_length.subConstant(winsz), winsz));
        // debug_print("frame: {f}\n", .{frame.shape()});
        frame = frame.mul(self.stddev.broad(frame.shape()).convert(frame.dtype()));
        debug_print("frame: {f}\n", .{frame.shape()});
        frame = frame.add(self.mean.broad(frame.shape()).convert(frame.dtype()));
        debug_print("frame: {f}\n", .{frame.shape()});

        const projected = self.latent_proj.forward(frame);
        debug_print("projected: {f}\n", .{projected.shape()});
        const output = self.decoder.forward(active_lengths, projected);
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

// --- Latent_Projection ---

/// Projects each denormalized audio latent from 32 to 512 features.
///
/// This is the decoder's bias-free Conv1d(32, 512, kernel_size=1). A single
/// kernel tap mixes latent coordinates independently at each frame, so it is
/// implemented as a matrix multiplication followed by an axis transpose.
/// The f32 checkpoint weight [512, 32, 1] comes from `latent_proj.weight`.
/// Input and output preserve the input's floating-point dtype.
///
/// Reference:
/// ```
/// latent_proj: Conv1d
/// └── weight: shape=(512, 32, 1), dtype=torch.float32
///     └── loaded: latent_proj.weight
/// ```
pub const Latent_Projection = struct {
    weight: zml.Tensor,

    /// Expects the checkpoint view at `latent_proj`.
    pub fn init(store: zml.io.TensorStore.View) Latent_Projection {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .weight = store.createReplicatedTensor("weight", .{ .d, .ld, .k }) };
    }

    /// Projects continuous decoder-space latents, assigning tags `.b`, `.s`, `.ld`.
    ///
    /// `input`: floating-point [B, S, 32], denormalized Phonon output. The caller
    /// first applies `normalized * flow_lm.emb_std + flow_lm.emb_mean` using the
    /// TTS checkpoint's statistics. S is the number of generated audio frames.
    ///
    /// Returns [B, 512, S] in `input.dtype()`, in the channels-first layout
    /// expected by the decoder's temporal upsampling convolution.
    ///
    /// Dimension tags:
    /// - `.b`: independent sequences; batch rows never mix.
    /// - `.s`: audio frames; projected independently and preserved in order.
    /// - `.ld`: 32 latent coordinates, contracted with the projection weight.
    /// - `.d`: 512 projected decoder features.
    /// - `.k`: the weight's single kernel tap, removed before the matrix product.
    ///
    /// Shape: [.b, .s, .ld=32] -> [.b, .d=512, .s].
    pub fn forward(self: Latent_Projection, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat() and input.rank() == 3);
        std.debug.assert(self.weight.dtype().isFloat() and self.weight.rank() == 3);
        std.debug.assert(self.weight.dim(.k) == 1);
        std.debug.assert(self.weight.dim(.ld) == 32 and self.weight.dim(.d) == 512);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const x = input.withTags(.{ .b, .s, .ld });
        debug_print("x: {f}\n", .{x.shape()});
        std.debug.assert(x.dim(.ld) == 32);
        const weight = self.weight.squeeze(.k).convert(x.dtype());
        debug_print("weight: {f}\n", .{weight.shape()});
        const projected = x.dot(weight, .ld);
        debug_print("projected: {f}\n", .{projected.shape()});
        const output = projected.transpose(.{ .b, .d, .s });
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

/// Temporal upsampling, two-layer transformer, and SEANet waveform synthesis.
/// This is the reference's PocketTTSAudioTokenizerDecoderModel; latent projection
/// belongs to its parent. Targets the decoder-only gradium/phonon-mimi config.
pub const Decoder = struct {
    upsample: Conv_Transpose_1D,
    transformer: Transformer,
    seanet: SEANet,

    pub fn init(store: zml.io.TensorStore.View) Decoder {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .upsample = .init(store.withPrefix("upsample"), 16, 512),
            .transformer = .init(store.withPrefix("decoder_transformer")),
            .seanet = .init(store.withPrefix("decoder_seanet")),
        };
    }

    pub fn bufferize(self: Decoder, allocator: std.mem.Allocator) !zml.Bufferized(Decoder) {
        const here = @typeName(@This()) ++ ".bufferize";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .upsample = try zml.mem.bufferize(allocator, Conv_Transpose_1D, &self.upsample),
            .transformer = try self.transformer.bufferize(allocator),
            .seanet = try self.seanet.bufferize(allocator),
        };
    }

    pub fn forward(self: Decoder, active_lengths: zml.Tensor, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("active_lengths: {f}\n", .{active_lengths.shape()});
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const upsampled = self.upsample.forward(null, input);
        debug_print("upsampled: {f}\n", .{upsampled.shape()});
        const transformer_input = upsampled.transpose(.{ .b, .s, .d });
        debug_print("transformer_input: {f}\n", .{transformer_input.shape()});
        const contextualized = self.transformer.forward(transformer_input);
        debug_print("contextualized: {f}\n", .{contextualized.shape()});
        const waveform_features = contextualized.transpose(.{ .b, .d, .s });
        debug_print("waveform_features: {f}\n", .{waveform_features.shape()});
        const output = self.seanet.forward(active_lengths, waveform_features);
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

/// Causal transposed convolution: inserts stride-1 zeros between input samples,
/// convolves with the reversed kernel, and trims K-stride samples from the right.
/// Used for depthwise 16x latent upsampling and dense 6x, 5x, 4x SEANet upsampling.
/// Checkpoint layout is [C_in, C_out/groups, K], unlike Conv1d's weight layout.
pub const Conv_Transpose_1D = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor,
    stride: i64,
    groups: i64 = 1,

    pub fn init(store: zml.io.TensorStore.View, stride: i64, groups: i64) Conv_Transpose_1D {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .weight = store.createReplicatedTensor("conv.weight", .{ .in, .out, .k }),
            .bias = if (store.hasKey("conv.bias")) store.createReplicatedTensor("conv.bias", .{.d}) else null,
            .stride = stride,
            .groups = groups,
        };
    }

    /// `input`: floating [B, C_in, S], from latent projection or a SEANet ELU.
    /// Returns [B, C_out, S*stride] in input.dtype(). Tags .b batch, .d channels,
    /// .s time; weight .in/.out describe channels and .k the kernel's time taps.
    /// `idx` labels repeated upsamplers; null keeps standalone calls unindexed.
    pub fn forward(self: Conv_Transpose_1D, comptime idx: ?usize, input: zml.Tensor) zml.Tensor {
        const here = if (idx) |i|
            std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), i })
        else
            @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), if (idx) |i|
            std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), i })
        else
            @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const features = input.withTags(.{ .b, .d, .s });
        debug_print("features: {f}\n", .{features.shape()});
        const raw_output = self.convolve(features);
        debug_print("raw_output: {f}\n", .{raw_output.shape()});
        const finalized_length = features.dim(.s) * self.stride;
        const finalized = raw_output.slice(.s, .{ .end = finalized_length });
        debug_print("finalized: {f}\n", .{finalized.shape()});
        const output = self.add_bias(finalized);
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }

    fn convolve(self: Conv_Transpose_1D, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".convolve";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".convolve", .{});
        defer compiler.popLocation();

        std.debug.assert(input.dtype().isFloat() and input.dim(.s) > 0);
        std.debug.assert(self.stride > 0 and self.weight.dim(.k) >= self.stride);
        const kernel = self.convolution_kernel(input.dtype());
        debug_print("kernel: {f}\n", .{kernel.shape()});
        const padding = self.weight.dim(.k) - 1;
        const output = input.conv1d(kernel, .{
            .lhs_dilation = self.stride,
            .padding = &.{ padding, padding },
            .window_reversal = true,
            .feature_group_count = self.groups,
        }).withTags(.{ .b, .d, .s });
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }

    /// PyTorch [C_in,C_out/groups,K] -> StableHLO [C_out,C_in/groups,K].
    /// Swap channel axes within each group so groups remain independent.
    fn convolution_kernel(self: Conv_Transpose_1D, dtype: zml.DataType) zml.Tensor {
        const here = @typeName(@This()) ++ ".convolution_kernel";

        debug_enter(here);
        defer debug_exit(here);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".convolution_kernel", .{});
        defer compiler.popLocation();

        const channels_per_group = @divExact(self.weight.dim(.in), self.groups);
        const grouped = self.weight.convert(dtype).splitAxis(.in, .{ .g = self.groups, .ci = channels_per_group });
        debug_print("grouped: {f}\n", .{grouped.shape()});
        const output_first = grouped.transpose(.{ .g, .out, .ci, .k });
        debug_print("output_first: {f}\n", .{output_first.shape()});
        const output = output_first.merge(.{ .co = .{ .g, .out } });
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }

    fn add_bias(self: Conv_Transpose_1D, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".add_bias";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".add_bias", .{});
        defer compiler.popLocation();

        const output = if (self.bias) |bias| input.add(bias.convert(input.dtype()).broad(input.shape())) else input;
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

/// The two decoder transformer layers. There is no final LayerNorm.
pub const Transformer = struct {
    layers: [2]Transformer_Layer,

    pub fn init(store: zml.io.TensorStore.View) Transformer {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .layers = .{ .init(store.withPrefix("layers.0")), .init(store.withPrefix("layers.1")) } };
    }

    // ZML bufferize does not descend fixed arrays; initialize each layer explicitly.
    pub fn bufferize(self: Transformer, allocator: std.mem.Allocator) !zml.Bufferized(Transformer) {
        const here = @typeName(@This()) ++ ".bufferize";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers: zml.Bufferized(Transformer) = undefined;
        for (&self.layers, &buffers.layers) |*layer, *target| target.* = try zml.mem.bufferize(allocator, Transformer_Layer, layer);
        return buffers;
    }

    /// `input`: floating [B, S, 512], upsampled latent projection transposed to
    /// sequence-first layout. Returns contextual features for SEANet, same shape
    /// and dtype, tagged .b batch/.s positions/.d features.
    pub fn forward(self: Transformer, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        var hidden = input;
        debug_print("hidden: {f}\n", .{hidden.shape()});
        inline for (self.layers, 0..) |layer, i| {
            hidden = layer.forward(i, hidden);
            debug_print("hidden: {f}\n", .{hidden.shape()});
        }

        debug_print("output: {f}\n", .{hidden.shape()});
        return hidden;
    }
};

/// One pre-norm transformer layer: attention and MLP each have their own learned
/// LayerScale and residual connection. Parameters come from decoder_transformer.layers.N.
pub const Transformer_Layer = struct {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: Layer_Norm,
    post_attention_layernorm: Layer_Norm,
    self_attn_layer_scale: Layer_Scale,
    mlp_layer_scale: Layer_Scale,

    pub fn init(store: zml.io.TensorStore.View) Transformer_Layer {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .self_attn = .init(store.withPrefix("self_attn")),
            .mlp = .init(store.withPrefix("mlp")),
            .input_layernorm = .init(store.withPrefix("input_layernorm")),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm")),
            .self_attn_layer_scale = .init(store.withPrefix("self_attn_layer_scale")),
            .mlp_layer_scale = .init(store.withPrefix("mlp_layer_scale")),
        };
    }

    /// `input`: floating [B, S, 512], upsampler output or the preceding layer's
    /// residual features. Returns the same shape/dtype, tags .b batch/.s time/.d features.
    /// `idx` identifies this repeated layer in compiler source locations.
    pub fn forward(self: Transformer_Layer, comptime idx: usize, input: zml.Tensor) zml.Tensor {
        const here = std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx });

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx }), .{});
        defer compiler.popLocation();

        const hidden = input.withTags(.{ .b, .s, .d });
        debug_print("hidden: {f}\n", .{hidden.shape()});
        const normalized = self.input_layernorm.forward(hidden);
        debug_print("normalized: {f}\n", .{normalized.shape()});
        const attention = self.self_attn.forward(normalized);
        debug_print("attention: {f}\n", .{attention.shape()});
        const scaled_attention = self.self_attn_layer_scale.forward(attention);
        debug_print("scaled_attention: {f}\n", .{scaled_attention.shape()});
        const residual = hidden.add(scaled_attention);
        debug_print("residual: {f}\n", .{residual.shape()});
        const output = self.add_mlp_residual(residual);
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }

    /// Normalize, transform, and scale the MLP branch before adding [B,.s,.d=512] residual.
    fn add_mlp_residual(self: Transformer_Layer, residual: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".add_mlp_residual";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("residual: {f}\n", .{residual.shape()});

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".add_mlp_residual", .{});
        defer compiler.popLocation();

        const normalized = self.post_attention_layernorm.forward(residual);
        debug_print("normalized: {f}\n", .{normalized.shape()});
        const update = self.mlp.forward(normalized);
        debug_print("update: {f}\n", .{update.shape()});
        const scaled_update = self.mlp_layer_scale.forward(update);
        debug_print("scaled_update: {f}\n", .{scaled_update.shape()});
        const output = residual.add(scaled_update);
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

/// Feature-wise LayerNorm used before attention and before the MLP, epsilon 1e-5.
/// Checkpoint vectors are [512]. ZML computes normalization statistics in f32.
pub const Layer_Norm = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Layer_Norm {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .weight = store.createReplicatedTensor("weight", .{.d}), .bias = store.createReplicatedTensor("bias", .{.d}) };
    }

    /// `input`: floating [B, S, 512], residual features from the upsampler or the
    /// preceding transformer branch. Returns normalized features of the same
    /// shape and dtype. Tags .b independent sequences, .s positions, .d features.
    pub fn forward(self: Layer_Norm, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        std.debug.assert(input.dtype().isFloat());
        const norm: zml.nn.LayerNorm = .{ .weight = self.weight.convert(input.dtype()), .bias = self.bias.convert(input.dtype()), .eps = 1e-5 };
        const features = input.withTags(.{ .b, .s, .d });
        debug_print("features: {f}\n", .{features.shape()});
        const output = norm.forward(features);
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

/// Eight-head causal attention with split-half RoPE (theta 10000) and a 251-slot
/// window. Separate Q/K/V checkpoint projections already use the HF head layout.
/// Parameters convert to the input floating dtype;
/// ZML's RoPE trigonometry and softmax use f32 internally.
pub const Attention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Attention {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .q_proj = .init(store.createReplicatedTensor("q_proj.weight", .{ .out, .d }), null, .d),
            .k_proj = .init(store.createReplicatedTensor("k_proj.weight", .{ .out, .d }), null, .d),
            .v_proj = .init(store.createReplicatedTensor("v_proj.weight", .{ .out, .d }), null, .d),
            .o_proj = .init(store.createReplicatedTensor("o_proj.weight", .{ .out, .d }), null, .d),
        };
    }

    /// `input`: floating [B, S, 512], pre-attention normalized decoder features.
    /// Positions start at zero. Returns [B, S, 512] in input.dtype(), before the
    /// caller's LayerScale/residual. Tags .b batch, .s time, .d features; internally
    /// .q queries, .k keys, .h eight heads, .hd 64 coordinates per head.
    pub fn forward(self: Attention, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const hidden = input.withTags(.{ .b, .s, .d });
        debug_print("hidden: {f}\n", .{hidden.shape()});
        const positions = zml.Tensor.arange(.{ .end = hidden.dim(.s) }, .i32).withTags(.{.s});
        debug_print("positions: {f}\n", .{positions.shape()});
        const projected = self.project_qkv(hidden);
        debug_print("projected.queries: {f}\n", .{projected.queries.shape()});
        debug_print("projected.keys: {f}\n", .{projected.keys.shape()});
        debug_print("projected.values: {f}\n", .{projected.values.shape()});
        const rotated = apply_rope(projected, positions);
        debug_print("rotated.queries: {f}\n", .{rotated.queries.shape()});
        debug_print("rotated.keys: {f}\n", .{rotated.keys.shape()});
        debug_print("rotated.values: {f}\n", .{rotated.values.shape()});
        const mask = sliding_window_mask(positions, hidden.dim(.s), hidden.dtype());
        debug_print("mask: {f}\n", .{mask.shape()});
        const output = self.attend_and_project(rotated.queries, rotated.keys, rotated.values, mask);
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }

    /// Before RoPE all fields are [B, new_positions, 8, 64], tagged .b/.s/.h/.hd.
    /// After RoPE, queries use .q instead of .s and keys/values use .k instead.
    const QKV = struct {
        queries: zml.Tensor,
        keys: zml.Tensor,
        values: zml.Tensor,
    };

    /// Separate checkpoint projections, each [B,.s,512] -> [B,.s,.h=8,.hd=64].
    fn project_qkv(self: Attention, hidden: zml.Tensor) QKV {
        const here = @typeName(@This()) ++ ".project_qkv";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("hidden: {f}\n", .{hidden.shape()});

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".project_qkv", .{});
        defer compiler.popLocation();

        std.debug.assert(hidden.dtype().isFloat() and hidden.dim(.d) == 512);
        const output: QKV = .{
            .queries = self.q_proj.forward(hidden, hidden.dtype()).splitAxis(.out, .{ .h = 8, .hd = 64 }),
            .keys = self.k_proj.forward(hidden, hidden.dtype()).splitAxis(.out, .{ .h = 8, .hd = 64 }),
            .values = self.v_proj.forward(hidden, hidden.dtype()).splitAxis(.out, .{ .h = 8, .hd = 64 }),
        };
        debug_print("output.queries: {f}\n", .{output.queries.shape()});
        debug_print("output.keys: {f}\n", .{output.keys.shape()});
        debug_print("output.values: {f}\n", .{output.values.shape()});
        return output;
    }

    /// Rotate Q/K at absolute i32 positions [.s]. Mimi's separate checkpoint
    /// weights use split-half real/imaginary coordinates. Values are not rotated.
    fn apply_rope(projected: QKV, positions: zml.Tensor) QKV {
        const here = @typeName(@This()) ++ ".apply_rope";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("positions: {f}\n", .{positions.shape()});
        debug_print("projected.queries: {f}\n", .{projected.queries.shape()});
        debug_print("projected.keys: {f}\n", .{projected.keys.shape()});
        debug_print("projected.values: {f}\n", .{projected.values.shape()});

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".apply_rope", .{});
        defer compiler.popLocation();

        const options: zml.nn.RopeOpts = .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = 10000 } },
        };
        const output: QKV = .{
            .queries = zml.nn.rope(projected.queries, positions, options).rename(.{ .s = .q }),
            .keys = zml.nn.rope(projected.keys, positions, options).rename(.{ .s = .k }),
            .values = projected.values.rename(.{ .s = .k }),
        };
        debug_print("output.queries: {f}\n", .{output.queries.shape()});
        debug_print("output.keys: {f}\n", .{output.keys.shape()});
        debug_print("output.values: {f}\n", .{output.values.shape()});
        return output;
    }

    /// Absolute query positions [.s] -> additive [.q,.k] mask. Each query
    /// sees itself and its 250 predecessors; future and older positions are blocked.
    fn sliding_window_mask(positions: zml.Tensor, capacity: i64, dtype: zml.DataType) zml.Tensor {
        const here = @typeName(@This()) ++ ".sliding_window_mask";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("positions: {f}\n", .{positions.shape()});

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".sliding_window_mask", .{});
        defer compiler.popLocation();

        const shape: zml.Shape = .init(.{ .q = positions.dim(.s), .k = capacity }, .i32);
        const queries = positions.rename(.{ .s = .q }).broad(shape);
        debug_print("queries: {f}\n", .{queries.shape()});
        const keys = zml.Tensor.iota(shape, .k);
        debug_print("keys: {f}\n", .{keys.shape()});
        const distance = queries.sub(keys);
        debug_print("distance: {f}\n", .{distance.shape()});
        const past_or_present = distance.cmp(.GE, zml.Tensor.scalar(0, .i32));
        debug_print("past_or_present: {f}\n", .{past_or_present.shape()});
        const within_window = distance.cmp(.LT, zml.Tensor.scalar(251, .i32));
        debug_print("within_window: {f}\n", .{within_window.shape()});
        const visible = past_or_present.logical(.AND, within_window);
        debug_print("visible: {f}\n", .{visible.shape()});
        const output = visible.select(zml.Tensor.scalar(0, dtype), zml.Tensor.scalar(-std.math.inf(f32), dtype));
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }

    /// Attend from [B, new_positions, 8, 64] queries to keys/values, then merge
    /// heads and project to [B, new_positions, 512], tagged .b/.s/.d.
    /// .out distinguishes the output projection's result from its contracted .d axis.
    fn attend_and_project(self: Attention, queries: zml.Tensor, keys: zml.Tensor, values: zml.Tensor, mask: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".attend_and_project";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("queries: {f}\n", .{queries.shape()});
        debug_print("keys: {f}\n", .{keys.shape()});
        debug_print("values: {f}\n", .{values.shape()});
        debug_print("mask: {f}\n", .{mask.shape()});

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".attend_and_project", .{});
        defer compiler.popLocation();

        const attended = zml.nn.sdpa(queries, keys, values, .{ .attn_mask = mask });
        debug_print("attended: {f}\n", .{attended.shape()});
        const features = attended.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        debug_print("features: {f}\n", .{features.shape()});
        const output = self.o_proj.forward(features, queries.dtype()).rename(.{ .out = .d });
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

/// Learned per-feature multiplier on each transformer branch before residual addition.
pub const Layer_Scale = struct {
    scale: zml.Tensor,
    pub fn init(store: zml.io.TensorStore.View) Layer_Scale {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .scale = store.createReplicatedTensor("scale", .{.d}) };
    }

    /// `input`: floating [B, S, 512], projected attention or MLP output.
    /// Returns the same shape/dtype scaled by checkpoint [512] parameters.
    /// Tags .b batch, .s positions, .d features; only .d selects the multiplier.
    pub fn forward(self: Layer_Scale, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        std.debug.assert(input.dtype().isFloat());
        const x = input.withTags(.{ .b, .s, .d });
        debug_print("x: {f}\n", .{x.shape()});
        const output = x.mul(self.scale.convert(x.dtype()).broad(x.shape()));
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

/// Bias-free 512 -> 2048 -> 512 feed-forward branch. gradium/phonon-mimi uses
/// exact GELU (erf), not the tanh approximation in the library's default config.
pub const MLP = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    pub fn init(store: zml.io.TensorStore.View) MLP {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .fc1 = .init(store.createReplicatedTensor("fc1.weight", .{ .ff, .d }), null, .d),
            .fc2 = .init(store.createReplicatedTensor("fc2.weight", .{ .d, .ff }), null, .ff),
        };
    }

    /// `input`: floating [B, S, 512], post-attention normalized residual features.
    /// Returns [B, S, 512] in input.dtype(), before LayerScale and residual addition.
    /// Tags .b batch, .s positions, .d model features, .ff intermediate features.
    /// erf is evaluated in f32 internally, then converted back to the input dtype.
    pub fn forward(self: MLP, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        std.debug.assert(input.dtype().isFloat());
        const features = input.withTags(.{ .b, .s, .d });
        debug_print("features: {f}\n", .{features.shape()});
        const expanded = self.fc1.forward(features, features.dtype());
        debug_print("expanded: {f}\n", .{expanded.shape()});
        const activated = gelu_erf(expanded);
        debug_print("activated: {f}\n", .{activated.shape()});
        const output = self.fc2.forward(activated, features.dtype());
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }

    /// Exact GELU on [B,.s,.ff=2048]. erf runs in f32 and is converted back.
    fn gelu_erf(features: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".gelu_erf";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("features: {f}\n", .{features.shape()});

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".gelu_erf", .{});
        defer compiler.popLocation();

        const erf_term = features.scale(1.0 / @sqrt(2.0)).erf().convert(features.dtype());
        debug_print("erf_term: {f}\n", .{erf_term.shape()});
        const output = features.scale(0.5).mul(erf_term.addConstant(1));
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

/// Converts 200 Hz transformer features into a 24 kHz mono waveform. A K=7
/// convolution is followed by three ELU/upsample/residual stages (6x,5x,4x),
/// then ELU and a K=3 output convolution. Channels descend 512 -> 256 -> 128 -> 64 -> 1.
/// Parameters come from `decoder.decoder_seanet.layers`; convolutions are causal.
pub const SEANet = struct {
    initial: Conv1d,
    upsample: [3]Conv_Transpose_1D,
    residual: [3]Resnet_Block,
    final: Conv1d,

    pub fn init(store: zml.io.TensorStore.View) SEANet {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .initial = .init(store.withPrefix("layers.0")),
            .upsample = .{
                .init(store.withPrefix("layers.2"), 6, 1),
                .init(store.withPrefix("layers.5"), 5, 1),
                .init(store.withPrefix("layers.8"), 4, 1),
            },
            .residual = .{
                .init(store.withPrefix("layers.3")),
                .init(store.withPrefix("layers.6")),
                .init(store.withPrefix("layers.9")),
            },
            .final = .init(store.withPrefix("layers.11")),
        };
    }

    pub fn bufferize(self: SEANet, allocator: std.mem.Allocator) !zml.Bufferized(SEANet) {
        const here = @typeName(@This()) ++ ".bufferize";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var buffers: zml.Bufferized(SEANet) = .{
            .initial = try zml.mem.bufferize(allocator, Conv1d, &self.initial),
            .upsample = undefined,
            .residual = undefined,
            .final = try zml.mem.bufferize(allocator, Conv1d, &self.final),
        };
        for (&self.upsample, &buffers.upsample) |*layer, *target| target.* = try zml.mem.bufferize(allocator, Conv_Transpose_1D, layer);
        for (&self.residual, &buffers.residual) |*layer, *target| target.* = try zml.mem.bufferize(allocator, Resnet_Block, layer);
        return buffers;
    }

    /// `input`: floating [B, 512, S], transformer output transposed to channels
    /// first. Returns [B, 1, S*120] audio samples in input.dtype(), with no output
    /// clipping. Tags .b independent waveforms, .d channels, .s time at each rate.
    pub fn forward(self: SEANet, active_lengths: zml.Tensor, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("active_lengths: {f}\n", .{active_lengths.shape()});
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        var hidden = input;
        debug_print("hidden: {f}\n", .{hidden.shape()});

        std.debug.assert(active_lengths.dtype() == .i32 and active_lengths.rank() == 1 and active_lengths.dim(0) == 2);
        const active_frame_length = active_lengths.slice(0, .single(1)); // [2] i32 -> scalar i32 frame count
        debug_print("active_frame_length: {f}\n", .{active_frame_length.shape()});

        {
            const index = active_frame_length.mul(.constant(.init(.i32, 16))).subConstant(24).maximum(.constant(.init(.i32, 0)));
            debug_print("index: {f}\n", .{index.shape()});
            hidden = hidden.slice(.s, .dyn(index, 24));
            debug_print("hidden: {f}\n", .{hidden.shape()});
        }

        hidden = self.initial.forward(hidden);
        debug_print("hidden: {f}\n", .{hidden.shape()});

        inline for (self.upsample, self.residual, 0..) |upsampler, residual_block, i| {
            hidden = zml.nn.elu(hidden, 1);
            debug_print("hidden: {f}\n", .{hidden.shape()});

            hidden = upsampler.forward(i, hidden);
            debug_print("hidden: {f}\n", .{hidden.shape()});

            hidden = residual_block.forward(i, hidden);
            debug_print("hidden: {f}\n", .{hidden.shape()});
        }

        hidden = zml.nn.elu(hidden, 1);
        debug_print("hidden: {f}\n", .{hidden.shape()});

        hidden = self.final.forward(hidden);
        debug_print("hidden: {f}\n", .{hidden.shape()});

        {
            const index = active_frame_length.subConstant(1).minimum(.constant(.init(.i32, 1)));
            debug_print("index: {f}\n", .{index.shape()});
            hidden = hidden.slice(.s, .dyn(index.mul(.constant(.init(.i32, 960))), 1920));
            debug_print("hidden: {f}\n", .{hidden.shape()});
        }

        debug_print("output: {f}\n", .{hidden.shape()});
        return hidden;
    }
};

/// Mimi's causal, zero-padded, stride-one convolution. Checkpoint weights use
/// [output channels, input channels, kernel taps]; all decoder convolutions have
/// dilation one. The first sample sees zeros to its left, never future samples.
pub const Conv1d = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    /// Expects a PocketTTSAudioTokenizerConv1d checkpoint view (above `conv`).
    pub fn init(store: zml.io.TensorStore.View) Conv1d {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .weight = store.createReplicatedTensor("conv.weight", .{ .out, .in, .k }),
            .bias = store.createReplicatedTensor("conv.bias", .{.d}),
        };
    }

    /// `input`: floating [B, C_in, S], from the preceding SEANet activation.
    /// Returns [B, C_out, S] in input.dtype(). Tags: .b independent sequences,
    /// .d channels, .s time; weight .in is contracted, .out becomes .d, .k taps.
    pub fn forward(self: Conv1d, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const features = input.withTags(.{ .b, .d, .s });
        debug_print("features: {f}\n", .{features.shape()});
        std.debug.assert(features.dtype().isFloat());
        const padded = features.pad(0, .{ .s = zml.Tensor.Pad{ .low = self.weight.dim(.k) - 1 } });
        debug_print("padded: {f}\n", .{padded.shape()});
        const output = self.convolve(padded);
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }

    fn convolve(self: Conv1d, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".convolve";

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".convolve", .{});
        defer compiler.popLocation();

        const y = input.conv1d(self.weight.convert(input.dtype()), .{}).withTags(.{ .b, .d, .s });
        debug_print("y: {f}\n", .{y.shape()});
        const output = y.add(self.bias.convert(input.dtype()).broad(y.shape()));
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

/// SEANet residual branch: ELU, causal K=3 convolution reducing channels by two,
/// ELU, K=1 convolution restoring channels, then the identity skip connection.
pub const Resnet_Block = struct {
    conv1: Conv1d,
    conv2: Conv1d,

    pub fn init(store: zml.io.TensorStore.View) Resnet_Block {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .conv1 = .init(store.withPrefix("block.1")), .conv2 = .init(store.withPrefix("block.3")) };
    }

    /// `input`: floating SEANet features [B, C, S] from an upsampling convolution.
    /// Returns the same shape and dtype, assigning tags .b batch, .d channels, .s time.
    /// `idx` identifies this repeated layer in compiler source locations.
    pub fn forward(self: Resnet_Block, comptime idx: usize, input: zml.Tensor) zml.Tensor {
        const here = std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx });

        debug_enter(here);
        defer debug_exit(here);
        debug_print("input: {f}\n", .{input.shape()});

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx }), .{});
        defer compiler.popLocation();

        const residual = input.withTags(.{ .b, .d, .s });
        debug_print("residual: {f}\n", .{residual.shape()});
        const activated = zml.nn.elu(residual, 1);
        debug_print("activated: {f}\n", .{activated.shape()});
        const reduced = self.conv1.forward(activated);
        debug_print("reduced: {f}\n", .{reduced.shape()});
        const reactivated = zml.nn.elu(reduced, 1);
        debug_print("reactivated: {f}\n", .{reactivated.shape()});
        const restored = self.conv2.forward(reactivated);
        debug_print("restored: {f}\n", .{restored.shape()});
        const output = residual.add(restored);
        debug_print("output: {f}\n", .{output.shape()});
        return output;
    }
};

fn debug_enter(name: []const u8) void {
    if (debug_depth == 0) {
        // Read once per graph traversal. The process does not mutate its environment.
        debug_filter = if (std.c.getenv("CMC_FILTER")) |value| std.mem.span(value) else null;
        if (debug_filter) |filter| {
            if (filter.len == 0) debug_filter = null;
        }
    }
    if (debug_selected_depth == null) {
        if (debug_filter == null or std.mem.eql(u8, debug_filter.?, name)) {
            debug_selected_depth = debug_depth;
        }
    }
    debug_print("vvv {s} vvv\n", .{name});
    debug_depth += 1;
}

fn debug_exit(name: []const u8) void {
    std.debug.assert(debug_depth > 0);
    debug_depth -= 1;
    debug_print("^^^ {s} ^^^\n", .{name});
    if (debug_selected_depth == debug_depth) debug_selected_depth = null;
}

fn debug_print(comptime fmt: []const u8, args: anytype) void {
    const selected_depth = debug_selected_depth orelse return;
    const indent: std.fmt.Alt(usize, format_debug_indent) = .{ .data = debug_depth - selected_depth };
    std.debug.print("{f}" ++ fmt, .{indent} ++ args);
}

fn format_debug_indent(depth: usize, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.splatByteAll(' ', depth * 4);
}
