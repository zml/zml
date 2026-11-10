const std = @import("std");
const zml = @import("zml");

const util = @import("util.zig");
const test_data = @import("transformer_test_data.zig");

test {
    std.testing.refAllDecls(@This());
}

/// Stores the attention history reused between speech-generation steps.
///
/// `k` and `v` are floating-point `[.layer=6, .b=B, .k=C, .h=8, .hd=64]`: one key
/// and value vector per layer, batch row, sequence slot, and attention head.
/// Keys already include their absolute-position RoPE rotation; values do not.
/// C is fixed prompt-plus-generation capacity, rather than the current length.
/// `init` and `initBuffer` allocate f32 history. Attention preserves each cache
/// tensor's dtype and converts selected keys/values to the hidden input's dtype.
///
/// The host allocates finite zero-filled buffers before prefill and carries the
/// updated cache between calls. Attention writes new positions and masks future
/// slots. These tensors are runtime history, separate from checkpoint parameters;
/// the host owns their buffers and manages donation at the prediction boundary.
pub const KV_Cache = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub fn init(batch: i64, capacity: i64) KV_Cache {
        std.debug.assert(batch > 0 and capacity > 0);
        const shape = zml.Shape.init(.{ .layer = 6, .b = batch, .k = capacity, .h = 8, .hd = 64 }, .f32);
        return .{ .k = .fromShape(shape), .v = .fromShape(shape) };
    }

    pub fn initBuffer(self: KV_Cache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KV_Cache) {
        const zeros = try zml.Slice.alloc(allocator, self.k.shape());
        defer zeros.free(allocator);
        @memset(zeros.items(f32), 0);
        var k = try zml.Buffer.fromSlice(io, platform, zeros);
        errdefer k.deinit();
        const v = try zml.Buffer.fromSlice(io, platform, zeros);
        return .{ .k = k, .v = v };
    }

    pub fn deinitBuffer(buffers: *zml.Bufferized(KV_Cache)) void {
        zml.Buffer.deinitAll(KV_Cache, buffers);
    }

    pub fn reuseBuffer(self: KV_Cache, origin: KV_Cache) KV_Cache {
        const k = self.k.reuseBuffer(origin.k); // [6,B,C,8,64] f32 -> same shape/dtype, donating origin K
        const v = self.v.reuseBuffer(origin.v); // [6,B,C,8,64] f32 -> same shape/dtype, donating origin V
        return .{ .k = k, .v = v };
    }
};

/// Returns transformer features together with their updated attention history.
///
/// `hidden` is floating-point `[.b=B, .s=Q, .d=512]`, one feature vector per newly
/// processed position. Q is the prompt length during prefill and 1 during decode.
/// Decoder layers return residual features; the model applies final normalization
/// before returning this type. `cache` retains the fixed-capacity K/V tensors
/// for all six layers, including the positions processed by this call. Attention
/// preserves the hidden input's dtype and each cache tensor's dtype; model entry
/// points currently use f32.
pub const CachedHidden = struct {
    hidden: zml.Tensor,
    cache: KV_Cache,
};

// --- Gradium_Phonon_TTS_Model ---
// The complete transformer, which runs multiple passes of the inner transformer block (aka Gradium_Phonon_TTS_Decoder_Layer).

/// Turns text, voice, and generated audio into contextual transformer features.
///
/// Token embeddings are replaced at special positions by projected voice frames,
/// a learned audio-start vector, or completed audio latents. Six decoder layers
/// mix context using causal attention, then final LayerNorm produces the features
/// used by the EOS and flow heads. Each position can see itself and earlier
/// positions in the same unpadded sequence.
///
/// `forward` processes the complete prefix and audio history on every call.
/// `forward_prefill` processes the prompt once and fills a K/V cache;
/// `forward_decode` consumes only the last completed audio frame and uses that
/// cache for earlier context. Both cached paths return features and updated history.
/// The layer-loop methods accept already prepared features; only the cached loop
/// also applies the model's final normalization.
///
/// Inputs include integer IDs `[B, S]`, f32 voice features `[B, P, 512]`, and
/// f32 normalized audio history `[B, A, 32]`. Output features are f32 `[B, S, 512]`
/// for full-prefix execution, or `[B, Q, 512]` for Q new cached positions.
/// B counts sequences, P voice frames, and A completed generated frames.
///
/// Parameters come from the root checkpoint view: transcript embeddings, input
/// projections, `flow_lm.transformer.layers.0..5`, and `flow_lm.out_norm`.
/// Stored bf16 tensors are used in f32. This port targets phonon-31m-en with
/// 8 attention heads of 64 features, RoPE theta 10000, and norm epsilon 1e-5.
///
/// Reference:
/// ```
/// model: GradiumPhononTTSModel
/// ├── embed_tokens: Embedding
/// │   └── weight: shape=(4001, 512), dtype=torch.float32
/// │       ├── loaded: model.embed_tokens.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight
/// ├── layers: ModuleList
/// │   └── 0..5: GradiumPhononTTSDecoderLayer
/// │       ├── self_attn: GradiumPhononTTSAttention
/// │       │   ├── q_proj: Linear
/// │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       │   │       ├── loaded: model.layers.5.self_attn.q_proj.weight
/// │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │       │   ├── k_proj: Linear
/// │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       │   │       ├── loaded: model.layers.5.self_attn.k_proj.weight
/// │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │       │   ├── v_proj: Linear
/// │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       │   │       ├── loaded: model.layers.5.self_attn.v_proj.weight
/// │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │       │   └── o_proj: Linear
/// │       │       └── weight: shape=(512, 512), dtype=torch.float32
/// │       │           ├── loaded: model.layers.5.self_attn.o_proj.weight
/// │       │           └── checkpoint: flow_lm.transformer.layers.5.self_attn.out_proj.weight
/// │       ├── mlp: GradiumPhononTTSMLP
/// │       │   ├── activation_fn: GELUActivation
/// │       │   ├── fc1: Linear
/// │       │   │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │       │   │       ├── loaded: model.layers.5.mlp.fc1.weight
/// │       │   │       └── checkpoint: flow_lm.transformer.layers.5.linear1.weight
/// │       │   └── fc2: Linear
/// │       │       └── weight: shape=(512, 2048), dtype=torch.float32
/// │       │           ├── loaded: model.layers.5.mlp.fc2.weight
/// │       │           └── checkpoint: flow_lm.transformer.layers.5.linear2.weight
/// │       ├── input_layernorm: LayerNorm
/// │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │       │   │   ├── loaded: model.layers.5.input_layernorm.weight
/// │       │   │   └── checkpoint: flow_lm.transformer.layers.5.norm1.weight
/// │       │   └── bias: shape=(512,), dtype=torch.float32
/// │       │       ├── loaded: model.layers.5.input_layernorm.bias
/// │       │       └── checkpoint: flow_lm.transformer.layers.5.norm1.bias
/// │       └── post_attention_layernorm: LayerNorm
/// │           ├── weight: shape=(512,), dtype=torch.float32
/// │           │   ├── loaded: model.layers.5.post_attention_layernorm.weight
/// │           │   └── checkpoint: flow_lm.transformer.layers.5.norm2.weight
/// │           └── bias: shape=(512,), dtype=torch.float32
/// │               ├── loaded: model.layers.5.post_attention_layernorm.bias
/// │               └── checkpoint: flow_lm.transformer.layers.5.norm2.bias
/// ├── norm: LayerNorm
/// │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   ├── loaded: model.norm.weight
/// │   │   └── checkpoint: flow_lm.out_norm.weight
/// │   └── bias: shape=(512,), dtype=torch.float32
/// │       ├── loaded: model.norm.bias
/// │       └── checkpoint: flow_lm.out_norm.bias
/// ├── rotary_emb: GradiumPhononTTSRotaryEmbedding
/// ├── audio_prompt_proj: Linear
/// │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       ├── loaded: model.audio_prompt_proj.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight
/// ├── audio_latent_proj: Linear
/// │   └── weight: shape=(512, 32), dtype=torch.float32
/// │       ├── loaded: model.audio_latent_proj.weight
/// │       └── checkpoint: flow_lm.input_linear.weight
/// ├── num_speakers_embed: Embedding
/// │   └── weight: shape=(32, 16), dtype=torch.float32
/// │       ├── loaded: model.num_speakers_embed.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.num_speakers.embed.weight
/// ├── num_speakers_proj: Linear
/// │   └── weight: shape=(512, 16), dtype=torch.float32
/// │       ├── loaded: model.num_speakers_proj.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight
/// ├── duration_delta_proj: Linear
/// │   └── weight: shape=(512, 128), dtype=torch.float32
/// │       ├── loaded: model.duration_delta_proj.weight
/// │       └── checkpoint: flow_lm.condition_provider.conditioners.duration_delta.output_proj.weight
/// └── bos_latent: shape=(32,), dtype=torch.float32
///     ├── loaded: model.bos_latent
///     └── checkpoint: flow_lm.bos_emb
/// ```
pub const Gradium_Phonon_TTS_Model = struct {
    embed_tokens: Embedding,
    input_projection: Input_Projection,
    layers: [num_layers]Gradium_Phonon_TTS_Decoder_Layer,
    norm: zml.nn.LayerNorm,

    // FIX: This port currently targets the six-layer phonon-31m-en checkpoint (here and everywhere else).
    const num_layers = 6;

    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Model {
        var self: Gradium_Phonon_TTS_Model = .{
            .embed_tokens = .init(store.withPrefix("flow_lm.condition_provider.conditioners.transcript_in_segment.embed")),
            .input_projection = .init(store),
            .layers = undefined,
            .norm = .{
                .weight = store.createReplicatedTensor("flow_lm.out_norm.weight", .{.d}),
                .bias = store.createReplicatedTensor("flow_lm.out_norm.bias", .{.d}),
                .eps = 1e-5,
            },
        };
        for (&self.layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("flow_lm.transformer.layers").withLayer(i));
        }
        return self;
    }

    /// Runs input preparation, all six decoder layers, and final normalization.
    ///
    /// `token_ids`: integer `[B, S]`, normally i64, the complete unpadded sequence
    /// from the tokenizer and generation loop. Contains text and control IDs;
    /// each completed audio frame adds one ID 6. S must be nonzero.
    /// `voice`: f32 `[B, P, 512]`, ordered speaker features loaded from the voice
    /// asset and transposed by the caller from `[B, 512, P]`. Each row has P ID-5 markers.
    /// `audio_latents`: f32 `[B, A, 32]`, the generation loop's complete normalized
    /// frame history, ordered like the A ID-6 markers. Initially A=0.
    /// All inputs share batch size B; positions start at zero on every call.
    ///
    /// Returns f32 `[B, S, 512]`, final normalized contextual features for every
    /// position. The prediction wrapper selects the last position for both heads.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent sequences, shared by token IDs and voice frames.
    /// - `.s`: sequence; token positions, preserved in the output.
    /// - `.p`: prompt frames; ordered voice frames inserted at audio-prompt markers.
    /// - `.ad`: audio-prompt features; 512 input features per voice frame.
    /// - `.a`: completed generated frames, in marker order; zero initially.
    /// - `.ld`: 32 normalized latent features per completed frame.
    /// - `.d`: model dimension; 512 output features per token, normalized by the final norm.
    /// Internal projection/attention tags are documented in their forward methods.
    ///
    /// Inputs: integer IDs `[.b, .s]`, f32 voice `[.b, .p, .ad=512]`,
    /// and f32 complete generated history `[.b, .a, .ld=32]`.
    /// Output: f32 `[.b, .s, .d=512]`.
    pub fn forward(self: Gradium_Phonon_TTS_Model, token_ids: zml.Tensor, voice: zml.Tensor, audio_latents: zml.Tensor) zml.Tensor {
        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Model", .{});
        defer compiler.popLocation();

        const embeddings = self.embed_tokens.forward(token_ids).convert(.f32); // [B,S] integer -> [B,S,d=512] f32
        const prepared = self.input_projection.forward(embeddings, token_ids, voice, audio_latents); // [B,S,512] f32, [B,S] integer, [B,P,512] f32, [B,A,32] f32 -> [B,S,512] f32
        const hidden = self.forward_layers(prepared); // [B,S,512] f32 -> [B,S,512] f32

        const weight = self.norm.weight.convert(.f32); // [d=512] bf16 -> [d=512] f32
        const bias = self.norm.bias.?.convert(.f32); // [d=512] bf16 -> [d=512] f32
        const norm: zml.nn.LayerNorm = .{ .weight = weight, .bias = bias, .eps = self.norm.eps };
        const output = norm.forward(hidden); // [B,S,512] f32 -> [B,S,512] f32
        return output;
    }

    /// Runs the six decoder layers on already prepared sequence features.
    ///
    /// `input`: f32 `[B, S, 512]`, output of this model's `input_projection`,
    /// combining text embeddings with voice, audio-start, and generated-frame
    /// replacements. This method assigns tags `.b`, `.s`, and `.d`.
    ///
    /// Returns f32 `[B, S, 512]`, the sixth layer's residual features. The caller
    /// applies the final model norm. This uncached path starts positions at zero
    /// and recomputes the full unpadded sequence.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent sequences processed together, never mixed.
    /// - `.s`: sequence; token positions. Each layer's causal attention can use
    ///   the current and earlier positions within the same sequence.
    /// - `.d`: model dimension; 512 features describing each token, transformed
    ///   by every layer while retaining the same width for residual additions.
    /// Internal attention and MLP tags are documented in their forward methods.
    ///
    /// Shape: `[.b, .s, .d=512] -> [.b, .s, .d=512]`, with f32 input and output.
    pub fn forward_layers(self: Gradium_Phonon_TTS_Model, input: zml.Tensor) zml.Tensor {
        std.debug.assert(input.dtype() == .f32);
        std.debug.assert(input.rank() == 3);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Model", .{});
        defer compiler.popLocation();

        var hidden_states = input.withTags(.{ .b, .s, .d }); // [B,S,512] f32 -> [b=B,s=S,d=512] f32
        for (self.layers, 0..) |layer, i| {
            compiler.pushLocationFmt(@src(), "layer[{d}]", .{i});
            hidden_states = layer.forward(hidden_states); // [B,S,512] f32 -> [B,S,512] f32
            compiler.popLocation();
        }
        return hidden_states;
    }

    /// Processes the initial prompt and fills its attention-cache prefix.
    ///
    /// `token_ids`: integer `[B, S]`, normally i64, the tokenizer's complete
    /// unpadded prompt ending at audio BOS (ID 4), passed by the prediction wrapper.
    /// `voice`: f32 `[B, P, 512]`, ordered features from the selected voice asset,
    /// transposed by the host and passed by the wrapper; one frame per ID-5 marker.
    /// `cache`: host-allocated K/V storage, each f32 `[6, B, C, 8, 64]`, initially
    /// filled with finite zeros and with C >= S. No generated history exists yet.
    ///
    /// Returns final normalized f32 hidden features `[B, S, 512]` and the cache
    /// with prompt positions 0..S-1 written for all layers. The wrapper selects
    /// the last position to condition the first audio-frame prediction.
    ///
    /// Dimension tags:
    /// - `.b`: sequences; `.s`: S prompt positions; `.d`: 512 hidden features.
    /// - `.p`: P voice frames; `.ad`: 512 voice features.
    /// - Cache: `.layer`=6, `.k`=C slots, `.h`=8 heads, `.hd`=64 head features.
    ///
    /// Dimension tags:
    /// - `.b`: sequences; `.s`: S prompt positions; `.d`: 512 hidden features.
    /// - `.p`: P voice frames; `.ad`: 512 voice features.
    /// - Cache: `.layer`=6, `.k`=C slots, `.h`=8 heads, `.hd`=64 head features.
    ///
    /// Dimension tags:
    /// - `.b`: sequences; `.s`: S prompt positions; `.d`: 512 hidden features.
    /// - `.p`: P voice frames; `.ad`: 512 voice features.
    /// - Cache: `.layer`=6, `.k`=C slots, `.h`=8 heads, `.hd`=64 head features.
    pub fn forward_prefill(self: Gradium_Phonon_TTS_Model, token_ids: zml.Tensor, voice: zml.Tensor, cache: KV_Cache) CachedHidden {
        const embeddings = self.embed_tokens.forward(token_ids).convert(.f32); // [B,P] integer -> [B,P,d=512] f32
        const empty = zml.Tensor.constantTensor(.init(.{ .b = token_ids.dim(0), .a = 0, .ld = 32 }, .f32), &.{}); // empty host data -> [B,0,32] f32
        const prepared = self.input_projection.forward(embeddings, token_ids, voice, empty); // [B,P,512], IDs, voice, empty history -> [B,P,512] f32
        const position = zml.Tensor.scalar(0, .i32); // host zero -> [] i32
        return self.forward_cached_layers(prepared, position, cache); // [B,P,512] + cache f32, [] i32 -> hidden + updated cache f32
    }

    /// Consumes the last completed audio frame using cached earlier context.
    ///
    /// `frame`: f32 `[B, 1, 32]`, the latest normalized audio latent produced by
    /// the sampler, selected from output history by the generation loop. The model
    /// creates a single ID-6 marker and projects this frame into transformer features.
    /// `position`: scalar i32 `[]`, the first new absolute sequence slot, supplied
    /// by the generation loop (0 for prefill, then the next unprocessed slot).
    /// It offsets RoPE and the causal mask; the host must ensure 0 <= position <= C-1.
    /// `cache`: runtime K/V history supplied by the caller, with each f32 tensor
    /// shaped `[.layer=6, .b=B, .k=C, .h=8, .hd=64]`. C is fixed capacity;
    /// keys include RoPE and values do not. Earlier prefix slots must already be
    /// populated, and unused slots must contain finite values.
    ///
    /// Returns final normalized f32 features `[B, 1, 512]` and updated cache,
    /// writing this position in all layers. The wrapper uses these features to
    /// predict the next frame; prior text, voice, and audio are already cached.
    ///
    /// Dimension tags:
    /// - Frame: `.b`=B sequences, `.a`=1 new audio frame, `.ld`=32 latent coordinates.
    /// - Hidden: `.b`, `.s`=1 position, `.d`=512 features.
    /// - Cache: `.layer`=6, `.k`=C slots, `.h`=8 heads, `.hd`=64 head features.
    ///
    /// Dimension tags:
    /// - Frame: `.b`=B sequences, `.a`=1 new audio frame, `.ld`=32 latent coordinates.
    /// - Hidden: `.b`, `.s`=1 position, `.d`=512 features.
    /// - Cache: `.layer`=6, `.k`=C slots, `.h`=8 heads, `.hd`=64 head features.
    ///
    /// Dimension tags:
    /// - Frame: `.b`=B sequences, `.a`=1 new audio frame, `.ld`=32 latent coordinates.
    /// - Hidden: `.b`, `.s`=1 position, `.d`=512 features.
    /// - Cache: `.layer`=6, `.k`=C slots, `.h`=8 heads, `.hd`=64 head features.
    pub fn forward_decode(self: Gradium_Phonon_TTS_Model, frame: zml.Tensor, position: zml.Tensor, cache: KV_Cache) CachedHidden {
        std.debug.assert(frame.rank() == 3 and frame.dim(1) == 1 and frame.dim(2) == 32);
        const ids_shape = zml.Shape.init(.{ .b = frame.dim(0), .s = 1 }, .i64);
        const marker = zml.Tensor.scalar(6, .i64); // host marker -> [] i64
        const ids = marker.broad(ids_shape); // [] i64 -> [B,s=1] i64
        const embeddings = self.embed_tokens.forward(ids).convert(.f32); // [B,1] i64 -> [B,1,d=512] f32
        const voice = zml.Tensor.constantTensor(.init(.{ .b = frame.dim(0), .p = 0, .ad = 512 }, .f32), &.{}); // empty host data -> [B,0,512] f32
        const prepared = self.input_projection.forward(embeddings, ids, voice, frame); // [B,1,512], IDs, empty voice, [B,1,32] -> [B,1,512] f32
        return self.forward_cached_layers(prepared, position, cache); // [B,1,512] + cache f32, [] i32 -> hidden + updated cache f32
    }

    /// Runs the six cached decoder layers and the model's final normalization.
    ///
    /// `input`: f32 `[B, Q, 512]`, prepared features from `input_projection`:
    /// the full prompt during prefill, or one projected audio frame during decode.
    /// `position`: scalar i32 `[]`, the first new absolute sequence slot, supplied
    /// by the generation loop (0 for prefill, then the next unprocessed slot).
    /// It offsets RoPE and the causal mask; the host must ensure 0 <= position <= C-Q.
    /// `cache`: runtime K/V history supplied by the caller, with each f32 tensor
    /// shaped `[.layer=6, .b=B, .k=C, .h=8, .hd=64]`. C is fixed capacity;
    /// keys include RoPE and values do not. Earlier prefix slots must already be
    /// populated, and unused slots must contain finite values.
    ///
    /// Returns final normalized f32 hidden features `[B, Q, 512]` and the cache
    /// updated at positions position..position+Q-1 for all six layers.
    ///
    /// Dimension tags:
    /// - `.b`: independent sequences; `.s`: Q new positions; `.d`: 512 features.
    /// - `.layer`: six decoder layers; `.k`: C cache slots; `.h`: eight heads;
    ///   `.hd`: 64 features per head. Cache capacity is separate from Q.
    pub fn forward_cached_layers(self: Gradium_Phonon_TTS_Model, input: zml.Tensor, position: zml.Tensor, cache: KV_Cache) CachedHidden {
        var hidden = input;
        var updated = cache;
        for (self.layers, 0..) |layer, index| {
            const result = layer.forward_cached(hidden, position, updated, index); // [B,Q,512] + cache f32, [] i32 -> hidden + updated cache f32
            hidden = result.hidden;
            updated = result.cache;
        }
        const weight = self.norm.weight.convert(.f32); // [d=512] bf16 -> [d=512] f32
        const bias = self.norm.bias.?.convert(.f32); // [d=512] bf16 -> [d=512] f32
        const norm: zml.nn.LayerNorm = .{ .weight = weight, .bias = bias, .eps = self.norm.eps };
        const output = norm.forward(hidden); // [B,Q,512] f32 -> [B,Q,512] f32
        return .{ .hidden = output, .cache = updated };
    }

    pub fn load(
        self: *const Gradium_Phonon_TTS_Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Model) {
        // bufferize does not traverse arrays, so load each layer separately.
        var buffers: zml.Bufferized(Gradium_Phonon_TTS_Model) = undefined;
        buffers.embed_tokens = try self.embed_tokens.load(allocator, io, platform, store);
        errdefer Embedding.unloadBuffers(&buffers.embed_tokens);
        buffers.input_projection = try self.input_projection.load(allocator, io, platform, store);
        errdefer Input_Projection.unloadBuffers(&buffers.input_projection);

        buffers.norm = try zml.mem.bufferize(allocator, zml.nn.LayerNorm, &self.norm);
        errdefer zml.Buffer.deinitAll(zml.nn.LayerNorm, &buffers.norm);
        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();
        try loader.load(io, zml.nn.LayerNorm, &self.norm, &buffers.norm, store, .{});
        try loader.await(io);

        var loaded: usize = 0;
        errdefer for (buffers.layers[0..loaded]) |*layer| {
            Gradium_Phonon_TTS_Decoder_Layer.unloadBuffers(layer);
        };

        for (&self.layers, &buffers.layers) |*layer, *layer_buffers| {
            layer_buffers.* = try layer.load(allocator, io, platform, store);
            loaded += 1;
        }

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Model)) void {
        Embedding.unloadBuffers(&self.embed_tokens);
        Input_Projection.unloadBuffers(&self.input_projection);
        zml.Buffer.deinitAll(zml.nn.LayerNorm, &self.norm);
        for (&self.layers) |*layer| {
            Gradium_Phonon_TTS_Decoder_Layer.unloadBuffers(layer);
        }
    }
};

test Gradium_Phonon_TTS_Model {
    const allocator = std.testing.allocator;
    const bazel_builtin = @import("bazel_builtin");
    const runfiles = try zml.bazel.runfiles(bazel_builtin.current_repository);

    for (0..4) |index| {
        const key = try std.fmt.allocPrint(
            allocator,
            "zml/examples/phonon/assets/model.call.{d}.forward.activations.safetensors",
            .{index},
        );
        defer allocator.free(key);
        const activations_path = try runfiles.rlocationAlloc(allocator, key) orelse return error.ActivationsNotFound;
        defer allocator.free(activations_path);
        try util.test_module_inputs(
            Gradium_Phonon_TTS_Model,
            util.gradium_phonon_31m_en,
            activations_path,
            "",
            .{ "input_ids", "audio_prompt_latents", "audio_latents" },
            .forward,
        );
    }
}

// --- Embedding ---

/// Looks up the initial feature vector for each text or control token.
///
/// Input: integer IDs `[B, S]` in [0, 4001), normally i64 from the tokenizer
/// and generation loop. Output: `[B, S, 512]` in the table's floating-point dtype,
/// one learned table row per ID. These features have not yet mixed sequence
/// context. The input projection
/// subsequently replaces voice, audio-start, and generated-audio positions.
///
/// The bf16 vocabulary table `[4001, 512]` comes from
/// `flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight`.
/// Lookup gathers stored rows and preserves the table's dtype.
///
/// Reference:
/// ```
/// embed_tokens: Embedding
/// └── weight: shape=(4001, 512), dtype=torch.float32
///     ├── loaded: model.embed_tokens.weight
///     └── checkpoint: flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight
/// ```
const Embedding = struct {
    weight: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Embedding {
        return .{
            .weight = store.createReplicatedTensor("weight", .{ .voc, .d }),
        };
    }

    // -- Description --

    /// Looks up one learned feature vector per text or control token.
    ///
    /// `input`: integer `[B, S]`, normally i64, IDs from the tokenizer and
    /// generation loop, passed by the model. Every value must be in [0, 4001).
    /// This method assigns tags `.b` and `.s`.
    ///
    /// Returns `[B, S, 512]` in `weight.dtype()`, raw embeddings passed to
    /// `Input_Projection` for voice/audio replacements before transformer context mixing.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent sequences of token IDs.
    /// - `.s`: sequence; token positions, preserved in the output.
    /// - `.voc`: vocabulary; the 4001 table rows indexed by the input IDs.
    ///   Gathering replaces this axis with the input's `.b` and `.s` axes.
    /// - `.d`: model dimension; the 512 features stored in each selected row.
    ///
    /// Shape: integer IDs `[.b, .s]` -> floating-point embeddings `[.b, .s, .d=512]`.
    pub fn forward(self: Embedding, input: zml.Tensor) zml.Tensor {
        std.debug.assert(input.dtype().isInteger());
        std.debug.assert(self.weight.dtype().isFloat());
        std.debug.assert(input.rank() == 2);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Embedding", .{});
        defer compiler.popLocation();

        const ids = input.withTags(.{ .b, .s }); // [B,S] integer -> [b=B,s=S] integer
        return self.weight.gather(.{ .voc = ids }, .{}); // [voc=4001,d=512] + [B,S] integer indices -> [B,S,d=512], preserving weight dtype
    }

    // -- Runtime --

    pub fn load(
        self: *const Embedding,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Embedding) {
        var buffers = try zml.mem.bufferize(allocator, Embedding, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Embedding, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Embedding)) void {
        zml.Buffer.deinitAll(Embedding, self);
    }
};

test Embedding {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Embedding = .{
        .weight = .fromShape(.init(.{ .voc = 4001, .d = 512 }, .bf16)),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2 }, .i64);
    var input: [2][2]i64 = undefined;
    for (&input) |*batch| {
        for (batch) |*id| id.* = random.intRangeLessThan(i64, 0, 4001);
    }
    const weights = try zml.Slice.alloc(allocator, layer.weight.shape());
    defer weights.free(allocator);
    for (weights.items(zml.floats.BFloat16)) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);

    var weight_buffer = try zml.Buffer.fromSlice(io, platform, weights);
    defer weight_buffer.deinit();
    const weight_buffers: zml.Bufferized(Embedding) = .{ .weight = weight_buffer };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.embedding;
    const output_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, .bf16);
    const expected_slice = try zml.Slice.alloc(allocator, output_shape);
    defer expected_slice.free(allocator);
    for (expected_slice.items(zml.floats.BFloat16), std.mem.bytesAsSlice(f32, std.mem.asBytes(&expected))) |*value, recorded| {
        value.* = .fromF32(recorded);
    }
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        const typed_layer: Embedding = .{
            .weight = .fromShape(layer.weight.shape().withDtype(dtype)),
        };
        var typed_exe = try platform.compile(allocator, io, typed_layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(.i32)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(output_shape.withDtype(dtype)));
    }
}

// --- Input_Projection ---

/// Builds transformer inputs by inserting voice and audio into token embeddings.
///
/// Text positions retain their embedding. ID 5 receives the corresponding projected
/// voice frame; ID 4 receives the projected learned audio-start latent; ID 6
/// receives the corresponding projected generated frame. Audio-start and generated
/// positions also receive speaker-count and duration conditioning. Frame order
/// must match marker order separately in each unpadded sequence.
///
/// Inputs: floating-point embeddings `[B, S, 512]`, integer IDs `[B, S]`, floating-point voice
/// `[B, P, 512]`, and floating-point normalized audio history `[B, A, 32]`.
/// Output: `[B, S, 512]` in the embeddings' dtype, ready for the first decoder layer.
/// P and A may be zero. Cached decode supplies only one ID-6 position and frame.
///
/// Parameters come from the root checkpoint view: speaker feature projection
/// `flow_lm.condition_provider.conditioners.speaker_wavs.output_proj`, audio
/// projection `flow_lm.input_linear`, learned latent `flow_lm.bos_emb`, and the
/// `num_speakers` and `duration_delta` conditioners. All projections are bias-free;
/// stored bf16 tensors, voice, and audio history are converted to the embeddings' dtype.
/// Host fields default to one speaker and
/// zero duration delta; the latter generates 128 sinusoidal features in host f32,
/// then converts them to the embeddings' dtype.
const Input_Projection = struct {
    audio_prompt_proj: zml.nn.Linear,
    audio_latent_proj: zml.nn.Linear,
    bos_latent: zml.Tensor,
    num_speakers_embed: zml.Tensor,
    num_speakers_proj: zml.nn.Linear,
    duration_delta_proj: zml.nn.Linear,

    num_speakers: i64 = 1,
    duration_delta: f32 = 0,

    pub fn init(store: zml.io.TensorStore.View) Input_Projection {
        return .{
            .audio_prompt_proj = .init(store.createReplicatedTensor(
                "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
                .{ .d, .ad },
            ), null, .ad),
            .audio_latent_proj = .init(store.createReplicatedTensor("flow_lm.input_linear.weight", .{ .d, .ld }), null, .ld),
            .bos_latent = store.createReplicatedTensor("flow_lm.bos_emb", .{.ld}),
            .num_speakers_embed = store.createReplicatedTensor(
                "flow_lm.condition_provider.conditioners.num_speakers.embed.weight",
                .{ .voc, .spk },
            ),
            .num_speakers_proj = .init(store.createReplicatedTensor(
                "flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight",
                .{ .d, .spk },
            ), null, .spk),
            .duration_delta_proj = .init(store.createReplicatedTensor(
                "flow_lm.condition_provider.conditioners.duration_delta.output_proj.weight",
                .{ .d, .dur },
            ), null, .dur),
        };
    }

    // -- Description --

    /// Replaces control-token embeddings with the corresponding voice/audio features.
    ///
    /// `input`: floating-point `[B, S, 512]`, raw token embeddings from `Embedding.forward`.
    /// `token_ids`: integer `[B, S]`, normally i64, the same tokenizer/generation
    /// IDs used for that lookup. IDs 5, 4, and 6 select voice, audio-start, and
    /// generated-frame replacements; other positions retain `input`.
    /// `voice`: floating-point `[B, P, 512]`, ordered speaker features from the loaded voice
    /// asset, passed by the model. Each row must have exactly P ID-5 markers.
    /// `audio_latents`: floating-point `[B, A, 32]`, normalized completed frames from the
    /// generation loop, in ID-6 marker order with exactly A markers per row.
    /// The full-prefix path passes the whole history; prefill passes A=0, and
    /// cached decode passes one frame with one ID-6 position and P=0.
    /// All inputs share B and are unpadded; P and A may be zero.
    ///
    /// Returns `[B, S, 512]` in `input.dtype()`, prepared input features for the first
    /// decoder layer. Voice, audio history, and parameters are converted to this dtype.
    /// Projected BOS and generated frames include speaker/duration context.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent sequences, shared by IDs, embeddings and voices.
    /// - `.s`: sequence; token positions. IDs select which positions to replace.
    /// - `.d`: model dimension; 512 features at each output sequence position.
    /// - `.p`: prompt frames; voice-frame order, gathered into ID 5 positions.
    /// - `.ad`: input audio-prompt features; 512 features contracted by voice projection.
    /// - `.a`: generated audio frames; ordered history, gathered into ID 6 positions.
    /// - `.ld`: latent dimension; 32 features in BOS and each generated frame.
    /// - `.voc`: speaker-count vocabulary; the row selected by num_speakers.
    /// - `.spk`: speaker-count embedding dimension; 16 features projected to `.d`.
    /// - `.dur`: duration embedding dimension; 128 sinusoidal features projected to `.d`.
    ///
    /// Inputs: floating-point embeddings `[.b, .s, .d]`, integer IDs `[.b, .s]`,
    /// floating-point voice frames `[.b, .p, .ad]` and latents `[.b, .a, .ld=32]`.
    /// Output: `[.b, .s, .d]` in `input.dtype()`.
    pub fn forward(
        self: Input_Projection,
        input: zml.Tensor,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        audio_latents: zml.Tensor,
    ) zml.Tensor {
        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and input.rank() == 3);
        std.debug.assert(token_ids.dtype().isInteger() and token_ids.rank() == 2);
        std.debug.assert(voice.dtype().isFloat() and voice.rank() == 3);
        std.debug.assert(audio_latents.dtype().isFloat() and audio_latents.rank() == 3);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Input_Projection", .{});
        defer compiler.popLocation();

        // T denotes input.dtype() below.
        const x = input.withTags(.{ .b, .s, .d }); // [B,S,512] T -> [b=B,s=S,d=512] T
        const ids = token_ids.withTags(.{ .b, .s }); // [B,S] integer -> [b=B,s=S] integer
        const frames = voice.convert(dtype).withTags(.{ .b, .p, .ad }); // [B,P,512] float -> [b=B,p=P,ad=512] T
        const latents = audio_latents.convert(dtype).withTags(.{ .b, .a, .ld }); // [B,A,32] float -> [b=B,a=A,ld=32] T
        std.debug.assert(x.dim(.b) == ids.dim(.b) and x.dim(.s) == ids.dim(.s));
        std.debug.assert(x.dim(.b) == frames.dim(.b));
        std.debug.assert(x.dim(.b) == latents.dim(.b) and latents.dim(.ld) == 32);

        const speaker_count = zml.Tensor.constant(.{ .i64 = self.num_speakers }); // host speaker count -> [] i64
        const speaker_row = self.num_speakers_embed.gather(.{ .voc = speaker_count }, .{}); // [voc=32,spk=16] bf16 + [] i64 index -> [spk=16] bf16
        const speaker = speaker_row.convert(dtype); // [spk=16] bf16 -> [spk=16] T
        const speaker_condition = self.num_speakers_proj.forward(speaker, dtype); // [spk=16] T -> [d=512] T

        // Construct fixed duration conditioning while tracing, not once per token.
        var duration_features: [128]f32 = undefined;
        for (0..64) |i| {
            const exponent = @as(f32, @floatFromInt(i)) / 63.0;
            const phase = 1000.0 * self.duration_delta / std.math.pow(f32, 10000.0, exponent);
            duration_features[i] = @cos(phase);
            duration_features[i + 64] = @sin(phase);
        }
        const duration = zml.Tensor.constantTensor(.init(.{ .dur = 128 }, .f32), std.mem.asBytes(&duration_features)).convert(dtype); // host [128] f32 -> [dur=128] T
        const duration_condition = self.duration_delta_proj.forward(duration, dtype); // [dur=128] T -> [d=512] T
        const condition = speaker_condition.add(duration_condition); // [d=512] T + [d=512] T -> [d=512] T

        const bos_latent = self.bos_latent.convert(dtype); // [ld=32] bf16 -> [ld=32] T
        const projected_bos = self.audio_latent_proj.forward(bos_latent, dtype); // [ld=32] T -> [d=512] T
        const bos = projected_bos.add(condition); // [d=512] T + [d=512] T -> [d=512] T
        const bos_id = zml.Tensor.scalar(4, ids.dtype()); // host 4 -> [] integer
        const bos_mask = ids.cmp(.EQ, bos_id); // [B,S] integer == [] integer -> [B,S] bool
        const broadcast_bos_mask = bos_mask.broad(x.shape()); // [B,S] bool -> [B,S,d=512] bool
        const broadcast_bos = bos.broad(x.shape()); // [d=512] T -> [B,S,d=512] T
        var output = broadcast_bos_mask.select(broadcast_bos, x); // [B,S,512] bool, 2 x [B,S,512] T -> [B,S,512] T

        // Skip empty histories: there is no frame to gather from a zero-length axis.
        if (frames.dim(.p) > 0) {
            const marker = zml.Tensor.scalar(5, ids.dtype()); // host 5 -> [] integer
            const mask = ids.cmp(.EQ, marker); // [B,S] integer == [] integer -> [B,S] bool
            // Occurrence order pairs markers with frames; pre-marker indices are masked out.
            const counts = mask.convert(.i32); // [B,S] bool -> [B,S] i32
            const cumulative = counts.cumulativeSum(.s); // [B,S] i32 -> [B,S] i32
            const zero_based = cumulative.subConstant(1); // [B,S] i32 - scalar i32 -> [B,S] i32
            const zero = zml.Tensor.scalar(0, .i32); // host 0 -> [] i32
            const indices = zero_based.maximum(zero); // [B,S] i32, [] i32 -> [B,S] i32
            const projected = self.audio_prompt_proj.forward(frames, dtype); // [B,p=P,ad=512] T -> [B,p=P,d=512] T
            const replacements = projected.gather(.{ .p = indices }, .{}); // [B,p=P,512] T + [B,S] i32 indices -> [B,S,512] T
            const broadcast_mask = mask.broad(x.shape()); // [B,S] bool -> [B,S,d=512] bool
            output = broadcast_mask.select(replacements, output); // [B,S,512] bool, 2 x [B,S,512] T -> [B,S,512] T
        }

        // Skip empty histories: there is no frame to gather from a zero-length axis.
        if (latents.dim(.a) > 0) {
            const marker = zml.Tensor.scalar(6, ids.dtype()); // host 6 -> [] integer
            const mask = ids.cmp(.EQ, marker); // [B,S] integer == [] integer -> [B,S] bool
            // Occurrence order pairs markers with frames; pre-marker indices are masked out.
            const counts = mask.convert(.i32); // [B,S] bool -> [B,S] i32
            const cumulative = counts.cumulativeSum(.s); // [B,S] i32 -> [B,S] i32
            const zero_based = cumulative.subConstant(1); // [B,S] i32 - scalar i32 -> [B,S] i32
            const zero = zml.Tensor.scalar(0, .i32); // host 0 -> [] i32
            const indices = zero_based.maximum(zero); // [B,S] i32, [] i32 -> [B,S] i32
            const projected = self.audio_latent_proj.forward(latents, dtype); // [B,a=A,ld=32] T -> [B,a=A,d=512] T
            const broadcast_condition = condition.broad(projected.shape()); // [d=512] T -> [B,a=A,d=512] T
            const conditioned = projected.add(broadcast_condition); // [B,A,512] T + [B,A,512] T -> [B,A,512] T
            const replacements = conditioned.gather(.{ .a = indices }, .{}); // [B,a=A,512] T + [B,S] i32 indices -> [B,S,512] T
            const broadcast_mask = mask.broad(x.shape()); // [B,S] bool -> [B,S,d=512] bool
            output = broadcast_mask.select(replacements, output); // [B,S,512] bool, 2 x [B,S,512] T -> [B,S,512] T
        }
        return output;
    }

    // -- Runtime --

    pub fn load(
        self: *const Input_Projection,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Input_Projection) {
        var buffers = try zml.mem.bufferize(allocator, Input_Projection, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Input_Projection, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Input_Projection)) void {
        zml.Buffer.deinitAll(Input_Projection, self);
    }
};

test Input_Projection {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Input_Projection = .{
        .audio_prompt_proj = .init(.fromShape(.init(.{ .d = 512, .ad = 512 }, .bf16)), null, .ad),
        .audio_latent_proj = .init(.fromShape(.init(.{ .d = 512, .ld = 32 }, .bf16)), null, .ld),
        .bos_latent = .fromShape(.init(.{ .ld = 32 }, .bf16)),
        .num_speakers_embed = .fromShape(.init(.{ .voc = 32, .spk = 16 }, .bf16)),
        .num_speakers_proj = .init(.fromShape(.init(.{ .d = 512, .spk = 16 }, .bf16)), null, .spk),
        .duration_delta_proj = .init(.fromShape(.init(.{ .d = 512, .dur = 128 }, .bf16)), null, .dur),
        .num_speakers = random.intRangeLessThan(i64, 0, 32),
        .duration_delta = random.float(f32),
    };
    var weight_buffers = try zml.mem.bufferize(allocator, Input_Projection, &layer);
    const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &layer);
    defer allocator.free(tensors);
    const buffers = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &weight_buffers);
    defer allocator.free(buffers);
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |buffer| buffer.deinit();
    for (tensors, buffers) |tensor, buffer| {
        const values = try zml.Slice.alloc(allocator, tensor.shape());
        defer values.free(allocator);
        for (values.items(zml.floats.BFloat16)) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
        buffer.* = try zml.Buffer.fromSlice(io, platform, values);
        loaded += 1;
    }

    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 7, .d = 512 }, .f32);
    var input: [2][7][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    // Different marker orders in each batch row exercise per-sequence frame indexing.
    // Also cover empty voice/audio histories, where both gathers must be skipped.
    for ([_]usize{ 2, 0 }) |frame_count| {
        const ids_shape: zml.Shape = .init(.{ .b = 2, .s = 7 }, .i64);
        var ids: [2][7]i64 = .{ .{ 7, 5, 4, 6, 5, 6, 7 }, .{ 6, 5, 7, 5, 4, 7, 6 } };
        for (&ids) |*batch| {
            for (batch) |*id| {
                if (id.* == 7 or (frame_count == 0 and (id.* == 5 or id.* == 6))) {
                    id.* = random.intRangeLessThan(i64, 7, 4001);
                }
            }
        }
        var ids_buffer = try zml.Buffer.fromBytes(io, platform, ids_shape, std.mem.asBytes(&ids));
        defer ids_buffer.deinit();

        const voice_shape: zml.Shape = .init(.{ .b = 2, .p = frame_count, .ad = 512 }, .f32);
        const voice = try zml.Slice.alloc(allocator, voice_shape);
        defer voice.free(allocator);
        for (voice.items(f32)) |*value| value.* = random.float(f32) * 2 - 1;
        var voice_buffer = try zml.Buffer.fromSlice(io, platform, voice);
        defer voice_buffer.deinit();

        const audio_shape: zml.Shape = .init(.{ .b = 2, .a = frame_count, .ld = 32 }, .f32);
        const audio = try zml.Slice.alloc(allocator, audio_shape);
        defer audio.free(allocator);
        for (audio.items(f32)) |*value| value.* = random.float(f32) * 2 - 1;
        var audio_buffer = try zml.Buffer.fromSlice(io, platform, audio);
        defer audio_buffer.deinit();

        var exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape),
            zml.Tensor.fromShape(ids_shape),
            zml.Tensor.fromShape(voice_shape),
            zml.Tensor.fromShape(audio_shape),
        }, .{});
        defer exe.deinit();
        var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer, ids_buffer, voice_buffer, audio_buffer });
        defer output.deinit();

        const expected = if (frame_count > 0) &test_data.input_projection else &test_data.input_projection_empty;
        const expected_slice = zml.Slice.init(input_shape, std.mem.asBytes(expected));
        try zml.testing.expectClose(io, expected_slice, output, .{
            .absolute_tolerance = 1e-4,
            .relative_tolerance = 1e-4,
            .minimum_close_fraction = 1,
        });

        // Leave auxiliary inputs in f32 to check conversion to the primary input dtype.
        for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
            var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
                zml.Tensor.fromShape(input_shape.withDtype(dtype)),
                zml.Tensor.fromShape(ids_shape),
                zml.Tensor.fromShape(voice_shape),
                zml.Tensor.fromShape(audio_shape),
            }, .{});
            defer typed_exe.deinit();
            try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(input_shape.withDtype(dtype)));
        }
    }
}

// --- Gradium_Phonon_TTS_Decoder_Layer ---
// The transformer block, which runs in multiple passes.
//
// Broadly:
// ```
// a = input_layernorm(x)
// a = self_attn(a)
// y = x + a
//
// m = post_attention_layernorm(y)
// m = mlp(m)
// output = y + m
// ```

/// Mixes sequence context, then refines each position's features.
///
/// The attention branch normalizes its input, performs causal self-attention,
/// and adds the result to the original features. The MLP branch normalizes that
/// sum, applies a 512 -> 2048 -> 512 network with GELU, and adds another residual.
/// Input and output are f32 `[B, S, 512]`; batch rows remain independent.
///
/// `forward` recomputes a complete sequence starting at position zero.
/// `forward_cached` processes Q new positions using stored K/V history and returns
/// the updated cache along with f32 features `[B, Q, 512]`.
///
/// Each layer has separate bf16 parameters under
/// `flow_lm.transformer.layers.{index}`: `norm1`, `self_attn.in_proj`,
/// `self_attn.out_proj`, `norm2`, `linear1`, and `linear2`.
/// Computation uses f32, norm epsilon 1e-5, and exact erf GELU.
///
/// Reference:
/// ```
/// GradiumPhononTTSDecoderLayer
/// ├── self_attn: GradiumPhononTTSAttention
/// │   ├── q_proj: Linear
/// │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       ├── loaded: model.layers.0.self_attn.q_proj.weight
/// │   │       └── checkpoint: flow_lm.transformer.layers.0.self_attn.in_proj.weight
/// │   ├── k_proj: Linear
/// │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       ├── loaded: model.layers.0.self_attn.k_proj.weight
/// │   │       └── checkpoint: flow_lm.transformer.layers.0.self_attn.in_proj.weight
/// │   ├── v_proj: Linear
/// │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       ├── loaded: model.layers.0.self_attn.v_proj.weight
/// │   │       └── checkpoint: flow_lm.transformer.layers.0.self_attn.in_proj.weight
/// │   └── o_proj: Linear
/// │       └── weight: shape=(512, 512), dtype=torch.float32
/// │           ├── loaded: model.layers.0.self_attn.o_proj.weight
/// │           └── checkpoint: flow_lm.transformer.layers.0.self_attn.out_proj.weight
/// ├── mlp: GradiumPhononTTSMLP
/// │   ├── activation_fn: GELUActivation
/// │   ├── fc1: Linear
/// │   │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │   │       ├── loaded: model.layers.0.mlp.fc1.weight
/// │   │       └── checkpoint: flow_lm.transformer.layers.0.linear1.weight
/// │   └── fc2: Linear
/// │       └── weight: shape=(512, 2048), dtype=torch.float32
/// │           ├── loaded: model.layers.0.mlp.fc2.weight
/// │           └── checkpoint: flow_lm.transformer.layers.0.linear2.weight
/// ├── input_layernorm: LayerNorm
/// │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   ├── loaded: model.layers.0.input_layernorm.weight
/// │   │   └── checkpoint: flow_lm.transformer.layers.0.norm1.weight
/// │   └── bias: shape=(512,), dtype=torch.float32
/// │       ├── loaded: model.layers.0.input_layernorm.bias
/// │       └── checkpoint: flow_lm.transformer.layers.0.norm1.bias
/// └── post_attention_layernorm: LayerNorm
///     ├── weight: shape=(512,), dtype=torch.float32
///     │   ├── loaded: model.layers.0.post_attention_layernorm.weight
///     │   └── checkpoint: flow_lm.transformer.layers.0.norm2.weight
///     └── bias: shape=(512,), dtype=torch.float32
///         ├── loaded: model.layers.0.post_attention_layernorm.bias
///         └── checkpoint: flow_lm.transformer.layers.0.norm2.bias
/// ```
const Gradium_Phonon_TTS_Decoder_Layer = struct {
    input_layernorm: Input_Layer_Norm,
    self_attn: Gradium_Phonon_TTS_Attention,
    post_attention_layernorm: Post_Attention_Layer_Norm,
    mlp: Gradium_Phonon_TTS_MLP,

    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Decoder_Layer {
        return .{
            .input_layernorm = .init(store),
            .self_attn = .init(store),
            .post_attention_layernorm = .init(store),
            .mlp = .init(store),
        };
    }

    /// Runs a complete sequence through attention and MLP residual branches.
    ///
    /// `input`: f32 `[B, S, 512]`, prepared model inputs for the first decoder
    /// layer or residual features from the preceding layer. This method assigns
    /// tags `.b`, `.s`, and `.d`; sequences must be unpadded.
    ///
    /// Returns f32 `[B, S, 512]`, contextual features after both residual additions:
    /// `x = input + attention(norm1(input)); output = x + mlp(norm2(x))`.
    /// The output feeds the next decoder layer or the model's final normalization.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension, the 512 features describing each token. Layer norm
    ///   normalizes across these features; the MLP transforms them independently per token.
    /// - `.b`: batch; independent sequences, never mixed by the layer.
    /// - `.s`: sequence; token positions. Causal attention mixes information from
    ///   earlier positions and the current position; norm and MLP act per token.
    /// All branches retain these tags so residual additions align corresponding
    /// tokens and features. Attention's internal tags are documented in its forward.
    ///
    /// Shape: `[.b, .s, .d=512] -> [.b, .s, .d=512]`, with f32 input and output.
    pub fn forward(self: Gradium_Phonon_TTS_Decoder_Layer, input: zml.Tensor) zml.Tensor {
        std.debug.assert(input.dtype() == .f32);
        std.debug.assert(input.rank() == 3);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Decoder_Layer", .{});
        defer compiler.popLocation();

        const x = input.withTags(.{ .b, .s, .d }); // [B,S,512] f32 -> [b=B,s=S,d=512] f32

        const attention_input = self.input_layernorm.forward(x); // [B,S,512] f32 -> [B,S,512] f32
        const attention_output = self.self_attn.forward_uncached(attention_input); // [B,S,512] f32 -> [B,S,512] f32
        const attended = x.add(attention_output); // [B,S,512] f32 + [B,S,512] f32 -> [B,S,512] f32

        const mlp_input = self.post_attention_layernorm.forward(attended); // [B,S,512] f32 -> [B,S,512] f32
        const mlp_output = self.mlp.forward(mlp_input); // [B,S,512] f32 -> [B,S,512] f32
        const output = attended.add(mlp_output); // [B,S,512] f32 + [B,S,512] f32 -> [B,S,512] f32
        return output;
    }

    /// Runs attention and MLP residual branches for newly supplied positions.
    ///
    /// `input`: f32 `[B, Q, 512]`, prepared model features for layer 0 or the
    /// preceding decoder layer's output, passed by `forward_cached_layers`.
    /// `position`: scalar i32 `[]`, the first new absolute sequence slot, supplied
    /// by the generation loop (0 for prefill, then the next unprocessed slot).
    /// It offsets RoPE and the causal mask; the host must ensure 0 <= position <= C-Q.
    /// `cache`: runtime K/V history supplied by the caller, with each f32 tensor
    /// shaped `[.layer=6, .b=B, .k=C, .h=8, .hd=64]`. C is fixed capacity;
    /// keys include RoPE and values do not. Earlier prefix slots must already be
    /// populated, and unused slots must contain finite values.
    /// `layer_index`: host usize selecting this layer's cache slice, supplied by
    /// the model's loop (0..5); other layer slices are retained.
    ///
    /// Returns residual features f32 `[B, Q, 512]` and updated cache for this layer.
    /// The next layer consumes the features; the model normalizes after layer 5.
    ///
    /// Dimension tags:
    /// - `.b`: independent sequences; `.s`: Q new positions; `.d`: 512 features.
    /// - `.layer`: six decoder layers; `.k`: C cache slots; `.h`: eight heads;
    ///   `.hd`: 64 features per head. Cache capacity is separate from Q.
    ///
    /// Dimension tags:
    /// - `.b`: independent sequences; `.s`: Q new positions; `.d`: 512 features.
    /// - `.layer`: six decoder layers; `.k`: C cache slots; `.h`: eight heads;
    ///   `.hd`: 64 features per head. Cache capacity is separate from Q.
    ///
    /// Dimension tags:
    /// - `.b`: independent sequences; `.s`: Q new positions; `.d`: 512 features.
    /// - `.layer`: six decoder layers; `.k`: C cache slots; `.h`: eight heads;
    ///   `.hd`: 64 features per head. Cache capacity is separate from Q.
    pub fn forward_cached(self: Gradium_Phonon_TTS_Decoder_Layer, input: zml.Tensor, position: zml.Tensor, cache: KV_Cache, layer_index: usize) CachedHidden {
        const attention_input = self.input_layernorm.forward(input); // [B,Q,512] f32 -> [B,Q,512] f32
        const attention = self.self_attn.forward_cached(attention_input, position, cache, layer_index); // [B,Q,512] + cache f32, [] i32 -> [B,Q,512] + updated cache f32
        const attended = input.add(attention.hidden); // 2 x [B,Q,512] f32 -> [B,Q,512] f32
        const mlp_input = self.post_attention_layernorm.forward(attended); // [B,Q,512] f32 -> [B,Q,512] f32
        const mlp_output = self.mlp.forward(mlp_input); // [B,Q,512] f32 -> [B,Q,512] f32
        const output = attended.add(mlp_output); // 2 x [B,Q,512] f32 -> [B,Q,512] f32
        return .{ .hidden = output, .cache = attention.cache };
    }

    pub fn load(
        self: *const Gradium_Phonon_TTS_Decoder_Layer,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Decoder_Layer) {
        var buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Decoder_Layer, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Gradium_Phonon_TTS_Decoder_Layer, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Decoder_Layer)) void {
        zml.Buffer.deinitAll(Gradium_Phonon_TTS_Decoder_Layer, self);
    }
};

test Gradium_Phonon_TTS_Decoder_Layer {
    const allocator = std.testing.allocator;
    const bazel_builtin = @import("bazel_builtin");
    const runfiles = try zml.bazel.runfiles(bazel_builtin.current_repository);

    // Each iteration uses that layer's own weights and captured Python input/output.
    // The separate transformer-loop test checks the six layers composed together.
    for (0..Gradium_Phonon_TTS_Model.num_layers) |index| {
        const key = try std.fmt.allocPrint(allocator, "zml/examples/phonon/assets/model.layers.{d}.activations.safetensors", .{index});
        defer allocator.free(key);
        const activations_path = try runfiles.rlocationAlloc(allocator, key) orelse return error.ActivationsNotFound;
        defer allocator.free(activations_path);
        const prefix = try std.fmt.allocPrint(allocator, "flow_lm.transformer.layers.{d}", .{index});
        defer allocator.free(prefix);

        try util.test_module(
            Gradium_Phonon_TTS_Decoder_Layer,
            util.gradium_phonon_31m_en,
            activations_path,
            prefix,
        );
    }
}

// --- Gradium_Phonon_TTS_Attention ---

/// Lets each sequence position gather information from its causal context.
///
/// A packed projection creates queries, keys, and values for eight heads of
/// 64 features. RoPE rotates queries and keys using absolute sequence positions.
/// Scaled dot-product attention weights visible values, then an output projection
/// combines the heads into 512 features. The decoder layer adds the residual.
///
/// Input and output: floating-point `[B, Q, 512]`, preserving the input dtype.
/// The standalone `forward_uncached` uses
/// the entire supplied sequence as context, starting at zero. `forward_cached`
/// writes Q new positions to the selected layer's cache and attends to the stored
/// prefix plus causally visible new positions. Both paths require unpadded inputs.
///
/// Parameters come from `flow_lm.transformer.layers.{index}`:
/// `self_attn.in_proj.weight` [1536, 512] and `self_attn.out_proj.weight` [512, 512].
/// Both are bias-free and converted from bf16 to the input dtype. RoPE uses
/// interleaved pairs and theta 10000; attention uses scale 1/sqrt(64) and softmax
/// over key positions. ZML computes RoPE trigonometry and attention softmax in
/// f32 internally, then converts back to the input dtype.
///
/// Reference:
/// ```
/// self_attn: GradiumPhononTTSAttention
/// ├── q_proj: Linear
/// │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       ├── loaded: model.layers.1.self_attn.q_proj.weight
/// │       └── checkpoint: flow_lm.transformer.layers.1.self_attn.in_proj.weight
/// ├── k_proj: Linear
/// │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       ├── loaded: model.layers.1.self_attn.k_proj.weight
/// │       └── checkpoint: flow_lm.transformer.layers.1.self_attn.in_proj.weight
/// ├── v_proj: Linear
/// │   └── weight: shape=(512, 512), dtype=torch.float32
/// │       ├── loaded: model.layers.1.self_attn.v_proj.weight
/// │       └── checkpoint: flow_lm.transformer.layers.1.self_attn.in_proj.weight
/// └── o_proj: Linear
///     └── weight: shape=(512, 512), dtype=torch.float32
///         ├── loaded: model.layers.1.self_attn.o_proj.weight
///         └── checkpoint: flow_lm.transformer.layers.1.self_attn.out_proj.weight
/// ```
const Gradium_Phonon_TTS_Attention = struct {
    qkv_proj: zml.nn.Linear,
    out_proj: zml.nn.Linear,

    const num_heads = 8;
    const head_dim = 64;

    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Attention {
        return .{
            .qkv_proj = .init(store.createReplicatedTensor("self_attn.in_proj.weight", .{ .qkv, .d }), null, .d),
            .out_proj = .init(store.createReplicatedTensor("self_attn.out_proj.weight", .{ .dout, .d }), null, .d),
        };
    }

    // -- Description --

    /// Computes causal attention over the complete supplied sequence.
    ///
    /// `input`: floating-point `[B, S, 512]`, normalized features produced by the decoder
    /// layer's `Input_Layer_Norm`. This method assigns tags `.b`, `.s`, and `.d`.
    /// The sequence is unpadded, with positions starting at zero and no prior cache.
    ///
    /// Returns `[B, S, 512]` in `input.dtype()`, projected attention features.
    /// The decoder layer adds them to its original input; this method performs no
    /// residual addition.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent sequences processed together, never mixed.
    /// - `.s`: sequence; the token positions in each input/output sequence.
    /// - `.d`: model dimension; 512 features describing a token. The packed Q/K/V
    ///   projection contracts this axis, and merging the heads restores it.
    /// - `.qkv`: packed query/key/value features; the projection produces 1536
    ///   values per token, chunked into three groups of 512. Each group is then
    ///   split into `.h` and `.hd` without changing its values.
    /// - `.h`: attention heads; 8 independent attention computations per token.
    /// - `.hd`: head dimension; 64 features within each head. RoPE rotates pairs
    ///   of these features in Q/K; attention contracts Q/K over this axis to
    ///   compute scores and retains V's features in the weighted result.
    /// - `.q`: query positions; `.s` renamed on Q, the tokens requesting information.
    /// - `.k`: key/value positions; `.s` renamed on K/V, the tokens supplying it.
    ///   Scores and the causal mask have separate `.q` and `.k` axes, even though
    ///   both lengths equal `.s` here. Softmax and the weighted sum run over `.k`;
    ///   the mask forbids keys after each query's position.
    /// - `.dout`: output projection features (512); distinct from its contracted
    ///   `.d` input axis, then renamed to `.d` for the caller.
    ///
    /// Shape: `[.b, .s, .d=512] -> [.b, .s, .d=512]`. Internally Q has shape
    /// `[.b, .q, .h=8, .hd=64]` and K/V have `[.b, .k, .h=8, .hd=64]`.
    pub fn forward_uncached(self: Gradium_Phonon_TTS_Attention, input: zml.Tensor) zml.Tensor {
        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat());
        std.debug.assert(input.rank() == 3);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Attention", .{});
        defer compiler.popLocation();

        // T denotes input.dtype() below.
        const x = input.withTags(.{ .b, .s, .d }); // [B,S,512] T -> [b=B,s=S,d=512] T
        std.debug.assert(x.dim(.d) == num_heads * head_dim);

        const packed_projection = self.qkv_proj.forward(x, dtype); // [B,S,d=512] T -> [B,S,qkv=1536] T
        const projections = packed_projection.chunkExact(.qkv, 3); // [B,S,qkv=1536] T -> 3 x [B,S,qkv=512] T
        const q_heads = projections[0].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }); // [B,S,qkv=512] T -> [B,S,h=8,hd=64] T
        const k_heads = projections[1].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }); // [B,S,qkv=512] T -> [B,S,h=8,hd=64] T
        const v_heads = projections[2].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }); // [B,S,qkv=512] T -> [B,S,h=8,hd=64] T
        const v = v_heads.rename(.{ .s = .k }); // [B,s=S,8,64] T -> [B,k=S,8,64] T

        // The original packed weights use adjacent real/imaginary coordinates.
        // Transformers permutes Q/K rows for its split-half RoPE convention instead.
        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .interleaved,
            .scaling = .{ .default = .{ .rope_theta = 10000 } },
        };
        const rotated_q = zml.nn.rope(q_heads, null, rope_opts); // [B,S,8,64] T -> [B,S,8,64] T
        const q = rotated_q.rename(.{ .s = .q }); // [B,s=S,8,64] T -> [B,q=S,8,64] T
        const rotated_k = zml.nn.rope(k_heads, null, rope_opts); // [B,S,8,64] T -> [B,S,8,64] T
        const k = rotated_k.rename(.{ .s = .k }); // [B,s=S,8,64] T -> [B,k=S,8,64] T

        // Zero permits attention; negative infinity excludes future keys.
        const mask = zml.nn.causalAttnMask(.{ .q = x.dim(.s), .k = x.dim(.s) }, dtype, null); // host S -> [q=S,k=S] T
        const attended = zml.nn.sdpa(q, k, v, .{ .attn_mask = mask }); // Q [B,q=S,8,64], K/V [B,k=S,8,64], mask [S,S] T -> [B,q=S,8,64] T
        const merged = attended.merge(.{ .d = .{ .h, .hd } }); // [B,q=S,h=8,hd=64] T -> [B,q=S,d=512] T
        const joined = merged.rename(.{ .q = .s }); // [B,q=S,512] T -> [B,s=S,512] T
        const projected = self.out_proj.forward(joined, dtype); // [B,S,d=512] T -> [B,S,dout=512] T
        const output = projected.rename(.{ .dout = .d }); // [B,S,dout=512] T -> [B,S,d=512] T
        return output;
    }

    /// Attends from new positions to their cached prefix and visible new context.
    ///
    /// `input`: floating-point `[B, Q, 512]`, the decoder layer's pre-attention normalized
    /// features from `Input_Layer_Norm`, for Q prompt positions or one decode position.
    /// `position`: scalar i32 `[]`, the first new absolute sequence slot, supplied
    /// by the generation loop (0 for prefill, then the next unprocessed slot).
    /// It offsets RoPE and the causal mask; the host must ensure 0 <= position <= C-Q.
    /// `cache`: runtime K/V history supplied by the caller, with each floating-point tensor
    /// shaped `[.layer=6, .b=B, .k=C, .h=8, .hd=64]`. C is fixed capacity;
    /// keys include RoPE and values do not. Earlier prefix slots must already be
    /// populated, and unused slots must contain finite values. Each cache tensor
    /// retains its dtype: new entries are converted on write, and selected cached
    /// keys/values are converted to `input.dtype()` for attention.
    /// `layer_index`: host usize (0..5), passed by the decoder layer to select its
    /// cache slice. New rotated keys and unrotated values replace Q slots there.
    ///
    /// Returns projected features `[B, Q, 512]` in `input.dtype()` and updated
    /// cache tensors in their respective input dtypes.
    /// The decoder layer performs the residual addition. Positions after each query
    /// are masked; earlier prefix positions must have been written by prior calls.
    ///
    /// Dimension tags:
    /// - `.b`: independent sequences; `.s`: Q new positions; `.d`: 512 features.
    /// - `.layer`: six decoder layers; `.k`: C cache slots; `.h`: eight heads;
    ///   `.hd`: 64 features per head. Cache capacity is separate from Q.
    /// - `.q`: Q query positions; `.qkv`: 1536 packed query/key/value features.
    /// - `.dout`: 512 output projection features, renamed to `.d`.
    /// Queries use `[.b, .q, .h, .hd]`; cached keys/values use `[.b, .k, .h, .hd]`.
    pub fn forward_cached(
        self: Gradium_Phonon_TTS_Attention,
        input: zml.Tensor,
        position: zml.Tensor,
        cache: KV_Cache,
        layer_index: usize,
    ) CachedHidden {
        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and input.rank() == 3);
        std.debug.assert(cache.k.dtype().isFloat() and cache.v.dtype().isFloat());
        // T denotes input.dtype() below; cache tensors retain their own dtypes.
        const x = input.withTags(.{ .b, .s, .d }); // [B,Q,512] T -> [b=B,s=Q,d=512] T
        std.debug.assert(position.rank() == 0 and position.dtype() == .i32);
        std.debug.assert(x.dim(.d) == num_heads * head_dim);
        std.debug.assert(x.dim(.b) == cache.k.dim(.b) and x.dim(.s) <= cache.k.dim(.k));
        const packed_projection = self.qkv_proj.forward(x, dtype); // [B,Q,d=512] T -> [B,Q,qkv=1536] T
        const projections = packed_projection.chunkExact(.qkv, 3); // [B,Q,1536] T -> 3 x [B,Q,qkv=512] T
        const q_heads = projections[0].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }); // [B,Q,512] T -> [B,Q,h=8,hd=64] T
        const k_heads = projections[1].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }); // [B,Q,512] T -> [B,Q,h=8,hd=64] T
        const v_heads = projections[2].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }); // [B,Q,512] T -> [B,Q,h=8,hd=64] T

        const offsets = zml.Tensor.arange(.{ .end = x.dim(.s) }, .i32); // host Q -> [Q] i32
        const tagged_offsets = offsets.withTags(.{.s}); // [Q] i32 -> [s=Q] i32
        const start = position.broad(tagged_offsets.shape()); // [] i32 -> [s=Q] i32
        const positions = tagged_offsets.add(start); // [s=Q] i32 + [s=Q] i32 -> [s=Q] i32
        const rope_opts: zml.nn.RopeOpts = .{ .layout = .interleaved, .scaling = .{ .default = .{ .rope_theta = 10000 } } };
        const rotated_q = zml.nn.rope(q_heads, positions, rope_opts); // [B,Q,8,64] T + [Q] positions -> [B,Q,8,64] T
        const q = rotated_q.rename(.{ .s = .q }); // [B,s=Q,8,64] T -> [B,q=Q,8,64] T
        const rotated_k = zml.nn.rope(k_heads, positions, rope_opts); // [B,Q,8,64] T + [Q] positions -> [B,Q,8,64] T
        const new_k = rotated_k.rename(.{ .s = .k }); // [B,s=Q,8,64] T -> [B,k=Q,8,64] T
        const new_v = v_heads.rename(.{ .s = .k }); // [B,s=Q,8,64] T -> [B,k=Q,8,64] T
        const layer = zml.Tensor.scalar(@as(i32, @intCast(layer_index)), .i32); // host layer index -> [] i32
        const updated_k = cache.k.dynamicUpdateSlice(.{ .layer = layer, .k = position }, new_k.convert(cache.k.dtype())); // [6,B,C,8,64] + [B,Q,8,64] cache dtype -> [6,B,C,8,64] cache dtype
        const updated_v = cache.v.dynamicUpdateSlice(.{ .layer = layer, .k = position }, new_v.convert(cache.v.dtype())); // [6,B,C,8,64] + [B,Q,8,64] cache dtype -> [6,B,C,8,64] cache dtype
        const layer_keys = updated_k.slice(.layer, .{ .start = @intCast(layer_index), .end = @intCast(layer_index + 1) }); // [6,B,C,8,64] cache dtype -> [1,B,C,8,64] cache dtype
        const keys = layer_keys.squeeze(.layer).convert(dtype); // [1,B,C,8,64] cache dtype -> [B,k=C,8,64] T
        const layer_values = updated_v.slice(.layer, .{ .start = @intCast(layer_index), .end = @intCast(layer_index + 1) }); // [6,B,C,8,64] cache dtype -> [1,B,C,8,64] cache dtype
        const values = layer_values.squeeze(.layer).convert(dtype); // [1,B,C,8,64] cache dtype -> [B,k=C,8,64] T

        // Use absolute query positions: a decode query at P may consult keys 0..P.
        const query_positions = positions.rename(.{ .s = .q }); // [s=Q] i32 -> [q=Q] i32
        const mask_shape = zml.Shape.init(.{ .q = x.dim(.s), .k = cache.k.dim(.k) }, .i32);
        const key_positions = zml.Tensor.iota(mask_shape, .k); // host Q,C -> [q=Q,k=C] i32
        const broadcast_queries = query_positions.broad(mask_shape); // [q=Q] i32 -> [q=Q,k=C] i32
        const visible = key_positions.cmp(.LE, broadcast_queries); // 2 x [Q,C] i32 -> [Q,C] bool
        const allowed = zml.Tensor.scalar(0, dtype); // host zero -> [] T
        const blocked = zml.Tensor.scalar(-std.math.inf(f32), dtype); // host -infinity -> [] T
        const mask = visible.select(allowed, blocked); // [Q,C] bool + 2 scalars T -> [Q,C] T
        const attended = zml.nn.sdpa(q, keys, values, .{ .attn_mask = mask }); // [B,Q,8,64], 2 x [B,C,8,64], [Q,C] T -> [B,Q,8,64] T
        const merged = attended.merge(.{ .d = .{ .h, .hd } }); // [B,Q,8,64] T -> [B,q=Q,d=512] T
        const joined = merged.rename(.{ .q = .s }); // [B,q=Q,512] T -> [B,s=Q,512] T
        const projected = self.out_proj.forward(joined, dtype); // [B,Q,d=512] T -> [B,Q,dout=512] T
        const output = projected.rename(.{ .dout = .d }); // [B,Q,dout=512] T -> [B,Q,d=512] T
        return .{ .hidden = output, .cache = .{ .k = updated_k, .v = updated_v } };
    }

    // -- Runtime --

    pub fn load(
        self: *const Gradium_Phonon_TTS_Attention,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Attention) {
        var buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Attention, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Gradium_Phonon_TTS_Attention, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Attention)) void {
        zml.Buffer.deinitAll(Gradium_Phonon_TTS_Attention, self);
    }
};

test Gradium_Phonon_TTS_Attention {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Gradium_Phonon_TTS_Attention = .{
        .qkv_proj = .init(.fromShape(.init(.{ .qkv = 1536, .d = 512 }, .bf16)), null, .d),
        .out_proj = .init(.fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)), null, .d),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 3, .d = 512 }, .f32);
    var input: [2][3][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var qkv_weights: [1536][512]zml.floats.BFloat16 = undefined;
    for (&qkv_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var out_weights: [512][512]zml.floats.BFloat16 = undefined;
    for (&out_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }

    var qkv_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.qkv_proj.weight.shape(), std.mem.asBytes(&qkv_weights));
    defer qkv_weight_buffer.deinit();
    var out_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.out_proj.weight.shape(), std.mem.asBytes(&out_weights));
    defer out_weight_buffer.deinit();
    const weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_Attention) = .{
        .qkv_proj = .{ .weight = qkv_weight_buffer, .bias = null, .quantization = null },
        .out_proj = .{ .weight = out_weight_buffer, .bias = null, .quantization = null },
    };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward_uncached, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.attention_uncached;
    const expected_slice = zml.Slice.init(input_shape, std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward_uncached, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(input_shape.withDtype(dtype)));
    }

    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        const cache_shape: zml.Shape = .init(.{ .layer = 6, .b = 2, .k = 5, .h = 8, .hd = 64 }, dtype);
        // Check matching dtypes as well as separate K/V storage dtypes.
        for ([_]KV_Cache{
            .{ .k = .fromShape(cache_shape), .v = .fromShape(cache_shape) },
            .{ .k = .fromShape(cache_shape.withDtype(.f16)), .v = .fromShape(cache_shape.withDtype(.bf16)) },
        }) |cache| {
            var cached_exe = try platform.compile(allocator, io, layer, .forward_cached, .{
                zml.Tensor.fromShape(input_shape.withDtype(dtype)),
                zml.Tensor.fromShape(.init(.{}, .i32)),
                cache,
                @as(usize, 2),
            }, .{});
            defer cached_exe.deinit();
            try std.testing.expect(cached_exe.output_shapes[0].eqlWithTags(input_shape.withDtype(dtype)));
            try std.testing.expect(cached_exe.output_shapes[1].eqlWithTags(cache.k.shape()));
            try std.testing.expect(cached_exe.output_shapes[2].eqlWithTags(cache.v.shape()));
        }
    }
}

// --- Gradium_Phonon_TTS_MLP ---

/// Refines each position's features with a two-layer feed-forward network.
///
/// A bias-free linear layer expands 512 features to 2048, exact erf GELU acts
/// elementwise, and another bias-free layer returns to 512 features. Sequence
/// positions and batch rows are processed independently. The decoder layer
/// supplies normalized features and adds this module's output as a residual.
///
/// Input and output: floating-point `[..., .d=512]`, preserving all leading axes.
/// Parameters come from `flow_lm.transformer.layers.{index}`:
/// `linear1.weight` [2048, 512] and `linear2.weight` [512, 2048], stored in bf16
/// and converted to the input dtype.
///
/// Reference:
/// ```
/// mlp: GradiumPhononTTSMLP
/// ├── activation_fn: GELUActivation
/// ├── fc1: Linear
/// │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │       ├── loaded: model.layers.1.mlp.fc1.weight
/// │       └── checkpoint: flow_lm.transformer.layers.1.linear1.weight
/// └── fc2: Linear
///     └── weight: shape=(512, 2048), dtype=torch.float32
///         ├── loaded: model.layers.1.mlp.fc2.weight
///         └── checkpoint: flow_lm.transformer.layers.1.linear2.weight
/// ```
const Gradium_Phonon_TTS_MLP = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_MLP {
        return .{
            .fc1 = .init(store.createReplicatedTensor("linear1.weight", .{ .dout, .d }), null, .d),
            .fc2 = .init(store.createReplicatedTensor("linear2.weight", .{ .d, .dout }), null, .dout),
        };
    }

    // -- Description --

    /// Applies fc1 -> exact GELU -> fc2 independently at each position.
    ///
    /// `input`: floating-point `[..., 512]`, normalized features from the decoder layer's
    /// `Post_Attention_Layer_Norm`, after the attention residual addition.
    /// Usually `[B, S, 512]`, or `[B, 1, 512]` during cached decode. This method
    /// tags only the final axis `.d`, preserving any leading axes.
    ///
    /// Returns floating-point `[..., .d=512]`, the MLP residual contribution. The decoder
    /// layer adds it to the features from before pre-MLP normalization.
    ///
    /// Parameters and output use the input dtype. ZML's erf approximation is
    /// evaluated in f32 and converted back before the remaining GELU operations.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; 512 features per token. Contracted by fc1 and
    ///   restored by fc2 so the output can feed the next transformer operation.
    /// - `.dout`: fc1's output dimension; the expanded 2048 intermediate features
    ///   per token. GELU acts elementwise; fc2 contracts this axis back to `.d`.
    /// - Leading axes are preserved without mixing their entries. Conventionally
    ///   `.b` is batch (independent sequences) and `.s` is sequence (token positions).
    ///   Only the final axis is tagged here; untagged leading axes remain untagged.
    ///
    /// Shape: `[..., .d=512] -> [..., .dout=2048] -> [..., .d=512]`, preserving the input dtype.
    pub fn forward(self: Gradium_Phonon_TTS_MLP, input: zml.Tensor) zml.Tensor {
        std.debug.assert(input.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_MLP", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d}); // [...,512] -> [...,d=512]

        compiler.pushLocationFmt(@src(), "layer[fc1]", .{});
        const hidden = self.fc1.forward(x, x.dtype()); // [...,d=512] -> [...,dout=2048]
        compiler.popLocation();

        compiler.pushLocationFmt(@src(), "layer[gelu]", .{});
        const activated = gelu_erf(hidden); // [...,dout=2048] -> [...,dout=2048]
        compiler.popLocation();

        compiler.pushLocationFmt(@src(), "layer[fc2]", .{});
        const output = self.fc2.forward(activated, activated.dtype()); // [...,dout=2048] -> [...,d=512]
        compiler.popLocation();
        return output;
    }

    fn gelu_erf(x: zml.Tensor) zml.Tensor {
        // Tensor.erf returns f32; convert back before combining it with x.
        const erf_term = x.scale(1.0 / std.math.sqrt(2.0)).erf().convert(x.dtype());
        return x.scale(0.5).mul(erf_term.addConstant(1.0));
    }

    // -- Runtime --

    pub fn load(
        self: *const Gradium_Phonon_TTS_MLP,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_MLP) {
        var buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_MLP, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Gradium_Phonon_TTS_MLP, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_MLP)) void {
        zml.nn.Linear.unloadBuffers(&self.fc1);
        zml.nn.Linear.unloadBuffers(&self.fc2);
    }
};

test Gradium_Phonon_TTS_MLP {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Gradium_Phonon_TTS_MLP = .{
        .fc1 = .init(.fromShape(.init(.{ .dout = 2048, .d = 512 }, .bf16)), null, .d),
        .fc2 = .init(.fromShape(.init(.{ .d = 512, .dout = 2048 }, .bf16)), null, .dout),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, .f32);
    var input: [2][2][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var fc1_weights: [2048][512]zml.floats.BFloat16 = undefined;
    for (&fc1_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var fc2_weights: [512][2048]zml.floats.BFloat16 = undefined;
    for (&fc2_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }

    var fc1_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.fc1.weight.shape(), std.mem.asBytes(&fc1_weights));
    defer fc1_weight_buffer.deinit();
    var fc2_weight_buffer = try zml.Buffer.fromBytes(io, platform, layer.fc2.weight.shape(), std.mem.asBytes(&fc2_weights));
    defer fc2_weight_buffer.deinit();
    const weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_MLP) = .{
        .fc1 = .{ .weight = fc1_weight_buffer, .bias = null, .quantization = null },
        .fc2 = .{ .weight = fc2_weight_buffer, .bias = null, .quantization = null },
    };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.mlp;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .s = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = 2, .d = 512 }, dtype)));
    }
}

// --- Input_Layer_Norm ---

/// Normalizes features entering a decoder layer, before its attention branch.
///
/// Computes mean and population variance across each position's 512 features,
/// normalizes with epsilon 1e-5, then applies a learned per-feature scale and bias.
/// Batch rows and sequence positions retain separate statistics.
///
/// Input and output: floating-point `[..., .d=512]`, preserving all leading axes.
/// Parameters come from `flow_lm.transformer.layers.{index}.norm1`:
/// bf16 `weight` [512] and `bias` [512], converted to the input dtype. Statistics are computed
/// from the current input on every call.
///
/// Reference:
/// ```
/// input_layernorm: LayerNorm
/// ├── weight: shape=(512,), dtype=torch.float32
/// │   ├── loaded: model.layers.1.input_layernorm.weight
/// │   └── checkpoint: flow_lm.transformer.layers.1.norm1.weight
/// └── bias: shape=(512,), dtype=torch.float32
///     ├── loaded: model.layers.1.input_layernorm.bias
///     └── checkpoint: flow_lm.transformer.layers.1.norm1.bias
/// ```
const Input_Layer_Norm = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View) Input_Layer_Norm {
        return .{
            .weight = store.createReplicatedTensor("norm1.weight", .{.d}),
            .bias = store.createReplicatedTensor("norm1.bias", .{.d}),
            .eps = 1e-5, // TODO: shouldnt this be coming from json then?
        };
    }

    // -- Description --

    /// Normalizes each position before the decoder layer's attention branch.
    ///
    /// `input`: floating-point `[..., 512]`, the decoder layer's incoming residual features:
    /// prepared input projections for layer 0, or the preceding layer's output.
    /// Usually `[B, S, 512]`, or `[B, 1, 512]` during cached decode. This method
    /// tags only the final axis `.d`, preserving any leading axes.
    ///
    /// Returns floating-point `[..., .d=512]`, normalized and learned scale/shift features
    /// passed to self-attention. The parent retains `input` for the residual addition.
    ///
    /// Parameters and output use the input dtype. ZML's LayerNorm computes
    /// normalization statistics internally in f32 before converting back.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; the final axis of 512 features per token. Mean and
    ///   variance are computed across this axis, then learned weight and bias
    ///   (each shaped `[.d]`) scale and shift each normalized feature.
    /// - Leading axes are preserved. Conventionally `.b` is batch (independent
    ///   sequences) and `.s` is sequence (token positions). Statistics are separate
    ///   for every token in every sequence; neither axis is reduced or mixed.
    ///   Only the final axis is tagged here; untagged leading axes remain untagged.
    ///
    /// Shape: `[..., .d=512] -> [..., .d=512]`, preserving the floating-point dtype.
    pub fn forward(self: Input_Layer_Norm, input: zml.Tensor) zml.Tensor {
        std.debug.assert(input.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Input_Layer_Norm", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d}); // [...,512] -> [...,d=512]
        const weight = self.weight.convert(x.dtype()); // [d=512] bf16 -> [d=512]
        const bias = self.bias.convert(x.dtype()); // [d=512] bf16 -> [d=512]
        const norm: zml.nn.LayerNorm = .{ .weight = weight, .bias = bias, .eps = self.eps };

        compiler.pushLocationFmt(@src(), "layer[norm]", .{});
        const output = norm.forward(x); // [...,d=512] -> [...,d=512]
        compiler.popLocation();
        return output;
    }

    // -- Runtime --

    pub fn load(
        self: *const Input_Layer_Norm,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Input_Layer_Norm) {
        var buffers = try zml.mem.bufferize(allocator, Input_Layer_Norm, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Input_Layer_Norm, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Input_Layer_Norm)) void {
        zml.Buffer.deinitAll(Input_Layer_Norm, self);
    }
};

test Input_Layer_Norm {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Input_Layer_Norm = .{
        .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .eps = 1e-5,
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, .f32);
    var input: [2][2][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var weight_values: [512]zml.floats.BFloat16 = undefined;
    for (&weight_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var bias_values: [512]zml.floats.BFloat16 = undefined;
    for (&bias_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);

    var weight = try zml.Buffer.fromBytes(io, platform, layer.weight.shape(), std.mem.asBytes(&weight_values));
    defer weight.deinit();
    var bias = try zml.Buffer.fromBytes(io, platform, layer.bias.shape(), std.mem.asBytes(&bias_values));
    defer bias.deinit();
    const weight_buffers: zml.Bufferized(Input_Layer_Norm) = .{ .weight = weight, .bias = bias };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.input_layer_norm;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .s = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = 2, .d = 512 }, dtype)));
    }
}

// --- Post_Attention_Layer_Norm ---

/// Normalizes features after the attention residual addition, before the MLP branch.
///
/// Computes mean and population variance across each position's 512 features,
/// normalizes with epsilon 1e-5, then applies a learned per-feature scale and bias.
/// Batch rows and sequence positions retain separate statistics.
///
/// Input and output: floating-point `[..., .d=512]`, preserving all leading axes.
/// Parameters come from `flow_lm.transformer.layers.{index}.norm2`:
/// bf16 `weight` [512] and `bias` [512], converted to the input dtype. Statistics are computed
/// from the current input on every call.
///
/// Reference:
/// ```
/// post_attention_layernorm: LayerNorm
/// ├── weight: shape=(512,), dtype=torch.float32
/// │   ├── loaded: model.layers.1.post_attention_layernorm.weight
/// │   └── checkpoint: flow_lm.transformer.layers.1.norm2.weight
/// └── bias: shape=(512,), dtype=torch.float32
///     ├── loaded: model.layers.1.post_attention_layernorm.bias
///     └── checkpoint: flow_lm.transformer.layers.1.norm2.bias
/// ```
const Post_Attention_Layer_Norm = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View) Post_Attention_Layer_Norm {
        return .{
            .weight = store.createReplicatedTensor("norm2.weight", .{.d}),
            .bias = store.createReplicatedTensor("norm2.bias", .{.d}),
            .eps = 1e-5, // TODO: shouldnt this be coming from json then?
        };
    }

    // -- Description --

    /// Normalizes each position before the decoder layer's MLP branch.
    ///
    /// `input`: floating-point `[..., 512]`, the sum of the decoder layer's incoming features
    /// and its attention output. Usually `[B, S, 512]`, or `[B, 1, 512]` during
    /// cached decode. This method tags only the final axis `.d`, preserving any
    /// leading axes.
    ///
    /// Returns floating-point `[..., .d=512]`, normalized and learned scale/shift features
    /// passed to the MLP. The parent retains `input` for the next residual addition.
    ///
    /// Parameters and output use the input dtype. ZML's LayerNorm computes
    /// normalization statistics internally in f32 before converting back.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; the final axis of 512 features per token. Mean and
    ///   variance are computed across this axis, then learned weight and bias
    ///   (each shaped `[.d]`) scale and shift each normalized feature.
    /// - Leading axes are preserved. Conventionally `.b` is batch (independent
    ///   sequences) and `.s` is sequence (token positions). Statistics are separate
    ///   for every token in every sequence; neither axis is reduced or mixed.
    ///   Only the final axis is tagged here; untagged leading axes remain untagged.
    ///
    /// Shape: `[..., .d=512] -> [..., .d=512]`, preserving the floating-point dtype.
    pub fn forward(self: Post_Attention_Layer_Norm, input: zml.Tensor) zml.Tensor {
        std.debug.assert(input.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Post_Attention_Layer_Norm", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d}); // [...,512] -> [...,d=512]
        const weight = self.weight.convert(x.dtype()); // [d=512] bf16 -> [d=512]
        const bias = self.bias.convert(x.dtype()); // [d=512] bf16 -> [d=512]
        const norm: zml.nn.LayerNorm = .{ .weight = weight, .bias = bias, .eps = self.eps };

        compiler.pushLocationFmt(@src(), "layer[norm]", .{});
        const output = norm.forward(x); // [...,d=512] -> [...,d=512]
        compiler.popLocation();
        return output;
    }

    // -- Runtime --

    pub fn load(
        self: *const Post_Attention_Layer_Norm,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Post_Attention_Layer_Norm) {
        var buffers = try zml.mem.bufferize(allocator, Post_Attention_Layer_Norm, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Post_Attention_Layer_Norm, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Post_Attention_Layer_Norm)) void {
        zml.Buffer.deinitAll(Post_Attention_Layer_Norm, self);
    }
};

test Post_Attention_Layer_Norm {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const layer: Post_Attention_Layer_Norm = .{
        .weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .eps = 1e-5,
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, .f32);
    var input: [2][2][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| value.* = random.float(f32) * 2 - 1;
        }
    }
    var weight_values: [512]zml.floats.BFloat16 = undefined;
    for (&weight_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var bias_values: [512]zml.floats.BFloat16 = undefined;
    for (&bias_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);

    var weight = try zml.Buffer.fromBytes(io, platform, layer.weight.shape(), std.mem.asBytes(&weight_values));
    defer weight.deinit();
    var bias = try zml.Buffer.fromBytes(io, platform, layer.bias.shape(), std.mem.asBytes(&bias_values));
    defer bias.deinit();
    const weight_buffers: zml.Bufferized(Post_Attention_Layer_Norm) = .{ .weight = weight, .bias = bias };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.post_attention_layer_norm;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .s = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = 2, .d = 512 }, dtype)));
    }
}
