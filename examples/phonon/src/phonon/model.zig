const std = @import("std");
const zml = @import("zml");

// Graph construction runs on the host; concurrent compiler threads keep separate depths.
threadlocal var debug_depth: usize = 0;
threadlocal var debug_filter: ?[]const u8 = null;
threadlocal var debug_selected_depth: ?usize = null;

// A gentle reminder of the tags we're dealing with:
//
// | Tag     | Meaning                                                            | Size in this implementation                           |
// | ------- | ------------------------------------------------------------------ | ----------------------------------------------------: |
// | `.b`    | Batch: independent sequences                                       | 1 in the executable                                   |
// | `.s`    | Sequence positions                                                 | Prompt positions plus generated-history positions     |
// | `.d`    | Transformer or flow hidden feature dimension                       | 512                                                   |
// | `.dout` | A distinct output-feature axis during a projection                 | Depends on the layer                                  |
// | `.voc`  | Rows of an embedding vocabulary                                    | 4001 for text; 32 for speaker count                   |
// | `.p`    | Frames in the saved voice prompt                                   | 125 for the currently selected voice                  |
// | `.ad`   | Features per voice-prompt frame                                    | 512                                                   |
// | `.ld`   | Features per generated-audio latent / BOS latent                   | 32                                                    |
// | `.a`    | Completed generated-audio frames supplied as history               | 0 initially, then 1, 2, 3, …                          |
// | `.t`    | Scalar flow time per batch row                                     | 1; its value is progress within one frame's sampling  |
// | `.f`    | Timestep frequency/features                                        | 128 stored frequencies, then 256 cosine/sine features |
// | `.m`    | Condition-derived modulation features                              | 1536 for shift/scale/gate; 1024 for shift/scale       |
// | `.spk`  | Features in the speaker-count embedding                            | 16                                                    |
// | `.dur`  | Features in the duration encoding                                  | 128                                                   |
// | `.qkv`  | Packed projection output, or one chunk before splitting into heads | 1536, then 512                                        |
// | `.h`    | Attention heads                                                    | 8                                                     |
// | `.hd`   | Features per attention head                                        | 64                                                    |
// | `.q`    | Query positions in attention                                       | Same as `.s` during prefill                           |
// | `.k`    | Key/value positions in attention                                   | Same as `.s` during prefill                           |
// | `.hq`   | Internal SDPA query-head grouping axis                             | 1 here                                                |

// For reference, the entire pipeline:
//
// Text + selected voice
//   ↓
// Conversation template
//   ↓
// Expand voice placeholders
//   ↓
// Text normalization + tokenization
//   ↓
// Token IDs + voice latents + masks
//   ↓
// Embedding lookup + replace special positions with projected audio features
//   ↓
// Six transformer layers
//   ↓
// Final normalization
//   ↓
// Flow head + stop prediction
//   ↓
// Repeat to generate audio latent frames
//   ↓
// Convert latent representation for audio decoder
//   ↓
// Audio decoder neural network
//   ↓
// Waveform trimming / optional fading
//   ↓
// WAV file / playback

// --- Gradium_Phonon_TTS_For_Conditional_Generation ---

/// Combines the transformer, EOS head, and flow head for one audio-frame prediction.
///
/// The transformer builds contextual features from text, voice, and preceding
/// audio. Only its last position conditions the two heads: EOS produces a raw
/// stopping score, while the flow head predicts a velocity for a sampler-supplied
/// candidate audio latent and flow interval. The caller updates that candidate,
/// retains the completed frame, and applies the stopping policy.
///
/// `forward` consumes the prompt or one decode position and returns EOS scores
/// `[B, 1]` in f32, a completed frame `[B, 1, 32]` in the candidate dtype, and updated
/// audio history and active length. The one-step sampler adds the predicted
/// velocity to the candidate within the graph.
/// B counts independent sequences; active_lengths identifies the valid prefill prefix.
///
/// Parameters come from the root checkpoint view: `model` owns the transformer,
/// `flow_head` uses `flow_lm.flow_net`, and `eos_head` uses `flow_lm.out_eos`.
/// The children target phonon-31m-en and convert stored bf16 parameters to their
/// respective floating input dtypes.
/// Sampling, stopping, latent rescaling, and waveform decoding belong to callers.
///
/// Reference:
/// ```
/// GradiumPhononTTSForConditionalGeneration
/// ├── model: GradiumPhononTTSModel
/// │   ├── embed_tokens: Embedding
/// │   │   └── weight: shape=(4001, 512), dtype=torch.float32
/// │   │       ├── loaded: model.embed_tokens.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight
/// │   ├── layers: ModuleList
/// │   │   └── 0..5: GradiumPhononTTSDecoderLayer
/// │   │       ├── self_attn: GradiumPhononTTSAttention
/// │   │       │   ├── q_proj: Linear
/// │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │       ├── loaded: model.layers.5.self_attn.q_proj.weight
/// │   │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │   │       │   ├── k_proj: Linear
/// │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │       ├── loaded: model.layers.5.self_attn.k_proj.weight
/// │   │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │   │       │   ├── v_proj: Linear
/// │   │       │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │       ├── loaded: model.layers.5.self_attn.v_proj.weight
/// │   │       │   │       └── checkpoint: flow_lm.transformer.layers.5.self_attn.in_proj.weight
/// │   │       │   └── o_proj: Linear
/// │   │       │       └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │           ├── loaded: model.layers.5.self_attn.o_proj.weight
/// │   │       │           └── checkpoint: flow_lm.transformer.layers.5.self_attn.out_proj.weight
/// │   │       ├── mlp: GradiumPhononTTSMLP
/// │   │       │   ├── activation_fn: GELUActivation
/// │   │       │   ├── fc1: Linear
/// │   │       │   │   └── weight: shape=(2048, 512), dtype=torch.float32
/// │   │       │   │       ├── loaded: model.layers.5.mlp.fc1.weight
/// │   │       │   │       └── checkpoint: flow_lm.transformer.layers.5.linear1.weight
/// │   │       │   └── fc2: Linear
/// │   │       │       └── weight: shape=(512, 2048), dtype=torch.float32
/// │   │       │           ├── loaded: model.layers.5.mlp.fc2.weight
/// │   │       │           └── checkpoint: flow_lm.transformer.layers.5.linear2.weight
/// │   │       ├── input_layernorm: LayerNorm
/// │   │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │       │   │   ├── loaded: model.layers.5.input_layernorm.weight
/// │   │       │   │   └── checkpoint: flow_lm.transformer.layers.5.norm1.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: model.layers.5.input_layernorm.bias
/// │   │       │       └── checkpoint: flow_lm.transformer.layers.5.norm1.bias
/// │   │       └── post_attention_layernorm: LayerNorm
/// │   │           ├── weight: shape=(512,), dtype=torch.float32
/// │   │           │   ├── loaded: model.layers.5.post_attention_layernorm.weight
/// │   │           │   └── checkpoint: flow_lm.transformer.layers.5.norm2.weight
/// │   │           └── bias: shape=(512,), dtype=torch.float32
/// │   │               ├── loaded: model.layers.5.post_attention_layernorm.bias
/// │   │               └── checkpoint: flow_lm.transformer.layers.5.norm2.bias
/// │   ├── norm: LayerNorm
/// │   │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │   │   ├── loaded: model.norm.weight
/// │   │   │   └── checkpoint: flow_lm.out_norm.weight
/// │   │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       ├── loaded: model.norm.bias
/// │   │       └── checkpoint: flow_lm.out_norm.bias
/// │   ├── rotary_emb: GradiumPhononTTSRotaryEmbedding
/// │   ├── audio_prompt_proj: Linear
/// │   │   └── weight: shape=(512, 512), dtype=torch.float32
/// │   │       ├── loaded: model.audio_prompt_proj.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight
/// │   ├── audio_latent_proj: Linear
/// │   │   └── weight: shape=(512, 32), dtype=torch.float32
/// │   │       ├── loaded: model.audio_latent_proj.weight
/// │   │       └── checkpoint: flow_lm.input_linear.weight
/// │   ├── num_speakers_embed: Embedding
/// │   │   └── weight: shape=(32, 16), dtype=torch.float32
/// │   │       ├── loaded: model.num_speakers_embed.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.num_speakers.embed.weight
/// │   ├── num_speakers_proj: Linear
/// │   │   └── weight: shape=(512, 16), dtype=torch.float32
/// │   │       ├── loaded: model.num_speakers_proj.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.num_speakers.output_proj.weight
/// │   ├── duration_delta_proj: Linear
/// │   │   └── weight: shape=(512, 128), dtype=torch.float32
/// │   │       ├── loaded: model.duration_delta_proj.weight
/// │   │       └── checkpoint: flow_lm.condition_provider.conditioners.duration_delta.output_proj.weight
/// │   └── bos_latent: shape=(32,), dtype=torch.float32
/// │       ├── loaded: model.bos_latent
/// │       └── checkpoint: flow_lm.bos_emb
/// ├── flow_head: GradiumPhononTTSFlowHead
/// │   ├── input_proj: Linear
/// │   │   ├── weight: shape=(512, 32), dtype=torch.float32
/// │   │   │   ├── loaded: flow_head.input_proj.weight
/// │   │   │   └── checkpoint: flow_lm.flow_net.input_proj.weight
/// │   │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       ├── loaded: flow_head.input_proj.bias
/// │   │       └── checkpoint: flow_lm.flow_net.input_proj.bias
/// │   ├── cond_proj: Linear
/// │   │   ├── weight: shape=(512, 512), dtype=torch.float32
/// │   │   │   ├── loaded: flow_head.cond_proj.weight
/// │   │   │   └── checkpoint: flow_lm.flow_net.cond_embed.weight
/// │   │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       ├── loaded: flow_head.cond_proj.bias
/// │   │       └── checkpoint: flow_lm.flow_net.cond_embed.bias
/// │   ├── timestep_embeddings: ModuleList
/// │   │   └── 0..1: GradiumPhononTTSTimestepEmbedding
/// │   │       ├── fc1: Linear
/// │   │       │   ├── weight: shape=(512, 256), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.timestep_embeddings.1.fc1.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.timestep_embeddings.1.fc1.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.bias
/// │   │       ├── fc2: Linear
/// │   │       │   ├── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.timestep_embeddings.1.fc2.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.timestep_embeddings.1.fc2.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.bias
/// │   │       ├── act_fn: SiLUActivation
/// │   │       └── norm: GradiumPhononTTSTimestepNorm
/// │   │           └── weight: shape=(512,), dtype=torch.float32
/// │   │               ├── loaded: flow_head.timestep_embeddings.1.norm.weight
/// │   │               └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.3.alpha
/// │   ├── layers: ModuleList
/// │   │   └── 0..5: GradiumPhononTTSFlowResBlock
/// │   │       ├── norm: GradiumPhononTTSFlowLayerNorm
/// │   │       │   ├── weight: shape=(512,), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.layers.5.norm.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.layers.5.norm.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.bias
/// │   │       ├── fc1: Linear
/// │   │       │   ├── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.layers.5.fc1.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.layers.5.fc1.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.bias
/// │   │       ├── fc2: Linear
/// │   │       │   ├── weight: shape=(512, 512), dtype=torch.float32
/// │   │       │   │   ├── loaded: flow_head.layers.5.fc2.weight
/// │   │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.weight
/// │   │       │   └── bias: shape=(512,), dtype=torch.float32
/// │   │       │       ├── loaded: flow_head.layers.5.fc2.bias
/// │   │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.bias
/// │   │       ├── act_fn: SiLUActivation
/// │   │       └── adaln_proj: Linear
/// │   │           ├── weight: shape=(1536, 512), dtype=torch.float32
/// │   │           │   ├── loaded: flow_head.layers.5.adaln_proj.weight
/// │   │           │   └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.weight
/// │   │           └── bias: shape=(1536,), dtype=torch.float32
/// │   │               ├── loaded: flow_head.layers.5.adaln_proj.bias
/// │   │               └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.bias
/// │   └── final_layer: GradiumPhononTTSFlowFinalLayer
/// │       ├── norm: GradiumPhononTTSFlowLayerNorm
/// │       ├── adaln_proj: Linear
/// │       │   ├── weight: shape=(1024, 512), dtype=torch.float32
/// │       │   │   ├── loaded: flow_head.final_layer.adaln_proj.weight
/// │       │   │   └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.weight
/// │       │   └── bias: shape=(1024,), dtype=torch.float32
/// │       │       ├── loaded: flow_head.final_layer.adaln_proj.bias
/// │       │       └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.bias
/// │       ├── act_fn: SiLUActivation
/// │       └── proj: Linear
/// │           ├── weight: shape=(32, 512), dtype=torch.float32
/// │           │   ├── loaded: flow_head.final_layer.proj.weight
/// │           │   └── checkpoint: flow_lm.flow_net.final_layer.linear.weight
/// │           └── bias: shape=(32,), dtype=torch.float32
/// │               ├── loaded: flow_head.final_layer.proj.bias
/// │               └── checkpoint: flow_lm.flow_net.final_layer.linear.bias
/// └── eos_head: Linear
///     ├── weight: shape=(1, 512), dtype=torch.float32
///     │   ├── loaded: eos_head.weight
///     │   └── checkpoint: flow_lm.out_eos.weight
///     └── bias: shape=(1,), dtype=torch.float32
///         ├── loaded: eos_head.bias
///         └── checkpoint: flow_lm.out_eos.bias
/// ```
pub const Gradium_Phonon_TTS_For_Conditional_Generation = struct {
    model: Gradium_Phonon_TTS_Model,
    flow_head: Gradium_Phonon_TTS_Flow_Head,
    eos_head: EOS_Head,

    /// Returns one completed normalized audio frame and updated generation state.
    ///
    /// `eos_score`: floating-point `[.b=B, .dout=1]`, a raw stopping logit from the last
    /// transformer position. The host generation loop interprets it using its policy.
    /// `frame`: floating-point `[.b=B, .s=1, .ld=32]`, candidate plus predicted velocity.
    /// `history`: `[.b=B, .a=max_frames, .ld=32]`, including the completed frame.
    /// `active_lengths`: [2] i32, valid token positions followed by completed frames.
    /// Both counters increment by one. Frame, history, and active lengths donate
    /// their input storage; the caller retains returned handles.
    /// EOS follows the voice dtype; frame follows the candidate dtype. The caller
    /// releases the EOS buffer separately from the retained generation state.
    pub const Output = struct {
        eos_score: zml.Tensor,
        active_lengths: zml.Tensor,
        frame: zml.Tensor,
        history: zml.Tensor,
        history34: zml.Tensor,
        rng: zml.Tensor.Rng,
        /// Full-capacity K/V pairs with this call's positions updated; donate each layer's cache buffers.
        kv_caches: Gradium_Phonon_TTS_Model.KV_Caches,
    };

    /// Expects the root checkpoint view.
    pub fn init(store: zml.io.TensorStore.View, max_tokens: usize) Gradium_Phonon_TTS_For_Conditional_Generation {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .model = .init(store, max_tokens),
            .flow_head = .init(store.withPrefix("flow_lm.flow_net")),
            .eos_head = .init(store.withPrefix("flow_lm.out_eos")),
        };
    }

    /// Shared prefill/decode graph: consume incoming positions and generate one frame.
    ///
    /// `token_ids`: integer `[B, S]`, normally i64. Prefill uses the largest
    /// chunk's prompt width, padded with ID 6 after this chunk's valid prefix.
    /// Decode supplies one ID 6. The valid token count is in `active_lengths`.
    /// `attention_metadata` supplies backend runtime tensors (Metal's live-query
    /// count is zero to process every supplied query). `attention_parameters`
    /// selects the implementation while building the graph. Both are shared by
    /// all six layers and retained across calls; neither is donated.
    /// `voice`: floating-point `[B, P, 512]`, selected speaker features loaded from the voice
    /// asset as `[B, 512, P]` and transposed by the caller. P matches the number
    /// of ID-5 markers in the initial prompt, in frame order. Decode retains the
    /// same buffer, but only computes projections for the incoming position.
    /// `audio_latents`: floating-point `[B, A, 32]`, the generation loop's complete normalized
    /// audio history, with completed frames in its valid prefix and zeros initially.
    /// The graph selects only the relevant history slice before input projection.
    /// `candidate`: floating-point `[B, 32]`, the sampler's current candidate for the next
    /// frame in normalized latent space; freshly drawn noise in the one-step sampler.
    /// `active_lengths`: [2] i32 containing valid token positions and completed frames
    /// before this step. Both slots contain integer counts used directly for indexing.
    /// `start_time`: floating-point `[B, 1]`, starting flow progress supplied by the sampler.
    /// `end_time`: floating-point `[B, 1]`, ending flow progress supplied by the sampler.
    /// Times normally lie in [0, 1]; the current sampler uses 0 and 1. They describe
    /// flow progress, rather than audio timestamps. All inputs share batch size B.
    ///
    /// Returns EOS logits `[B, 1]`, a generated frame `[B, 1, 32]`, updated history,
    /// counters, RNG state and all six caches. The graph applies the one-step
    /// velocity update; the caller applies the stopping policy and retains state. EOS uses
    /// `voice.dtype()`; velocity uses `candidate.dtype()`. The transformer converts
    /// history and parameters to the voice dtype; the flow head converts its
    /// condition and times to the candidate dtype.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent prompts, candidates and sampling intervals.
    /// - `.s`: sequence; full-prefix positions, reduced to the final position for both heads.
    /// - `.p`: prompt frames; ordered voice frames, one per audio-prompt token.
    /// - `.ad`: 512 voice-prompt features per frame.
    /// - `.d`: 512 final transformer features conditioning both heads.
    /// - `.a`: completed generated frames in the full history; zero initially.
    /// - `.ld`: 32 normalized history/candidate/velocity components.
    /// - `.t`: one scalar flow time per batch row, for each interval endpoint.
    /// - `.dout`: one raw EOS score per batch row; no sigmoid or threshold applied.
    ///
    /// Inputs: integer IDs `[b,s]`, floating-point voice `[b,p,ad=512]`, complete
    /// history `[b,a,ld=32]`, candidate `[b,ld=32]`, and times `[b,t=1]` each.
    /// Outputs: eos_score `[b,dout=1]` in the voice dtype and velocity `[b,ld=32]`
    /// in the candidate dtype.
    pub fn forward(
        self: Gradium_Phonon_TTS_For_Conditional_Generation,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        audio_latents: zml.Tensor,
        history34: zml.Tensor,
        start_time: zml.Tensor,
        end_time: zml.Tensor,
        active_lengths: zml.Tensor,
        frame: zml.Tensor,
        rng: zml.Tensor.Rng,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) Output {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("token_ids", token_ids);
        debug_value("voice", voice);
        debug_value("audio_latents", audio_latents);
        debug_value("history34", history34);
        debug_value("start_time", start_time);
        debug_value("end_time", end_time);
        debug_value("active_lengths", active_lengths);
        debug_value("frame", frame);
        debug_value("rng", rng);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        std.debug.assert(active_lengths.dtype() == .i32 and active_lengths.rank() == 1 and active_lengths.dim(0) == 2);
        const active_token_length = active_lengths.slice(0, .single(0)); // [2] i32 -> scalar i32 token count
        debug_value("active_token_length", active_token_length);
        const active_frame_length = active_lengths.slice(0, .single(1)); // [2] i32 -> scalar i32 frame count
        debug_value("active_frame_length", active_frame_length);

        const next_rng, const noise = sample_noise(rng);
        debug_value("next_rng", next_rng);
        debug_value("noise", noise);

        // Prefill starts at zero; a one-position call consumes the latest audio frame.
        const start_position = active_token_length.subConstant(token_ids.dim(.s)).maximum(.constant(.init(.i32, 0))); // scalar i32 absolute token offset
        debug_value("start_position", start_position);
        const history_width = @min(audio_latents.dim(.a), token_ids.dim(.s));
        const history_start = active_frame_length.subConstant(history_width).maximum(.constant(.init(.i32, 0)));
        debug_value("history_start", history_start);
        const input_history = audio_latents.slice(.a, .dyn(history_start, history_width)); // [B,A,32] -> [B,min(A,S),32], same dtype
        debug_value("input_history", input_history);
        const input_voice = voice.slice(.p, .{ .len = @min(voice.dim(.p), token_ids.dim(.s)) }); // [B,P,512] -> [B,min(P,S),512], same dtype
        debug_value("input_voice", input_voice);
        const transformer_output = self.model.forward(token_ids, input_voice, input_history, start_position, active_token_length, attention_metadata, attention_parameters);
        debug_value("transformer_output", transformer_output);
        const hidden = transformer_output.hidden;
        debug_value("hidden", hidden);
        const local_length = active_token_length.sub(start_position); // scalar i32 global length -> valid positions in this call
        debug_value("local_length", local_length);
        const condition = last_valid_position(hidden, local_length);
        debug_value("condition", condition);
        const predict_output = self.predict(condition, noise, start_time, end_time);
        debug_value("predict_output", predict_output);

        const next_frame = noise.add(predict_output.velocity).insertAxes(1, .{.s});
        debug_value("next_frame", next_frame);
        const next_history = audio_latents.dynamicUpdateSlice(.{ .a = active_frame_length }, next_frame.rename(.{ .s = .a }));
        debug_value("next_history", next_history);

        const winsz = @min(next_history.dim(.a), 34);
        const next_history34 = next_history.slice(.a, .dyn(active_frame_length.addConstant(1).subConstant(winsz), winsz));
        debug_value("next_history34", next_history34);

        const next_active_lengths = active_lengths.addConstant(1); // [2] i32 -> [2] i32, advance both counts
        debug_value("next_active_lengths", next_active_lengths);

        const out: Output = .{
            .eos_score = predict_output.eos_score.convert(.f32), // Scalar host readback remains f32 for every transformer dtype.
            .active_lengths = next_active_lengths.reuseBuffer(active_lengths),
            .frame = next_frame.reuseBuffer(frame),
            .history = next_history.reuseBuffer(audio_latents),
            .history34 = next_history34.reuseBuffer(history34),
            .rng = next_rng,
            .kv_caches = transformer_output.kv_caches,
        };
        debug_value("out", out);

        std.debug.assert(active_lengths.shape().eqlWithTags(out.active_lengths.shape()));
        std.debug.assert(frame.shape().eqlWithTags(out.frame.shape()));
        std.debug.assert(audio_latents.shape().eqlWithTags(out.history.shape()));
        std.debug.assert(history34.shape().eqlWithTags(out.history34.shape()));
        std.debug.assert(rng._state.shape().eqlWithTags(out.rng._state.shape()));

        debug_value("output", out);
        return out;
    }

    fn sample_noise(rng: zml.Tensor.Rng) struct {
        zml.Tensor.Rng,
        zml.Tensor,
    } {
        const here = @typeName(@This()) ++ ".sample_noise";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("rng", rng);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".sample_noise", .{});
        defer compiler.popLocation();

        const shape: zml.Shape = .init(.{ .b = 1, .ld = 32 }, .f32);

        const rng1, const u = rng.uniform(shape, .{});
        debug_value("rng1", rng1);
        debug_value("u", u);
        const rng2, const v = rng1.uniform(shape, .{});
        debug_value("rng2", rng2);
        debug_value("v", v);

        // Box–Muller: two uniform samples -> one standard Gaussian sample.
        const one_minus_u = u.scale(-1).addConstant(1);
        debug_value("one_minus_u", one_minus_u);
        const radius = one_minus_u.log().scale(-2).sqrt();
        debug_value("radius", radius);
        const angle = v.scale(2 * std.math.pi);
        debug_value("angle", angle);
        const gaussian = radius.mul(angle.cos());
        debug_value("gaussian", gaussian);

        const noise = gaussian.scale(@sqrt(@as(f32, 0.3)));
        debug_value("noise", noise);
        debug_value("output.rng", rng2);
        debug_value("output.noise", noise);
        return .{ rng2, noise };
    }

    fn last_valid_position(hidden: zml.Tensor, active_token_length: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".last_valid_position";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("hidden", hidden);
        debug_value("active_token_length", active_token_length);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".last_valid_position", .{});
        defer compiler.popLocation();

        const index = active_token_length.addConstant(-1);
        debug_value("index", index);
        const debug_output: zml.Tensor = hidden.slice(.s, .dynSingle(index));
        debug_value("output", debug_output);
        return debug_output;
    }

    pub const Prediction_Output = struct {
        eos_score: zml.Tensor,
        velocity: zml.Tensor,
    };

    /// Context [B,512], candidate [B,32], and interval endpoints [B,1] ->
    /// EOS [B,1] in context dtype and velocity [B,32] in candidate dtype.
    fn predict(
        self: Gradium_Phonon_TTS_For_Conditional_Generation,
        condition: zml.Tensor,
        candidate: zml.Tensor,
        start: zml.Tensor,
        end: zml.Tensor,
    ) Prediction_Output {
        const here = @typeName(@This()) ++ ".predict";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("condition", condition);
        debug_value("candidate", candidate);
        debug_value("start", start);
        debug_value("end", end);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".predict", .{});
        defer compiler.popLocation();

        const debug_output: Prediction_Output = .{
            .eos_score = self.eos_head.forward(condition),
            .velocity = self.flow_head.forward(candidate, condition, start, end),
        };
        debug_value("output", debug_output);
        return debug_output;
    }

    pub fn bufferize(
        self: *const Gradium_Phonon_TTS_For_Conditional_Generation,
        allocator: std.mem.Allocator,
    ) !zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation) {
        return .{
            .model = try self.model.bufferize(allocator),
            .flow_head = try self.flow_head.bufferize(allocator),
            .eos_head = try zml.mem.bufferize(allocator, EOS_Head, &self.eos_head),
        };
    }
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
/// It performs embedding lookup, input replacement, all six decoder layers,
/// and final normalization. The caller supplies matching token IDs, voice frames,
/// and audio latents for the positions being processed.
///
/// Inputs include integer IDs `[B, S]`, floating-point voice `[B, P, 512]`, and
/// normalized audio history `[B, A, 32]`. Voice determines the compute/output dtype;
/// audio history and parameters are converted to match. Output features are
/// `[B, S, 512]`, one vector per supplied sequence position.
/// B counts sequences, P voice frames, and A completed generated frames.
///
/// Parameters come from the root checkpoint view: transcript embeddings, input
/// projections, `flow_lm.transformer.layers.0..5`, and `flow_lm.out_norm`.
/// Stored bf16 tensors are converted to the voice dtype. ZML evaluates normalization
/// statistics, RoPE trigonometry, softmax, and erf internally in f32, then converts
/// back. This port targets phonon-31m-en with 8 attention heads of 64 features,
/// RoPE theta 10000, and norm epsilon 1e-5.
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
    input_projection: Transformer_Input_Projection,
    layers: [num_layers]Gradium_Phonon_TTS_Decoder_Layer,
    norm: zml.nn.LayerNorm,

    // FIX: This port currently targets the six-layer phonon-31m-en checkpoint (here and everywhere else).
    const num_layers = 6;

    pub const KV_Caches = [num_layers]Gradium_Phonon_TTS_Attention.KV_Cache;

    pub const Output = struct {
        hidden: zml.Tensor,
        kv_caches: KV_Caches,
    };

    pub fn init(store: zml.io.TensorStore.View, max_tokens: usize) Gradium_Phonon_TTS_Model {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

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
            layer.* = .init(store.withPrefix("flow_lm.transformer.layers").withLayer(i), max_tokens);
        }
        return self;
    }

    /// Runs input preparation, all six decoder layers, and final normalization.
    ///
    /// `token_ids`: integer `[B, S]`, normally i64, the complete unpadded sequence
    /// from the tokenizer and generation loop. Contains text and control IDs;
    /// each completed audio frame adds one ID 6. S must be nonzero.
    /// `voice`: floating-point `[B, P, 512]`, ordered speaker features loaded from the voice
    /// asset and transposed by the caller from `[B, 512, P]`. Each row has P ID-5 markers.
    /// `audio_latents`: floating-point `[B, A, 32]`, the generation loop's complete normalized
    /// frame history, ordered like the A ID-6 markers. Initially A=0.
    /// All inputs share batch size B; positions start at zero on every call.
    /// Computation uses `voice.dtype()`; embeddings, parameters, and audio history
    /// are converted to that dtype.
    ///
    /// Returns `[B, S, 512]` in `voice.dtype()`, final normalized contextual features
    /// for every position. The prediction wrapper selects the last position for both heads.
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
    /// Inputs: integer IDs `[.b, .s]`, floating-point voice `[.b, .p, .ad=512]`,
    /// and floating-point complete generated history `[.b, .a, .ld=32]`.
    /// Output: `[.b, .s, .d=512]` in `voice.dtype()`.
    /// `start_position`: scalar i32 absolute index of the first incoming position.
    /// `active_token_length`: scalar i32 valid cache-prefix length, including this input.
    /// Returns the incoming positions' features as `hidden`, plus six full-capacity
    /// K/V pairs in `kv_caches`, with the incoming range updated and old entries retained.
    pub fn forward(
        self: Gradium_Phonon_TTS_Model,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        audio_latents: zml.Tensor,
        start_position: zml.Tensor,
        active_token_length: zml.Tensor,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) Output {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("token_ids", token_ids);
        debug_value("voice", voice);
        debug_value("audio_latents", audio_latents);

        debug_value("start_position", start_position);
        debug_value("active_token_length", active_token_length);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const dtype = voice.dtype();
        std.debug.assert(dtype.isFloat() and audio_latents.dtype().isFloat());

        var hidden = self.prepare_input(token_ids, voice, audio_latents);
        debug_value("hidden", hidden);
        var kv_caches: KV_Caches = undefined;
        inline for (self.layers, 0..) |layer, i| {
            const layer_output = layer.forward(i, hidden, start_position, active_token_length, attention_metadata, attention_parameters);
            hidden = layer_output.hidden;
            kv_caches[i] = layer_output.kv_cache;
            debug_value("hidden", hidden);
            debug_value(std.fmt.comptimePrint("kv_caches[{}]", .{i}), kv_caches[i]);
        }

        const debug_output: Output = .{
            .hidden = self.normalize_output(hidden),
            .kv_caches = kv_caches,
        };
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Token IDs [B,S], voice [B,P,512], and latents [B,A,32] -> [B,S,512].
    /// Voice determines compute dtype.
    fn prepare_input(self: Gradium_Phonon_TTS_Model, token_ids: zml.Tensor, voice: zml.Tensor, audio_latents: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".prepare_input";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("token_ids", token_ids);
        debug_value("voice", voice);
        debug_value("audio_latents", audio_latents);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".prepare_input", .{});
        defer compiler.popLocation();

        const embeddings = self.embed_tokens.forward(token_ids).convert(voice.dtype());
        debug_value("embeddings", embeddings);
        const debug_output: zml.Tensor = self.input_projection.forward(embeddings, token_ids, voice, audio_latents);
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Apply the learned final LayerNorm to [..., .d=512], preserving dtype.
    fn normalize_output(self: Gradium_Phonon_TTS_Model, hidden: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".normalize_output";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("hidden", hidden);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".normalize_output", .{});
        defer compiler.popLocation();

        const norm: zml.nn.LayerNorm = .{
            .weight = self.norm.weight.convert(hidden.dtype()),
            .bias = self.norm.bias.?.convert(hidden.dtype()),
            .eps = self.norm.eps,
        };
        const debug_output: zml.Tensor = norm.forward(hidden);
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Prepare buffer metadata, handling the fixed layer array explicitly.
    pub fn bufferize(self: *const Gradium_Phonon_TTS_Model, allocator: std.mem.Allocator) !zml.Bufferized(Gradium_Phonon_TTS_Model) {
        var buffers: zml.Bufferized(Gradium_Phonon_TTS_Model) = .{
            .embed_tokens = try zml.mem.bufferize(allocator, Embedding, &self.embed_tokens),
            .input_projection = try zml.mem.bufferize(allocator, Transformer_Input_Projection, &self.input_projection),
            .norm = try zml.mem.bufferize(allocator, zml.nn.LayerNorm, &self.norm),
            .layers = undefined,
        };
        for (&self.layers, &buffers.layers) |*layer, *buffer| {
            buffer.* = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Decoder_Layer, layer);
        }
        return buffers;
    }
};

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
//
// TODO: one could technically imagine caching all the embedding+voice/audio/BOS replacement stuff
const Embedding = struct {
    weight: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Embedding {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

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
    /// `Transformer_Input_Projection` for voice/audio replacements before transformer context mixing.
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
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isInteger());
        std.debug.assert(self.weight.dtype().isFloat());
        std.debug.assert(input.rank() == 2);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const ids = input.withTags(.{ .b, .s }); // [B,S] integer -> [b=B,s=S] integer
        debug_value("ids", ids);
        const debug_output: zml.Tensor = self.weight.gather(.{ .voc = ids }, .{});
        debug_value("output", debug_output);
        return debug_output; // [voc=4001,d=512] + [B,S] integer indices -> [B,S,d=512], preserving weight dtype
    }
};

// --- Transformer_Input_Projection ---

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
/// P and A may be zero.
///
/// Parameters come from the root checkpoint view: speaker feature projection
/// `flow_lm.condition_provider.conditioners.speaker_wavs.output_proj`, audio
/// projection `flow_lm.input_linear`, learned latent `flow_lm.bos_emb`, and the
/// `num_speakers` and `duration_delta` conditioners. All projections are bias-free;
/// stored bf16 tensors, voice, and audio history are converted to the embeddings' dtype.
/// Host fields default to one speaker and
/// zero duration delta; the latter generates 128 sinusoidal features in host f32,
/// then converts them to the embeddings' dtype.
const Transformer_Input_Projection = struct {
    audio_prompt_proj: zml.nn.Linear,
    audio_latent_proj: zml.nn.Linear,
    bos_latent: zml.Tensor,
    num_speakers_embed: zml.Tensor,
    num_speakers_proj: zml.nn.Linear,
    duration_delta_proj: zml.nn.Linear,

    num_speakers: i64 = 1,
    duration_delta: f32 = 0,

    pub fn init(store: zml.io.TensorStore.View) Transformer_Input_Projection {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

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
    /// The full-prefix path passes the whole history; the initial pass has A=0.
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
        self: Transformer_Input_Projection,
        input: zml.Tensor,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        audio_latents: zml.Tensor,
    ) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);
        debug_value("token_ids", token_ids);
        debug_value("voice", voice);
        debug_value("audio_latents", audio_latents);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and input.rank() == 3);
        std.debug.assert(token_ids.dtype().isInteger() and token_ids.rank() == 2);
        std.debug.assert(voice.dtype().isFloat() and voice.rank() == 3);
        std.debug.assert(audio_latents.dtype().isFloat() and audio_latents.rank() == 3);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const embeddings = input.withTags(.{ .b, .s, .d });
        debug_value("embeddings", embeddings);
        const ids = token_ids.withTags(.{ .b, .s });
        debug_value("ids", ids);
        const voice_frames = voice.convert(dtype).withTags(.{ .b, .p, .ad });
        debug_value("voice_frames", voice_frames);
        const audio_frames = audio_latents.convert(dtype).withTags(.{ .b, .a, .ld });
        debug_value("audio_frames", audio_frames);
        std.debug.assert(embeddings.dim(.b) == ids.dim(.b) and embeddings.dim(.s) == ids.dim(.s));
        std.debug.assert(embeddings.dim(.b) == voice_frames.dim(.b));
        std.debug.assert(embeddings.dim(.b) == audio_frames.dim(.b) and audio_frames.dim(.ld) == 32);

        const condition = self.speaker_condition(dtype).add(self.duration_condition(dtype));
        debug_value("condition", condition);
        const bos = self.audio_latent_proj.forward(self.bos_latent.convert(dtype), dtype).add(condition);
        debug_value("bos", bos);
        const bos_mask = ids.cmp(.EQ, zml.Tensor.scalar(4, ids.dtype())).broad(embeddings.shape());
        debug_value("bos_mask", bos_mask);
        var output = bos_mask.select(bos.broad(embeddings.shape()), embeddings);
        debug_value("output", output);

        // Empty histories have nothing to gather; the initial pass has no generated audio.
        if (voice_frames.dim(.p) > 0) {
            const projected = self.audio_prompt_proj.forward(voice_frames, dtype);
            debug_value("projected", projected);
            output = replace_frame_markers(output, ids, 5, projected, .p);
            debug_value("output", output);
        }
        if (audio_frames.dim(.a) > 0) {
            const projected = self.audio_latent_proj.forward(audio_frames, dtype);
            debug_value("projected", projected);
            const conditioned = projected.add(condition.broad(projected.shape()));
            debug_value("conditioned", conditioned);
            output = replace_frame_markers(output, ids, 6, conditioned, .a);
            debug_value("output", output);
        }
        debug_value("output", output);
        return output;
    }

    /// Host speaker count -> learned [.d=512] conditioning vector.
    fn speaker_condition(self: Transformer_Input_Projection, dtype: zml.DataType) zml.Tensor {
        const here = @typeName(@This()) ++ ".speaker_condition";

        debug_enter(here);
        defer debug_exit(here);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".speaker_condition", .{});
        defer compiler.popLocation();

        const count = zml.Tensor.constant(.{ .i64 = self.num_speakers });
        debug_value("count", count);
        const speaker = self.num_speakers_embed.gather(.{ .voc = count }, .{}).convert(dtype);
        debug_value("speaker", speaker);
        const debug_output: zml.Tensor = self.num_speakers_proj.forward(speaker, dtype);
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Host duration delta -> sinusoidal [.dur=128] features -> [.d=512].
    /// The scalar is fixed while tracing; this does not run a host loop per token.
    fn duration_condition(self: Transformer_Input_Projection, dtype: zml.DataType) zml.Tensor {
        const here = @typeName(@This()) ++ ".duration_condition";

        debug_enter(here);
        defer debug_exit(here);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".duration_condition", .{});
        defer compiler.popLocation();

        var features: [128]f32 = undefined;
        for (0..64) |i| {
            const exponent = @as(f32, @floatFromInt(i)) / 63.0;
            const phase = 1000.0 * self.duration_delta / std.math.pow(f32, 10000.0, exponent);
            features[i] = @cos(phase);
            features[i + 64] = @sin(phase);
        }
        const duration = zml.Tensor.constantTensor(.init(.{ .dur = 128 }, .f32), std.mem.asBytes(&features)).convert(dtype);
        debug_value("duration", duration);
        const debug_output: zml.Tensor = self.duration_delta_proj.forward(duration, dtype);
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Insert projected frames [B,frame_axis,512] at the matching IDs in [B,S].
    /// The first marker takes frame 0, the second frame 1, independently per row.
    /// Non-marker positions retain their embeddings. The caller skips empty frames.
    fn replace_frame_markers(embeddings: zml.Tensor, ids: zml.Tensor, marker_id: i64, frames: zml.Tensor, comptime frame_axis: @EnumLiteral()) zml.Tensor {
        const here = @typeName(@This()) ++ ".replace_frame_markers";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("embeddings", embeddings);
        debug_value("ids", ids);
        debug_value("frames", frames);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".replace_frame_markers", .{});
        defer compiler.popLocation();

        const mask = ids.cmp(.EQ, zml.Tensor.scalar(marker_id, ids.dtype()));
        debug_value("mask", mask);
        const indices = mask.convert(.i32).cumulativeSum(.s).subConstant(1).maximum(zml.Tensor.scalar(0, .i32));
        debug_value("indices", indices);
        // Both .b axes are shared, so gather selects frames within each batch row.
        const indexed_frames = frames.renameTag(frame_axis, .frame);
        debug_value("indexed_frames", indexed_frames);
        const replacements = indexed_frames.gather(.{ .frame = indices }, .{});
        debug_value("replacements", replacements);
        const debug_output: zml.Tensor = mask.broad(embeddings.shape()).select(replacements, embeddings);
        debug_value("output", debug_output);
        return debug_output;
    }
};

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
/// Input and output are floating-point `[B, S, 512]`, preserving the input dtype;
/// batch rows remain independent.
///
/// `forward` processes an incoming block at its absolute position, using
/// each layer's retained K/V history. The block may be full-width or one position.
///
/// Each layer has separate bf16 parameters under
/// `flow_lm.transformer.layers.{index}`: `norm1`, `self_attn.in_proj`,
/// `self_attn.out_proj`, `norm2`, `linear1`, and `linear2`.
/// Parameters are converted to the input dtype; norm epsilon is 1e-5 and GELU
/// uses erf. ZML evaluates normalization statistics, RoPE trigonometry, attention
/// softmax, and erf internally in f32, then converts back to the input dtype.
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

    pub const Output = struct {
        hidden: zml.Tensor,
        kv_cache: Gradium_Phonon_TTS_Attention.KV_Cache,
    };

    pub fn init(store: zml.io.TensorStore.View, max_tokens: usize) Gradium_Phonon_TTS_Decoder_Layer {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .input_layernorm = .init(store),
            .self_attn = .init(store, max_tokens),
            .post_attention_layernorm = .init(store),
            .mlp = .init(store),
        };
    }

    /// Runs a complete sequence through attention and MLP residual branches.
    ///
    /// `input`: floating-point `[B, S, 512]`, prepared model inputs for the first decoder
    /// layer or residual features from the preceding layer. This method assigns
    /// tags `.b`, `.s`, and `.d`; the valid token count excludes padded cache slots.
    ///
    /// Returns `[B, S, 512]` in `input.dtype()`, after both residual additions:
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
    /// Shape: `[.b, .s, .d=512] -> [.b, .s, .d=512]`, preserving the input dtype.
    /// `idx` identifies this repeated layer in compiler source locations.
    /// Returns the features as `hidden` alongside this layer's updated `kv_cache`.
    /// Scalar i32 `start_position` locates the incoming block; scalar i32
    /// `active_token_length` bounds the valid cache prefix, including this block.
    pub fn forward(self: Gradium_Phonon_TTS_Decoder_Layer, comptime idx: usize, input: zml.Tensor, start_position: zml.Tensor, active_token_length: zml.Tensor, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) Output {
        const here = std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx });

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        debug_value("start_position", start_position);
        debug_value("active_token_length", active_token_length);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat() and input.rank() == 3);
        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx }), .{});
        defer compiler.popLocation();

        const hidden = input.withTags(.{ .b, .s, .d });
        debug_value("hidden", hidden);
        const normalized = self.input_layernorm.forward(hidden);
        debug_value("normalized", normalized);
        const attention = self.self_attn.forward(normalized, start_position, active_token_length, attention_metadata, attention_parameters);
        debug_value("attention", attention);
        const residual = hidden.add(attention.hidden);
        debug_value("residual", residual);
        const debug_output: Output = .{
            .hidden = self.add_mlp_residual(residual),
            .kv_cache = attention.kv_cache,
        };
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Post-attention residual [B,S,512] -> normalize, MLP, residual addition.
    fn add_mlp_residual(self: Gradium_Phonon_TTS_Decoder_Layer, residual: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".add_mlp_residual";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("residual", residual);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".add_mlp_residual", .{});
        defer compiler.popLocation();

        const normalized = self.post_attention_layernorm.forward(residual);
        debug_value("normalized", normalized);
        const update = self.mlp.forward(normalized);
        debug_value("update", update);
        const debug_output: zml.Tensor = residual.add(update);
        debug_value("output", debug_output);
        return debug_output;
    }
};

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
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

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
    /// Usually `[B, S, 512]`. This method
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
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d});
        debug_value("x", x);
        const weight = self.weight.convert(x.dtype());
        debug_value("weight", weight);
        const bias = self.bias.convert(x.dtype());
        debug_value("bias", bias);
        const norm: zml.nn.LayerNorm = .{ .weight = weight, .bias = bias, .eps = self.eps };

        const output = norm.forward(x);
        debug_value("output", output);
        debug_value("output", output);
        return output;
    }
};

// --- Gradium_Phonon_TTS_Attention ---

/// Lets each sequence position gather information from its causal context.
///
/// A packed projection creates queries, keys, and values for eight heads of
/// 64 features. RoPE rotates queries and keys using absolute sequence positions.
/// Scaled dot-product attention weights visible values, then an output projection
/// combines the heads into 512 features. The decoder layer adds the residual.
///
/// Input and output: floating-point `[B, S, 512]`, preserving dtype.
/// `forward` processes the entire supplied sequence starting at zero.
/// Inputs must be unpadded.
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

    kv_cache: KV_Cache,

    pub const Output = struct {
        hidden: zml.Tensor,
        kv_cache: KV_Cache,
    };

    const num_heads = 8;
    const head_dim = 64;

    const KV_Cache = struct {
        k: zml.Tensor,
        v: zml.Tensor,

        const Bufferized = zml.Bufferized(KV_Cache);

        // pub fn init(kv_shape: zml.Shape, mesh: *const zml.Mesh) KV_Cache {
        //     const shape_sharded = kv_shape.withPartitioning(mesh, .{ .h = .model });
        //
        //     return .{
        //         .k = .fromShape(shape_sharded),
        //         .v = .fromShape(shape_sharded),
        //     };
        // }
        pub fn init(kv_shape: zml.Shape) KV_Cache {
            return .{ .k = .fromShape(kv_shape), .v = .fromShape(kv_shape) };
        }

        pub fn deinit(self: Bufferized) void {
            self.k.deinit();
            self.v.deinit();
        }

        pub fn reuse_buffers(self: KV_Cache, other: KV_Cache) KV_Cache {
            std.debug.assert(self.k.shape().eqlWithTags(other.k.shape()));
            std.debug.assert(self.v.shape().eqlWithTags(other.v.shape()));
            return .{
                .k = self.k.reuseBuffer(other.k),
                .v = self.v.reuseBuffer(other.v),
            };
        }
    };

    pub fn init(store: zml.io.TensorStore.View, max_tokens: usize) Gradium_Phonon_TTS_Attention {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .qkv_proj = .init(store.createReplicatedTensor("self_attn.in_proj.weight", .{ .qkv, .d }), null, .d),
            .out_proj = .init(store.createReplicatedTensor("self_attn.out_proj.weight", .{ .dout, .d }), null, .d),
            .kv_cache = .init(.init(.{ .b = 1, .k = max_tokens, .h = num_heads, .hd = head_dim }, .f32)),
        };
    }

    /// Shared cached causal attention for prompt-width prefill and one-position decode.
    ///
    /// `input`: floating [B, S, 512], pre-attention features from Input_Layer_Norm.
    /// All S incoming positions are processed together at scalar i32 `start_position`.
    /// Scalar i32 `active_token_length` is the valid cache prefix after updating it.
    /// Valid prefill queries end at active_token_length - 1; later queries are
    /// padding and their outputs are discarded. Decode supplies one valid query.
    /// The attention API's causal bound excludes unused cache entries from every
    /// valid query without a separate validity mask.
    /// Returns projected features [B, S, 512] in input.dtype(), before the caller's
    /// residual addition. Parameters are converted to the input dtype.
    /// Returns the features as `hidden` and a `kv_cache` with the incoming rotated
    /// K and projected V written into their absolute slots. Decode reads the other
    /// retained entries. Prefill writes only the prompt prefix and attends directly
    /// to its projected K/V; unused cache slots remain zero until decode fills them.
    ///
    /// Tags: .b independent sequences, .s input/output positions, .d model features.
    /// Attention uses .q for the S incoming queries and .k for the prompt during
    /// prefill or the full cache capacity during decode;
    /// .h=8 heads and .hd=64 coordinates within each head. Future keys are masked.
    pub fn forward(self: Gradium_Phonon_TTS_Attention, input: zml.Tensor, start_position: zml.Tensor, active_token_length: zml.Tensor, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) Output {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        debug_value("start_position", start_position);
        debug_value("active_token_length", active_token_length);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const hidden = input.withTags(.{ .b, .s, .d });
        debug_value("hidden", hidden);
        const projected = self.project_qkv(hidden);
        debug_value("projected", projected);
        const relative_positions = zml.Tensor.arange(.{ .end = hidden.dim(.s) }, .i32).withTags(.{.s}); // [S] i32 local positions
        debug_value("relative_positions", relative_positions);
        const positions = relative_positions.add(start_position.broad(relative_positions.shape())); // [S] i32 -> [S] i32 absolute positions
        debug_value("positions", positions);
        const rotated = apply_rope(projected, positions);
        debug_value("rotated", rotated);
        const next_cache: KV_Cache = .{
            .k = self.kv_cache.k.dynamicUpdateSlice(.{ .k = start_position }, rotated.keys), // [B,K,H,HD] + [B,S,H,HD] -> [B,K,H,HD], same dtype
            .v = self.kv_cache.v.dynamicUpdateSlice(.{ .k = start_position }, rotated.values), // [B,K,H,HD] + [B,S,H,HD] -> [B,K,H,HD], same dtype
        };
        debug_value("next_cache", next_cache);
        // The graph is compiled for either the whole initial prompt or one
        // incoming decode position; this choice is resolved while building it.
        const keys = if (hidden.dim(.s) > 1) rotated.keys else next_cache.k;
        debug_value("keys", keys);
        const values = if (hidden.dim(.s) > 1) rotated.values else next_cache.v;
        debug_value("values", values);
        const debug_output: Output = .{
            .hidden = self.attend_and_project(rotated.queries, keys, values, start_position, attention_metadata, attention_parameters),
            .kv_cache = next_cache.reuse_buffers(self.kv_cache),
        };
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Before RoPE all fields are [B, new_positions, 8, 64], tagged .b/.s/.h/.hd.
    /// After RoPE, queries use .q instead of .s and keys/values use .k instead.
    const QKV = struct {
        queries: zml.Tensor,
        keys: zml.Tensor,
        values: zml.Tensor,
    };

    /// [B, new_positions, 512] -> three head-split projections. The packed
    /// projection axis .qkv holds 1536 features: queries, then keys, then values.
    fn project_qkv(self: Gradium_Phonon_TTS_Attention, hidden: zml.Tensor) QKV {
        const here = @typeName(@This()) ++ ".project_qkv";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("hidden", hidden);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".project_qkv", .{});
        defer compiler.popLocation();

        std.debug.assert(hidden.dtype().isFloat() and hidden.rank() == 3);
        std.debug.assert(hidden.dim(.d) == num_heads * head_dim);
        const packed_features = self.qkv_proj.forward(hidden, hidden.dtype());
        debug_value("packed_features", packed_features);
        const projections = packed_features.chunkExact(.qkv, 3);
        debug_value("projections", projections);
        const debug_output: QKV = .{
            .queries = projections[0].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }),
            .keys = projections[1].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }),
            .values = projections[2].splitAxis(.qkv, .{ .h = num_heads, .hd = head_dim }),
        };
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Rotate Q/K at absolute positions [.s], or positions 0..S-1 when null.
    /// The checkpoint packs adjacent real/imaginary coordinates, so RoPE must
    /// use interleaved pairs. Values are not rotated. Assign attention axis tags.
    fn apply_rope(projected: QKV, positions: ?zml.Tensor) QKV {
        const here = @typeName(@This()) ++ ".apply_rope";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("positions", positions);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".apply_rope", .{});
        defer compiler.popLocation();

        const options: zml.nn.RopeOpts = .{
            .layout = .interleaved,
            .scaling = .{ .default = .{ .rope_theta = 10000 } },
        };
        const debug_output: QKV = .{
            .queries = zml.nn.rope(projected.queries, positions, options).rename(.{ .s = .q }),
            .keys = zml.nn.rope(projected.keys, positions, options).rename(.{ .s = .k }),
            .values = projected.values.rename(.{ .s = .k }),
        };
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Attend from [B, new_positions, 8, 64] queries to keys/values, then merge
    /// heads and project to [B, new_positions, 512], tagged .b/.s/.d.
    /// .dout distinguishes the output projection's result from its contracted .d axis.
    /// `start_position` is the scalar absolute offset of the first query, not a
    /// vector of all query positions. The attention backend adds each local offset.
    /// Q/K/V, positions, metadata, and result are replicated on the current mesh.
    fn attend_and_project(
        self: Gradium_Phonon_TTS_Attention,
        queries: zml.Tensor,
        keys: zml.Tensor,
        values: zml.Tensor,
        start_position: zml.Tensor,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) zml.Tensor {
        const here = @typeName(@This()) ++ ".attend_and_project";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("queries", queries);
        debug_value("keys", keys);
        debug_value("values", values);
        debug_value("start_position", start_position);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".attend_and_project", .{});
        defer compiler.popLocation();

        // Query j may read keys through start_position + j. Padded prefill
        // queries are independent of earlier valid queries and discarded later.
        const attended = zml.attention.attention(
            queries,
            keys,
            values,
            start_position,
            attention_metadata,
            attention_parameters,
            compiler.platform.replicated_mesh,
            .{
                .in_specs = .{ .q = .replicated, .k = .replicated, .v = .replicated },
                .out_specs = .replicated,
            },
        );
        debug_value("attended", attended);
        const features = attended.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        debug_value("features", features);
        const debug_output: zml.Tensor = self.out_proj.forward(features, queries.dtype()).rename(.{ .dout = .d });
        debug_value("output", debug_output);
        return debug_output;
    }
};

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
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

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
    /// and its attention output. Usually `[B, S, 512]`. This method tags only
    /// the final axis `.d`, preserving any
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
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d});
        debug_value("x", x);
        const weight = self.weight.convert(x.dtype());
        debug_value("weight", weight);
        const bias = self.bias.convert(x.dtype());
        debug_value("bias", bias);
        const norm: zml.nn.LayerNorm = .{ .weight = weight, .bias = bias, .eps = self.eps };

        const output = norm.forward(x);
        debug_value("output", output);
        debug_value("output", output);
        return output;
    }
};

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
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

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
    /// Usually `[B, S, 512]`. This method
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
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d});
        debug_value("x", x);

        const hidden = self.fc1.forward(x, x.dtype());
        debug_value("hidden", hidden);

        const activated = gelu_erf(hidden);
        debug_value("activated", activated);

        const output = self.fc2.forward(activated, activated.dtype());
        debug_value("output", output);
        debug_value("output", output);
        return output;
    }

    fn gelu_erf(x: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".gelu_erf";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("x", x);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".gelu_erf", .{});
        defer compiler.popLocation();
        // Evaluate GELU in f32, then restore the activation dtype at the boundary.
        const wide = x.convert(.f32);
        const erf_term = wide.scale(1.0 / std.math.sqrt(2.0)).erf();
        debug_value("erf_term", erf_term);
        const debug_output: zml.Tensor = wide.scale(0.5).mul(erf_term.addConstant(1.0)).convert(x.dtype());
        debug_value("output", debug_output);
        return debug_output;
    }
};

/// Scores whether speech generation should end (EOS: end of speech).
///
/// A learned linear layer combines 512 final, normalized transformer features
/// into one score: `score = dot(features, weight) + bias`. A higher score favors
/// ending speech. This raw score is called a logit; it is not a probability.
///
/// Input: floating-point `[..., .d=512]`. Output: `[..., .dout=1]` in the input dtype.
/// Each input feature vector gets its own score; batch and sequence axes are
/// preserved. During generation, the caller passes only the last sequence
/// position's features, producing one score per batch row.
///
/// The generation loop compares the score directly with its EOS threshold and
/// decides when to stop. This head applies no sigmoid or stopping policy.
///
/// Matches Python's `eos_head`, a biased `Linear(512, 1)`. Parameters come from
/// `flow_lm.out_eos`: bf16 weight `[1, 512]` and bias `[1]`, converted to the input
/// dtype by forward. The generation wrapper currently supplies f32 features.
///
/// Reference:
/// ```
/// eos_head: Linear
/// ├── weight: shape=(1, 512), dtype=torch.float32
/// │   ├── loaded: eos_head.weight
/// │   └── checkpoint: flow_lm.out_eos.weight
/// └── bias: shape=(1,), dtype=torch.float32
///     ├── loaded: eos_head.bias
///     └── checkpoint: flow_lm.out_eos.bias
/// ```
pub const EOS_Head = struct {
    projection: zml.nn.Linear,

    /// Expects the checkpoint view at flow_lm.out_eos.
    pub fn init(store: zml.io.TensorStore.View) EOS_Head {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .projection = .init(
                store.createReplicatedTensor("weight", .{ .dout, .d }),
                store.createReplicatedTensor("bias", .{.dout}),
                .d,
            ),
        };
    }

    /// Computes one raw end-of-speech score per input feature vector.
    ///
    /// `input`: final normalized hidden states from the transformer, encoding
    /// the text, voice, and audio context visible at each supplied position.
    /// Shape: floating-point `[..., 512]`, commonly `[B, S, 512]` for a sequence or
    /// `[B, 512]` for one selected position per batch row.
    /// During generation, the prediction wrapper selects the last position
    /// from the full-prefix transformer output,
    /// and passes `[B, 512]` here. The final axis is tagged `.d` by this method.
    ///
    /// Returns `[..., .dout=1]` in the input dtype, preserving all leading axes.
    /// Each value is a logit, not a probability; the generation loop applies the
    /// stopping threshold and policy.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; the final axis of 512 transformer features,
    ///   contracted with the weight matrix to compute a score.
    /// - `.dout`: output dimension; one score, with its learned bias added.
    /// - Leading axes are preserved, commonly `.b` for independent sequences
    ///   and `.s` for token positions. Untagged leading axes remain untagged.
    ///
    /// Shape: `[..., .d=512]` -> `[..., .dout=1]`, preserving the floating-point dtype.
    pub fn forward(self: EOS_Head, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat());
        std.debug.assert(input.rank() >= 1);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d}); // [...,512] -> [...,d=512]
        debug_value("x", x);
        const output = self.projection.forward(x, x.dtype()); // [...,d=512] -> [...,dout=1], same dtype
        debug_value("output", output);
        debug_value("output", output);
        return output;
    }
};

// --- Gradium_Phonon_TTS_Flow_Head ---

/// Predicts how to move a candidate audio latent toward the next speech frame.
///
/// The sampler starts with noise in a 32-component audio latent space. This head
/// predicts a velocity (a change in those coordinates) using the candidate,
/// the transformer's contextual features, and a flow-sampling interval. The
/// sampler uses that velocity to update the candidate; the result remains a
/// latent frame that must later be decoded into waveform samples.
///
/// The candidate is projected from 32 to 512 features. Separately, the projected
/// transformer condition is added to the average of two learned time embeddings.
/// Six residual blocks use this combined condition to refine the candidate's
/// hidden features, then the final layer projects them to 32 velocity components.
///
/// Inputs: floating-point candidate `[B, 32]`, transformer condition `[B, 512]`, and
/// start/end times `[B, 1]`. Output: velocity `[B, 32]` in the candidate dtype.
/// B is the number of independent candidate frames processed together.
///
/// Matches Python's `flow_head`. Parameters come from `flow_lm.flow_net`:
/// `input_proj`, `cond_embed`, `time_embed.0..1`, `res_blocks.0..5`, and
/// `final_layer`. Stored bf16 parameters are converted to the candidate input dtype by the
/// child modules.
///
/// Reference:
/// ```
///  flow_head: GradiumPhononTTSFlowHead
///  ├── input_proj: Linear
///  │   ├── weight: shape=(512, 32), dtype=torch.float32
///  │   │   ├── loaded: flow_head.input_proj.weight
///  │   │   └── checkpoint: flow_lm.flow_net.input_proj.weight
///  │   └── bias: shape=(512,), dtype=torch.float32
///  │       ├── loaded: flow_head.input_proj.bias
///  │       └── checkpoint: flow_lm.flow_net.input_proj.bias
///  ├── cond_proj: Linear
///  │   ├── weight: shape=(512, 512), dtype=torch.float32
///  │   │   ├── loaded: flow_head.cond_proj.weight
///  │   │   └── checkpoint: flow_lm.flow_net.cond_embed.weight
///  │   └── bias: shape=(512,), dtype=torch.float32
///  │       ├── loaded: flow_head.cond_proj.bias
///  │       └── checkpoint: flow_lm.flow_net.cond_embed.bias
///  ├── timestep_embeddings: ModuleList
///  │   └── 0..1: GradiumPhononTTSTimestepEmbedding
///  │       ├── fc1: Linear
///  │       │   ├── weight: shape=(512, 256), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.timestep_embeddings.1.fc1.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.timestep_embeddings.1.fc1.bias
///  │       │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.bias
///  │       ├── fc2: Linear
///  │       │   ├── weight: shape=(512, 512), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.timestep_embeddings.1.fc2.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.timestep_embeddings.1.fc2.bias
///  │       │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.bias
///  │       ├── act_fn: SiLUActivation
///  │       └── norm: GradiumPhononTTSTimestepNorm
///  │           └── weight: shape=(512,), dtype=torch.float32
///  │               ├── loaded: flow_head.timestep_embeddings.1.norm.weight
///  │               └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.3.alpha
///  ├── layers: ModuleList
///  │   └── 0..5: GradiumPhononTTSFlowResBlock
///  │       ├── norm: GradiumPhononTTSFlowLayerNorm
///  │       │   ├── weight: shape=(512,), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.layers.5.norm.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.layers.5.norm.bias
///  │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.bias
///  │       ├── fc1: Linear
///  │       │   ├── weight: shape=(512, 512), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.layers.5.fc1.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.layers.5.fc1.bias
///  │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.bias
///  │       ├── fc2: Linear
///  │       │   ├── weight: shape=(512, 512), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.layers.5.fc2.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.layers.5.fc2.bias
///  │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.bias
///  │       ├── act_fn: SiLUActivation
///  │       └── adaln_proj: Linear
///  │           ├── weight: shape=(1536, 512), dtype=torch.float32
///  │           │   ├── loaded: flow_head.layers.5.adaln_proj.weight
///  │           │   └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.weight
///  │           └── bias: shape=(1536,), dtype=torch.float32
///  │               ├── loaded: flow_head.layers.5.adaln_proj.bias
///  │               └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.bias
///  └── final_layer: GradiumPhononTTSFlowFinalLayer
///      ├── norm: GradiumPhononTTSFlowLayerNorm
///      ├── adaln_proj: Linear
///      │   ├── weight: shape=(1024, 512), dtype=torch.float32
///      │   │   ├── loaded: flow_head.final_layer.adaln_proj.weight
///      │   │   └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.weight
///      │   └── bias: shape=(1024,), dtype=torch.float32
///      │       ├── loaded: flow_head.final_layer.adaln_proj.bias
///      │       └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.bias
///      ├── act_fn: SiLUActivation
///      └── proj: Linear
///          ├── weight: shape=(32, 512), dtype=torch.float32
///          │   ├── loaded: flow_head.final_layer.proj.weight
///          │   └── checkpoint: flow_lm.flow_net.final_layer.linear.weight
///          └── bias: shape=(32,), dtype=torch.float32
///              ├── loaded: flow_head.final_layer.proj.bias
///              └── checkpoint: flow_lm.flow_net.final_layer.linear.bias
/// ```
pub const Gradium_Phonon_TTS_Flow_Head = struct {
    input_proj: Flow_Input_Projection,
    cond_proj: Condition_Projection,
    timestep_embeddings: [2]Gradium_Phonon_TTS_Timestep_Embedding,
    layers: [6]Gradium_Phonon_TTS_Flow_Res_Block,
    final_layer: Gradium_Phonon_TTS_Flow_Final_Layer,

    /// Targets the phonon-31m-en flow head: two time embeddings and six blocks.
    /// Expects a checkpoint view at flow_lm.flow_net.
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Flow_Head {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var self: Gradium_Phonon_TTS_Flow_Head = .{
            .input_proj = .init(store.withPrefix("input_proj")),
            .cond_proj = .init(store.withPrefix("cond_embed")),
            .timestep_embeddings = undefined,
            .layers = undefined,
            .final_layer = .init(store.withPrefix("final_layer")),
        };
        for (&self.timestep_embeddings, 0..) |*embedding, i| {
            embedding.* = .init(store.withPrefix("time_embed").withLayer(i));
        }
        for (&self.layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("res_blocks").withLayer(i));
        }
        return self;
    }

    /// Predicts a latent velocity for the supplied candidate and sampling interval.
    ///
    /// `input`: floating-point `[B, 32]`, the sampler's current candidate for the next audio
    /// frame in normalized latent space. The prediction wrapper passes it through
    /// from the sampler; in the current one-step sampler it is freshly drawn noise.
    /// `condition`: floating-point `[B, 512]`, final normalized transformer features encoding
    /// text, voice, and preceding audio context. The prediction wrapper selects
    /// the last position from the full-prefix transformer output.
    /// `start_time`: floating-point `[B, 1]`, the interval's starting flow-sampling progress.
    /// `end_time`: floating-point `[B, 1]`, the interval's ending flow-sampling progress.
    /// The sampler supplies both times, normally in [0, 1]; the current one-step
    /// sampler uses 0 and 1. These values describe progress from noise toward an
    /// audio latent, rather than token positions or waveform timestamps.
    /// All inputs must have the same batch size B; this method assigns their tags.
    ///
    /// Returns floating-point `[B, 32]`, a velocity in the candidate's latent coordinates.
    /// The caller applies the numerical update; for the current interval [0, 1],
    /// that is `frame = input + velocity`.
    ///
    /// The candidate input determines the computation and output dtype.
    /// Transformer conditioning, interval times, and parameters are converted to it.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent candidate frames, shared by all four inputs.
    /// - `.ld`: latent dimension; 32 components in the candidate and output velocity.
    /// - `.d`: hidden dimension; 512 projected features in the flow blocks and condition.
    /// - `.t`: one scalar flow time per row, embedded separately for start and end.
    ///   These are sampling progress in [0, 1], not audio or token positions.
    /// Internal frequency and modulation tags are documented by the child modules.
    ///
    /// Inputs: floating-point candidate `[.b, .ld=32]`, condition `[.b, .d=512]`,
    /// start/end times each `[.b, .t=1]`. Output: floating-point `[.b, .ld=32]`.
    pub fn forward(
        self: Gradium_Phonon_TTS_Flow_Head,
        input: zml.Tensor,
        condition: zml.Tensor,
        start_time: zml.Tensor,
        end_time: zml.Tensor,
    ) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);
        debug_value("condition", condition);
        debug_value("start_time", start_time);
        debug_value("end_time", end_time);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and condition.dtype().isFloat());
        std.debug.assert(start_time.dtype().isFloat() and end_time.dtype().isFloat());
        std.debug.assert(input.rank() == 2 and condition.rank() == 2);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const candidate = input.withTags(.{ .b, .ld });
        debug_value("candidate", candidate);
        const context = condition.convert(dtype).withTags(.{ .b, .d });
        debug_value("context", context);
        const start = start_time.convert(dtype).withTags(.{ .b, .t });
        debug_value("start", start);
        const end = end_time.convert(dtype).withTags(.{ .b, .t });
        debug_value("end", end);
        std.debug.assert(candidate.dim(.b) == context.dim(.b));
        std.debug.assert(candidate.dim(.b) == start.dim(.b) and candidate.dim(.b) == end.dim(.b));

        var hidden = self.input_proj.forward(candidate);
        debug_value("hidden", hidden);
        const combined_condition = self.combine_condition(context, start, end);
        debug_value("combined_condition", combined_condition);
        inline for (self.layers, 0..) |layer, i| {
            hidden = layer.forward(i, hidden, combined_condition);
            debug_value("hidden", hidden);
        }
        const debug_output: zml.Tensor = self.final_layer.forward(hidden, combined_condition);
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Context [B,.d=512] and interval endpoints [B,.t=1] -> [B,.d=512].
    /// Each endpoint has its own learned embedding; average before adding context.
    fn combine_condition(self: Gradium_Phonon_TTS_Flow_Head, context: zml.Tensor, start: zml.Tensor, end: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".combine_condition";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("context", context);
        debug_value("start", start);
        debug_value("end", end);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".combine_condition", .{});
        defer compiler.popLocation();
        var time_embeddings: [2]zml.Tensor = undefined;
        inline for (self.timestep_embeddings, .{ start, end }, 0..) |embedding, time, i| {
            time_embeddings[i] = embedding.forward(i, time);
            debug_value(std.fmt.comptimePrint("time_embeddings[{}]", .{i}), time_embeddings[i]);
        }
        const time_embedding = time_embeddings[0].add(time_embeddings[1]).scale(0.5);
        debug_value("time_embedding", time_embedding);
        const projected_context = self.cond_proj.forward(context);
        debug_value("projected_context", projected_context);
        const debug_output: zml.Tensor = projected_context.add(time_embedding);
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Prepare buffer metadata, handling fixed embedding and block arrays explicitly.
    pub fn bufferize(self: *const Gradium_Phonon_TTS_Flow_Head, allocator: std.mem.Allocator) !zml.Bufferized(Gradium_Phonon_TTS_Flow_Head) {
        var buffers: zml.Bufferized(Gradium_Phonon_TTS_Flow_Head) = .{
            .input_proj = try zml.mem.bufferize(allocator, Flow_Input_Projection, &self.input_proj),
            .cond_proj = try zml.mem.bufferize(allocator, Condition_Projection, &self.cond_proj),
            .final_layer = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Flow_Final_Layer, &self.final_layer),
            .timestep_embeddings = undefined,
            .layers = undefined,
        };
        for (&self.timestep_embeddings, &buffers.timestep_embeddings) |*embedding, *buffer| {
            buffer.* = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Timestep_Embedding, embedding);
        }
        for (&self.layers, &buffers.layers) |*layer, *buffer| {
            buffer.* = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Flow_Res_Block, layer);
        }
        return buffers;
    }
};

// --- Flow_Input_Projection ---

/// Maps a candidate audio latent into the flow head's hidden feature space.
///
/// A biased linear layer expands 32 latent coordinates to 512 hidden features.
/// The flow head passes these features to the first residual block.
///
/// Input: floating-point `[..., .ld=32]`. Output: `[..., .d=512]` in the input dtype.
/// Leading axes are preserved; the flow head supplies one candidate per batch
/// row, shaped `[B, 32]`.
///
/// Parameters come from `flow_lm.flow_net.input_proj`: bf16 weight `[512, 32]`
/// and bias `[512]`, converted to the input dtype by forward.
///
/// Reference:
/// ```
//  input_proj: Linear
//  ├── weight: shape=(512, 32), dtype=torch.float32
//  │   ├── loaded: flow_head.input_proj.weight
//  │   └── checkpoint: flow_lm.flow_net.input_proj.weight
//  └── bias: shape=(512,), dtype=torch.float32
//      ├── loaded: flow_head.input_proj.bias
//      └── checkpoint: flow_lm.flow_net.input_proj.bias
/// ```
pub const Flow_Input_Projection = struct {
    projection: zml.nn.Linear,

    /// Expects a checkpoint view at flow_lm.flow_net.input_proj.
    pub fn init(store: zml.io.TensorStore.View) Flow_Input_Projection {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .projection = .init(
            store.createReplicatedTensor("weight", .{ .d, .ld }),
            store.createReplicatedTensor("bias", .{.d}),
            .ld,
        ) };
    }

    // -- Description --

    /// Projects a candidate audio latent into flow hidden features.
    ///
    /// `input`: floating-point `[..., 32]`, the sampler's current candidate latent, passed
    /// by the flow head before any residual blocks. In the current one-step
    /// sampler this is fresh noise shaped `[B, 32]`. This method tags the final
    /// axis `.ld`; any leading batch or sequence axes are preserved.
    ///
    /// Returns floating-point `[..., .d=512]`, candidate features for the first residual
    /// block, computed by a biased linear layer without activation or normalization.
    ///
    /// Parameters are converted to the input dtype; the output keeps that dtype.
    ///
    /// Dimension tags:
    /// - `.ld`: latent dimension; 32 candidate audio features, contracted by the weight.
    /// - `.d`: hidden dimension; 512 output features with a learned bias.
    /// Shape: floating-point `[..., .ld=32]` -> floating-point `[..., .d=512]`.
    pub fn forward(self: Flow_Input_Projection, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat());
        std.debug.assert(input.rank() >= 1);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.ld}); // [...,32] -> [...,ld=32]
        debug_value("x", x);
        const projected = self.projection.forward(x, dtype); // [...,ld=32] -> [...,d=512]
        debug_value("projected", projected);
        debug_value("output", projected);
        return projected;
    }
};

// --- Gradium_Phonon_TTS_Timestep_Embedding ---

/// Encodes a scalar flow-sampling time as 512 conditioning features.
///
/// Multiplies the time by 128 stored frequencies and concatenates the cosines
/// and sines into 256 features. Two biased linear layers with SiLU between them
/// produce 512 features, followed by RMS normalization with a learned scale
/// and epsilon 1e-5. RMS normalization does not subtract the mean.
///
/// Input: floating-point times `[B, 1]`. Output: embeddings `[B, 512]` in the input dtype.
/// The flow head uses separate instances for interval start and end times,
/// then averages their outputs before adding the projected transformer condition.
///
/// Parameters come from `flow_lm.flow_net.time_embed.0` or `.1`: `freqs` [128],
/// `mlp.0` (256 -> 512), `mlp.2` (512 -> 512), and `mlp.3.alpha` [512].
/// Stored bf16 tensors are converted to the input dtype; frequencies retain checkpoint rounding.
///
/// Reference:
/// ```
//  GradiumPhononTTSTimestepEmbedding
//  ├── fc1: Linear
//  │   ├── weight: shape=(512, 256), dtype=torch.float32
//  │   │   ├── loaded: flow_head.timestep_embeddings.1.fc1.weight
//  │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.timestep_embeddings.1.fc1.bias
//  │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.bias
//  ├── fc2: Linear
//  │   ├── weight: shape=(512, 512), dtype=torch.float32
//  │   │   ├── loaded: flow_head.timestep_embeddings.1.fc2.weight
//  │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.timestep_embeddings.1.fc2.bias
//  │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.bias
//  ├── act_fn: SiLUActivation
//  └── norm: GradiumPhononTTSTimestepNorm
//      └── weight: shape=(512,), dtype=torch.float32
//          ├── loaded: flow_head.timestep_embeddings.1.norm.weight
//          └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.3.alpha
/// ```
pub const Gradium_Phonon_TTS_Timestep_Embedding = struct {
    freqs: zml.Tensor,
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    norm_weight: zml.Tensor,
    eps: f32 = 1e-5,

    /// Expects a view at flow_lm.flow_net.time_embed.0 (or .1).
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Timestep_Embedding {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .freqs = store.createReplicatedTensor("freqs", .{.f}),
            .fc1 = .init(
                store.createReplicatedTensor("mlp.0.weight", .{ .d, .f }),
                store.createReplicatedTensor("mlp.0.bias", .{.d}),
                .f,
            ),
            .fc2 = .init(
                store.createReplicatedTensor("mlp.2.weight", .{ .dout, .d }),
                store.createReplicatedTensor("mlp.2.bias", .{.dout}),
                .d,
            ),
            .norm_weight = store.createReplicatedTensor("mlp.3.alpha", .{.d}),
        };
    }

    // -- Description --

    /// Embeds a flow-sampling time using sinusoidal features and a learned MLP.
    ///
    /// `input`: floating-point `[B, 1]`, one scalar flow-sampling progress value per batch row,
    /// normally in [0, 1]. The flow head passes its `start_time` to instance 0 and
    /// its `end_time` to instance 1; both originate in the sampler. This method
    /// assigns tags `.b` and `.t`.
    ///
    /// Returns floating-point `[B, 512]`, a conditioning vector for that interval endpoint.
    /// The parent averages the two endpoint embeddings. Computation is
    /// cos/sin features -> fc1 -> SiLU -> fc2 -> learned RMS normalization.
    ///
    /// Parameters are converted to the input dtype; the output keeps that dtype.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent time values, embedded without mixing rows.
    /// - `.t`: one scalar flow time per row. Broadcast over frequencies, then removed.
    /// - `.f`: frequency features; 128 stored frequencies, expanded to 256 features
    ///   by concatenating all cosines followed by all sines. Contracted by fc1.
    /// - `.d`: hidden dimension; 512 features after fc1 and in the final embedding.
    ///   RMS normalization reduces over this axis without subtracting the mean;
    ///   the learned weight then scales each feature (no bias).
    /// - `.dout`: fc2's 512 output features, renamed to `.d` before normalization.
    ///
    /// Shape: `[.b, .t=1]` -> `[.b, .f=256]` -> `[.b, .d=512]`, preserving the input dtype.
    /// `idx` identifies this repeated layer in compiler source locations.
    pub fn forward(self: Gradium_Phonon_TTS_Timestep_Embedding, comptime idx: usize, input: zml.Tensor) zml.Tensor {
        const here = std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx });

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        std.debug.assert(input.dtype().isFloat() and input.rank() == 2);
        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx }), .{});
        defer compiler.popLocation();

        const times = input.withTags(.{ .b, .t });
        debug_value("times", times);
        std.debug.assert(times.dim(.t) == 1);
        const periodic = self.sinusoidal_features(times);
        debug_value("periodic", periodic);
        const hidden = self.fc1.forward(periodic, times.dtype()).silu();
        debug_value("hidden", hidden);
        const embedding = self.fc2.forward(hidden, times.dtype()).rename(.{ .dout = .d });
        debug_value("embedding", embedding);
        const debug_output: zml.Tensor = self.normalize_embedding(embedding);
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Times [B,.t=1] -> concatenated cos/sin features [B,.f=256].
    /// Use checkpoint frequencies, preserving their stored rounding.
    fn sinusoidal_features(self: Gradium_Phonon_TTS_Timestep_Embedding, times: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".sinusoidal_features";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("times", times);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".sinusoidal_features", .{});
        defer compiler.popLocation();

        const frequencies = self.freqs.convert(times.dtype());
        debug_value("frequencies", frequencies);
        const shape: zml.Shape = .init(.{ .b = times.dim(.b), .f = frequencies.dim(.f) }, times.dtype());
        const angles = times.rename(.{ .t = .f }).broad(shape).mul(frequencies.broad(shape));
        debug_value("angles", angles);
        const debug_output: zml.Tensor = zml.Tensor.concatenate(&.{ angles.cos(), angles.sin() }, .f);
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Learned RMS normalization of [B,.d=512], without centering.
    /// Preserve the reference's multiplication order: (weight * embedding) / RMS.
    fn normalize_embedding(self: Gradium_Phonon_TTS_Timestep_Embedding, embedding: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".normalize_embedding";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("embedding", embedding);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".normalize_embedding", .{});
        defer compiler.popLocation();

        const inverse_rms = embedding.mul(embedding).mean(.d).addConstant(self.eps).rsqrt();
        debug_value("inverse_rms", inverse_rms);
        const weight = self.norm_weight.convert(embedding.dtype()).broad(embedding.shape());
        debug_value("weight", weight);
        const debug_output: zml.Tensor = weight.mul(embedding).mul(inverse_rms.broad(embedding.shape()));
        debug_value("output", debug_output);
        return debug_output;
    }
};

// --- Condition_Projection ---

/// Maps transformer context into the flow head's conditioning feature space.
///
/// A biased linear layer transforms the final normalized transformer features.
/// The flow head adds the averaged start/end time embeddings to this projection;
/// the resulting condition guides every residual block and the final layer.
///
/// Input and output: floating-point `[..., .d=512]`, preserving dtype and leading axes.
/// During generation, the input is `[B, 512]`, selected from the last transformer
/// position by the prediction wrapper.
///
/// Parameters come from `flow_lm.flow_net.cond_embed`: bf16 weight `[512, 512]`
/// and bias `[512]`, converted to the input dtype by forward.
///
/// Reference:
/// ```
//  cond_proj: Linear
//  ├── weight: shape=(512, 512), dtype=torch.float32
//  │   ├── loaded: flow_head.cond_proj.weight
//  │   └── checkpoint: flow_lm.flow_net.cond_embed.weight
//  └── bias: shape=(512,), dtype=torch.float32
//      ├── loaded: flow_head.cond_proj.bias
//      └── checkpoint: flow_lm.flow_net.cond_embed.bias
/// ```
pub const Condition_Projection = struct {
    projection: zml.nn.Linear,

    /// Expects a checkpoint view at flow_lm.flow_net.cond_embed.
    pub fn init(store: zml.io.TensorStore.View) Condition_Projection {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{ .projection = .init(
            store.createReplicatedTensor("weight", .{ .dout, .d }),
            store.createReplicatedTensor("bias", .{.dout}),
            .d,
        ) };
    }

    // -- Description --

    /// Projects transformer context into flow conditioning features.
    ///
    /// `input`: floating-point `[..., 512]`, final normalized transformer hidden states,
    /// passed through the flow head's `condition` parameter. During generation,
    /// the prediction wrapper supplies the last position's features as `[B, 512]`,
    /// encoding the text, voice, and audio context visible there. This method tags
    /// the final axis `.d` and preserves any leading batch or sequence axes.
    ///
    /// Returns floating-point `[..., .d=512]`, a biased linear projection. The flow head adds
    /// the averaged time embeddings before passing it to the blocks and final layer.
    ///
    /// Parameters are converted to the input dtype; the output keeps that dtype.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; 512 transformer features, contracted by the weight.
    /// - `.dout`: 512 projected conditioning features, renamed to `.d` on output.
    /// Shape: floating-point `[..., .d=512]` -> floating-point `[..., .d=512]`.
    pub fn forward(self: Condition_Projection, input: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat());
        std.debug.assert(input.rank() >= 1);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d}); // [...,512] -> [...,d=512]
        debug_value("x", x);
        const projected = self.projection.forward(x, dtype); // [...,d=512] -> [...,dout=512]
        debug_value("projected", projected);
        const output = projected.rename(.{ .dout = .d }); // [...,dout=512] -> [...,d=512]
        debug_value("output", output);
        debug_value("output", output);
        return output;
    }
};

// --- Gradium_Phonon_TTS_Flow_Res_Block ---

/// Refines a candidate frame's hidden features using transformer and time context.
///
/// The condition produces three 512-feature vectors: shift, scale, and gate.
/// Shift and scale modulate the normalized input before a two-layer MLP;
/// the gate scales that MLP's update before it is added to the original input:
/// `output = input + gate * MLP(LayerNorm(input) * (1 + scale) + shift)`.
/// The gate is a raw learned multiplier, with no sigmoid.
///
/// Input and condition: floating-point `[B, 512]`. Output: `[B, 512]` in the input dtype.
/// Each batch row is processed independently. The flow head chains six blocks,
/// passing the same combined transformer/time condition to each one.
///
/// Parameters come from `flow_lm.flow_net.res_blocks.{index}`: `in_ln`
/// (learned scale and bias), `mlp.0` and `mlp.2` (512 -> 512, with SiLU between),
/// and `adaLN_modulation.1` (512 -> 1536, applied after SiLU).
/// Stored bf16 parameters are converted to the input dtype; LayerNorm epsilon is 1e-6.
///
/// Reference:
/// ```
//  GradiumPhononTTSFlowResBlock
//  ├── norm: GradiumPhononTTSFlowLayerNorm
//  │   ├── weight: shape=(512,), dtype=torch.float32
//  │   │   ├── loaded: flow_head.layers.5.norm.weight
//  │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.layers.5.norm.bias
//  │       └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.bias
//  ├── fc1: Linear
//  │   ├── weight: shape=(512, 512), dtype=torch.float32
//  │   │   ├── loaded: flow_head.layers.5.fc1.weight
//  │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.layers.5.fc1.bias
//  │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.bias
//  ├── fc2: Linear
//  │   ├── weight: shape=(512, 512), dtype=torch.float32
//  │   │   ├── loaded: flow_head.layers.5.fc2.weight
//  │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.layers.5.fc2.bias
//  │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.bias
//  ├── act_fn: SiLUActivation
//  └── adaln_proj: Linear
//      ├── weight: shape=(1536, 512), dtype=torch.float32
//      │   ├── loaded: flow_head.layers.5.adaln_proj.weight
//      │   └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.weight
//      └── bias: shape=(1536,), dtype=torch.float32
//          ├── loaded: flow_head.layers.5.adaln_proj.bias
//          └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.bias
/// ```
pub const Gradium_Phonon_TTS_Flow_Res_Block = struct {
    norm_weight: zml.Tensor,
    norm_bias: zml.Tensor,
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    adaln_proj: zml.nn.Linear,
    eps: f32 = 1e-6,

    /// Expects a view at flow_lm.flow_net.res_blocks.{index}.
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Flow_Res_Block {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .norm_weight = store.createReplicatedTensor("in_ln.weight", .{.d}),
            .norm_bias = store.createReplicatedTensor("in_ln.bias", .{.d}),
            .fc1 = .init(store.createReplicatedTensor("mlp.0.weight", .{ .dout, .d }), store.createReplicatedTensor(
                "mlp.0.bias",
                .{.dout},
            ), .d),
            .fc2 = .init(store.createReplicatedTensor("mlp.2.weight", .{ .dout, .d }), store.createReplicatedTensor(
                "mlp.2.bias",
                .{.dout},
            ), .d),
            .adaln_proj = .init(
                store.createReplicatedTensor("adaLN_modulation.1.weight", .{ .m, .d }),
                store.createReplicatedTensor("adaLN_modulation.1.bias", .{.m}),
                .d,
            ),
        };
    }

    /// Updates flow hidden features using a condition-dependent residual MLP.
    ///
    /// `input`: floating-point `[B, 512]`, the candidate's current hidden features, produced
    /// by `Flow_Input_Projection` for the first block or by the preceding residual block.
    /// `condition`: floating-point `[B, 512]`, the flow head's projected transformer context
    /// plus the average of its start/end time embeddings. The parent computes
    /// this once and passes the same tensor to all six blocks and the final layer.
    /// Batch rows correspond one-to-one; this method assigns tags `.b` and `.d`.
    ///
    /// Returns floating-point `[B, 512]`, the input plus a gated MLP update. SiLU(condition)
    /// produces shift, scale, and gate through a linear projection. Shift and scale
    /// modulate the normalized input; the gate multiplies the MLP output without
    /// a sigmoid before the residual addition.
    ///
    /// The input determines the computation and output dtype. The condition and
    /// parameters are converted to that dtype before computation.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent candidate frames, with one condition per row.
    /// - `.d`: 512 hidden features; LayerNorm reduces this axis separately per row.
    ///   Both the input and the combined transformer/time condition use this axis.
    /// - `.m`: 1536 modulation features; three consecutive groups of 512 for shift,
    ///   scale and gate. Each group is renamed to `.d` after splitting.
    /// - `.dout`: 512 output features of each MLP projection, renamed to `.d`.
    ///
    /// Shapes: input and condition floating-point `[.b, .d=512]` -> floating-point `[.b, .d=512]`.
    /// `idx` identifies this repeated layer in compiler source locations.
    pub fn forward(self: Gradium_Phonon_TTS_Flow_Res_Block, comptime idx: usize, input: zml.Tensor, condition: zml.Tensor) zml.Tensor {
        const here = std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx });

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);
        debug_value("condition", condition);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and condition.dtype().isFloat());
        std.debug.assert(input.rank() == 2 and condition.rank() == 2);
        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), std.fmt.comptimePrint("{s}.forward[{}]", .{ @typeName(@This()), idx }), .{});
        defer compiler.popLocation();

        const hidden = input.withTags(.{ .b, .d });
        debug_value("hidden", hidden);
        const context = condition.convert(dtype).withTags(.{ .b, .d });
        debug_value("context", context);
        std.debug.assert(hidden.dim(.b) == context.dim(.b) and hidden.dim(.d) == context.dim(.d));

        const modulation = self.condition_modulation(context);
        debug_value("modulation", modulation);
        const normalized = self.normalize_features(hidden);
        debug_value("normalized", normalized);
        const modulated = apply_modulation(normalized, modulation.shift, modulation.scale);
        debug_value("modulated", modulated);
        const residual = self.feed_forward(modulated);
        debug_value("residual", residual);
        // The gate is a raw learned multiplier, not a sigmoid probability.
        const debug_output: zml.Tensor = hidden.add(modulation.gate.mul(residual));
        debug_value("output", debug_output);
        return debug_output;
    }

    const Modulation = struct { shift: zml.Tensor, scale: zml.Tensor, gate: zml.Tensor };

    /// Condition [B,.d=512] -> three [B,.d=512] modulation vectors.
    /// The projection's .m axis packs shift, scale, and gate in that order.
    fn condition_modulation(self: Gradium_Phonon_TTS_Flow_Res_Block, condition: zml.Tensor) Modulation {
        const here = @typeName(@This()) ++ ".condition_modulation";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("condition", condition);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".condition_modulation", .{});
        defer compiler.popLocation();

        const projected = self.adaln_proj.forward(condition.silu(), condition.dtype());
        debug_value("projected", projected);
        const parts = projected.chunkExact(.m, 3);
        debug_value("parts", parts);
        const debug_output: Modulation = .{
            .shift = parts[0].rename(.{ .m = .d }),
            .scale = parts[1].rename(.{ .m = .d }),
            .gate = parts[2].rename(.{ .m = .d }),
        };
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Feature-wise LayerNorm with learned gain/bias; shape [B,.d=512].
    fn normalize_features(self: Gradium_Phonon_TTS_Flow_Res_Block, hidden: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".normalize_features";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("hidden", hidden);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".normalize_features", .{});
        defer compiler.popLocation();

        const normalized = normalize_feature_variance(hidden, self.eps);
        debug_value("normalized", normalized);
        const weight = self.norm_weight.convert(hidden.dtype()).broad(hidden.shape());
        debug_value("weight", weight);
        const bias = self.norm_bias.convert(hidden.dtype()).broad(hidden.shape());
        debug_value("bias", bias);
        const debug_output: zml.Tensor = normalized.mul(weight).add(bias);
        debug_value("output", debug_output);
        return debug_output;
    }

    /// Linear -> SiLU -> Linear, preserving [B,.d=512] and dtype.
    fn feed_forward(self: Gradium_Phonon_TTS_Flow_Res_Block, hidden: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".feed_forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("hidden", hidden);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".feed_forward", .{});
        defer compiler.popLocation();

        const expanded = self.fc1.forward(hidden, hidden.dtype()).rename(.{ .dout = .d });
        debug_value("expanded", expanded);
        const debug_output: zml.Tensor = self.fc2.forward(expanded.silu(), hidden.dtype()).rename(.{ .dout = .d });
        debug_value("output", debug_output);
        return debug_output;
    }
};

// --- Gradium_Phonon_TTS_Flow_Final_Layer ---

/// Converts the final flow hidden features into an audio latent velocity.
///
/// Normalizes the last residual block's output, applies a condition-derived
/// shift and scale, and projects the result from 512 features to 32 latent
/// velocity components. The sampler uses this velocity to update its candidate.
///
/// Input and condition: floating-point `[B, 512]`. Output: `[B, 32]` in the input dtype.
/// The condition is the same combined transformer/time vector supplied to all
/// six residual blocks. LayerNorm has no learned scale or bias here; its
/// epsilon is 1e-6. Modulation uses `normalized * (1 + scale) + shift`.
///
/// Parameters come from `flow_lm.flow_net.final_layer`: `adaLN_modulation.1`
/// (512 -> 1024, producing shift and scale after SiLU) and `linear` (512 -> 32).
/// Stored bf16 parameters are converted to the input dtype.
///
/// Reference:
/// ```
// GradiumPhononTTSFlowFinalLayer
// ├── norm: GradiumPhononTTSFlowLayerNorm
// ├── adaln_proj: Linear
// │   ├── weight: shape=(1024, 512), dtype=torch.float32
// │   │   ├── loaded: flow_head.final_layer.adaln_proj.weight
// │   │   └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.weight
// │   └── bias: shape=(1024,), dtype=torch.float32
// │       ├── loaded: flow_head.final_layer.adaln_proj.bias
// │       └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.bias
// ├── act_fn: SiLUActivation
// └── proj: Linear
//     ├── weight: shape=(32, 512), dtype=torch.float32
//     │   ├── loaded: flow_head.final_layer.proj.weight
//     │   └── checkpoint: flow_lm.flow_net.final_layer.linear.weight
//     └── bias: shape=(32,), dtype=torch.float32
//         ├── loaded: flow_head.final_layer.proj.bias
//         └── checkpoint: flow_lm.flow_net.final_layer.linear.bias
/// ```
pub const Gradium_Phonon_TTS_Flow_Final_Layer = struct {
    adaln_proj: zml.nn.Linear,
    proj: zml.nn.Linear,
    eps: f32 = 1e-6,

    /// Expects a view at flow_lm.flow_net.final_layer.
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Flow_Final_Layer {
        const here = @typeName(@This()) ++ ".init";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        return .{
            .adaln_proj = .init(
                store.createReplicatedTensor("adaLN_modulation.1.weight", .{ .m, .d }),
                store.createReplicatedTensor("adaLN_modulation.1.bias", .{.m}),
                .d,
            ),
            .proj = .init(
                store.createReplicatedTensor("linear.weight", .{ .ld, .d }),
                store.createReplicatedTensor("linear.bias", .{.ld}),
                .d,
            ),
        };
    }

    /// Normalizes and conditions the final hidden features, then predicts velocity.
    ///
    /// `input`: floating-point `[B, 512]`, the candidate's hidden features produced by the
    /// sixth and last flow residual block.
    /// `condition`: floating-point `[B, 512]`, projected transformer context plus the averaged
    /// start/end time embeddings, computed by the flow head. This is the same
    /// condition supplied to each residual block.
    /// Batch rows correspond one-to-one; this method assigns tags `.b` and `.d`.
    ///
    /// Returns floating-point `[B, 32]`, the latent velocity consumed by the sampler.
    /// LayerNorm has no learned weight or bias here. SiLU -> Linear produces the
    /// condition-derived shift and scale; there is no gate or residual addition.
    ///
    /// The input determines the computation and output dtype. The condition and
    /// parameters are converted to that dtype before computation.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent candidate frames, with one condition per row.
    /// - `.d`: 512 hidden features in both input and combined transformer/time
    ///   condition. Normalization reduces only this axis, separately per row.
    /// - `.m`: 1024 modulation features; consecutive 512-feature shift and scale
    ///   vectors, each renamed to `.d` after splitting.
    /// - `.ld`: latent dimension; 32 velocity components produced by the final
    ///   biased projection. The sampler uses these to update the candidate latent.
    ///
    /// Shapes: two floating-point `[.b, .d=512]` inputs -> floating-point `[.b, .ld=32]` output.
    pub fn forward(self: Gradium_Phonon_TTS_Flow_Final_Layer, input: zml.Tensor, condition: zml.Tensor) zml.Tensor {
        const here = @typeName(@This()) ++ ".forward";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("input", input);
        debug_value("condition", condition);

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and condition.dtype().isFloat());
        std.debug.assert(input.rank() == 2 and condition.rank() == 2);
        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const hidden = input.withTags(.{ .b, .d });
        debug_value("hidden", hidden);
        const context = condition.convert(dtype).withTags(.{ .b, .d });
        debug_value("context", context);
        std.debug.assert(hidden.dim(.b) == context.dim(.b) and hidden.dim(.d) == context.dim(.d));

        const modulation = self.condition_modulation(context);
        debug_value("modulation", modulation);
        // Final normalization has no learned gain or bias.
        const normalized = normalize_feature_variance(hidden, self.eps);
        debug_value("normalized", normalized);
        const modulated = apply_modulation(normalized, modulation.shift, modulation.scale);
        debug_value("modulated", modulated);
        const debug_output: zml.Tensor = self.proj.forward(modulated, dtype);
        debug_value("output", debug_output);
        return debug_output;
    }

    const Modulation = struct { shift: zml.Tensor, scale: zml.Tensor };

    /// Condition [B,.d=512] -> shift and scale, each [B,.d=512]. No residual gate.
    fn condition_modulation(self: Gradium_Phonon_TTS_Flow_Final_Layer, condition: zml.Tensor) Modulation {
        const here = @typeName(@This()) ++ ".condition_modulation";

        debug_enter(here);
        defer debug_exit(here);
        debug_value("condition", condition);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".condition_modulation", .{});
        defer compiler.popLocation();

        const projected = self.adaln_proj.forward(condition.silu(), condition.dtype());
        debug_value("projected", projected);
        const parts = projected.chunkExact(.m, 2);
        debug_value("parts", parts);
        const debug_output: Modulation = .{ .shift = parts[0].rename(.{ .m = .d }), .scale = parts[1].rename(.{ .m = .d }) };
        debug_value("output", debug_output);
        return debug_output;
    }
};

// --- Shared flow normalization operations ---

/// Center and normalize [B,.d] using population variance. Statistics stay in
/// the input dtype to preserve the flow reference's arithmetic.
fn normalize_feature_variance(hidden: zml.Tensor, eps: f32) zml.Tensor {
    const here = @typeName(@This()) ++ ".normalize_feature_variance";

    debug_enter(here);
    defer debug_exit(here);
    debug_value("hidden", hidden);

    const compiler = zml.Compiler.current();
    compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".normalize_feature_variance", .{});
    defer compiler.popLocation();

    const centered = hidden.sub(hidden.mean(.d).broad(hidden.shape()));
    debug_value("centered", centered);
    const inverse_std = centered.mul(centered).mean(.d).addConstant(eps).rsqrt();
    debug_value("inverse_std", inverse_std);
    const debug_output: zml.Tensor = centered.mul(inverse_std.broad(hidden.shape()));
    debug_value("output", debug_output);
    return debug_output;
}

/// Apply adaptive normalization's feature-wise scale and shift to [B,.d].
fn apply_modulation(normalized: zml.Tensor, shift: zml.Tensor, scale: zml.Tensor) zml.Tensor {
    const here = @typeName(@This()) ++ ".apply_modulation";

    debug_enter(here);
    defer debug_exit(here);
    debug_value("normalized", normalized);
    debug_value("shift", shift);
    debug_value("scale", scale);

    const compiler = zml.Compiler.current();
    compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".apply_modulation", .{});
    defer compiler.popLocation();

    const debug_output: zml.Tensor = normalized.mul(scale.addConstant(1)).add(shift);
    debug_value("output", debug_output);
    return debug_output;
}

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

// Structured outputs retain their field names; RNGs expose only their state shape.
fn debug_value(comptime name: []const u8, value: anytype) void {
    if (debug_selected_depth == null) return;
    const T = @TypeOf(value);
    if (T == zml.Tensor) {
        debug_print("{s}: {f}\n", .{ name, value.shape() });
    } else if (T == zml.Tensor.Rng) {
        debug_value(name ++ "._state", value._state);
    } else switch (@typeInfo(T)) {
        .@"struct" => |info| inline for (info.field_names) |field| {
            debug_value(name ++ "." ++ field, @field(value, field));
        },
        .array => |info| {
            if (info.child == zml.Tensor or info.child == Gradium_Phonon_TTS_Attention.KV_Cache) {
                inline for (value, 0..) |tensor, i| {
                    debug_value(std.fmt.comptimePrint("{s}[{}]", .{ name, i }), tensor);
                }
            }
        },
        .optional => |info| {
            if (info.child == zml.Tensor) {
                if (value) |tensor| {
                    debug_value(name, tensor);
                } else {
                    debug_print("{s}: null\n", .{name});
                }
            }
        },
        else => {},
    }
}
