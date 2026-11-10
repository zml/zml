const std = @import("std");
const zml = @import("zml");

const transformer = @import("transformer.zig");
const eos_head = @import("eos_head.zig");
const flow_head = @import("flow_head.zig");
const util = @import("util.zig");

test {
    std.testing.refAllDecls(@This());
}

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
/// `forward` recomputes the full prefix. `forward_prefill` processes the initial
/// prompt and fills a K/V cache; `forward_decode` processes one completed audio
/// frame using that cached context. Prefill already produces the prediction for
/// the first audio frame. All paths return f32 EOS scores `[B, 1]` and latent
/// velocities `[B, 32]`; cached paths also return updated attention history.
/// B counts independent sequences; inputs must be unpadded.
///
/// Parameters come from the root checkpoint view: `model` owns the transformer,
/// `flow_head` uses `flow_lm.flow_net`, and `eos_head` uses `flow_lm.out_eos`.
/// The children target phonon-31m-en and use stored bf16 parameters in f32.
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
    model: transformer.Gradium_Phonon_TTS_Model,
    flow_head: flow_head.Gradium_Phonon_TTS_Flow_Head,
    eos_head: eos_head.EOS_Head,

    /// Returns the two predictions used to complete one normalized audio frame.
    ///
    /// `eos_score`: f32 `[.b=B, .dout=1]`, a raw stopping logit from the last
    /// transformer position. The host generation loop interprets it using its policy.
    /// `velocity`: f32 `[.b=B, .ld=32]`, the flow head's latent velocity for the
    /// supplied candidate and interval. The sampler applies the numerical update.
    /// Both tensors preserve batch ordering. Executable results use
    /// `Bufferized(Output)`; the caller releases both buffers.
    pub const Output = struct {
        eos_score: zml.Tensor,
        velocity: zml.Tensor,
    };

    /// Expects the root checkpoint view.
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_For_Conditional_Generation {
        return .{
            .model = .init(store),
            .flow_head = .init(store.withPrefix("flow_lm.flow_net")),
            .eos_head = .init(store.withPrefix("flow_lm.out_eos")),
        };
    }

    /// Predicts an EOS score and flow velocity from the complete sequence prefix.
    ///
    /// `token_ids`: integer `[B, S]`, normally i64, the unpadded sequence from
    /// the tokenizer and generation loop. S>0; the prompt ends at audio BOS (ID 4)
    /// initially, then gains one ID 6 per completed generated frame.
    /// `voice`: f32 `[B, P, 512]`, selected speaker features loaded from the voice
    /// asset as `[B, 512, P]` and transposed by the caller. P matches the number
    /// of ID-5 markers in each sequence, in frame order.
    /// `audio_latents`: f32 `[B, A, 32]`, the generation loop's complete normalized
    /// audio history, one completed frame per ID-6 marker in order; A=0 initially.
    /// `candidate`: f32 `[B, 32]`, the sampler's current candidate for the next
    /// frame in normalized latent space; freshly drawn noise in the one-step sampler.
    /// `start_time`: f32 `[B, 1]`, starting flow progress supplied by the sampler.
    /// `end_time`: f32 `[B, 1]`, ending flow progress supplied by the sampler.
    /// Times normally lie in [0, 1]; the current sampler uses 0 and 1. They describe
    /// flow progress, rather than audio timestamps. All inputs share batch size B.
    ///
    /// Returns f32 EOS logits `[B, 1]` and velocities `[B, 32]`, both conditioned
    /// on only the last position of the transformer output. The caller applies the velocity
    /// update, stopping policy, and feedback of completed frames.
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
    /// Inputs: integer IDs `[b,s]`, f32 voice `[b,p,ad=512]`, f32 complete
    /// history `[b,a,ld=32]`, f32 candidate
    /// `[b,ld=32]`, and f32 start/end times `[b,t=1]` each.
    /// Outputs: f32 eos_score `[b,dout=1]` and velocity `[b,ld=32]`.
    pub fn forward(
        self: Gradium_Phonon_TTS_For_Conditional_Generation,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        audio_latents: zml.Tensor,
        candidate: zml.Tensor,
        start_time: zml.Tensor,
        end_time: zml.Tensor,
    ) Output {
        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_For_Conditional_Generation", .{});
        defer compiler.popLocation();

        const hidden = self.model.forward(token_ids, voice, audio_latents); // [B,S] integer, [B,P,512] f32, [B,A,32] f32 -> [B,S,d=512] f32
        std.debug.assert(hidden.dim(.s) > 0);
        const last = hidden.slice(.s, .{ .start = hidden.dim(.s) - 1, .len = 1 }); // [B,s=S,512] f32 -> [B,s=1,512] f32
        const condition = last.squeeze(.s); // [B,s=1,512] f32 -> [B,d=512] f32

        const score = self.eos_head.forward(condition); // [B,d=512] f32 -> [B,dout=1] f32
        const velocity = self.flow_head.forward(candidate, condition, start_time, end_time); // [B,32], [B,512], [B,1], [B,1] f32 -> [B,ld=32] f32
        return .{ .eos_score = score, .velocity = velocity };
    }

    /// Returns an EOS score, latent velocity, and updated attention history.
    ///
    /// `eos_score` is f32 `[.b=B, .dout=1]` and `velocity` is f32 `[.b=B, .ld=32]`,
    /// with the same meanings as `Output`. `cache` contains rotated keys and
    /// unrotated values, each f32 `[.layer=6, .b=B, .k=C, .h=8, .hd=64]`.
    /// The cached wrapper donates input cache storage to these returned tensors.
    /// Raw executable callers must destroy the old donated handles and retain the
    /// returned handles exactly once; the cache capacity C stays fixed.
    pub const CachedOutput = struct {
        eos_score: zml.Tensor,
        velocity: zml.Tensor,
        cache: transformer.KV_Cache,
    };

    /// Processes the complete prompt once and predicts the first audio frame.
    ///
    /// `ids`: integer `[B, S]`, normally i64, the tokenizer's unpadded prompt
    /// ending at audio BOS (ID 4), uploaded by the generation loop. S must be nonzero.
    /// `voice`: f32 `[B, P, 512]`, selected speaker features loaded from the voice
    /// asset and transposed by the host; one ordered frame per ID-5 marker.
    /// `candidate`: f32 `[B, 32]`, the sampler's candidate for the next normalized
    /// audio frame, initially noise; passed through to the flow head.
    /// `start`: f32 `[B, 1]`, the sampler-supplied interval's starting flow progress.
    /// `end`: f32 `[B, 1]`, the sampler-supplied interval's ending flow progress.
    /// The current one-step sampler supplies 0 and 1 and computes candidate + velocity.
    /// `cache`: host-allocated K/V storage, each f32 `[6, B, C, 8, 64]`, initially
    /// filled with finite zeros. C must be at least the prompt length S.
    /// All inputs share B. There are no completed generated frames yet.
    ///
    /// Returns EOS logits `[B, 1]`, latent velocities `[B, 32]`, and updated cache.
    /// Only the last prompt position conditions the heads. The cache contains all
    /// prompt positions and donates its storage to the returned handles.
    ///
    /// Dimension tags:
    /// - `.b`: sequences; `.s`: prompt positions; `.p`: voice frames; `.ad`: 512 voice features.
    /// - `.d`: 512 transformer features; `.dout`: one EOS score; `.ld`: 32 latent coordinates.
    /// - `.t`: one flow time per row; cache `.layer`=6, `.k`=C, `.h`=8, `.hd`=64.
    pub fn forward_prefill(self: Gradium_Phonon_TTS_For_Conditional_Generation, ids: zml.Tensor, voice: zml.Tensor, candidate: zml.Tensor, start: zml.Tensor, end: zml.Tensor, cache: transformer.KV_Cache) CachedOutput {
        const result = self.model.forward_prefill(ids, voice, cache); // IDs + voice + cache -> hidden [B,P,512] f32 + updated cache
        const last = result.hidden.slice(.s, .{ .start = result.hidden.dim(.s) - 1, .len = 1 }); // [B,P,512] f32 -> [B,1,512] f32
        const condition = last.squeeze(.s); // [B,1,512] f32 -> [B,d=512] f32
        const score = self.eos_head.forward(condition); // [B,512] f32 -> [B,1] f32
        const velocity = self.flow_head.forward(candidate, condition, start, end); // [B,32], [B,512], 2 x [B,1] f32 -> [B,32] f32
        const updated_cache = result.cache.reuseBuffer(cache); // 2 x [6,B,C,8,64] f32 -> same shapes/dtypes, donating input cache
        return .{ .eos_score = score, .velocity = velocity, .cache = updated_cache };
    }

    /// Consumes one completed audio frame and predicts the next using cached context.
    ///
    /// `frame`: f32 `[B, 1, 32]`, the latest completed normalized latent, selected
    /// from accumulated output history by the generation loop. It becomes the
    /// transformer's single new input position, rather than the new sampler candidate.
    /// `position`: scalar i32 `[]`, that frame's absolute sequence slot, supplied
    /// by the generation loop. The first decode slot follows the prompt; later calls
    /// advance by one. The host must keep it within the cache capacity.
    /// `candidate`: f32 `[B, 32]`, the sampler's candidate for the next normalized
    /// audio frame, initially noise; passed through to the flow head.
    /// `start`: f32 `[B, 1]`, the sampler-supplied interval's starting flow progress.
    /// `end`: f32 `[B, 1]`, the sampler-supplied interval's ending flow progress.
    /// The current one-step sampler supplies 0 and 1 and computes candidate + velocity.
    /// `cache`: f32 K/V history, each `[6, B, C, 8, 64]`, returned by the previous
    /// prefill/decode call. All earlier sequence slots must already be populated.
    /// Frame, candidate, times, and cache share batch size B.
    ///
    /// Returns EOS logits `[B, 1]`, latent velocities `[B, 32]`, and updated cache.
    /// The transformer produces `[B, 1, 512]` and the heads use its single position.
    /// Exactly this new slot is written per layer; cache storage is donated to the
    /// returned handles. The caller updates the candidate and applies stopping rules.
    ///
    /// Dimension tags:
    /// - `.b`: sequences; frame `.a`=1, hidden `.s`=1; `.d`=512 transformer features.
    /// - `.ld`: 32 latent coordinates; `.t`: one flow time; `.dout`: one EOS score.
    /// - Cache: `.layer`=6, `.k`=C slots, `.h`=8 heads, `.hd`=64 head features.
    pub fn forward_decode(self: Gradium_Phonon_TTS_For_Conditional_Generation, frame: zml.Tensor, position: zml.Tensor, candidate: zml.Tensor, start: zml.Tensor, end: zml.Tensor, cache: transformer.KV_Cache) CachedOutput {
        const result = self.model.forward_decode(frame, position, cache); // [B,1,32] + cache f32, [] i32 -> [B,1,512] f32 + updated cache
        const condition = result.hidden.squeeze(.s); // [B,1,512] f32 -> [B,d=512] f32
        const score = self.eos_head.forward(condition); // [B,512] f32 -> [B,1] f32
        const velocity = self.flow_head.forward(candidate, condition, start, end); // [B,32], [B,512], 2 x [B,1] f32 -> [B,32] f32
        const updated_cache = result.cache.reuseBuffer(cache); // 2 x [6,B,C,8,64] f32 -> same shapes/dtypes, donating input cache
        return .{ .eos_score = score, .velocity = velocity, .cache = updated_cache };
    }

    pub fn load(
        self: *const Gradium_Phonon_TTS_For_Conditional_Generation,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation) {
        // Delegate to children so their layer arrays are loaded and cleaned up correctly.
        var buffers: zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation) = undefined;
        buffers.model = try self.model.load(allocator, io, platform, store);
        errdefer transformer.Gradium_Phonon_TTS_Model.unloadBuffers(&buffers.model);
        buffers.flow_head = try self.flow_head.load(allocator, io, platform, store);
        errdefer flow_head.Gradium_Phonon_TTS_Flow_Head.unloadBuffers(&buffers.flow_head);
        buffers.eos_head = try self.eos_head.load(allocator, io, platform, store);
        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_For_Conditional_Generation)) void {
        transformer.Gradium_Phonon_TTS_Model.unloadBuffers(&self.model);
        flow_head.Gradium_Phonon_TTS_Flow_Head.unloadBuffers(&self.flow_head);
        eos_head.EOS_Head.unloadBuffers(&self.eos_head);
    }
};

test Gradium_Phonon_TTS_For_Conditional_Generation {
    const allocator = std.testing.allocator;
    const path = try util.assetPath(allocator, "zml/examples/phonon/assets/prediction.activations.safetensors");
    defer allocator.free(path);
    try util.test_module_outputs(
        Gradium_Phonon_TTS_For_Conditional_Generation,
        util.gradium_phonon_31m_en,
        path,
        "",
        .{ "input_ids", "voice", "audio_latents", "candidate", "start_time", "end_time" },
        .forward,
        .{ "eos_score", "velocity" },
    );
}
