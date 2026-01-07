const std = @import("std");
const zml = @import("zml");

const llama = @import("llama.zig");

const LlamaLM = llama.LlamaLM;

// pub const Sampling = struct {
//     embed_tokens: zml.nn.Embedding,

//     pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !Sampling {
//         _ = allocator; // autofix
//         return .{
//             .embed_tokens = store.withPrefix("embed_tokens").createTensor("weight"),
//         };
//     }

//     pub fn loadBuffers(self: *const Sampling, allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, platform: zml.Platform) !zml.Bufferized(Sampling) {
//         return zml.io.loadBuffersFromId(allocator, io, self, store.withPrefix("embed_tokens"), platform);
//     }

//     pub fn forward(
//         self: LlamaLM,
//         lm_head_: ?zml.nn.Linear,
//         out_: Tensor,
//         rng: Tensor.Rng,
//         opts: zml.nn.SamplingStrategy,
//     ) struct { Tensor, Tensor.Rng } {
//         const out = out_.withPartialTags(.{ .s, .d });

//         var logits = blk: {
//             if (lm_head_) |lm_head| {
//                 break :blk lm_head.forward(out).rename(.{ .dout = .d });
//             } else {
//                 break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .d);
//             }
//         };

//         if (logits.shape().hasTag(.voc) == null)
//             logits = logits.rename(.{ .d = .voc });

//         const next_tokens, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
//         return .{ next_tokens, new_rng };
//     }
// };

pub const LlamaLayers = struct {
    embedding_exe: zml.Exe,
    layers_exe: zml.Exe,
    sampling_exe: zml.Exe,
    // embedding_buffers: zml.Bufferized(zml.nn.Embedding),
    // layers_buffers: []zml.Bufferized(llama.LlamaLayer),
    // sampling_buffers: zml.Bufferized(Sampling),

    pub fn compile(
        parent_allocator: std.mem.Allocator,
        io: std.Io,
        platform: zml.Platform,
        model_instance: *const LlamaLM,
        tokens_tensor: zml.Tensor,
        idx_tensor: zml.Tensor,
        kv_cache: llama.KvCache,
        rng: zml.Tensor.Rng,
        //store: zml.io.TensorStore
    ) LlamaLayers {
        const embedding = model_instance.model.embed_tokens;
        const embedding_exe = try platform.compile(parent_allocator, io, embedding, .forward, .{tokens_tensor});

        const hidden_states_tensor = zml.Tensor.init(zml.Shape.init(.{ .b = tokens_tensor.dim(.b), .d = model_instance.config.hidden_size }, .bf16), .bf16);
        const layer = model_instance.model.layers[0];
        const layer_exe = try platform.compile(parent_allocator, io, layer, .forward, .{ hidden_states_tensor, idx_tensor, kv_cache });

        // must handle lm_head
        const sampling_exe = try platform.compile(parent_allocator, io, model_instance, .sampleTokens, .{ hidden_states_tensor, rng, model_instance.sampling_opts });

        return .{
            .embedding_exe = embedding_exe,
            .layer_exe = layer_exe,
            .sampling_exe = sampling_exe,
        };
    }
};
