const std = @import("std");

const zml = @import("zml");
const stdx = @import("stdx");

const Tensor = @import("main.zig").Tensor;
const TensorDescriptor = @import("main.zig").TensorDescriptor;
const BufferStore4 = @import("main.zig").BufferStore4;
const BufferStore5 = @import("main.zig").BufferStore5;
const Bufferized = @import("main.zig").Bufferized;
const loadBuffersFromId = @import("main.zig").loadBuffersFromId;
const multiTransfer = @import("main.zig").multiTransfer;
const singleTransfer = @import("main.zig").singleTransfer;
const structTransfer = @import("main.zig").structTransfer;
const initBufferizedFrom = @import("main.zig").initBufferizedFrom;
const compile = @import("main.zig").compile;

pub const LlamaLM = struct {
    pub const Config = struct {
        bos_token_id: u32,
        eos_token_id: stdx.json.Union(union(enum) {
            int: u32,
            ints: []u32,
        }),
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rope_theta: f32,
        max_position_embeddings: usize,
        rms_norm_eps: f32,
        hf_rope_impl: bool = true,
        //rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = {} },
    };

    pub const Options = struct {
        //sampling_strategy: ?zml.nn.SamplingStrategy,
        max_seq_len: usize,
    };

    lm_head: ?Linear,
    model: Llama,

    // Options controlling generation
    //gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(allocator: std.mem.Allocator, buffer_store: BufferStore5.View, config: LlamaLM.Config) !LlamaLM {
        return .{
            .lm_head = null,
            .model = try .init(allocator, buffer_store.withPrefix("model"), config),
            .config = config,
        };
    }

    pub fn deinit(self: LlamaLM, allocator: std.mem.Allocator) void {
        allocator.free(self.model.layers);
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, llama: LlamaLM, buffer_store: BufferStore5.View, platform: zml.Platform) !LlamaLMBufferized {
        return .{
            .lm_head = null,
            .model = try Llama.loadBuffers(allocator, llama.model, buffer_store.withPrefix("model"), platform),
        };
    }
};

pub const LlamaLMBufferized = struct {
    lm_head: ?Bufferized(Linear),
    model: Bufferized(Llama),

    pub fn deinit(self: LlamaLMBufferized, allocator: std.mem.Allocator) void {
        allocator.free(self.model.layers);
    }
};

pub const Linear = struct {
    weight: Tensor,
    bias: ?Tensor = null,
    tag: zml.Shape.Tag,

    pub fn initWithTag(buffer_store: BufferStore5.View, tag: anytype) Linear {
        return .{
            .weight = buffer_store.createTensorWithTags("weight", .{ .dout, tag }),
            .bias = buffer_store.maybeCreateTensorWithTags("bias", .{.dout}),
            .tag = zml.Shape.toTag(tag),
        };
    }
};

pub const Llama = struct {
    embed_tokens: struct { weight: Tensor },
    norm: RmsNorm,
    layers: []TransformerLayer,

    max_seq_len: u32,
    num_heads: i64,
    num_kv_heads: i64,

    pub fn init(allocator: std.mem.Allocator, buffer_store: BufferStore5.View, config: LlamaLM.Config) !Llama {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            var buffer: [16]u8 = undefined;
            const number_prefix = try std.fmt.bufPrint(&buffer, "layers.{d}", .{i});
            layer.* = .init(buffer_store.withPrefix(number_prefix), config);
        }

        return .{
            .embed_tokens = .{ .weight = buffer_store.createTensorWithTags("embed_tokens.weight", .{ .x, .y }) },
            .norm = .init(buffer_store.withPrefix("norm"), config),
            .layers = layers,
            .max_seq_len = 256,
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
        };
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, model: Llama, buffer_store: BufferStore5.View, platform: zml.Platform) !Bufferized(Llama) {
        std.debug.print("Loading Llama\n", .{});
        const layers = try allocator.alloc(Bufferized(TransformerLayer), model.layers.len);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            var buffer: [16]u8 = undefined;
            const number_prefix = try std.fmt.bufPrint(&buffer, "layers.{d}", .{i});
            std.debug.print("Loading layer {d}\n", .{i});
            layer.* = try TransformerLayer.loadBuffers(allocator, model.layers[i], buffer_store.withPrefix(number_prefix), platform);
        }

        return .{
            .embed_tokens = try loadBuffersFromId(allocator, model.embed_tokens, buffer_store.withPrefix("embed_tokens"), platform),
            .layers = layers,
            .norm = try loadBuffersFromId(allocator, model.norm, buffer_store.withPrefix("norm"), platform),
        };
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    pub fn init(buffer_store: BufferStore5.View, config: LlamaLM.Config) RmsNorm {
        return .{
            .weight = buffer_store.createTensorWithTags("weight", .{.d}),
            .eps = config.rms_norm_eps,
        };
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(buffer_store: BufferStore5.View, config: LlamaLM.Config) TransformerLayer {
        return .{
            .input_layernorm = .init(buffer_store.withPrefix("input_layernorm"), config),
            .self_attn = SelfAttn.init(buffer_store.withPrefix("self_attn"), config),
            .post_attention_layernorm = .init(buffer_store.withPrefix("post_attention_layernorm"), config),
            .mlp = .init(buffer_store.withPrefix("mlp")),
        };
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, layer: TransformerLayer, buffer_store: BufferStore5.View, platform: zml.Platform) !Bufferized(TransformerLayer) {
        return .{
            .input_layernorm = try loadBuffersFromId(allocator, layer.input_layernorm, buffer_store.withPrefix("input_layernorm"), platform),
            .self_attn = try SelfAttn.loadBuffers(allocator, layer.self_attn, buffer_store.withPrefix("self_attn"), platform),
            .post_attention_layernorm = try loadBuffersFromId(allocator, layer.post_attention_layernorm, buffer_store.withPrefix("post_attention_layernorm"), platform),
            .mlp = try loadBuffersFromId(allocator, layer.mlp, buffer_store.withPrefix("mlp"), platform),
        };
    }
};

pub const SelfAttn = struct {
    qkv_proj: Linear,
    //q_proj: Linear,
    //k_proj: Linear,
    //v_proj: Linear,

    o_proj: Linear,
    num_heads: i64,
    num_kv_heads: i64,

    pub fn init(buffer_store: BufferStore5.View, config: LlamaLM.Config) SelfAttn {
        const q_weight_shape = buffer_store.getShape("q_proj.weight").?;
        const k_weight_shape = buffer_store.getShape("k_proj.weight").?;
        const v_weight_shape = buffer_store.getShape("v_proj.weight").?;
        const qkv_weight_shape = zml.Shape.concatenate(&.{ q_weight_shape, k_weight_shape, v_weight_shape }, 0);
        return .{
            .qkv_proj = .{ .weight = Tensor.init(qkv_weight_shape).withTags(.{ .dout, .d }), .tag = zml.Shape.toTag(.d) },
            .o_proj = .initWithTag(buffer_store.withPrefix("o_proj"), .d),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
        };
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, self_attn: SelfAttn, buffer_store: BufferStore5.View, platform: zml.Platform) !Bufferized(SelfAttn) {
        const q_proj = Linear.initWithTag(buffer_store.withPrefix("q_proj"), .d);
        const k_proj = Linear.initWithTag(buffer_store.withPrefix("k_proj"), .d);
        const v_proj = Linear.initWithTag(buffer_store.withPrefix("v_proj"), .d);

        const exe2 = try compile(allocator, mergeQkv2, .{ q_proj, k_proj, v_proj }, platform);
        defer exe2.deinit();

        var args = try exe2.args(allocator);
        defer args.deinit(allocator);

        var results = try exe2.results(allocator);
        defer results.deinit(allocator);

        const q_proj_buffers = try loadBuffersFromId(allocator, q_proj, buffer_store.withPrefix("q_proj"), platform);
        const k_proj_buffers = try loadBuffersFromId(allocator, k_proj, buffer_store.withPrefix("k_proj"), platform);
        const v_proj_buffers = try loadBuffersFromId(allocator, v_proj, buffer_store.withPrefix("v_proj"), platform);

        args.set(.{ q_proj_buffers, k_proj_buffers, v_proj_buffers });
        exe2.call(args, &results);

        var qkv_proj: Bufferized(Linear) = undefined;
        initBufferizedFrom(self_attn.qkv_proj, &qkv_proj);

        results.fill(&qkv_proj);
        errdefer qkv_proj.weight.deinit();
        errdefer if (qkv_proj.bias != null) qkv_proj.bias.?.deinit();

        const o_proj = try loadBuffersFromId(allocator, self_attn.o_proj, buffer_store.withPrefix("o_proj"), platform);

        return .{
            .qkv_proj = qkv_proj,
            .o_proj = o_proj,
        };
    }
};

fn mergeQkv(q: Tensor, k: Tensor, v: Tensor) Tensor {
    return Tensor.concatenate(&.{ q, k, v }, 0);
}

fn mergeQkv2(q: Linear, k: Linear, v: Linear) Linear {
    return .{
        .weight = Tensor.concatenate(&.{ q.weight, k.weight, v.weight }, 0),
        .bias = if (q.bias != null) Tensor.concatenate(&.{ q.bias.?, k.bias.?, v.bias.? }, 0) else null,
        .tag = q.tag,
    };
}

const Mlp = struct {
    up_proj: Linear, // (dim -> hidden_dim)
    gate_proj: Linear, // (dim -> hidden_dim)
    down_proj: Linear, // (hidden_dim -> dim)

    pub fn init(buffer_store: BufferStore5.View) Mlp {
        return .{
            .up_proj = .initWithTag(buffer_store.withPrefix("up_proj"), .d),
            .gate_proj = .initWithTag(buffer_store.withPrefix("gate_proj"), .d),
            .down_proj = .initWithTag(buffer_store.withPrefix("down_proj"), .d),
        };
    }
};
