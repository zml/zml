const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");
const dit = @import("acedit.zig");

const dialects = @import("mlir/dialects");


pub const AceEnc_handler = struct {
    model: AceEnc,
    silence: SilenceGenerator,
    params: Params,
    config: dit.ConfigXl,
    exes: Exes,
    model_buffers: zml.Bufferized(AceEnc),
    silence_buffers: zml.Bufferized(SilenceGenerator),
    shardings: main.Shardings,
    
    pub fn init(zml_handler: *main.Zml_handler, text_len: u32, lyric_len: u32, target_duration: u32, audiocodes: usize) !AceEnc_handler {
        zml_handler.tic(&zml_handler.timers.enc.init);
        const repo_m = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.acedit);
        const repo_s = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.silence);

        var registry_m: zml.safetensors.TensorRegistry = try .fromRepo(zml_handler.allocator, zml_handler.io, repo_m);
        defer registry_m.deinit();
        
        var registry_s: zml.safetensors.TensorRegistry = try .fromRepoFile(zml_handler.allocator, zml_handler.io, repo_s, "silence_latent.safetensors");
        defer registry_s.deinit();
        
        std.log.info("ENC parse config and safetensors", .{});
        var config: dit.ConfigXl = undefined;
        if (zml_handler.uris.is_xl) {
            const parsed_config = try main.parseConfig(dit.ConfigXl, zml_handler.allocator, zml_handler.io, repo_m);
            defer parsed_config.deinit();
            config = try parsed_config.value.dupe(zml_handler.allocator);
        } else {
            const parsed_config = try main.parseConfig(dit.ConfigBase, zml_handler.allocator, zml_handler.io, repo_m);
            defer parsed_config.deinit();
            const xl_config = try dit.ConfigXl.fromBase(parsed_config.value);
            config = try xl_config.dupe(zml_handler.allocator);
        }
        std.log.info("ENC parsed", .{});        

        var store_m: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry_m);
        var store_s: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry_s);
        defer store_m.deinit();
        defer store_s.deinit();
        std.log.info("ENC init models", .{});
        const model: AceEnc = try .init(zml_handler.allocator, store_m.view(), config);
        std.log.info("ENC encoder initialized", .{});
        const silence: SilenceGenerator = .init(store_s.view(), config, target_duration * 25);
        std.log.info("ENC silence initialized", .{});
        
        const params: Params = .{
            .text_emb = .init(.{ .s = text_len, .d = config.text_hidden_dim }, .bf16),
            .lyric_emb = .init(.{ .s = lyric_len, .d = config.text_hidden_dim }, .bf16),
            .timbre_latent = .init(.{ .a = config.timbre_hidden_dim, .t = config.timbre_fix_frame }, .bf16),
            .audio_codes = .init(.{ .s = audiocodes }, .u32),
            .src_audio = .init(.{ .a = config.audio_acoustic_hidden_dim, .t = target_duration * 25 }, .bf16),
            .encoded_lyric = .init(.{ .s = lyric_len, .d = config.encoder_hidden_size }, .bf16),
            .encoded_timbre = .init(.{ .s = 1, .d = config.encoder_hidden_size }, .bf16),
        };
        
        const shardings: main.Shardings = try .init(zml_handler.platform);

        zml_handler.toc(&zml_handler.timers.enc.init);
        zml_handler.tic(&zml_handler.timers.enc.compile);
        
        std.log.info("ENC compile models", .{});
        const exes = try compileModel(zml_handler, model, silence, params, shardings);
        std.log.info("ENC compiled", .{});

        zml_handler.toc(&zml_handler.timers.enc.compile);
        zml_handler.tic(&zml_handler.timers.enc.load);
        
        std.log.info("ENC load buffers", .{});
        const model_buffers = try model.load(zml_handler, &store_m, &shardings.all());
        std.log.info("ENC loaded encoder buffers", .{});
        const silence_buffers = try silence.load(zml_handler, &store_s, &shardings.all());
        std.log.info("ENC loaded silence buffers", .{});
        
        zml_handler.toc(&zml_handler.timers.enc.load);
        
        return .{
            .model = model,
            .silence = silence,
            .params = params,
            .config = config,
            .exes = exes,
            .model_buffers = model_buffers,
            .silence_buffers = silence_buffers,
            .shardings = shardings,
        };
    }

    pub fn compileModel(zml_handler: *main.Zml_handler, model: AceEnc, silence: SilenceGenerator, params: Params, shardings: main.Shardings) !Exes {
        const shardings_arr = shardings.all();
        const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };

        var encode_lyric_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: LyricEncoder, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{
                        params_.lyric_emb }, opts_);
            }
        }.call, .{ zml_handler, model.lyric_encoder, params, opts });
        var encode_lyric_future_awaited = false;
        errdefer if (!encode_lyric_future_awaited) if (encode_lyric_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var encode_timbre_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: TimbreEncoder, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{
                        params_.timbre_latent }, opts_);
            }
        }.call, .{ zml_handler, model.timbre_encoder, params, opts });
        var encode_timbre_future_awaited = false;
        errdefer if (!encode_timbre_future_awaited) if (encode_timbre_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var encode_audiocodes_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AudioCodeEncoder, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{
                        params_.audio_codes }, opts_);
            }
        }.call, .{ zml_handler, model.audiocode_encoder, params, opts });
        var encode_audiocodes_future_awaited = false;
        errdefer if (!encode_audiocodes_future_awaited) if (encode_audiocodes_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};
        
        var encode_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceEnc, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward,
                    .{ params_.text_emb, params_.encoded_lyric, params_.encoded_timbre, params_.src_audio }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var encode_future_awaited = false;
        errdefer if (!encode_future_awaited) if (encode_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};
        
        var silence_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: SilenceGenerator, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{ }, opts_);
            }
        }.call, .{ zml_handler, silence, opts });
        var silence_future_awaited = false;
        errdefer if (!silence_future_awaited) if (silence_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};
        
        const encode_lyric_exe = try encode_lyric_future.await(zml_handler.io);
        encode_lyric_future_awaited = true;
        
        const encode_timbre_exe = try encode_timbre_future.await(zml_handler.io);
        encode_timbre_future_awaited = true;
        
        const encode_audiocodes_exe = try encode_audiocodes_future.await(zml_handler.io);
        encode_audiocodes_future_awaited = true;

        const encode_exe = try encode_future.await(zml_handler.io);
        encode_future_awaited = true;
        
        const silence_exe = try silence_future.await(zml_handler.io);
        silence_future_awaited = true;

        return .{
            .encode_lyric_exe = encode_lyric_exe,
            .encode_lyric_args = try encode_lyric_exe.args(zml_handler.allocator),
            .encode_lyric_results = try encode_lyric_exe.results(zml_handler.allocator),
            .encode_timbre_exe = encode_timbre_exe,
            .encode_timbre_args = try encode_timbre_exe.args(zml_handler.allocator),
            .encode_timbre_results = try encode_timbre_exe.results(zml_handler.allocator),
            .encode_audiocodes_exe = encode_audiocodes_exe,
            .encode_audiocodes_args = try encode_audiocodes_exe.args(zml_handler.allocator),
            .encode_audiocodes_results = try encode_audiocodes_exe.results(zml_handler.allocator),
            .encode_exe = encode_exe,
            .encode_args = try encode_exe.args(zml_handler.allocator),
            .encode_results = try encode_exe.results(zml_handler.allocator),
            .silence_exe = silence_exe,
            .silence_args = try silence_exe.args(zml_handler.allocator),
            .silence_results = try silence_exe.results(zml_handler.allocator),
        };
    }
    
    pub fn unloadBuffers(self: *AceEnc_handler, allocator: std.mem.Allocator) void {
        AceEnc.unloadBuffers(&self.model_buffers, allocator);
        SilenceGenerator.unloadBuffers(&self.silence_buffers);
    }

    pub fn deinit(self: *AceEnc_handler, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.exes.deinit(allocator);
    }
    
};


pub const Params = struct {
    // caption Enceddings, expected shape: [s_text, d_emb]
    text_emb: zml.Tensor,
    // lyric embeddings, expected shape: [s_lyric, d_emb]
    lyric_emb: zml.Tensor,
    // timbre source, expected shape: [config.timbre_fix_frame, d_audio]
    timbre_latent: zml.Tensor,
    // audio codes, expected shape: [t_5hz, 1]
    audio_codes: zml.Tensor,
    // source audio, expected shape: [t_25hz, d_audio]
    src_audio: zml.Tensor,

    encoded_lyric: zml.Tensor,
    encoded_timbre: zml.Tensor,
};

pub const LayerType = enum {
    full_attention,
    sliding_attention,
};

pub const Exes = struct {
    encode_lyric_exe: zml.Exe,
    encode_lyric_args: zml.Exe.Arguments,
    encode_lyric_results: zml.Exe.Results,

    encode_timbre_exe: zml.Exe,
    encode_timbre_args: zml.Exe.Arguments,
    encode_timbre_results: zml.Exe.Results,

    encode_audiocodes_exe: zml.Exe,
    encode_audiocodes_args: zml.Exe.Arguments,
    encode_audiocodes_results: zml.Exe.Results,

    encode_exe: zml.Exe,
    encode_args: zml.Exe.Arguments,
    encode_results: zml.Exe.Results,

    silence_exe: zml.Exe,
    silence_args: zml.Exe.Arguments,
    silence_results: zml.Exe.Results,

    pub fn deinit(self: Exes, allocator: std.mem.Allocator) void {
        self.encode_lyric_exe.deinit();
        self.encode_lyric_args.deinit(allocator);
        self.encode_lyric_results.deinit(allocator);
        self.encode_timbre_exe.deinit();
        self.encode_timbre_args.deinit(allocator);
        self.encode_timbre_results.deinit(allocator);
        self.encode_audiocodes_exe.deinit();
        self.encode_audiocodes_args.deinit(allocator);
        self.encode_audiocodes_results.deinit(allocator);
        self.encode_exe.deinit();
        self.encode_args.deinit(allocator);
        self.encode_results.deinit(allocator);
        self.silence_exe.deinit();
        self.silence_args.deinit(allocator);
        self.silence_results.deinit(allocator);
    }
};


pub const SilenceGenerator = struct {
    // dim = [1, 64, 15000] = [.b, .a, .t_25hz] where 15000 in 25hz is 600s the max audio output length
    silence_latent: zml.Tensor,
    time_timbre: u32,
    time_25hz: u32,
    
    pub fn init(store: zml.io.TensorStore.View, config: dit.ConfigXl, time_25hz: u32) SilenceGenerator {
        return .{
            .silence_latent = store.createTensor("silence_latent", .{ .batch, .audio, .time }, null),
            .time_timbre = config.timbre_fix_frame,
            .time_25hz = time_25hz,
        };
    }
    
    pub fn load(self: *const SilenceGenerator, zml_handler: *main.Zml_handler, store: *zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(SilenceGenerator) {
        return zml.io.load(SilenceGenerator, self, zml_handler.arena.allocator(), zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 128 * 1024 * 1024,
        });
    }
    
    pub fn unloadBuffers(self: *zml.Bufferized(SilenceGenerator)) void {
        self.silence_latent.deinit();
    }
    
    // we return 2 silence slices : one for the timbre reference, of length time_timbre
    // and one for the source audio reference, of length time_25hz
    pub fn forward(self: SilenceGenerator) struct {zml.Tensor, zml.Tensor } {
        const full_audio_slice: zml.Tensor.Slice = .{ .start = 0, .end = self.silence_latent.dim(.audio) };
        const target_timbre_slice: zml.Tensor.Slice = .{ .start = 0, .end = self.time_timbre };
        const target_time_slice: zml.Tensor.Slice = .{ .start = 0, .end = self.time_25hz };
        
        const timbre_slice = self.silence_latent.squeeze(.batch).slice(&.{ full_audio_slice, target_timbre_slice });
        const audio_slice = self.silence_latent.squeeze(.batch).slice(&.{ full_audio_slice, target_time_slice });
        
        return .{ timbre_slice.convert(.bf16), audio_slice.convert(.bf16) };
    }
};


pub const AceEnc = struct {
    text_encoder: TextEncoder,
    lyric_encoder: LyricEncoder,
    timbre_encoder: TimbreEncoder,
    audiocode_encoder: AudioCodeEncoder,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: dit.ConfigXl) !AceEnc {
        return .{
            .text_encoder = try .init(store.withPrefix("encoder")),
            .lyric_encoder = try .init(allocator, store.withPrefix("encoder"), config),
            .timbre_encoder = try .init(allocator, store.withPrefix("encoder"), config),
            .audiocode_encoder = try .init(allocator, store, config),
        };
    }

    pub fn load(self: *const AceEnc, zml_handler: *main.Zml_handler, store: *zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(AceEnc) {
        var progress = zml_handler.progress.start("Load Enc weights", store.registry.tensors.count());
        defer progress.end();
        return zml.io.load(AceEnc, self, zml_handler.allocator, zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
            .parallelism = 16,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = &progress,
        });
    }

    pub fn deinit(self: *const AceEnc, allocator: std.mem.Allocator) void {
        self.lyric_encoder.deinit(allocator);
        self.timbre_encoder.deinit(allocator);
        self.audiocode_encoder.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AceEnc), allocator: std.mem.Allocator) void {
         TextEncoder.unloadBuffers(&self.text_encoder);
         LyricEncoder.unloadBuffers(&self.lyric_encoder, allocator);
         TimbreEncoder.unloadBuffers(&self.timbre_encoder, allocator);
         AudioCodeEncoder.unloadBuffers(&self.audiocode_encoder, allocator);
    }

    pub fn forward(self: AceEnc, text_emb: zml.Tensor, encoded_lyric: zml.Tensor, encoded_timbre: zml.Tensor, src_latent: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        // dim [s_text, d_emb_cond]
        const encoded_text = self.text_encoder.forward(text_emb);
        
        // dim [s_text + s_lyric + 1, d_emb_cond]
        const encoded_conditions = zml.Tensor.concatenate(&.{ encoded_lyric, encoded_timbre, encoded_text }, .s);
        
        // dim [a, t_25hz]
        const latent_mask = zml.Tensor.constant(zml.DataType.constant(.bf16, 1)).broad(src_latent.shape());
        
        // dim [t_25hz, 2 * a]
        const context_latents = zml.Tensor.concatenate(&.{ src_latent, latent_mask }, .a).transpose(.{ .t, .a });
        
        return .{ context_latents, encoded_conditions };
    }
};


pub const TextEncoder = struct {
    text_projector: zml.nn.Linear,
    
    pub fn init(store: zml.io.TensorStore.View) !TextEncoder {
        return .{
            .text_projector = .init(store.createTensor("text_projector.weight", .{ .d_out, .d }, null), null, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextEncoder)) void {
        self.text_projector.weight.deinit();
    }

    pub fn forward(self: TextEncoder, text_emb: zml.Tensor) zml.Tensor {
        return self.text_projector.forward(text_emb).rename(.{ .d_out = .d });
    }
};


pub const LyricEncoder = struct {
    lyric_projector: zml.nn.Linear,
    lyric_layers: []EncoderLayer,
    lyric_norm: RmsNorm,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: dit.ConfigXl) !LyricEncoder {
        const layers = try allocator.alloc(EncoderLayer, config.num_lyric_encoder_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("lyric_encoder.layers").withLayer(i), config);
        }
        return .{
            .lyric_projector = .init(
                store.createTensor("lyric_encoder.embed_tokens.weight", .{ .d_out, .d }, null),
                store.createTensor("lyric_encoder.embed_tokens.bias", .{ .d_out }, null),
                .d),
            .lyric_layers = layers,
            .lyric_norm = .init(store.withPrefix("lyric_encoder.norm"), config.rms_norm_eps),
        };
    }

    pub fn deinit(self: *const LyricEncoder, allocator: std.mem.Allocator) void {
        allocator.free(self.lyric_layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LyricEncoder), allocator: std.mem.Allocator) void {
        self.lyric_projector.weight.deinit();
        if (self.lyric_projector.bias) |*bias| bias.deinit();
        for (self.lyric_layers) |*layer| {
            EncoderLayer.unloadBuffers(layer);
        }
        allocator.free(self.lyric_layers);
        RmsNorm.unloadBuffers(&self.lyric_norm);
    }

    pub fn forward(self: LyricEncoder, lyric_emb: zml.Tensor) zml.Tensor {
        var lyric_proj = self.lyric_projector.forward(lyric_emb).rename(.{ .d_out = .d });
        for (self.lyric_layers) |layer| {
            // lyrics use full bidirectionnal attention : no masking
            lyric_proj = layer.forward(lyric_proj, null);
        }
        lyric_proj = self.lyric_norm.forward(lyric_proj);
        return lyric_proj;
    }
};

pub const EncoderLayer = struct {
    id: u32,
    input_layernorm: RmsNorm,
    self_attn: SelfAttention,
    post_attention_layernorm: RmsNorm,
    mlp: MlpLayer,

    pub fn init(id: u32, store: zml.io.TensorStore.View, config: dit.ConfigXl) !EncoderLayer {
        return .{
            .id = id,
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
            .mlp = try .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(EncoderLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttention.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        MlpLayer.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: EncoderLayer, x0: zml.Tensor, attn_mask: ?zml.Tensor) zml.Tensor {
        const x0_norm = self.input_layernorm.forward(x0);
        const delta_attn = self.self_attn.forward(x0_norm, attn_mask);
        const x1 = x0.add(delta_attn);
        const x1_norm = self.post_attention_layernorm.forward(x1);
        const delta_mlp = self.mlp.forward(x1_norm);
        return x1.add(delta_mlp);
    }
};

pub const SelfAttention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: dit.ConfigXl) !SelfAttention {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, null), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, null), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, null), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, null), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .num_heads = @intCast(config.encoder_num_attention_heads),
            .num_kv_heads = @intCast(config.encoder_num_key_value_heads),
            .head_dim = @intCast(config.head_dim),
            .rope_opts = .{
                .layout = .sequential,
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttention)) void {
        self.q_proj.weight.deinit();
        self.k_proj.weight.deinit();
        self.v_proj.weight.deinit();
        self.o_proj.weight.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    // full/window bidirectional self attention, no kv caching. the attn_mask is precomputed
    pub fn forward(self: SelfAttention, x: zml.Tensor, attn_mask: ?zml.Tensor) zml.Tensor {
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = self.head_dim });
        
        const pos_index = zml.Tensor.arange(.{ .end = x.dim(.s) }, .u32).withTags(.{ .s });
        
        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });
        
        const attn_heads_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
        const attn_output = attn_heads_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn_output);
        return delta.rename(.{ .d_out = .d });
    }
};

pub const MlpLayer = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) !MlpLayer {
        return .{
            .up_proj = store.createTensor("up_proj.weight", .{ .d_out, .d }, null),
            .gate_proj = store.createTensor("gate_proj.weight", .{ .d_out, .d }, null),
            .down_proj = store.createTensor("down_proj.weight", .{ .d, .d_out }, null),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MlpLayer)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: MlpLayer, input: zml.Tensor) zml.Tensor {
        const up_projection = input.dot(self.up_proj, .d);
        const gate_projection = input.dot(self.gate_proj, .d);
        const activation = gate_projection.silu().mul(up_projection);
        return activation.dot(self.down_proj, .d_out);
    }
};

pub const RmsNorm = struct {
    weights: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weights = store.createTensor("weight", .{ .d_out }, null),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weights.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const normalized = zml.nn.rmsNorm(input, .d, self.eps);
        return normalized.mul(self.weights.withTags(.{ .d }).broad(input.shape()));
    }
};


pub const TimbreEncoder = struct {
    embed_timbre: zml.nn.Linear,
    timbre_layers: []EncoderLayer,
    timbre_norm: RmsNorm,
    special_tokens: zml.Tensor,
    sliding_window: u32,
    
    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: dit.ConfigXl) !TimbreEncoder {
        const layers = try allocator.alloc(EncoderLayer, config.num_timbre_encoder_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("timbre_encoder.layers").withLayer(i), config);
        }
        return .{
            .embed_timbre = .init(
                store.createTensor("timbre_encoder.embed_tokens.weight", .{ .d, .a }, null),
                store.createTensor("timbre_encoder.embed_tokens.bias", .{ .d }, null),
                .a),
            .timbre_layers = layers,
            .timbre_norm = .init(store.withPrefix("timbre_encoder.norm"), config.rms_norm_eps),
            .special_tokens = store.createTensor("timbre_encoder.special_token", .{ .singleton, .singleton, .d }, null),
            .sliding_window = config.sliding_window,
        };
    }

    pub fn deinit(self: *const TimbreEncoder, allocator: std.mem.Allocator) void {
        allocator.free(self.timbre_layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TimbreEncoder), allocator: std.mem.Allocator) void {
        self.embed_timbre.weight.deinit();
        if (self.embed_timbre.bias) |*bias| bias.deinit();
        for (self.timbre_layers) |*layer| {
            EncoderLayer.unloadBuffers(layer);
        }
        allocator.free(self.timbre_layers);
        self.special_tokens.deinit();
        RmsNorm.unloadBuffers(&self.timbre_norm);
    }

    pub fn forward(self: TimbreEncoder, timbre_latent: zml.Tensor) zml.Tensor {
        var timbre_emb = self.embed_timbre.forward(timbre_latent);
        // the special tokens appending is commented in the python reference
        // inputs_embeds = torch.cat([self.special_token.expand(inputs_embeds.shape[0], 1, -1), inputs_embeds], dim=1)
        // the encoder layers assume sequence length to be tagged with s
        timbre_emb = timbre_emb.rename(.{ .t = .s });
        const window_attention_mask = createBidirectionalWindowMask(timbre_emb.dim(.s), self.sliding_window);
        for (self.timbre_layers) |layer| {
            // timbre use alternating window/full bidirectionnal attention
            const actual_mask = if (layer.id % 2 == 0) window_attention_mask else null;
            timbre_emb = layer.forward(timbre_emb, actual_mask);
        }
        timbre_emb = self.timbre_norm.forward(timbre_emb);
        // the timbre embedding is the first token of the sequence (it's a special token)
        timbre_emb = timbre_emb.choose1d(.s, 0).insertAxes(0, .{ .s });
        return timbre_emb;
    }
};


pub const AudioCodeEncoder = struct {
    dequantizer: AudioCodeDequantizer,
    embed_tokens: zml.nn.Linear,
    layers: []EncoderLayer,
    norm: RmsNorm,
    proj_out: zml.nn.Linear,
    special_tokens: zml.Tensor,
    pool_window_size: u32,
    sliding_window: u32,
    
    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: dit.ConfigXl) !AudioCodeEncoder {
        const layers = try allocator.alloc(EncoderLayer, config.num_attention_pooler_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("detokenizer.layers").withLayer(i), config);
        }
        return .{
            .dequantizer = .init(store.withPrefix("tokenizer.quantizer"), config),
            .embed_tokens = .init(
                store.createTensor("detokenizer.embed_tokens.weight", .{ .d_out, .d }, null),
                store.createTensor("detokenizer.embed_tokens.bias", .{ .d_out }, null),
                .d,
            ),
            .layers = layers,
            .norm = .init(store.withPrefix("detokenizer.norm"), config.rms_norm_eps),
            .proj_out = .init(
                store.createTensor("detokenizer.proj_out.weight", .{ .d_out, .d }, null),
                store.createTensor("detokenizer.proj_out.bias", .{ .d_out }, null),
                .d,
            ),
            .special_tokens = store.createTensor("detokenizer.special_tokens", .{ .b, .p, .d }, null),
            .pool_window_size = config.pool_window_size,
            .sliding_window = config.sliding_window,
        };
    }

    pub fn deinit(self: AudioCodeEncoder, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AudioCodeEncoder), allocator: std.mem.Allocator) void {
        AudioCodeDequantizer.unloadBuffers(&self.dequantizer);
        self.embed_tokens.weight.deinit();
        if (self.embed_tokens.bias) |*bias| bias.deinit();
        for (self.layers) |*layer| {
            EncoderLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
        self.proj_out.weight.deinit();
        if (self.proj_out.bias) |*bias| bias.deinit();
        self.special_tokens.deinit();
    }

    pub fn forward(self: AudioCodeEncoder, audio_codes_int: zml.Tensor) zml.Tensor {
        const x = self.dequantizer.dequantize(audio_codes_int);
        // Expected input: [t_code, d]
        var hidden_states = self.embed_tokens.forward(x.withTags(.{ .t_code, .d }));
        hidden_states = hidden_states.rename(.{ .t_code = .t });
        hidden_states = hidden_states.rename(.{ .d_out = .d });
        // expand each token into pool_window_size patches
        // [t, d] -> [t, p = pool_window_size, d]
        hidden_states = hidden_states.insertAxes(1, .{ .p });
        hidden_states = hidden_states.repeat1d(.p, self.pool_window_size);
        // add special tokens
        // TODO: in python these seem to be initialized at random instead of from the tensor file
        // special_tokens = nn.Parameter(torch.randn(1, config.pool_window_size, config.hidden_size) * 0.02)
        const special_tokens = self.special_tokens.squeeze(.b).broad(hidden_states.shape());
        hidden_states = hidden_states.add(special_tokens);
        // encoder layers process tensors of dim [b, s, d]
        hidden_states = hidden_states.rename(.{ .t = .b, .p = .s });
        // audiocodes encoder uses alternating window/full bidirectional attention
        const window_attention_mask = createBidirectionalWindowMask(hidden_states.dim(.s), self.sliding_window);
        for (self.layers) |layer| {
            if (hidden_states.dim(.b) == 0) break;
            const actual_mask = if (layer.id % 2 == 0) window_attention_mask else null;
            hidden_states = layer.forward(hidden_states, actual_mask);
        }
        hidden_states = self.norm.forward(hidden_states);
        // project from encoder hidden dim into input audio channel dimension (d -> a, 2048 -> 64)
        hidden_states = self.proj_out.forward(hidden_states);
        // rename back to [t, p, d], with d now being the audio dimension
        hidden_states = hidden_states.rename(.{ .d_out = .a, .s = .p, .b = .t });
        
        // depatch time dimension from 5hz * p = 5 to 25hz : [t, p, d] -> [t * p, d]
        return hidden_states.merge(.{ .t = .{ .t, .p }});
    }
    
};

pub const AudioCodeDequantizer = struct {
    project_in: zml.nn.Linear, // for quantization
    project_out: zml.nn.Linear, // for dequantization
    fsq_input_levels: []const u32,

    pub fn init(store: zml.io.TensorStore.View, config: dit.ConfigXl) AudioCodeDequantizer {
        return .{
            .project_in = .init(
                store.createTensor("project_in.weight", .{ .d_out, .d }, null),
                store.createTensor("project_in.bias", .{ .d_out }, null),
                .d,
            ),
            .project_out = .init(
                store.createTensor("project_out.weight", .{ .d_out, .d }, null),
                store.createTensor("project_out.bias", .{ .d_out }, null),
                .d,
            ),
            .fsq_input_levels = config.fsq_input_levels,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AudioCodeDequantizer)) void {
        self.project_in.weight.deinit();
        if (self.project_in.bias) |*bias| bias.deinit();
        self.project_out.weight.deinit();
        if (self.project_out.bias) |*bias| bias.deinit();
    }

    pub fn dequantize(self: AudioCodeDequantizer, audio_codes: zml.Tensor) zml.Tensor {
        const code_ids = audio_codes.withTags(.{ .t_code });
        
        // from ResidualFSQ implementation : this is the implicit codebook
        
        const levels = self.fsq_input_levels;
        
        const q0 = code_ids.remainder(zml.Tensor.scalar(levels[0], code_ids.dtype()).broad(code_ids.shape()));
        var rem = code_ids.div(zml.Tensor.scalar(levels[0], code_ids.dtype()).broad(code_ids.shape()));

        const q1 = rem.remainder(zml.Tensor.scalar(levels[1], rem.dtype()).broad(rem.shape()));
        rem = rem.div(zml.Tensor.scalar(levels[1], rem.dtype()).broad(rem.shape()));

        const q2 = rem.remainder(zml.Tensor.scalar(levels[2], rem.dtype()).broad(rem.shape()));
        rem = rem.div(zml.Tensor.scalar(levels[2], rem.dtype()).broad(rem.shape()));

        const q3 = rem.remainder(zml.Tensor.scalar(levels[3], rem.dtype()).broad(rem.shape()));
        rem = rem.div(zml.Tensor.scalar(levels[3], rem.dtype()).broad(rem.shape()));

        const q4 = rem.remainder(zml.Tensor.scalar(levels[4], rem.dtype()).broad(rem.shape()));
        const q5 = rem.div(zml.Tensor.scalar(levels[4], rem.dtype()).broad(rem.shape()));

        const q0f = normalizeQuantLevel(q0, levels[0]);
        const q1f = normalizeQuantLevel(q1, levels[1]);
        const q2f = normalizeQuantLevel(q2, levels[2]);
        const q3f = normalizeQuantLevel(q3, levels[3]);
        const q4f = normalizeQuantLevel(q4, levels[4]);
        const q5f = normalizeQuantLevel(q5, levels[5]);
        
        const fsq_features = zml.Tensor.stack(&.{ q0f, q1f, q2f, q3f, q4f, q5f }, .last, .d);

        // this is the residual scales and sum
        // ResidualFSQ uses: scales[q, d] = levels[d] ** (-q)
        //
        // In the current inference path used by ACE-Step audio-code hints,
        // indices arrive with a single residual quantizer axis (Q = 1), so:
        // - only q = 0 is present
        // - scale = 1 for every FSQ dimension
        // - summation over quantizers is therefore identity
        const summed = fsq_features.convert(.bf16); // NUM
        
        // out projection 
        return self.project_out.forward(summed).rename(.{ .d_out = .d });
    }
    
    fn normalizeQuantLevel(x: zml.Tensor, level: u32) zml.Tensor {
        const xf = x.convert(.f32);
        const denom = @as(f32, @floatFromInt(@max(level - 1, 1)));
        return xf.div(zml.Tensor.scalar(denom, .f32).broad(xf.shape())).scale(2.0).addConstant(-1.0);
    }

};


pub fn createBidirectionalWindowMask(seq_len: i64, window_len: u32) zml.Tensor {
     const attn_shape = zml.Shape.init(.{ .q = seq_len, .k = seq_len }, .bf16);
     const window = zml.DataType.constant(.i32, window_len);
     
    // tokens at pos i attend to tokens at pos [i - w, i + w] ie pos j st. |i - j| <= w
    const q_idx = zml.Tensor.iota(attn_shape, .q);
    const k_idx = zml.Tensor.iota(attn_shape, .k);
    const idx_dist = zml.Tensor.abs(q_idx.sub(k_idx));
    const max_dist = zml.Tensor.constant(window).broad(attn_shape);
    const is_in_window = idx_dist.cmp(.LE, max_dist);

    const zeros = zml.Tensor.zeroes(attn_shape);
    const minf = zml.floats.Float32.toF32(zml.floats.Float32.minus_inf);
    const minus_inf = zml.Tensor.constant(zml.DataType.constant(.bf16, minf)).broad(attn_shape);
     
    return zml.Tensor.select(is_in_window, zeros, minus_inf);
}

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
