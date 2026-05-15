const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");

const dialects = @import("mlir/dialects");


pub const AceDit_handler = struct {
    model: AceDit,
    params: Params,
    config: ConfigXl,
    exes: Exes,
    model_buffers: zml.Bufferized(AceDit),
    shardings: main.Shardings,
    
    pub fn init(zml_handler: *main.Zml_handler, target_duration: u32, conditions_len: i64) !AceDit_handler {
        zml_handler.tic(&zml_handler.timers.dit.init);
        const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.acedit);
        var registry: zml.safetensors.TensorRegistry = try .fromRepo(zml_handler.allocator, zml_handler.io, repo);
        defer registry.deinit();
            
        std.log.info("DiT parse config and safetensors", .{});
        var config: ConfigXl = undefined;
        if (zml_handler.uris.is_xl) {
            const parsed_config = try main.parseConfig(ConfigXl, zml_handler.allocator, zml_handler.io, repo);
            defer parsed_config.deinit();
            config = try parsed_config.value.dupe(zml_handler.allocator);
        } else {
            const parsed_config = try main.parseConfig(ConfigBase, zml_handler.allocator, zml_handler.io, repo);
            defer parsed_config.deinit();
            const xl_config = try ConfigXl.fromBase(parsed_config.value);
            config = try xl_config.dupe(zml_handler.allocator);
        }
        std.log.info("DiT parsed", .{});

        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();
        std.log.info("DiT init model", .{});
        const model: AceDit = try .init(zml_handler.allocator, store.view().withPrefix("decoder"), config);
        std.log.info("DiT initialized", .{});
        
        const params: Params = .{
            .t_curr = .init(.{}, .f32),
            .t_next = .init(.{}, .f32),
            .x = .init(.{ .t = 25 * target_duration, .a = config.timbre_hidden_dim }, .bf16),
            .context_latents = .init(.{ .t = 25 * target_duration, .a = 2 * config.timbre_hidden_dim }, .bf16),
            .y = .init(.{ .s = conditions_len, .d = config.encoder_hidden_size }, .bf16),
            .y_proj = .init(.{ .s = conditions_len, .d = config.hidden_size }, .bf16),
            .mask = .init(.{ .q = @divFloor(25 * target_duration + 1, 2), .k = @divFloor(25 * target_duration + 1, 2) }, .bf16),
            .hidden_states = .init(.{ .t = @divFloor(25 * target_duration + 1, 2), .d = config.hidden_size }, .bf16),
            .temb = .init(.{ .d_emb = config.hidden_size }, .bf16),
            .timestep_proj = .init(.{ .n2 = config.fsq_input_levels.len, .d_emb = config.hidden_size }, .bf16),
        };
        
        const shardings: main.Shardings = try .init(zml_handler.platform);

        zml_handler.toc(&zml_handler.timers.dit.init);
        zml_handler.tic(&zml_handler.timers.dit.compile);
        
        std.log.info("DiT compile model", .{});
        const exes = try compileModel(zml_handler, model, params, shardings);
        std.log.info("DiT compiled", .{});

        zml_handler.toc(&zml_handler.timers.dit.compile);
        zml_handler.tic(&zml_handler.timers.dit.load);
        
        std.log.info("DiT load buffers", .{});
        const model_buffers = try model.load(zml_handler, &store, &shardings.all());
        std.log.info("DiT loaded", .{});

        zml_handler.toc(&zml_handler.timers.dit.load);
        
        return .{
            .model = model,
            .params = params,
            .config = config,
            .exes = exes,
            .model_buffers = model_buffers,
            .shardings = shardings,
        };
    }

    pub fn compileModel(zml_handler: *main.Zml_handler, model: AceDit, params: Params, shardings: main.Shardings) !Exes {
        const shardings_arr = shardings.all();
        const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };

        var pre_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceDit, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .preprocess, .{
                        params_.t_curr, params_.x, params_.context_latents, params_.y }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var pre_future_awaited = false;
        errdefer if (!pre_future_awaited) if (pre_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var layer_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: DiTLayer, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .forward, .{
                        params_.hidden_states, params_.y_proj, params_.timestep_proj, params_.mask }, opts_);
            }
        }.call, .{ zml_handler, model.layers[0], params, opts });
        var layer_future_awaited = false;
        errdefer if (!layer_future_awaited) if (layer_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        var post_future = try zml_handler.io.concurrent(struct {
            fn call(zml_handler_: *main.Zml_handler, model_: AceDit, params_: Params, opts_: zml.module.CompilationOptions) !zml.Exe {
                return zml_handler_.platform.compile(zml_handler_.allocator, zml_handler_.io, model_, .postprocess,
                    .{ params_.t_curr, params_.t_next, params_.x, params_.hidden_states, params_.temb }, opts_);
            }
        }.call, .{ zml_handler, model, params, opts });
        var post_future_awaited = false;
        errdefer if (!post_future_awaited) if (post_future.cancel(zml_handler.io)) |v| v.deinit() else |_| {};

        const pre_future_exe = try pre_future.await(zml_handler.io);
        pre_future_awaited = true;

        const layer_future_exe = try layer_future.await(zml_handler.io);
        layer_future_awaited = true;

        const post_future_exe = try post_future.await(zml_handler.io);
        post_future_awaited = true;

        return .{
            .preprocess_exe = pre_future_exe,
            .preprocess_args = try pre_future_exe.args(zml_handler.allocator),
            .preprocess_results = try pre_future_exe.results(zml_handler.allocator),
            .layer_exe = layer_future_exe,
            .layer_args = try layer_future_exe.args(zml_handler.allocator),
            .layer_results = try layer_future_exe.results(zml_handler.allocator),
            .postprocess_exe = post_future_exe,
            .postprocess_args = try post_future_exe.args(zml_handler.allocator),
            .postprocess_results = try post_future_exe.results(zml_handler.allocator),
        };
    }
    
    pub fn unloadBuffers(self: *AceDit_handler, allocator: std.mem.Allocator) void {
        AceDit.unloadBuffers(&self.model_buffers, allocator);
    }

    pub fn deinit(self: *AceDit_handler, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.exes.deinit(allocator);
    }
    
};


pub const Params = struct {
    // current timestamp, scalar in [0, 1]
    t_curr: zml.Tensor,
    // next timestamp, scalar in [0, 1]. is used to compute the time step when iterating the noised latent
    t_next: zml.Tensor,
    // noisy audio channel latent that will be denoised, dim [t_25hz, a]
    x: zml.Tensor,
    // context latents : one audio channel for source latent, one for latent mask, dim [2 * a, t_25hz]
    context_latents: zml.Tensor,
    // encoder conditions : information about text, lyric and reference timbre
    // lives in another space, is attended by the cross attention. dim = [s, d_enc]
    y: zml.Tensor,

    // intermediate arguments
    y_proj: zml.Tensor,
    mask: zml.Tensor,
    hidden_states: zml.Tensor,
    temb: zml.Tensor,
    timestep_proj: zml.Tensor,
};

pub const ConfigBase = struct {
    attention_bias: bool,
    attention_dropout: f32,
    audio_acoustic_hidden_dim: u32,
    data_proportion: f32,
    dtype: []const u8,
    fsq_dim: u32,
    fsq_input_levels: []const u32,
    fsq_input_num_quantizers: u32,
    head_dim: u32,
    hidden_act: []u8,
    hidden_size: u32,
    in_channels: u32,
    initializer_range: f32,
    intermediate_size: u32,
    layer_types: []const DecoderLayerType,
    max_position_embeddings: u32,
    num_attention_heads: u32,
    num_attention_pooler_hidden_layers: u32,
    num_audio_decoder_hidden_layers: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    num_lyric_encoder_hidden_layers: u32,
    num_timbre_encoder_hidden_layers: u32,
    patch_size: u32,
    pool_window_size: u32,
    rms_norm_eps: f32,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    rope_theta: f32,
    sliding_window: u32,
    text_hidden_dim: u32,
    timbre_fix_frame: u32,
    timbre_hidden_dim: u32,
    timestep_mu: f32,
    timestep_sigma: f32,
    use_cache: bool,
    use_sliding_window: bool,
    vocab_size: u32,

    pub fn dupe(self: ConfigBase, allocator: std.mem.Allocator) !ConfigBase {
        return .{
            .attention_bias = self.attention_bias,
            .attention_dropout = self.attention_dropout,
            .audio_acoustic_hidden_dim = self.audio_acoustic_hidden_dim,
            .data_proportion = self.data_proportion,
            .dtype = try allocator.dupe(u8, self.dtype),
            .fsq_dim = self.fsq_dim,
            .fsq_input_levels = try allocator.dupe(u32, self.fsq_input_levels),
            .fsq_input_num_quantizers = self.fsq_input_num_quantizers,
            .head_dim = self.head_dim,
            .hidden_act = try allocator.dupe(u8, self.hidden_act),
            .hidden_size = self.hidden_size,
            .in_channels = self.in_channels,
            .initializer_range = self.initializer_range,
            .intermediate_size = self.intermediate_size,
            .layer_types = try allocator.dupe(DecoderLayerType, self.layer_types),
            .max_position_embeddings = self.max_position_embeddings,
            .num_attention_heads = self.num_attention_heads,
            .num_attention_pooler_hidden_layers = self.num_attention_pooler_hidden_layers,
            .num_audio_decoder_hidden_layers = self.num_audio_decoder_hidden_layers,
            .num_hidden_layers = self.num_hidden_layers,
            .num_key_value_heads = self.num_key_value_heads,
            .num_lyric_encoder_hidden_layers = self.num_lyric_encoder_hidden_layers,
            .num_timbre_encoder_hidden_layers = self.num_timbre_encoder_hidden_layers,
            .patch_size = self.patch_size,
            .pool_window_size = self.pool_window_size,
            .rms_norm_eps = self.rms_norm_eps,
            .rope_scaling = self.rope_scaling,
            .rope_theta = self.rope_theta,
            .sliding_window = self.sliding_window,
            .text_hidden_dim = self.text_hidden_dim,
            .timbre_fix_frame = self.timbre_fix_frame,
            .timbre_hidden_dim = self.timbre_hidden_dim,
            .timestep_mu = self.timestep_mu,
            .timestep_sigma = self.timestep_sigma,
            .use_cache = self.use_cache,
            .use_sliding_window = self.use_sliding_window,
            .vocab_size = self.vocab_size,
        };
    }

    pub fn deinit(self: ConfigBase, allocator: std.mem.Allocator) void {
        allocator.free(self.dtype);
        allocator.free(self.fsq_input_levels);
        allocator.free(self.hidden_act);
        allocator.free(self.layer_types);
    }
};

pub const ConfigXl = struct {
    attention_bias: bool,
    attention_dropout: f32,
    audio_acoustic_hidden_dim: u32,
    data_proportion: f32,
    dtype: []const u8,
    encoder_hidden_size: u32,
    encoder_intermediate_size: u32,
    encoder_num_attention_heads: u32,
    encoder_num_key_value_heads: u32,
    fsq_dim: u32,
    fsq_input_levels: []const u32,
    fsq_input_num_quantizers: u32,
    head_dim: u32,
    hidden_act: []u8,
    hidden_size: u32,
    in_channels: u32,
    initializer_range: f32,
    intermediate_size: u32,
    layer_types: []const DecoderLayerType,
    max_position_embeddings: u32,
    num_attention_heads: u32,
    num_attention_pooler_hidden_layers: u32,
    num_audio_decoder_hidden_layers: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    num_lyric_encoder_hidden_layers: u32,
    num_timbre_encoder_hidden_layers: u32,
    patch_size: u32,
    pool_window_size: u32,
    rms_norm_eps: f32,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    rope_theta: f32,
    sliding_window: u32,
    text_hidden_dim: u32,
    timbre_fix_frame: u32,
    timbre_hidden_dim: u32,
    timestep_mu: f32,
    timestep_sigma: f32,
    use_cache: bool,
    use_sliding_window: bool,
    vocab_size: u32,

    pub fn fromBase(base: ConfigBase) !ConfigXl {
        return .{
            .attention_bias = base.attention_bias,
            .attention_dropout = base.attention_dropout,
            .audio_acoustic_hidden_dim = base.audio_acoustic_hidden_dim,
            .data_proportion = base.data_proportion,
            .dtype = base.dtype,
            .encoder_hidden_size = base.hidden_size,
            .encoder_intermediate_size = base.intermediate_size,
            .encoder_num_attention_heads = base.num_attention_heads,
            .encoder_num_key_value_heads = base.num_key_value_heads,
            .fsq_dim = base.fsq_dim,
            .fsq_input_levels = base.fsq_input_levels,
            .fsq_input_num_quantizers = base.fsq_input_num_quantizers,
            .head_dim = base.head_dim,
            .hidden_act = base.hidden_act,
            .hidden_size = base.hidden_size,
            .in_channels = base.in_channels,
            .initializer_range = base.initializer_range,
            .intermediate_size = base.intermediate_size,
            .layer_types = base.layer_types,
            .max_position_embeddings = base.max_position_embeddings,
            .num_attention_heads = base.num_attention_heads,
            .num_attention_pooler_hidden_layers = base.num_attention_pooler_hidden_layers,
            .num_audio_decoder_hidden_layers = base.num_audio_decoder_hidden_layers,
            .num_hidden_layers = base.num_hidden_layers,
            .num_key_value_heads = base.num_key_value_heads,
            .num_lyric_encoder_hidden_layers = base.num_lyric_encoder_hidden_layers,
            .num_timbre_encoder_hidden_layers = base.num_timbre_encoder_hidden_layers,
            .patch_size = base.patch_size,
            .pool_window_size = base.pool_window_size,
            .rms_norm_eps = base.rms_norm_eps,
            .rope_scaling = base.rope_scaling,
            .rope_theta = base.rope_theta,
            .sliding_window = base.sliding_window,
            .text_hidden_dim = base.text_hidden_dim,
            .timbre_fix_frame = base.timbre_fix_frame,
            .timbre_hidden_dim = base.timbre_hidden_dim,
            .timestep_mu = base.timestep_mu,
            .timestep_sigma = base.timestep_sigma,
            .use_cache = base.use_cache,
            .use_sliding_window = base.use_sliding_window,
            .vocab_size = base.vocab_size,
        };
    }

    pub fn dupe(self: ConfigXl, allocator: std.mem.Allocator) !ConfigXl {
        return .{
            .attention_bias = self.attention_bias,
            .attention_dropout = self.attention_dropout,
            .audio_acoustic_hidden_dim = self.audio_acoustic_hidden_dim,
            .data_proportion = self.data_proportion,
            .dtype = try allocator.dupe(u8, self.dtype),
            .encoder_hidden_size = self.encoder_hidden_size,
            .encoder_intermediate_size = self.encoder_intermediate_size,
            .encoder_num_attention_heads = self.encoder_num_attention_heads,
            .encoder_num_key_value_heads = self.encoder_num_key_value_heads,
            .fsq_dim = self.fsq_dim,
            .fsq_input_levels = try allocator.dupe(u32, self.fsq_input_levels),
            .fsq_input_num_quantizers = self.fsq_input_num_quantizers,
            .head_dim = self.head_dim,
            .hidden_act = try allocator.dupe(u8, self.hidden_act),
            .hidden_size = self.hidden_size,
            .in_channels = self.in_channels,
            .initializer_range = self.initializer_range,
            .intermediate_size = self.intermediate_size,
            .layer_types = try allocator.dupe(DecoderLayerType, self.layer_types),
            .max_position_embeddings = self.max_position_embeddings,
            .num_attention_heads = self.num_attention_heads,
            .num_attention_pooler_hidden_layers = self.num_attention_pooler_hidden_layers,
            .num_audio_decoder_hidden_layers = self.num_audio_decoder_hidden_layers,
            .num_hidden_layers = self.num_hidden_layers,
            .num_key_value_heads = self.num_key_value_heads,
            .num_lyric_encoder_hidden_layers = self.num_lyric_encoder_hidden_layers,
            .num_timbre_encoder_hidden_layers = self.num_timbre_encoder_hidden_layers,
            .patch_size = self.patch_size,
            .pool_window_size = self.pool_window_size,
            .rms_norm_eps = self.rms_norm_eps,
            .rope_scaling = self.rope_scaling,
            .rope_theta = self.rope_theta,
            .sliding_window = self.sliding_window,
            .text_hidden_dim = self.text_hidden_dim,
            .timbre_fix_frame = self.timbre_fix_frame,
            .timbre_hidden_dim = self.timbre_hidden_dim,
            .timestep_mu = self.timestep_mu,
            .timestep_sigma = self.timestep_sigma,
            .use_cache = self.use_cache,
            .use_sliding_window = self.use_sliding_window,
            .vocab_size = self.vocab_size,
        };
    }

    pub fn deinit(self: ConfigXl, allocator: std.mem.Allocator) void {
        allocator.free(self.dtype);
        allocator.free(self.fsq_input_levels);
        allocator.free(self.hidden_act);
        allocator.free(self.layer_types);
    }
};

pub const DecoderLayerType = enum {
    full_attention,
    sliding_attention,
};

pub const Exes = struct {
    preprocess_exe: zml.Exe,
    preprocess_args: zml.Exe.Arguments,
    preprocess_results: zml.Exe.Results,
    
    layer_exe: zml.Exe,
    layer_args: zml.Exe.Arguments,
    layer_results: zml.Exe.Results,
    
    postprocess_exe: zml.Exe,
    postprocess_args: zml.Exe.Arguments,
    postprocess_results: zml.Exe.Results,

    pub fn deinit(self: Exes, allocator: std.mem.Allocator) void {
        self.preprocess_exe.deinit();
        self.preprocess_args.deinit(allocator);
        self.preprocess_results.deinit(allocator);
        self.layer_exe.deinit();
        self.layer_args.deinit(allocator);
        self.layer_results.deinit(allocator);
        self.postprocess_exe.deinit();
        self.postprocess_args.deinit(allocator);
        self.postprocess_results.deinit(allocator);
    }
};


pub const AceDit = struct {
    condition_embedder: zml.nn.Linear,
    proj_in: PatchIn,
    proj_out: PatchOut,
    time_embed: TimestepEmbedding,
    time_embed_r: TimestepEmbedding,
    layers: []DiTLayer,
    norm_out: RmsNorm,
    scale_shift_table: zml.Tensor,
    config: ConfigXl,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: ConfigXl) !AceDit {
        const layers = try allocator.alloc(DiTLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .condition_embedder = .init(
                store.createTensor("condition_embedder.weight", .{ .d_out, .d }, null),
                store.createTensor("condition_embedder.bias", .{ .d_out }, null),
                .d,
            ),
            .proj_in = .init(store.withPrefix("proj_in"), config),
            .proj_out = .init(store.withPrefix("proj_out"), config),
            .time_embed = .init(store.withPrefix("time_embed"), config),
            .time_embed_r = .init(store.withPrefix("time_embed_r"), config),
            .layers = layers,
            .norm_out = .init(store.withPrefix("norm_out"), config.rms_norm_eps),
            .scale_shift_table = store.createTensor("scale_shift_table", .{ .n1, .n2, .d }, null),
            .config = config,
        };
    }
    
    pub fn deinit(self: AceDit, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn load(self: *const AceDit, zml_handler: *main.Zml_handler, store: *zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(AceDit) {
        var progress = zml_handler.progress.start("Load DiT weights", store.registry.tensors.count());
        defer progress.end();
        return zml.io.load(AceDit, self, zml_handler.allocator, zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
            .parallelism = 16,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = &progress,
        });
    }
    
    pub fn unloadBuffers(self: *zml.Bufferized(AceDit), allocator: std.mem.Allocator) void {
        self.condition_embedder.weight.deinit();
        if (self.condition_embedder.bias) |*bias| bias.deinit();
        PatchIn.unloadBuffers(&self.proj_in);
        PatchOut.unloadBuffers(&self.proj_out);
        TimestepEmbedding.unloadBuffers(&self.time_embed);
        TimestepEmbedding.unloadBuffers(&self.time_embed_r);
        for (self.layers) |*layer| {
            DiTLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm_out);
        self.scale_shift_table.deinit();
    }

    pub fn preprocess(self: AceDit, t_curr: zml.Tensor, x: zml.Tensor, context_latents: zml.Tensor, y_enc: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        var y = self.condition_embedder.forward(y_enc.withTags(.{ .s, .d }));
        y = y.rename(.{ .d_out = .d });
        
        // embed timesteps
        const temb_t, const timestep_proj_t = self.time_embed.forward(t_curr);
        const temb_r, const timestep_proj_r = self.time_embed_r.forward(t_curr.sub(t_curr));
        const temb = temb_t.add(temb_r);
        const timestep_proj = timestep_proj_t.add(timestep_proj_r);
        
        var hidden_states = self.mergeAndPadLatents(x, context_latents);

        // embed latents into decoder hidden dim
        // proj_in projects three concatenated audio channels into the embedding space
        // time is patched with size 2 so patchIn does [t, 3 * a] -> [t / 2, d_emb]
        hidden_states = self.proj_in.forward(hidden_states);

        return .{ y, hidden_states, temb, timestep_proj };
    }

    pub fn mergeAndPadLatents(self: AceDit, x: zml.Tensor, context_latents: zml.Tensor) zml.Tensor {
        // merge the three input audio channels : context_latents = [src_audio, mask] with x = [target_audio]
        var hidden_states = zml.Tensor.concatenate(&.{ context_latents, x }, .a);
        // pad hidden_states so that time dim is a multiple of patch_size
        const d_time: u32 = @intCast(hidden_states.dim(.t));
        if (d_time % self.config.patch_size != 0) {
            const d_pad: u32 = self.config.patch_size - (d_time % self.config.patch_size);
            // [.a, .t] -> [.a, .t + d_pad]
            const pad_shape = zml.Shape.init(.{ d_pad, hidden_states.dim(.a) }, hidden_states.dtype());
            hidden_states = zml.Tensor.concatenate(&.{ hidden_states, zml.Tensor.zeroes(pad_shape) }, .t);
        }
        return hidden_states;
    }

    pub fn postprocess(self: AceDit, t_curr: zml.Tensor, t_next: zml.Tensor, x: zml.Tensor, hidden_states_: zml.Tensor, temb: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        // adaptive layer norm based on scale shift
        var hidden_states = self.scale_shift(hidden_states_, temb);
        
        // project back latent embeddings into original latent space
        // proj_out projects from the embedding space onto a single audio channel
        // also depatches the time : [t / 2, d_emb] -> [t, a]
        hidden_states = self.proj_out.forward(hidden_states);
           
        // remove padding from output : this is now the predicted flow velocity v
        const v = hidden_states.slice1d(.t, .{ .start = 0, .end = x.dim(.t) });
        
        // update noised latent x with one step of flow matching
        const dt = t_next.sub(t_curr).broad(x.shape()).convert(.bf16);
        const x_next = x.add(v.mul(dt));
     
        return .{ x_next.reuseBuffer(x), x_next.withTags(.{ .t, .a }).transpose(.{ .a, .t }) };
    }
    
    pub fn scale_shift(self: AceDit, hidden_states: zml.Tensor, temb: zml.Tensor) zml.Tensor {
        // scale shift parameters for adaptive output normalization
        const out_mod = self.scale_shift_table.squeeze(.n1).add(temb.insertAxes(0, .{ .n2 }).broad(self.scale_shift_table.squeeze(.n1).shape()));
        const shift = out_mod.choose1d(.n2, 0);
        const scale = out_mod.choose1d(.n2, 1);

        // apply adaptive layer norm: norm(x) * (1 + scale) + shift
        var res = hidden_states;
        res = self.norm_out.forward(res);
        res = res.mul(scale.addConstant(1.0).broad(res.shape()));
        res = res.add(shift.broad(res.shape()));
        
        return res;
    }
    
};


pub const TimestepEmbedding = struct {
    time_proj_weight: zml.Tensor,
    time_proj_bias: zml.Tensor,
    linear_1: zml.nn.Linear,
    linear_2: zml.nn.Linear,
    config: ConfigXl,

    pub fn init(store: zml.io.TensorStore.View, config: ConfigXl) TimestepEmbedding {
        return .{
            .time_proj_weight = store.createTensor("time_proj.weight", .{ .d_emb_6, .d_emb }, null),
            .time_proj_bias = store.createTensor("time_proj.bias", .{ .d_emb_6 }, null),
            .linear_1 = .init(
                store.createTensor("linear_1.weight", .{ .d_emb, .d256 }, null),
                store.createTensor("linear_1.bias", .{ .d_emb }, null),
                .d256,
            ),
            .linear_2 = .init(
                store.createTensor("linear_2.weight", .{ .d_emb_out, .d_emb }, null),
                store.createTensor("linear_2.bias", .{ .d_emb_out }, null),
                .d_emb,
            ),
            .config = config,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TimestepEmbedding)) void {
        self.time_proj_weight.deinit();
        self.time_proj_bias.deinit();
        self.linear_1.weight.deinit();
        if (self.linear_1.bias) |*bias| bias.deinit();
        self.linear_2.weight.deinit();
        if (self.linear_2.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: TimestepEmbedding, timestep: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const t256 = timeEmbedding(timestep).convert(.bf16);
        var temb = self.linear_1.forward(t256);
        temb = temb.silu();
        temb = self.linear_2.forward(temb).rename(.{ .d_emb_out = .d_emb });
        var timestep_proj = temb.silu().dot(self.time_proj_weight, .d_emb);
        timestep_proj = timestep_proj.add(self.time_proj_bias.broad(timestep_proj.shape()));
        timestep_proj = timestep_proj.splitAxis(.d_emb_6, .{ .n2 = 6, .d_emb = self.config.hidden_size });
        return .{ temb, timestep_proj };
    }
    
    fn timeEmbedding(timestep: zml.Tensor) zml.Tensor {
        // NUM : this step if performed in f32 precision, then converted to bf16
        // timestep is a scalar tensor containing the value of the current timestep in [0, 1]
        const d: u32 = 128;
        const s: f32 = - 0.07195578415; // -ln(10000) / d
        const idx = zml.Tensor.arange(.{ .end = d }, .f32).withTags(.{ .d128 });
        const scale = zml.Tensor.scalar(s, .f32);
        const freqs = idx.mul(scale.broad(idx.shape())).exp(); // [exp(-s*i)] i = 0..d128-1
        const t = timestep.mul(zml.Tensor.scalar(1000.0, .f32)).appendAxes(.{ .d128 });
        const args = freqs.mul(t.broad(freqs.shape()));
        const cos = args.cos();
        const sin = args.sin();
        return zml.Tensor.concatenate(&.{ cos, sin }, .d128).rename(.{ .d128 = .d256 });
    }
    
};

pub const DiTLayer = struct {
    id: u32,
    self_attn_norm: RmsNorm,
    self_attn: SelfAttention,
    cross_attn_norm: RmsNorm,
    cross_attn: CrossAttention,
    mlp_norm: RmsNorm,
    mlp: MlpLayer,
    scale_shift_table: zml.Tensor,

    pub fn init(id: u32, store: zml.io.TensorStore.View, config: ConfigXl) !DiTLayer {
        return .{
            .id = id,
            .self_attn_norm = .init(store.withPrefix("self_attn_norm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
            .cross_attn_norm = .init(store.withPrefix("cross_attn_norm"), config.rms_norm_eps),
            .cross_attn = try .init(store.withPrefix("cross_attn"), config),
            .mlp_norm = .init(store.withPrefix("mlp_norm"), config.rms_norm_eps),
            .mlp = try .init(store.withPrefix("mlp")),
            .scale_shift_table = store.createTensor("scale_shift_table", .{ .n1, .n2, .d }, null),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DiTLayer)) void {
        RmsNorm.unloadBuffers(&self.self_attn_norm);
        SelfAttention.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.cross_attn_norm);
        CrossAttention.unloadBuffers(&self.cross_attn);
        RmsNorm.unloadBuffers(&self.mlp_norm);
        MlpLayer.unloadBuffers(&self.mlp);
        self.scale_shift_table.deinit();
    }
    
    pub fn forward(self: DiTLayer, x_: zml.Tensor, y: zml.Tensor, time_emb: zml.Tensor, attn_mask: zml.Tensor) zml.Tensor {
        const x = x_.rename(.{ .t = .s });
        const x_norm = self.scaleShift(x, time_emb);
        
        const delta_self = self.self_attn.forward(x_norm, attn_mask);
        
        // gated residual connection: x = x + attn_output * gate
        const layer_mod = self.scale_shift_table.squeeze(.n1).add(time_emb.rename(.{ .d_emb = .d }));
        const gate_msa = layer_mod.choose1d(.n2, 2);
        // gated residual connection: x = x + attn_output * gate
        const x1 = x.add(delta_self.mul(gate_msa.broad(delta_self.shape())));

        // step 2: cross-attention (always full)
        const x1_norm = self.cross_attn_norm.forward(x1);
        const delta_cross = self.cross_attn.forward(x1_norm, y);
        const x2 = x1.add(delta_cross);

        // step 3: mlp with adaptive layer norm
        return self.mlpAln(x2, time_emb).rename(.{ .s = .t });
    }
    
    pub fn scaleShift(self: DiTLayer, x: zml.Tensor, time_emb: zml.Tensor) zml.Tensor {
        // extract scale-shift parameters for adaptive layer norm from timestep embeddings
        const layer_mod = self.scale_shift_table.squeeze(.n1).add(time_emb.rename(.{ .d_emb = .d }));
        const shift_msa = layer_mod.choose1d(.n2, 0);
        const scale_msa = layer_mod.choose1d(.n2, 1);
        
        // step 1: self-attention with adaptive layer norm (AdaLN)
        // adaptive normalization: norm(x) * (1 + scale) + shift
        var x_norm = self.self_attn_norm.forward(x);
        x_norm = x_norm.mul(scale_msa.addConstant(1.0).broad(x_norm.shape()));
        x_norm = x_norm.add(shift_msa.broad(x_norm.shape()));
        
        return x_norm.reuseBuffer(x);
    }
  
    pub fn mlpAln(self: DiTLayer, x: zml.Tensor, time_emb: zml.Tensor) zml.Tensor {
        const layer_mod = self.scale_shift_table.squeeze(.n1).add(time_emb.rename(.{ .d_emb = .d }));
        const c_gate_msa = layer_mod.choose1d(.n2, 5);
        const c_scale_msa = layer_mod.choose1d(.n2, 4);
        const c_shift_msa = layer_mod.choose1d(.n2, 3);
        var x_mlp_norm = self.mlp_norm.forward(x);
        x_mlp_norm = x_mlp_norm.mul(c_scale_msa.addConstant(1.0).broad(x_mlp_norm.shape()));
        x_mlp_norm = x_mlp_norm.add(c_shift_msa.broad(x_mlp_norm.shape()));
        const mlp_delta = self.mlp.forward(x_mlp_norm);
        return x.add(mlp_delta.mul(c_gate_msa.broad(mlp_delta.shape())));
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
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: ConfigXl) !SelfAttention {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, null), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, null), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, null), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, null), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
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

    // full bidirectional self attention, no kv caching
    pub fn forward(self: SelfAttention, x: zml.Tensor, attn_mask: zml.Tensor) zml.Tensor {
        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });
        
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

pub const CrossAttention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: u32,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: ConfigXl) !CrossAttention {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, null), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, null), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, null), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, null), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .head_dim = config.head_dim,
            .rope_opts = .{
                .layout = .sequential,
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(CrossAttention)) void {
        self.q_proj.weight.deinit();
        self.k_proj.weight.deinit();
        self.v_proj.weight.deinit();
        self.o_proj.weight.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    // cross attention :
    // - queries on the x space (hidden_states in embedded latent space)
    // - keys/values on the y space (encoder_hidden_states in embedded conditions space)
    // - cross attention doesn't use rope
    pub fn forward(self: CrossAttention, x: zml.Tensor, y: zml.Tensor) zml.Tensor {
        var k = self.k_proj.forward(y).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var v = self.v_proj.forward(y).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = self.head_dim });
        
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });
        q = q.rename(.{ .s = .q });
        
        const attn_heads_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = null, .allow_cudnn = true });
        const attn_output = attn_heads_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });

        const delta = self.o_proj.forward(attn_output);
        return delta.rename(.{ .d_out = .d });
    }

};

pub const PatchIn = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    config: ConfigXl,

    pub fn init(store: zml.io.TensorStore.View, config: ConfigXl) PatchIn {
        return .{
            .weight = store.createTensor("1.weight", .{ .d, .c_in, .patch }, null),
            .bias = store.createTensor("1.bias", .{ .d }, null),
            .config = config,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(PatchIn)) void {
        self.weight.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: PatchIn, x: zml.Tensor) zml.Tensor {
        var x_bct = x.insertAxes(0, .{ .b }).withTags(.{ .b, .t, .d });
        x_bct = x_bct.transpose(.{ .b, .d, .t });
        var out = x_bct.conv1d(
            self.weight,
            .{ .window_strides = self.config.patch_size },
        );
        out = out.add(self.bias.broad(out.shape()));
        return out.transpose(.{ .b, .t, .d }).squeeze(.b);
    }
};

pub const PatchOut = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    config: ConfigXl,

    pub fn init(store: zml.io.TensorStore.View, config: ConfigXl) PatchOut {
        return .{
            .weight = store.createTensor("1.weight", .{ .d_in, .c_out, .patch }, null),
            .bias = store.createTensor("1.bias", .{ .d }, null),
            .config = config,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(PatchOut)) void {
        self.weight.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: PatchOut, x: zml.Tensor) zml.Tensor {
        var x_bct = x.insertAxes(0, .{ .b }).withTags(.{ .b, .t, .d });
        x_bct = x_bct.transpose(.{ .b, .d, .t });
        const patch = self.config.patch_size;
        // emulate conv transpose by inserting zeros in the patched dimension and calling regular conv
        const pad = patch - 1;
        const padding_b: zml.Tensor.Pad = .{};
        const padding_d: zml.Tensor.Pad = .{};
        const padding_t: zml.Tensor.Pad = .{ .low = pad, .interior = pad, .high = pad };
        const paddings = .{
            .b = padding_b,
            .d = padding_d,
            .t = padding_t,
        };
        const expanded = x_bct.pad(0, paddings);
        // flip weights instead of using window_reversal = true because using
        // window_reversal = true is very slow
        const flipped_weight = self.weight.reverse(.{ .patch });
        // transpose convolution : reverse kernel output dimensions
        var out = expanded.conv1d(flipped_weight, .{  .kernel_output_feature_dimension = 1, .kernel_input_feature_dimension = 0 });
        out = out.squeeze(.b).transpose(.{ .t, .d });
        out = out.add(self.bias.broad(out.shape()));
        return out;
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