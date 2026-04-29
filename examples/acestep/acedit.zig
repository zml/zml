const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");

const dialects = @import("mlir/dialects");

const hz_type = .f32;


pub const AceDit_handler = struct {
    model: AceDit,
    params: Params,
    config: Config,
    diffuse_exe: zml.Exe,
    model_buffers: zml.Bufferized(AceDit),
    shardings: main.Shardings,
    
    pub fn initFromFile(zml_handler: main.Zml_handler, target_duration: u32, conditions_len: i64) !AceDit_handler {
        const model_path = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-turbo//model.safetensors";
        const config_path = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-turbo//config.json";
        var registry: zml.safetensors.TensorRegistry = try .fromPath(zml_handler.allocator, zml_handler.io, model_path);
        defer registry.deinit();
        
        //try main.printSafetensors(zml_handler.allocator, zml_handler.io, model_path);
    
        std.log.info("DiT parse config and safetensors", .{});
        const parsed_config = try parseConfig(zml_handler, config_path);
        defer parsed_config.deinit();
        const config = try parsed_config.value.dupe(zml_handler.allocator);
        std.log.info("DiT parsed", .{});

        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();
        std.log.info("DiT init model", .{});
        const model: AceDit = try .init(zml_handler.allocator, store.view().withPrefix("decoder"), config);
        std.log.info("DiT initialized", .{});
        
        const params: Params = .{
            .t_curr = .init(.{}, .f32),
            .t_next = .init(.{}, .f32),
            .x = .init(.{ .t = 25 * target_duration, .a = config.timbre_hidden_dim }, hz_type),
            .context_latents = .init(.{ .t = 25 * target_duration, .a = 2 * config.timbre_hidden_dim }, hz_type),
            .y = .init(.{ .s = conditions_len, .d = config.hidden_size }, hz_type),
        };
        
        const shardings: main.Shardings = try .init(zml_handler.platform);
        
        std.log.info("DiT compile model", .{});
        const diffuse_exe = try compileModel(zml_handler, model, params, shardings);
        std.log.info("DiT compiled", .{});
        
        std.log.info("DiT load buffers", .{});
        const model_buffers = try model.load(zml_handler, &store, &shardings.all());
        std.log.info("DiT loaded", .{});
        
        return .{
            .model = model,
            .params = params,
            .config = config,
            .diffuse_exe = diffuse_exe,
            .model_buffers = model_buffers,
            .shardings = shardings,
        };
    }
    
    pub fn unloadBuffers(self: *AceDit_handler) void {
        AceDit.unloadBuffers(&self.model_buffers);
    }

    pub fn deinit(self: *AceDit_handler, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.config.deinit(allocator);
        self.diffuse_exe.deinit();
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
};

pub const Config = struct {
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

    pub fn dupe(self: Config, allocator: std.mem.Allocator) !Config {
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

    pub fn deinit(self: Config, allocator: std.mem.Allocator) void {
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

pub fn parseConfig(zml_handler: main.Zml_handler, path: []const u8) !std.json.Parsed(Config) {
    const parsed_config = blk: {
        const config_json_file = try std.Io.Dir.openFileAbsolute(zml_handler.io, path, .{});
        defer config_json_file.close(zml_handler.io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(zml_handler.io, &config_json_buffer);
        var reader = std.json.Reader.init(zml_handler.allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(Config, zml_handler.allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();
    return parsed_config;
}

pub fn compileModel(zml_handler: main.Zml_handler, model: AceDit, params: Params, shardings: main.Shardings) !zml.Exe {
    const shardings_arr = shardings.all();
    const opts: zml.module.CompilationOptions = .{
        .shardings = &shardings_arr,
    };
    return zml_handler.platform.compile(
        zml_handler.allocator,
        zml_handler.io,
        model,
        .forward,
        .{ params.t_curr, params.t_next, params.x, params.context_latents, params.y },
        opts,
    );
}


pub const AceDit = struct {
    condition_embedder: zml.nn.Linear,
    proj_in: PatchIn,
    proj_out: PatchOut,
    time_embed: TimestepEmbedding,
    time_embed_r: TimestepEmbedding,
    layers: []DiTLayer,
    norm_out: RmsNorm,
    scale_shift_table: zml.Tensor,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !AceDit {
        const layers = try allocator.alloc(DiTLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .condition_embedder = .init(
                store.createTensor("condition_embedder.weight", .{ .d_out, .d }, .{ .d_out = .model }),
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

    pub fn load(self: *const AceDit, zml_handler: main.Zml_handler, store: *zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(AceDit) {
        return zml.io.load(AceDit, self, zml_handler.arena.allocator(), zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 128 * 1024 * 1024,
        });
    }
    
    pub fn unloadBuffers(self: *zml.Bufferized(AceDit)) void {
        self.condition_embedder.weight.deinit();
        if (self.condition_embedder.bias) |*bias| bias.deinit();
        PatchIn.unloadBuffers(&self.proj_in);
        PatchOut.unloadBuffers(&self.proj_out);
        TimestepEmbedding.unloadBuffers(&self.time_embed);
        TimestepEmbedding.unloadBuffers(&self.time_embed_r);
        for (self.layers) |*layer| {
            DiTLayer.unloadBuffers(layer);
        }
        RmsNorm.unloadBuffers(&self.norm_out);
        self.scale_shift_table.deinit();
    }

    // given a given latent state x, and a timestamp t, computes the predicted flow velocity v by running the full decoder model,
    // conditioned on the encoder hidden states y. we also return it's transposate, as it's relatively small, because on the last
    // step of diffusion, the final latent must be in transposed shape for the decoder.
    pub fn forward(self: AceDit, t_curr: zml.Tensor, t_next: zml.Tensor, x: zml.Tensor, context_latents: zml.Tensor, y_enc: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        var y = self.condition_embedder.convert(hz_type).forward(y_enc.withTags(.{ .s, .d }));
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

        // pass through decoder transformer layers
        // decoder uses alternating full/sliding attention layers, each contaning a cross attention and a self attention
        hidden_states = hidden_states.rename(.{ .t = .s });
        const window_attention_mask = createBidirectionalWindowMask(hidden_states.dim(.s), self.config.sliding_window);
        for (self.layers) |layer| {
            const actual_mask = if (layer.id % 2 == 0) window_attention_mask else null;
            hidden_states = layer.forward(hidden_states, y, timestep_proj, actual_mask);
        }
        hidden_states = hidden_states.rename(.{ .s = .t });
        
        // adaptive layer norm based on scale shift
        hidden_states = self.scale_shift(hidden_states, temb);
        
        // project back latent embeddings into original latent space
        // proj_out projects from the embedding space onto a single audio channel
        // also depatches the time : [t / 2, d_emb] -> [t, a]
        hidden_states = self.proj_out.forward(hidden_states);
           
        // remove padding from output : this is now the predicted flow velocity v
        const v = hidden_states.slice1d(.t, .{ .start = 0, .end = x.dim(.t) });
        
        // update noised latent x with one step of flow matching
        const dt = t_next.sub(t_curr).broad(x.shape());
        const x_next = x.add(v.mul(dt));
     
        return .{ x_next, x_next.withTags(.{ .t, .a }).transpose(.{ .a, .t }) };
    }
    
    pub fn forwardNoIter(self: AceDit, t_curr: zml.Tensor, t_next: zml.Tensor, x: zml.Tensor, context_latents: zml.Tensor, y_enc: zml.Tensor) zml.Tensor {
        var y = self.condition_embedder.convert(hz_type).forward(y_enc.withTags(.{ .s, .d }));
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

        // pass through decoder transformer layers
        // decoder uses alternating full/sliding attention layers, each contaning a cross attention and a self attention
        hidden_states = hidden_states.rename(.{ .t = .s });
        const window_attention_mask = createBidirectionalWindowMask(hidden_states.dim(.s), self.config.sliding_window);
        for (self.layers) |layer| {
            const actual_mask = if (layer.id % 2 == 0) window_attention_mask else null;
            hidden_states = layer.forward(hidden_states, y, timestep_proj, actual_mask);
        }
        hidden_states = hidden_states.rename(.{ .s = .t });
        
        // adaptive layer norm based on scale shift
        hidden_states = self.scale_shift(hidden_states, temb);
        
        // project back latent embeddings into original latent space
        // proj_out projects from the embedding space onto a single audio channel
        // also depatches the time : [t / 2, d_emb] -> [t, a]
        hidden_states = self.proj_out.forward(hidden_states);
           
        // remove padding from output : this is now the predicted flow velocity v
        const v = hidden_states.slice1d(.t, .{ .start = 0, .end = x.dim(.t) });
        
        _ = t_next;
        return v;
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
    
    pub fn forwardLayer(self: AceDit, x: zml.Tensor, y: zml.Tensor, t: zml.Tensor, layer: DiTLayer, window: bool) zml.Tensor {
        const window_attention_mask = createBidirectionalWindowMask(x.dim(.s), self.config.sliding_window);
        const actual_mask = if (window) window_attention_mask else null;
        return layer.forward(x, y, t, actual_mask);
    }
    
    pub fn scale_shift(self: AceDit, hidden_states: zml.Tensor, temb: zml.Tensor) zml.Tensor {
        // scale shift parameters for adaptive output normalization
        const out_mod = self.scale_shift_table.convert(hz_type).squeeze(.n1).add(temb.insertAxes(0, .{ .n2 }).broad(self.scale_shift_table.squeeze(.n1).shape()));
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


const AceDitIterWrapper = struct {
    dit: AceDit,
    pub fn forward(wrapper: AceDitIterWrapper, x: zml.Tensor, context_latents: zml.Tensor, y: zml.Tensor, t_curr: zml.Tensor, t_next: zml.Tensor) zml.Tensor {
        const t_curr_ = t_curr;
        const t_next_ = t_next;
        const x_ = x.withTags(.{ .b, .t, .a }).squeeze(.b);
        const context_latents_ = context_latents.withTags(.{ .b, .t, .a }).squeeze(.b);
        const y_ = y.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.dit.forward(t_curr_, t_next_, x_, context_latents_, y_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitFullWrapper = struct {
    dit: AceDit,
    pub fn forward(wrapper: AceDitFullWrapper, t_curr: zml.Tensor, t_next: zml.Tensor, x: zml.Tensor, context_latents: zml.Tensor, y: zml.Tensor) zml.Tensor {
        const t_curr_ = t_curr.withTags(.{ .t }).squeeze(.t);
        const t_next_ = t_next.withTags(.{ .t }).squeeze(.t);
        const x_ = x.withTags(.{ .b, .t, .a }).squeeze(.b);
        const context_latents_ = context_latents.withTags(.{ .b, .t, .a }).squeeze(.b);
        const y_ = y.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.dit.forwardNoIter(t_curr_, t_next_, x_, context_latents_, y_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitTimeembedWrapper = struct {
    time_embed: TimestepEmbedding,
    pub fn forward(wrapper: AceDitTimeembedWrapper, t: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const t_ = t.withTags(.{ .t }).squeeze(.t);
        const out1, const out2 = wrapper.time_embed.forward(t_);
        return .{ out1.insertAxes(0, .{ .b }), out2.insertAxes(0, .{ .b }) };
    }
};

const AceDitMergepadWrapper = struct {
    dit: AceDit,
    pub fn forward(wrapper: AceDitMergepadWrapper, context_latents: zml.Tensor, x: zml.Tensor) zml.Tensor {
        const context_latents_ = context_latents.withTags(.{ .b, .t, .a }).squeeze(.b);
        const x_ = x.withTags(.{ .b, .t, .a }).squeeze(.b);
        
        const output = wrapper.dit.mergeAndPadLatents(x_, context_latents_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitWindowLayerWrapper = struct {
    dit: AceDit,
    pub fn forward(wrapper: AceDitWindowLayerWrapper, x: zml.Tensor, y: zml.Tensor, t: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        const y_ = y.withTags(.{ .b, .s, .d }).squeeze(.b);
        const t_ = t.withTags(.{ .b, .t, .d_emb }).squeeze(.b);
        const output = wrapper.dit.forwardLayer(x_, y_, t_, wrapper.dit.layers[0], true);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitFullLayerWrapper = struct {
    dit: AceDit,
    pub fn forward(wrapper: AceDitFullLayerWrapper, x: zml.Tensor, y: zml.Tensor, t: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        const y_ = y.withTags(.{ .b, .s, .d }).squeeze(.b);
        const t_ = t.withTags(.{ .b, .t, .d_emb }).squeeze(.b);
        const output = wrapper.dit.forwardLayer(x_, y_, t_, wrapper.dit.layers[1], false);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitScaleshiftWrapper = struct {
    dit: AceDit,
    pub fn forward(wrapper: AceDitScaleshiftWrapper, x: zml.Tensor, t: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .t, .d }).squeeze(.b);
        const t_ = t.withTags(.{ .b, .d }).squeeze(.b);
        const output = wrapper.dit.scale_shift(x_, t_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitProjinWrapper = struct {
    proj_in: PatchIn,
    pub fn forward(wrapper: AceDitProjinWrapper, x: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .t, .a }).squeeze(.b);
        const output = wrapper.proj_in.forward(x_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitProjoutWrapper = struct {
    proj_out: PatchOut,
    pub fn forward(wrapper: AceDitProjoutWrapper, x: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .t, .a }).squeeze(.b);
        const output = wrapper.proj_out.forward(x_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitLayerssWrapper = struct {
    layer: DiTLayer,
    pub fn forward(wrapper: AceDitLayerssWrapper, x: zml.Tensor, time_emb: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        const time_emb_ = time_emb.withTags(.{ .b, .t, .d_emb }).squeeze(.b);
        const output = wrapper.layer.scaleShift(x_, time_emb_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitBetwAttnWrapper = struct {
    layer: DiTLayer,
    pub fn forward(wrapper: AceDitBetwAttnWrapper, x: zml.Tensor, delta: zml.Tensor, time_emb: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        const delta_ = delta.withTags(.{ .b, .s, .d }).squeeze(.b);
        const time_emb_ = time_emb.withTags(.{ .b, .t, .d_emb }).squeeze(.b);
        const output = wrapper.layer.betweenAttn(x_, delta_, time_emb_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitMlpWrapper = struct {
    layer: DiTLayer,
    pub fn forward(wrapper: AceDitMlpWrapper, x: zml.Tensor, time_emb: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        const time_emb_ = time_emb.withTags(.{ .b, .t, .d_emb }).squeeze(.b);
        const output = wrapper.layer.mlpF(x_, time_emb_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitCrossWrapper = struct {
    layer: DiTLayer,
    pub fn forward(wrapper: AceDitCrossWrapper, x: zml.Tensor, y: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        const y_ = y.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.layer.cross_attn.forward(x_, y_);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitPreCrossWrapper = struct {
    layer: DiTLayer,
    pub fn forward(wrapper: AceDitPreCrossWrapper, x: zml.Tensor, y: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        const y_ = y.withTags(.{ .b, .s, .d }).squeeze(.b);
        var q, var k, var v = wrapper.layer.cross_attn.preAttn(x_, y_);
        q = q.withTags(.{ .h, .s, .hd }).insertAxes(0, .{ .b }).transpose(.{ .b, .s, .h, .hd });
        k = k.withTags(.{ .h, .s, .hd }).insertAxes(0, .{ .b }).transpose(.{ .b, .s, .h, .hd });
        v = v.withTags(.{ .h, .s, .hd }).insertAxes(0, .{ .b }).transpose(.{ .b, .s, .h, .hd });
        return .{ q, k, v };
    }
};

const AceDitSelfFullWrapper = struct {
    layer: DiTLayer,
    pub fn forward(wrapper: AceDitSelfFullWrapper, x: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.layer.self_attn.forward(x_, null);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitSelfWindowWrapper = struct {
    layer: DiTLayer,
    pub fn forward(wrapper: AceDitSelfWindowWrapper, x: zml.Tensor) zml.Tensor {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        const mask = createBidirectionalWindowMask(x_.dim(.s), 128);
        const output = wrapper.layer.self_attn.forward(x_, mask);
        return output.insertAxes(0, .{ .b });
    }
};

const AceDitPreSelfWrapper = struct {
    layer: DiTLayer,
    pub fn forward(wrapper: AceDitPreSelfWrapper, x: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        const x_ = x.withTags(.{ .b, .s, .d }).squeeze(.b);
        var q, var k, var v = wrapper.layer.self_attn.preAttn(x_);
        q = q.withTags(.{ .h, .s, .hd }).insertAxes(0, .{ .b }).transpose(.{ .b, .s, .h, .hd });
        k = k.withTags(.{ .h, .s, .hd }).insertAxes(0, .{ .b }).transpose(.{ .b, .s, .h, .hd });
        v = v.withTags(.{ .h, .s, .hd }).insertAxes(0, .{ .b }).transpose(.{ .b, .s, .h, .hd });
        return .{ q, k, v };
    }
};


pub const TimestepEmbedding = struct {
    time_proj_weight: zml.Tensor,
    time_proj_bias: zml.Tensor,
    linear_1: zml.nn.Linear,
    linear_2: zml.nn.Linear,
    config: Config,

    pub fn init(store: zml.io.TensorStore.View, config: Config) TimestepEmbedding {
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
        const t256 = timeEmbedding(timestep);
        var temb = self.linear_1.convert(hz_type).forward(t256);
        temb = temb.silu();
        temb = self.linear_2.convert(hz_type).forward(temb).rename(.{ .d_emb_out = .d_emb });

        var timestep_proj = temb.silu().dot(self.time_proj_weight.convert(hz_type), .d_emb);
        timestep_proj = timestep_proj.add(self.time_proj_bias.convert(hz_type).broad(timestep_proj.shape()));
        timestep_proj = timestep_proj.splitAxis(.d_emb_6, .{ .n2 = 6, .d_emb = self.config.hidden_size });

        return .{ temb, timestep_proj };
    }
    
    fn timeEmbedding(timestep: zml.Tensor) zml.Tensor {
        // timestep is a scalar tensor containing the value of the current timestep in [0, 1]
        const d: u32 = 128;
        const s: f32 = - 0.07195578415; // -ln(10000) / d
        const idx = zml.Tensor.arange(.{ .end = d }, hz_type).withTags(.{ .d128 });
        const scale = zml.Tensor.scalar(s, hz_type);
        const freqs = idx.mul(scale.broad(idx.shape())).exp(); // [exp(-s*i)] i = 0..d128-1
        const t = timestep.mul(zml.Tensor.scalar(1000.0, hz_type)).appendAxes(.{ .d128 });
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

    pub fn init(id: u32, store: zml.io.TensorStore.View, config: Config) !DiTLayer {
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
    
    pub fn forward(self: DiTLayer, x: zml.Tensor, y: zml.Tensor, time_emb: zml.Tensor, attn_mask: ?zml.Tensor) zml.Tensor {
        const x_norm = self.scaleShift(x, time_emb);
        
        const delta_self = self.self_attn.forward(x_norm, attn_mask);
        
        // gated residual connection: x = x + attn_output * gate
        const x1 = self.betweenAttn(x, delta_self, time_emb);

        // step 2: cross-attention (always full)
        const x1_norm = self.cross_attn_norm.forward(x1);
        const delta_cross = self.cross_attn.forward(x1_norm, y);
        const x2 = x1.add(delta_cross);

        // step 3: mlp with adaptive layer norm
        return self.mlpF(x2, time_emb);
    }
    
    pub fn scaleShift(self: DiTLayer, x: zml.Tensor, time_emb: zml.Tensor) zml.Tensor {
        // extract scale-shift parameters for adaptive layer norm from timestep embeddings
        const layer_mod = self.scale_shift_table.convert(hz_type).squeeze(.n1).add(time_emb.rename(.{ .d_emb = .d }));
        const shift_msa = layer_mod.choose1d(.n2, 0);
        const scale_msa = layer_mod.choose1d(.n2, 1);
        
        // step 1: self-attention with adaptive layer norm (AdaLN)
        // adaptive normalization: norm(x) * (1 + scale) + shift
        var x_norm = self.self_attn_norm.forward(x);
        x_norm = x_norm.mul(scale_msa.addConstant(1.0).broad(x_norm.shape()));
        x_norm = x_norm.add(shift_msa.broad(x_norm.shape()));
        
        return x_norm;
    }
    
    pub fn betweenAttn(self: DiTLayer, x: zml.Tensor, delta: zml.Tensor, time_emb: zml.Tensor) zml.Tensor {
        const layer_mod = self.scale_shift_table.convert(hz_type).squeeze(.n1).add(time_emb.rename(.{ .d_emb = .d }));
        const gate_msa = layer_mod.choose1d(.n2, 2);
        // gated residual connection: x = x + attn_output * gate
        return x.add(delta.mul(gate_msa.broad(delta.shape())));
    }
    
    pub fn mlpF(self: DiTLayer, x: zml.Tensor, time_emb: zml.Tensor) zml.Tensor {
        const layer_mod = self.scale_shift_table.convert(hz_type).squeeze(.n1).add(time_emb.rename(.{ .d_emb = .d }));
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

    pub fn init(store: zml.io.TensorStore.View, config: Config) !SelfAttention {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, .{ .d = .model }), null, .d),
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
    pub fn forward(self: SelfAttention, x: zml.Tensor, attn_mask: ?zml.Tensor) zml.Tensor {
        const q, const k, const v = self.preAttn(x);
        
        const attn_heads_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
        const attn_output = attn_heads_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        
        const delta = self.o_proj.convert(hz_type).forward(attn_output);
        
        return delta.rename(.{ .d_out = .d });
    }
    
    pub fn preAttn(self: SelfAttention, x: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        var q = self.q_proj.convert(hz_type).forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.convert(hz_type).forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });
        var v = self.v_proj.convert(hz_type).forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });
        
        const pos_index = zml.Tensor.arange(.{ .end = x.dim(.s) }, .u32).withTags(.{ .s });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        return .{ q, k, v };
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

    pub fn init(store: zml.io.TensorStore.View, config: Config) !CrossAttention {
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
        const q, const k, const v = self.preAttn(x, y);
        
        const attn_heads_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = null, .allow_cudnn = true });
        const attn_output = attn_heads_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });

        const delta = self.o_proj.convert(hz_type).forward(attn_output);
        return delta.rename(.{ .d_out = .d });
    }
    
    pub fn preAttn(self: CrossAttention, x: zml.Tensor, y: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        var k = self.k_proj.convert(hz_type).forward(y).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var v = self.v_proj.convert(hz_type).forward(y).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var q = self.q_proj.convert(hz_type).forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = self.head_dim });
        
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });
        q = q.rename(.{ .s = .q });
        
        return .{ q, k, v };
    }
};

pub const PatchIn = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    config: Config,

    pub fn init(store: zml.io.TensorStore.View, config: Config) PatchIn {
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
            self.weight.convert(hz_type),
            .{ .window_strides = self.config.patch_size },
        );
        out = out.add(self.bias.convert(hz_type).broad(out.shape()));
        return out.transpose(.{ .b, .t, .d }).squeeze(.b);
    }
};

pub const PatchOut = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    config: Config,

    pub fn init(store: zml.io.TensorStore.View, config: Config) PatchOut {
        return .{
            .weight = store.createTensor("1.weight", .{ .d_in, .c_out, .patch }, .{ .d_in = .model }),
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
        const flipped_weight = self.weight.convert(hz_type).reverse(.{ .patch });
        // transpose convolution : reverse kernel output dimensions
        var out = expanded.conv1d(flipped_weight, .{  .kernel_output_feature_dimension = 1, .kernel_input_feature_dimension = 0 });
        out = out.squeeze(.b).transpose(.{ .t, .d });
        out = out.add(self.bias.convert(hz_type).broad(out.shape()));
        return out;
    }
};

pub const MlpLayer = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) !MlpLayer {
        return .{
            .up_proj = store.createTensor("up_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .gate_proj = store.createTensor("gate_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .down_proj = store.createTensor("down_proj.weight", .{ .d, .d_out }, .{ .d = .model }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MlpLayer)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: MlpLayer, input: zml.Tensor) zml.Tensor {
        const up_projection = input.dot(self.up_proj.convert(hz_type), .d);
        const gate_projection = input.dot(self.gate_proj.convert(hz_type), .d);
        const activation = gate_projection.silu().mul(up_projection);
        return activation.dot(self.down_proj.convert(hz_type), .d_out);
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
        return normalized.mul(self.weights.convert(hz_type).withTags(.{ .d }).broad(input.shape()));
    }
};


pub fn createBidirectionalWindowMask(seq_len: i64, window_len: u32) zml.Tensor {
     const attn_shape = zml.Shape.init(.{ .q = seq_len, .k = seq_len }, hz_type);
     const window = zml.DataType.constant(.i32, window_len);
     
    // tokens at pos i attend to tokens at pos [i - w, i + w] ie pos j st. |i - j| <= w
    const q_idx = zml.Tensor.iota(attn_shape, .q);
    const k_idx = zml.Tensor.iota(attn_shape, .k);
    const idx_dist = zml.Tensor.abs(q_idx.sub(k_idx));
    const max_dist = zml.Tensor.constant(window).broad(attn_shape);
    const is_in_window = idx_dist.cmp(.LE, max_dist);

    const zeros = zml.Tensor.zeroes(attn_shape);
    const minf = zml.floats.Float32.minus_inf;
    const minus_inf = zml.Tensor.constant(zml.DataType.constant(hz_type, zml.floats.Float32.toF32(minf))).broad(attn_shape);

    return zml.Tensor.select(is_in_window, zeros, minus_inf);
}


pub fn testModel(zml_handler: main.Zml_handler, acedit: AceDit_handler) !void {
    const activations_path = "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-turbo//activations.safetensors";
    
    try main.printSafetensors(zml_handler.allocator, zml_handler.io, activations_path);

    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(zml_handler.allocator, zml_handler.io, activations_path);
    defer activations_registry.deinit();

    var activations_store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &activations_registry);
    defer activations_store.deinit();
    
    std.log.info("Test activations : dit iter", .{});
    const w_iter: AceDitIterWrapper = .{ .dit = acedit.model };
    const wb_iter: zml.Bufferized(AceDitIterWrapper) = .{ .dit = acedit.model_buffers };
    try testLayer(w_iter, wb_iter, "model", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    
    std.log.info("Test activations : dit layer 0 pre cross", .{});
    const w_pc: AceDitPreCrossWrapper = .{ .layer = acedit.model.layers[0] };
    const wb_pc: zml.Bufferized(AceDitPreCrossWrapper) = .{ .layer = acedit.model_buffers.layers[0] };
    try testLayer(w_pc, wb_pc, "model.layers.0.pre_cross_attn", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit layer 0 cross", .{});
    const w_c: AceDitCrossWrapper = .{ .layer = acedit.model.layers[0] };
    const wb_c: zml.Bufferized(AceDitCrossWrapper) = .{ .layer = acedit.model_buffers.layers[0] };
    try testLayer(w_c, wb_c, "model.layers.0.cross_attn", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit layer 0 ss", .{});
    const w_lss: AceDitLayerssWrapper = .{ .layer = acedit.model.layers[0] };
    const wb_lss: zml.Bufferized(AceDitLayerssWrapper) = .{ .layer = acedit.model_buffers.layers[0] };
    try testLayer(w_lss, wb_lss, "model.layers.0.scale_shift", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit layer 0 pre self attn", .{});
    const w_pselff0: AceDitPreSelfWrapper = .{ .layer = acedit.model.layers[0] };
    const wb_pselff0: zml.Bufferized(AceDitPreSelfWrapper) = .{ .layer = acedit.model_buffers.layers[0] };
    try testLayer(w_pselff0, wb_pselff0, "model.layers.0.pre_self_attn", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit layer 0 self attn : window", .{});
    const w_selfw: AceDitSelfWindowWrapper = .{ .layer = acedit.model.layers[0] };
    const wb_selfw: zml.Bufferized(AceDitSelfWindowWrapper) = .{ .layer = acedit.model_buffers.layers[0] };
    try testLayer(w_selfw, wb_selfw, "model.layers.0.self_attn", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit layer 1 pre self attn", .{});
    const w_pselff: AceDitPreSelfWrapper = .{ .layer = acedit.model.layers[1] };
    const wb_pselff: zml.Bufferized(AceDitPreSelfWrapper) = .{ .layer = acedit.model_buffers.layers[1] };
    try testLayer(w_pselff, wb_pselff, "model.layers.1.pre_self_attn", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit layer 1 self attn : full", .{});
    const w_selff: AceDitSelfFullWrapper = .{ .layer = acedit.model.layers[1] };
    const wb_selff: zml.Bufferized(AceDitSelfFullWrapper) = .{ .layer = acedit.model_buffers.layers[1] };
    try testLayer(w_selff, wb_selff, "model.layers.1.self_attn", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit layer 0 betw", .{});
    const w_bet: AceDitBetwAttnWrapper = .{ .layer = acedit.model.layers[0] };
    const wb_bet: zml.Bufferized(AceDitBetwAttnWrapper) = .{ .layer = acedit.model_buffers.layers[0] };
    try testLayer(w_bet, wb_bet, "model.layers.0.betw_attn", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit layer 0 mlp", .{});
    const w_mlp: AceDitMlpWrapper = .{ .layer = acedit.model.layers[0] };
    const wb_mlp: zml.Bufferized(AceDitMlpWrapper) = .{ .layer = acedit.model_buffers.layers[0] };
    try testLayer(w_mlp, wb_mlp, "model.layers.0.mlp", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    
    std.log.info("Test activations : dit time embed", .{});
    const w_temb: AceDitTimeembedWrapper = .{ .time_embed = acedit.model.time_embed };
    const wb_temb: zml.Bufferized(AceDitTimeembedWrapper) = .{ .time_embed = acedit.model_buffers.time_embed };
    try testLayer(w_temb, wb_temb, "model.time_embed", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit timer embed r", .{});
    const w_tembr: AceDitTimeembedWrapper = .{ .time_embed = acedit.model.time_embed_r };
    const wb_tembr: zml.Bufferized(AceDitTimeembedWrapper) = .{ .time_embed = acedit.model_buffers.time_embed_r };
    try testLayer(w_tembr, wb_tembr, "model.time_embed_r", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit merge pad", .{});
    const w_mergep: AceDitMergepadWrapper = .{ .dit = acedit.model, };
    const wb_mergep: zml.Bufferized(AceDitMergepadWrapper) = .{ .dit = acedit.model_buffers };
    try testLayer(w_mergep, wb_mergep, "model.merge_pad", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit proj in", .{});
    const w_projin: AceDitProjinWrapper = .{ .proj_in = acedit.model.proj_in, };
    const wb_projin: zml.Bufferized(AceDitProjinWrapper) = .{ .proj_in = acedit.model_buffers.proj_in };
    try testLayer(w_projin, wb_projin, "model.proj_in", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit scaleshift", .{});
    const w_ss: AceDitScaleshiftWrapper = .{ .dit = acedit.model, };
    const wb_ss: zml.Bufferized(AceDitScaleshiftWrapper) = .{ .dit = acedit.model_buffers };
    try testLayer(w_ss, wb_ss, "model.scale_shift", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit proj out", .{});
    const w_projout: AceDitProjoutWrapper = .{ .proj_out = acedit.model.proj_out };
    const wb_projout: zml.Bufferized(AceDitProjoutWrapper) = .{ .proj_out = acedit.model_buffers.proj_out };
    try testLayer(w_projout, wb_projout, "model.proj_out", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    
    std.log.info("Test activations : dit layer 0 (window)", .{});
    const w_w: AceDitWindowLayerWrapper = .{ .dit = acedit.model, };
    const wb_w: zml.Bufferized(AceDitWindowLayerWrapper) = .{ .dit = acedit.model_buffers };
    try testLayer(w_w, wb_w, "model.layers.0", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit layer 1 (full)", .{});
    const w_f: AceDitFullLayerWrapper = .{ .dit = acedit.model, };
    const wb_f: zml.Bufferized(AceDitFullLayerWrapper) = .{ .dit = acedit.model_buffers };
    try testLayer(w_f, wb_f, "model.layers.1", zml_handler, activations_store.view(), &acedit.shardings.all());
    
    std.log.info("Test activations : dit full step", .{});
    const w_dit: AceDitFullWrapper = .{ .dit = acedit.model };
    const wb_dit: zml.Bufferized(AceDitFullWrapper) = .{ .dit = acedit.model_buffers };
    try testLayer(w_dit, wb_dit, "model", zml_handler, activations_store.view(), &acedit.shardings.all());
}

pub fn testLayer(wrapper: anytype, buffers: anytype, layername: []const u8, zml_handler: main.Zml_handler, store: zml.io.TensorStore.View, shardings: []const zml.sharding.Sharding) !void {
    try zml.testing.testLayer(zml_handler.allocator, zml_handler.io, zml_handler.platform, wrapper, .forward, store, layername, buffers, shardings, .{});
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