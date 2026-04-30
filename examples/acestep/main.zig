const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const acellm_ = @import("acellm.zig");
const aceemb_ = @import("aceemb.zig");
const aceenc_ = @import("aceenc.zig");
const acedit_ = @import("acedit.zig");
const acevae_ = @import("acevae.zig");
const inference = @import("inference.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Shardings = struct {
    replicated: zml.sharding.Sharding,
    model: zml.sharding.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        const model_mesh: zml.sharding.LogicalMesh = try .init("model", .{ .model = .high_bandwidth });
        const model_sharding_strategy: zml.sharding.Strategy = try .suggest(model_mesh, platform.physical_mesh);
        return .{
            .replicated = try zml.sharding.replicatedSharding(platform),
            .model = try .initFromStrategy(platform, model_mesh, model_sharding_strategy),
        };
    }

    pub fn all(self: *const Shardings) [2]zml.sharding.Sharding {
        return .{ self.replicated, self.model };
    }
};

pub const Zml_handler = struct {
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    io: std.Io,
    platform: *zml.Platform,
    
    pub fn fromInit(init: std.process.Init) !Zml_handler {
        if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
            var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
            defer working_dir.close(init.io);
            try std.process.setCurrentDir(init.io, working_dir);
        }
        
        return .{
            .arena = init.arena,
            .allocator = init.gpa,
            .io = init.io,
            .platform = try .auto(init.gpa, init.io, .{}),
        };
    }
    
    pub fn deinit(self: *Zml_handler) void {
        self.platform.deinit(self.allocator);
    }
};


pub fn main(init: std.process.Init) !void {
    var zml_handler: Zml_handler = try .fromInit(init);
    defer zml_handler.deinit();
    
    //try test5Hz(zml_handler);
    //try testEmb(zml_handler);
    //try testEnc(zml_handler);
    //try testDit(zml_handler);
    //try testVae(zml_handler);
    //try testVaeExampleWav(zml_handler);
    //try testVaeDiffused(zml_handler);
    
    //try testDiffusion(zml_handler);
    
    try runFullPipeline(zml_handler);
    
}

pub fn runFullPipeline(zml_handler: Zml_handler) !void {
    const raw_prompt = "a short electric guitar solo\n\ninstrumental: true";
    
    // text2music : think = false, initial latents initialized from noise
    // cover : think = true, initial latents initialized from audio codes
    const think = true;
    // text2music, overrides the generated duration metadata
    const target_duration: u32 = 12;
    
    // ------------------------------------------------
    // Thinking/Inspiration phase : 5Hz LLM model
    // ------------------------------------------------
    
    var acellm = try acellm_.AceLlm_handler.initFromFile(zml_handler);
    defer acellm.deinit(zml_handler.allocator);

    var audio_metadata: inference.AudioMetadata = try inference.runPhase1(raw_prompt, zml_handler, &acellm);
    if (!think) audio_metadata.setDuration(12);
    const actual_duration = std.fmt.parseUnsigned(u32, audio_metadata.duration, 10) catch target_duration;
    defer audio_metadata.deinit(zml_handler.allocator);
    
    const audio_codes: inference.AudioCodes = if (think) try inference.runPhase2(audio_metadata, zml_handler, &acellm) else .empty();
    defer audio_codes.deinit(zml_handler.allocator);

    acellm.unloadBuffers();
        
    // ------------------------------------------------
    // The text inputs of the DiT need to be embedded
    // using the AceEmb model embedding, not 5Hz
    // ------------------------------------------------

    const full_emb, const partial_emb = try aceemb_.embeddingLengths(zml_handler, audio_metadata);
    var aceemb = try aceemb_.AceEmb_handler.initFromFile(zml_handler, full_emb, partial_emb);
    defer aceemb.deinit(zml_handler.allocator);

    const text_emb: inference.TextEmbedding = try inference.embedTextInputs(zml_handler, audio_metadata, &aceemb);
    defer text_emb.deinit(zml_handler.allocator);
    
    aceemb.unloadBuffers();
    
    // ------------------------------------------------
    // Encoding phase : prepare input latents and
    // encoded conditions for diffusion
    // ------------------------------------------------
    
    const int_codes = try audio_codes.getIntCodes(zml_handler.allocator);
    defer zml_handler.allocator.free(int_codes);
    
    var aceenc = try aceenc_.AceEnc_handler.initFromFile(zml_handler, text_emb.textLen(), text_emb.lyricLen(),  actual_duration, int_codes.len);
    defer aceenc.deinit(zml_handler.allocator);
    
    const diffuse_args: inference.InitialLatents = try inference.prepareLatents(zml_handler, &aceenc, text_emb, int_codes, actual_duration);
    defer diffuse_args.deinit(zml_handler.allocator);
    
    aceenc.unloadBuffers();
    
    // ------------------------------------------------
    // Generation phase : diffusion with DiT model
    // ------------------------------------------------

    var acedit = try acedit_.AceDit_handler.initFromFile(zml_handler, actual_duration, diffuse_args.encoder_conditions.shape.dim(.s_enc));
    defer acedit.deinit(zml_handler.allocator);

    const diffused_latents: inference.DiffusedLatents = try inference.runDiffusion(zml_handler, &acedit, diffuse_args);
    defer diffused_latents.deinit(zml_handler.allocator);

    acedit.unloadBuffers();
    
    // ------------------------------------------------
    // Output latents of the DiT model are decoded
    // with the VAE model
    // ------------------------------------------------

    var acevae = try acevae_.AceVae_handler.initFromFile(zml_handler, actual_duration);
    defer acevae.deinit(zml_handler.allocator);

    const decoded_audio: inference.DecodedAudio = try inference.decodeAudioLatents(zml_handler, &acevae, diffused_latents);
    defer decoded_audio.deinit(zml_handler.allocator);

    acevae.unloadBuffers();
    
    // ------------------------------------------------
    // Export decoded audio as WAV
    // ------------------------------------------------
    
     try exportDecodedAudioAsWav(zml_handler.io, decoded_audio, "decoded_audio.wav");
}

pub fn exportDecodedAudioAsWav(io: std.Io, decoded_audio: inference.DecodedAudio, output_path: []const u8) !void {
    const audio = decoded_audio.audio;
    const shape = audio.shape;

    if (shape.rank() != 2) return error.InvalidAudioRank;
    if (audio.dtype() != .f32) return error.UnsupportedAudioType;

    const num_channels_i64 = shape.dim(0);
    const num_frames_i64 = shape.dim(1);

    const num_frames: u32 = std.math.cast(u32, num_frames_i64) orelse return error.AudioTooLong;
    const num_channels: u16 = std.math.cast(u16, num_channels_i64) orelse return error.InvalidChannelCount;
    const sample_rate: u32 = 48_000;
    const bytes_per_sample: u16 = @sizeOf(f32);
    const bits_per_sample: u16 = bytes_per_sample * 8;
    const block_align: u16 = num_channels * bytes_per_sample;
    const byte_rate: u32 = sample_rate * block_align;

    const samples = audio.constItems(f32);
    const expected_samples_u64 = @as(u64, num_frames) * num_channels;
    const expected_samples: usize = std.math.cast(usize, expected_samples_u64) orelse return error.AudioTooLarge;
    if (samples.len != expected_samples) return error.InvalidAudioBufferSize;

    const data_chunk_size_u64 = @as(u64, samples.len) * @sizeOf(f32);
    const data_chunk_size: u32 = std.math.cast(u32, data_chunk_size_u64) orelse return error.AudioTooLarge;
    const fact_chunk_size: u32 = 4;
    const riff_chunk_size_u64: u64 = 36 + 12 + data_chunk_size;
    const riff_chunk_size: u32 = std.math.cast(u32, riff_chunk_size_u64) orelse return error.AudioTooLarge;

    var file = try std.Io.Dir.createFile(.cwd(), io, output_path, .{ .truncate = true });
    defer file.close(io);

    var file_writer = file.writer(io, &.{});
    const writer: *std.Io.Writer = &file_writer.interface;

    try writer.writeAll("RIFF");
    try writer.writeInt(u32, riff_chunk_size, .little);
    try writer.writeAll("WAVE");

    try writer.writeAll("fmt ");
    try writer.writeInt(u32, 16, .little);
    try writer.writeInt(u16, 3, .little);
    try writer.writeInt(u16, num_channels, .little);
    try writer.writeInt(u32, sample_rate, .little);
    try writer.writeInt(u32, byte_rate, .little);
    try writer.writeInt(u16, block_align, .little);
    try writer.writeInt(u16, bits_per_sample, .little);

    try writer.writeAll("fact");
    try writer.writeInt(u32, fact_chunk_size, .little);
    try writer.writeInt(u32, num_frames, .little);

    try writer.writeAll("data");
    try writer.writeInt(u32, data_chunk_size, .little);

    for (0..num_frames) |frame_idx| {
        for (0..num_channels) |channel_idx| {
            const sample = samples[channel_idx * num_frames + frame_idx];
            try writer.writeInt(u32, @bitCast(sample), .little);
        }
    }

    try writer.flush();
    log.info("Exported decoded audio to {s}", .{ output_path });
}

pub fn printSafetensors(allocator: std.mem.Allocator, io: std.Io, fpath: []const u8) !void {
    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fpath);
    defer registry.deinit();
    std.log.info("Found {d} activations in {s}", .{ registry.tensors.count(), fpath });

    // Print model shapes
    const tensors: zml.safetensors.Tensors = registry.tensors;
    const data = tensors.entries;
    for (0..data.len) |i| {
        const entry = data.get(i);
        const tensor: zml.safetensors.Tensor = tensors.get(entry.key).?;
        std.log.info("Tensor(name={s} shape={f} size={d})", .{
            tensor.name,
            tensor.shape,
            tensor.byteSize(),
        });
    }
}


pub fn test5Hz(zml_handler: Zml_handler) !void {
    var llm = try acellm_.AceLlm_handler.initFromFile(zml_handler);
    defer llm.deinit(zml_handler.allocator);
    try acellm_.testModel(zml_handler, llm);
    llm.unloadBuffers();
}

pub fn testEmb(zml_handler: Zml_handler) !void {
    const path = "//Users//sboulmier//zml//examples//acestep//models//Qwen3-Embedding-0.6B//activations.safetensors";
    try printSafetensors(zml_handler.allocator, zml_handler.io, path);
    const lyric_tokens = try getSlice(zml_handler, path, "lyric_token_ids", .i64);
    const text_tokens = try getSlice(zml_handler, path, "text_token_ids", .i64);
    defer lyric_tokens.free(zml_handler.allocator);
    defer text_tokens.free(zml_handler.allocator);
    
    const lyric_tokens_i64 = lyric_tokens.items(i64);
    const text_tokens_i64 = text_tokens.items(i64);
    const nb_lyric = lyric_tokens_i64.len;
    const nb_text = text_tokens_i64.len;
    
    var emb = try aceemb_.AceEmb_handler.initFromFile(zml_handler, @intCast(nb_text), @intCast(nb_lyric));
    defer emb.deinit(zml_handler.allocator);

    const lyric_tokens_u32 = try zml_handler.allocator.alloc(u32, nb_lyric);
    const text_tokens_u32 = try zml_handler.allocator.alloc(u32, nb_text);
    defer zml_handler.allocator.free(lyric_tokens_u32);
    defer zml_handler.allocator.free(text_tokens_u32);

    for (lyric_tokens_i64, 0..) |token, i| {
        lyric_tokens_u32[i] = std.math.cast(u32, token) orelse return error.InvalidTokenValue;
    }
    for (text_tokens_i64, 0..) |token, i| {
        text_tokens_u32[i] = std.math.cast(u32, token) orelse return error.InvalidTokenValue;
    }

    const lyric_embedding = try inference.generateTextEmbedding(zml_handler, &emb, lyric_tokens_u32, true);
    const text_embedding = try inference.generateTextEmbedding(zml_handler, &emb, text_tokens_u32, false);
    defer lyric_embedding.free(zml_handler.allocator);
    defer text_embedding.free(zml_handler.allocator);

    const lyric_activation = try getSlice(zml_handler, path, "lyric_embedding", .f32);
    const text_activation = try getSlice(zml_handler, path, "text_embedding", .f32);
    defer lyric_activation.free(zml_handler.allocator);
    defer text_activation.free(zml_handler.allocator);

    var lyric_max_abs_error: f32 = 0.0;
    var lyric_max_rel_error: f32 = 0.0;
    for (lyric_embedding.constItems(f32), lyric_activation.constItems(f32)) |actual, expected| {
        const abs_error = @abs(actual - expected);
        const rel_error = abs_error / (1.0 + @abs(actual) + @abs(expected));
        if (abs_error > lyric_max_abs_error) lyric_max_abs_error = abs_error;
        if (rel_error > lyric_max_rel_error) lyric_max_rel_error = rel_error;
    }

    var text_max_abs_error: f32 = 0.0;
    var text_max_rel_error: f32 = 0.0;
    for (text_embedding.constItems(f32), text_activation.constItems(f32)) |actual, expected| {
        const abs_error = @abs(actual - expected);
        const rel_error = abs_error / (1.0 + @abs(actual) + @abs(expected));
        if (abs_error > text_max_abs_error) text_max_abs_error = abs_error;
        if (rel_error > text_max_rel_error) text_max_rel_error = rel_error;
    }

    std.log.info("EMB lyric max absolute error: {d}", .{lyric_max_abs_error});
    std.log.info("EMB lyric max relative error: {d}", .{lyric_max_rel_error});
    std.log.info("EMB text max absolute error: {d}", .{text_max_abs_error});
    std.log.info("EMB text max relative error: {d}", .{text_max_rel_error});

    emb.unloadBuffers();
}

pub fn testEnc(zml_handler: Zml_handler) !void {
    var enc = try aceenc_.AceEnc_handler.initFromFile(zml_handler, 68, 30, 120, 0);
    defer enc.deinit(zml_handler.allocator);
    try aceenc_.testModel(zml_handler, enc);
    enc.unloadBuffers();
}

pub fn testDit(zml_handler: Zml_handler) !void {
    var dit = try acedit_.AceDit_handler.initFromFile(zml_handler, 3000, 65);
    defer dit.deinit(zml_handler.allocator);
    try acedit_.testModel(zml_handler, dit);
    dit.unloadBuffers();
}

pub fn testVae(zml_handler: Zml_handler) !void {
    var vae = try acevae_.AceVae_handler.initFromFile(zml_handler, 300);
    defer vae.deinit(zml_handler.allocator);
    try acevae_.testModel(zml_handler, vae);
    vae.unloadBuffers();
}

pub fn testVaeExampleWav(zml_handler: Zml_handler) !void {
    const path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//example_encoded.safetensors";
    const encoded_slice = try getSlice(zml_handler, path, "latents_tc");
    const t = encoded_slice.shape.dim(1);
    defer encoded_slice.free(zml_handler.allocator);
    
    var acevae = try acevae_.AceVae_handler.initFromFile(zml_handler, @intCast(@divExact(t, 25)));
    defer acevae.deinit(zml_handler.allocator);
    
    const decoded_audio: inference.DecodedAudio = try inference.decodeAudioLatents(zml_handler, &acevae, .{ .x = encoded_slice });
    defer decoded_audio.deinit(zml_handler.allocator);
    acevae.unloadBuffers();
    
    try exportDecodedAudioAsWav(zml_handler.io, decoded_audio, "decoded_audio.wav");
}

pub fn testVaeDiffused(zml_handler: Zml_handler) !void {
    const path = "//Users//sboulmier//zml//examples//acestep//models//Oobleck-vae//diffused_latents4.safetensors";
    const diffused_slice = try getSlice(zml_handler, path, "diffused_latents", .f32);
    const t = diffused_slice.shape.dim(1);
    defer diffused_slice.free(zml_handler.allocator);
    
    var acevae = try acevae_.AceVae_handler.initFromFile(zml_handler, @intCast(@divExact(t, 25)));
    defer acevae.deinit(zml_handler.allocator);
    
    const decoded_audio: inference.DecodedAudio = try inference.decodeAudioLatents(zml_handler, &acevae, .{ .x = diffused_slice });
    defer decoded_audio.deinit(zml_handler.allocator);
    acevae.unloadBuffers();
    
    try exportDecodedAudioAsWav(zml_handler.io, decoded_audio, "decoded_audio.wav");
}

pub fn testDiffusion(zml_handler: Zml_handler) !void {
    //x shape torch.Size([1, 300, 64])
    //encoder_hidden_states shape torch.Size([1, 65, 2048])
    //context_latents shape torch.Size([1, 300, 128])
    const initial_latents: inference.InitialLatents = .{
        .x = try getSlice(zml_handler, "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-turbo//dit_args.safetensors", "xt", .f32),
        .context_latents = try getSlice(zml_handler, "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-turbo//dit_args.safetensors", "context_latents", .f32),
        .encoder_conditions = try getSlice(zml_handler, "//Users//sboulmier//zml//examples//acestep//models//acestep-v15-turbo//dit_args.safetensors", "encoder_hidden_states", .f32),
    };
    defer initial_latents.deinit(zml_handler.allocator);
    std.log.info("Slices extracted", .{});
    
    var acedit = try acedit_.AceDit_handler.initFromFile(zml_handler, 12, initial_latents.encoder_conditions.shape.dim(0));
    defer acedit.deinit(zml_handler.allocator);

    // Test model activations
    //try acedit.testModel(zml_handler);

    const diffused_latents: inference.DiffusedLatents = try inference.runDiffusion(zml_handler, &acedit, initial_latents);
    defer diffused_latents.deinit(zml_handler.allocator);

    //try diffused_latents.print(zml_handler.io);
    acedit.unloadBuffers();
    
    // ------------------------------------------------
    // Output latents of the DiT model are decoded
    // with the VAE model
    // ------------------------------------------------

    var acevae = try acevae_.AceVae_handler.initFromFile(zml_handler, 12);
    defer acevae.deinit(zml_handler.allocator);
    
    // Test model activations
    //try acevae.testModel(zml_handler);

    const decoded_audio: inference.DecodedAudio = try inference.decodeAudioLatents(zml_handler, &acevae, diffused_latents);
    defer decoded_audio.deinit(zml_handler.allocator);

    //try decoded_audio.print(zml_handler.io);
    acevae.unloadBuffers();
    
    // ------------------------------------------------
    // Export decoded audio as WAV
    // ------------------------------------------------
    
     try exportDecodedAudioAsWav(zml_handler.io, decoded_audio, "decoded_audio.wav");
}

pub fn getSlice(zml_handler: Zml_handler, file_name: []const u8, tensor_name: []const u8, dtype: anytype) !zml.Slice {
    std.log.info("Getting slice {s}", .{tensor_name});
    var registry: zml.safetensors.TensorRegistry = try .fromPath(zml_handler.allocator, zml_handler.io, file_name);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
    defer store.deinit();
    const model: TensorExtractor = .init(store.view(), tensor_name);
    const shardings: Shardings = try .init(zml_handler.platform);
    const shardings_arr = shardings.all();
    const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };
    const extract_exe = try zml_handler.platform.compile(zml_handler.allocator, zml_handler.io, model, .forward, .{}, opts);
    defer extract_exe.deinit();
    var model_buffers = try model.load(zml_handler, &store, &shardings.all());
    defer TensorExtractor.unloadBuffers(&model_buffers);
    const slice: zml.Slice = try .alloc(zml_handler.allocator, .init(model.shape(), dtype));
    var buffer: zml.Buffer = try .fromSlice(zml_handler.io, zml_handler.platform, slice, shardings.all()[0]);
    defer buffer.deinit();
    var extract_args = try extract_exe.args(zml_handler.allocator);
    defer extract_args.deinit(zml_handler.allocator);
    var extract_results = try extract_exe.results(zml_handler.allocator);
    defer extract_results.deinit(zml_handler.allocator);
    extract_args.set(.{ model_buffers });
    extract_exe.call(extract_args, &extract_results);
    extract_results.fill(.{ &buffer });
    try buffer.toSlice(zml_handler.io, slice);
    return slice;
}

const TensorExtractor = struct {
    tensor: zml.Tensor,
    pub fn init(store: zml.io.TensorStore.View, tensor_name: []const u8) TensorExtractor {
        return .{
            .tensor = store.createTensor(tensor_name, null, null),
        };
    }
    pub fn load(self: *const TensorExtractor, zml_handler: Zml_handler, store: *zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(TensorExtractor) {
        return zml.io.load(TensorExtractor, self, zml_handler.arena.allocator(), zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 128 * 1024 * 1024,
        });
    }
    pub fn unloadBuffers(self: *zml.Bufferized(TensorExtractor)) void {
        self.tensor.deinit();
    }
    pub fn shape(self: TensorExtractor) zml.Shape {
        return self.tensor.shape();
    }
    pub fn forward(self: TensorExtractor) zml.Tensor {
        return self.tensor;
    }
};
