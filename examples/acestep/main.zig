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
    
    const raw_prompt = "a short electric guitar solo\n\ninstrumental: true";
    
    // ------------------------------------------------
    // Thinking/Inspiration phase : 5Hz LLM model
    // ------------------------------------------------
    
    var acellm = try acellm_.AceLlm_handler.initFromFile(zml_handler);
    defer acellm.deinit(zml_handler.allocator);

    // Test model activations
    //try acellm.testModel(zml_handler);

    const audio_metadata = try inference.runPhase1(raw_prompt, zml_handler, &acellm);
    //const audio_metadata: inference.AudioMetadata = .initExample();
    defer audio_metadata.deinit(zml_handler.allocator);
    const audio_codes = try inference.runPhase2(audio_metadata, zml_handler, &acellm);
    //const audio_codes: inference.AudioCodes = try .initExample(zml_handler.allocator);
    defer audio_codes.deinit(zml_handler.allocator);

    acellm.unloadBuffers();
    
    // ------------------------------------------------
    // The text inputs of the DiT need to be embedded
    // using the AceEmb model embedding, not 5Hz
    // ------------------------------------------------

    const full_emb, const partial_emb = try aceemb_.embeddingLengths(zml_handler, audio_metadata);
    var aceemb = try aceemb_.AceEmb_handler.initFromFile(zml_handler, full_emb, partial_emb);
    defer aceemb.deinit(zml_handler.allocator);

    // Test model activations
    //try acedit.testModel(zml_handler, aceemb);

    const text_emb: inference.TextEmbedding = try inference.embedTextInputs(zml_handler, audio_metadata, &aceemb);
    defer text_emb.deinit(zml_handler.allocator);
    
    //try text_emb.print(zml_handler.io);
    aceemb.unloadBuffers();
    
    // ------------------------------------------------
    // Encoding phase : prepare input latents and
    // encoded conditions for diffusion
    // ------------------------------------------------
    
    const int_codes = try audio_codes.getIntCodes(zml_handler.allocator);
    defer zml_handler.allocator.free(int_codes);
    
    var aceenc = try aceenc_.AceEnc_handler.initFromFile(zml_handler, text_emb.textLen(), text_emb.lyricLen(), audio_codes.len());
    defer aceenc.deinit(zml_handler.allocator);

    // Test model activations
    //try aceenc.testModel(zml_handler);
    
    const diffuse_args: inference.InitialLatents = try inference.prepareLatents(zml_handler, &aceenc, text_emb, int_codes);
    defer diffuse_args.deinit(zml_handler.allocator);
    
    //try diffuse_args.print(zml_handler.io);
    aceenc.unloadBuffers();
    
    // ------------------------------------------------
    // Generation phase : diffusion with DiT model
    // ------------------------------------------------

    var acedit = try acedit_.AceDit_handler.initFromFile(zml_handler, 5 * audio_codes.len(), diffuse_args.encoder_conditions.shape.dim(.s_enc));
    defer acedit.deinit(zml_handler.allocator);

    // Test model activations
    //try acedit.testModel(zml_handler);

    const diffused_latents: inference.DiffusedLatents = try inference.runDiffusion(zml_handler, &acedit, diffuse_args);
    defer diffused_latents.deinit(zml_handler.allocator);

    //try diffused_latents.print(zml_handler.io);
    acedit.unloadBuffers();
    
    // ------------------------------------------------
    // Output latents of the DiT model are decoded
    // with the VAE model
    // ------------------------------------------------

    var acevae = try acevae_.AceVae_handler.initFromFile(zml_handler, 5 * audio_codes.len());
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

pub fn exportDecodedAudioAsWav(io: std.Io, decoded_audio: inference.DecodedAudio, output_path: []const u8) !void {
    const audio = decoded_audio.audio;
    const shape = audio.shape;

    if (shape.rank() != 2) return error.InvalidAudioRank;

    const num_frames_i64 = shape.dim(0);
    const num_channels_i64 = shape.dim(1);

    if (num_channels_i64 != 2) return error.InvalidChannelCount;
    if (audio.dtype() != .f32) return error.UnsupportedAudioType;

    const num_frames: u32 = std.math.cast(u32, num_frames_i64) orelse return error.AudioTooLong;
    const num_channels: u16 = std.math.cast(u16, num_channels_i64) orelse return error.InvalidChannelCount;
    const sample_rate: u32 = 48_000;
    const bits_per_sample: u16 = 16;
    const bytes_per_sample: u16 = bits_per_sample / 8;
    const block_align: u16 = num_channels * bytes_per_sample;
    const byte_rate: u32 = sample_rate * block_align;
    const data_chunk_size_u64 = @as(u64, num_frames) * block_align;
    const data_chunk_size: u32 = std.math.cast(u32, data_chunk_size_u64) orelse return error.AudioTooLarge;
    const riff_chunk_size_u64: u64 = 36 + data_chunk_size;
    const riff_chunk_size: u32 = std.math.cast(u32, riff_chunk_size_u64) orelse return error.AudioTooLarge;

    const samples = audio.constItems(f32);
    const expected_samples_u64 = @as(u64, num_frames) * num_channels;
    const expected_samples: usize = std.math.cast(usize, expected_samples_u64) orelse return error.AudioTooLarge;
    if (samples.len != expected_samples) return error.InvalidAudioBufferSize;

    var file = try std.Io.Dir.createFile(.cwd(), io, output_path, .{ .truncate = true });
    defer file.close(io);

    var file_writer = file.writer(io, &.{});
    const writer: *std.Io.Writer = &file_writer.interface;

    try writer.writeAll("RIFF");
    try writer.writeInt(u32, riff_chunk_size, .little);
    try writer.writeAll("WAVE");

    try writer.writeAll("fmt ");
    try writer.writeInt(u32, 16, .little);
    try writer.writeInt(u16, 1, .little);
    try writer.writeInt(u16, num_channels, .little);
    try writer.writeInt(u32, sample_rate, .little);
    try writer.writeInt(u32, byte_rate, .little);
    try writer.writeInt(u16, block_align, .little);
    try writer.writeInt(u16, bits_per_sample, .little);

    try writer.writeAll("data");
    try writer.writeInt(u32, data_chunk_size, .little);

    for (samples) |sample| {
        const clamped = std.math.clamp(sample, -1.0, 1.0);
        const pcm_value: i16 = if (clamped <= -1.0)
            -32768
        else
            @intFromFloat(clamped * 32767.0);
        try writer.writeInt(i16, pcm_value, .little);
    }

    try writer.flush();
    log.info("Exported decoded audio to {s}", .{output_path});
}

pub fn printSafetensors(allocator: std.mem.Allocator, io: std.Io, fpath: []const u8) !void {
    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fpath);
    defer registry.deinit();
    std.log.debug("Found {d} activations in {s}", .{ registry.tensors.count(), fpath });

    // Print model shapes
    const tensors: zml.safetensors.Tensors = registry.tensors;
    const data = tensors.entries;
    for (0..data.len) |i| {
        const entry = data.get(i);
        const tensor: zml.safetensors.Tensor = tensors.get(entry.key).?;
        std.log.debug("Tensor(name={s} shape={f} size={d})", .{
            tensor.name,
            tensor.shape,
            tensor.byteSize(),
        });
    }
}
