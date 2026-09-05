const std = @import("std");
const builtin = @import("builtin");
const zml = @import("zml");
const layers = @import("layers.zig");
const model = @import("model.zig");
const codec = @import("codec.zig");
const tokenizer = @import("tokenizer.zig");
const audio = @import("audio_io.zig");
const Tensor = zml.Tensor;

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    model: []const u8,
    text: []const u8,
    output: []const u8,
    voice: []const u8 = "casual_female",
    max_frames: u32 = 375,
    steps: u32 = 7,
    guidance: f32 = 1.2,
    seed: u64 = 0,
    dump_dir: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    var debug_allocator: std.heap.DebugAllocator(.{ .thread_safe = true }) = .init;
    defer if (builtin.mode == .Debug) std.debug.assert(debug_allocator.deinit() == .ok);
    const allocator = if (builtin.mode == .Debug) debug_allocator.allocator() else std.heap.smp_allocator;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if (args.text.len == 0 or args.max_frames == 0 or args.max_frames > 1500 or args.steps == 0 or args.steps > 50 or !std.math.isFinite(args.guidance) or args.guidance < 0 or args.guidance > 5) return error.InvalidArguments;
    const dir = try std.Io.Dir.cwd().openDir(io, args.model, .{});
    defer dir.close(io);
    try validateConfig(allocator, io, dir);
    const voice = try audio.Voice.load(allocator, io, dir, args.voice);
    defer voice.deinit(allocator);
    const tekken = try dir.readFileAlloc(io, "tekken.json", allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(tekken);
    std.log.info("Loading Tekken text tokenizer...", .{});
    var tok = try tokenizer.load(allocator, tekken);
    defer tok.deinit();
    var encoder = try tok.encoder();
    defer encoder.deinit();
    const text_ids = try encoder.encodeAlloc(allocator, args.text);
    defer allocator.free(text_ids);
    const prompt_ids = try tokenizer.prompt(allocator, text_ids, voice.frames);
    defer allocator.free(prompt_ids);
    if (prompt_ids.len > 4096) return error.TextTooLong;
    std.log.info("Voice {s}: {} frames; text: {} tokens", .{ args.voice, voice.frames, text_ids.len });
    try dump(allocator, io, args.dump_dir, "prompt.u32", std.mem.sliceAsBytes(prompt_ids));

    const file = try dir.openFile(io, "consolidated.safetensors", .{});
    defer file.close(io);
    var registry = try zml.safetensors.fetchRegistry(allocator, io, dir, file);
    defer registry.deinit();
    var store = zml.io.TensorStore.fromRegistry(allocator, &registry);
    defer store.deinit();
    const platform = try zml.Platform.auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    std.log.info("Selected platform: {f}", .{platform.fmtVerbose()});
    const llm = try model.LanguageModel.init(allocator, store.view());
    defer allocator.free(llm.blocks);
    const acoustic = try model.AcousticModel.init(allocator, store.view());
    defer allocator.free(acoustic.blocks);
    const decoder = try codec.Codec.init(allocator, store.view());
    defer allocator.free(decoder.convs);
    defer allocator.free(decoder.blocks);
    var cache_spec: model.Cache = undefined;
    // Metal flash attention requires the cache sequence axis to be aligned.
    const cache_length = std.mem.alignForward(usize, prompt_ids.len + args.max_frames, 64);
    for (&cache_spec) |*entry| entry.* = .init(@intCast(cache_length));

    std.log.info("Compiling language-model prefill...", .{});
    var prefill_exe = try platform.compile(allocator, io, llm, .prefill, .{
        Tensor.init(.{ .s = prompt_ids.len }, .u32),
        Tensor.init(.{ .s = voice.frames, .d = 3072 }, .bf16),
        cache_spec,
    }, .{ .program_name = "voxtral_tts_prefill" });
    defer prefill_exe.deinit();
    std.log.info("Compiling language-model decode...", .{});
    var decode_exe = try platform.compile(allocator, io, llm, .decode, .{
        Tensor.init(.{ .cb = 37 }, .u32), Tensor.init(.{}, .u32), cache_spec,
    }, .{ .program_name = "voxtral_tts_decode" });
    defer decode_exe.deinit();
    std.log.info("Compiling acoustic flow matching ({} steps)...", .{args.steps});
    var acoustic_exe = try platform.compile(allocator, io, acoustic, .generate, .{
        Tensor.init(.{ .s = 1, .d = 3072 }, .bf16), Tensor.init(.{ .cb = 36 }, .f32), args.steps, args.guidance,
    }, .{ .program_name = "voxtral_tts_acoustic" });
    defer acoustic_exe.deinit();

    std.log.info("Loading model weights...", .{});
    var llm_buffers = try layers.load(model.LanguageModel, &llm, allocator, io, platform, &store);
    defer allocator.free(llm_buffers.blocks);
    defer layers.unload(&llm_buffers);
    var acoustic_buffers = try layers.load(model.AcousticModel, &acoustic, allocator, io, platform, &store);
    defer allocator.free(acoustic_buffers.blocks);
    defer layers.unload(&acoustic_buffers);
    var codec_buffers = try layers.load(codec.Codec, &decoder, allocator, io, platform, &store);
    defer allocator.free(codec_buffers.convs);
    defer allocator.free(codec_buffers.blocks);
    defer layers.unload(&codec_buffers);
    var cache = try initCache(allocator, io, platform, cache_spec);
    defer layers.unload(&cache);
    var prefill = try prefill_exe.runner(allocator);
    defer prefill.deinit(allocator);
    var decode = try decode_exe.runner(allocator);
    defer decode.deinit(allocator);
    var generate = try acoustic_exe.runner(allocator);
    defer generate.deinit(allocator);
    var hidden: zml.Buffer = undefined;
    {
        var tokens = try zml.Buffer.fromSlice(io, platform, .init(.init(.{ .s = prompt_ids.len }, .u32), std.mem.sliceAsBytes(prompt_ids)), .replicated);
        defer tokens.deinit();
        var voice_buffer = try zml.Buffer.fromSlice(io, platform, .init(.init(.{ .s = voice.frames, .d = 3072 }, .bf16), voice.bytes), .replicated);
        defer voice_buffer.deinit();
        var updated: zml.Bufferized(model.Cache) = undefined;
        std.log.info("Generating speech...", .{});
        prefill.run(.{ llm_buffers, tokens, voice_buffer, cache }, .{ &hidden, &updated });
        layers.unload(&cache);
        cache = updated;
    }
    defer hidden.deinit();
    if (args.dump_dir != null) {
        const slice = try hidden.toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        try dump(allocator, io, args.dump_dir, "hidden.bf16", slice.data());
    }
    var prng: std.Random.DefaultPrng = .init(args.seed);
    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(allocator);
    var reached_eos = false;
    const start: std.Io.Timestamp = .now(io, .awake);
    for (0..args.max_frames) |frame| {
        var noise: [36]f32 = undefined;
        for (&noise) |*value| value.* = prng.random().floatNorm(f32);
        if (frame == 0) try dump(allocator, io, args.dump_dir, "noise.f32", std.mem.asBytes(&noise));
        var noise_buffer = try zml.Buffer.fromSlice(io, platform, .init(.init(.{ .cb = 36 }, .f32), std.mem.asBytes(&noise)), .replicated);
        defer noise_buffer.deinit();
        var codes: zml.Buffer = undefined;
        generate.run(.{ acoustic_buffers, hidden, noise_buffer }, .{&codes});
        defer codes.deinit();
        const result = try codes.toSliceAlloc(allocator, io);
        defer result.free(allocator);
        if (result.items(u32)[0] == 1) {
            reached_eos = true;
            break;
        }
        try generated.appendSlice(allocator, result.items(u32));
        if (frame + 1 == args.max_frames) break;
        var index = try zml.Buffer.scalar(io, platform, prompt_ids.len + frame, .u32);
        defer index.deinit();
        var next: zml.Buffer = undefined;
        var updated: zml.Bufferized(model.Cache) = undefined;
        decode.run(.{ llm_buffers, codes, index, cache }, .{ &next, &updated });
        hidden.deinit();
        layers.unload(&cache);
        hidden = next;
        cache = updated;
        if ((frame + 1) % 25 == 0) std.log.info("Generated {d:.1}s of speech", .{@as(f32, @floatFromInt(frame + 1)) / 12.5});
    }
    const frames = generated.items.len / 37;
    if (frames == 0) return error.NoAudioGenerated;
    if (!reached_eos) std.log.warn("Reached --max-frames; output is truncated", .{});
    std.log.info("Generated {} audio frames in {f}", .{ frames, start.untilNow(io, .awake) });
    try dump(allocator, io, args.dump_dir, "codes.u32", std.mem.sliceAsBytes(generated.items));
    std.log.info("Compiling waveform decoder for {} frames...", .{frames});
    var codec_exe = try platform.compile(allocator, io, decoder, .forward, .{Tensor.init(.{ .s = frames, .cb = 37 }, .u32)}, .{ .program_name = "voxtral_tts_waveform" });
    defer codec_exe.deinit();
    var codec_runner = try codec_exe.runner(allocator);
    defer codec_runner.deinit(allocator);
    var codes = try zml.Buffer.fromSlice(io, platform, .init(.init(.{ .s = frames, .cb = 37 }, .u32), std.mem.sliceAsBytes(generated.items)), .replicated);
    defer codes.deinit();
    var waveform: zml.Buffer = undefined;
    codec_runner.run(.{ codec_buffers, codes }, .{&waveform});
    defer waveform.deinit();
    const result = try waveform.toSliceAlloc(allocator, io);
    defer result.free(allocator);
    try dump(allocator, io, args.dump_dir, "audio.f32", result.data());
    try audio.writeWav(io, args.output, result.items(f32));
    std.log.info("Saved {s} ({d:.2}s, mono 24 kHz)", .{ args.output, @as(f32, @floatFromInt(result.items(f32).len)) / 24000 });
    if (args.dump_dir != null) {
        var trace_exe = try platform.compile(allocator, io, decoder, .stages, .{Tensor.init(.{ .s = frames, .cb = 37 }, .u32)}, .{ .program_name = "voxtral_tts_codec_trace" });
        defer trace_exe.deinit();
        var runner = try trace_exe.runner(allocator);
        defer runner.deinit(allocator);
        var trace: [10]zml.Buffer = undefined;
        runner.run(.{ codec_buffers, codes }, .{&trace});
        defer layers.unload(&trace);
        for (trace, 0..) |buffer, i| {
            const slice = try buffer.toSliceAlloc(allocator, io);
            defer slice.free(allocator);
            var name: [32]u8 = undefined;
            try dump(allocator, io, args.dump_dir, try std.fmt.bufPrint(&name, "codec-{d}.bf16", .{i}), slice.data());
        }
    }
}

fn initCache(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, spec: model.Cache) !zml.Bufferized(model.Cache) {
    const zeros = try zml.Slice.alloc(allocator, spec[0].k.shape());
    defer zeros.free(allocator);
    @memset(zeros.data(), 0);
    var result: zml.Bufferized(model.Cache) = undefined;
    var initialized: usize = 0;
    errdefer for (result[0..initialized]) |*entry| layers.unload(entry);
    for (&result) |*entry| {
        entry.k = try .fromSlice(io, platform, zeros, .replicated);
        errdefer entry.k.deinit();
        entry.v = try .fromSlice(io, platform, zeros, .replicated);
        initialized += 1;
    }
    return result;
}

fn dump(allocator: std.mem.Allocator, io: std.Io, path: ?[]const u8, name: []const u8, bytes: []const u8) !void {
    const root = path orelse return;
    const full = try std.fs.path.join(allocator, &.{ root, name });
    defer allocator.free(full);
    const file = try std.Io.Dir.cwd().createFile(io, full, .{ .exclusive = true });
    defer file.close(io);
    var writer = file.writer(io, &.{});
    try writer.interface.writeAll(bytes);
}

fn validateConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    const bytes = try dir.readFileAlloc(io, "params.json", allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.UnsupportedCheckpoint;
    const kind = root.object.get("model_type") orelse return error.UnsupportedCheckpoint;
    if (kind != .string or !std.mem.eql(u8, kind.string, "voxtral_tts")) return error.UnsupportedCheckpoint;
    inline for (.{ .{ "dim", 3072 }, .{ "n_layers", 26 }, .{ "head_dim", 128 }, .{ "hidden_dim", 9216 }, .{ "n_heads", 32 }, .{ "n_kv_heads", 8 }, .{ "vocab_size", 131072 } }) |entry| {
        const value = root.object.get(entry[0]) orelse return error.UnsupportedCheckpoint;
        if (value != .integer or value.integer != entry[1]) return error.UnsupportedCheckpoint;
    }
    try expectNumber(root, "rope_theta", 1000000);
    try expectNumber(root, "norm_eps", 1e-5);
    try expectBool(root, "causal", true);
    try expectBool(root, "use_biases", false);
    const multimodal = try object(root, "multimodal");
    const audio_model = try object(multimodal, "audio_model_args");
    try expectNumber(audio_model, "semantic_codebook_size", 8192);
    try expectNumber(audio_model, "acoustic_codebook_size", 21);
    try expectNumber(audio_model, "n_acoustic_codebook", 36);
    const acoustic = try object(audio_model, "acoustic_transformer_args");
    inline for (.{ .{ "dim", 3072 }, .{ "n_layers", 3 }, .{ "head_dim", 128 }, .{ "hidden_dim", 9216 }, .{ "n_heads", 32 }, .{ "n_kv_heads", 8 } }) |entry| try expectNumber(acoustic, entry[0], entry[1]);
    const codec_args = try object(multimodal, "audio_tokenizer_args");
    inline for (.{ .{ "dim", 1024 }, .{ "hidden_dim", 4096 }, .{ "head_dim", 128 }, .{ "n_heads", 8 }, .{ "n_kv_heads", 8 }, .{ "semantic_dim", 256 }, .{ "acoustic_dim", 36 }, .{ "pretransform_patch_size", 240 }, .{ "patch_proj_kernel_size", 7 }, .{ "sampling_rate", 24000 }, .{ "attn_sliding_window_size", 16 } }) |entry| try expectNumber(codec_args, entry[0], entry[1]);
    try expectNumber(codec_args, "norm_eps", 1e-2);
    try expectNumber(codec_args, "qk_norm_eps", 1e-6);
    inline for (.{ "causal", "qk_norm", "layer_scale", "conv_weight_norm", "half_attn_window_upon_downsampling" }) |key| try expectBool(codec_args, key, true);
    try expectBool(codec_args, "use_biases", false);
    inline for (.{ .{ "decoder_transformer_lengths_str", "2,2,2,2" }, .{ "decoder_convs_kernels_str", "3,4,4,4" }, .{ "decoder_convs_strides_str", "1,2,2,2" } }) |entry| {
        const value = codec_args.object.get(entry[0]) orelse return error.UnsupportedCheckpoint;
        if (value != .string or !std.mem.eql(u8, value.string, entry[1])) return error.UnsupportedCheckpoint;
    }
}

fn expectBool(root: std.json.Value, key: []const u8, expected: bool) !void {
    const value = root.object.get(key) orelse return error.UnsupportedCheckpoint;
    if (value != .bool or value.bool != expected) return error.UnsupportedCheckpoint;
}

fn object(root: std.json.Value, key: []const u8) !std.json.Value {
    const value = root.object.get(key) orelse return error.UnsupportedCheckpoint;
    if (value != .object) return error.UnsupportedCheckpoint;
    return value;
}

fn expectNumber(root: std.json.Value, key: []const u8, expected: f64) !void {
    const value = root.object.get(key) orelse return error.UnsupportedCheckpoint;
    const number: f64 = switch (value) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.UnsupportedCheckpoint,
    };
    if (number != expected) return error.UnsupportedCheckpoint;
}
