const std = @import("std");
const zml = @import("zml");
const l = @import("layers.zig");
const m = @import("model.zig");
const codec = @import("codec.zig");
const audio = @import("audio.zig");
const sampling = @import("sampling.zig");
const T = zml.Tensor;
const Args = struct {
    model: []const u8,
    text: []const u8,
    output: []const u8 = "output.wav",
    instruction: []const u8 = "Speak clearly and naturally.",
    ref_audio: ?[]const u8 = null,
    ref_text: ?[]const u8 = null,
    max_frames: u32 = 750,
    temperature: f32 = 0.9,
    top_k: u32 = 50,
    repetition_penalty: f32 = 1.1,
    seed: u64 = 42,
    dump_dir: ?[]const u8 = null,
};
pub fn run(init: std.process.Init, comptime clone: bool) !void {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if (std.mem.trim(u8, args.text, " \t\r\n").len == 0 or args.max_frames == 0 or args.max_frames > 1500 or !std.math.isFinite(args.temperature) or args.temperature < 0 or args.top_k == 0 or args.top_k > 2048 or !std.math.isFinite(args.repetition_penalty) or args.repetition_penalty <= 0) return error.InvalidArguments;
    if (clone) {
        if (args.ref_audio == null or args.ref_text == null or std.mem.trim(u8, args.ref_text.?, " \r\n\t").len == 0) {
            std.log.err("voice_cloning requires --ref-audio and --ref-text", .{});
            return error.MissingReference;
        }
    } else if (args.ref_audio != null or args.ref_text != null) return error.UseVoiceCloningBinary;
    const dir = try std.Io.Dir.cwd().openDir(io, args.model, .{});
    defer dir.close(io);
    try validate(a, io, dir);
    const wave: ?[]f32 = if (clone) try audio.readWav(a, try std.Io.Dir.cwd().readFileAlloc(io, args.ref_audio.?, a, .limited(32 * 1024 * 1024))) else null;
    const tokenizer_bytes = try dir.readFileAlloc(io, "tokenizer.json", a, .limited(64 * 1024 * 1024));
    var tok = try zml.tokenizer.Tokenizer.fromBytes(a, tokenizer_bytes);
    defer tok.deinit();
    var enc = try tok.encoder();
    defer enc.deinit();
    const target = try encodeSegment(a, &enc, try std.fmt.allocPrint(a, "[S0]<ins_bos>{s}<ins_eos>{s}", .{ args.instruction, args.text }));
    const reference: ?[]u32 = if (clone) try encodeSegment(a, &enc, try std.fmt.allocPrint(a, "[S0]{s}", .{std.mem.trim(u8, args.ref_text.?, " \r\n\t")})) else null;
    const ref_frames = if (wave) |w| (w.len + 1919) / 1920 else 0;
    const prompt_len = target.len + (if (reference) |r| r.len + ref_frames + 1 else @as(usize, 0));
    if (prompt_len + args.max_frames > 2048) return error.ContextTooLong;
    const platform = try zml.Platform.init(a, io, .metal, .{});
    defer platform.deinit(a, io);
    std.log.info("Selected platform: {f}", .{platform.fmtVerbose()});
    const file = try dir.openFile(io, "model.safetensors.index.json", .{});
    defer file.close(io);
    var registry = try zml.safetensors.fetchRegistry(a, io, dir, file);
    defer registry.deinit();
    var store = zml.io.TensorStore.fromRegistry(a, &registry);
    defer store.deinit();
    const codec_dir = try dir.openDir(io, "audio_tokenizer", .{});
    defer codec_dir.close(io);
    const codec_file = try codec_dir.openFile(io, "model.safetensors", .{});
    defer codec_file.close(io);
    var codec_registry = try zml.safetensors.fetchRegistry(a, io, codec_dir, codec_file);
    defer codec_registry.deinit();
    var codec_store = zml.io.TensorStore.fromRegistry(a, &codec_registry);
    defer codec_store.deinit();
    const prompt = try a.alloc(u8, prompt_len * 2048 * 2);
    var cursor: usize = 0;
    {
        const text = try m.TextEncoder.init(a, store.view());
        std.log.info("Loading text encoder...", .{});
        var weights = try l.load(m.TextEncoder, &text, a, io, platform, &store);
        defer l.unload(&weights);
        if (reference) |r| {
            const data = try encodeText(a, io, platform, text, weights, r);
            @memcpy(prompt[0..data.len], data);
            cursor = data.len;
        }
        const data = try encodeText(a, io, platform, text, weights, target);
        const offset = if (clone) cursor + (ref_frames + 1) * 2048 * 2 else cursor;
        @memcpy(prompt[offset..][0..data.len], data);
    }
    const llm = try m.Backbone.init(a, store.view());
    var llm_weights = try l.load(m.Backbone, &llm, a, io, platform, &store);
    defer l.unload(&llm_weights);
    if (wave) |w| {
        const encoder = try codec.Encoder.init(a, codec_store.view());
        std.log.info("Compiling reference audio encoder ({} samples)...", .{w.len});
        var exe = try platform.compile(a, io, encoder, .forward, .{T.init(.{ .samples = w.len }, .f32)}, .{ .program_name = "breeze_reference_audio" });
        defer exe.deinit();
        var weights = try l.load(codec.Encoder, &encoder, a, io, platform, &codec_store);
        defer l.unload(&weights);
        var runner = try exe.runner(a);
        defer runner.deinit(a);
        var input = try buffer(io, platform, .init(.{ .samples = w.len }, .f32), std.mem.sliceAsBytes(w));
        defer input.deinit();
        var codes: zml.Buffer = undefined;
        var latent: zml.Buffer = undefined;
        runner.run(.{ weights, input }, .{ &codes, &latent });
        defer codes.deinit();
        defer latent.deinit();
        if (args.dump_dir != null) {
            const encoded = try latent.toSliceAlloc(a, io);
            defer encoded.free(a);
            try dump(a, io, args.dump_dir, "reference_latent.f32", encoded.data());
        }
        const slice = try codes.toSliceAlloc(a, io);
        defer slice.free(a);
        try dump(a, io, args.dump_dir, "reference_codes.u32", slice.data());
        // The all-zero frame is the audio EOS marker used between reference and target text.
        const ids = try a.alloc(u32, (ref_frames + 1) * 16);
        @memcpy(ids[0 .. ref_frames * 16], slice.items(u32));
        @memset(ids[ref_frames * 16 ..], 0);
        var embed_exe = try platform.compile(a, io, llm, .embed, .{T.init(.{ .s = ref_frames + 1, .cb = 16 }, .u32)}, .{ .program_name = "breeze_reference_embed" });
        defer embed_exe.deinit();
        var embed_runner = try embed_exe.runner(a);
        defer embed_runner.deinit(a);
        var id_buffer = try buffer(io, platform, .init(.{ .s = ref_frames + 1, .cb = 16 }, .u32), std.mem.sliceAsBytes(ids));
        defer id_buffer.deinit();
        var embedded: zml.Buffer = undefined;
        embed_runner.run(.{ llm_weights, id_buffer }, .{&embedded});
        defer embedded.deinit();
        const data = try embedded.toSliceAlloc(a, io);
        defer data.free(a);
        @memcpy(prompt[cursor..][0..data.data().len], data.data());
    }
    try dump(a, io, args.dump_dir, "prompt.bf16", prompt);
    var spec: m.BackboneCache = undefined;
    for (&spec) |*c| c.* = .init(std.mem.alignForward(usize, prompt_len + args.max_frames, 64), 8, 128);
    var depth_spec: m.DepthCache = undefined;
    for (&depth_spec) |*c| c.* = .init(16, 2, 128);
    const depth = try m.Depth.init(a, store.view());
    std.log.info("Compiling backbone prefill/decode and depth decoder...", .{});
    var prefill_exe = try platform.compile(a, io, llm, .prefill, .{ T.init(.{ .s = prompt_len, .d = 2048 }, .bf16), spec }, .{ .program_name = "breeze_prefill" });
    defer prefill_exe.deinit();
    var decode_exe = try platform.compile(a, io, llm, .decode, .{ T.init(.{ .s = 1, .cb = 16 }, .u32), T.init(.{}, .u32), spec }, .{ .program_name = "breeze_decode" });
    defer decode_exe.deinit();
    var depth_pre_exe = try platform.compile(a, io, depth, .prefill, .{ T.init(.{ .s = 1, .d = 2048 }, .bf16), T.init(.{}, .u32), depth_spec }, .{ .program_name = "breeze_depth_prefill" });
    defer depth_pre_exe.deinit();
    var depth_dec_exe = try platform.compile(a, io, depth, .decode, .{ T.init(.{}, .u32), T.init(.{}, .u32), depth_spec }, .{ .program_name = "breeze_depth_decode" });
    defer depth_dec_exe.deinit();
    var depth_weights = try l.load(m.Depth, &depth, a, io, platform, &store);
    defer l.unload(&depth_weights);
    var cache = try initCache(a, io, platform, spec);
    defer l.unload(&cache);
    var depth_cache = try initCache(a, io, platform, depth_spec);
    defer l.unload(&depth_cache);
    var prefill = try prefill_exe.runner(a);
    defer prefill.deinit(a);
    var decode = try decode_exe.runner(a);
    defer decode.deinit(a);
    var dpre = try depth_pre_exe.runner(a);
    defer dpre.deinit(a);
    var ddec = try depth_dec_exe.runner(a);
    defer ddec.deinit(a);
    var input = try buffer(io, platform, .init(.{ .s = prompt_len, .d = 2048 }, .bf16), prompt);
    defer input.deinit();
    var hidden: zml.Buffer = undefined;
    var logits: zml.Buffer = undefined;
    var updated: zml.Bufferized(m.BackboneCache) = undefined;
    prefill.run(.{ llm_weights, input, cache }, .{ &hidden, &logits, &updated });
    l.unload(&cache);
    cache = updated;
    defer hidden.deinit();
    defer logits.deinit();
    const first_hidden = try hidden.toSliceAlloc(a, io);
    defer first_hidden.free(a);
    try dump(a, io, args.dump_dir, "hidden.bf16", first_hidden.data());
    var rng: std.Random.DefaultPrng = .init(args.seed);
    var generated: std.ArrayList(u32) = .empty;
    var history: std.ArrayList(u32) = .empty;
    var reached_eos = false;
    std.log.info("Generating speech...", .{});
    for (0..args.max_frames) |frame| {
        const scores = try logits.toSliceAlloc(a, io);
        defer scores.free(a);
        const score_values = try asFloats(a, scores);
        if (frame == 0) try dump(a, io, args.dump_dir, "logits.f32", std.mem.sliceAsBytes(score_values));
        const first = try sampling.sample(score_values, rng.random(), args.temperature, args.top_k, history.items, args.repetition_penalty, true);
        if (first == 2051) {
            reached_eos = true;
            break;
        }
        try history.append(a, first);
        var ids: [16]u32 = undefined;
        ids[0] = first;
        var first_buffer = try zml.Buffer.scalar(io, platform, first, .u32);
        defer first_buffer.deinit();
        var dlogits: zml.Buffer = undefined;
        var dnext: zml.Bufferized(m.DepthCache) = undefined;
        dpre.run(.{ depth_weights, hidden, first_buffer, depth_cache }, .{ &dlogits, &dnext });
        l.unload(&depth_cache);
        depth_cache = dnext;
        for (1..16) |cb| {
            const ds = try dlogits.toSliceAlloc(a, io);
            defer ds.free(a);
            const depth_scores = try asFloats(a, ds);
            if (frame == 0 and cb == 1) try dump(a, io, args.dump_dir, "depth_logits.f32", std.mem.sliceAsBytes(depth_scores));
            ids[cb] = try sampling.sample(depth_scores, rng.random(), args.temperature, args.top_k, &.{}, 1, false);
            dlogits.deinit();
            if (cb < 15) {
                var token = try zml.Buffer.scalar(io, platform, ids[cb], .u32);
                defer token.deinit();
                var idx = try zml.Buffer.scalar(io, platform, cb + 1, .u32);
                defer idx.deinit();
                ddec.run(.{ depth_weights, token, idx, depth_cache }, .{ &dlogits, &dnext });
                l.unload(&depth_cache);
                depth_cache = dnext;
            }
        }
        try generated.appendSlice(a, &ids);
        if (frame + 1 == args.max_frames) break;
        var codes = try buffer(io, platform, .init(.{ .s = 1, .cb = 16 }, .u32), std.mem.asBytes(&ids));
        defer codes.deinit();
        var index = try zml.Buffer.scalar(io, platform, prompt_len + frame, .u32);
        defer index.deinit();
        var next_hidden: zml.Buffer = undefined;
        var next_logits: zml.Buffer = undefined;
        decode.run(.{ llm_weights, codes, index, cache }, .{ &next_hidden, &next_logits, &updated });
        hidden.deinit();
        logits.deinit();
        l.unload(&cache);
        hidden = next_hidden;
        logits = next_logits;
        cache = updated;
        if (frame == 0 and args.dump_dir != null) {
            const next_scores = try logits.toSliceAlloc(a, io);
            defer next_scores.free(a);
            try dump(a, io, args.dump_dir, "decode_logits.f32", std.mem.sliceAsBytes(try asFloats(a, next_scores)));
        }
        if ((frame + 1) % 25 == 0) std.log.info("Generated {d:.1}s", .{@as(f32, @floatFromInt(frame + 1)) / 12.5});
    }
    const frames = generated.items.len / 16;
    if (frames == 0) return error.NoAudioGenerated;
    if (!reached_eos) std.log.warn("Reached --max-frames; output is truncated", .{});
    try dump(a, io, args.dump_dir, "codes.u32", std.mem.sliceAsBytes(generated.items));
    const decoder = try codec.Decoder.init(a, codec_store.view());
    var decoder_weights = try l.load(codec.Decoder, &decoder, a, io, platform, &codec_store);
    defer l.unload(&decoder_weights);
    // Match Qwen's offline chunked decoding with 25 frames of left context.
    var samples: std.ArrayList(f32) = .empty;
    var start: usize = 0;
    while (start < frames) {
        const end = @min(start + 300, frames);
        const context: usize = @min(start, 25);
        const count = end - start + context;
        std.log.info("Decoding waveform frames {}..{}...", .{ start, end });
        var exe = try platform.compile(a, io, decoder, .forward, .{T.init(.{ .s = count, .cb = 16 }, .u32)}, .{ .program_name = "breeze_waveform" });
        defer exe.deinit();
        var runner = try exe.runner(a);
        defer runner.deinit(a);
        var codes = try buffer(io, platform, .init(.{ .s = count, .cb = 16 }, .u32), std.mem.sliceAsBytes(generated.items[(start - context) * 16 .. end * 16]));
        defer codes.deinit();
        var output: zml.Buffer = undefined;
        runner.run(.{ decoder_weights, codes }, .{&output});
        defer output.deinit();
        const result = try output.toSliceAlloc(a, io);
        defer result.free(a);
        try samples.appendSlice(a, result.items(f32)[context * 1920 ..]);
        start = end;
    }
    try dump(a, io, args.dump_dir, "audio.f32", std.mem.sliceAsBytes(samples.items));
    try audio.writeWav(io, args.output, samples.items);
    std.log.info("Saved {s} ({d:.2}s, mono 24 kHz)", .{ args.output, @as(f32, @floatFromInt(samples.items.len)) / 24000 });
}
fn buffer(io: std.Io, p: *const zml.Platform, shape: zml.Shape, bytes: []const u8) !zml.Buffer {
    return .fromSlice(io, p, .init(shape, bytes), .replicated);
}
fn encodeText(a: std.mem.Allocator, io: std.Io, p: *const zml.Platform, text: m.TextEncoder, weights: zml.Bufferized(m.TextEncoder), ids: []const u32) ![]u8 {
    std.log.info("Compiling text encoder for {} tokens...", .{ids.len});
    var exe = try p.compile(a, io, text, .forward, .{T.init(.{ .s = ids.len }, .u32)}, .{ .program_name = "breeze_text_encoder" });
    defer exe.deinit();
    var runner = try exe.runner(a);
    defer runner.deinit(a);
    var input = try buffer(io, p, .init(.{ .s = ids.len }, .u32), std.mem.sliceAsBytes(ids));
    defer input.deinit();
    var output: zml.Buffer = undefined;
    runner.run(.{ weights, input }, .{&output});
    defer output.deinit();
    const slice = try output.toSliceAlloc(a, io);
    defer slice.free(a);
    return a.dupe(u8, slice.data());
}
fn initCache(a: std.mem.Allocator, io: std.Io, p: *const zml.Platform, spec: anytype) !zml.Bufferized(@TypeOf(spec)) {
    const zero = try zml.Slice.alloc(a, spec[0].k.shape());
    defer zero.free(a);
    @memset(zero.data(), 0);
    var result: zml.Bufferized(@TypeOf(spec)) = undefined;
    for (&result) |*c| {
        c.k = try .fromSlice(io, p, zero, .replicated);
        c.v = try .fromSlice(io, p, zero, .replicated);
    }
    return result;
}
fn asFloats(a: std.mem.Allocator, s: zml.Slice) ![]const f32 {
    if (s.dtype() == .f32) return s.items(f32);
    const out = try a.alloc(f32, s.data().len / 2);
    for (out, 0..) |*v, i| {
        const b = std.mem.readInt(u16, s.data()[i * 2 ..][0..2], .little);
        v.* = @bitCast(@as(u32, b) << 16);
    }
    return out;
}
fn dump(a: std.mem.Allocator, io: std.Io, dir: ?[]const u8, name: []const u8, bytes: []const u8) !void {
    const root = dir orelse return;
    const path = try std.fs.path.join(a, &.{ root, name });
    const f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    var w = f.writer(io, &.{});
    try w.interface.writeAll(bytes);
    try w.interface.flush();
}
fn validate(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    const model = try dir.readFileAlloc(io, "config.json", a, .limited(1024 * 1024));
    const codec_config = try dir.readFileAlloc(io, "audio_tokenizer/config.json", a, .limited(1024 * 1024));
    try @import("config.zig").validate(a, model, codec_config);
}

fn encodeSegment(a: std.mem.Allocator, enc: *zml.tokenizer.Tokenizer.Encoder, text: []const u8) ![]u32 {
    enc.reset();
    const body = try enc.encodeAlloc(a, text);
    const ids = try a.alloc(u32, body.len + 1);
    // IREE returns raw tokens; the reference adds BOS to each independent text segment.
    ids[0] = 2;
    @memcpy(ids[1..], body);
    return ids;
}
