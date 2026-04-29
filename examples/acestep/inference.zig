const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const main = @import("main.zig");
const acellm_ = @import("acellm.zig");
const aceemb_ = @import("aceemb.zig");
const aceenc_ = @import("aceenc.zig");
const acedit_ = @import("acedit.zig");
const acevae_ = @import("acevae.zig");

const cfg: f32 = 1.0;
const seed: u128 = 0;
const hz_type = .f32;

pub const AudioMetadata = struct {
    bpm: []const u8,
    caption: []const u8,
    duration: []const u8,
    genres: []const u8,
    keyscale: []const u8,
    language: []const u8,
    timesignature: []const u8,
    lyric: []const u8,
    
    pub fn initExample() AudioMetadata {
        return .{
            .bpm = "N/A",
            .caption = "",
            .duration = "12",
            .genres = "N/A",
            .keyscale = "N/A",
            .language = "unknown",
            .timesignature = "N/A",
            .lyric = "[Instrumental]",
        };
    }

    const field_names = [_][]const u8{
        "bpm:",
        "caption:",
        "duration:",
        "genres:",
        "keyscale:",
        "language:",
        "timesignature:",
        "lyric:",
    };

    fn isFieldStart(line: []const u8) bool {
        const trimmed = std.mem.trimStart(u8, line, " \t");
        for (field_names) |f| {
            if (std.mem.startsWith(u8, trimmed, f)) return true;
        }
        return false;
    }

    fn extractFieldBlock(allocator: std.mem.Allocator, input: []const u8, field: []const u8) ![]const u8 {
        var it = std.mem.splitScalar(u8, input, '\n');
        var found = false;
        var start: usize = 0;
        var end: usize = input.len;
        while (it.next()) |line| {
            const trimmed = std.mem.trimStart(u8, line, " \t");
            if (!found) {
                if (std.mem.startsWith(u8, trimmed, field)) {
                    found = true;
                    // Compute start index
                    const line_offset = @intFromPtr(line.ptr) - @intFromPtr(input.ptr);
                    const trim_offset = @intFromPtr(trimmed.ptr) - @intFromPtr(line.ptr);
                    start = line_offset + trim_offset + field.len;
                }
            } else {
                if (isFieldStart(line)) {
                    end = @intFromPtr(line.ptr) - @intFromPtr(input.ptr);
                    break;
                }
            }
        }
        if (!found) return "";
        const trimmed = std.mem.trim(u8, input[start..end], " \t\r\n");
        const output: []u8 = try allocator.alloc(u8, std.mem.replacementSize(u8, trimmed, "\n  ", " "));
        _ = std.mem.replace(u8, trimmed, "\n  ", " ", output);
        return output;
    }

    pub fn initFromString(allocator: std.mem.Allocator, input: []const u8) !AudioMetadata {
        var it = std.mem.splitSequence(u8, input, "</think>");
        const trimmed = it.first()[7..];
        return .{
            .bpm = try extractFieldBlock(allocator, trimmed, "bpm:"),
            .caption = try extractFieldBlock(allocator, trimmed, "caption:"),
            .duration = try extractFieldBlock(allocator, trimmed, "duration:"),
            .genres = try extractFieldBlock(allocator, trimmed, "genres:"),
            .keyscale = try extractFieldBlock(allocator, trimmed, "keyscale:"),
            .language = try extractFieldBlock(allocator, trimmed, "language:"),
            .timesignature = try extractFieldBlock(allocator, trimmed, "timesignature:"),
            .lyric = try extractFieldBlock(allocator, trimmed, "lyric:"),
        };
    }
    
    pub fn setDuration(self: *AudioMetadata, allocator: std.mem.Allocator, duration: []const u8) void {
        allocator.free(self.duration);
        self.duration = try allocator.dupe(u8, duration);
    }

    pub fn deinit(self: AudioMetadata, allocator: std.mem.Allocator) void {
        allocator.free(self.bpm);
        allocator.free(self.caption);
        allocator.free(self.duration);
        allocator.free(self.genres);
        allocator.free(self.keyscale);
        allocator.free(self.language);
        allocator.free(self.timesignature);
        allocator.free(self.lyric);
    }
};

pub const AudioCodes = struct {
    token_id: []u32,
    string: []u8,
    
    pub fn initExample(allocator: std.mem.Allocator) !AudioCodes {
        const codes = [_]u32{ };
        const str = "";
        
        return .{
            .token_id = try allocator.dupe(u32, &codes),
            .string = try allocator.dupe(u8, str),
        };
    }
    
    pub fn len(self: AudioCodes) u32 {
        return @intCast(self.token_id.len);
    }
    
    pub fn getIntCodes(self: AudioCodes, allocator: std.mem.Allocator) ![]u32 {
        const prefix = "<|audio_code_";
        const suffix = "|>";
        var result: std.ArrayList(u32) = try .initCapacity(allocator, 0);
        errdefer result.deinit(allocator);
        var i: usize = 0;
        while (i < self.string.len) {
            if (!std.mem.startsWith(u8, self.string[i..], prefix)) {
                return error.InvalidFormat;
            }
            i += prefix.len;
            const start = i;
            while (i < self.string.len and std.ascii.isDigit(self.string[i])) : (i += 1) {}
            if (i == start) {
                return error.InvalidFormat;
            }
            const value = try std.fmt.parseInt(u32, self.string[start..i], 10);
            if (!std.mem.startsWith(u8, self.string[i..], suffix)) {
                return error.InvalidFormat;
            }
            i += suffix.len;
            try result.append(allocator, value);
        }
        return result.toOwnedSlice(allocator);
    }
    
    pub fn deinit(self: AudioCodes, allocator: std.mem.Allocator) void {
        allocator.free(self.token_id);
        allocator.free(self.string);
    }
};

pub const TextEmbedding = struct {
    caption_embedding: zml.Slice,
    lyric_embedding: zml.Slice,
    
    pub fn textLen(self: TextEmbedding) u32 {
        return @intCast(self.caption_embedding.shape.dim(.s));
    }
    
    pub fn lyricLen(self: TextEmbedding) u32 {
        return @intCast(self.lyric_embedding.shape.dim(.s));
    }
    
    pub fn print(self: TextEmbedding, io: std.Io) !void {
        var stdout = std.Io.File.stdout().writer(io, &.{});
        const writer: *std.Io.Writer = &stdout.interface;
        const options: std.fmt.Number = .{};
        
        std.log.info("Caption embedding shape", .{});
        try self.caption_embedding.shape.format(writer);
        try self.caption_embedding.prettyPrint(writer, options);
        
        std.log.info("Lyric embedding shape", .{});
        try self.lyric_embedding.shape.format(writer);
        try self.lyric_embedding.prettyPrint(writer, options);
    }
    
    pub fn deinit(self: TextEmbedding, allocator: std.mem.Allocator) void {
        self.caption_embedding.free(allocator);
        self.lyric_embedding.free(allocator);
    }
};

pub const InitialLatents = struct {
    x: zml.Slice,
    context_latents: zml.Slice,
    encoder_conditions: zml.Slice,
    
    pub fn print(self: InitialLatents, io: std.Io) !void {
        var stdout = std.Io.File.stdout().writer(io, &.{});
        const writer: *std.Io.Writer = &stdout.interface;
        const options: std.fmt.Number = .{};
        
        std.log.info("Initial latents shapes", .{});
        try self.x.shape.format(writer);
        try self.context_latents.shape.format(writer);
        try self.encoder_conditions.shape.format(writer);
        try self.x.prettyPrint(writer, options);
        try self.context_latents.prettyPrint(writer, options);
        try self.encoder_conditions.prettyPrint(writer, options);
    }
    
    pub fn deinit(self: InitialLatents, allocator: std.mem.Allocator) void {
        self.x.free(allocator);
        self.context_latents.free(allocator);
        self.encoder_conditions.free(allocator);
    }
};

pub const DiffusedLatents = struct {
    x: zml.Slice,
    
    pub fn print(self: DiffusedLatents, io: std.Io) !void {
        var stdout = std.Io.File.stdout().writer(io, &.{});
        const writer: *std.Io.Writer = &stdout.interface;
        const options: std.fmt.Number = .{};
        
        std.log.info("Encoded text shape", .{});
        try self.x.shape.format(writer);
        try self.x.prettyPrint(writer, options);
    }
    
    pub fn deinit(self: DiffusedLatents, allocator: std.mem.Allocator) void {
        self.x.free(allocator);
    }
};

pub const DecodedAudio = struct {
    audio: zml.Slice,
    
    pub fn deinit(self: DecodedAudio, allocator: std.mem.Allocator) void {
        self.audio.free(allocator);
    }
    
    pub fn print(self: DecodedAudio, io: std.Io) !void {
        var stdout = std.Io.File.stdout().writer(io, &.{});
        const writer: *std.Io.Writer = &stdout.interface;
        const options: std.fmt.Number = .{};
        
        std.log.info("Encoded text shape", .{});
        try self.audio.shape.format(writer);
        try self.audio.prettyPrint(writer, options);
    }
};


pub fn runPhase1(raw_prompt: []const u8, zml_handler: main.Zml_handler, acellm: *acellm_.AceLlm_handler) !AudioMetadata {
    std.log.info("5Hz phase I : inspiration from initial prompt", .{});
    //std.log.info("########### prompt start ###########:\n{s}", .{raw_prompt});
    //std.log.info("###########  prompt end  ###########", .{});
    
    const inspi_tokens = try tokenizeInspirationPrompt(zml_handler.allocator, acellm.tokenizer, raw_prompt);
    defer zml_handler.allocator.free(inspi_tokens);

    const inspi_result = try generateInspirationText(zml_handler, acellm, inspi_tokens);
    defer zml_handler.allocator.free(inspi_result);

    const metadata: AudioMetadata = try .initFromString(zml_handler.allocator, inspi_result);
    return metadata;
}

pub fn runPhase2(metadata: AudioMetadata, zml_handler: main.Zml_handler, acellm: *acellm_.AceLlm_handler) !AudioCodes {
    std.log.info("5hz phase II : audio codes generation", .{});
    const cond_tok, const uncond_tok = try tokenizeGenerationPrompt(zml_handler.allocator, acellm.tokenizer, metadata);
    defer zml_handler.allocator.free(cond_tok);
    defer zml_handler.allocator.free(uncond_tok);

    std.log.debug("5Hz conditional input tokens {any}", .{ cond_tok });
    std.log.debug("5Hz unconditional input tokens {any}", .{ uncond_tok });
    
    try acellm.resetKvCache(zml_handler);

    const audiocodes, const audiostr = try generateAudioCodes(zml_handler, acellm, cond_tok, uncond_tok, metadata);
        
    return .{ .token_id = audiocodes, .string = audiostr };
}

pub fn embedTextInputs(zml_handler: main.Zml_handler, audio_metadata: AudioMetadata, aceemb: *aceemb_.AceEmb_handler) !TextEmbedding {
    std.log.info("EMB start Text Input Embedding", .{});
    
    const caption_tok = try tokenizeInputCaption(zml_handler.allocator, aceemb.tokenizer, audio_metadata);
    const lyric_tok = try tokenizeInputLyrics(zml_handler.allocator, aceemb.tokenizer, audio_metadata);
    defer zml_handler.allocator.free(caption_tok);
    defer zml_handler.allocator.free(lyric_tok);
    
    std.log.info("EMB caption tokens : {d}", .{ caption_tok.len });
    std.log.info("EMB lyrics tokens : {d}", .{ lyric_tok.len} );
    
    const caption_emb = try generateTextEmbedding(zml_handler, aceemb, caption_tok, false);
    const lyric_emb = try generateTextEmbedding(zml_handler, aceemb, lyric_tok, true);
    
    return .{ .caption_embedding = caption_emb, .lyric_embedding = lyric_emb };
}


pub fn tokenizeInspirationPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var formatted_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer formatted_prompt.deinit(allocator);

    try formatted_prompt.appendSlice(allocator, "<|im_start|>system\n");
    try formatted_prompt.appendSlice(allocator, "# Instruction\nExpand the user's input into a more detailed and specific musical description:\n\n");
    try formatted_prompt.appendSlice(allocator, "<|im_end|>");

    try formatted_prompt.appendSlice(allocator, "\n<|im_start|>user\n");
    try formatted_prompt.appendSlice(allocator, prompt);
    try formatted_prompt.appendSlice(allocator, "<|im_end|>\n");

    try formatted_prompt.appendSlice(allocator, "<|im_start|>assistant\n\n");
    
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 10);
    try tokens.appendSlice(allocator, try encoder.encode(formatted_prompt.items));

    return tokens.toOwnedSlice(allocator);
}

pub fn tokenizeGenerationPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, metadata: AudioMetadata) !struct { []u32, []u32 } {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var prompt_before_cot: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer prompt_before_cot.deinit(allocator);
    try prompt_before_cot.appendSlice(allocator, "<|im_start|>system\n");
    try prompt_before_cot.appendSlice(allocator, "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    try prompt_before_cot.appendSlice(allocator, "<|im_end|>\n");
    try prompt_before_cot.appendSlice(allocator, "<|im_start|>user\n");
    try prompt_before_cot.appendSlice(allocator, "# Caption\n");
    try prompt_before_cot.appendSlice(allocator, metadata.caption);
    try prompt_before_cot.appendSlice(allocator, "\n\n");
    try prompt_before_cot.appendSlice(allocator, "# Lyric\n");
    try prompt_before_cot.appendSlice(allocator, "[Instrumental]\n");
    try prompt_before_cot.appendSlice(allocator, "<|im_end|>\n");
    try prompt_before_cot.appendSlice(allocator, "<|im_start|>assistant\n");
    try prompt_before_cot.appendSlice(allocator, "<think>");
    
    var prompt_cot: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer prompt_cot.deinit(allocator);
    try prompt_cot.appendSlice(allocator, "\nbpm: ");
    try prompt_cot.appendSlice(allocator, metadata.bpm);
    try prompt_cot.appendSlice(allocator, "\nduration: ");
    try prompt_cot.appendSlice(allocator, metadata.duration);
    try prompt_cot.appendSlice(allocator, "\nkeyscale: ");
    try prompt_cot.appendSlice(allocator, metadata.keyscale);
    try prompt_cot.appendSlice(allocator, "\nlanguage: ");
    try prompt_cot.appendSlice(allocator, metadata.language);
    try prompt_cot.appendSlice(allocator, "\ntimesignature: ");
    try prompt_cot.appendSlice(allocator, metadata.timesignature);
    
    var prompt_empty_cot: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer prompt_empty_cot.deinit(allocator);
    try prompt_empty_cot.appendSlice(allocator, "\n");
    
    var prompt_after_cot: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer prompt_after_cot.deinit(allocator);
    try prompt_after_cot.appendSlice(allocator, "\n</think>\n\n");
    try prompt_after_cot.appendSlice(allocator, "<|im_end|>\n");

    var cond_tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    var uncond_tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    
    // same before
    try cond_tokens.appendSlice(allocator, try encoder.encode(prompt_before_cot.items));
    try uncond_tokens.appendSlice(allocator, try encoder.encode(prompt_before_cot.items));
    // cond has CoT content, uncond has empty think block
    try cond_tokens.appendSlice(allocator, try encoder.encode(prompt_cot.items));
    try uncond_tokens.appendSlice(allocator, try encoder.encode(prompt_empty_cot.items));
    // same after
    try cond_tokens.appendSlice(allocator, try encoder.encode(prompt_after_cot.items));
    try uncond_tokens.appendSlice(allocator, try encoder.encode(prompt_after_cot.items));
    
    return .{ try cond_tokens.toOwnedSlice(allocator), try uncond_tokens.toOwnedSlice(allocator) };
}

pub fn tokenizeInputCaption(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, metadata: AudioMetadata) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    
    var formatted_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer formatted_prompt.deinit(allocator);
    
    try formatted_prompt.appendSlice(allocator, "# Instruction\n");
    try formatted_prompt.appendSlice(allocator, "Fill the audio semantic mask based on the given conditions:\n\n");
    try formatted_prompt.appendSlice(allocator, "# Caption\n");
    try formatted_prompt.appendSlice(allocator, metadata.caption);
    try formatted_prompt.appendSlice(allocator, "\n# Metas\n");
    try formatted_prompt.appendSlice(allocator, "- bmp: ");
    try formatted_prompt.appendSlice(allocator, metadata.bpm);
    try formatted_prompt.appendSlice(allocator, "\n- timesignature: ");
    try formatted_prompt.appendSlice(allocator, metadata.timesignature);
    try formatted_prompt.appendSlice(allocator, "\n- keyscale: ");
    try formatted_prompt.appendSlice(allocator, metadata.keyscale);
    try formatted_prompt.appendSlice(allocator, "\n- duration: ");
    try formatted_prompt.appendSlice(allocator, metadata.duration);
    try formatted_prompt.appendSlice(allocator, "\n<|endoftext|>\n");
    
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    try tokens.appendSlice(allocator, try encoder.encode(formatted_prompt.items));
    return tokens.toOwnedSlice(allocator);
}

pub fn tokenizeInputLyrics(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, metadata: AudioMetadata) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    
    var formatted_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer formatted_prompt.deinit(allocator);
    
    try formatted_prompt.appendSlice(allocator, "# Language\n");
    try formatted_prompt.appendSlice(allocator, metadata.language);
    try formatted_prompt.appendSlice(allocator, "\n\n# Lyrics\n");
    try formatted_prompt.appendSlice(allocator, metadata.lyric);
    try formatted_prompt.appendSlice(allocator, "<|endoftext|>");
    
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    try tokens.appendSlice(allocator, try encoder.encode(formatted_prompt.items));
    return tokens.toOwnedSlice(allocator);
}


pub fn generateInspirationText(zml_handler: main.Zml_handler, acellm: *acellm_.AceLlm_handler, prompt_tok: []const u32) ![]u8 {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acellm.params.shardings.replicated;
    const platform = zml_handler.platform;
    
    var tokenizer_decoder = try acellm.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    //std.log.info("Call 5Hz model with formatted prompt", .{});
    //std.log.info("########### prompt start ###########\n{s}", .{try tokenizer_decoder.decode(prompt_tok)});
    //std.log.info("###########  prompt end  ###########", .{});

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);

    var prefill_args = try acellm.prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);
    var prefill_results = try acellm.prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    // prepare device buffers for the prefill tokens and their positions
    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ acellm.options.seq_len }, .u32));
    defer prefill_tokens_slice.free(allocator);
    @memcpy(prefill_tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);

    var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer prefill_tokens_buffer.deinit();
    var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, sharding);
    defer prefill_token_pos_buffer.deinit();

    const voc_len = acellm.phase.phase1_mask.len;

    const mask_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    defer mask_slice.free(allocator);
    @memcpy(mask_slice.items(f32)[0..acellm.phase.phase1_mask.len], acellm.phase.phase1_mask);
    var phase_mask_buffer: zml.Buffer = try .fromSlice(io, platform, mask_slice, sharding);
    defer phase_mask_buffer.deinit();

    const pen_value: f32 = 2.0;
    const pos_pen_values: []f32 = try allocator.alloc(f32, voc_len);
    const neg_pen_values: []f32 = try allocator.alloc(f32, voc_len);
    defer allocator.free(pos_pen_values);
    defer allocator.free(neg_pen_values);
    for (0..voc_len) |i| {
        pos_pen_values[i] = 1.0;
        neg_pen_values[i] = 1.0;
    }
    for (0..prompt_tok.len) |i| {
        const token_id = prompt_tok[i];
        pos_pen_values[token_id] /= pen_value;
        neg_pen_values[token_id] *= pen_value;
    }
    const pos_pen_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    const neg_pen_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    defer pos_pen_slice.free(allocator);
    defer neg_pen_slice.free(allocator);
    @memcpy(pos_pen_slice.items(f32)[0..voc_len], pos_pen_values);
    @memcpy(neg_pen_slice.items(f32)[0..voc_len], neg_pen_values);
    var pos_pen_buffer: zml.Buffer = try .fromSlice(io, platform, pos_pen_slice, sharding);
    var neg_pen_buffer: zml.Buffer = try .fromSlice(io, platform, neg_pen_slice, sharding);
    defer pos_pen_buffer.deinit();
    defer neg_pen_buffer.deinit();

    const logits_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .voc = voc_len }, .f32));
    defer logits_slice.free(allocator);
    var logits_buffer: zml.Buffer = try .fromSlice(io, platform, logits_slice, sharding);
    defer logits_buffer.deinit();

    std.log.info("5Hz run prefill with sequence of {d} tokens", .{prompt_tok.len});
    prefill_args.set(.{ acellm.model_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, acellm.kv_cache_buffers, rng_buffers, phase_mask_buffer, pos_pen_buffer, neg_pen_buffer, true });
    acellm.prefill_exe.call(prefill_args, &prefill_results);
    prefill_results.fill(.{ &prefill_tokens_buffer, &acellm.kv_cache_buffers, &rng_buffers, &logits_buffer });

    try logits_buffer.toSlice(io, logits_slice);
    try prefill_tokens_buffer.toSlice(io, prefill_tokens_slice);
    generated_token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[prompt_tok.len - 1];

    // Prepare for token-by-token generation,
    std.log.info("5Hz prepare decode", .{});
    var decode_args = try acellm.decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try acellm.decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    // start with the token generated based on the full prompt.
    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, sharding);
    defer current_token_buffer.deinit();
    const output_tokens_len = acellm.options.seq_len - prompt_tok.len - 1;
    var num_tokens_generated: usize = 0;
    var result: std.ArrayList(u8) = try .initCapacity(allocator, 0);

    std.log.info("5Hz run decode", .{});
    //var stdout = std.Io.File.stdout().writer(io, &.{});
    //var writer: *std.Io.Writer = &stdout.interface;
    generation: for (0..output_tokens_len + 1) |i| {
        // collect and print generated sequence
        num_tokens_generated += 1;
        const generated_token = generated_token_slice.items(u32)[0];
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try result.appendSlice(allocator, chunk);
            //try writer.writeAll(chunk);
            //try writer.flush();
        } else {
            std.log.info("ERROR could not decode token: {d}", .{generated_token});
        }
        pos_pen_values[generated_token] /= pen_value;
        neg_pen_values[generated_token] *= pen_value;
        @memcpy(pos_pen_slice.items(f32)[0..voc_len], pos_pen_values);
        @memcpy(neg_pen_slice.items(f32)[0..voc_len], neg_pen_values);

        // check for eos
        if (i == output_tokens_len) break :generation;
        switch (acellm.config.eos_token_id.value) {
            .int => |eos| if (generated_token == @as(u32, @intCast(eos))) break :generation,
            .ints => |eos_list| {
                for (eos_list) |eos| {
                    if (generated_token == @as(u32, @intCast(eos))) break :generation;
                }
            },
        }
        // current token pos needs to go into a zml.Buffer
        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(prompt_tok.len + i)}));
        var token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice, sharding);
        defer token_pos_buffer.deinit();

        var pos_pen_buff: zml.Buffer = try .fromSlice(io, platform, pos_pen_slice, sharding);
        var neg_pen_buff: zml.Buffer = try .fromSlice(io, platform, neg_pen_slice, sharding);
        defer pos_pen_buff.deinit();
        defer neg_pen_buff.deinit();
        // call to generate the next token
        decode_args.set(.{ acellm.model_buffers, current_token_buffer, token_pos_buffer, acellm.kv_cache_buffers, rng_buffers, phase_mask_buffer, pos_pen_buff, neg_pen_buff, false });
        acellm.decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, &acellm.kv_cache_buffers, &rng_buffers, &logits_buffer });
        // extract the generated token from the buffer
        try logits_buffer.toSlice(io, logits_slice);
        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    std.log.info("5Hz done, generated {d} tokens", .{num_tokens_generated});
    return result.toOwnedSlice(allocator);
}

pub fn generateAudioCodes(zml_handler: main.Zml_handler, acellm: *acellm_.AceLlm_handler, cond_tok: []const u32, uncond_tok: []const u32, metadata: AudioMetadata) !struct { []u32, []u8 } {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acellm.params.shardings.replicated;
    const platform = zml_handler.platform;
    
    var tokenizer_decoder = try acellm.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    //std.log.info("Call 5Hz model with formatted prompt : conditional", .{});
    //std.log.info("########### prompt start ###########\n{s}", .{try tokenizer_decoder.decode(cond_tok)});
    //std.log.info("###########  prompt end  ###########", .{});
    //std.log.info("Call 5Hz model with formatted prompt : unconditional", .{});
    //std.log.info("########### prompt start ###########\n{s}", .{try tokenizer_decoder.decode(uncond_tok)});
    //std.log.info("###########  prompt end  ###########", .{});

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);

    var prefill_args = try acellm.prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);
    var prefill_results = try acellm.prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    // prepare device buffers for the prefill tokens and their positions
    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ acellm.options.seq_len }, .u32));
    defer prefill_tokens_slice.free(allocator);
    var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer prefill_tokens_buffer.deinit();
    var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, sharding);
    defer prefill_token_pos_buffer.deinit();

    const voc_len = acellm.phase.phase1_mask.len;

    const mask_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    defer mask_slice.free(allocator);
    @memcpy(mask_slice.items(f32)[0..acellm.phase.phase1_mask.len], acellm.phase.phase2_mask);
    var phase_mask_buffer: zml.Buffer = try .fromSlice(io, platform, mask_slice, sharding);
    defer phase_mask_buffer.deinit();

    const pen_value: f32 = 1.0;
    const pos_pen_values: []f32 = try allocator.alloc(f32, voc_len);
    const neg_pen_values: []f32 = try allocator.alloc(f32, voc_len);
    defer allocator.free(pos_pen_values);
    defer allocator.free(neg_pen_values);
    for (0..voc_len) |i| {
        pos_pen_values[i] = 1.0;
        neg_pen_values[i] = 1.0;
    }
    for (0..cond_tok.len) |i| {
        const token_id = cond_tok[i];
        pos_pen_values[token_id] /= pen_value;
        neg_pen_values[token_id] *= pen_value;
    }
    const pos_pen_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    const neg_pen_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    defer pos_pen_slice.free(allocator);
    defer neg_pen_slice.free(allocator);
    @memcpy(pos_pen_slice.items(f32)[0..voc_len], pos_pen_values);
    @memcpy(neg_pen_slice.items(f32)[0..voc_len], neg_pen_values);
    var pos_pen_buffer: zml.Buffer = try .fromSlice(io, platform, pos_pen_slice, sharding);
    var neg_pen_buffer: zml.Buffer = try .fromSlice(io, platform, neg_pen_slice, sharding);
    defer pos_pen_buffer.deinit();
    defer neg_pen_buffer.deinit();

    const cond_logits_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .voc = voc_len }, .f32));
    const uncond_logits_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .voc = voc_len }, .f32));
    defer cond_logits_slice.free(allocator);
    defer uncond_logits_slice.free(allocator);
    var cond_logits_buffer: zml.Buffer = try .fromSlice(io, platform, cond_logits_slice, sharding);
    var uncond_logits_buffer: zml.Buffer = try .fromSlice(io, platform, uncond_logits_slice, sharding);
    defer cond_logits_buffer.deinit();
    defer uncond_logits_buffer.deinit();

    std.log.info("5Hz run cond prefill with sequence of {d} tokens", .{ cond_tok.len });
    @memcpy(prefill_tokens_slice.items(u32)[0..cond_tok.len], cond_tok);
    prefill_args.set(.{ acellm.model_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, acellm.kv_cache_buffers, rng_buffers, phase_mask_buffer, pos_pen_buffer, neg_pen_buffer, true });
    acellm.prefill_exe.call(prefill_args, &prefill_results);
    prefill_results.fill(.{ &prefill_tokens_buffer, &acellm.kv_cache_buffers, &rng_buffers, &cond_logits_buffer });
    
    std.log.info("5Hz run uncond prefill with sequence of {d} tokens", .{ uncond_tok.len });
    @memcpy(prefill_tokens_slice.items(u32)[0..uncond_tok.len], uncond_tok);
    prefill_args.set(.{ acellm.model_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, acellm.kv_cache_buffers_cfg, rng_buffers, phase_mask_buffer, pos_pen_buffer, neg_pen_buffer, true });
    acellm.prefill_exe.call(prefill_args, &prefill_results);
    prefill_results.fill(.{ &prefill_tokens_buffer, &acellm.kv_cache_buffers_cfg, &rng_buffers, &uncond_logits_buffer });

    // combine logits
    try cond_logits_buffer.toSlice(io, cond_logits_slice);
    try uncond_logits_buffer.toSlice(io, uncond_logits_slice);
    var clo: []f32 = cond_logits_slice.items(f32);
    var ulo: []f32 = uncond_logits_slice.items(f32);
    var max_logit: f32 = -1e20;
    var argmax_logit: u32 = 0;
    for (0..clo.len) |i| {
        clo[i] = ulo[i] + cfg * (clo[i] - ulo[i]);
        if (clo[i] > max_logit) {
            max_logit = clo[i];
            argmax_logit = @intCast(i);
        }
    }   
    generated_token_slice.items(u32)[0] = argmax_logit;

    // Prepare for token-by-token generation,
    std.log.info("5Hz prepare decode", .{});
    var decode_args = try acellm.decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try acellm.decode_exe.results(allocator);
    defer decode_results.deinit(allocator);
    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, sharding);
    defer current_token_buffer.deinit();
    const nb_audio_codes = 5 * try std.fmt.parseInt(u32, metadata.duration, 10);
    const max_output_tokens = acellm.options.seq_len - cond_tok.len;
    var result_tok: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    var result_str: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    std.log.info("5Hz run decode, need {d} audio codes", .{ nb_audio_codes });
    //var stdout = std.Io.File.stdout().writer(io, &.{});
    //var writer: *std.Io.Writer = &stdout.interface;
    for (0..max_output_tokens) |i| {
        // collect and print generated sequence
        const generated_token = generated_token_slice.items(u32)[0];
        const chunk = try tokenizer_decoder.decode(&.{ generated_token });
        try result_tok.append(allocator, generated_token);
        try result_str.appendSlice(allocator, chunk);
        //try writer.writeAll(chunk);
        //try writer.flush();
        if (result_tok.items.len == nb_audio_codes) break;
        
        // update penalty values
        pos_pen_values[generated_token] /= pen_value;
        neg_pen_values[generated_token] *= pen_value;
        @memcpy(pos_pen_slice.items(f32)[0..voc_len], pos_pen_values);
        @memcpy(neg_pen_slice.items(f32)[0..voc_len], neg_pen_values);

        // current token pos needs to go into a zml.Buffer
        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(cond_tok.len + i)}));
        var token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice, sharding);
        defer token_pos_buffer.deinit();

        var pos_pen_buff: zml.Buffer = try .fromSlice(io, platform, pos_pen_slice, sharding);
        var neg_pen_buff: zml.Buffer = try .fromSlice(io, platform, neg_pen_slice, sharding);
        defer pos_pen_buff.deinit();
        defer neg_pen_buff.deinit();
        
        // call to generate the next cond logits
        decode_args.set(.{ acellm.model_buffers, current_token_buffer, token_pos_buffer, acellm.kv_cache_buffers, rng_buffers, phase_mask_buffer, pos_pen_buff, neg_pen_buff, false });
        acellm.decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, &acellm.kv_cache_buffers, &rng_buffers, &cond_logits_buffer });
        // call to generate the next uncond logits
        decode_args.set(.{ acellm.model_buffers, current_token_buffer, token_pos_buffer, acellm.kv_cache_buffers_cfg, rng_buffers, phase_mask_buffer, pos_pen_buff, neg_pen_buff, true });
        acellm.decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, &acellm.kv_cache_buffers_cfg, &rng_buffers, &uncond_logits_buffer });
        
        // combine logits
        try cond_logits_buffer.toSlice(io, cond_logits_slice);
        try uncond_logits_buffer.toSlice(io, uncond_logits_slice);
        clo = cond_logits_slice.items(f32);
        ulo = uncond_logits_slice.items(f32);
        max_logit = -1e20;
        argmax_logit = 0;
        for (0..clo.len) |ii| {
            clo[ii] = ulo[ii] + cfg * (clo[ii] - ulo[ii]);
            if (clo[ii] > max_logit) {
                max_logit = clo[ii];
                argmax_logit = @intCast(ii);
            }
        }
        generated_token_slice.items(u32)[0] = argmax_logit;
    }
    std.log.info("5Hz done, generated {d} tokens", .{ result_tok.items.len });
    return .{ try result_tok.toOwnedSlice(allocator), try result_str.toOwnedSlice(allocator) };
}

pub fn generateTextEmbedding(zml_handler: main.Zml_handler, aceemb: *aceemb_.AceEmb_handler, tokens: []const u32, token_embedding: bool) !zml.Slice {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = aceemb.params.shardings.replicated;
    const platform = zml_handler.platform;
    
    var tokenizer_decoder = try aceemb.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    //std.log.info("Call Qwen Emb model with formatted prompt", .{});
    //std.log.info("########### prompt start ###########\n{s}", .{try tokenizer_decoder.decode(tokens)});
    //std.log.info("###########  prompt end  ###########", .{});
    //std.log.info("Basic token embedding mode: {any}", .{token_embedding});
    
    const seq_len = tokens.len;
    const emb_dim = aceemb.config.hidden_size;
    
    // the result embeddings we return
    const embedding_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = seq_len, .d = emb_dim }, .f32));
    var embedding_buffer: zml.Buffer = try .fromSlice(io, platform, embedding_slice, sharding);
    defer embedding_buffer.deinit();

    const tokens_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = seq_len }, .u32));
    defer tokens_slice.free(allocator);
    @memcpy(tokens_slice.items(u32)[0..tokens.len], tokens);
    var tokens_buffer: zml.Buffer = try .fromSlice(io, platform, tokens_slice, sharding);
    defer tokens_buffer.deinit();

    const exe = if (token_embedding) aceemb.partial_embed_exe else aceemb.full_embed_exe;
    var embed_args = try exe.args(allocator);
    defer embed_args.deinit(allocator);
    var embed_results = try exe.results(allocator);
    defer embed_results.deinit(allocator);
    embed_args.set(.{ aceemb.model_buffers, tokens_buffer });
    exe.call(embed_args, &embed_results);
    
    embed_results.fill(.{ &embedding_buffer });
    std.log.info("EMB done embedding", .{});    

    try embedding_buffer.toSlice(io, embedding_slice);
    return embedding_slice;
}

pub fn prepareLatents(zml_handler: main.Zml_handler, aceenc: *aceenc_.AceEnc_handler, text_emb: TextEmbedding, audio_codes: []u32, target_duration: u32) !InitialLatents {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = aceenc.shardings.replicated;
    const platform = zml_handler.platform;

    const caption_len = text_emb.caption_embedding.shape.dim(0);
    const lyric_len = text_emb.lyric_embedding.shape.dim(0);
    const emb_dim = aceenc.config.hidden_size;
    const audio_dim = aceenc.config.timbre_hidden_dim;
    const t_timbre = aceenc.config.timbre_fix_frame;
    const t_25hz: i64 = 25 * target_duration;
    const s_enc = caption_len + lyric_len + 1;
    
    std.log.info("ENC generate timbre reference of length {d} and {d} from silence latent", .{ t_timbre, t_25hz });
    
    const timbre_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_dim, .t = t_timbre }, .f32));
    defer timbre_slice.free(allocator);
    var timbre_buffer: zml.Buffer = try .fromSlice(io, platform, timbre_slice, sharding);
    defer timbre_buffer.deinit();
    
    const src_audio_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_dim, .t = t_25hz }, .f32));
    defer src_audio_slice.free(allocator);
    var src_audio_buffer: zml.Buffer = try .fromSlice(io, platform, src_audio_slice, sharding);
    defer src_audio_buffer.deinit();
    
    var silence_args = try aceenc.silence_exe.args(allocator);
    defer silence_args.deinit(allocator);
    var silence_results = try aceenc.silence_exe.results(allocator);
    defer silence_results.deinit(allocator);
    
    silence_args.set(.{ aceenc.silence_buffers });
    aceenc.silence_exe.call(silence_args, &silence_results);
    silence_results.fill(.{ &timbre_buffer, &src_audio_buffer });
    
    std.log.info("ENC call encoder cap={d}x{d} lyr={d}x{d} tim={d}x{d} audioc={d} src={d}x{d}", .{
        caption_len, emb_dim,
        lyric_len, emb_dim,
        audio_dim, t_timbre,
        audio_codes.len,
        audio_dim, t_25hz,
    });
    
    // the result latents we return
    const x_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .t = t_25hz, .a = audio_dim }, hz_type));
    const context_latents_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .t = t_25hz, .a = 2 * audio_dim }, hz_type));
    const encoded_conditions_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s_enc = s_enc, .d = emb_dim }, hz_type));
    var x_buffer: zml.Buffer = try .fromSlice(io, platform, x_slice, sharding);
    var context_latents_buffer: zml.Buffer = try .fromSlice(io, platform, context_latents_slice, sharding);
    var encoded_conditions_buffer: zml.Buffer = try .fromSlice(io, platform, encoded_conditions_slice, sharding);
    defer x_buffer.deinit();
    defer context_latents_buffer.deinit();
    defer encoded_conditions_buffer.deinit();
    
    var text_emb_buffer: zml.Buffer = try .fromSlice(io, platform, text_emb.caption_embedding, sharding);
    var lyric_emb_buffer: zml.Buffer = try .fromSlice(io, platform, text_emb.lyric_embedding, sharding);
    defer text_emb_buffer.deinit();
    defer lyric_emb_buffer.deinit();
    
    const audio_codes_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = audio_codes.len }, .u32));
    defer audio_codes_slice.free(allocator);
    @memcpy(audio_codes_slice.items(u32)[0..audio_codes.len], audio_codes);
    var audio_codes_buffer: zml.Buffer = try .fromSlice(io, platform, audio_codes_slice, sharding);
    defer audio_codes_buffer.deinit();
    
    var encode_args = try aceenc.encode_exe.args(allocator);
    defer encode_args.deinit(allocator);
    var encode_results = try aceenc.encode_exe.results(allocator);
    defer encode_results.deinit(allocator);
    encode_args.set(.{ aceenc.model_buffers, text_emb_buffer, lyric_emb_buffer, timbre_buffer, audio_codes_buffer, src_audio_buffer });
    aceenc.encode_exe.call(encode_args, &encode_results);
    encode_results.fill(.{ &x_buffer, &context_latents_buffer, &encoded_conditions_buffer });
    std.log.info("ENC done encoding, output shape : context={d}x{d} conditions={d}x{d}", .{ 2 * audio_dim, t_25hz, s_enc, emb_dim });

    try x_buffer.toSlice(io, x_slice);
    try context_latents_buffer.toSlice(io, context_latents_slice);
    try encoded_conditions_buffer.toSlice(io, encoded_conditions_slice);
        
    return .{
        .x = x_slice,
        .context_latents = context_latents_slice,
        .encoder_conditions = encoded_conditions_slice,
    };
}

pub fn runDiffusion(zml_handler: main.Zml_handler, acedit: *acedit_.AceDit_handler, latents: InitialLatents) !DiffusedLatents {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acedit.shardings.replicated;
    const platform = zml_handler.platform;

    const audio_dim = acedit.config.timbre_hidden_dim;
    const t_25hz = latents.context_latents.shape.dim(0);

    std.log.info("DiT call with input size : {d}x{d} {d}x{d}", .{ t_25hz, audio_dim, latents.encoder_conditions.shape.dim(0), latents.encoder_conditions.shape.dim(1) });
    
    var diffuse_args = try acedit.diffuse_exe.args(allocator);
    defer diffuse_args.deinit(allocator);
    var diffuse_results = try acedit.diffuse_exe.results(allocator);
    defer diffuse_results.deinit(allocator);
    
    // prepare arguments buffers
    var x_buffer: zml.Buffer = try .fromSlice(io, platform, latents.x, sharding);
    var context_latents_buffer: zml.Buffer = try .fromSlice(io, platform, latents.context_latents, sharding);
    var encoded_conditions_buffer: zml.Buffer = try .fromSlice(io, platform, latents.encoder_conditions, sharding);
    defer x_buffer.deinit();
    defer context_latents_buffer.deinit();
    defer encoded_conditions_buffer.deinit();
    
    const result_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_dim, .t = t_25hz }, .f32));
    var result_buffer: zml.Buffer = try .fromSlice(io, platform, result_slice, sharding);
    defer result_buffer.deinit();
    
    // the full forward pass on the dit model is one iteration of the denoising
    const timestamps: [9]f32 = .{ 1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3, 0.0 };
    const steps = timestamps.len - 1;
    for (0..steps) |i| {
        std.log.info("DiT ************* step {d}", .{ i });
        var t_curr: zml.Buffer = try zml.Buffer.scalar(io, platform, timestamps[i], .f32, sharding);
        var t_next: zml.Buffer = try zml.Buffer.scalar(io, platform, timestamps[i + 1], .f32, sharding);
        defer t_curr.deinit();
        defer t_next.deinit();
        diffuse_args.set(.{ acedit.model_buffers, t_curr, t_next, x_buffer, context_latents_buffer, encoded_conditions_buffer });
        acedit.diffuse_exe.callOpts(io, diffuse_args, &diffuse_results, .{ .wait = true });
        diffuse_results.fill(.{ &x_buffer, &result_buffer });
    }
    
    try result_buffer.toSlice(io, result_slice);
    return .{ .x = result_slice };    
}

pub fn decodeAudioLatents(zml_handler: main.Zml_handler, acevae: *acevae_.AceVae_handler, latents: DiffusedLatents) !DecodedAudio {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acevae.shardings.replicated;
    const platform = zml_handler.platform;
        
    const t_25hz = latents.x.shape.dim(1);
    var t_48khz = t_25hz;
    for (acevae.config.downsampling_ratios) |ratio| {
        t_48khz *= ratio;
    }
    const audio_dim = latents.x.shape.dim(0);
    
    std.log.info("VAE call decode with input size : {d}x{d}", .{ audio_dim, t_25hz });
    
    // the result latents we return
    const audio_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = 2, .t = t_48khz }, hz_type));
    var audio_buffer: zml.Buffer = try .fromSlice(io, platform, audio_slice, sharding);
    defer audio_buffer.deinit();
    
    var latent_buffer: zml.Buffer = try .fromSlice(io, platform, latents.x, sharding);
    defer latent_buffer.deinit();
    
    var decode_args = try acevae.decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try acevae.decode_exe.results(allocator);
    defer decode_results.deinit(allocator);
    decode_args.set(.{ acevae.model_buffers, latent_buffer });
    acevae.decode_exe.callOpts(io, decode_args, &decode_results, .{ .wait = true });
    decode_results.fill(.{ &audio_buffer });
    std.log.info("VAE done decoding, output shape : {d}x{d}", .{ 2, t_48khz });
    
    try audio_buffer.toSlice(io, audio_slice);
        
    return .{ .audio = audio_slice };
}