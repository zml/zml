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

const hz_type = main.hz_type;

// TODO : make inference instanciable, so we don't pass as many args,
// and that we can store there all different structs of results to
// compute the prompts sizes and so on more cleanly (only one for eg)

pub const AudioMetadata = struct {
    bpm: []const u8,
    caption: []const u8,
    duration: []const u8,
    genres: []const u8,
    keyscale: []const u8,
    language: []const u8,
    timesignature: []const u8,
    lyric: []const u8,

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
    
    pub fn setDuration(self: *AudioMetadata, allocator: std.mem.Allocator, duration: []const u8) !void {
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
    
    pub fn empty(allocator: std.mem.Allocator) !AudioCodes {
        return .{
            .token_id = try allocator.alloc(u32, 0),
            .string = try allocator.alloc(u8, 0),
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


pub fn runPhase1(raw_prompt: []const u8, zml_handler: *main.Zml_handler, acellm: *acellm_.AceLlm_handler) !AudioMetadata {
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

pub fn runPhase2(metadata: AudioMetadata, zml_handler: *main.Zml_handler, acecfg: *acellm_.AceCfg_handler) !AudioCodes {
    std.log.info("5hz phase II : audio codes generation", .{});
    const cond_tok, const uncond_tok = try tokenizeGenerationPrompt(zml_handler.allocator, acecfg.llm.tokenizer, metadata);
    defer zml_handler.allocator.free(cond_tok);
    defer zml_handler.allocator.free(uncond_tok);

    std.log.debug("5Hz conditional input tokens {any}", .{ cond_tok });
    std.log.debug("5Hz unconditional input tokens {any}", .{ uncond_tok });
    
    const audiocodes, const audiostr = try generateAudioCodes(zml_handler, acecfg, cond_tok, uncond_tok, metadata);
        
    return .{ .token_id = audiocodes, .string = audiostr };
}

pub fn embedTextInputs(zml_handler: *main.Zml_handler, audio_metadata: AudioMetadata, aceemb: *aceemb_.AceEmb_handler) !TextEmbedding {
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

    var system_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer system_prompt.deinit(allocator);
    try system_prompt.appendSlice(allocator, "<|im_start|>system\n");
    try system_prompt.appendSlice(allocator, "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    try system_prompt.appendSlice(allocator, "<|im_end|>\n");

    var user_cond_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer user_cond_prompt.deinit(allocator);
    try user_cond_prompt.appendSlice(allocator, "<|im_start|>user\n");
    try user_cond_prompt.appendSlice(allocator, "# Caption\n");
    try user_cond_prompt.appendSlice(allocator, metadata.caption);
    try user_cond_prompt.appendSlice(allocator, "\n\n");
    try user_cond_prompt.appendSlice(allocator, "# Lyric\n");
    try user_cond_prompt.appendSlice(allocator, metadata.lyric);
    try user_cond_prompt.appendSlice(allocator, "\n<|im_end|>\n");

    var user_uncond_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer user_uncond_prompt.deinit(allocator);
    try user_uncond_prompt.appendSlice(allocator, "<|im_start|>user\n");
    try user_uncond_prompt.appendSlice(allocator, "NO USER INPUT");
    try user_uncond_prompt.appendSlice(allocator, "<|im_end|>\n");

    var assistant_cond_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer assistant_cond_prompt.deinit(allocator);
    try assistant_cond_prompt.appendSlice(allocator, "<|im_start|>assistant\n");
    try assistant_cond_prompt.appendSlice(allocator, "<think>");
    try assistant_cond_prompt.appendSlice(allocator, "\nbpm: ");
    try assistant_cond_prompt.appendSlice(allocator, metadata.bpm);
    try assistant_cond_prompt.appendSlice(allocator, "\nduration: ");
    try assistant_cond_prompt.appendSlice(allocator, metadata.duration);
    try assistant_cond_prompt.appendSlice(allocator, "\nkeyscale: ");
    try assistant_cond_prompt.appendSlice(allocator, metadata.keyscale);
    //try assistant_cond_prompt.appendSlice(allocator, "\nlanguage: ");
    //try assistant_cond_prompt.appendSlice(allocator, metadata.language);
    try assistant_cond_prompt.appendSlice(allocator, "\ntimesignature: ");
    try assistant_cond_prompt.appendSlice(allocator, metadata.timesignature);
    try assistant_cond_prompt.appendSlice(allocator, "\n</think>\n\n");
    
    var assistant_uncond_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer assistant_uncond_prompt.deinit(allocator);
    try assistant_uncond_prompt.appendSlice(allocator, "<|im_start|>assistant\n");
    try assistant_uncond_prompt.appendSlice(allocator, "<think>\n\n");
    try assistant_uncond_prompt.appendSlice(allocator, "</think>\n\n");
    
    var cond_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer cond_prompt.deinit(allocator);
    try cond_prompt.appendSlice(allocator, system_prompt.items);
    try cond_prompt.appendSlice(allocator, user_cond_prompt.items);
    try cond_prompt.appendSlice(allocator, assistant_cond_prompt.items);

    var uncond_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer uncond_prompt.deinit(allocator);
    try uncond_prompt.appendSlice(allocator, system_prompt.items);
    try uncond_prompt.appendSlice(allocator, user_uncond_prompt.items);
    try uncond_prompt.appendSlice(allocator, assistant_uncond_prompt.items);
    
    var cond_tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    var uncond_tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    
    try cond_tokens.appendSlice(allocator, try encoder.encode(cond_prompt.items));
    try uncond_tokens.appendSlice(allocator, try encoder.encode(uncond_prompt.items));

    return .{ try cond_tokens.toOwnedSlice(allocator), try uncond_tokens.toOwnedSlice(allocator) };
}

pub fn tokenizeInputCaption(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, metadata: AudioMetadata) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    
    var formatted_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer formatted_prompt.deinit(allocator);
    
    try formatted_prompt.appendSlice(allocator, "# Instruction\n");
    // TODO
    //try formatted_prompt.appendSlice(allocator, "Fill the audio semantic mask based on the given conditions:\n\n");
    try formatted_prompt.appendSlice(allocator, "Generate audio semantic tokens based on the given conditions:\n\n");
    try formatted_prompt.appendSlice(allocator, "# Caption\n");
    try formatted_prompt.appendSlice(allocator, metadata.caption);
    try formatted_prompt.appendSlice(allocator, "\n\n# Metas\n");
    try formatted_prompt.appendSlice(allocator, "- bpm: ");
    try formatted_prompt.appendSlice(allocator, metadata.bpm);
    try formatted_prompt.appendSlice(allocator, "\n- timesignature: ");
    try formatted_prompt.appendSlice(allocator, metadata.timesignature);
    try formatted_prompt.appendSlice(allocator, "\n- keyscale: ");
    try formatted_prompt.appendSlice(allocator, metadata.keyscale);
    try formatted_prompt.appendSlice(allocator, "\n- duration: ");
    try formatted_prompt.appendSlice(allocator, metadata.duration);
    try formatted_prompt.appendSlice(allocator, "\n<|endoftext|>\n");

    std.log.info("text input\n{s}", .{formatted_prompt.items});
    
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    try tokens.appendSlice(allocator, try encoder.encode(formatted_prompt.items));
    return tokens.toOwnedSlice(allocator);
}

pub fn tokenizeInputLyrics(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, metadata: AudioMetadata) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    
    var formatted_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer formatted_prompt.deinit(allocator);
    
    try formatted_prompt.appendSlice(allocator, "# Languages\n");
    try formatted_prompt.appendSlice(allocator, metadata.language);
    try formatted_prompt.appendSlice(allocator, "\n\n# Lyric\n");
    try formatted_prompt.appendSlice(allocator, metadata.lyric);
    try formatted_prompt.appendSlice(allocator, "<|endoftext|>");

    std.log.info("lyric input\n{s}", .{formatted_prompt.items});
    
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    try tokens.appendSlice(allocator, try encoder.encode(formatted_prompt.items));
    return tokens.toOwnedSlice(allocator);
}


pub fn generateInspirationText(zml_handler: *main.Zml_handler, acellm: *acellm_.AceLlm_handler, prompt_tok: []const u32) ![]u8 {
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

    var prefill_args = try acellm.prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);
    var prefill_results = try acellm.prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);
    var decode_args = try acellm.decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try acellm.decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    var token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer token_slice.free(allocator);
    token_slice.items(u32)[0] = 0;
    var token_buffer: zml.Buffer = try .fromSlice(io, platform, token_slice, sharding);
    defer token_buffer.deinit();

    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ acellm.options.seq_len }, .u32));
    defer prefill_tokens_slice.free(allocator);
    @memcpy(prefill_tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);
    var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer prefill_tokens_buffer.deinit();

    std.log.info("5Hz run prefill with sequence of {d} tokens", .{ prompt_tok.len });
    prefill_args.set(.{ acellm.model_buffers, prefill_tokens_buffer, token_buffer, acellm.kv_cache_buffers, rng_buffers });
    acellm.prefill_exe.call(prefill_args, &prefill_results);
    prefill_results.fill(.{ &prefill_tokens_buffer, &acellm.kv_cache_buffers, &rng_buffers });

    try prefill_tokens_buffer.toSlice(io, prefill_tokens_slice);
    token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[prompt_tok.len - 1];

    std.log.info("5Hz run decode", .{});
    const output_tokens_len = acellm.options.seq_len - prompt_tok.len - 1;
    var num_tokens_generated: usize = 0;
    var result: std.ArrayList(u8) = try .initCapacity(allocator, 0);

    var stdout = std.Io.File.stdout().writer(io, &.{});
    var writer: *std.Io.Writer = &stdout.interface;
    generation: for (0..output_tokens_len + 1) |i| {
        num_tokens_generated += 1;
        const generated_token = token_slice.items(u32)[0];
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try result.appendSlice(allocator, chunk);
            try writer.writeAll(chunk);
            try writer.flush();
        } else {
            std.log.info("ERROR could not decode token: {d}", .{ generated_token });
        }

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
        
        // token buffer was used to pass 0 as token_index to the prefill
        // now it's used to pass the token we want to predict from to the decode
        token_buffer.deinit();
        token_buffer = try .fromSlice(io, platform, token_slice, sharding);
        // we then need a new 1 token buffer to pass the token_index
        token_slice.items(u32)[0] = @intCast(prompt_tok.len + i);
        var decode_token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_slice, sharding);
        defer decode_token_pos_buffer.deinit();

        // call to generate the next token
        decode_args.set(.{ acellm.model_buffers, token_buffer, decode_token_pos_buffer, acellm.kv_cache_buffers, rng_buffers });
        acellm.decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &token_buffer, &acellm.kv_cache_buffers, &rng_buffers });
        // extract the generated token from the buffer
        try token_buffer.toSlice(io, token_slice);
    }
    try writer.writeAll("\n");
    try writer.flush();
    std.log.info("5Hz done, generated {d} tokens", .{ num_tokens_generated });
    return result.toOwnedSlice(allocator);
}

pub fn generateAudioCodes(zml_handler: *main.Zml_handler, acecfg: *acellm_.AceCfg_handler, cond_tok: []const u32, uncond_tok: []const u32, metadata: AudioMetadata) !struct { []u32, []u8 } {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acecfg.params.shardings.replicated;
    const platform = zml_handler.platform;
    
    var tokenizer_decoder = try acecfg.llm.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const nb_audio_codes = 5 * try std.fmt.parseInt(u32, metadata.duration, 10);
    const cond_tot = cond_tok.len + nb_audio_codes;
    const uncond_tot = uncond_tok.len + nb_audio_codes;

    var result_tok: std.ArrayList(u32) = try .initCapacity(allocator, nb_audio_codes);
    var result_str: std.ArrayList(u8) = try .initCapacity(allocator, nb_audio_codes * 15);

    //std.log.info("Call 5Hz model with formatted prompt : conditional", .{});
    //std.log.info("########### prompt start ###########\n{s}", .{try tokenizer_decoder.decode(cond_tok)});
    //std.log.info("###########  prompt end  ###########", .{});
    //std.log.info("Call 5Hz model with formatted prompt : unconditional", .{});
    //std.log.info("########### prompt start ###########\n{s}", .{try tokenizer_decoder.decode(uncond_tok)});
    //std.log.info("###########  prompt end  ###########", .{});

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var prefill_args = try acecfg.prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);
    var prefill_results = try acecfg.prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);
    var decode_args = try acecfg.decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try acecfg.decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    // buffer and slice for a .s = 1, val = 0 tensor
    var zero_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer zero_slice.free(allocator);
    zero_slice.items(u32)[0] = 0;
    var zero_buffer: zml.Buffer = try .fromSlice(io, platform, zero_slice, sharding);
    defer zero_buffer.deinit();
    
    // in prefill, we need to pass the prompt length to the .forward to it knows which logits to use to compute next token
    // this will also be used to get the generated token and pass it to .forward during decode
    var cond_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    var uncond_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer cond_token_slice.free(allocator);
    defer uncond_token_slice.free(allocator);
    cond_token_slice.items(u32)[0] = @intCast(cond_tok.len - 1);
    uncond_token_slice.items(u32)[0] = @intCast(uncond_tok.len - 1);
    var cond_token_buffer: zml.Buffer = try .fromSlice(io, platform, cond_token_slice, sharding);
    var uncond_token_buffer: zml.Buffer = try .fromSlice(io, platform, uncond_token_slice, sharding);
    defer cond_token_buffer.deinit();
    defer uncond_token_buffer.deinit();

    const cond_prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ cond_tot }, .u32));
    const uncond_prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ uncond_tot }, .u32));
    defer cond_prefill_tokens_slice.free(allocator);
    defer uncond_prefill_tokens_slice.free(allocator);
    @memcpy(cond_prefill_tokens_slice.items(u32)[0..cond_tok.len], cond_tok);
    @memcpy(uncond_prefill_tokens_slice.items(u32)[0..uncond_tok.len], uncond_tok);
    var cond_prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, cond_prefill_tokens_slice, sharding);
    var uncond_prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, uncond_prefill_tokens_slice, sharding);
    defer cond_prefill_tokens_buffer.deinit();
    defer uncond_prefill_tokens_buffer.deinit();

    std.log.info("5Hz run CFG prefill with sequence of {d}/{d} tokens", .{ cond_tot, uncond_tot });
    prefill_args.set(.{
        acecfg.llm.model_buffers,
        cond_prefill_tokens_buffer, uncond_prefill_tokens_buffer,
        zero_buffer, zero_buffer,
        cond_token_buffer, uncond_token_buffer,
        acecfg.cond_kv_cache_buffers, acecfg.uncond_kv_cache_buffers,
        rng_buffers,
    });
    acecfg.prefill_exe.callOpts(io, prefill_args, &prefill_results, .{ .wait = true });
    prefill_results.fill(.{ &cond_token_buffer, &uncond_token_buffer, &acecfg.cond_kv_cache_buffers, &acecfg.uncond_kv_cache_buffers, &rng_buffers });

    try cond_token_buffer.toSlice(io, cond_token_slice);
    try uncond_token_buffer.toSlice(io, uncond_token_slice);
    
    std.log.info("5Hz run decode CFG, need {d} audio codes", .{ nb_audio_codes });
    var stdout = std.Io.File.stdout().writer(io, &.{});
    var writer: *std.Io.Writer = &stdout.interface;
    for (0..nb_audio_codes) |i| {
        // collect and print generated sequence
        const generated_token = cond_token_slice.items(u32)[0];
        const chunk = try tokenizer_decoder.decode(&.{ generated_token });
        try result_tok.append(allocator, generated_token);
        try result_str.appendSlice(allocator, chunk);
        try writer.writeAll(chunk);
        try writer.flush();
        if (result_tok.items.len == nb_audio_codes) break;

        // in decode, we need to pass the position of the last generated token to the .forward
        var cond_pos_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
        var uncond_pos_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
        defer cond_pos_slice.free(allocator);
        defer uncond_pos_slice.free(allocator);
        cond_pos_slice.items(u32)[0] = @intCast(cond_tok.len + i);
        uncond_pos_slice.items(u32)[0] = @intCast(uncond_tok.len + i);
        var cond_pos_buffer: zml.Buffer = try .fromSlice(io, platform, cond_pos_slice, sharding);
        var uncond_pos_buffer: zml.Buffer = try .fromSlice(io, platform, uncond_pos_slice, sharding);
        defer cond_pos_buffer.deinit();
        defer uncond_pos_buffer.deinit();
        
        decode_args.set(.{
            acecfg.llm.model_buffers,
            cond_token_buffer, uncond_token_buffer,
            cond_pos_buffer, uncond_pos_buffer,
            zero_buffer, zero_buffer,
            acecfg.cond_kv_cache_buffers, acecfg.uncond_kv_cache_buffers,
            rng_buffers,
        });
        acecfg.decode_exe.callOpts(io, decode_args, &decode_results, .{ .wait = true });
        decode_results.fill(.{ &cond_token_buffer, &uncond_token_buffer, &acecfg.cond_kv_cache_buffers, &acecfg.uncond_kv_cache_buffers, &rng_buffers });
        
        try cond_token_buffer.toSlice(io, cond_token_slice);
        try uncond_token_buffer.toSlice(io, uncond_token_slice);
    }
    try writer.writeAll("\n");
    try writer.flush();
    std.log.info("5Hz CFG done, generated {d} tokens", .{ result_tok.items.len });
    return .{ try result_tok.toOwnedSlice(allocator), try result_str.toOwnedSlice(allocator) };
}

pub fn generateTextEmbedding(zml_handler: *main.Zml_handler, aceemb: *aceemb_.AceEmb_handler, tokens: []const u32, token_embedding: bool) !zml.Slice {
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
    const embedding_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = seq_len, .d = emb_dim }, hz_type));
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

pub fn prepareLatents(zml_handler: *main.Zml_handler, aceenc: *aceenc_.AceEnc_handler, text_emb: TextEmbedding, audio_codes: []u32, target_duration: u32) !InitialLatents {
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

pub fn runDiffusion(zml_handler: *main.Zml_handler, acedit: *acedit_.AceDit_handler, latents: InitialLatents) !DiffusedLatents {
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

pub fn decodeAudioLatents(zml_handler: *main.Zml_handler, acevae: *acevae_.AceVae_handler, latents: DiffusedLatents) !DecodedAudio {
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