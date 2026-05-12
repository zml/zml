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

    pub fn setDuration(self: *AudioMetadata, allocator: std.mem.Allocator, duration: i64) !void {
        const max_len = 20;
        var buf: [max_len]u8 = undefined;
        const numAsString = try std.fmt.bufPrint(&buf, "{}", .{ duration });
        allocator.free(self.duration);
        self.duration = try allocator.dupe(u8, numAsString);
    }

    pub fn duration_s(self: *AudioMetadata) !u32 {
        return try std.fmt.parseUnsigned(u32, self.duration, 10);
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


pub fn tokenizeInspirationPrompt(zml_handler: *main.Zml_handler, tokenizer: zml.tokenizer.Tokenizer) ![]u32 {
    const allocator = zml_handler.allocator;
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var formatted_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer formatted_prompt.deinit(allocator);

    try formatted_prompt.appendSlice(allocator, "<|im_start|>system\n");
    try formatted_prompt.appendSlice(allocator, "# Instruction\nExpand the user's input into a more detailed and specific musical description:\n\n");
    try formatted_prompt.appendSlice(allocator, "<|im_end|>");

    try formatted_prompt.appendSlice(allocator, "\n<|im_start|>user\n");
    try formatted_prompt.appendSlice(allocator, zml_handler.args.prompt);
    if (zml_handler.args.instru) {
        try formatted_prompt.appendSlice(allocator, "\n\ninstrumental=true");
    }
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

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var zero_buffer: zml.Buffer = try .scalar(io, platform, 0, .u32, sharding);
    defer zero_buffer.deinit();
    var prompt_buffer: zml.Buffer = try .scalar(io, platform, prompt_tok.len - 1, .u32, sharding);
    defer prompt_buffer.deinit();

    var token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer token_slice.free(allocator);
    var token_buffer: zml.Buffer = try .fromSlice(io, platform, token_slice, sharding);
    defer token_buffer.deinit();
    var token_embed_buffer: zml.Buffer = undefined;
    defer token_embed_buffer.deinit();
    var logits_buffer: zml.Buffer = undefined;
    defer logits_buffer.deinit();

    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ acellm.options.seq_len }, .u32));
    defer prefill_tokens_slice.free(allocator);
    @memcpy(prefill_tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);
    var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer prefill_tokens_buffer.deinit();
    var prefill_embed_buffer: zml.Buffer = undefined;
    defer prefill_embed_buffer.deinit();

    const layer_index_slices = try allocator.alloc(zml.Slice, acellm.config.num_hidden_layers);
    defer {
        for (layer_index_slices) |*s| s.free(allocator);
        allocator.free(layer_index_slices);
    }
    for (0..acellm.config.num_hidden_layers) |i| {
        layer_index_slices[i] = try zml.Slice.alloc(allocator, .init(.{}, .u32));
        layer_index_slices[i].items(u32)[0] = @intCast(i);
    }
    const layer_index_buffers = try allocator.alloc(zml.Buffer, acellm.config.num_hidden_layers);
    defer {
        for (layer_index_buffers) |*s| s.deinit();
        allocator.free(layer_index_buffers);
    }
    for (0..acellm.config.num_hidden_layers) |i| {
        layer_index_buffers[i] = try zml.Buffer.fromSlice(io, platform, layer_index_slices[i], sharding);
    }

    std.log.info("5Hz run prefill with seq_len/prompt_len of {d}/{d} tokens", .{ acellm.options.seq_len, prompt_tok.len });
    zml_handler.tic(&zml_handler.timers.llm.prefill);

    acellm.exes.prefill_embed_args.set(.{ acellm.model_buffers, prefill_tokens_buffer });
    acellm.exes.prefill_embed_exe.callOpts(io, acellm.exes.prefill_embed_args, &acellm.exes.prefill_embed_results, .{ .wait = true });
    acellm.exes.prefill_embed_results.fill(.{ &prefill_embed_buffer });

    for (0..acellm.config.num_hidden_layers) |i| {
        acellm.exes.prefill_layer_args.set(.{ acellm.model_buffers.layers[i], prefill_embed_buffer, zero_buffer, acellm.kv_cache_buffers, layer_index_buffers[i] });
        acellm.exes.prefill_layer_exe.callOpts(io, acellm.exes.prefill_layer_args, &acellm.exes.prefill_layer_results, .{ .wait = true });
        acellm.exes.prefill_layer_results.fill(.{ &prefill_embed_buffer, &acellm.kv_cache_buffers });
    }
    acellm.exes.prefill_select_args.set(.{ acellm.model_buffers, prefill_embed_buffer, prompt_buffer });
    acellm.exes.prefill_select_exe.callOpts(io, acellm.exes.prefill_select_args, &acellm.exes.prefill_select_results, .{ .wait = true });
    acellm.exes.prefill_select_results.fill(.{ &token_embed_buffer });

    acellm.exes.logits_args.set(.{ acellm.model_buffers, token_embed_buffer });
    acellm.exes.logits_exe.callOpts(io, acellm.exes.logits_args, &acellm.exes.logits_results, .{ .wait = true });
    acellm.exes.logits_results.fill(.{ &logits_buffer });

    acellm.exes.sample_args.set(.{ acellm.model_buffers, logits_buffer, rng_buffers });
    acellm.exes.sample_exe.callOpts(io, acellm.exes.sample_args, &acellm.exes.sample_results, .{ .wait = true });
    acellm.exes.sample_results.fill(.{ &token_buffer, &rng_buffers });

    try token_buffer.toSlice(io, token_slice);
    zml_handler.toc(&zml_handler.timers.llm.prefill);

    std.log.info("5Hz run decode", .{});
    zml_handler.tic(&zml_handler.timers.llm.decode);

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
        if (i == output_tokens_len) break :generation;
        switch (acellm.config.eos_token_id.value) {
            .int => |eos| if (generated_token == @as(u32, @intCast(eos))) break :generation,
            .ints => |eos_list| {
                for (eos_list) |eos| {
                    if (generated_token == @as(u32, @intCast(eos))) break :generation;
                }
            },
        }

        // we need a new 1 token buffer to pass the token_index
        var pos_buffer: zml.Buffer = try .scalar(io, platform, prompt_tok.len + i, .u32, sharding);
        defer pos_buffer.deinit();

        // call to generate the next token
        acellm.exes.decode_embed_args.set(.{ acellm.model_buffers, token_buffer });
        acellm.exes.decode_embed_exe.callOpts(io, acellm.exes.decode_embed_args, &acellm.exes.decode_embed_results, .{ .wait = true });
        acellm.exes.decode_embed_results.fill(.{ &token_embed_buffer });
        for (0..acellm.config.num_hidden_layers) |ii| {
            acellm.exes.decode_layer_args.set(.{ acellm.model_buffers.layers[ii], token_embed_buffer, pos_buffer, acellm.kv_cache_buffers, layer_index_buffers[ii] });
            acellm.exes.decode_layer_exe.callOpts(io, acellm.exes.decode_layer_args, &acellm.exes.decode_layer_results, .{ .wait = true });
            acellm.exes.decode_layer_results.fill(.{ &token_embed_buffer, &acellm.kv_cache_buffers });
        }
        acellm.exes.logits_args.set(.{ acellm.model_buffers, token_embed_buffer });
        acellm.exes.logits_exe.callOpts(io, acellm.exes.logits_args, &acellm.exes.logits_results, .{ .wait = true });
        acellm.exes.logits_results.fill(.{ &logits_buffer });
        acellm.exes.sample_args.set(.{ acellm.model_buffers, logits_buffer, rng_buffers });
        acellm.exes.sample_exe.callOpts(io, acellm.exes.sample_args, &acellm.exes.sample_results, .{ .wait = true });
        acellm.exes.sample_results.fill(.{ &token_buffer, &rng_buffers });

        // extract the generated token from the buffer
        try token_buffer.toSlice(io, token_slice);
    }
    try writer.writeAll("\n");
    try writer.flush();
    std.log.info("5Hz done, generated {d} tokens", .{ num_tokens_generated });
    zml_handler.toc(&zml_handler.timers.llm.decode);
    return result.toOwnedSlice(allocator);
}

pub fn generateAudioCodes(zml_handler: *main.Zml_handler, acecfg: *acellm_.AceCfg_handler, cond_tok: []const u32, uncond_tok: []const u32, metadata: AudioMetadata) !AudioCodes {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acecfg.params.shardings.replicated;
    const platform = zml_handler.platform;

    var tokenizer_decoder = try acecfg.llm.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const nb_audio_codes = 5 * try std.fmt.parseInt(u32, metadata.duration, 10);

    var result_tok: std.ArrayList(u32) = try .initCapacity(allocator, nb_audio_codes);
    var result_str: std.ArrayList(u8) = try .initCapacity(allocator, nb_audio_codes * 15);

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var zero_buffer: zml.Buffer = try .scalar(io, platform, 0, .u32, sharding);
    defer zero_buffer.deinit();
    var cond_prompt_buffer: zml.Buffer = try .scalar(io, platform, cond_tok.len - 1, .u32, sharding);
    defer cond_prompt_buffer.deinit();
    var uncond_prompt_buffer: zml.Buffer = try .scalar(io, platform, uncond_tok.len - 1, .u32, sharding);
    defer uncond_prompt_buffer.deinit();

    var token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer token_slice.free(allocator);
    var token_buffer: zml.Buffer = try .fromSlice(io, platform, token_slice, sharding);
    defer token_buffer.deinit();
    var cond_embed_buffer: zml.Buffer = undefined;
    defer cond_embed_buffer.deinit();
    var uncond_embed_buffer: zml.Buffer = undefined;
    defer uncond_embed_buffer.deinit();
    var cond_logits_buffer: zml.Buffer = undefined;
    defer cond_logits_buffer.deinit();
    var uncond_logits_buffer: zml.Buffer = undefined;
    defer uncond_logits_buffer.deinit();

    const cond_prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ acecfg.llm.options.seq_len }, .u32));
    const uncond_prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ acecfg.llm.options.seq_len }, .u32));
    defer cond_prefill_tokens_slice.free(allocator);
    defer uncond_prefill_tokens_slice.free(allocator);
    @memcpy(cond_prefill_tokens_slice.items(u32)[0..cond_tok.len], cond_tok);
    @memcpy(uncond_prefill_tokens_slice.items(u32)[0..uncond_tok.len], uncond_tok);
    var cond_prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, cond_prefill_tokens_slice, sharding);
    var uncond_prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, uncond_prefill_tokens_slice, sharding);
    defer cond_prefill_tokens_buffer.deinit();
    defer uncond_prefill_tokens_buffer.deinit();
    var cond_prefill_embed_buffer: zml.Buffer = undefined;
    defer cond_prefill_embed_buffer.deinit();
    var uncond_prefill_embed_buffer: zml.Buffer = undefined;
    defer uncond_prefill_embed_buffer.deinit();

    const layer_index_slices = try allocator.alloc(zml.Slice, acecfg.llm.config.num_hidden_layers);
    defer {
        for (layer_index_slices) |*s| s.free(allocator);
        allocator.free(layer_index_slices);
    }
    for (0..acecfg.llm.config.num_hidden_layers) |i| {
        layer_index_slices[i] = try zml.Slice.alloc(allocator, .init(.{}, .u32));
        layer_index_slices[i].items(u32)[0] = @intCast(i);
    }
    const layer_index_buffers = try allocator.alloc(zml.Buffer, acecfg.llm.config.num_hidden_layers);
    defer {
        for (layer_index_buffers) |*s| s.deinit();
        allocator.free(layer_index_buffers);
    }
    for (0..acecfg.llm.config.num_hidden_layers) |i| {
        layer_index_buffers[i] = try zml.Buffer.fromSlice(io, platform, layer_index_slices[i], sharding);
    }

    std.log.info("5Hz run CFG prefill", .{});
    zml_handler.tic(&zml_handler.timers.cfg.prefill);

    // embed cond tokens
    acecfg.llm.exes.prefill_embed_args.set(.{ acecfg.llm.model_buffers, cond_prefill_tokens_buffer });
    acecfg.llm.exes.prefill_embed_exe.callOpts(io, acecfg.llm.exes.prefill_embed_args, &acecfg.llm.exes.prefill_embed_results, .{ .wait = true });
    acecfg.llm.exes.prefill_embed_results.fill(.{ &cond_prefill_embed_buffer });
    // embed uncond tokens
    acecfg.llm.exes.prefill_embed_args.set(.{ acecfg.llm.model_buffers, uncond_prefill_tokens_buffer });
    acecfg.llm.exes.prefill_embed_exe.callOpts(io, acecfg.llm.exes.prefill_embed_args, &acecfg.llm.exes.prefill_embed_results, .{ .wait = true });
    acecfg.llm.exes.prefill_embed_results.fill(.{ &uncond_prefill_embed_buffer });
    for (0..acecfg.llm.config.num_hidden_layers) |i| {
        // layer i the cond embeds
        acecfg.llm.exes.prefill_layer_args.set(.{ acecfg.llm.model_buffers.layers[i], cond_prefill_embed_buffer, zero_buffer, acecfg.cond_kv_cache_buffers, layer_index_buffers[i] });
        acecfg.llm.exes.prefill_layer_exe.callOpts(io, acecfg.llm.exes.prefill_layer_args, &acecfg.llm.exes.prefill_layer_results, .{ .wait = true });
        acecfg.llm.exes.prefill_layer_results.fill(.{ &cond_prefill_embed_buffer, &acecfg.cond_kv_cache_buffers });
        // layer i the uncond embeds
        acecfg.llm.exes.prefill_layer_args.set(.{ acecfg.llm.model_buffers.layers[i], uncond_prefill_embed_buffer, zero_buffer, acecfg.uncond_kv_cache_buffers, layer_index_buffers[i] });
        acecfg.llm.exes.prefill_layer_exe.callOpts(io, acecfg.llm.exes.prefill_layer_args, &acecfg.llm.exes.prefill_layer_results, .{ .wait = true });
        acecfg.llm.exes.prefill_layer_results.fill(.{ &uncond_prefill_embed_buffer, &acecfg.uncond_kv_cache_buffers });
    }
    // select cond embed
    acecfg.llm.exes.prefill_select_args.set(.{ acecfg.llm.model_buffers, cond_prefill_embed_buffer, cond_prompt_buffer });
    acecfg.llm.exes.prefill_select_exe.callOpts(io, acecfg.llm.exes.prefill_select_args, &acecfg.llm.exes.prefill_select_results, .{ .wait = true });
    acecfg.llm.exes.prefill_select_results.fill(.{ &cond_embed_buffer });
    // select uncond embed
    acecfg.llm.exes.prefill_select_args.set(.{ acecfg.llm.model_buffers, uncond_prefill_embed_buffer, uncond_prompt_buffer });
    acecfg.llm.exes.prefill_select_exe.callOpts(io, acecfg.llm.exes.prefill_select_args, &acecfg.llm.exes.prefill_select_results, .{ .wait = true });
    acecfg.llm.exes.prefill_select_results.fill(.{ &uncond_embed_buffer });
    // compute cond logits
    acecfg.llm.exes.logits_args.set(.{ acecfg.llm.model_buffers, cond_embed_buffer });
    acecfg.llm.exes.logits_exe.callOpts(io, acecfg.llm.exes.logits_args, &acecfg.llm.exes.logits_results, .{ .wait = true });
    acecfg.llm.exes.logits_results.fill(.{ &cond_logits_buffer });
    // compute uncond logits
    acecfg.llm.exes.logits_args.set(.{ acecfg.llm.model_buffers, uncond_embed_buffer });
    acecfg.llm.exes.logits_exe.callOpts(io, acecfg.llm.exes.logits_args, &acecfg.llm.exes.logits_results, .{ .wait = true });
    acecfg.llm.exes.logits_results.fill(.{ &uncond_logits_buffer });
    // combine them with cfg
    acecfg.exes.cfg_args.set(.{ acecfg.llm.model_buffers, cond_logits_buffer, uncond_logits_buffer });
    acecfg.exes.cfg_exe.callOpts(io, acecfg.exes.cfg_args, &acecfg.exes.cfg_results, .{ .wait = true });
    acecfg.exes.cfg_results.fill(.{ &cond_logits_buffer });
    // sample next token
    acecfg.exes.sample_args.set(.{ acecfg.llm.model_buffers, cond_logits_buffer, rng_buffers });
    acecfg.exes.sample_exe.callOpts(io, acecfg.exes.sample_args, &acecfg.exes.sample_results, .{ .wait = true });
    acecfg.exes.sample_results.fill(.{ &token_buffer, &rng_buffers });

    try token_buffer.toSlice(io, token_slice);
    zml_handler.toc(&zml_handler.timers.cfg.prefill);

    std.log.info("5Hz run decode CFG, need {d} audio codes", .{ nb_audio_codes });
    zml_handler.tic(&zml_handler.timers.cfg.decode);
    var stdout = std.Io.File.stdout().writer(io, &.{});
    var writer: *std.Io.Writer = &stdout.interface;
    for (0..nb_audio_codes) |i| {
        // collect and print generated sequence
        const generated_token = token_slice.items(u32)[0];
        const chunk = try tokenizer_decoder.decode(&.{ generated_token });
        try result_tok.append(allocator, generated_token);
        try result_str.appendSlice(allocator, chunk);
        try writer.writeAll(chunk);
        if (result_tok.items.len == nb_audio_codes) break;

        var cond_pos_buffer: zml.Buffer = try .scalar(io, platform, cond_tok.len + i, .u32, sharding);
        var uncond_pos_buffer: zml.Buffer = try .scalar(io, platform, uncond_tok.len + i, .u32, sharding);
        defer cond_pos_buffer.deinit();
        defer uncond_pos_buffer.deinit();

        // embed cond tokens
        acecfg.llm.exes.decode_embed_args.set(.{ acecfg.llm.model_buffers, token_buffer });
        acecfg.llm.exes.decode_embed_exe.callOpts(io, acecfg.llm.exes.decode_embed_args, &acecfg.llm.exes.decode_embed_results, .{ .wait = true });
        acecfg.llm.exes.decode_embed_results.fill(.{ &cond_embed_buffer });
        // embed uncond tokens
        acecfg.llm.exes.decode_embed_args.set(.{ acecfg.llm.model_buffers, token_buffer });
        acecfg.llm.exes.decode_embed_exe.callOpts(io, acecfg.llm.exes.decode_embed_args, &acecfg.llm.exes.decode_embed_results, .{ .wait = true });
        acecfg.llm.exes.decode_embed_results.fill(.{ &uncond_embed_buffer });
        for (0..acecfg.llm.config.num_hidden_layers) |ii| {
            // layer ii the cond embeds
            acecfg.llm.exes.decode_layer_args.set(.{ acecfg.llm.model_buffers.layers[ii], cond_embed_buffer, cond_pos_buffer, acecfg.cond_kv_cache_buffers, layer_index_buffers[ii] });
            acecfg.llm.exes.decode_layer_exe.callOpts(io, acecfg.llm.exes.decode_layer_args, &acecfg.llm.exes.decode_layer_results, .{ .wait = true });
            acecfg.llm.exes.decode_layer_results.fill(.{ &cond_embed_buffer, &acecfg.cond_kv_cache_buffers });
            // layer ii the uncond embeds
            acecfg.llm.exes.decode_layer_args.set(.{ acecfg.llm.model_buffers.layers[ii], uncond_embed_buffer, uncond_pos_buffer, acecfg.uncond_kv_cache_buffers, layer_index_buffers[ii] });
            acecfg.llm.exes.decode_layer_exe.callOpts(io, acecfg.llm.exes.decode_layer_args, &acecfg.llm.exes.decode_layer_results, .{ .wait = true });
            acecfg.llm.exes.decode_layer_results.fill(.{ &uncond_embed_buffer, &acecfg.uncond_kv_cache_buffers });
        }
        // compute cond logits
        acecfg.llm.exes.logits_args.set(.{ acecfg.llm.model_buffers, cond_embed_buffer });
        acecfg.llm.exes.logits_exe.callOpts(io, acecfg.llm.exes.logits_args, &acecfg.llm.exes.logits_results, .{ .wait = true });
        acecfg.llm.exes.logits_results.fill(.{ &cond_logits_buffer });
        // compute uncond logits
        acecfg.llm.exes.logits_args.set(.{ acecfg.llm.model_buffers, uncond_embed_buffer });
        acecfg.llm.exes.logits_exe.callOpts(io, acecfg.llm.exes.logits_args, &acecfg.llm.exes.logits_results, .{ .wait = true });
        acecfg.llm.exes.logits_results.fill(.{ &uncond_logits_buffer });
        // combine them with cfg
        acecfg.exes.cfg_args.set(.{ acecfg.llm.model_buffers, cond_logits_buffer, uncond_logits_buffer });
        acecfg.exes.cfg_exe.callOpts(io, acecfg.exes.cfg_args, &acecfg.exes.cfg_results, .{ .wait = true });
        acecfg.exes.cfg_results.fill(.{ &cond_logits_buffer });
        // sample next token
        acecfg.exes.sample_args.set(.{ acecfg.llm.model_buffers, cond_logits_buffer, rng_buffers, true });
        acecfg.exes.sample_exe.callOpts(io, acecfg.exes.sample_args, &acecfg.exes.sample_results, .{ .wait = true });
        acecfg.exes.sample_results.fill(.{ &token_buffer, &rng_buffers });
        try token_buffer.toSlice(io, token_slice);
    }
    try writer.writeAll("\n");
    try writer.flush();
    std.log.info("5Hz CFG done, generated {d} tokens", .{ result_tok.items.len });
    zml_handler.toc(&zml_handler.timers.cfg.decode);
    return .{
        .token_id = try result_tok.toOwnedSlice(allocator),
        .string = try result_str.toOwnedSlice(allocator),
    };
}

pub fn generateTextEmbedding(zml_handler: *main.Zml_handler, aceemb: *aceemb_.AceEmb_handler, tokenizer: zml.tokenizer.Tokenizer, tokens: []const u32, token_embedding: bool) !zml.Slice {
    if (token_embedding) {
        zml_handler.tic(&zml_handler.timers.emb.decode);
    } else {
        zml_handler.tic(&zml_handler.timers.emb.prefill);
    }

    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = aceemb.params.shardings.replicated;
    const platform = zml_handler.platform;

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const seq_len = tokens.len;
    const emb_dim = aceemb.config.hidden_size;

    // the result embeddings we return
    const embedding_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = seq_len, .d = emb_dim }, .bf16));
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

    if (token_embedding) {
        zml_handler.toc(&zml_handler.timers.emb.decode);
    } else {
        zml_handler.toc(&zml_handler.timers.emb.prefill);
    }

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

    const timbre_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_dim, .t = t_timbre }, .bf16));
    defer timbre_slice.free(allocator);
    var timbre_buffer: zml.Buffer = try .fromSlice(io, platform, timbre_slice, sharding);
    defer timbre_buffer.deinit();

    const src_audio_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_dim, .t = t_25hz }, .bf16));
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
    const x_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .t = t_25hz, .a = audio_dim }, .bf16));
    const context_latents_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .t = t_25hz, .a = 2 * audio_dim }, .bf16));
    const encoded_conditions_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s_enc = s_enc, .d = emb_dim }, .bf16));
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

    zml_handler.tic(&zml_handler.timers.enc.prefill);

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

    zml_handler.toc(&zml_handler.timers.enc.prefill);

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

    const result_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_dim, .t = t_25hz }, .bf16));
    var result_buffer: zml.Buffer = try .fromSlice(io, platform, result_slice, sharding);
    defer result_buffer.deinit();

    // the full forward pass on the dit model is one iteration of the denoising
    const timestamps: [9]f32 = .{ 1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3, 0.0 };
    const steps = timestamps.len - 1;
    zml_handler.tic(&zml_handler.timers.dit.decode);
    var dit_time: std.Io.Duration = .{ .nanoseconds = 0 };
    for (0..steps) |i| {
        std.log.info("DiT ************* step {d}/{d}", .{ i+1, steps });
        var t_curr: zml.Buffer = try zml.Buffer.scalar(io, platform, timestamps[i], .f32, sharding);
        var t_next: zml.Buffer = try zml.Buffer.scalar(io, platform, timestamps[i + 1], .f32, sharding);
        defer t_curr.deinit();
        defer t_next.deinit();
        zml_handler.tic(&zml_handler.timers.dit.prefill);
        diffuse_args.set(.{ acedit.model_buffers, t_curr, t_next, x_buffer, context_latents_buffer, encoded_conditions_buffer });
        acedit.diffuse_exe.callOpts(io, diffuse_args, &diffuse_results, .{ .wait = true });
        diffuse_results.fill(.{ &x_buffer, &result_buffer });
        zml_handler.toc(&zml_handler.timers.dit.prefill);
        dit_time.nanoseconds += zml_handler.timers.dit.prefill.nanoseconds;
    }
    zml_handler.timers.dit.prefill.nanoseconds = dit_time.nanoseconds;
    zml_handler.toc(&zml_handler.timers.dit.decode);

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
    const audio_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = 2, .t = t_48khz }, .f32));
    var audio_buffer: zml.Buffer = try .fromSlice(io, platform, audio_slice, sharding);
    defer audio_buffer.deinit();

    var latent_buffer: zml.Buffer = try .fromSlice(io, platform, latents.x, sharding);
    defer latent_buffer.deinit();

    var decode_args = try acevae.decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try acevae.decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    zml_handler.tic(&zml_handler.timers.vae.prefill);

    decode_args.set(.{ acevae.model_buffers, latent_buffer });
    acevae.decode_exe.callOpts(io, decode_args, &decode_results, .{ .wait = true });
    decode_results.fill(.{ &audio_buffer });
    std.log.info("VAE done decoding, output shape : {d}x{d}", .{ 2, t_48khz });

    try audio_buffer.toSlice(io, audio_slice);

    zml_handler.toc(&zml_handler.timers.vae.prefill);

    return .{ .audio = audio_slice };
}

pub fn decodeAudioLatentsTiled(zml_handler: *main.Zml_handler, acevae: *acevae_.AceVae_handler, latents: DiffusedLatents, decode_t: u32) !DecodedAudio {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acevae.shardings.replicated;
    const platform = zml_handler.platform;

    var decode_args = try acevae.decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try acevae.decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    // the latent space is 25hz : latent_frames is always a multiple of 25
    // we use that to simplify the tiling logic
    const overlap = 25;
    const stride = decode_t * 25;
    const chunk_frames = stride + 2 * overlap;

    // [f1, f2] in latent space has coord [F1, F2] in audio space with Fi = fi * upsampling_ratio
    var upsampling_ratio: u32 = 1;
    for (acevae.config.downsampling_ratios) |ratio| {
        upsampling_ratio *= ratio;
    }

    const latent_frames: u32 = @intCast(latents.x.shape.dim(1));
    const audio_frames: u32 = @intCast(latent_frames * upsampling_ratio);
    const decoded_chunk_frames: u32 = chunk_frames * upsampling_ratio;

    const audio_dim: u32 = @intCast(latents.x.shape.dim(0));
    const audio_channels: u32 = 2;

    std.log.info("VAE call decode with input size : {d}x{d}", .{ audio_dim, latent_frames });

    // the result audio : we write into this the core of each decoded chunk
    const audio_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_channels, .t = audio_frames }, .f32));

    // chunk slice/buffer to decode
    const encoded_chunk_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_dim, .t = chunk_frames }, .bf16));
    defer encoded_chunk_slice.free(allocator);
    const decoded_chunk_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_channels, .t = decoded_chunk_frames }, .f32));
    defer decoded_chunk_slice.free(allocator);
    var decoded_chunk_buffer: zml.Buffer = try .fromSlice(io, platform, decoded_chunk_slice, sharding);
    defer decoded_chunk_buffer.deinit();

    zml_handler.tic(&zml_handler.timers.vae.prefill);

    var core_start: usize = 0;
    var core_end: usize = core_start + stride;
    var win_start: usize = undefined;
    var win_end: usize = undefined;
    
    while (true) {
        var last_chunk = false;
        if (core_start < overlap) {
            // this is first chunk, put all overlap to the right
            win_start = 0;
            win_end = core_end + 2 * overlap;
        } else if (core_end + overlap >= latent_frames) {
            // this is the last chunk, put all overlap to the left
            last_chunk = true;
            win_end = latent_frames;
            core_end = latent_frames;
            core_start = core_end - stride;
            win_start = core_start - 2 * overlap;
        } else {
            // this is a middle chunk, put overlap on both sides
            win_start = core_start - overlap;
            win_end = core_end + overlap;
        }
        std.log.info("core = [{d}..{d}] win = [{d}..{d}]", .{ core_start, core_end, win_start, win_end });
        // move the chunk data from latents.x to encoded_chunk_slice, assume tensors are stored in row major
        for (0..audio_dim) |i| {
            for (win_start..win_end) |j| {
                const j_chunk = j - win_start;
                encoded_chunk_slice.items(zml.floats.BFloat16)[i * chunk_frames + j_chunk] = latents.x.items(zml.floats.BFloat16)[i * latent_frames + j];
            }
        }
        // send the slice to the GPU
        var encoded_chunk_buffer: zml.Buffer = try .fromSlice(io, platform, encoded_chunk_slice, sharding);
        defer encoded_chunk_buffer.deinit();
        // decode it
        decode_args.set(.{ acevae.model_buffers, encoded_chunk_buffer });
        acevae.decode_exe.callOpts(io, decode_args, &decode_results, .{ .wait = true });
        decode_results.fill(.{ &decoded_chunk_buffer });
        // send the decoded chunk back to the CPU
        try decoded_chunk_buffer.toSlice(io, decoded_chunk_slice);
        // write the decoded chunk to the right place in audio_frames
        // we decoded [core_start, core_end] to upsampling_ratio * [core_start, core_end]
        // decoded_chunk is [O|O|core]
        const decoded_core_start = 2 * overlap * upsampling_ratio;
        const decoded_core_end = decoded_chunk_frames;
        for (0..audio_channels) |i| {
            for (decoded_core_start..decoded_core_end) |j| {
                audio_slice.items(f32)[i * audio_frames + j] = decoded_chunk_slice.items(f32)[i * decoded_chunk_frames + j];
            }
        }
        if (last_chunk) break;
        // slide the chunk by stride
        core_start += stride;
        core_end += stride;
    }

    std.log.info("VAE done decoding, output shape : {d}x{d}", .{ 2, audio_frames });
    zml_handler.toc(&zml_handler.timers.vae.prefill);

    return .{ .audio = audio_slice };
}


fn tokensPerSecond(duration: std.Io.Duration, tokens: u64) f64 {
    if (tokens == 0) return 0;
    const seconds = @as(f64, @floatFromInt(duration.toNanoseconds())) / 1e9;
    if (seconds <= 0) return 0;
    return @as(f64, @floatFromInt(tokens)) / seconds;
}
