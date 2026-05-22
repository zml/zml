const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const Tokenizer = zml.tokenizer.Tokenizer;
const Zml_handler = main.Zml_handler;

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
            .lyric = try extractLyricBlock(allocator, input),
        };
    }

    pub fn extractLyricBlock(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
        var split_1 = std.mem.splitSequence(u8, input, "# Lyric");
        _ = split_1.first();
        if (split_1.next()) |lyric_start| {
            var split_2 = std.mem.splitSequence(u8, lyric_start[1..], "<|im_end|>");
            const lyric = split_2.first();
            return try allocator.dupe(u8, lyric);
        } else {
            std.log.info("ERROR in parsing inspiration output", .{});
            return "";
        }
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

    pub fn empty(allocator: std.mem.Allocator) AudioMetadata {
        return .{
            .bpm = try allocator.dupe(u8, "N/A"),
            .caption = try allocator.dupe(u8, ""),
            .duration = try allocator.dupe(u8, "N/A"),
            .genres = try allocator.dupe(u8, "N/A"),
            .keyscale = try allocator.dupe(u8, "N/A"),
            .language = try allocator.dupe(u8, "N/A"),
            .timesignature = try allocator.dupe(u8, "N/A"),
            .lyric = try allocator.dupe(u8, ""),
        };
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

    pub fn duration_s(self: AudioCodes) u32 {
        // quantized space is 5Hz
        return @divExact(self.token_id.len, 5);
    }

    pub fn deinit(self: AudioCodes, allocator: std.mem.Allocator) void {
        allocator.free(self.token_id);
        allocator.free(self.string);
    }
};

pub const ContextLatents = struct {
    latents: zml.Slice,
    conditions: zml.Slice,

    pub fn print(self: ContextLatents, io: std.Io) !void {
        var stdout = std.Io.File.stdout().writer(io, &.{});
        const writer: *std.Io.Writer = &stdout.interface;
        const options: std.fmt.Number = .{};

        std.log.info("Initial latents shapes", .{});
        try self.latents.shape.format(writer);
        try self.conditions.shape.format(writer);
        try self.latents.prettyPrint(writer, options);
        try self.conditions.prettyPrint(writer, options);
    }

    pub fn deinit(self: ContextLatents, allocator: std.mem.Allocator) void {
        self.latents.free(allocator);
        self.conditions.free(allocator);
    }
};

pub const AudioLatents = struct {
    // dim [a, t]
    x: zml.Slice,

    pub fn print(self: AudioLatents, io: std.Io) !void {
        var stdout = std.Io.File.stdout().writer(io, &.{});
        const writer: *std.Io.Writer = &stdout.interface;
        const options: std.fmt.Number = .{};

        std.log.info("Audio latents shape", .{});
        try self.x.shape.format(writer);
        try self.x.prettyPrint(writer, options);
    }

    pub fn duration_s(self: AudioLatents) u32 {
        // latent space is 25hz
        return @divExact(self.x.shape.dim(1), 25);
    }

    pub fn deinit(self: AudioLatents, allocator: std.mem.Allocator) void {
        self.x.free(allocator);
    }
};

pub const AudioFrames = struct {
    // dim [2, t]
    audio: zml.Slice,

    pub fn deinit(self: AudioFrames, allocator: std.mem.Allocator) void {
        self.audio.free(allocator);
    }

    pub fn print(self: AudioFrames, io: std.Io) !void {
        var stdout = std.Io.File.stdout().writer(io, &.{});
        const writer: *std.Io.Writer = &stdout.interface;
        const options: std.fmt.Number = .{};

        std.log.info("Audio frames shape", .{});
        try self.audio.shape.format(writer);
        try self.audio.prettyPrint(writer, options);
    }

    pub fn duration_s(self: AudioFrames) u32 {
        // audio frames are 48khz
        return @divExact(self.audio.shape.dim(1), 48_000);
    }
};

// ------------------------------------------------
//                 5Hz tasks
// ------------------------------------------------

pub fn tokenizeInspirationPrompt(zml_handler: *Zml_handler, tokenizer: Tokenizer) ![]u32 {
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

pub fn tokenizeGenerationPrompt(allocator: std.mem.Allocator, tokenizer: Tokenizer, metadata: AudioMetadata) !struct { []u32, []u32 } {
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

    std.log.info("Cond prompt:\n{s}", .{ cond_prompt.items });
    std.log.info("Uncond prompt:\n{s}", .{ uncond_prompt.items });

    var cond_tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    var uncond_tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);

    try cond_tokens.appendSlice(allocator, try encoder.encode(cond_prompt.items));
    try uncond_tokens.appendSlice(allocator, try encoder.encode(uncond_prompt.items));

    return .{ try cond_tokens.toOwnedSlice(allocator), try uncond_tokens.toOwnedSlice(allocator) };
}

pub fn generateInspirationText(zml_handler: *Zml_handler, acellm: *acellm_.AceLlm_handler, prompt_tok: []const u32) ![]u8 {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acellm.shardings.replicated;
    const platform = zml_handler.platform;

    var tokenizer_decoder = try acellm.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, zml_handler.args.seed, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var zero_buffer: zml.Buffer = try .scalar(io, platform, 0, .u32, sharding);
    defer zero_buffer.deinit();

    var token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer token_slice.free(allocator);

    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ .b = 2, .s = acellm.options.seq_len }, .u32));
    defer prefill_tokens_slice.free(allocator);
    @memcpy(prefill_tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);
    @memcpy(prefill_tokens_slice.items(u32)[acellm.options.seq_len..][0..prompt_tok.len], prompt_tok);

    var buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer buffer.deinit();
    
    var x1_buffer: zml.Buffer = undefined;
    var x2_buffer: zml.Buffer = undefined;
    var q1_buffer: zml.Buffer = undefined;
    var q2_buffer: zml.Buffer = undefined;
    var k1_buffer: zml.Buffer = undefined;
    var k2_buffer: zml.Buffer = undefined;
    var v1_buffer: zml.Buffer = undefined;
    var v2_buffer: zml.Buffer = undefined;
    var delta1_buffer: zml.Buffer = undefined;
    
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

    acellm.exes.prefill_embed_args.set(.{ acellm.model_buffers, buffer });
    acellm.exes.prefill_embed_exe.call(acellm.exes.prefill_embed_args, &acellm.exes.prefill_embed_results);
    //buffer.deinit();
    acellm.exes.prefill_embed_results.fill(.{ &buffer });

    for (0..acellm.config.num_hidden_layers) |i| {
        acellm.exes.prefill_layer_pre_args.set(.{ acellm.model_buffers.layers[i], buffer });
        acellm.exes.prefill_layer_pre_exe.call(acellm.exes.prefill_layer_pre_args, &acellm.exes.prefill_layer_pre_results);
        acellm.exes.prefill_layer_pre_results.fill(.{ &x1_buffer, &q1_buffer, &k1_buffer, &v1_buffer, &x2_buffer, &q2_buffer, &k2_buffer, &v2_buffer });

        acellm.exes.prefill_layer_attn_args.set(.{ acellm.model_buffers.layers[i], x1_buffer, zero_buffer, acellm.kv_cache_buffers, layer_index_buffers[i], q1_buffer, k1_buffer, v1_buffer });
        acellm.exes.prefill_layer_attn_exe.call(acellm.exes.prefill_layer_attn_args, &acellm.exes.prefill_layer_attn_results);
        acellm.exes.prefill_layer_attn_results.fill(.{ &delta1_buffer, &acellm.kv_cache_buffers });

        acellm.exes.prefill_layer_post_args.set(.{ acellm.model_buffers.layers[i], x1_buffer, delta1_buffer, x1_buffer, delta1_buffer });
        acellm.exes.prefill_layer_post_exe.call(acellm.exes.prefill_layer_post_args, &acellm.exes.prefill_layer_post_results);
        //buffer.deinit();
        acellm.exes.prefill_layer_post_results.fill(.{ &buffer });
        x1_buffer.deinit();
        x2_buffer.deinit();
        q1_buffer.deinit();
        q2_buffer.deinit();
        k1_buffer.deinit();
        k2_buffer.deinit();
        v1_buffer.deinit();
        v2_buffer.deinit();
        delta1_buffer.deinit();
    }
    const prompt_slice: zml.Slice = try .alloc(allocator, .init(.{ .b = 2, .s = 1 }, .u32));
    defer prompt_slice.free(allocator);
    prompt_slice.items(u32)[0] = @intCast(prompt_tok.len - 1);
    prompt_slice.items(u32)[1] = @intCast(prompt_tok.len - 1);
    var prompt_buffer: zml.Buffer = try .fromSlice(io, platform, prompt_slice, sharding);
    defer prompt_buffer.deinit();
    acellm.exes.prefill_select_args.set(.{ acellm.model_buffers, buffer, prompt_buffer });
    acellm.exes.prefill_select_exe.call(acellm.exes.prefill_select_args, &acellm.exes.prefill_select_results);
    //buffer.deinit();
    acellm.exes.prefill_select_results.fill(.{ &buffer });

    acellm.exes.logits_args.set(.{ acellm.model_buffers, buffer });
    acellm.exes.logits_exe.call(acellm.exes.logits_args, &acellm.exes.logits_results);
    //buffer.deinit();
    acellm.exes.logits_results.fill(.{ &buffer });

    acellm.exes.sample_args.set(.{ acellm.model_buffers, buffer, rng_buffers });
    acellm.exes.sample_exe.call(acellm.exes.sample_args, &acellm.exes.sample_results);
    //zml.Tensor.Rng.deinitBuffer(&rng_buffers);
    //buffer.deinit();
    acellm.exes.sample_results.fill(.{ &buffer, &rng_buffers });

    try buffer.toSlice(io, token_slice);
    zml_handler.toc(&zml_handler.timers.llm.prefill);

    std.log.info("5Hz run decode", .{});
    zml_handler.tic(&zml_handler.timers.llm.decode);

    const decode_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ .b = 2, .s = 1 }, .u32));
    defer decode_tokens_slice.free(allocator);
    
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
        decode_tokens_slice.items(u32)[0] = generated_token;
        decode_tokens_slice.items(u32)[1] = generated_token;
        buffer.deinit();
        buffer = try .fromSlice(io, platform, decode_tokens_slice, sharding);

        // we need a new 1 token buffer to pass the token_index
        var pos_buffer: zml.Buffer = try .scalar(io, platform, prompt_tok.len + i, .u32, sharding);
        defer pos_buffer.deinit();

        // call to generate the next token
        acellm.exes.decode_embed_args.set(.{ acellm.model_buffers, buffer });
        acellm.exes.decode_embed_exe.call(acellm.exes.decode_embed_args, &acellm.exes.decode_embed_results);
        buffer.deinit();
        acellm.exes.decode_embed_results.fill(.{ &buffer });
        for (0..acellm.config.num_hidden_layers) |ii| {
            acellm.exes.decode_layer_pre_args.set(.{ acellm.model_buffers.layers[ii], buffer });
            acellm.exes.decode_layer_pre_exe.call(acellm.exes.decode_layer_pre_args, &acellm.exes.decode_layer_pre_results);
            acellm.exes.decode_layer_pre_results.fill(.{ &x1_buffer, &q1_buffer, &k1_buffer, &v1_buffer, &x2_buffer, &q2_buffer, &k2_buffer, &v2_buffer });

            acellm.exes.decode_layer_attn_args.set(.{ acellm.model_buffers.layers[ii], x1_buffer, pos_buffer, acellm.kv_cache_buffers, layer_index_buffers[ii], q1_buffer, k1_buffer, v1_buffer });
            acellm.exes.decode_layer_attn_exe.call(acellm.exes.decode_layer_attn_args, &acellm.exes.decode_layer_attn_results);
            acellm.exes.decode_layer_attn_results.fill(.{ &delta1_buffer, &acellm.kv_cache_buffers });

            acellm.exes.decode_layer_post_args.set(.{ acellm.model_buffers.layers[ii], x1_buffer, delta1_buffer, x1_buffer, delta1_buffer });
            acellm.exes.decode_layer_post_exe.call(acellm.exes.decode_layer_post_args, &acellm.exes.decode_layer_post_results);
            buffer.deinit();
            acellm.exes.decode_layer_post_results.fill(.{ &buffer });
            x1_buffer.deinit();
            x2_buffer.deinit();
            q1_buffer.deinit();
            q2_buffer.deinit();
            k1_buffer.deinit();
            k2_buffer.deinit();
            v1_buffer.deinit();
            v2_buffer.deinit();
            delta1_buffer.deinit();
        }
        acellm.exes.logits_args.set(.{ acellm.model_buffers, buffer });
        acellm.exes.logits_exe.call(acellm.exes.logits_args, &acellm.exes.logits_results);
        buffer.deinit();
        acellm.exes.logits_results.fill(.{ &buffer });

        acellm.exes.sample_args.set(.{ acellm.model_buffers, buffer, rng_buffers });
        acellm.exes.sample_exe.call(acellm.exes.sample_args, &acellm.exes.sample_results);
        buffer.deinit();
        zml.Tensor.Rng.deinitBuffer(&rng_buffers);
        acellm.exes.sample_results.fill(.{ &buffer, &rng_buffers });

        // extract the generated token from the buffer
        try buffer.toSlice(io, token_slice);
    }
    try writer.writeAll("\n");
    try writer.flush();
    std.log.info("5Hz done, generated {d} tokens", .{ num_tokens_generated });
    zml_handler.toc(&zml_handler.timers.llm.decode);
    return result.toOwnedSlice(allocator);
}

pub fn generateAudioCodes(zml_handler: *Zml_handler, acecfg: *acellm_.AceCfg_handler, cond_tok: []const u32, uncond_tok: []const u32, metadata: AudioMetadata) !AudioCodes {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acecfg.llm.shardings.replicated;
    const platform = zml_handler.platform;
    const acellm = acecfg.llm;

    var tokenizer_decoder = try acecfg.llm.tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const nb_audio_codes = 5 * try std.fmt.parseInt(u32, metadata.duration, 10);

    var result_tok: std.ArrayList(u32) = try .initCapacity(allocator, nb_audio_codes);
    var result_str: std.ArrayList(u8) = try .initCapacity(allocator, nb_audio_codes * 15);

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, zml_handler.args.seed, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var zero_buffer: zml.Buffer = try .scalar(io, platform, 0, .u32, sharding);
    defer zero_buffer.deinit();
    
    var token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer token_slice.free(allocator);

    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ .b = 2, .s = acecfg.llm.options.seq_len }, .u32));
    defer prefill_tokens_slice.free(allocator);
    // in .b = 0, we put the cond prompt
    // in .b = 1, we put the uncond prompt
    @memcpy(prefill_tokens_slice.items(u32)[0..cond_tok.len], cond_tok);
    @memcpy(prefill_tokens_slice.items(u32)[acecfg.llm.options.seq_len..][0..uncond_tok.len], uncond_tok);

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

    var buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer buffer.deinit();
    
    var x1_buffer: zml.Buffer = undefined;
    var x2_buffer: zml.Buffer = undefined;
    var q1_buffer: zml.Buffer = undefined;
    var q2_buffer: zml.Buffer = undefined;
    var k1_buffer: zml.Buffer = undefined;
    var k2_buffer: zml.Buffer = undefined;
    var v1_buffer: zml.Buffer = undefined;
    var v2_buffer: zml.Buffer = undefined;
    var delta1_buffer: zml.Buffer = undefined;
    var delta2_buffer: zml.Buffer = undefined;

    std.log.info("5Hz run CFG prefill", .{});
    zml_handler.tic(&zml_handler.timers.cfg.prefill);
    
    acellm.exes.prefill_embed_args.set(.{ acellm.model_buffers, buffer });
    acellm.exes.prefill_embed_exe.call(acellm.exes.prefill_embed_args, &acellm.exes.prefill_embed_results);
    buffer.deinit();
    acellm.exes.prefill_embed_results.fill(.{ &buffer });

    for (0..acellm.config.num_hidden_layers) |i| {
        acellm.exes.prefill_layer_pre_args.set(.{ acellm.model_buffers.layers[i], buffer });
        acellm.exes.prefill_layer_pre_exe.call(acellm.exes.prefill_layer_pre_args, &acellm.exes.prefill_layer_pre_results);
        acellm.exes.prefill_layer_pre_results.fill(.{ &x1_buffer, &q1_buffer, &k1_buffer, &v1_buffer, &x2_buffer, &q2_buffer, &k2_buffer, &v2_buffer });
        
        acellm.exes.prefill_layer_attn_args.set(.{ acellm.model_buffers.layers[i], x1_buffer, zero_buffer, acecfg.cond_kv_cache_buffers, layer_index_buffers[i], q1_buffer, k1_buffer, v1_buffer });
        acellm.exes.prefill_layer_attn_exe.call(acellm.exes.prefill_layer_attn_args, &acellm.exes.prefill_layer_attn_results);
        acellm.exes.prefill_layer_attn_results.fill(.{ &delta1_buffer, &acecfg.cond_kv_cache_buffers });

        acellm.exes.prefill_layer_attn_args.set(.{ acellm.model_buffers.layers[i], x2_buffer, zero_buffer, acecfg.uncond_kv_cache_buffers, layer_index_buffers[i], q2_buffer, k2_buffer, v2_buffer });
        acellm.exes.prefill_layer_attn_exe.call(acellm.exes.prefill_layer_attn_args, &acellm.exes.prefill_layer_attn_results);
        acellm.exes.prefill_layer_attn_results.fill(.{ &delta2_buffer, &acecfg.uncond_kv_cache_buffers });

        acellm.exes.prefill_layer_post_args.set(.{ acellm.model_buffers.layers[i], x1_buffer, delta1_buffer, x2_buffer, delta2_buffer });
        acellm.exes.prefill_layer_post_exe.call(acellm.exes.prefill_layer_post_args, &acellm.exes.prefill_layer_post_results);
        buffer.deinit();
        acellm.exes.prefill_layer_post_results.fill(.{ &buffer });
        x1_buffer.deinit();
        x2_buffer.deinit();
        q1_buffer.deinit();
        q2_buffer.deinit();
        k1_buffer.deinit();
        k2_buffer.deinit();
        v1_buffer.deinit();
        v2_buffer.deinit();
        delta1_buffer.deinit();
        delta2_buffer.deinit();
    }
    const prompt_slice: zml.Slice = try .alloc(allocator, .init(.{ .b = 2, .s = 1 }, .u32));
    defer prompt_slice.free(allocator);
    prompt_slice.items(u32)[0] = @intCast(cond_tok.len - 1);
    prompt_slice.items(u32)[1] = @intCast(uncond_tok.len - 1);
    var prompt_buffer: zml.Buffer = try .fromSlice(io, platform, prompt_slice, sharding);
    defer prompt_buffer.deinit();
    acellm.exes.prefill_select_args.set(.{ acellm.model_buffers, buffer, prompt_buffer });
    acellm.exes.prefill_select_exe.call(acellm.exes.prefill_select_args, &acellm.exes.prefill_select_results);
    buffer.deinit();
    acellm.exes.prefill_select_results.fill(.{ &buffer });

    acellm.exes.logits_args.set(.{ acellm.model_buffers, buffer });
    acellm.exes.logits_exe.call(acellm.exes.logits_args, &acellm.exes.logits_results);
    buffer.deinit();
    acellm.exes.logits_results.fill(.{ &buffer });

    acecfg.exes.cfg_args.set(.{ acellm.model_buffers, buffer });
    acecfg.exes.cfg_exe.call(acecfg.exes.cfg_args, &acecfg.exes.cfg_results);
    buffer.deinit();
    acecfg.exes.cfg_results.fill(.{ &buffer });

    acecfg.exes.sample_args.set(.{ acellm.model_buffers, buffer, rng_buffers });
    acecfg.exes.sample_exe.call(acecfg.exes.sample_args, &acecfg.exes.sample_results);
    zml.Tensor.Rng.deinitBuffer(&rng_buffers);
    buffer.deinit();
    acecfg.exes.sample_results.fill(.{ &buffer, &rng_buffers });

    try buffer.toSlice(io, token_slice);
    zml_handler.toc(&zml_handler.timers.cfg.prefill);

    std.log.info("5Hz run decode CFG, need {d} audio codes", .{ nb_audio_codes });
    zml_handler.tic(&zml_handler.timers.cfg.decode);

    const decode_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{ .b = 2, .s = 1 }, .u32));
    defer decode_tokens_slice.free(allocator);
    
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
        defer cond_pos_buffer.deinit();
        var uncond_pos_buffer: zml.Buffer = try .scalar(io, platform, uncond_tok.len + i, .u32, sharding);
        defer uncond_pos_buffer.deinit();

        decode_tokens_slice.items(u32)[0] = generated_token;
        decode_tokens_slice.items(u32)[1] = generated_token;
        buffer.deinit();
        buffer = try .fromSlice(io, platform, decode_tokens_slice, sharding);

        // call to generate the next token
        acellm.exes.decode_embed_args.set(.{ acellm.model_buffers, buffer });
        acellm.exes.decode_embed_exe.call(acellm.exes.decode_embed_args, &acellm.exes.decode_embed_results);
        buffer.deinit();
        acellm.exes.decode_embed_results.fill(.{ &buffer });
        for (0..acellm.config.num_hidden_layers) |ii| {
            acellm.exes.decode_layer_pre_args.set(.{ acellm.model_buffers.layers[ii], buffer });
            acellm.exes.decode_layer_pre_exe.call(acellm.exes.decode_layer_pre_args, &acellm.exes.decode_layer_pre_results);
            acellm.exes.decode_layer_pre_results.fill(.{ &x1_buffer, &q1_buffer, &k1_buffer, &v1_buffer, &x2_buffer, &q2_buffer, &k2_buffer, &v2_buffer });

            acellm.exes.decode_layer_attn_args.set(.{ acellm.model_buffers.layers[ii], x1_buffer, cond_pos_buffer, acecfg.cond_kv_cache_buffers, layer_index_buffers[ii], q1_buffer, k1_buffer, v1_buffer });
            acellm.exes.decode_layer_attn_exe.call(acellm.exes.decode_layer_attn_args, &acellm.exes.decode_layer_attn_results);
            acellm.exes.decode_layer_attn_results.fill(.{ &delta1_buffer, &acecfg.cond_kv_cache_buffers });

            acellm.exes.decode_layer_attn_args.set(.{ acellm.model_buffers.layers[ii], x2_buffer, uncond_pos_buffer, acecfg.uncond_kv_cache_buffers, layer_index_buffers[ii], q2_buffer, k2_buffer, v2_buffer });
            acellm.exes.decode_layer_attn_exe.call(acellm.exes.decode_layer_attn_args, &acellm.exes.decode_layer_attn_results);
            acellm.exes.decode_layer_attn_results.fill(.{ &delta2_buffer, &acecfg.uncond_kv_cache_buffers });

            acellm.exes.decode_layer_post_args.set(.{ acellm.model_buffers.layers[ii], x1_buffer, delta1_buffer, x2_buffer, delta2_buffer });
            acellm.exes.decode_layer_post_exe.call(acellm.exes.decode_layer_post_args, &acellm.exes.decode_layer_post_results);
            buffer.deinit();
            acellm.exes.decode_layer_post_results.fill(.{ &buffer });
            x1_buffer.deinit();
            x2_buffer.deinit();
            q1_buffer.deinit();
            q2_buffer.deinit();
            k1_buffer.deinit();
            k2_buffer.deinit();
            v1_buffer.deinit();
            v2_buffer.deinit();
            delta1_buffer.deinit();
            delta2_buffer.deinit();            
        }
        acellm.exes.logits_args.set(.{ acellm.model_buffers, buffer });
        acellm.exes.logits_exe.call(acellm.exes.logits_args, &acellm.exes.logits_results);
        buffer.deinit();
        acellm.exes.logits_results.fill(.{ &buffer });

        acecfg.exes.cfg_args.set(.{ acellm.model_buffers, buffer });
        acecfg.exes.cfg_exe.call(acecfg.exes.cfg_args, &acecfg.exes.cfg_results);
        buffer.deinit();
        acecfg.exes.cfg_results.fill(.{ &buffer });
    
        acecfg.exes.sample_args.set(.{ acellm.model_buffers, buffer, rng_buffers });
        acecfg.exes.sample_exe.call(acecfg.exes.sample_args, &acecfg.exes.sample_results);
        zml.Tensor.Rng.deinitBuffer(&rng_buffers);
        buffer.deinit();
        acecfg.exes.sample_results.fill(.{ &buffer, &rng_buffers });
        
        try buffer.toSlice(io, token_slice);
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

// ------------------------------------------------
//                 EMB tasks
// ------------------------------------------------

pub fn tokenizeInputCaption(allocator: std.mem.Allocator, tokenizer: Tokenizer, metadata: AudioMetadata, cover: bool) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var formatted_prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer formatted_prompt.deinit(allocator);

    try formatted_prompt.appendSlice(allocator, "# Instruction\n");
    if (cover) {
        try formatted_prompt.appendSlice(allocator, "Generate audio semantic tokens based on the given conditions:\n\n");
    } else {
        try formatted_prompt.appendSlice(allocator, "Fill the audio semantic mask based on the given conditions:\n\n");
    }
    try formatted_prompt.appendSlice(allocator, "# Caption\n");
    try formatted_prompt.appendSlice(allocator, metadata.caption);
    // make both tokenizations same length
    if (!cover) try formatted_prompt.appendSlice(allocator, " ");
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

pub fn tokenizeInputLyrics(allocator: std.mem.Allocator, tokenizer: Tokenizer, metadata: AudioMetadata) ![]u32 {
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

pub fn embedLyric(zml_handler: *Zml_handler, aceemb: *aceemb_.AceEmb_handler, lyric_tokens: []const u32) !zml.Slice {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = aceemb.params.shardings.replicated;
    const platform = zml_handler.platform;

    zml_handler.tic(&zml_handler.timers.emb.prefill);

    const lyric_embedding_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceemb.options.seq_len, .d = aceemb.config.hidden_size }, .bf16));
    defer lyric_embedding_slice.free(allocator);
    var lyric_embedding_buffer: zml.Buffer = undefined;
    defer lyric_embedding_buffer.deinit();

    const lyric_tokens_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceemb.options.seq_len }, .u32));
    defer lyric_tokens_slice.free(allocator);
    @memcpy(lyric_tokens_slice.items(u32)[0..lyric_tokens.len], lyric_tokens);
    var lyric_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, lyric_tokens_slice, sharding);
    defer lyric_tokens_buffer.deinit();

    aceemb.exes.lyric_embed_args.set(.{ aceemb.model_buffers, lyric_tokens_buffer });
    aceemb.exes.lyric_embed_exe.call(aceemb.exes.lyric_embed_args, &aceemb.exes.lyric_embed_results);
    aceemb.exes.lyric_embed_results.fill(.{ &lyric_embedding_buffer });

    try lyric_embedding_buffer.toSlice(io, lyric_embedding_slice);

    zml_handler.toc(&zml_handler.timers.emb.prefill);

    const lyric_embedding: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = lyric_tokens.len, .d = aceemb.config.hidden_size }, .bf16));
    const n: usize = lyric_tokens.len * aceemb.config.hidden_size;
    @memcpy(lyric_embedding.items(zml.floats.BFloat16)[0..n], lyric_embedding_slice.items(zml.floats.BFloat16)[0..n]);

    return lyric_embedding;
}

pub fn embedText(zml_handler: *Zml_handler, aceemb: *aceemb_.AceEmb_handler, text_tokens: []const u32) !zml.Slice {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = aceemb.params.shardings.replicated;
    const platform = zml_handler.platform;

    zml_handler.tic(&zml_handler.timers.emb.decode);

    const text_embedding_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceemb.options.seq_len, .d = aceemb.config.hidden_size }, .bf16));
    defer text_embedding_slice.free(allocator);
    var text_embedding_buffer: zml.Buffer = undefined;
    defer text_embedding_buffer.deinit();

    const text_tokens_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceemb.options.seq_len }, .u32));
    defer text_tokens_slice.free(allocator);
    @memcpy(text_tokens_slice.items(u32)[0..text_tokens.len], text_tokens);
    var text_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, text_tokens_slice, sharding);
    defer text_tokens_buffer.deinit();

    aceemb.exes.text_embed_args.set(.{ aceemb.model_buffers, text_tokens_buffer });
    aceemb.exes.text_embed_exe.call(aceemb.exes.text_embed_args, &aceemb.exes.text_embed_results);
    aceemb.exes.text_embed_results.fill(.{ &text_embedding_buffer });
    for (0..aceemb.config.num_hidden_layers) |i| {
        aceemb.exes.layer_args.set(.{ aceemb.model_buffers.layers[i], text_embedding_buffer });
        aceemb.exes.layer_exe.call(aceemb.exes.layer_args, &aceemb.exes.layer_results);
        aceemb.exes.layer_results.fill(.{ &text_embedding_buffer });
    }
    aceemb.exes.norm_args.set(.{ aceemb.model_buffers, text_embedding_buffer });
    aceemb.exes.norm_exe.call(aceemb.exes.norm_args, &aceemb.exes.norm_results);
    aceemb.exes.norm_results.fill(.{ &text_embedding_buffer });

    try text_embedding_buffer.toSlice(io, text_embedding_slice);

    zml_handler.toc(&zml_handler.timers.emb.decode);

    const text_embedding: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = text_tokens.len, .d = aceemb.config.hidden_size }, .bf16));
    const n: usize = text_tokens.len * aceemb.config.hidden_size;
    @memcpy(text_embedding.items(zml.floats.BFloat16)[0..n], text_embedding_slice.items(zml.floats.BFloat16)[0..n]);

    return text_embedding;
}

// ------------------------------------------------
//                 ENC tasks
// ------------------------------------------------

pub fn prepareDiffusionLatents(zml_handler: *Zml_handler, aceenc: *aceenc_.AceEnc_handler, duration: u32, audio_codes: ?[]u32, audio_latents: ?AudioLatents) !zml.Slice {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = aceenc.shardings.replicated;
    const platform = zml_handler.platform;

    const audio_dim = aceenc.config.timbre_hidden_dim;
    const t_25hz = 25 * duration;
    const n: usize = @intCast(t_25hz * audio_dim);

    zml_handler.tic(&zml_handler.timers.enc.prefill);

    // the diffusion latents are composed of two parts : first a source latent that is the structural scaffolding the
    // diffusion will use to denoise towards, and second, a logical mask that constrains temporally where the diffusion acts
    // the source latent can come from dequantized and detokenized audio codes, from silence latent or from source audio

    const mask_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .t = t_25hz, .a = audio_dim }, .bf16));
    defer mask_slice.free(allocator);
    const one = zml.floats.BFloat16.fromF32(1.0);
    for (0..n) |i| {
        mask_slice.items(zml.floats.BFloat16)[i] = one;
    }

    const latents_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .t = aceenc.options.seq_len_time * 25, .a = audio_dim }, .bf16));
    defer latents_slice.free(allocator);

    const return_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .t = t_25hz, .a = 2 * audio_dim }, .bf16));

    if (audio_codes) |codes| {
        std.log.info("ENC init source latents from audiocodes", .{});
        const audio_codes_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceenc.options.seq_len_time * 5 }, .u32));
        defer audio_codes_slice.free(allocator);
        @memcpy(audio_codes_slice.items(u32)[0..codes.len], codes);
        var audio_codes_buffer: zml.Buffer = try .fromSlice(io, platform, audio_codes_slice, sharding);
        defer audio_codes_buffer.deinit();
        aceenc.exes.encode_audiocodes_args.set(.{ aceenc.model_buffers.audiocode_encoder, audio_codes_buffer });
        aceenc.exes.encode_audiocodes_exe.call(aceenc.exes.encode_audiocodes_args, &aceenc.exes.encode_audiocodes_results);
        var latent_buffer: zml.Buffer = undefined;
        defer latent_buffer.deinit();
        aceenc.exes.encode_audiocodes_results.fill(.{ &latent_buffer });
        try latent_buffer.toSlice(io, latents_slice);
    } else if (audio_latents) |audio| {
        std.log.info("ENC init source latents from audio latents", .{});
        @memcpy(latents_slice.items(zml.floats.BFloat16)[0..n], audio.x.items(zml.floats.BFloat16));
    } else {
        std.log.info("ENC init source latents from silence latents", .{});
        aceenc.exes.silence_args.set(.{ aceenc.silence_buffers });
        aceenc.exes.silence_exe.call(aceenc.exes.silence_args, &aceenc.exes.silence_results);
        var latent_buffer: zml.Buffer = undefined;
        defer latent_buffer.deinit();
        aceenc.exes.silence_results.fill(.{ &latent_buffer });
        try latent_buffer.toSlice(io, latents_slice);
    }

    // concatenate slices rows
    for (0..t_25hz) |i| {
        for (0..audio_dim) |j| {
            return_slice.items(zml.floats.BFloat16)[i * (2 * audio_dim) + j] = latents_slice.items(zml.floats.BFloat16)[i * audio_dim + j];
            return_slice.items(zml.floats.BFloat16)[i * (2 * audio_dim) + j + audio_dim] = mask_slice.items(zml.floats.BFloat16)[i * audio_dim + j];
        }
    }

    zml_handler.toc(&zml_handler.timers.enc.prefill);

    return return_slice;
}

pub fn prepareDiffusionConditions(zml_handler: *Zml_handler, aceenc: *aceenc_.AceEnc_handler, cap_emb: zml.Slice, lyric_emb: zml.Slice, style_latents: ?AudioLatents) !zml.Slice {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = aceenc.shardings.replicated;
    const platform = zml_handler.platform;

    const caption_len = cap_emb.shape.dim(0);
    const lyric_len = lyric_emb.shape.dim(0);
    const input_dim = cap_emb.shape.dim(1);
    const emb_dim = aceenc.config.encoder_hidden_size;
    const t_timbre_25hz = aceenc.config.timbre_fix_frame;
    const audio_dim = aceenc.config.timbre_hidden_dim;
    const s_enc = caption_len + 1 + lyric_len;

    const n_cap_in: usize = @intCast(input_dim * caption_len);
    const n_tim_in: usize = @intCast(audio_dim * t_timbre_25hz);
    const n_lyr_in: usize = @intCast(input_dim * lyric_len);
    
    zml_handler.tic(&zml_handler.timers.enc.decode);

    // the diffusion conditions are composed of three parts :
    // - the encoded caption embedding (full embed)
    // - the encoded timbre/style reference, either created from a audio file or from silence latent
    // - the encoded lyric embedding (token embed)

    const caption_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceenc.options.seq_len_text, .d = emb_dim }, .bf16));
    defer caption_slice.free(allocator);
    const lyric_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceenc.options.seq_len_text, .d = emb_dim }, .bf16));
    defer lyric_slice.free(allocator);
    const timbre_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1, .d = emb_dim }, .bf16));
    defer timbre_slice.free(allocator);

    const result_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = s_enc, .d = emb_dim }, .bf16));

    // encode text
    std.log.info("ENC encoding text", .{});
    const caption_input_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceenc.options.seq_len_text, .d = input_dim }, .bf16));
    defer caption_input_slice.free(allocator);
    @memcpy(caption_input_slice.items(zml.floats.BFloat16)[0..n_cap_in], cap_emb.items(zml.floats.BFloat16));
    var caption_input_buffer: zml.Buffer = try .fromSlice(io, platform, caption_input_slice, sharding);
    defer caption_input_buffer.deinit();
    var caption_output_buffer: zml.Buffer = undefined;
    defer caption_output_buffer.deinit();
    aceenc.exes.encode_text_args.set(.{ aceenc.model_buffers.text_encoder, caption_input_buffer });
    aceenc.exes.encode_text_exe.call(aceenc.exes.encode_text_args, &aceenc.exes.encode_text_results);
    aceenc.exes.encode_text_results.fill(.{ &caption_output_buffer });
    try caption_output_buffer.toSlice(io, caption_slice);

    // encode lyric
    std.log.info("ENC encoding lyric", .{});
    const lyric_input_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceenc.options.seq_len_text, .d = input_dim }, .bf16));
    defer lyric_input_slice.free(allocator);
    @memcpy(lyric_input_slice.items(zml.floats.BFloat16)[0..n_lyr_in], lyric_emb.items(zml.floats.BFloat16));
    var lyric_input_buffer: zml.Buffer = try .fromSlice(io, platform, lyric_input_slice, sharding);
    defer lyric_input_buffer.deinit();
    var lyric_output_buffer: zml.Buffer = undefined;
    defer lyric_output_buffer.deinit();
    const mask_slice = try createBidirectionalRangeMask(allocator, aceenc.options.seq_len_text, lyric_len);
    defer mask_slice.free(allocator);
    var mask_buffer: zml.Buffer = try .fromSlice(io, platform, mask_slice, sharding);
    defer mask_buffer.deinit();
    aceenc.exes.encode_lyric_args.set(.{ aceenc.model_buffers.lyric_encoder, lyric_input_buffer, mask_buffer });
    aceenc.exes.encode_lyric_exe.call(aceenc.exes.encode_lyric_args, &aceenc.exes.encode_lyric_results);
    aceenc.exes.encode_lyric_results.fill(.{ &lyric_output_buffer });
    try lyric_output_buffer.toSlice(io, lyric_slice);

    // encode timbre
    const timbre_input_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = t_timbre_25hz, .d = audio_dim }, .bf16));
    defer timbre_input_slice.free(allocator);
    if (style_latents) |latents| {
        std.log.info("ENC encoding timbre from reference audio", .{});
        @memcpy(timbre_input_slice.items(zml.floats.BFloat16)[0..n_tim_in], latents.x.items(zml.floats.BFloat16));
    } else {
        std.log.info("ENC encoding timbre from silence latent", .{});
        const silence_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = aceenc.options.seq_len_time * 25, .d = audio_dim }, .bf16));
        defer silence_slice.free(allocator);
        var silence_buffer: zml.Buffer = undefined;
        defer silence_buffer.deinit();
        aceenc.exes.silence_args.set(.{ aceenc.silence_buffers });
        aceenc.exes.silence_exe.call(aceenc.exes.silence_args, &aceenc.exes.silence_results);
        aceenc.exes.silence_results.fill(.{ &silence_buffer });
        try silence_buffer.toSlice(io, silence_slice);
        @memcpy(timbre_input_slice.items(zml.floats.BFloat16)[0..n_tim_in], silence_slice.items(zml.floats.BFloat16)[0..n_tim_in]);
    }
    var timbre_input_buffer: zml.Buffer = try .fromSlice(io, platform, timbre_input_slice, sharding);
    defer timbre_input_buffer.deinit();
    var timbre_output_buffer: zml.Buffer = undefined;
    defer timbre_output_buffer.deinit();
    aceenc.exes.encode_timbre_args.set(.{ aceenc.model_buffers.timbre_encoder, timbre_input_buffer });
    aceenc.exes.encode_timbre_exe.call(aceenc.exes.encode_timbre_args, &aceenc.exes.encode_timbre_results);
    aceenc.exes.encode_timbre_results.fill(.{ &timbre_output_buffer });
    try timbre_output_buffer.toSlice(io, timbre_slice);

    // merge slices
    const n_cap: usize = @intCast(emb_dim * caption_len);
    const n_tim: usize = @intCast(emb_dim * 1);
    const n_lyr: usize = @intCast(emb_dim * lyric_len);
    const cap_start = 0;
    const cap_end = cap_start + n_cap;
    const timbre_start = cap_end;
    const timbre_end = timbre_start + n_tim;
    const lyr_start = timbre_end;
    const lyr_end = lyr_start + n_lyr;
    
    @memcpy(result_slice.items(zml.floats.BFloat16)[cap_start..cap_end], caption_slice.items(zml.floats.BFloat16)[0..n_cap]);
    @memcpy(result_slice.items(zml.floats.BFloat16)[timbre_start..timbre_end], timbre_slice.items(zml.floats.BFloat16)[0..n_tim]);
    @memcpy(result_slice.items(zml.floats.BFloat16)[lyr_start..lyr_end], lyric_slice.items(zml.floats.BFloat16)[0..n_lyr]);

    zml_handler.toc(&zml_handler.timers.enc.decode);
    
    return result_slice;
}

pub fn createBidirectionalRangeMask(allocator: std.mem.Allocator, seq_len: i64, seq_range: i64) !zml.Slice {
    var mask: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .q = seq_len, .k = seq_len }, .bf16));
    const mask_items = mask.items(zml.floats.BFloat16);
    const zero = zml.floats.BFloat16.fromF32(0.0);
    const minus_inf = zml.floats.BFloat16.fromF32(zml.floats.Float32.toF32(zml.floats.Float32.minus_inf));
    const seq_len_u: usize = @intCast(seq_len);
    for (0..seq_len_u) |q| {
        for (0..seq_len_u) |k| {
            const idx = q * seq_len_u + k;
            mask_items[idx] = if (q < seq_range and k < seq_range) zero else minus_inf;
        }
    }
    return mask;
}

// ------------------------------------------------
//                 DiT tasks
// ------------------------------------------------

pub fn runCoverDiffusion(zml_handler: *Zml_handler, acedit: *acedit_.AceDit_handler, context: ContextLatents, id: usize) !AudioLatents {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acedit.shardings.replicated;
    const platform = zml_handler.platform;

    const t = context.latents.shape.dim(0);
    const a = @divExact(context.latents.shape.dim(1), 2);
    const s = context.conditions.shape.dim(0);
    const d = context.conditions.shape.dim(1);

    std.log.info("DiT call with input size : {d}x{d} {d}x{d}", .{ t, a, s, d });

    const timestamps: [9]f32 = .{ 1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3, 0.0 };

    // populate x with gaussian noise seeded with id
    const x: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .t = t, .a = a }, .bf16));
    defer x.free(allocator);
    const seed: u64 = @intCast(id);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const dimT: usize = @intCast(t);
    const dimA: usize = @intCast(a);
    for (0..dimT) |i| {
        for (0..dimA) |j| {
            const rand: f32 = random.floatNorm(f32);
            x.items(zml.floats.BFloat16)[j + i * dimA] = zml.floats.BFloat16.fromF32(rand);
        }
    }

    // prepare arguments buffers
    var x_buffer: zml.Buffer = try .fromSlice(io, platform, x, sharding);
    var latents_buffer: zml.Buffer = try .fromSlice(io, platform, context.latents, sharding);
    var conditions_buffer: zml.Buffer = try .fromSlice(io, platform, context.conditions, sharding);
    defer x_buffer.deinit();
    defer latents_buffer.deinit();
    defer conditions_buffer.deinit();

    const full_mask_slice = try createBidirectionalFullMask(allocator, @divFloor(t + 1, 2));
    defer full_mask_slice.free(allocator);
    var full_mask_buffer: zml.Buffer = try .fromSlice(io, platform, full_mask_slice, sharding);
    defer full_mask_buffer.deinit();
    const sliding_mask_slice = try createBidirectionalWindowMask(allocator, @divFloor(t + 1, 2), acedit.config.sliding_window);
    defer sliding_mask_slice.free(allocator);
    var sliding_mask_buffer: zml.Buffer = try .fromSlice(io, platform, sliding_mask_slice, sharding);
    defer sliding_mask_buffer.deinit();

    const result_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = a, .t = t }, .bf16));
    var result_buffer: zml.Buffer = try .fromSlice(io, platform, result_slice, sharding);
    defer result_buffer.deinit();

    // the full forward pass on the dit model is one iteration of the denoising
    const steps = timestamps.len - 1;
    zml_handler.tic(&zml_handler.timers.dit.prefill);
    for (0..steps) |i| {
        var y_proj_buffer: zml.Buffer = try zml.Buffer.scalar(io, platform, 0, .bf16, sharding);
        defer y_proj_buffer.deinit();
        var hidden_states_buffer: zml.Buffer = undefined;
        defer hidden_states_buffer.deinit();
        var temb_buffer: zml.Buffer = try zml.Buffer.scalar(io, platform, 0, .bf16, sharding);
        defer temb_buffer.deinit();
        var timestep_proj_buffer: zml.Buffer = try zml.Buffer.scalar(io, platform, 0, .bf16, sharding);
        defer timestep_proj_buffer.deinit();
        var t_curr: zml.Buffer = try zml.Buffer.scalar(io, platform, timestamps[i], .f32, sharding);
        var t_next: zml.Buffer = try zml.Buffer.scalar(io, platform, timestamps[i + 1], .f32, sharding);
        defer t_curr.deinit();
        defer t_next.deinit();
        std.log.info("DiT ************* step {d}/{d}, is_cover: {any}", .{ i+1, steps, true });
        acedit.exes.preprocess_args.set(.{ acedit.model_buffers, t_curr, x_buffer, latents_buffer, conditions_buffer });
        acedit.exes.preprocess_exe.call(acedit.exes.preprocess_args, &acedit.exes.preprocess_results);
        y_proj_buffer.deinit();
        temb_buffer.deinit();
        timestep_proj_buffer.deinit();
        acedit.exes.preprocess_results.fill(.{ &y_proj_buffer, &hidden_states_buffer, &temb_buffer, &timestep_proj_buffer });
        for (0..acedit.config.num_hidden_layers) |ii| {
            const mask_buffer = if (acedit.config.layer_types[ii] == .sliding_attention) sliding_mask_buffer else full_mask_buffer;
            acedit.exes.layer_args.set(.{ acedit.model_buffers.layers[ii], hidden_states_buffer, y_proj_buffer, timestep_proj_buffer, mask_buffer });
            acedit.exes.layer_exe.call(acedit.exes.layer_args, &acedit.exes.layer_results);
            acedit.exes.layer_results.fill(.{ &hidden_states_buffer });
        }
        acedit.exes.postprocess_args.set(.{ acedit.model_buffers, t_curr, t_next, x_buffer, hidden_states_buffer, temb_buffer });
        acedit.exes.postprocess_exe.call(acedit.exes.postprocess_args, &acedit.exes.postprocess_results);
        result_buffer.deinit();
        acedit.exes.postprocess_results.fill(.{ &x_buffer, &result_buffer });
    }
    try result_buffer.toSlice(io, result_slice);

    zml_handler.toc(&zml_handler.timers.dit.prefill);
    
    return .{ .x = result_slice };
}

pub fn runRemixDiffusion(zml_handler: *Zml_handler, acedit: *acedit_.AceDit_handler, source: AudioLatents, context_cover: ContextLatents, context_non_cover: ContextLatents, id: usize, match_level: u8) !AudioLatents {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acedit.shardings.replicated;
    const platform = zml_handler.platform;

    // for now, contexts have same size
    const t = context_cover.latents.shape.dim(0);
    const a = @divExact(context_cover.latents.shape.dim(1), 2);
    const s = context_cover.conditions.shape.dim(0);
    const d = context_cover.conditions.shape.dim(1);

    std.log.info("DiT call with input size : {d}x{d} {d}x{d}", .{ t, a, s, d });

    const timestamps: [9]f32 = .{ 1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3, 0.0 };
    const noise = timestamps[match_level];

    // populate x with gaussian noise seeded with id
    const x: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .t = t, .a = a }, .bf16));
    defer x.free(allocator);
    var prng = std.Random.DefaultPrng.init(@intCast(id));
    const random = prng.random();
    const dimT: usize = @intCast(t);
    const dimA: usize = @intCast(a);
    std.log.info("Renoising reference audio with noise level {d}, starting DiT from step {d}", .{ noise, match_level });
    for (0..dimT) |i| {
        for (0..dimA) |j| {
            const rand: f32 = random.floatNorm(f32);
            // apply noise to s.x to initialize x : we only add noise in quantity that matches a level of the schedule,
            // and we then start the diffusion from that level.
            const target: f32 = source.x.items(zml.floats.BFloat16)[i + j * dimT].toF32(); // source.x is [a, t] because it comes from VAE encode
            const noised = noise * rand + (1.0 - noise) * target;
            x.items(zml.floats.BFloat16)[j + i * dimA] = zml.floats.BFloat16.fromF32(noised); // x is [t, a]
        }
    }

    // prepare arguments buffers
    var x_buffer: zml.Buffer = try .fromSlice(io, platform, x, sharding);
    var latents_cover_buffer: zml.Buffer = try .fromSlice(io, platform, context_cover.latents, sharding);
    var conditions_cover_buffer: zml.Buffer = try .fromSlice(io, platform, context_cover.conditions, sharding);
    var latents_non_cover_buffer: zml.Buffer = try .fromSlice(io, platform, context_non_cover.latents, sharding);
    var conditions_non_cover_buffer: zml.Buffer = try .fromSlice(io, platform, context_non_cover.conditions, sharding);
    defer x_buffer.deinit();
    defer latents_cover_buffer.deinit();
    defer conditions_cover_buffer.deinit();
    defer latents_non_cover_buffer.deinit();
    defer conditions_non_cover_buffer.deinit();

    const full_mask_slice = try createBidirectionalFullMask(allocator, @divFloor(t + 1, 2));
    defer full_mask_slice.free(allocator);
    var full_mask_buffer: zml.Buffer = try .fromSlice(io, platform, full_mask_slice, sharding);
    defer full_mask_buffer.deinit();
    const sliding_mask_slice = try createBidirectionalWindowMask(allocator, @divFloor(t + 1, 2), acedit.config.sliding_window);
    defer sliding_mask_slice.free(allocator);
    var sliding_mask_buffer: zml.Buffer = try .fromSlice(io, platform, sliding_mask_slice, sharding);
    defer sliding_mask_buffer.deinit();

    const result_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = a, .t = t }, .bf16));
    var result_buffer: zml.Buffer = try .fromSlice(io, platform, result_slice, sharding);
    defer result_buffer.deinit();

    // the full forward pass on the dit model is one iteration of the denoising
    const steps = timestamps.len - 1;
    zml_handler.tic(&zml_handler.timers.dit.prefill);
    for (match_level..steps) |i| {
        var y_proj_buffer: zml.Buffer = try zml.Buffer.scalar(io, platform, 0, .bf16, sharding);
        defer y_proj_buffer.deinit();
        var hidden_states_buffer: zml.Buffer = undefined;
        defer hidden_states_buffer.deinit();
        var temb_buffer: zml.Buffer = try zml.Buffer.scalar(io, platform, 0, .bf16, sharding);
        defer temb_buffer.deinit();
        var timestep_proj_buffer: zml.Buffer = try zml.Buffer.scalar(io, platform, 0, .bf16, sharding);
        defer timestep_proj_buffer.deinit();

        var t_curr: zml.Buffer = try zml.Buffer.scalar(io, platform, timestamps[i], .f32, sharding);
        var t_next: zml.Buffer = try zml.Buffer.scalar(io, platform, timestamps[i + 1], .f32, sharding);
        defer t_curr.deinit();
        defer t_next.deinit();
        const is_cover = i < zml_handler.args.cover_strength;
        const latents_buffer = if (is_cover) latents_cover_buffer else latents_non_cover_buffer;
        const conditions_buffer = if (is_cover) conditions_cover_buffer else conditions_non_cover_buffer;
        std.log.info("DiT ************* step {d}/{d}, is_cover: {any}", .{ i+1, steps, is_cover });

        acedit.exes.preprocess_args.set(.{ acedit.model_buffers, t_curr, x_buffer, latents_buffer, conditions_buffer });
        acedit.exes.preprocess_exe.call(acedit.exes.preprocess_args, &acedit.exes.preprocess_results);
        y_proj_buffer.deinit();
        temb_buffer.deinit();
        timestep_proj_buffer.deinit();
        acedit.exes.preprocess_results.fill(.{ &y_proj_buffer, &hidden_states_buffer, &temb_buffer, &timestep_proj_buffer });
        for (0..acedit.config.num_hidden_layers) |ii| {
            const mask_buffer = if (acedit.config.layer_types[ii] == .sliding_attention) sliding_mask_buffer else full_mask_buffer;
            acedit.exes.layer_args.set(.{ acedit.model_buffers.layers[ii], hidden_states_buffer, y_proj_buffer, timestep_proj_buffer, mask_buffer });
            acedit.exes.layer_exe.call(acedit.exes.layer_args, &acedit.exes.layer_results);
            acedit.exes.layer_results.fill(.{ &hidden_states_buffer });
        }
        acedit.exes.postprocess_args.set(.{ acedit.model_buffers, t_curr, t_next, x_buffer, hidden_states_buffer, temb_buffer });
        acedit.exes.postprocess_exe.call(acedit.exes.postprocess_args, &acedit.exes.postprocess_results);
        result_buffer.deinit();
        acedit.exes.postprocess_results.fill(.{ &x_buffer, &result_buffer });
    }
    
    try result_buffer.toSlice(io, result_slice);

    zml_handler.toc(&zml_handler.timers.dit.prefill);
    
    return .{ .x = result_slice };
}

pub fn createBidirectionalFullMask(allocator: std.mem.Allocator, seq_len: i64) !zml.Slice {
    var mask: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .q = seq_len, .k = seq_len }, .bf16));
    const mask_items = mask.items(zml.floats.BFloat16);
    const zero = zml.floats.BFloat16.fromF32(0.0);
    const seq_len_u: usize = @intCast(seq_len);
    for (0..seq_len_u) |q| {
        for (0..seq_len_u) |k| {
            const idx = q * seq_len_u + k;
            mask_items[idx] = zero;
        }
    }
    return mask;
}

pub fn createBidirectionalWindowMask(allocator: std.mem.Allocator, seq_len: i64, window_len: u32) !zml.Slice {
    var mask: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .q = seq_len, .k = seq_len }, .bf16));
    const zeros = zml.floats.BFloat16.fromF32(0.0);
    const minus_inf = zml.floats.BFloat16.fromF32(zml.floats.Float32.toF32(zml.floats.Float32.minus_inf));
    const mask_items = mask.items(zml.floats.BFloat16);
    const seq_len_u: usize = @intCast(seq_len);
    const window_len_i64: i64 = @intCast(window_len);
    for (0..seq_len_u) |q| {
        for (0..seq_len_u) |k| {
            const q_i64: i64 = @intCast(q);
            const k_i64: i64 = @intCast(k);
            const idx = q * seq_len_u + k;
            mask_items[idx] = if (@abs(q_i64 - k_i64) <= window_len_i64) zeros else minus_inf;
        }
    }
    return mask;
}

// ------------------------------------------------
//                 VAE tasks
// ------------------------------------------------

pub fn encodeAudioLatents(zml_handler: *Zml_handler, acevae: *acevae_.AceVaeEncoder_handler, audio_input: AudioFrames) !AudioLatents {
    const io = zml_handler.io;
    const allocator = zml_handler.allocator;
    const sharding = acevae.shardings.replicated;
    const platform = zml_handler.platform;

    var encode_args = try acevae.encode_exe.args(allocator);
    defer encode_args.deinit(allocator);
    var encode_results = try acevae.encode_exe.results(allocator);
    defer encode_results.deinit(allocator);

    const overlap = 25;
    const stride = acevae_.decode_t * 25;
    const chunk_frames = stride + 2 * overlap;
    var upsampling_ratio: u32 = 1;
    for (acevae.config.downsampling_ratios) |ratio| {
        upsampling_ratio *= ratio;
    }

    const audio_channels: usize = @intCast(audio_input.audio.shape.dim(0));
    const audio_frames: usize = @intCast(audio_input.audio.shape.dim(1));
    const latent_dim: usize = acevae.config.decoder_input_channels;
    const latent_frames: usize = @divFloor(audio_frames, upsampling_ratio);
    const chunk_latent_frames: usize = chunk_frames;
    const chunk_audio_frames: usize = chunk_frames * upsampling_ratio;

    std.log.info("VAE call encode with input size : {d}x{d}", .{ audio_channels, audio_frames });

    const latents_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = latent_dim, .t = latent_frames }, .bf16));

    const audio_chunk_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = audio_channels, .t = chunk_audio_frames }, .f32));
    defer audio_chunk_slice.free(allocator);

    const latent_chunk_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .a = latent_dim, .t = chunk_latent_frames }, .bf16));
    defer latent_chunk_slice.free(allocator);
    var latent_chunk_buffer: zml.Buffer = try zml.Buffer.fromSlice(io, platform, latent_chunk_slice, sharding);
    defer latent_chunk_buffer.deinit();

    zml_handler.tic(&zml_handler.timers.vae.prefill);

    var core_start: usize = 0;
    var core_end: usize = core_start + stride;
    var win_start: usize = undefined;
    var win_end: usize = undefined;

    while (true) {
        var last_chunk = false;
        if (core_start == 0) {
            win_start = 0;
            win_end = core_end + 2 * overlap;
        } else if (core_end + overlap >= latent_frames) {
            last_chunk = true;
            win_end = latent_frames;
            core_end = latent_frames;
            core_start = core_end - stride;
            win_start = core_start - 2 * overlap;
        } else {
            win_start = core_start - overlap;
            win_end = core_end + overlap;
        }

        const win_audio_start = win_start * upsampling_ratio;
        const win_audio_end = win_end * upsampling_ratio;
        for (0..audio_channels) |i| {
            for (win_audio_start..win_audio_end) |j| {
                const j_chunk = j - win_audio_start;
                if (j < audio_frames) {
                    audio_chunk_slice.items(f32)[i * chunk_audio_frames + j_chunk] = audio_input.audio.items(f32)[i * audio_frames + j];
                }
            }
        }

        var audio_chunk_buffer: zml.Buffer = try .fromSlice(io, platform, audio_chunk_slice, sharding);
        defer audio_chunk_buffer.deinit();

        encode_args.set(.{ acevae.model_buffers, audio_chunk_buffer, latent_chunk_buffer });
        acevae.encode_exe.call(encode_args, &encode_results);
        encode_results.fill(.{ &latent_chunk_buffer });

        try latent_chunk_buffer.toSlice(io, latent_chunk_slice);

        const chunk_core_start = (core_start - win_start);
        for (0..latent_dim) |i| {
            for (core_start..core_end) |j| {
                const j_chunk = (j - core_start) + chunk_core_start;
                latents_slice.items(zml.floats.BFloat16)[i * latent_frames + j] = latent_chunk_slice.items(zml.floats.BFloat16)[i * chunk_latent_frames + j_chunk];
            }
        }

        if (last_chunk) break;
        core_start += stride;
        core_end += stride;
    }

    std.log.info("VAE done encoding, output shape : {d}x{d}", .{ latent_dim, latent_frames });
    zml_handler.toc(&zml_handler.timers.vae.prefill);

    return .{ .x = latents_slice };
}

pub fn decodeAudioLatents(zml_handler: *Zml_handler, acevae: *acevae_.AceVaeDecoder_handler, latents: AudioLatents) !AudioFrames {
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
    const stride = acevae_.decode_t * 25;
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
    var decoded_chunk_buffer: zml.Buffer = try zml.Buffer.fromSlice(io, platform, decoded_chunk_slice, sharding);
    defer decoded_chunk_buffer.deinit();

    zml_handler.tic(&zml_handler.timers.vae.prefill);

    var core_start: usize = 0;
    var core_end: usize = core_start + stride;
    var win_start: usize = undefined;
    var win_end: usize = undefined;

    while (true) {
        var last_chunk = false;
        if (core_start == 0) {
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
        decode_args.set(.{ acevae.model_buffers, encoded_chunk_buffer, decoded_chunk_buffer });
        acevae.decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &decoded_chunk_buffer });
        // send the decoded chunk back to the CPU
        try decoded_chunk_buffer.toSlice(io, decoded_chunk_slice);
        // write the decoded chunk to the right place in audio_frames
        const decoded_core_start = core_start * upsampling_ratio;
        const decoded_core_end = core_end * upsampling_ratio;
        const chunk_core_start = (core_start - win_start) * upsampling_ratio;
        for (0..audio_channels) |i| {
            for (decoded_core_start..decoded_core_end) |j| {
                const j_chunk = j - decoded_core_start + chunk_core_start;
                audio_slice.items(f32)[i * audio_frames + j] = decoded_chunk_slice.items(f32)[i * decoded_chunk_frames + j_chunk];
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
