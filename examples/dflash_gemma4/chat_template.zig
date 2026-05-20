const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.@"dflash_gemma4/chat_template");

pub const ChatTemplate = struct {
    config: Config,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !ChatTemplate {
        const tokenizer_config = try parseTokenizerConfig(allocator, io, repo);
        defer tokenizer_config.deinit();

        const source = try loadTemplateSource(allocator, io, repo, tokenizer_config.value);
        defer allocator.free(source);

        if (!isSupportedGemmaTemplate(source)) {
            log.err("Gemma 4 DFlash requires a supported Gemma 4 single-user chat template with add_generation_prompt support", .{});
            return error.UnsupportedGemmaChatTemplate;
        }

        var config = try Config.fromTokenizerConfig(allocator, tokenizer_config.value);
        errdefer config.deinit(allocator);

        return .{
            .config = config,
        };
    }

    pub fn deinit(self: *ChatTemplate, allocator: std.mem.Allocator) void {
        self.config.deinit(allocator);
    }

    pub fn tokenizePrompt(
        self: *const ChatTemplate,
        allocator: std.mem.Allocator,
        tokenizer: *zml.tokenizer.Tokenizer,
        prompt: []const u8,
    ) ![]u32 {
        const rendered = try self.renderUserPrompt(allocator, prompt);
        defer allocator.free(rendered);

        var encoder = try tokenizer.encoder();
        defer encoder.deinit();
        return try encoder.encodeAlloc(allocator, rendered);
    }

    fn renderUserPrompt(self: *const ChatTemplate, allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
        const bos = self.config.bos_token orelse "<bos>";
        return try std.fmt.allocPrint(
            allocator,
            "{s}<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{s}<turn|>\n<|turn>model\n",
            .{ bos, prompt },
        );
    }
};

pub fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try repo.openFile(io, "tokenizer.json", .{});
    defer file.close(io);

    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);

    return try .fromBytes(allocator, bytes);
}

fn loadTemplateSource(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo: std.Io.Dir,
    tokenizer_config: std.json.Value,
) ![]const u8 {
    const jinja_file = repo.openFile(io, "chat_template.jinja", .{}) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (jinja_file) |file| {
        defer file.close(io);
        var reader = file.reader(io, &.{});
        return try reader.interface.readAlloc(allocator, try file.length(io));
    }

    if (getChatTemplateFromConfig(tokenizer_config)) |template| {
        return try allocator.dupe(u8, template);
    }

    const legacy_file = repo.openFile(io, "chat_template.json", .{}) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (legacy_file) |file| {
        defer file.close(io);
        var reader = file.reader(io, &.{});
        const payload = try reader.interface.readAlloc(allocator, try file.length(io));
        defer allocator.free(payload);

        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, payload, .{ .allocate = .alloc_if_needed });
        defer parsed.deinit();

        if (getLegacyChatTemplateFromConfig(parsed.value)) |template| {
            return try allocator.dupe(u8, template);
        }
    }

    log.err("No chat template found (checked chat_template.jinja, tokenizer_config.json, chat_template.json)", .{});
    return error.NoChatTemplate;
}

fn getChatTemplateFromConfig(tokenizer_config: std.json.Value) ?[]const u8 {
    if (tokenizer_config != .object) return null;
    const ct = tokenizer_config.object.get("chat_template") orelse return null;
    return switch (ct) {
        .string => |s| s,
        .array => |arr| blk: {
            var first_template: ?[]const u8 = null;
            for (arr.items) |item| {
                if (item != .object) continue;
                const name = if (item.object.get("name")) |n| switch (n) {
                    .string => |s| s,
                    else => continue,
                } else continue;
                const tmpl = if (item.object.get("template")) |t| switch (t) {
                    .string => |s| s,
                    else => continue,
                } else continue;
                if (first_template == null) first_template = tmpl;
                if (std.mem.eql(u8, name, "default")) break :blk tmpl;
            }
            break :blk first_template;
        },
        else => null,
    };
}

fn getLegacyChatTemplateFromConfig(chat_template_json: std.json.Value) ?[]const u8 {
    if (chat_template_json != .object) return null;
    const ct = chat_template_json.object.get("chat_template") orelse return null;
    return switch (ct) {
        .string => |s| s,
        else => null,
    };
}

fn isSupportedGemmaTemplate(source: []const u8) bool {
    const has_turn_tokens = std.mem.indexOf(u8, source, "<|turn>") != null and
        std.mem.indexOf(u8, source, "<turn|>") != null and
        std.mem.indexOf(u8, source, "model") != null;
    const can_start_generation = std.mem.indexOf(u8, source, "add_generation_prompt") != null and
        std.mem.indexOf(u8, source, "<|channel>thought") != null;

    return has_turn_tokens and can_start_generation;
}

const Config = struct {
    bos_token: ?[]const u8 = null,
    eos_token: ?[]const u8 = null,

    fn fromTokenizerConfig(allocator: std.mem.Allocator, config: std.json.Value) !Config {
        var out: Config = .{};
        errdefer out.deinit(allocator);

        if (config == .object) {
            if (config.object.get("bos_token")) |val| {
                if (extractTokenString(val)) |s| out.bos_token = try allocator.dupe(u8, s);
            }
            if (config.object.get("eos_token")) |val| {
                if (extractTokenString(val)) |s| out.eos_token = try allocator.dupe(u8, s);
            }
        }

        return out;
    }

    fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        if (self.bos_token) |token| allocator.free(token);
        if (self.eos_token) |token| allocator.free(token);
    }

    fn extractTokenString(val: std.json.Value) ?[]const u8 {
        return switch (val) {
            .string => |s| s,
            .object => |obj| if (obj.get("content")) |content| switch (content) {
                .string => |s| s,
                else => null,
            } else null,
            else => null,
        };
    }
};

fn parseTokenizerConfig(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !std.json.Parsed(std.json.Value) {
    const file = try repo.openFile(io, "tokenizer_config.json", .{});
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);
    var json_reader = std.json.Reader.init(allocator, &reader.interface);
    defer json_reader.deinit();

    return try std.json.parseFromTokenSource(std.json.Value, allocator, &json_reader, .{ .allocate = .alloc_if_needed });
}
