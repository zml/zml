const std = @import("std");
const zml = @import("zml");

pub const Qwen2TokenizerFast = struct {
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,

    pub const ChatMessage = struct {
        role: []const u8,
        content: []const u8,
    };

    pub fn from_pretrained(allocator: std.mem.Allocator, io: std.Io, repo_id: []const u8, options: struct { subfolder: ?[]const u8 = null }) !Qwen2TokenizerFast {
        var buffer: [1024]u8 = undefined;
        const subfolder = options.subfolder orelse "";

        // Try tokenizer.json first (Fast tokenizer)
        const path = try std.fmt.bufPrint(&buffer, "{s}/{s}/tokenizer.json", .{ repo_id, subfolder });

        // We use zml.tokenizer.Tokenizer.fromFile which detects json/pb
        const tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, path);

        return Qwen2TokenizerFast{
            .tokenizer = tokenizer,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Qwen2TokenizerFast) void {
        self.tokenizer.deinit();
    }

    pub fn apply_chat_template(self: Qwen2TokenizerFast, messages: []const ChatMessage, options: struct {
        tokenize: bool = false,
        add_generation_prompt: bool = true,
        enable_thinking: bool = false,
    }) ![]const u8 {
        var result = try std.ArrayList(u8).initCapacity(self.allocator, 1024);

        // Simple Qwen2 chat template implementation
        // <|im_start|>role\ncontent<|im_end|>\n

        for (messages) |msg| {
            try result.appendSlice(self.allocator, "<|im_start|>");
            try result.appendSlice(self.allocator, msg.role);
            try result.appendSlice(self.allocator, "\n");
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, "<|im_end|>\n");
        }

        if (options.add_generation_prompt) {
            try result.appendSlice(self.allocator, "<|im_start|>assistant\n");
            // To match Python output observed:
            try result.appendSlice(self.allocator, "<think>\n\n</think>\n\n");
        }

        return result.toOwnedSlice(self.allocator);
    }

    pub const TokenizeOutput = struct {
        input_ids: zml.Buffer,
        attention_mask: zml.Buffer,
    };

    pub fn tokenize(self: *Qwen2TokenizerFast, io: std.Io, platform: *const zml.Platform, text: []const u8, options: struct {
        padding: []const u8 = "max_length",
        max_length: usize = 20,
        truncation: bool = true,
        return_tensors: []const u8 = "pt",
    }) !TokenizeOutput {
        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const ids = try encoder.encode(text);

        const pad_id = self.tokenizer.tokenToId("<|endoftext|>") orelse 151643; // Default or internal

        var final_ids = try std.ArrayList(i64).initCapacity(self.allocator, options.max_length);
        defer final_ids.deinit(self.allocator);

        var final_mask = try std.ArrayList(i64).initCapacity(self.allocator, options.max_length);
        defer final_mask.deinit(self.allocator);

        const len = ids.len;
        const take = if (options.truncation and len > options.max_length) options.max_length else len;

        for (0..take) |i| {
            final_ids.appendAssumeCapacity(@intCast(ids[i]));
            final_mask.appendAssumeCapacity(1);
        }

        if (take < options.max_length) {
            const pad_len = options.max_length - take;
            for (0..pad_len) |_| {
                final_ids.appendAssumeCapacity(@intCast(pad_id));
                final_mask.appendAssumeCapacity(0);
            }
        }

        // Convert to zml.Buffer
        const shape = zml.Shape.init(.{ 1, @as(i64, @intCast(options.max_length)) }, .i64);

        const ids_buf = try zml.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(final_ids.items));
        const mask_buf = try zml.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(final_mask.items));

        return TokenizeOutput{
            .input_ids = ids_buf,
            .attention_mask = mask_buf,
        };
    }
};
