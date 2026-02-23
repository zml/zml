const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.tokenization_qwen2_fast);

pub const Qwen2TokenizerFast = struct {
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,

    pub const ChatMessage = struct {
        role: []const u8,
        content: []const u8,
    };

    pub fn fromPretrained(allocator: std.mem.Allocator, io: std.Io, repo_dir: std.Io.Dir, options: struct { subfolder: ?[]const u8 = "tokenizer" }) !@This() {
        const subfolder = options.subfolder orelse "";

        const tokenizer_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{subfolder});
        defer allocator.free(tokenizer_path);

        const bytes = label_block_bytes: {
            const file = try repo_dir.openFile(io, tokenizer_path, .{});
            defer file.close(io);
            var reader = file.reader(io, &.{});
            break :label_block_bytes try reader.interface.readAlloc(allocator, try file.length(io));
        };
        defer allocator.free(bytes);

        return @This(){
            .tokenizer = try zml.tokenizer.Tokenizer.fromBytes(allocator, io, bytes),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.tokenizer.deinit();
    }

    pub fn applyChatTemplate(self: @This(), messages: []const ChatMessage, options: struct {
        add_generation_prompt: bool = true,
    }) ![]const u8 {
        const im_start = "<|im_start|>";
        const im_sep = "\n";
        const im_end = "<|im_end|>\n";
        const gen_header = "<|im_start|>assistant\n";
        const gen_think = "<think>\n\n</think>\n\n";

        var total_size: usize = 0;

        for (messages) |msg| {
            total_size += im_start.len;
            total_size += msg.role.len;
            total_size += im_sep.len;
            total_size += msg.content.len;
            total_size += im_end.len;
        }

        if (options.add_generation_prompt) {
            total_size += gen_header.len + gen_think.len;
        }

        var result: std.ArrayList(u8) = try std.ArrayList(u8).initCapacity(self.allocator, total_size);
        errdefer result.deinit(self.allocator);

        for (messages) |msg| {
            try result.appendSlice(self.allocator, im_start);
            try result.appendSlice(self.allocator, msg.role);
            try result.appendSlice(self.allocator, im_sep);
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, im_end);
        }

        if (options.add_generation_prompt) {
            try result.appendSlice(self.allocator, gen_header);
            try result.appendSlice(self.allocator, gen_think);
        }

        return result.toOwnedSlice(self.allocator);
    }

    pub const TokenizeOutput = struct {
        input_ids: zml.Buffer,
        attention_mask: zml.Buffer,
        pub fn deinit(self: *@This()) void {
            self.input_ids.deinit();
            self.attention_mask.deinit();
        }
    };

    pub fn tokenize(self: *@This(), io: std.Io, platform: *const zml.Platform, text: []const u8, options: struct {
        max_length: usize = 512,
        truncation: bool = true,
    }) !TokenizeOutput {
        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();

        const ids: []const u32 = try encoder.encode(text);

        const pad_id: u32 = self.tokenizer.tokenToId("<|endoftext|>") orelse 151643;

        var final_ids: std.ArrayList(i64) = try std.ArrayList(i64).initCapacity(self.allocator, options.max_length);
        defer final_ids.deinit(self.allocator);

        var final_mask: std.ArrayList(i64) = try std.ArrayList(i64).initCapacity(self.allocator, options.max_length);
        defer final_mask.deinit(self.allocator);

        const len: usize = ids.len;
        const take: usize = if (options.truncation and len > options.max_length) options.max_length else len;

        for (0..take) |idx| {
            final_ids.appendAssumeCapacity(@intCast(ids[idx]));
            final_mask.appendAssumeCapacity(1);
        }

        if (take < options.max_length) {
            const pad_len = options.max_length - take;
            for (0..pad_len) |_| {
                final_ids.appendAssumeCapacity(@intCast(pad_id));
                final_mask.appendAssumeCapacity(0);
            }
        }

        const shape: zml.Shape = zml.Shape.init(.{ 1, @as(i64, @intCast(options.max_length)) }, .i64);

        return TokenizeOutput{
            .input_ids = try zml.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(final_ids.items)),
            .attention_mask = try zml.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(final_mask.items)),
        };
    }

    pub fn pipelineRun(allocator: std.mem.Allocator, io: std.Io, repo_dir: std.Io.Dir, platform: *const zml.Platform, progress: ?*std.Progress.Node, prompt: []const u8, max_length: usize) !TokenizeOutput {
        if (progress) |p| {
            p.increaseEstimatedTotalItems(1);
            var node = p.start("Executing tokenizer...", 1);
            defer node.end();
        }
        // Tokenizer Setup
        var tokenizer = try @This().fromPretrained(allocator, io, repo_dir, .{ .subfolder = "tokenizer" });
        defer tokenizer.deinit();
        const messages = [_]@This().ChatMessage{
            .{ .role = "user", .content = prompt },
        };
        const text_templated = try tokenizer.applyChatTemplate(&messages, .{
            .add_generation_prompt = true,
        });
        defer allocator.free(text_templated);
        // log.info("text_templated: from {s} to {s}", .{ prompt, text_templated });

        // Tokenize
        return try tokenizer.tokenize(io, platform, text_templated, .{
            .max_length = max_length,
            .truncation = true,
        });
    }
};
