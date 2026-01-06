const std = @import("std");

const stdx = @import("stdx");
const zml_tokenizer = @import("zml/tokenizer");

const log = std.log.scoped(.@"//zml/tokenizer");

const Flags = struct {
    tokenizer: []const u8,
    prompt: []const u8,
    expected: []const u8 = "",
    verbose: bool = false,
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    const io = threaded.io();

    const args = stdx.flags.parseProcessArgs(Flags);

    log.info("\tLoading tokenizer from {s}", .{args.tokenizer});
    var tokenizer = try zml_tokenizer.Tokenizer.fromFile(allocator, io, args.tokenizer);
    log.info("✅\tLoaded tokenizer from {s}", .{args.tokenizer});
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    const prompt_tok = try encoder.encode(args.prompt);

    log.info("Input: {s}\nOutput: {any}", .{ args.prompt, prompt_tok });

    var errors: u8 = 0;
    {
        const reconstructed = try decoder.decode(prompt_tok);
        if (!std.mem.eql(u8, args.prompt, reconstructed)) {
            log.err("Reconstructed string from tokens doesn't match source: {s}", .{reconstructed});
            errors += 1;
        }
    }

    if (args.expected.len > 0) {
        var expected: std.ArrayList(u32) = try .initCapacity(allocator, args.prompt.len);
        var it = std.mem.splitSequence(u8, args.expected, ",");
        while (it.next()) |int_token| {
            const tok = try std.fmt.parseInt(u32, int_token, 10);
            try expected.append(allocator, tok);
        }
        if (!std.mem.eql(u32, expected.items, prompt_tok)) {
            log.err("Doesn't match expected: {any}", .{expected.items});
            errors += 1;
        }
    }

    if (errors == 0) log.info("All good !", .{});

    std.process.exit(errors);
}
