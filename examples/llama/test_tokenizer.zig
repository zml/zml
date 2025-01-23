const std = @import("std");
const log = std.log.scoped(.@"//llama:test_tokenizer");

const flags = @import("tigerbeetle/flags");
const zml = @import("zml");

const Flags = struct {
    tokenizer: []const u8,
    prompt: []const u8 =
        \\Examples of titles:
        \\📉 Stock Market Trends
        \\🍪 Perfect Chocolate Chip Recipe
        \\Evolution of Music Streaming
        \\Remote Work Productivity Tips
        \\Artificial Intelligence in Healthcare
        \\🎮 Video Game Development Insights
        \\
    ,
    expected: []const u8 = "128000,41481,315,15671,512,9468,241,231,12937,8152,50730,198,9468,235,103,24118,39520,32013,26371,198,35212,3294,315,10948,45910,198,25732,5664,5761,1968,26788,198,9470,16895,22107,304,39435,198,9468,236,106,8519,4140,11050,73137,198",
    verbose: bool = false,
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    const allocator = gpa.allocator();

    var raw_args = std.process.args();
    const args = flags.parse(&raw_args, Flags);

    log.info("\tLoading tokenizer from {s}", .{args.tokenizer});
    var tokenizer = try zml.aio.detectFormatAndLoadTokenizer(allocator, args.tokenizer);
    log.info("✅\tLoaded tokenizer from {s}", .{args.tokenizer});
    defer tokenizer.deinit();

    const prompt_tok = try tokenizer.encode(allocator, args.prompt, .{ .debug = args.verbose });

    log.info("Input: {s}\nOutput: {d}", .{ args.prompt, prompt_tok });
    if (args.expected.len > 0) {
        var expected = try std.ArrayList(u32).initCapacity(allocator, args.prompt.len);
        var it = std.mem.splitSequence(u8, args.expected, ",");
        while (it.next()) |int_token| {
            const tok = try std.fmt.parseInt(u32, int_token, 10);
            try expected.append(tok);
        }
        if (std.mem.eql(u32, expected.items, prompt_tok)) {
            log.info("All good !", .{});
        } else {
            log.err("Doesn't match expected: {d}", .{expected.items});
        }
    }
}
