const std = @import("std");

const zml = @import("zml");
const meta = zml.meta;
const flags = @import("tigerbeetle/flags");

const log = std.log.scoped(.tokenizer_test);

pub fn main() !void {
    const CliArgs = struct {
        pub const help =
            \\ tokenizer_test --tokenizer=vocab.json --prompt='Hello world' --expected=23,1304
        ;
        tokenizer: []const u8,
        prompt: []const u8,
        expected: []const u8,
    };

    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);
    const expected = try parseIntList(u32, allocator, cli_args.expected);
    defer allocator.free(expected);

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();

    const tokenizer_path = cli_args.tokenizer;
    log.info("   Loading tokenizer from {s}", .{tokenizer_path});
    var tokenizer = try zml.aio.detectFormatAndLoadTokenizer(allocator, tokenizer_path);
    log.info("✅ Loaded tokenizer from {s}: {?}, {}", .{ tokenizer_path, tokenizer.normalizer, tokenizer.special_tokens });
    defer tokenizer.deinit();

    const prompt_tok = tokenizer.encode(allocator, cli_args.prompt, .{}) catch unreachable;
    defer allocator.free(prompt_tok);

    std.testing.expectEqualSlices(u32, expected, prompt_tok) catch std.process.exit(1);
    log.info("✅ tests ok !", .{});
}

fn parseIntList(T: type, allocator: std.mem.Allocator, raw_bytes: []const u8) ![]const T {
    const expected_tokens = std.mem.count(u8, raw_bytes, ",") + 1;
    const expected = try allocator.alloc(u32, expected_tokens);
    var in_progress = raw_bytes;
    for (expected) |*token| {
        const has_next_comma = std.mem.indexOfScalar(u8, in_progress, ',');
        const token_str = if (has_next_comma) |n| in_progress[0..n] else in_progress;
        token.* = try std.fmt.parseInt(u32, token_str, 10);
        if (has_next_comma) |n| in_progress = in_progress[n + 1 ..];
    }
    return expected;
}
