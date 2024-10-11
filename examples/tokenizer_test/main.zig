const std = @import("std");
const zml = @import("zml");
const meta = zml.meta;
const flags = @import("tigerbeetle/flags");
const log = std.log.scoped(.tokenizer_test);

pub fn main() !void {
    const CliArgs = struct {
        pub const help =
            \\ tokenizer_test --tokenizer=vocab.json --prompt='path/to/prompt.txt'
        ;
        tokenizer: []const u8,
        prompt: []const u8,
    };

    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const tokenizer_path = cli_args.tokenizer;
    var tokenizer = try zml.aio.detectFormatAndLoadTokenizer(allocator, tokenizer_path);
    defer tokenizer.deinit();

    const prompt_file = try std.fs.cwd().openFile(cli_args.prompt, .{});
    defer prompt_file.close();
    const prompt = try prompt_file.readToEndAlloc(allocator, (try prompt_file.metadata()).size());
    defer allocator.free(prompt);
    const prompt_tok = tokenizer.encode(allocator, prompt, .{}) catch unreachable;
    defer allocator.free(prompt_tok);
    const outw = std.io.getStdOut().writer();
    for (prompt_tok, 0..) |tok, i| {
        if (i != prompt_tok.len - 1) {
            try outw.print("{d},", .{tok});
        } else {
            try outw.print("{d}", .{tok});
        }
    }
}
