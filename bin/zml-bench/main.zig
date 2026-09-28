const std = @import("std");
const bench = @import("bench");
const tui = @import("tui");

const help =
    \\Usage: bazel run //bin/zml-bench -- [options]
    \\
    \\Benchmark an OpenAI-compatible streaming chat completions endpoint.
    \\  --endpoint URL       Default: http://127.0.0.1:8000/v1
    \\  --model NAME         Default: zml_model
    \\  --prompt TEXT        Default: Write a website in C
    \\  --batch N            Concurrent requests, 1-256 (default: 16)
    \\  --max-tokens N       Omit to use the server's limit
    \\  --temperature N      Nonnegative; omit to use the server default
    \\  --max-completion-tokens N  Supersedes --max-tokens
    \\  --thinking VALUE     auto/off/on, effort name, or budget 0-100
    \\  --reasoning-effort VALUE  Alias for --thinking (model dependent)
    \\  --system TEXT        System message
    \\  --stop TEXT          Stop string or JSON array of up to four strings
    \\  --tools JSON         JSON array of tool definitions
    \\  --documents JSON     JSON array of documents for the chat template
    \\  --report PATH        Save the final JSON report
    \\  --headless           Run one batch and print a JSON report
    \\  --help               Show this help
    \\
    \\Authentication: OPENAI_API_KEY environment variable (optional).
    \\TUI: Tab/Shift-Tab fields, Ctrl-S send, Esc stop, PgUp/PgDn scroll,
    \\     Ctrl-O options, Ctrl-R report, Ctrl-C quit. Click fields to edit, panes for full-screen output.
    \\     In full-screen output: Esc back, Home/End scroll/follow.
    \\
;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var config: bench.Config = .{};
    var headless = false;
    var reportPath: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help")) {
            var buffer: [4096]u8 = undefined;
            var stdout = std.Io.File.stdout().writer(init.io, &buffer);
            try stdout.interface.writeAll(help);
            try stdout.interface.flush();
            return;
        }
        if (std.mem.eql(u8, arg, "--headless")) {
            headless = true;
            continue;
        }
        i += 1;
        if (i == args.len) return error.MissingOptionValue;
        const value = args[i];
        if (std.mem.eql(u8, arg, "--endpoint")) {
            config.endpoint = value;
        } else if (std.mem.eql(u8, arg, "--model")) {
            config.model = value;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            config.prompt = value;
        } else if (std.mem.eql(u8, arg, "--batch")) {
            config.batch = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            config.maxTokens = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            config.temperature = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, arg, "--max-completion-tokens")) {
            config.maxCompletionTokens = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, arg, "--thinking") or std.mem.eql(u8, arg, "--reasoning-effort")) {
            config.reasoningEffort = value;
        } else if (std.mem.eql(u8, arg, "--system")) {
            config.systemPrompt = value;
        } else if (std.mem.eql(u8, arg, "--stop")) {
            config.stop = value;
        } else if (std.mem.eql(u8, arg, "--tools")) {
            config.tools = value;
        } else if (std.mem.eql(u8, arg, "--documents")) {
            config.documents = value;
        } else if (std.mem.eql(u8, arg, "--report")) {
            reportPath = value;
        } else return error.UnknownOption;
    }
    try config.validate();
    const apiKey = init.environ_map.get("OPENAI_API_KEY");
    if (!headless) return tui.run(allocator, init.io, config, apiKey, reportPath);

    const batch = try bench.Batch.create(allocator, init.io, config, apiKey);
    defer batch.destroy();
    while (batch.summary().active > 0) {
        try init.io.sleep(.fromMilliseconds(25), .awake);
    }
    const summary = batch.summary();
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    try batch.writeReport(&stdout.interface);
    if (reportPath) |path| try batch.saveReport(path);
    try stdout.interface.flush();
    if (summary.failed > 0 or summary.stopped > 0) return error.BenchmarkFailed;
}
