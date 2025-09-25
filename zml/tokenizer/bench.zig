const std = @import("std");

const stdx = @import("stdx");
const tokenizer = @import("zml/tokenizer");
const tools = @import("tools");

pub const std_options: std.Options = .{
    .log_level = .debug,
};

const Cli = struct {
    // model_dir: []const u8 = "/Users/guw/models/openai/gpt-oss-20b",
    model_dir: []const u8 = "/Users/guw/models/meta/Llama-3.1-8B-Instruct",
    input: []const u8 = "",
};

const sample_text = @embedFile("bench.zig");

const edge_cases =
    \\
    \\  two spaces
    \\   three spaces
    \\    four spaces
    \\     five spaces
    \\loading took {D}\nHellow world
;

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    var tracer = tools.Tracer.init("bench");

    const cli = stdx.flags.parseProcessArgs(Cli);
    const tokenizer_path = try std.fs.path.join(allocator, &.{ cli.model_dir, "tokenizer.json" });

    const text: []const u8 = if (cli.input.len > 0) txt: {
        const file = try std.fs.cwd().openFile(cli.input, .{});
        var allocating: std.Io.Writer.Allocating = try .initCapacity(allocator, (try file.stat()).size);
        var reader = file.reader(&.{});
        _ = try reader.interface.stream(&allocating.writer, .unlimited);
        break :txt allocating.written();
    } else sample_text;
    defer {
        if (text.ptr != sample_text.ptr) allocator.free(text);
    }

    var timer: std.time.Timer = try .start();

    var hf_tokenizer: tokenizer.Tokenizer = load: {
        timer.reset();
        defer std.debug.print("HUF tokenizer: loading took {D}\n", .{timer.lap()});

        break :load try .fromFile(allocator, tokenizer_path);
    };
    defer hf_tokenizer.deinit();

    var hf_encoder = try hf_tokenizer.encoder();
    defer hf_encoder.deinit();
    const hf_tokens = tok: {
        timer.reset();
        defer std.debug.print("HUF tokenizer: tokenization took {D}\n", .{timer.lap()});

        const f = tracer.frameStart("HF");
        defer tracer.frameEnd(f, "HF");
        break :tok try hf_encoder.encode(text);
    };

    var zml_tokenizer: tokenizer.Tokenizer = load: {
        timer.reset();
        defer std.debug.print("ZML tokenizer: loading took {D}\n", .{timer.lap()});

        const h = try allocator.create(tokenizer.homemade.Tokenizer);
        h.* = try tokenizer.homemade.fromHfJson(allocator, tokenizer_path);

        // for (&[_]u32{ 271, 309, 92, 3392, 21170, 77, 739, 672 }) |i| {
        //     std.debug.print("Tokens {d}: '{s}'\n", .{ i, h.lookupPiece(i) });
        // }
        break :load .{ .homemade = h };
    };
    defer zml_tokenizer.deinit();

    var zml_encoder = try zml_tokenizer.encoder();
    defer zml_encoder.deinit();
    const zml_tokens = tok: {
        timer.reset();
        defer std.debug.print("ZML tokenizer: tokenization took {D}\n", .{timer.lap()});

        const f = tracer.frameStart("ZML");
        defer tracer.frameEnd(f, "ZML");
        break :tok try zml_encoder.encode(text);
    };

    // std.debug.print("{s}\n", .{sample_text});
    std.debug.print("HUF {any}\n", .{hf_tokens});
    std.debug.print("ZML {any}\n", .{zml_tokens});
}
