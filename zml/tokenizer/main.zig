const std = @import("std");
const tokenizer = @import("zml/tokenizer");

pub fn main() !void {
    const model2 = "/private/var/tmp/_bazel_steeve/a67b810d44f2a673ebbd5bab86ccd5cc/external/zml~~huggingface~Meta-Llama-3.1-8B-Instruct/tokenizer.json";

    var sp = try tokenizer.Tokenizer.from_file(std.heap.c_allocator, model2);
    defer sp.deinit();

    std.debug.print("Loaded model\n", .{});

    var encoder = try sp.encoder();
    defer encoder.deinit();

    var decoder = try sp.decoder();
    defer decoder.deinit();

    const ids = try encoder.encode("Hello, world! plane pouet plane");
    const decoded = try decoder.decode(ids);

    std.debug.print("{d}\n{s}\n", .{ ids, decoded });
}
