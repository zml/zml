const std = @import("std");
const c = @import("c");
const HFTokenizer = @import("hftokenizers").HFTokenizer;

pub fn main() !void {
    const tokenizer = try HFTokenizer.fromFile("/private/var/tmp/_bazel_steeve/a67b810d44f2a673ebbd5bab86ccd5cc/external/zml~~huggingface~Meta-Llama-3.1-8B-Instruct/tokenizer.json");
    defer tokenizer.deinit();

    const input = "Hello, world! plane pouet plane";
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    const encoded = try encoder.encode(input);
    var pouet = std.ArrayList(u32).init(std.heap.c_allocator);
    defer pouet.deinit();

    // try pouet.appendSlice(encoded.ids);

    var t = try std.time.Timer.start();
    for (0..100) |_| {
        try pouet.appendSlice(encoded);
        t.reset();
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        const decoded = try decoder.decode(pouet.items);
        const elapsed = t.lap();
        // std.debug.print("{any} {any} {d}us\n", .{tokenizer, encoded, elapsed / std.time.ns_per_us});
        std.debug.print("{any} {any} {s} {d}ns {d}us\n", .{ tokenizer, encoded, decoded, elapsed, elapsed / std.time.ns_per_us });
    }
}
