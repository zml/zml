const std = @import("std");
const c = @import("c");
const HFTokenizers = @import("hftokenizers").HFTokenizers;

pub fn main() !void {
    const tokenizer = HFTokenizers.init("/private/var/tmp/_bazel_steeve/a67b810d44f2a673ebbd5bab86ccd5cc/external/zml~~huggingface~Meta-Llama-3.1-8B-Instruct/tokenizer.json");
    defer HFTokenizers.deinit(tokenizer);

    const input = "Hello, world! plane pouet plane";
    var encoded = HFTokenizers.encode(tokenizer, input);
    defer encoded.deinit();
    var pouet = std.ArrayList(u32).init(std.heap.c_allocator);
    defer pouet.deinit();

    // try pouet.appendSlice(encoded.ids);

    var t = try std.time.Timer.start();
    for (0..100) |i| {
        _ = i; // autofix
        try pouet.appendSlice(encoded.ids);
        t.reset();
        var decoded = HFTokenizers.decode(tokenizer, pouet.items);
        defer decoded.deinit();
        const elapsed = t.lap();
        // std.debug.print("{any} {any} {d}us\n", .{tokenizer, encoded, elapsed / std.time.ns_per_us});
        std.debug.print("{any} {any} {s} {d}ns {d}us\n", .{ tokenizer, encoded, decoded.str, elapsed, elapsed / std.time.ns_per_us });
    }
}
