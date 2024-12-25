const std = @import("std");
const c = @import("c");

pub const HFTokenizers = opaque {
    pub const Encoded = struct {
        ids: []const u32,

        pub fn deinit(self: *Encoded) void {
            c.hf_tokenizers_tokens_drop(c.zig_slice{ .ptr = @ptrCast(@constCast(self.ids.ptr)), .len = @intCast(self.ids.len) });
        }
    };

    pub const Decoded = struct {
        str: []const u8,

        pub fn deinit(self: *Decoded) void {
            c.hf_tokenizers_str_drop(c.zig_slice{ .ptr = @ptrCast(@constCast(self.str.ptr)), .len = @intCast(self.str.len) });
        }
    };
    pub fn init(model: []const u8) *HFTokenizers {
        return @ptrCast(c.hf_tokenizers_new(c.zig_slice{ .ptr = @constCast(model.ptr), .len = @intCast(model.len) }));
    }

    pub fn deinit(self: *HFTokenizers) void {
        return c.hf_tokenizers_drop(self);
    }

    pub fn encode(self: *HFTokenizers, input: []const u8) Encoded {
        const ret = c.hf_tokenizers_encode(self, c.zig_slice{ .ptr = @constCast(input.ptr), .len = @intCast(input.len) });
        return .{
            .ids = @as([*c]const u32, @alignCast(@ptrCast(ret.ptr)))[0..ret.len]
        };
    }

    pub fn decode(self: *HFTokenizers, input: []const u32) Decoded {
        const ret = c.hf_tokenizers_decode(self, c.zig_slice{ .ptr = @ptrCast(@constCast(input.ptr)), .len = @intCast(input.len) });
        return .{
            .str = @as([*c]const u8, @alignCast(@ptrCast(ret.ptr)))[0..ret.len]
        };
    }
};

pub fn as_path(path: []const u8) [std.fs.max_path_bytes:0]u8 {
    var result: [std.fs.max_path_bytes:0]u8 = undefined;
    @memcpy(result[0..path.len], path);
    result[path.len] = 0;
    return result;
}

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
        std.debug.print("{any} {any} {s} {d}ns {d}us\n", .{tokenizer, encoded, decoded.str, elapsed, elapsed / std.time.ns_per_us});
    }
}
