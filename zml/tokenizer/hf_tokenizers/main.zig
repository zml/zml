const std = @import("std");
const c = @import("c");

pub const HFTokenizers = opaque {
    pub fn init(model: []const u8) *HFTokenizers {
        return @ptrCast(c.hf_tokenizers_new(c.zig_slice{ .ptr = @constCast(model.ptr), .len = @intCast(model.len) }));
    }

    pub fn deinit(self: *HFTokenizers) void {
        return c.hf_tokenizers_drop(self);
    }

    pub fn encode(self: *HFTokenizers, input: []const u8) []i32 {
        const ret = c.hf_tokenizers_encode(self, c.zig_slice{ .ptr = @constCast(input.ptr), .len = @intCast(input.len) });
        return @as([*c]i32, @alignCast(@ptrCast(ret.ptr)))[0..ret.len];
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
    const encoded = HFTokenizers.encode(tokenizer, input);
    std.debug.print("{any} {any}\n", .{tokenizer, encoded});
}
