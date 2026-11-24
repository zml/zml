const zml = @import("zml");
const stdx = zml.stdx;

const Tensor = zml.Tensor;

/// Debug-assert that `x` has exactly the given tags (in any order), and return it.
pub fn assertedTags(x: Tensor, comptime tags: anytype) Tensor {
    const n_tags = tags.len;
    const expected = comptime blk: {
        // Braces are doubled because this is spliced into a format string.
        var s: []const u8 = "{{";
        for (tags, 0..) |t, i| {
            s = s ++ (if (i == 0) "" else ", ") ++ "." ++ @tagName(t);
        }
        break :blk s ++ "}}";
    };
    stdx.debug.assert(
        x.rank() == n_tags and x.shape().hasTags(tags),
        "Expected tensor with tags " ++ expected ++ ", got {f}",
        .{x},
    );
    return x;
}

/// Debug-assert that `x` has exactly the given tags (in any order).
pub fn assertTags(x: Tensor, comptime tags: anytype) void {
    _ = assertedTags(x, tags);
}

