const std = @import("std");

pub const StringReplaceReader = struct {
    interface: std.Io.Reader = .{
        .buffer = &.{},
        .vtable = &.{
            .stream = stream,
        },
        .seek = 0,
        .end = 0,
    },
    in: *std.Io.Reader,
    pattern: []const u8,
    replace_by: []const u8,

    pub fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *StringReplaceReader = @fieldParentPtr("interface", r);
        if (std.mem.eql(u8, try self.in.peek(self.pattern.len), self.pattern)) {
            self.in.toss(self.pattern.len);
            try w.writeAll(self.replace_by);
            return self.replace_by.len;
        }
        return self.in.streamDelimiterLimit(w, self.pattern[0], limit) catch |err| switch (err) {
            std.Io.Reader.StreamDelimiterLimitError.StreamTooLong => return 0,
            else => |e| return e,
        };
    }
};
