const std = @import("std");
const vaxis = @import("vaxis");

const OutputView = @This();

rows: std.ArrayList(usize) = .empty,
contentLen: usize = 0,
width: u16 = 0,
height: u16 = 1,
top: usize = 0,
following: bool = true,

pub fn deinit(self: *OutputView, allocator: std.mem.Allocator) void {
    self.rows.deinit(allocator);
}

pub fn reset(self: *OutputView) void {
    self.rows.clearRetainingCapacity();
    self.contentLen = 0;
    self.width = 0;
    self.top = 0;
    self.following = true;
}

pub fn scrollBy(self: *OutputView, delta: i32) void {
    self.top = if (delta < 0) self.top -| @as(usize, @intCast(-delta)) else self.top +| @as(usize, @intCast(delta));
    self.following = delta > 0 and self.top >= self.maxTop();
    self.top = @min(self.top, self.maxTop());
}

pub fn home(self: *OutputView) void {
    self.following = false;
    self.top = 0;
}

pub fn maxTop(self: *const OutputView) usize {
    return self.rows.items.len -| self.height;
}

/// Screen cells borrow frameAllocator's copies until Vaxis.render completes.
pub fn draw(self: *OutputView, allocator: std.mem.Allocator, frameAllocator: std.mem.Allocator, win: vaxis.Window, text: []const u8) !void {
    self.height = @max(1, win.height);
    try self.reflow(allocator, win, text);
    if (self.following) self.top = self.maxTop();
    self.top = @min(self.top, self.maxTop());
    const end = @min(self.rows.items.len, self.top + win.height);
    for (self.top..end) |row| {
        const startByte = self.rows.items[row];
        const endByte = if (row + 1 < self.rows.items.len) self.rows.items[row + 1] else text.len;
        const line = try frameAllocator.dupe(u8, std.mem.trimEnd(u8, text[startByte..endByte], "\n"));
        _ = win.print(&.{.{ .text = line }}, .{ .row_offset = @intCast(row - self.top), .wrap = .none });
    }
}

fn reflow(self: *OutputView, allocator: std.mem.Allocator, win: vaxis.Window, text: []const u8) !void {
    if (self.width != win.width or text.len < self.contentLen) self.rows.clearRetainingCapacity();
    if (self.rows.items.len > 0 and self.width == win.width and self.contentLen == text.len) return;
    self.width = win.width;
    if (self.rows.items.len == 0) try self.rows.append(allocator, 0);
    // Only the last wrapped line can change as streamed text is appended.
    const start = self.rows.getLast();
    var iter = vaxis.unicode.graphemeIterator(text[start..]);
    var offset = start;
    var col: usize = 0;
    while (iter.next()) |grapheme| {
        const bytes = grapheme.bytes(text[start..]);
        if (std.mem.eql(u8, bytes, "\n")) {
            try self.rows.append(allocator, offset + bytes.len);
            col = 0;
        } else {
            const width = win.gwidth(bytes);
            if (col > 0 and col + width > @max(1, win.width)) {
                try self.rows.append(allocator, offset);
                col = 0;
            }
            col += width;
        }
        offset += bytes.len;
    }
    self.contentLen = text.len;
}

test "wraps Unicode, extends streamed rows, and preserves scrollback until following resumes" {
    const allocator = std.testing.allocator;
    var vx = try vaxis.init(allocator, .{});
    var discard: std.Io.Writer.Discarding = .init(&.{});
    defer vx.deinit(allocator, &discard.writer);
    try vx.resize(allocator, &discard.writer, .{ .cols = 4, .rows = 2, .x_pixel = 0, .y_pixel = 0 });
    vx.screen.width_method = .unicode;
    var frame: std.heap.ArenaAllocator = .init(allocator);
    defer frame.deinit();
    var view: OutputView = .{};
    defer view.deinit(allocator);
    try view.draw(allocator, frame.allocator(), vx.window(), "abcd\n界e\nlast");
    try std.testing.expectEqualSlices(usize, &.{ 0, 5, 10 }, view.rows.items);
    try std.testing.expectEqual(@as(usize, 1), view.top);
    view.home();
    try view.draw(allocator, frame.allocator(), vx.window(), "abcd\n界e\nlast more");
    try std.testing.expectEqualSlices(usize, &.{ 0, 5, 10, 14, 18 }, view.rows.items);
    try std.testing.expectEqual(@as(usize, 0), view.top);
    view.scrollBy(100);
    try std.testing.expect(view.following);
    try std.testing.expectEqual(@as(usize, 3), view.top);
    try vx.resize(allocator, &discard.writer, .{ .cols = 12, .rows = 2, .x_pixel = 0, .y_pixel = 0 });
    try view.draw(allocator, frame.allocator(), vx.window(), "abcd\n界e\nlast more");
    try std.testing.expectEqualSlices(usize, &.{ 0, 5, 10 }, view.rows.items);
}
