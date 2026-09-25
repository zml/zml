const std = @import("std");
const log = std.log.scoped(.json_writer);

pub const State = enum(u8) { empty, in, closed };

pub fn dict(writer: anytype) @TypeOf(writer).Error!Dict(@TypeOf(writer)) {
    var top_level: List(@TypeOf(writer)) = .{ .writer = writer };
    return try top_level.openDict();
}

pub fn list(writer: anytype) @TypeOf(writer).Error!List(@TypeOf(writer)) {
    var top_level: List(@TypeOf(writer)) = .{ .writer = writer };
    return try top_level.openList();
}

pub fn Dict(Writer: type) type {
    const Error = Writer.Error;

    return struct {
        writer: Writer,
        state: State = .empty,

        const D = @This();

        pub fn write(self: *D, key: []const u8, x: anytype) Error!void {
            std.debug.assert(self.state != .closed);
            const comma = if (self.state == .empty) "" else ",";
            try std.fmt.format(self.writer, "{s}\"{s}\":", .{ comma, key });
            try std.json.stringify(x, .{}, self.writer);
            self.state = .in;
        }

        pub fn writeTemplate(self: *D, key: []const u8, comptime template: []const u8, args: anytype) Error!void {
            std.debug.assert(self.state != .closed);
            const comma = if (self.state == .empty) "" else ",";
            try std.fmt.format(self.writer, "{s}\"{s}\":", .{ comma, key });
            try std.fmt.format(self.writer, template, args);
            self.state = .in;
        }

        pub fn close(self: *D) void {
            std.debug.assert(self.state != .closed);
            self.writer.writeAll("}") catch {
                // No error allow to use close in a defer block.
                // But here we swallow the error which can go undetected if this was the last character of the file.
                // To do this properly we would need to specialized for a specific writer
            };
            self.state = .closed;
        }

        pub fn openList(self: *D, key: []const u8) Error!List(Writer) {
            std.debug.assert(self.state != .closed);
            const comma = if (self.state == .empty) "" else ",";
            try std.fmt.format(self.writer, "{s}\"{s}\":[", .{ comma, key });
            self.state = .in;
            return .{ .writer = self.writer };
        }

        pub fn openDict(self: *D, key: []const u8) Error!D {
            std.debug.assert(self.state != .closed);
            const comma = if (self.state == .empty) "" else ",";
            try std.fmt.format(self.writer, "{s}\"{s}\":{{", .{ comma, key });
            self.state = .in;
            return .{ .writer = self.writer };
        }
    };
}

pub fn List(Writer: type) type {
    const Error = Writer.Error;

    return struct {
        writer: Writer,
        state: State = .empty,

        const L = @This();

        pub fn write(self: *L, x: anytype) Error!void {
            std.debug.assert(self.state != .closed);
            const comma = if (self.state == .empty) "" else ",";
            try self.writer.writeAll(comma);
            try std.json.stringify(x, .{}, self.writer);
            self.state = .in;
        }

        pub fn writeTemplate(self: *L, comptime template: []const u8, args: anytype) Error!void {
            std.debug.assert(self.state != .closed);
            const comma = if (self.state == .empty) "" else ",";
            try self.writer.writeAll(comma);
            try std.fmt.format(self.writer, template, args);
            self.state = .in;
        }

        pub fn close(self: *L) void {
            std.debug.assert(self.state != .closed);
            self.writer.writeAll("]") catch {
                // we swallow the error which can go undetected if this was the last character
            };
            self.state = .closed;
        }

        pub fn openList(self: *L) Error!L {
            std.debug.assert(self.state != .closed);
            const comma = if (self.state == .empty) "" else ",";
            try std.fmt.format(self.writer, "{s}[", .{comma});
            self.state = .in;
            return .{ .writer = self.writer };
        }

        pub fn openDict(self: *L) Error!Dict(Writer) {
            std.debug.assert(self.state != .closed);
            const comma = if (self.state == .empty) "" else ",";
            try self.writer.print("{s}{{", .{comma});
            self.state = .in;
            return .{ .writer = self.writer };
        }
    };
}
