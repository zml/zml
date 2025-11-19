const std = @import("std");

pub const Reader = struct {
    reader: std.fs.File.Reader,
    remaining: std.Io.Limit,
    interface: std.Io.Reader,

    pub fn init(reader: std.fs.File.Reader, limit: std.Io.Limit) Reader {
        return .{
            .reader = reader,
            .remaining = limit,
            .interface = .{
                .vtable = &.{
                    .stream = stream,
                    .discard = discard,
                },
                .buffer = &.{},
                .seek = 0,
                .end = 0,
            },
        };
    }

    fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *Reader = @fieldParentPtr("interface", r);
        const combined_limit = limit.min(self.remaining);
        const n = try self.reader.interface.stream(w, combined_limit);
        self.remaining = self.remaining.subtract(n).?;
        return n;
    }

    fn discard(r: *std.Io.Reader, limit: std.Io.Limit) std.Io.Reader.Error!usize {
        const self: *Reader = @fieldParentPtr("interface", r);
        const combined_limit = limit.min(self.remaining);
        const n = try self.reader.interface.discard(combined_limit);
        self.remaining = self.remaining.subtract(n).?;
        return n;
    }
};

pub const Loader = struct {
    pub fn init() Loader {
        return .{};
    }

    pub fn deinit(self: Loader) void {
        _ = self; // autofix
    }

    pub fn open(self: Loader, allocator: std.mem.Allocator, uri: std.Uri) !Resource {
        _ = self; // autofix
        const resource: Resource = try .init(allocator, uri);

        return resource;
    }
};

pub const Resource = struct {
    file: std.fs.File,
    size: usize,
    start: usize,

    pub fn init(allocator: std.mem.Allocator, uri: std.Uri) !Resource {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        const raw_query = if (uri.query) |query| try query.toRawMaybeAlloc(arena.allocator()) else "";
        var it = std.mem.tokenizeScalar(u8, raw_query, '&');
        var start: ?usize = null;
        var end: ?usize = null;
        while (it.next()) |component| {
            var it2 = std.mem.tokenizeScalar(u8, component, '=');
            const key = it2.next().?;
            const value = it2.next() orelse "";
            if (std.mem.eql(u8, key, "start")) {
                start = try std.fmt.parseInt(usize, value, 10);
            } else if (std.mem.eql(u8, key, "end")) {
                end = try std.fmt.parseInt(usize, value, 10);
            }
        }

        std.debug.print("start: {any} - end: {any}\n", .{ start, end });

        const path = try uri.path.toRawMaybeAlloc(arena.allocator());

        const file = try std.fs.openFileAbsolute(path, .{});

        const size = end.? - start.?;

        return .{
            .file = file,
            .size = size,
            .start = start.?,
        };
    }

    pub fn deinit(self: Resource) void {
        self.file.close();
    }

    pub fn reader(self: Resource, buffer: []u8) Reader {
        var r = std.fs.File.Reader.init(self.file, buffer);
        r.seekTo(self.start) catch unreachable;
        const r2 = Reader.init(r, .limited(self.size));
        return r2;
    }
};
