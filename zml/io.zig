const std = @import("std");
pub const file = @import("io/file.zig");

pub const ResourceKind = enum { file };

pub const Loader = struct {
    file: ?*file.Loader = null,

    pub fn init(allocator: std.mem.Allocator) Loader {
        _ = allocator; // autofix
        return .{};
    }

    pub fn deinit(self: Loader) void {
        _ = self; // autofix
    }

    pub fn open(self: Loader, allocator: std.mem.Allocator, uri: std.Uri) !Resource {
        inline for (std.meta.tags(ResourceKind)) |tag| {
            if (std.mem.eql(u8, uri.scheme, @tagName(tag))) {
                const maybe_loader = @field(self, @tagName(tag));
                if (maybe_loader) |loader| {
                    return @unionInit(Resource, @tagName(tag), try loader.open(allocator, uri));
                }
            }
        }
        return error.NoLoaderFound;
    }

    pub fn register(self: *Loader, kind: ResourceKind, loader: anytype) void {
        @field(self, @tagName(kind)) = loader;
    }
};

pub const Resource = union(ResourceKind) {
    file: file.Resource,

    pub fn deinit(self: Resource) void {
        switch (self) {
            inline else => |v| v.deinit(),
        }
    }

    pub fn reader(self: Resource, buffer: []u8) Reader {
        return switch (self) {
            .file => |v| .{ .file = v.reader(buffer) },
        };
    }
};

pub const Reader = union(ResourceKind) {
    file: file.Reader,

    pub fn interface(self: *Reader) *std.io.Reader {
        return switch (self.*) {
            .file => |*v| &v.interface,
        };
    }
};
