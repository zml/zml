const std = @import("std");

const safetensors = @import("safetensors.zig");
const Shape = @import("shape.zig").Shape;

const log = std.log.scoped(.@"zml/weights");

test {
    std.testing.refAllDecls(@This());
}

pub const Weights = std.StringArrayHashMapUnmanaged(Weight);
pub const Metadatas = safetensors.Metadatas;

pub const ReaderOpts = struct {
    alignment: ?std.mem.Alignment = null,
};

pub const Backend = union(enum) {
    safetensors: struct {
        file_uri: []const u8,
        offset: u64,
    },
};

pub const Weight = struct {
    name: []const u8,
    shape: Shape,
    backend: Backend,

    pub fn byteSize(self: Weight) u64 {
        return self.shape.byteSize();
    }

    pub fn reader(self: Weight, io: std.Io, buffer: []u8, opts: ReaderOpts) !WeightReader {
        return switch (self.backend) {
            .safetensors => |loc| .{
                .name = self.name,
                .shape = self.shape,
                .inner = .{ .safetensors = try safetensors.TensorReader.init(io, self.name, self.shape, loc.file_uri, loc.offset, buffer, opts) },
            },
        };
    }

    pub fn format(self: Weight, writer: *std.Io.Writer) !void {
        switch (self.backend) {
            .safetensors => |loc| try writer.print("Weight(name={s} shape={f} size={d}, offset={d}, file_uri={s})", .{
                self.name,
                self.shape,
                self.byteSize(),
                loc.offset,
                loc.file_uri,
            }),
        }
    }
};

pub const WeightReader = struct {
    name: []const u8,
    shape: Shape,
    inner: union(enum) {
        safetensors: safetensors.TensorReader,
    },

    pub fn interface(self: *WeightReader) *std.Io.Reader {
        return switch (self.inner) {
            .safetensors => |*reader| &reader.interface,
        };
    }

    pub fn deinit(self: *WeightReader) void {
        switch (self.inner) {
            .safetensors => |*reader| reader.deinit(),
        }
    }
};

pub const Registry = struct {
    arena: std.heap.ArenaAllocator,

    tensors: Weights,
    metadata: Metadatas,

    mutex: std.Io.Mutex = .init,

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
            .metadata = .{},
        };
    }

    pub fn initWithMetadata(
        allocator: std.mem.Allocator,
        metadata: Metadatas,
    ) !Registry {
        var self: Registry = .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
            .metadata = .empty,
        };

        try self.mergeMetadata(metadata);

        return self;
    }

    pub fn fromRepo(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
    ) !Registry {
        const entrypoint = safetensors.resolveModelEntrypoint(io, repo) catch |err| switch (err) {
            error.FileNotFound => {
                log.warn("Unsupported checkpoint: no safetensors model in repository", .{});
                return error.UnsupportedCheckpoint;
            },
            else => |e| return e,
        };
        return try safetensors.fetchRegistry(allocator, io, repo, entrypoint);
    }

    pub fn fromPath(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
    ) !Registry {
        if (isSafetensorsPath(path)) {
            var repo = try safetensors.resolveModelRepo(io, path);
            return try safetensors.fetchRegistry(allocator, io, repo, try repo.openFile(io, path, .{ .mode = .read_only }));
        }

        if (looksLikeCheckpointFile(path)) {
            log.warn("Unsupported checkpoint format: {s}", .{path});
            return error.UnsupportedCheckpoint;
        }

        const repo = safetensors.resolveModelRepo(io, path) catch |err| switch (err) {
            error.FileNotFound, error.InvalidPath => {
                log.warn("Unsupported checkpoint format: {s}", .{path});
                return error.UnsupportedCheckpoint;
            },
            else => |e| return e,
        };
        return fromRepo(allocator, io, repo);
    }

    pub fn deinit(self: *Registry) void {
        const allocator = self.arena.allocator();
        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }

    pub fn mergeMetadata(
        self: *Registry,
        other: Metadatas,
    ) !void {
        const allocator = self.arena.allocator();

        var it = other.iterator();
        while (it.next()) |entry| {
            const key = try allocator.dupe(u8, entry.key_ptr.*);
            const value = try entry.value_ptr.*.clone(allocator);

            const gop = try self.metadata.getOrPut(allocator, key);
            if (gop.found_existing) {
                gop.value_ptr.*.deinit(allocator);
                gop.value_ptr.* = value;
                log.debug("Overwrote existing metadata key={s} with value={f}", .{ key, value });
            } else {
                gop.value_ptr.* = value;
                log.debug("Added new metadata key={s} with value={f}", .{ key, value });
            }
        }
    }

    pub fn register(self: *Registry, weight: Weight) !void {
        const allocator = self.arena.allocator();

        var copy = weight;
        copy.name = try allocator.dupe(u8, weight.name);
        switch (copy.backend) {
            .safetensors => |*loc| loc.file_uri = try allocator.dupe(u8, loc.file_uri),
        }

        try self.tensors.put(allocator, copy.name, copy);
    }

    pub fn registerTensor(self: *Registry, weight: Weight) !void {
        return self.register(weight);
    }

    pub fn reader(
        self: *Registry,
        io: std.Io,
        tensor_name: []const u8,
        buffer: []u8,
    ) !WeightReader {
        const weight = self.tensors.get(tensor_name) orelse {
            log.err("Tensor {s} not found in registry", .{tensor_name});
            return error.TensorNotFound;
        };

        return try weight.reader(io, buffer, .{});
    }

    pub fn iterator(self: *Registry) Weights.Iterator {
        return self.tensors.iterator();
    }

    pub fn totalBytes(self: *Registry) u64 {
        var total: u64 = 0;

        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            total += entry.value_ptr.byteSize();
        }

        return total;
    }
};

pub fn isSafetensorsPath(path: []const u8) bool {
    return std.mem.endsWith(u8, path, ".safetensors.index.json") or
        std.mem.endsWith(u8, path, ".safetensors");
}

fn looksLikeCheckpointFile(path: []const u8) bool {
    const ext = std.fs.path.extension(path);
    return ext.len > 0 and !std.mem.eql(u8, ext, "");
}

test "sniff safetensors vs unknown checkpoint paths" {
    try std.testing.expect(isSafetensorsPath("model.safetensors"));
    try std.testing.expect(isSafetensorsPath("model.safetensors.index.json"));
    try std.testing.expect(!isSafetensorsPath("model.gguf"));
    try std.testing.expect(!isSafetensorsPath("model.pt"));
    try std.testing.expect(!isSafetensorsPath("/tmp/my-repo"));
}

test "fromPath rejects unknown checkpoint files" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    try std.testing.expectError(error.UnsupportedCheckpoint, Registry.fromPath(allocator, io, "model.gguf"));
    try std.testing.expectError(error.UnsupportedCheckpoint, Registry.fromPath(allocator, io, "weights.npz"));
}
