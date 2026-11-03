const std = @import("std");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const testing = @import("testing.zig");

pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);

const log = std.log.scoped(.@"zml/safetensors");

const BYTES_HEADER = 8;

/// A checkpoint tensor: its name, its shape, and where its bytes are, at
/// `offset` of the file `file_uri` names. The URI is what the io that read
/// the header resolved, a VFS URI under a VFS, so read the tensor through
/// an io that resolves it alike. A registry's entries own nothing; the
/// registry owns their strings.
pub const Tensor = struct {
    file_uri: []const u8,
    name: []const u8,
    shape: Shape,
    offset: u64,

    pub fn byteSize(self: Tensor) u64 {
        return self.shape.byteSize();
    }

    /// Reads the whole tensor into `buffer`, which holds exactly its bytes
    /// (`OutOfBounds` otherwise). Opens `file_uri` through `io` for the
    /// call, failing as that io's open does; a file that ends inside the
    /// tensor is `UnexpectedEndOfFile`.
    pub fn read(self: Tensor, io: std.Io, buffer: []u8) !void {
        if (buffer.len != self.byteSize()) return error.OutOfBounds;
        const file = try std.Io.Dir.openFile(.cwd(), io, self.file_uri, .{ .mode = .read_only });
        defer file.close(io);
        return self.readRange(io, file, buffer, 0);
    }

    pub const RangeError = std.Io.File.ReadPositionalError || error{ OutOfBounds, UnexpectedEndOfFile };

    /// Reads `buffer.len` bytes of the tensor from `tensor_offset`, relative
    /// to its data, out of `file`, which the caller opened at `file_uri`. A
    /// range past the tensor is `OutOfBounds`, a file that ends inside it
    /// `UnexpectedEndOfFile`. The reads are positional, so concurrent calls
    /// may share one file.
    pub fn readRange(self: Tensor, io: std.Io, file: std.Io.File, buffer: []u8, tensor_offset: u64) RangeError!void {
        const tensor_size = self.byteSize();
        if (tensor_offset > tensor_size or buffer.len > tensor_size - tensor_offset) return error.OutOfBounds;
        if (buffer.len == 0) return;
        const file_offset = std.math.add(u64, self.offset, tensor_offset) catch return error.OutOfBounds;
        _ = std.math.add(u64, file_offset, buffer.len) catch return error.OutOfBounds;
        if (try file.readPositionalAll(io, buffer, file_offset) != buffer.len) return error.UnexpectedEndOfFile;
    }

    pub fn format(self: Tensor, writer: *std.Io.Writer) !void {
        try writer.print("Tensor(name={s} shape={f} size={d}, offset={d}, file_uri={s})", .{
            self.name,
            self.shape,
            self.byteSize(),
            self.offset,
            self.file_uri,
        });
    }
};

test "Tensor.readRange reads tensor-relative ranges within the tensor and its file" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile(io, "tensor.bin", .{ .read = true });
    defer file.close(io);
    try file.writePositionalAll(io, "prefix0123456789", 0);

    var tensor: Tensor = .{
        .file_uri = "unused",
        .name = "tensor",
        .shape = .init(.{10}, .u8),
        .offset = "prefix".len,
    };
    var middle: [4]u8 = undefined;
    try tensor.readRange(io, file, &middle, 3);
    try std.testing.expectEqualStrings("3456", &middle);

    var empty: [0]u8 = .{};
    try tensor.readRange(io, file, &empty, 10);
    try std.testing.expectError(error.OutOfBounds, tensor.readRange(io, file, &empty, 11));

    var too_large: [2]u8 = undefined;
    try std.testing.expectError(error.OutOfBounds, tensor.readRange(io, file, &too_large, 9));

    // A file that ends inside the tensor.
    tensor.shape = .init(.{12}, .u8);
    var tail: [3]u8 = undefined;
    try std.testing.expectError(error.UnexpectedEndOfFile, tensor.readRange(io, file, &tail, 9));

    tensor.offset = std.math.maxInt(u64);
    var overflow: [1]u8 = undefined;
    try std.testing.expectError(error.OutOfBounds, tensor.readRange(io, file, &overflow, 0));
    try std.testing.expectError(error.OutOfBounds, tensor.readRange(io, file, &overflow, 1));
}

test "Tensor.read reads the whole tensor from its file" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile(io, "tensor.bin", .{ .read = true });
    try file.writePositionalAll(io, "prefix0123456789", 0);

    var path_buffer: [1024]u8 = undefined;
    const path_len = try file.realPath(io, &path_buffer);
    file.close(io);

    const tensor: Tensor = .{
        .file_uri = path_buffer[0..path_len],
        .name = "tensor",
        .shape = .init(.{10}, .u8),
        .offset = "prefix".len,
    };
    var whole: [10]u8 = undefined;
    try tensor.read(io, &whole);
    try std.testing.expectEqualStrings("0123456789", &whole);
    try std.testing.expectError(error.OutOfBounds, tensor.read(io, whole[0..9]));
}

/// The tensors of one or more checkpoints by name, read from their
/// safetensors headers; no tensor data is read. It owns every name and URI.
pub const TensorRegistry = struct {
    arena: std.heap.ArenaAllocator,

    tensors: Tensors,

    /// An empty registry.
    pub fn init(allocator: std.mem.Allocator) TensorRegistry {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
        };
    }

    /// The checkpoint in `repo`: its `model.safetensors.index.json` and the
    /// files it lists, else `model.safetensors`. `repo` stays the caller's
    /// to close.
    pub fn fromRepo(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
    ) !TensorRegistry {
        const entrypoint, const kind, _ = try resolveModelEntrypoint(io, repo);
        defer entrypoint.close(io);

        var registry: TensorRegistry = .init(allocator);
        errdefer registry.deinit();
        try addCheckpoint(io, &registry, repo, entrypoint, kind, "");
        return registry;
    }

    /// The checkpoint at `path`, read through `io`: a `.safetensors` file,
    /// a `.safetensors.index.json` and the files it lists, or a directory
    /// holding `model.safetensors.index.json` or `model.safetensors`. A
    /// relative path is from the cwd; under a VFS, `path` may be a remote
    /// URI. A missing path or entrypoint is `FileNotFound`, a denied
    /// directory or entrypoint `PermissionDenied`; a file opened by name,
    /// the one given or one an index lists, fails as `io`'s open does (a
    /// denied local file is `AccessDenied`). A malformed header, such as
    /// one whose offsets leave the file in a truncated download or one
    /// shorter than its 8-byte length field, or an index without a
    /// `weight_map` from tensor names to file names, is
    /// `InvalidSafetensorsHeader`, and a name two of its files repeat
    /// `DuplicateTensor`, both logged.
    pub fn fromPath(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
    ) !TensorRegistry {
        const repo_path, const file_kind = locateCheckpoint(path);
        var repo = try resolveModelRepo(io, repo_path);
        defer repo.close(io);
        const kind = file_kind orelse return fromRepo(allocator, io, repo);

        // `repo` is the file's directory: the file opens by its base name.
        const entrypoint = try repo.openFile(io, std.fs.path.basename(path), .{ .mode = .read_only });
        defer entrypoint.close(io);

        var registry: TensorRegistry = .init(allocator);
        errdefer registry.deinit();
        try addCheckpoint(io, &registry, repo, entrypoint, kind, "");
        return registry;
    }

    /// Adds every tensor of the checkpoint in `repo`, read as `fromRepo`
    /// reads it, under `name` and a dot: `View.withPrefix(name)` reaches
    /// them. A second checkpoint, such as a speculative drafter's, joins
    /// the registry under a name of its own, and one store then holds both.
    /// Adding under a name that a tensor already here is named under fails
    /// with `error.DuplicateTensor`, logged, before anything is read. `repo`
    /// stays the caller's to close.
    pub fn addRepo(self: *TensorRegistry, io: std.Io, repo: std.Io.Dir, name: []const u8) !void {
        const prefix = try std.mem.concat(self.arena.child_allocator, u8, &.{ name, "." });
        defer self.arena.child_allocator.free(prefix);
        // A checkpoint added beside another owns its prefix: no tensor already
        // here may be named under it.
        for (self.tensors.keys()) |key| {
            if (std.mem.startsWith(u8, key, prefix)) {
                log.warn("Cannot add a checkpoint under {s}: tensor {s} is already named under it", .{ name, key });
                return error.DuplicateTensor;
            }
        }
        const entrypoint, const kind, _ = try resolveModelEntrypoint(io, repo);
        defer entrypoint.close(io);
        try addCheckpoint(io, self, repo, entrypoint, kind, prefix);
    }

    /// Frees every entry, its name and URI.
    pub fn deinit(self: *TensorRegistry) void {
        const allocator = self.arena.allocator();
        self.tensors.deinit(allocator);
        self.arena.deinit();
    }

    /// Adds `tensor`, its name and file URI copied; a name already here is
    /// `DuplicateTensor`, logged.
    pub fn registerTensor(
        self: *TensorRegistry,
        tensor: Tensor,
    ) !void {
        const allocator = self.arena.allocator();

        var tensor_copy = tensor;

        tensor_copy.name = try allocator.dupe(u8, tensor.name);
        tensor_copy.file_uri = try allocator.dupe(u8, tensor.file_uri);

        const entry = try self.tensors.getOrPut(allocator, tensor_copy.name);
        if (entry.found_existing) {
            log.warn("Tensor {s} of {s} is already registered from {s}", .{ tensor.name, tensor.file_uri, entry.value_ptr.file_uri });
            return error.DuplicateTensor;
        }
        entry.value_ptr.* = tensor_copy;
    }
};

/// What a checkpoint's entrypoint is: an index of the safetensors files
/// holding its tensors, or the one file holding them.
const EntrypointKind = enum { index, safetensors };

/// Where the checkpoint at `path` is (`TensorRegistry.fromPath`): its
/// directory, and the kind of the file `path` names there, or null when
/// `path` names the directory itself.
fn locateCheckpoint(path: []const u8) struct { []const u8, ?EntrypointKind } {
    const kind: ?EntrypointKind = if (std.mem.endsWith(u8, path, ".safetensors.index.json"))
        .index
    else if (std.mem.endsWith(u8, path, ".safetensors"))
        .safetensors
    else
        null;
    return .{ if (kind == null) path else std.fs.path.dirname(path) orelse ".", kind };
}

/// Adds the tensors of the checkpoint whose entrypoint is open in `repo` to
/// `registry` under `prefix`. An index's files are read from `repo`.
fn addCheckpoint(
    io: std.Io,
    registry: *TensorRegistry,
    repo: std.Io.Dir,
    entrypoint: std.Io.File,
    kind: EntrypointKind,
    prefix: []const u8,
) !void {
    switch (kind) {
        .safetensors => return parseSafetensors(io, registry, entrypoint, prefix),
        .index => {
            var arena = std.heap.ArenaAllocator.init(registry.arena.child_allocator);
            defer arena.deinit();
            const scratch = arena.allocator();

            var index_uri_buf: [1024]u8 = undefined;
            const index_uri = index_uri_buf[0..try entrypoint.realPath(io, &index_uri_buf)];

            var index_reader = entrypoint.reader(io, try scratch.alloc(u8, 64 * 1024));
            const file_names = indexFiles(scratch, &index_reader.interface) catch |err| switch (err) {
                error.OutOfMemory => |e| return e,
                error.ReadFailed => return index_reader.err orelse err,
                else => {
                    log.warn("Invalid safetensors index {s}: needs a weight_map from tensor names to file names ({t})", .{ index_uri, err });
                    return error.InvalidSafetensorsHeader;
                },
            };
            for (file_names) |file_name| {
                const file = try repo.openFile(io, file_name, .{});
                defer file.close(io);
                try parseSafetensors(io, registry, file, prefix);
            }
        },
    }
}

/// The files a safetensors index maps tensors to, each once, in the order
/// its `weight_map` first names them, allocated in `arena`. An index
/// without a `weight_map` from tensor names to file names fails with the
/// JSON parser's error, unlogged.
fn indexFiles(arena: std.mem.Allocator, index: *std.Io.Reader) ![]const []const u8 {
    var json_reader: std.json.Reader = .init(arena, index);
    // The default allocation, `.alloc_always` over a `json.Reader`, copies
    // every string out of `index`'s buffer, which the next refill overwrites.
    const Index = struct { weight_map: std.json.ArrayHashMap([]const u8) };
    const parsed = try std.json.parseFromTokenSourceLeaky(Index, arena, &json_reader, .{ .ignore_unknown_fields = true });
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (parsed.weight_map.map.values()) |file_name| try files.put(arena, file_name, {});
    return files.keys();
}

/// The safetensors files `TensorRegistry.fromPath` reads for `path`, named
/// in their directory, which is returned open for the caller to close: the
/// file a `.safetensors` path names, the files a `.safetensors.index.json`
/// lists, else those the directory's `model.safetensors.index.json` lists,
/// or its `model.safetensors`. The names are allocated in `arena`. No
/// header is read and nothing is logged: a failure here is one `fromPath`
/// reports, with its context.
pub fn checkpointFiles(arena: std.mem.Allocator, io: std.Io, path: []const u8) !struct { std.Io.Dir, []const []const u8 } {
    const repo_path, const file_kind = locateCheckpoint(path);
    const repo = try std.Io.Dir.openDir(.cwd(), io, repo_path, .{});
    errdefer repo.close(io);
    const entrypoint, const kind, const name = if (file_kind) |kind|
        .{ try repo.openFile(io, std.fs.path.basename(path), .{ .mode = .read_only }), kind, std.fs.path.basename(path) }
    else
        try resolveModelEntrypoint(io, repo);
    defer entrypoint.close(io);
    const names = switch (kind) {
        .safetensors => try arena.dupe([]const u8, &.{name}),
        .index => names: {
            var index_reader = entrypoint.reader(io, try arena.alloc(u8, 64 * 1024));
            break :names indexFiles(arena, &index_reader.interface) catch |err| switch (err) {
                error.ReadFailed => return index_reader.err orelse err,
                else => |e| return e,
            };
        },
    };
    return .{ repo, names };
}

/// Adds the tensors of the safetensors `file` to `registry`, each named
/// `prefix` followed by its key in the header. A header that is not an
/// object of tensors, each with a `dtype`, a `shape` and `data_offsets`
/// spanning exactly its bytes within the file, is
/// `InvalidSafetensorsHeader`, and a name already registered
/// `DuplicateTensor`, both logged.
pub fn parseSafetensors(
    io: std.Io,
    registry: *TensorRegistry,
    file: std.Io.File,
    prefix: []const u8,
) !void {
    var arena = std.heap.ArenaAllocator.init(registry.arena.child_allocator);
    defer arena.deinit();

    var reader = file.reader(io, try arena.allocator().alloc(u8, 64 * 1024));

    var file_uri_buf: [1024]u8 = undefined;
    const file_uri_len = try file.realPath(io, &file_uri_buf);
    const file_uri = file_uri_buf[0..file_uri_len];

    // `File.Reader` returns `ReadFailed` with its cause in `err`, which a
    // failed seek leaves unset.
    const json_header_length: u64 = reader.interface.takeInt(u64, .little) catch |err| switch (err) {
        error.EndOfStream => {
            log.warn("Invalid safetensors header in {s}: shorter than its {d}-byte header length", .{ file_uri, BYTES_HEADER });
            return error.InvalidSafetensorsHeader;
        },
        error.ReadFailed => return reader.err orelse err,
    };
    // The header and every tensor's data must lie inside the file.
    const file_length = try file.length(io);
    if (json_header_length > file_length -| BYTES_HEADER) {
        log.warn("Invalid safetensors header in {s}: {d} bytes of header in a file of {d}", .{ file_uri, json_header_length, file_length });
        return error.InvalidSafetensorsHeader;
    }
    const data_start_offset = BYTES_HEADER + json_header_length;
    const json_data = try arena.allocator().alloc(u8, @intCast(json_header_length));

    reader.interface.readSliceAll(json_data) catch |err| return switch (err) {
        error.ReadFailed => reader.err orelse err,
        error.EndOfStream => err,
    };

    const headers = std.json.parseFromSliceLeaky(std.json.ArrayHashMap(std.json.Value), arena.allocator(), json_data, .{}) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        else => {
            log.warn("Invalid safetensors header in {s}: not a JSON object ({t})", .{ file_uri, err });
            return error.InvalidSafetensorsHeader;
        },
    };
    const data_len = file_length - data_start_offset;

    // `data_offsets` are `[start, end)` after the header.
    const Entry = struct { dtype: []const u8, shape: []const i64, data_offsets: [2]u64 };
    var it = headers.map.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;

        // Free-form strings, which nothing reads.
        if (std.mem.eql(u8, key, "__metadata__")) continue;

        const entry = std.json.parseFromValueLeaky(Entry, arena.allocator(), kv.value_ptr.*, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            else => {
                log.warn("Invalid safetensors header in {s}: tensor {s} needs a string dtype, an integer shape and two unsigned data_offsets ({t})", .{ file_uri, key, err });
                return error.InvalidSafetensorsHeader;
            },
        };

        if (entry.shape.len > Shape.MAX_RANK) {
            log.warn("Cannot load {s}: tensor {s} has {d} dims, more than {d}", .{ file_uri, key, entry.shape.len, Shape.MAX_RANK });
            return error.TooManyDimensions;
        }
        const dtype = stringToDtype(entry.dtype) orelse {
            log.warn("Cannot load {s}: tensor {s} has dtype {s}, which zml does not support", .{ file_uri, key, entry.dtype });
            return error.UnsupportedDataType;
        };

        const start, const end = entry.data_offsets;
        const byte_size = checkedByteSize(entry.shape, dtype);
        if (byte_size == null or start > end or end > data_len or end - start != byte_size.?) {
            log.warn("Invalid safetensors header in {s}: tensor {s} has data_offsets [{d},{d}], which must span the bytes of its {s} shape {any} within the file's {d} data bytes", .{ file_uri, key, start, end, entry.dtype, entry.shape, data_len });
            return error.InvalidSafetensorsHeader;
        }

        const tensor: Tensor = .{
            .file_uri = file_uri,
            .name = if (prefix.len == 0) key else try std.mem.concat(arena.allocator(), u8, &.{ prefix, key }),
            .shape = .init(entry.shape, dtype),
            .offset = data_start_offset + start,
        };

        try registry.registerTensor(tensor);
    }
}

/// The bytes of `dims` of `dtype`, as `Shape.byteSize` counts them, or null
/// on a negative dim or a bit count past u64.
fn checkedByteSize(dims: []const i64, dtype: DataType) ?u64 {
    var bits: u64 = dtype.bitSizeOf();
    for (dims) |d| bits = std.math.mul(u64, bits, std.math.cast(u64, d) orelse return null) catch return null;
    return std.math.divCeil(u64, bits, 8) catch unreachable;
}

test "parseSafetensors rejects a malformed header, data_offsets that do not span their tensor within the file, or a name already registered" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const max = "18446744073709551615";
    // One U8 tensor over the file's 4 data bytes. `header` overrides the
    // whole header, `header_len` the header's length field.
    const cases = [_]struct { offsets: []const u8 = "[0,4]", shape: []const u8 = "[4]", header: ?[]const u8 = null, header_len: ?u64 = null, valid: bool = false }{
        .{ .valid = true },
        .{ .offsets = "[4,0]" },
        .{ .offsets = "[0,3]" },
        .{ .offsets = "[0,5]" },
        .{ .offsets = "[-4,4]" },
        .{ .offsets = "[0," ++ max ++ "]" },
        .{ .offsets = "[" ++ max ++ "," ++ max ++ "]" },
        .{ .offsets = "[0,18446744073709551616]" },
        .{ .offsets = "[0]" },
        .{ .shape = "[-4]" },
        .{ .shape = "[4294967296,4294967296]" },
        .{ .header = "{\"x\":{\"dtype\":\"U8\",\"shape\":[4]}}" },
        .{ .header = "{\"x\":{\"dtype\":\"U8\",\"shape\":4,\"data_offsets\":[0,4]}}" },
        .{ .header = "[]" },
        .{ .header = "{\"x\":" },
        .{ .header_len = 1 << 20 },
        .{ .header_len = std.math.maxInt(u64) },
    };
    for (cases, 0..) |case, index| {
        var header_buffer: [128]u8 = undefined;
        const header = case.header orelse try std.fmt.bufPrint(&header_buffer, "{{\"x\":{{\"dtype\":\"U8\",\"shape\":{s},\"data_offsets\":{s}}}}}", .{ case.shape, case.offsets });
        var name_buffer: [32]u8 = undefined;
        const file = try tmp.dir.createFile(io, try std.fmt.bufPrint(&name_buffer, "{d}.safetensors", .{index}), .{ .read = true });
        defer file.close(io);
        var length: [BYTES_HEADER]u8 = undefined;
        std.mem.writeInt(u64, &length, case.header_len orelse header.len, .little);
        try file.writePositionalAll(io, &length, 0);
        try file.writePositionalAll(io, header, length.len);
        try file.writePositionalAll(io, &.{ 1, 2, 3, 4 }, length.len + header.len);

        var registry: TensorRegistry = .init(allocator);
        defer registry.deinit();
        const result = parseSafetensors(io, &registry, file, "");
        if (case.valid) {
            try result;
            try std.testing.expectEqual(@as(u64, BYTES_HEADER + header.len), registry.tensors.get("x").?.offset);
            // A name already registered, as one repeated across a checkpoint's
            // shards.
            try std.testing.expectError(error.DuplicateTensor, parseSafetensors(io, &registry, file, ""));
        } else {
            try std.testing.expectError(error.InvalidSafetensorsHeader, result);
        }
    }
}

test "addRepo adds a second checkpoint under a prefix" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // The target is one file, the drafter an index over one shard; both name
    // their tensor `w`.
    const target = try tmp.dir.createDirPathOpen(io, "target", .{});
    defer target.close(io);
    try testing.writeSafetensorsFile(allocator, io, target, "model.safetensors", &.{
        .{ .name = "w", .dtype = "U8", .dims = &.{4}, .bytes = &.{ 1, 2, 3, 4 } },
    });
    const drafter = try tmp.dir.createDirPathOpen(io, "drafter", .{});
    defer drafter.close(io);
    try testing.writeSafetensorsFile(allocator, io, drafter, "shard.safetensors", &.{
        .{ .name = "w", .dtype = "U8", .dims = &.{4}, .bytes = &.{ 5, 6, 7, 8 } },
    });
    try drafter.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"shard.safetensors\"}}" });

    var registry: TensorRegistry = try .fromRepo(allocator, io, target);
    defer registry.deinit();
    try registry.addRepo(io, drafter, "drafter");
    try std.testing.expectError(error.DuplicateTensor, registry.addRepo(io, drafter, "drafter"));

    try std.testing.expectEqual(@as(usize, 2), registry.tensors.count());
    const cases = [_]struct { name: []const u8, bytes: [4]u8 }{
        .{ .name = "w", .bytes = .{ 1, 2, 3, 4 } },
        .{ .name = "drafter.w", .bytes = .{ 5, 6, 7, 8 } },
    };
    for (cases) |case| {
        const tensor = registry.tensors.get(case.name).?;
        try std.testing.expectEqualStrings(case.name, tensor.name);
        var bytes: [4]u8 = undefined;
        try tensor.read(io, &bytes);
        try std.testing.expectEqualSlices(u8, &case.bytes, &bytes);
    }
}

test "fromPath opens a file, an index or a directory by a path relative to the cwd" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try testing.writeSafetensorsFile(allocator, io, tmp.dir, "layer.safetensors", &.{
        .{ .name = "w", .dtype = "U8", .dims = &.{4}, .bytes = &.{ 1, 2, 3, 4 } },
    });
    for ([_][]const u8{ "layer.safetensors.index.json", "model.safetensors.index.json" }) |index| {
        try tmp.dir.writeFile(io, .{ .sub_path = index, .data = "{\"weight_map\":{\"w\":\"layer.safetensors\"}}" });
    }

    for ([_][]const u8{ "layer.safetensors", "layer.safetensors.index.json", "." }) |name| {
        // `tmpDir` is `.zig-cache/tmp/<sub_path>` under the cwd.
        var path_buffer: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buffer, ".zig-cache/tmp/{s}/{s}", .{ &tmp.sub_path, name });
        var registry: TensorRegistry = try .fromPath(allocator, io, path);
        defer registry.deinit();
        var bytes: [4]u8 = undefined;
        try registry.tensors.get("w").?.read(io, &bytes);
        try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &bytes);
    }
}

test "fromPath refuses a file shorter than its header length or an index without a weight_map" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const files = [_]struct { []const u8, []const u8 }{
        .{ "short.safetensors", &.{ 1, 2, 3 } },
        .{ "list.safetensors.index.json", "[]" },
        .{ "unmapped.safetensors.index.json", "{\"metadata\":{\"total_size\":4}}" },
        .{ "unnamed.safetensors.index.json", "{\"weight_map\":{\"w\":4}}" },
    };
    for (files) |file| {
        const name, const data = file;
        try tmp.dir.writeFile(io, .{ .sub_path = name, .data = data });
        // `tmpDir` is `.zig-cache/tmp/<sub_path>` under the cwd.
        var path_buffer: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buffer, ".zig-cache/tmp/{s}/{s}", .{ &tmp.sub_path, name });
        try std.testing.expectError(error.InvalidSafetensorsHeader, TensorRegistry.fromPath(allocator, io, path));
    }
}

test "fromRepo reads every file an index longer than its read buffer names" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    // About 90 KiB of weight map, read through a 64 KiB buffer: file names
    // straddle refills.
    const shards = 16;
    const tensors_per_shard = 64;
    var index: std.Io.Writer.Allocating = .init(scratch);
    try index.writer.writeAll("{\"metadata\":{\"total_size\":1024},\"weight_map\":{");
    for (0..shards) |shard| {
        const file_name = try std.fmt.allocPrint(scratch, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ shard + 1, shards });
        var tensors: [tensors_per_shard]testing.TensorBytes = undefined;
        for (&tensors, 0..) |*tensor, expert| {
            const name = try std.fmt.allocPrint(scratch, "model.layers.{d}.mlp.experts.{d}.down_proj.weight", .{ shard, expert });
            tensor.* = .{ .name = name, .dtype = "U8", .dims = &.{1}, .bytes = &.{0} };
            try index.writer.print("{s}\"{s}\":\"{s}\"", .{ if (shard + expert == 0) "" else ",", name, file_name });
        }
        try testing.writeSafetensorsFile(allocator, io, tmp.dir, file_name, &tensors);
    }
    try index.writer.writeAll("}}");
    try std.testing.expect(index.written().len > 64 * 1024);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });

    var registry: TensorRegistry = try .fromRepo(allocator, io, tmp.dir);
    defer registry.deinit();
    // Each file holds its own tensors: a misread file name would lose some.
    try std.testing.expectEqual(@as(usize, shards * tensors_per_shard), registry.tensors.count());
}

test "checkpointFiles names the files fromPath reads" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    // The index names a twice and leaves out c, which sits beside its files
    // as a Mistral-format `consolidated.safetensors` does.
    try testing.writeSafetensorsFile(allocator, io, tmp.dir, "a.safetensors", &.{
        .{ .name = "w", .dtype = "U8", .dims = &.{1}, .bytes = &.{1} },
        .{ .name = "x", .dtype = "U8", .dims = &.{1}, .bytes = &.{2} },
    });
    try testing.writeSafetensorsFile(allocator, io, tmp.dir, "b.safetensors", &.{
        .{ .name = "y", .dtype = "U8", .dims = &.{1}, .bytes = &.{3} },
    });
    try testing.writeSafetensorsFile(allocator, io, tmp.dir, "c.safetensors", &.{
        .{ .name = "w", .dtype = "U8", .dims = &.{1}, .bytes = &.{4} },
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"a.safetensors\",\"y\":\"b.safetensors\",\"x\":\"a.safetensors\"}}" });
    const single = try tmp.dir.createDirPathOpen(io, "single", .{});
    defer single.close(io);
    try testing.writeSafetensorsFile(allocator, io, single, "model.safetensors", &.{
        .{ .name = "w", .dtype = "U8", .dims = &.{1}, .bytes = &.{5} },
    });

    const cases = [_]struct { []const u8, []const []const u8 }{
        .{ ".", &.{ "a.safetensors", "b.safetensors" } },
        .{ "model.safetensors.index.json", &.{ "a.safetensors", "b.safetensors" } },
        .{ "c.safetensors", &.{"c.safetensors"} },
        .{ "single", &.{"model.safetensors"} },
    };
    // `tmpDir` is `.zig-cache/tmp/<sub_path>` under the cwd.
    var path_buffer: [128]u8 = undefined;
    for (cases) |case| {
        const name, const expected = case;
        const path = try std.fmt.bufPrint(&path_buffer, ".zig-cache/tmp/{s}/{s}", .{ &tmp.sub_path, name });
        const repo, const files = try checkpointFiles(scratch, io, path);
        defer repo.close(io);
        try std.testing.expectEqual(expected.len, files.len);
        for (expected, files) |expected_file, file| {
            try std.testing.expectEqualStrings(expected_file, file);
            (try repo.openFile(io, file, .{})).close(io);
        }
        // The registry reads these files and no other.
        var registry: TensorRegistry = try .fromPath(allocator, io, path);
        defer registry.deinit();
        var read: std.StringArrayHashMapUnmanaged(void) = .empty;
        for (registry.tensors.values()) |tensor| try read.put(scratch, std.fs.path.basename(tensor.file_uri), {});
        try std.testing.expectEqual(read.count(), files.len);
        for (files) |file| try std.testing.expect(read.contains(file));
    }
    const missing = try std.fmt.bufPrint(&path_buffer, ".zig-cache/tmp/{s}/missing", .{&tmp.sub_path});
    try std.testing.expectError(error.FileNotFound, checkpointFiles(scratch, io, missing));
}

pub const ModelPathResolutionError = error{
    FileNotFound,
    PermissionDenied,
    InvalidPath,
    Canceled,
};

/// Opens the model directory at `path`, relative to the cwd. A path that is
/// not a directory, such as a file's, is `InvalidPath`, logged.
pub fn resolveModelRepo(
    io: std.Io,
    path: []const u8,
) ModelPathResolutionError!std.Io.Dir {
    return std.Io.Dir.openDir(.cwd(), io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => error.FileNotFound,
        error.AccessDenied, error.PermissionDenied => error.PermissionDenied,
        error.Canceled => error.Canceled,
        else => {
            log.warn("{s} is not a model directory: {t}", .{ path, err });
            return error.InvalidPath;
        },
    };
}

test "resolveModelRepo refuses a file" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "" });

    // `tmpDir` is `.zig-cache/tmp/<sub_path>` under the cwd.
    var path_buffer: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buffer, ".zig-cache/tmp/{s}/model.safetensors", .{&tmp.sub_path});
    try std.testing.expectError(error.InvalidPath, resolveModelRepo(io, path));
}

/// The entrypoint of the model directory `repo`, open, with its kind and
/// its name there: `model.safetensors.index.json`, else `model.safetensors`.
fn resolveModelEntrypoint(io: std.Io, repo: std.Io.Dir) ModelPathResolutionError!struct { std.Io.File, EntrypointKind, []const u8 } {
    // When neither opens, a denied probe is what is reported. On S3 it may
    // mean absent: a bucket that may not be listed answers 403 for a
    // missing key.
    var denied = false;
    const entrypoints = [_]struct { []const u8, EntrypointKind }{
        .{ "model.safetensors.index.json", .index },
        .{ "model.safetensors", .safetensors },
    };
    for (entrypoints) |entrypoint| {
        const name, const kind = entrypoint;
        if (repo.openFile(io, name, .{ .mode = .read_only })) |file| {
            return .{ file, kind, name };
        } else |err| switch (err) {
            error.Canceled => return ModelPathResolutionError.Canceled,
            error.AccessDenied, error.PermissionDenied => denied = true,
            else => {},
        }
    }
    return if (denied) ModelPathResolutionError.PermissionDenied else ModelPathResolutionError.FileNotFound;
}

test "resolveModelEntrypoint falls back past a denied index and reports a denied probe" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const served = try tmp.dir.createDirPathOpen(io, "served", .{});
    defer served.close(io);
    try served.writeFile(io, .{ .sub_path = "model.safetensors", .data = "" });

    // The testing io, but a missing file is `PermissionDenied`: S3 answers
    // 403 for a missing key to a principal without `s3:ListBucket`.
    var vtable = io.vtable.*;
    vtable.dirOpenFile = struct {
        fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.OpenFileOptions) std.Io.File.OpenError!std.Io.File {
            return std.testing.io.vtable.dirOpenFile(userdata, dir, sub_path, options) catch |err| switch (err) {
                error.FileNotFound => error.PermissionDenied,
                else => err,
            };
        }
    }.dirOpenFile;
    const unlisted: std.Io = .{ .userdata = io.userdata, .vtable = &vtable };

    const file, const kind, const name = try resolveModelEntrypoint(unlisted, served);
    defer file.close(io);
    try std.testing.expectEqual(.safetensors, kind);
    try std.testing.expectEqualStrings("model.safetensors", name);
    // `tmp.dir` holds neither entrypoint.
    try std.testing.expectError(error.PermissionDenied, resolveModelEntrypoint(unlisted, tmp.dir));
    try std.testing.expectError(error.FileNotFound, resolveModelEntrypoint(io, tmp.dir));
}

fn stringToDtype(safetensor_type: []const u8) ?DataType {
    const map = std.StaticStringMap(DataType).initComptime(.{
        .{ "F64", .f64 },
        .{ "F32", .f32 },
        .{ "F16", .f16 },
        .{ "BF16", .bf16 },
        .{ "F8_E8M0", .f8e8m0 },
        .{ "F8_E4M3", .f8e4m3fn },
        .{ "F4_E2M1", .f4e2m1 },
        .{ "I64", .i64 },
        .{ "I32", .i32 },
        .{ "I16", .i16 },
        .{ "I8", .i8 },
        .{ "U64", .u64 },
        .{ "U32", .u32 },
        .{ "U16", .u16 },
        .{ "U8", .u8 },
        .{ "BOOL", .bool },
    });

    return map.get(safetensor_type);
}
