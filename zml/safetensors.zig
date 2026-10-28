const std = @import("std");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const Dims = Shape.DimsArray;

pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);

const log = std.log.scoped(.@"zml/safetensors");

const BYTES_HEADER = 8;

pub const ExactPositionalError = std.Io.File.ReadPositionalError || error{
    OutOfBounds,
    UnexpectedEndOfFile,
};

test "Tensor.readRange reads tensor-relative ranges within the tensor" {
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

test "Tensor.readRange reports a truncated backing file" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile(io, "tensor.bin", .{ .read = true });
    defer file.close(io);
    try file.writePositionalAll(io, "prefix0123456789", 0);

    const tensor: Tensor = .{
        .file_uri = "unused",
        .name = "tensor",
        .shape = .init(.{12}, .u8),
        .offset = "prefix".len,
    };
    var tail: [3]u8 = undefined;
    try std.testing.expectError(error.UnexpectedEndOfFile, tensor.readRange(io, file, &tail, 9));
}

test "concurrent Tensor.readRange calls on one file preserve offsets" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile(io, "tensor.bin", .{ .read = true });
    defer file.close(io);
    try file.writePositionalAll(io, "prefixABCDEFGHI", 0);

    const tensor: Tensor = .{
        .file_uri = "unused",
        .name = "tensor",
        .shape = .init(.{9}, .u8),
        .offset = "prefix".len,
    };

    var chunks: [3][3]u8 = undefined;
    var start: std.Io.Event = .unset;
    var read_done: [3]std.Io.Event = @splat(.unset);
    var release_completion: [3]std.Io.Event = @splat(.unset);
    var completed: [3]std.Io.Event = @splat(.unset);
    var group: std.Io.Group = .init;
    const Job = struct {
        fn run(
            tensor_: *const Tensor,
            file_: std.Io.File,
            destination: []u8,
            offset: u64,
            start_: *std.Io.Event,
            read_done_: *std.Io.Event,
            release_completion_: *std.Io.Event,
            completed_: *std.Io.Event,
            io_: std.Io,
        ) std.Io.Cancelable!void {
            try start_.wait(io_);
            tensor_.readRange(io_, file_, destination, offset) catch unreachable;
            read_done_.set(io_);
            try release_completion_.wait(io_);
            completed_.set(io_);
        }
    };
    for (0..3) |i| {
        try group.concurrent(io, Job.run, .{
            &tensor,
            file,
            &chunks[i],
            i * 3,
            &start,
            &read_done[i],
            &release_completion[i],
            &completed[i],
            io,
        });
    }

    start.set(io);
    for (&read_done) |*event| try event.wait(io);
    for ([_]usize{ 2, 0, 1 }) |i| {
        release_completion[i].set(io);
        try completed[i].wait(io);
    }
    try group.await(io);

    var assembled: [9]u8 = undefined;
    for (&chunks, 0..) |*chunk, i| @memcpy(assembled[i * 3 ..][0..3], chunk);
    try std.testing.expectEqualStrings("ABCDEFGHI", &assembled);
}

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

    pub const ReadError = std.Io.File.OpenError || ExactPositionalError;

    pub fn byteSize(self: Tensor) u64 {
        return self.shape.byteSize();
    }

    /// Reads the whole tensor into `buffer`, which holds exactly its bytes
    /// (`OutOfBounds` otherwise). Opens `file_uri` through `io` for the
    /// call, failing as that io's open does (under a VFS, see its error
    /// contract); a file that ends inside the tensor is
    /// `UnexpectedEndOfFile`.
    pub fn read(self: Tensor, io: std.Io, buffer: []u8) ReadError!void {
        if (buffer.len != self.byteSize()) return error.OutOfBounds;
        const file = try std.Io.Dir.openFile(.cwd(), io, self.file_uri, .{ .mode = .read_only });
        defer file.close(io);
        return self.readRange(io, file, buffer, 0);
    }

    /// Reads `buffer.len` bytes of the tensor from `tensor_offset`, relative
    /// to its data, out of `file`, which the caller opened at `file_uri`. A
    /// range past the tensor is `OutOfBounds`, a file that ends inside it
    /// `UnexpectedEndOfFile`. The reads are positional, so concurrent calls
    /// may share one file.
    pub fn readRange(self: Tensor, io: std.Io, file: std.Io.File, buffer: []u8, tensor_offset: u64) ExactPositionalError!void {
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

/// The tensors of one or more checkpoints by name, read from their
/// safetensors headers; no tensor data is read. It owns every name and URI.
/// Build it complete, then hand it to `zml.io.TensorStore.init`, which owns
/// it from then on and frees it.
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
        const entrypoint, const kind = try resolveModelEntrypoint(io, repo);
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
    /// denied local file is `AccessDenied`). A header whose offsets leave
    /// the file, as in a truncated download, is `InvalidSafetensorsHeader`,
    /// logged.
    pub fn fromPath(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
    ) !TensorRegistry {
        const file_kind: ?EntrypointKind = if (std.mem.endsWith(u8, path, ".safetensors.index.json"))
            .index
        else if (std.mem.endsWith(u8, path, ".safetensors"))
            .safetensors
        else
            null;
        var repo = try resolveModelRepo(io, if (file_kind == null) path else std.fs.path.dirname(path) orelse ".");
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
    /// with `error.DuplicateTensor`, before anything is read. `repo` stays
    /// the caller's to close. A registry a store owns takes no more
    /// (`TensorStore.init`).
    pub fn addRepo(self: *TensorRegistry, io: std.Io, repo: std.Io.Dir, name: []const u8) !void {
        const prefix = try std.mem.concat(self.arena.child_allocator, u8, &.{ name, "." });
        defer self.arena.child_allocator.free(prefix);
        // A checkpoint added beside another owns its prefix: no tensor already
        // here may be named under it.
        for (self.tensors.keys()) |key| {
            if (std.mem.startsWith(u8, key, prefix)) return error.DuplicateTensor;
        }
        const entrypoint, const kind = try resolveModelEntrypoint(io, repo);
        defer entrypoint.close(io);
        try addCheckpoint(io, self, repo, entrypoint, kind, prefix);
    }

    /// Frees every entry, its name and URI. Not for a registry handed to a
    /// store, which `TensorStore.deinit` frees.
    pub fn deinit(self: *TensorRegistry) void {
        const allocator = self.arena.allocator();
        self.tensors.deinit(allocator);
        self.arena.deinit();
    }

    /// Adds `tensor`, its name and file URI copied. A registry a store owns
    /// takes no more (`TensorStore.init`).
    pub fn registerTensor(
        self: *TensorRegistry,
        tensor: Tensor,
    ) !void {
        const allocator = self.arena.allocator();

        var tensor_copy = tensor;

        tensor_copy.name = try allocator.dupe(u8, tensor.name);
        tensor_copy.file_uri = try allocator.dupe(u8, tensor.file_uri);

        try self.tensors.put(allocator, tensor_copy.name, tensor_copy);
    }
};

/// What a checkpoint's entrypoint is: an index of the safetensors files
/// holding its tensors, or the one file holding them.
const EntrypointKind = enum { index, safetensors };

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

            var index_reader = entrypoint.reader(io, try scratch.alloc(u8, 64 * 1024));
            var json_reader: std.json.Reader = .init(scratch, &index_reader.interface);
            const index = try std.json.parseFromTokenSourceLeaky(std.json.Value, scratch, &json_reader, .{ .allocate = .alloc_if_needed });
            const weight_map = index.object.get("weight_map") orelse {
                log.warn("No weight_map attribute found in index", .{});
                return;
            };

            // The files the weight map names, each read once.
            var files: std.StringArrayHashMapUnmanaged(void) = .empty;
            var it = weight_map.object.iterator();
            while (it.next()) |entry| try files.put(scratch, entry.value_ptr.string, {});
            for (files.keys()) |file_name| {
                const file = try repo.openFile(io, file_name, .{});
                defer file.close(io);
                try parseSafetensors(io, registry, file, prefix);
            }
        },
    }
}

/// Adds the tensors of the safetensors `file` to `registry`, each named
/// `prefix` followed by its key in the header.
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

    const json_header_length: u64 = try reader.interface.takeInt(u64, .little);
    // The header and every tensor's data must lie inside the file.
    const file_length = try file.length(io);
    if (json_header_length > file_length -| BYTES_HEADER) {
        log.warn("Invalid safetensors header in {s}: {d} bytes of header in a file of {d}", .{ file_uri, json_header_length, file_length });
        return error.InvalidSafetensorsHeader;
    }
    const data_start_offset = BYTES_HEADER + json_header_length;
    const json_data = try arena.allocator().alloc(u8, @intCast(json_header_length));

    try reader.interface.readSliceAll(json_data);

    const headers: std.json.Value = try std.json.parseFromSliceLeaky(
        std.json.Value,
        arena.allocator(),
        json_data,
        .{},
    );

    var it = headers.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        // Free-form strings, which nothing reads.
        if (std.mem.eql(u8, key, "__metadata__")) continue;

        const shape_field = value.object.get("shape").?.array;

        if (shape_field.items.len > Shape.MAX_RANK) {
            log.warn("Can't load tensor {s}, too many dims: {}", .{ key, shape_field.items.len });
            return error.TooManyDimensions;
        }

        const offset_field = value.object.get("data_offsets").?;
        const start = dataStart(offset_field, file_length - data_start_offset) orelse {
            log.warn("Invalid safetensors header in {s}: tensor {s} has data_offsets {f}, which must run forward within the file's {d} data bytes", .{ file_uri, key, std.json.fmt(offset_field, .{}), file_length - data_start_offset });
            return error.InvalidSafetensorsHeader;
        };
        const dtype = try stringToDtype(value.object.get("dtype").?.string);

        var dims: Dims = .empty;
        for (shape_field.items) |d| {
            dims.appendAssumeCapacity(d.integer);
        }

        const tensor: Tensor = .{
            .file_uri = file_uri,
            .name = if (prefix.len == 0) key else try std.mem.concat(arena.allocator(), u8, &.{ prefix, key }),
            .shape = .init(dims.slice(), dtype),
            .offset = data_start_offset + start,
        };

        try registry.registerTensor(tensor);
    }
}

/// The start of a tensor's `data_offsets`, `[start, end)` after the header,
/// when they run forward and end within `data_len` bytes.
fn dataStart(offsets: std.json.Value, data_len: u64) ?u64 {
    if (offsets != .array or offsets.array.items.len != 2) return null;
    var bounds: [2]u64 = undefined;
    for (&bounds, offsets.array.items) |*bound, item| bound.* = switch (item) {
        .integer => |integer| std.math.cast(u64, integer) orelse return null,
        // Integers past i64.
        .number_string => |digits| std.fmt.parseInt(u64, digits, 10) catch return null,
        else => return null,
    };
    if (bounds[0] > bounds[1] or bounds[1] > data_len) return null;
    return bounds[0];
}

test "parseSafetensors rejects a header or data_offsets that run backward, overflow or end past the file" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const max = "18446744073709551615";
    // One tensor of 4 bytes, the file's only data. `header_len` overrides
    // the header's length field.
    const cases = [_]struct { offsets: []const u8, header_len: ?u64 = null, valid: bool = false }{
        .{ .offsets = "[0,4]", .valid = true },
        .{ .offsets = "[4,0]" },
        .{ .offsets = "[0,5]" },
        .{ .offsets = "[-4,4]" },
        .{ .offsets = "[0," ++ max ++ "]" },
        .{ .offsets = "[" ++ max ++ "," ++ max ++ "]" },
        .{ .offsets = "[0,18446744073709551616]" },
        .{ .offsets = "[0]" },
        .{ .offsets = "[0,4]", .header_len = 1 << 20 },
        .{ .offsets = "[0,4]", .header_len = std.math.maxInt(u64) },
    };
    for (cases, 0..) |case, index| {
        var header_buffer: [128]u8 = undefined;
        const header = try std.fmt.bufPrint(&header_buffer, "{{\"x\":{{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":{s}}}}}", .{case.offsets});
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
        } else {
            try std.testing.expectError(error.InvalidSafetensorsHeader, result);
        }
    }
}

test "addRepo adds a second checkpoint under a prefix" {
    const TensorStore = @import("io/TensorStore.zig");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const writeCheckpoint = struct {
        /// A safetensors file of one tensor, `w`, of 4 bytes.
        fn call(io_: std.Io, dir: std.Io.Dir, sub_path: []const u8, bytes: *const [4]u8) !void {
            const header = "{\"w\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}";
            const file = try dir.createFile(io_, sub_path, .{});
            defer file.close(io_);
            var length: [BYTES_HEADER]u8 = undefined;
            std.mem.writeInt(u64, &length, header.len, .little);
            try file.writePositionalAll(io_, &length, 0);
            try file.writePositionalAll(io_, header, length.len);
            try file.writePositionalAll(io_, bytes, length.len + header.len);
        }
    }.call;
    // The target is one file, the drafter an index over one shard; both name
    // their tensor `w`.
    const target = try tmp.dir.createDirPathOpen(io, "target", .{});
    defer target.close(io);
    try writeCheckpoint(io, target, "model.safetensors", &.{ 1, 2, 3, 4 });
    const drafter = try tmp.dir.createDirPathOpen(io, "drafter", .{});
    defer drafter.close(io);
    try writeCheckpoint(io, drafter, "shard.safetensors", &.{ 5, 6, 7, 8 });
    try drafter.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"shard.safetensors\"}}" });

    var store: TensorStore = store: {
        var registry: TensorRegistry = try .fromRepo(allocator, io, target);
        errdefer registry.deinit();
        try registry.addRepo(io, drafter, "drafter");
        try std.testing.expectError(error.DuplicateTensor, registry.addRepo(io, drafter, "drafter"));
        break :store .init(allocator, registry, .{});
    };
    defer store.deinit();

    const root = store.view();
    try std.testing.expectEqual(@as(usize, 2), root.count());
    const cases = [_]struct { view: TensorStore.View, name: []const u8, bytes: [4]u8 }{
        .{ .view = root, .name = "w", .bytes = .{ 1, 2, 3, 4 } },
        .{ .view = root.withPrefix("drafter"), .name = "drafter.w", .bytes = .{ 5, 6, 7, 8 } },
    };
    for (cases) |case| {
        const tensor = case.view.get("w").?;
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
    {
        const header = "{\"w\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}";
        const file = try tmp.dir.createFile(io, "layer.safetensors", .{});
        defer file.close(io);
        var length: [BYTES_HEADER]u8 = undefined;
        std.mem.writeInt(u64, &length, header.len, .little);
        try file.writePositionalAll(io, &length, 0);
        try file.writePositionalAll(io, header, length.len);
        try file.writePositionalAll(io, &.{ 1, 2, 3, 4 }, length.len + header.len);
    }
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

fn resolveModelEntrypoint(io: std.Io, repo: std.Io.Dir) ModelPathResolutionError!struct { std.Io.File, EntrypointKind } {
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
            return .{ file, kind };
        } else |err| switch (err) {
            error.Canceled => return ModelPathResolutionError.Canceled,
            error.AccessDenied, error.PermissionDenied => denied = true,
            else => {},
        }
    }
    return if (denied) ModelPathResolutionError.PermissionDenied else ModelPathResolutionError.FileNotFound;
}

test "resolveModelEntrypoint falls back past a denied index and reports a denied probe" {
    const VFS = @import("vfs");
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // `served` has only the single file, behind an index S3 refuses with the
    // 403 a principal without `s3:ListBucket` gets for a missing key.
    var object: [8]u8 = @splat(0);
    var server = try VFS.MockServer.init(io, &object, .{
        .objects = &.{.{ .path = "/bucket/served/model.safetensors", .bytes = &object }},
        .refusals = &.{
            .{ .path = "/bucket/served/model.safetensors.index.json", .status = .forbidden },
            .{ .path = "/bucket/denied/model.safetensors.index.json", .status = .forbidden },
            .{ .path = "/bucket/denied/model.safetensors", .status = .forbidden },
            .{ .path = "/bucket/missing/model.safetensors.index.json", .status = .not_found },
            .{ .path = "/bucket/missing/model.safetensors", .status = .not_found },
        },
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var endpoint_buffer: [64]u8 = undefined;
    const endpoint = try std.fmt.bufPrint(&endpoint_buffer, "http://127.0.0.1:{d}", .{server.port()});
    var s3: VFS.S3 = try .init(allocator, io, &client, .{ .endpoint_url = endpoint, .region = "us-east-1" });
    defer s3.deinit();

    {
        const repo = try std.Io.Dir.openDir(.cwd(), s3.io(), "bucket/served", .{});
        defer repo.close(s3.io());
        const file, const kind = try resolveModelEntrypoint(s3.io(), repo);
        defer file.close(s3.io());
        try std.testing.expectEqual(.safetensors, kind);
    }
    {
        const repo = try std.Io.Dir.openDir(.cwd(), s3.io(), "bucket/denied", .{});
        defer repo.close(s3.io());
        try std.testing.expectError(error.PermissionDenied, resolveModelEntrypoint(s3.io(), repo));
    }
    {
        const repo = try std.Io.Dir.openDir(.cwd(), s3.io(), "bucket/missing", .{});
        defer repo.close(s3.io());
        try std.testing.expectError(error.FileNotFound, resolveModelEntrypoint(s3.io(), repo));
    }

    try server.stop(io);
}

fn stringToDtype(safetensor_type: []const u8) !DataType {
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

    return map.get(safetensor_type) orelse {
        log.err("Unsupported safetensor data type: {s}", .{safetensor_type});
        return error.UnsupportedDataType;
    };
}
