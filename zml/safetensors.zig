const std = @import("std");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const Dims = Shape.DimsArray;

pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);

const log = std.log.scoped(.@"zml/safetensors");

const BYTES_HEADER = 8;
pub const max_positional_iovecs: usize = if (@TypeOf(std.posix.IOV_MAX) == void) 64 else std.posix.IOV_MAX;

pub const ExactPositionalError = std.Io.File.ReadPositionalError || error{
    OutOfBounds,
    UnexpectedEndOfFile,
};

/// Completes an exact absolute positional scatter read, resuming short reads
/// and respecting the platform iovec limit.
/// Scatters `[file_offset, file_offset + total)` into `buffers`, where
/// `total` is their combined length. The file may end early only past
/// `minimum` bytes: the bytes after it are padding a caller added to align
/// the read, and the end of the file cuts them. Returns the bytes read.
pub fn readFilePositionalAllV(
    io: std.Io,
    file: std.Io.File,
    buffers: []const []u8,
    file_offset: u64,
    minimum: u64,
) ExactPositionalError!u64 {
    var read_size: u64 = 0;
    for (buffers) |buffer| {
        const len = std.math.cast(u64, buffer.len) orelse return error.OutOfBounds;
        read_size = std.math.add(u64, read_size, len) catch return error.OutOfBounds;
    }
    _ = std.math.add(u64, file_offset, read_size) catch return error.OutOfBounds;
    if (minimum > read_size) return error.OutOfBounds;
    if (read_size == 0) return 0;

    var batch: [max_positional_iovecs][]u8 = undefined;
    var buffer_index: usize = 0;
    var buffer_offset: usize = 0;
    var completed: u64 = 0;
    while (completed < read_size) {
        var batch_len: usize = 0;
        var batch_size: u64 = 0;
        var scan_index = buffer_index;
        var scan_offset = buffer_offset;
        while (scan_index < buffers.len and batch_len < batch.len) : (scan_index += 1) {
            const buffer = buffers[scan_index];
            if (scan_offset >= buffer.len) {
                scan_offset = 0;
                continue;
            }
            batch[batch_len] = buffer[scan_offset..];
            batch_size += batch[batch_len].len;
            batch_len += 1;
            scan_offset = 0;
        }
        if (batch_len == 0) return error.UnexpectedEndOfFile;

        const absolute_offset = std.math.add(u64, file_offset, completed) catch return error.OutOfBounds;
        const bytes_read = try file.readPositional(io, batch[0..batch_len], absolute_offset);
        if (bytes_read == 0) {
            if (completed >= minimum) return completed;
            return error.UnexpectedEndOfFile;
        }
        completed += @intCast(bytes_read);
        // A short read past `minimum` leaves only padding unread, which no
        // caller addresses; the call that would fetch it (or confirm the
        // end of the file) starts at an unaligned offset, which a file read
        // directly rejects.
        if (bytes_read < batch_size and completed >= minimum) return completed;

        try advanceScatter(buffers, &buffer_index, &buffer_offset, bytes_read);
    }
    return completed;
}

fn advanceScatter(
    buffers: []const []u8,
    buffer_index: *usize,
    buffer_offset: *usize,
    bytes: usize,
) error{UnexpectedEndOfFile}!void {
    var remaining = bytes;
    while (remaining > 0) {
        while (buffer_index.* < buffers.len and buffer_offset.* == buffers[buffer_index.*].len) {
            buffer_index.* += 1;
            buffer_offset.* = 0;
        }
        if (buffer_index.* >= buffers.len) return error.UnexpectedEndOfFile;
        const available = buffers[buffer_index.*].len - buffer_offset.*;
        if (remaining < available) {
            buffer_offset.* += remaining;
            return;
        }
        remaining -= available;
        buffer_index.* += 1;
        buffer_offset.* = 0;
    }
}

const CountingPositionalIo = struct {
    calls: usize = 0,
    largest_scatter: usize = 0,
    last_offset: u64 = 0,
    max_result: ?usize = null,

    const vtable: std.Io.VTable = blk: {
        var result = std.testing.io.vtable.*;
        result.fileReadPositional = fileReadPositional;
        break :blk result;
    };

    fn io(self: *CountingPositionalIo) std.Io {
        return .{
            .userdata = self,
            .vtable = &vtable,
        };
    }

    fn fileReadPositional(
        userdata: ?*anyopaque,
        _: std.Io.File,
        data: []const []u8,
        offset: u64,
    ) std.Io.File.ReadPositionalError!usize {
        const self: *CountingPositionalIo = @ptrCast(@alignCast(userdata.?));
        self.calls += 1;
        self.largest_scatter = @max(self.largest_scatter, data.len);
        self.last_offset = offset;

        var requested: usize = 0;
        for (data) |buffer| requested +|= buffer.len;
        var remaining = @min(requested, self.max_result orelse requested);
        var position: usize = 0;
        for (data) |buffer| {
            const take = @min(remaining, buffer.len);
            for (buffer[0..take]) |*byte| {
                byte.* = @truncate(offset + position);
                position += 1;
            }
            remaining -= take;
            if (remaining == 0) break;
        }
        return position;
    }
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

test "readFilePositionalAllV batches more than IOV_MAX buffers" {
    const io = std.testing.io;
    const len = max_positional_iovecs + 1;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile(io, "tensor.bin", .{ .read = true });
    defer file.close(io);
    var contents: [len]u8 = undefined;
    for (&contents, 0..) |*byte, i| byte.* = @truncate(i);
    try file.writePositionalAll(io, &contents, 0);

    var output: [len]u8 = undefined;
    var buffers: [len][]u8 = undefined;
    for (&buffers, 0..) |*buffer, i| buffer.* = output[i..][0..1];
    try std.testing.expectEqual(@as(u64, len), try readFilePositionalAllV(io, file, &buffers, 0, len));
    try std.testing.expectEqualSlices(u8, &contents, &output);
}

test "readFilePositionalAllV resumes short reads" {
    var counting: CountingPositionalIo = .{ .max_result = 1 };
    const io = counting.io();
    const file: std.Io.File = .{ .handle = 0, .flags = .{ .nonblocking = false } };

    var output: [4]u8 = undefined;
    _ = try readFilePositionalAllV(io, file, &.{ output[0..1], output[1..3], output[3..4] }, 7, output.len);
    try std.testing.expectEqual(@as(usize, 4), counting.calls);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8, 9, 10 }, &output);
}

test "advanceScatter resumes partial reads across boundaries" {
    var storage: [6]u8 = undefined;
    const buffers = [_][]u8{ storage[0..2], storage[2..2], storage[2..5], storage[5..6] };
    var index: usize = 0;
    var offset: usize = 0;
    try advanceScatter(&buffers, &index, &offset, 1);
    try std.testing.expectEqual(@as(usize, 0), index);
    try std.testing.expectEqual(@as(usize, 1), offset);
    try advanceScatter(&buffers, &index, &offset, 2);
    try std.testing.expectEqual(@as(usize, 2), index);
    try std.testing.expectEqual(@as(usize, 1), offset);
    try advanceScatter(&buffers, &index, &offset, 3);
    try std.testing.expectEqual(@as(usize, 4), index);
    try std.testing.expectEqual(@as(usize, 0), offset);
    try std.testing.expectError(error.UnexpectedEndOfFile, advanceScatter(&buffers, &index, &offset, 1));
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
        _ = try readFilePositionalAllV(io, file, &.{buffer}, file_offset, buffer.len);
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

    /// The checkpoint in `repo` (`addRepo` with no prefix). `repo` stays
    /// the caller's to close.
    pub fn fromRepo(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
    ) !TensorRegistry {
        var registry: TensorRegistry = .init(allocator);
        errdefer registry.deinit();
        try registry.addRepo(io, repo, "");
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
        var repo = try resolveModelRepo(io, path);
        defer repo.close(io);

        // `repo` is the file's directory: the file opens by its base name.
        const entrypoint_file, const entrypoint_path = if (std.mem.endsWith(u8, path, ".safetensors.index.json") or std.mem.endsWith(u8, path, ".safetensors"))
            .{ try repo.openFile(io, std.fs.path.basename(path), .{ .mode = .read_only }), path }
        else
            try resolveModelEntrypoint(io, repo);
        defer entrypoint_file.close(io);

        var registry: TensorRegistry = .init(allocator);
        errdefer registry.deinit();
        try addCheckpoint(allocator, io, &registry, repo, entrypoint_file, entrypoint_path, "");
        return registry;
    }

    /// Adds every tensor of the checkpoint in `repo` (its
    /// `model.safetensors.index.json` and the files it lists, else
    /// `model.safetensors`) under `prefix` followed by its name. A second
    /// checkpoint, such as a speculative drafter's, joins the registry under
    /// a prefix of its own, and one store then holds both; reach the second
    /// through `View.withPrefix`. Adding under a prefix that a tensor
    /// already here is named under fails with `error.DuplicateTensor`,
    /// before anything is read. `repo` stays the caller's to close. A
    /// registry a store owns takes no more (`TensorStore.init`).
    pub fn addRepo(self: *TensorRegistry, io: std.Io, repo: std.Io.Dir, prefix: []const u8) !void {
        // A checkpoint added beside another owns its prefix: no tensor already
        // here may be named under it.
        for (self.tensors.keys()) |name| {
            if (std.mem.startsWith(u8, name, prefix)) return error.DuplicateTensor;
        }
        const entrypoint_file, const entrypoint_path = try resolveModelEntrypoint(io, repo);
        defer entrypoint_file.close(io);
        try addCheckpoint(self.arena.child_allocator, io, self, repo, entrypoint_file, entrypoint_path, prefix);
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

    pub fn iterator(self: *TensorRegistry) Tensors.Iterator {
        return self.tensors.iterator();
    }

    pub fn totalBytes(self: *TensorRegistry) u64 {
        var total: u64 = 0;

        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            total += entry.value_ptr.byteSize();
        }

        return total;
    }
};

/// Adds the tensors of the checkpoint whose entrypoint `model` is open at
/// `model_path` in `repo`, one safetensors file or an index of the files of
/// `repo` to read, to `registry` under `prefix`.
fn addCheckpoint(
    allocator: std.mem.Allocator,
    io: std.Io,
    registry: *TensorRegistry,
    repo: std.Io.Dir,
    model: std.Io.File,
    model_path: []const u8,
    prefix: []const u8,
) !void {
    if (std.mem.endsWith(u8, model_path, ".safetensors.index.json")) {
        const file_reader_buf = try allocator.alloc(u8, 64 * 1024);
        defer allocator.free(file_reader_buf);

        var index_reader = model.reader(io, file_reader_buf);

        var safetensors_index = try parseSafetensorsIndex(
            allocator,
            &index_reader.interface,
        );
        defer safetensors_index.deinit();

        return parseSafetensorsIndexFiles(
            allocator,
            io,
            registry,
            safetensors_index,
            repo,
            prefix,
        );
    }
    if (std.mem.endsWith(u8, model_path, ".safetensors")) {
        return parseSafetensors(allocator, io, registry, model, prefix);
    }
    return error.InvalidPath;
}

/// Adds the tensors of the safetensors `file` to `registry`, each named
/// `prefix` followed by its key in the header.
pub fn parseSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    registry: *TensorRegistry,
    file: std.Io.File,
    prefix: []const u8,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const header_buf = try allocator.alloc(u8, 64 * 1024);
    defer allocator.free(header_buf);

    var reader = file.reader(io, header_buf);

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
        const result = parseSafetensors(allocator, io, &registry, file, "");
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
        try registry.addRepo(io, drafter, "drafter.");
        try std.testing.expectError(error.DuplicateTensor, registry.addRepo(io, drafter, "drafter."));
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

test "fromPath opens a file or an index by a path relative to the cwd" {
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
    try tmp.dir.writeFile(io, .{ .sub_path = "layer.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"layer.safetensors\"}}" });

    for ([_][]const u8{ "layer.safetensors", "layer.safetensors.index.json" }) |name| {
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

const SafetensorsIndex = struct {
    arena: std.heap.ArenaAllocator,

    /// The files the index's `weight_map` names, each once.
    files: std.StringArrayHashMapUnmanaged(void),

    pub fn init(allocator: std.mem.Allocator) SafetensorsIndex {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .files = .empty,
        };
    }

    pub fn deinit(self: *SafetensorsIndex) void {
        const allocator = self.arena.allocator();
        self.files.deinit(allocator);
        self.arena.deinit();
    }
};

fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
) !SafetensorsIndex {
    var safetensors_index: SafetensorsIndex = .init(allocator);
    errdefer safetensors_index.deinit();

    const arena_allocator = safetensors_index.arena.allocator();
    var json_reader: std.json.Reader = .init(arena_allocator, reader);

    const index = try std.json.parseFromTokenSourceLeaky(
        std.json.Value,
        arena_allocator,
        &json_reader,
        .{ .allocate = .alloc_if_needed },
    );

    const weight_map = index.object.get("weight_map");

    if (weight_map) |wm| {
        var it = wm.object.iterator();

        while (it.next()) |entry| {
            try safetensors_index.files.put(arena_allocator, entry.value_ptr.string, {});
        }
    } else {
        log.warn("No weight_map attribute found in index", .{});
    }

    return safetensors_index;
}

fn parseSafetensorsIndexFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    registry: *TensorRegistry,
    safetensors_index: SafetensorsIndex,
    repo: std.Io.Dir,
    prefix: []const u8,
) !void {
    const filenames = safetensors_index.files.keys();
    for (filenames) |filename| {
        const file = try repo.openFile(io, filename, .{});
        defer file.close(io);
        try parseSafetensors(allocator, io, registry, file, prefix);
    }
}

pub const ModelPathResolutionError = error{
    FileNotFound,
    PermissionDenied,
    InvalidPath,
    Canceled,
} || std.mem.Allocator.Error;

pub fn resolveModelRepo(
    io: std.Io,
    path: []const u8,
) ModelPathResolutionError!std.Io.Dir {
    if (std.mem.endsWith(u8, path, ".safetensors.index.json") or
        std.mem.endsWith(u8, path, ".safetensors"))
    {
        const dir_path = std.fs.path.dirname(path) orelse ".";

        const dir = std.Io.Dir.openDir(.cwd(), io, dir_path, .{}) catch |e| {
            return switch (e) {
                error.FileNotFound => ModelPathResolutionError.FileNotFound,
                error.AccessDenied, error.PermissionDenied => ModelPathResolutionError.PermissionDenied,
                error.Canceled => ModelPathResolutionError.Canceled,
                else => ModelPathResolutionError.InvalidPath,
            };
        };

        return dir;
    }

    const dir = std.Io.Dir.openDir(.cwd(), io, path, .{}) catch |e| {
        return switch (e) {
            error.FileNotFound => ModelPathResolutionError.FileNotFound,
            error.AccessDenied, error.PermissionDenied => ModelPathResolutionError.PermissionDenied,
            error.Canceled => ModelPathResolutionError.Canceled,
            else => ModelPathResolutionError.InvalidPath,
        };
    };

    return dir;
}

fn resolveModelEntrypoint(io: std.Io, repo: std.Io.Dir) ModelPathResolutionError!struct { std.Io.File, []const u8 } {
    // When neither opens, a denied probe is what is reported. On S3 it may
    // mean absent: a bucket that may not be listed answers 403 for a
    // missing key.
    var denied = false;
    for ([_][]const u8{ "model.safetensors.index.json", "model.safetensors" }) |name| {
        if (repo.openFile(io, name, .{ .mode = .read_only })) |file| {
            return .{ file, name };
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
    var server = try VFS.mock_server.MockServer.init(io, &object, .{
        .objects = &.{.{ .path = "/bucket/served/model.safetensors", .bytes = &object }},
        .refusals = &.{
            .{ .path = "/bucket/served/model.safetensors.index.json", .status = .forbidden },
            .{ .path = "/bucket/denied/model.safetensors.index.json", .status = .forbidden },
            .{ .path = "/bucket/denied/model.safetensors", .status = .forbidden },
            .{ .path = "/bucket/missing/model.safetensors.index.json", .status = .not_found },
            .{ .path = "/bucket/missing/model.safetensors", .status = .not_found },
        },
    });
    var server_group: std.Io.Group = .init;
    try VFS.mock_server.startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer VFS.mock_server.cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var endpoint_buffer: [64]u8 = undefined;
    const endpoint = try std.fmt.bufPrint(&endpoint_buffer, "http://127.0.0.1:{d}", .{server.port()});
    var s3: VFS.S3 = try .init(allocator, io, &client, .{ .endpoint_url = endpoint, .region = "us-east-1" });
    defer s3.deinit();

    {
        const repo = try std.Io.Dir.openDir(.cwd(), s3.io(), "bucket/served", .{});
        defer repo.close(s3.io());
        const file, const name = try resolveModelEntrypoint(s3.io(), repo);
        defer file.close(s3.io());
        try std.testing.expectEqualStrings("model.safetensors", name);
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

    server_group.cancel(io);
    server_joined = true;
    try server.check();
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
