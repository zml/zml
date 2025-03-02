const asynk = @import("async");
const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../../zml.zig");
const pickle = @import("pickle.zig");
const py = @import("py.zig");
const eval = @import("eval.zig");
const HostBuffer = zml.HostBuffer;

const testing = std.testing;
const log = std.log.scoped(.@"zml/aio");

// TODO(cryptodeal): use zml.aio.PrefixBuilder instead
const StringBuilder = std.ArrayListUnmanaged(u8);

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(File);
}

pub const File = struct {
    buffer_file: zml.aio.MemoryMappedFile,
    /// Map names to sub file
    file_map: std.StringArrayHashMapUnmanaged(FileEntry) = .{},
    tar_file: ?TarStream = null,
    is_zip_file: bool,
    zip_prefix: []const u8 = &.{},
    pickle_subfile: struct { start: u64 = 0, len: usize },

    pub const FileEntry = struct {
        version_needed_to_extract: u16,
        flags: u16,
        compression_method: std.zip.CompressionMethod,
        last_modification_time: u16,
        last_modification_date: u16,
        header_zip_offset: u64,
        crc32: u32,
        filename_len: u32,
        compressed_size: u64,
        uncompressed_size: u64,
        file_offset: u64,

        pub fn init(entry: anytype) FileEntry {
            return .{
                .version_needed_to_extract = entry.version_needed_to_extract,
                .flags = @as(u16, @bitCast(entry.flags)),
                .compression_method = entry.compression_method,
                .last_modification_time = entry.last_modification_time,
                .last_modification_date = entry.last_modification_date,
                .header_zip_offset = entry.header_zip_offset,
                .crc32 = entry.crc32,
                .filename_len = entry.filename_len,
                .compressed_size = entry.compressed_size,
                .uncompressed_size = entry.uncompressed_size,
                .file_offset = entry.file_offset,
            };
        }
    };

    const magic = "PK\x03\x04";

    pub fn fromTarFile(allocator: std.mem.Allocator, mapped: zml.aio.MemoryMappedFile, file: std.tar.Iterator(asynk.File.Reader).File) !File {
        const tar_file = try TarStream.init(file);
        const file_magic = try tar_file.reader().readBytesNoEof(magic.len);
        try tar_file.seekTo(0);
        var res: File = .{
            .buffer_file = mapped,
            .tar_file = tar_file,
            .is_zip_file = std.mem.eql(u8, &file_magic, magic),
            .pickle_subfile = .{ .len = try tar_file.getEndPos() },
        };
        if (res.is_zip_file) {
            try res.parseZipHeaders(allocator, tar_file.seekableStream());
        }
        return res;
    }

    pub fn init(allocator: std.mem.Allocator, mmap_file: zml.aio.MemoryMappedFile) !File {
        const file_magic = try mmap_file.file.reader().readBytesNoEof(magic.len);
        try mmap_file.file.seekTo(0);
        var res: File = .{
            .buffer_file = mmap_file,
            .is_zip_file = std.mem.eql(u8, &file_magic, magic),
            .pickle_subfile = .{ .len = mmap_file.data.len },
        };

        if (res.is_zip_file) {
            try res.parseZipHeaders(allocator, mmap_file.file.seekableStream());
        }
        return res;
    }

    pub fn close(self: *File) void {
        self.buffer_file.deinit();
    }

    pub fn parsePickle(self: *File, allocator: std.mem.Allocator) ![]const pickle.Op {
        return if (self.tar_file) |tar_file| {
            try tar_file.seekTo(self.pickle_subfile.start);
            var buffered = std.io.bufferedReader(tar_file.reader());
            return try pickle.parse(allocator, buffered.reader(), self.pickle_subfile.len);
        } else {
            const file = self.buffer_file.file;
            try file.seekTo(self.pickle_subfile.start);
            var buffered = std.io.bufferedReader(file.reader());
            return try pickle.parse(allocator, buffered.reader(), self.pickle_subfile.len);
        };
    }

    fn parseZipHeaders(self: *File, allocator: std.mem.Allocator, seekable_stream: anytype) !void {
        var file_map: std.StringArrayHashMapUnmanaged(FileEntry) = .{};

        var iter = try std.zip.Iterator(@TypeOf(seekable_stream)).init(seekable_stream);
        var filename_buf: [std.fs.max_path_bytes]u8 = undefined;
        while (try iter.next()) |entry| {
            const filename = filename_buf[0..entry.filename_len];
            try seekable_stream.seekTo(entry.header_zip_offset + @sizeOf(std.zip.CentralDirectoryFileHeader));
            const len = try seekable_stream.context.reader().readAll(filename);
            if (len != filename.len) return error.ZipBadFileOffset;
            if (isBadFilename(filename)) return error.ZipBadFilename;
            std.mem.replaceScalar(u8, filename, '\\', '/'); // normalize path separators
            try file_map.put(allocator, try allocator.dupe(u8, filename), FileEntry.init(entry));
        }

        self.file_map = file_map;
        var file_iter = file_map.iterator();
        while (file_iter.next()) |e| {
            const entry = e.value_ptr.*;
            const filename = e.key_ptr.*;
            if (!std.mem.endsWith(u8, filename, "data.pkl")) continue;

            self.zip_prefix = filename[0 .. filename.len - "data.pkl".len];

            const local_data_header_offset: u64 = local_data_header_offset: {
                switch (entry.compression_method) {
                    .store => {},
                    .deflate => {
                        // TODO(cryptodeal): handle decompress
                        @panic("TODO support use of `deflate`");
                    },
                    else => @panic("TODO support other modes of compression"),
                }
                const local_header = blk: {
                    try seekable_stream.seekTo(entry.file_offset);
                    break :blk try seekable_stream.context.reader().readStructEndian(std.zip.LocalFileHeader, .little);
                };
                if (!std.mem.eql(u8, &local_header.signature, &std.zip.local_file_header_sig))
                    return error.ZipBadFileOffset;
                if (local_header.version_needed_to_extract != entry.version_needed_to_extract)
                    return error.ZipMismatchVersionNeeded;
                if (local_header.last_modification_time != entry.last_modification_time)
                    return error.ZipMismatchModTime;
                if (local_header.last_modification_date != entry.last_modification_date)
                    return error.ZipMismatchModDate;

                if (@as(u16, @bitCast(local_header.flags)) != entry.flags)
                    return error.ZipMismatchFlags;
                if (local_header.crc32 != 0 and local_header.crc32 != entry.crc32)
                    return error.ZipMismatchCrc32;
                if (local_header.compressed_size != 0 and
                    local_header.compressed_size != entry.compressed_size)
                    return error.ZipMismatchCompLen;
                if (local_header.uncompressed_size != 0 and
                    local_header.uncompressed_size != entry.uncompressed_size)
                    return error.ZipMismatchUncompLen;
                if (local_header.filename_len != entry.filename_len)
                    return error.ZipMismatchFilenameLen;

                break :local_data_header_offset @as(u64, local_header.filename_len) +
                    @as(u64, local_header.extra_len);
            };

            const local_data_file_offset: u64 =
                @as(u64, entry.file_offset) +
                @as(u64, @sizeOf(std.zip.LocalFileHeader)) +
                local_data_header_offset;
            self.pickle_subfile = .{ .start = local_data_file_offset, .len = entry.uncompressed_size };
            return;
        }

        log.err("Could not find file ending in `data.pkl` in archive", .{});
        return error.PickleNotFound;
    }

    fn basicTypeCheck(object: *const py.Object, module: []const u8, class: []const u8) bool {
        return switch (object.member) {
            .raw => |raw| return (std.mem.eql(u8, module, raw.global.module) and
                std.mem.eql(u8, class, raw.global.class)),
            else => false,
        };
    }

    pub fn parseModel(self: File, values: []const py.Any, store: *zml.aio.BufferStore) !void {
        var prefix_buf: [1024]u8 = undefined;
        const allocator = store.arena.allocator();
        for (values) |item| {
            try self.parseValue(allocator, store, StringBuilder.initBuffer(&prefix_buf), item);
        }
    }

    pub fn parseValue(self: File, allocator: std.mem.Allocator, store: *zml.aio.BufferStore, prefix: StringBuilder, v: py.Any) !void {
        // log.warn("Parsing {}", .{v});
        switch (v) {
            .app, .object, .global => |object| {
                if (!(try self.parseTorchGlobal(allocator, store, prefix, v))) {
                    try self.parseValue(allocator, store, prefix, object.member);
                    for (object.args) |item| {
                        try self.parseValue(allocator, store, prefix, item);
                    }
                    if (object.kwargs.len % 2 != 0) return error.InvalidInput;
                    const n_kwargs = @divExact(object.kwargs.len, 2);

                    for (0..n_kwargs) |i| {
                        const key, const val = object.kwargs[2 * i ..][0..2].*;
                        // kwargs can only be keyed by string.
                        if (key != .string) return error.InvalidInput;
                        // Handle Pytorch specific fields
                        const s = key.string;
                        if (std.mem.eql(u8, s, "_modules") or std.mem.eql(u8, s, "_parameters") or std.mem.eql(u8, s, "_buffers")) {
                            try self.parseValue(allocator, store, prefix, val);
                        } else {
                            var new_prefix = prefix;
                            if (prefix.items.len > 0) {
                                new_prefix.appendAssumeCapacity('.');
                            }
                            new_prefix.appendSliceAssumeCapacity(s);
                            try self.parseValue(allocator, store, new_prefix, val);
                        }
                    }
                }
            },
            .set_state => |set_state| {
                // `set_state` contains info about python struct being constructed
                switch (set_state.obj) {
                    .object => |obj| switch (obj.member) {
                        .raw => |raw| switch (raw) {
                            .global => |global| {
                                // in this case, we can capture the name of the python type
                                // which can be used for codegen (e.g. `torch.nn.modules.conv.Conv2d`)
                                var new_prefix = prefix;
                                if (prefix.items.len > 0) {
                                    new_prefix.appendAssumeCapacity('.');
                                }
                                new_prefix.appendSliceAssumeCapacity("_gen_type_helper");
                                const key = try allocator.dupe(u8, new_prefix.items);
                                const d = try store._metadata.getOrPut(allocator, key);
                                if (d.found_existing) {
                                    log.err("Duplicate key: {s}", .{new_prefix.items});
                                    allocator.free(key);
                                } else {
                                    const val = try std.mem.join(allocator, ".", &.{ global.module, global.class });
                                    d.value_ptr.* = .{ .string = val };
                                }
                            },
                            else => try self.parseValue(allocator, store, prefix, set_state.obj), // parse normally
                        },
                        else => try self.parseValue(allocator, store, prefix, set_state.obj), // parse normally
                    },
                    else => try self.parseValue(allocator, store, prefix, set_state.obj), // parse normally
                }
                try self.parseValue(allocator, store, prefix, set_state.state);
            },
            .pers_id => |pers_id| try self.parseValue(allocator, store, prefix, pers_id.ref),
            .seq => |seq| {
                switch (seq.type) {
                    .list, .tuple, .set, .frozen_set => {
                        if (seq.values.len == 0) return;
                        var valid_slice = true;
                        switch (seq.values[0]) {
                            inline .int64, .float64, .boolval => |val0, tag| {
                                const ItemType = switch (tag) {
                                    .int64 => i64,
                                    .float64 => f64,
                                    .boolval => bool,
                                    else => unreachable,
                                };
                                var values: std.ArrayListUnmanaged(ItemType) = .{};
                                try values.append(allocator, val0);
                                for (seq.values[1..], 1..) |val, i| {
                                    if (std.meta.activeTag(val) != tag) valid_slice = false;
                                    if (valid_slice) {
                                        try values.append(allocator, @field(val, @tagName(tag)));
                                    } else {
                                        var new_prefix = prefix;
                                        if (prefix.items.len > 0) {
                                            new_prefix.appendAssumeCapacity('.');
                                        }
                                        new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                                        try self.parseValue(allocator, store, new_prefix, val);
                                    }
                                }

                                if (valid_slice) {
                                    try store._metadata.put(
                                        allocator,
                                        try allocator.dupe(u8, prefix.items),
                                        try zml.aio.Metadata.copySlice(allocator, values.items),
                                    );
                                } else {
                                    for (values.items, 0..) |val, i| {
                                        var new_prefix = prefix;
                                        if (prefix.items.len > 0) {
                                            new_prefix.appendAssumeCapacity('.');
                                        }
                                        new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                                        const new_tag = switch (tag) {
                                            .int64 => "int",
                                            .float64 => "float",
                                            .boolval => "bool",
                                            else => unreachable, // we are already inside a switch
                                        };
                                        try store._metadata.put(allocator, try allocator.dupe(u8, new_prefix.items), @unionInit(zml.aio.Metadata, new_tag, val));
                                    }
                                }
                            },
                            else => {
                                for (seq.values, 0..) |item, i| {
                                    var new_prefix = prefix;
                                    if (v.isPrimitive()) {
                                        if (prefix.items.len > 0) {
                                            new_prefix.appendAssumeCapacity('.');
                                        }
                                        new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                                    }
                                    try self.parseValue(allocator, store, new_prefix, item);
                                }
                            },
                        }
                    },
                    .dict => {
                        const n = @divExact(seq.values.len, 2);
                        log.debug("found dict with {} entries", .{n});
                        for (0..n) |i| {
                            const key, const val = seq.values[2 * i ..][0..2].*;
                            switch (key) {
                                .string => |s| {
                                    // Handle Pytorch specific fields.
                                    if (std.mem.eql(u8, s, "_modules") or std.mem.eql(u8, s, "_parameters") or std.mem.eql(u8, s, "_buffers")) {
                                        try self.parseValue(allocator, store, prefix, val);
                                    } else {
                                        var new_prefix = prefix;
                                        if (prefix.items.len > 0) {
                                            new_prefix.appendAssumeCapacity('.');
                                        }
                                        new_prefix.appendSliceAssumeCapacity(s);

                                        try self.parseValue(allocator, store, new_prefix, val);
                                    }
                                },
                                .int64 => |int| {
                                    var new_prefix = prefix;
                                    if (prefix.items.len > 0) {
                                        new_prefix.appendAssumeCapacity('.');
                                    }
                                    new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), int, 10, .lower, .{});
                                    try self.parseValue(allocator, store, new_prefix, val);
                                },
                                inline else => |_, tag| {
                                    log.debug("Ignoring unsupported key type found in torch file: {s}", .{@tagName(tag)});
                                    continue;
                                },
                            }
                        }
                    },
                }
            },
            .bytes => |val| {
                const key = try allocator.dupe(u8, prefix.items);
                const d = try store._metadata.getOrPut(allocator, key);
                if (d.found_existing) {
                    log.warn("Duplicate key: {s}", .{prefix.items});
                    allocator.free(key);
                } else d.value_ptr.* = .{ .string = val };
            },
            inline .float64, .int64, .boolval, .bigint, .string => |val| {
                const key = try allocator.dupe(u8, prefix.items);
                const d = try store._metadata.getOrPut(allocator, key);
                if (d.found_existing) {
                    log.warn("Duplicate key: {s}", .{prefix.items});
                    allocator.free(key);
                } else {
                    d.value_ptr.* = zml.aio.Metadata.wrap(val);
                }
            },
            else => {},
        }
    }

    fn parseTorchGlobal(self: File, allocator: std.mem.Allocator, store: *zml.aio.BufferStore, prefix: StringBuilder, v: py.Any) !bool {
        return switch (v) {
            .global => |object| {
                if (try self.parseTensor(allocator, object)) |host_buffer| {
                    const key = try allocator.dupe(u8, prefix.items);
                    const entry = try store.buffers.getOrPut(allocator, key);
                    if (entry.found_existing) {
                        log.warn("Duplicate key: {s}", .{key});
                        allocator.free(key);
                    }

                    try store.registerBuffer(
                        allocator,
                        key,
                        host_buffer.shape(),
                        host_buffer.data,
                    );
                    return true;
                } else if (basicTypeCheck(object, "torch", "Size")) {
                    const size = object.args;
                    const key = try allocator.dupe(u8, prefix.items);
                    const entry = try store._metadata.getOrPut(allocator, key);
                    if (entry.found_existing) {
                        log.warn("Duplicate key: {s}", .{prefix.items});
                        allocator.free(key);
                    }
                    const d = try allocator.alloc(i64, size.len);
                    for (d, 0..) |*di, i| di.* = size[i].int64;
                    entry.value_ptr.* = .{ .array_int = d };
                    return true;
                } else if (basicTypeCheck(object, "fractions", "Fraction")) {
                    const fraction_str = object.args[0].string;
                    if (std.mem.indexOfScalar(u8, fraction_str, '/')) |split_idx| {
                        {
                            var new_prefix = prefix;
                            new_prefix.appendSliceAssumeCapacity(".numerator");
                            try store._metadata.put(allocator, try allocator.dupe(u8, new_prefix.items), .{ .int = try std.fmt.parseInt(i64, fraction_str[0..split_idx], 10) });
                        }
                        {
                            var new_prefix = prefix;
                            new_prefix.appendSliceAssumeCapacity(".denominator");
                            try store._metadata.put(allocator, try allocator.dupe(u8, new_prefix.items), .{ .int = try std.fmt.parseInt(i64, fraction_str[split_idx + 1 ..], 10) });
                        }
                        return true;
                    }
                }
                return false;
            },
            else => false,
        };
    }

    fn parseTensor(self: File, tmp_allocator: std.mem.Allocator, object: *py.Object) !?zml.HostBuffer {
        if (!basicTypeCheck(object, "torch._utils", "_rebuild_tensor_v2")) {
            return null;
        }

        const args = object.args;
        if (args.len < 4 or
            args[0] != .pers_id or
            args[1] != .int64 or
            args[2] != .seq or args[2].seq.type != .tuple or
            args[3] != .seq or args[3].seq.type != .tuple)
        {
            log.err("Unexpected py.Any in call to torch._utils._rebuild_tensor_v2: {}", .{object.*});
            return error.InvalidInput;
        }

        const pid: *py.PersId = args[0].pers_id;
        var offset: u64 = @intCast(args[1].int64);
        const raw_dims: py.Sequence = args[2].seq;
        const raw_strides: py.Sequence = args[3].seq;
        const dims = try parseDims(raw_dims.values);
        var strides = try parseDims(raw_strides.values);

        const dtype, const storage_file = try parseStorage(pid.ref);
        // Pytorch store "item" strides, while ZML uses byte strides.
        for (strides.slice()) |*s| s.* *= dtype.sizeOf();
        // Same thing for the offset.
        offset = offset * dtype.sizeOf();

        const filename = try std.mem.join(tmp_allocator, "", &.{ self.zip_prefix, "data/", storage_file });
        defer tmp_allocator.free(filename);

        // The offset in the pickle is the offset inside the storage_file.
        // But .pt are made of several files, so we need to append the file offset.
        const storage = try self.getStorage(filename);
        return HostBuffer.fromStridedSlice(
            zml.Shape.init(dims.constSlice(), dtype),
            storage[offset..],
            strides.constSlice(),
        );
    }

    fn parseStorage(val: py.Any) !struct { zml.DataType, []const u8 } {
        if (val != .seq) return error.InvalidInput;
        const sargs = val.seq.values;
        if (val.seq.type == .tuple and
            sargs.len >= 5 and
            sargs[0] == .string and std.mem.eql(u8, sargs[0].string, "storage") and
            sargs[1] == .raw and sargs[1].raw == .global and
            sargs[2] == .string and
            sargs[3] == .string)
        {
            const op = sargs[1].raw.global;
            const storage_file = sargs[2].string;
            // const sdev = sargs[3].string;
            if (!std.mem.eql(u8, "torch", op.module) or
                !std.mem.endsWith(u8, op.class, "Storage"))
                return error.InvalidInput;

            return .{
                try storageToDtype(op.class),
                storage_file,
            };
        } else {
            return error.InvalidInput;
        }
    }

    /// Given the name of one of the files in the .pt tarball,
    /// return the slice of the memory-mapped .pt corresponding to it.
    fn getStorage(self: File, filename: []const u8) ![]const u8 {
        const maybe_entry = self.file_map.get(filename);
        if (maybe_entry == null) {
            std.log.err("Could not find file ending in `{s}` in archive", .{filename});
            return error.TensorNotFound;
        }
        const entry = maybe_entry.?;
        const base_offset: u64 = if (self.tar_file) |t| t.start else 0;
        const file_offset: u64 = base_offset + entry.file_offset;
        const file = self.buffer_file.file;
        try file.seekTo(entry.file_offset);
        const local_header = try file.reader().readStructEndian(std.zip.LocalFileHeader, .little);

        if (!std.mem.eql(u8, &local_header.signature, &std.zip.local_file_header_sig))
            return error.ZipBadFileOffset;
        if (local_header.compressed_size != 0 and
            local_header.compressed_size != entry.compressed_size)
            return error.ZipMismatchCompLen;
        if (local_header.uncompressed_size != 0 and
            local_header.uncompressed_size != entry.uncompressed_size)
            return error.ZipMismatchUncompLen;
        if (local_header.filename_len != entry.filename_len)
            return error.ZipMismatchFilenameLen;

        const start = file_offset +
            @sizeOf(std.zip.LocalFileHeader) +
            @as(u64, local_header.filename_len) +
            @as(u64, local_header.extra_len);
        return self.buffer_file.mappedSlice(start, entry.uncompressed_size);
    }

    fn parseDims(values: []py.Any) error{InvalidInput}!zml.Shape.DimsArray {
        stdx.debug.assert(values.len <= zml.Tensor.MAX_RANK, "Found Pytorch tensor with unsupported rank {}", .{values.len});
        var result: zml.Shape.DimsArray = .{};
        for (values) |val| {
            switch (val) {
                .int64 => |d| result.appendAssumeCapacity(d),
                else => return error.InvalidInput,
            }
        }
        return result;
    }
};

/// Convert from a torch.<type>Storage to a `zml.DataType`.
/// TODO: make this future proof, storage type are going to get replaced with torch.UntypedStorage
/// See https://pytorch.org/docs/stable/storage.html
fn storageToDtype(storage_type: []const u8) !zml.DataType {
    const torch_type = storage_type[0 .. storage_type.len - "Storage".len];
    const map = std.StaticStringMap(zml.DataType).initComptime(.{
        .{ "Double", .f64 },
        .{ "Float", .f32 },
        .{ "Half", .f16 },
        .{ "Long", .i64 },
        .{ "Int", .i32 },
        .{ "Short", .i16 },
        .{ "Char", .i8 },
        .{ "Byte", .u8 },
        .{ "Bool", .bool },
        .{ "BFloat16", .bf16 },
        .{ "ComplexDouble", .c128 },
        .{ "ComplexFloat", .c64 },
        // QUInt8Storage
        // QInt8Storage
        // QInt32Storage
        // QUInt4x2Storage
        // QUInt2x4Storage
    });

    return map.get(torch_type) orelse {
        log.err("Unsupported torch storage type: {s}", .{storage_type});
        return error.UnsupportedDataType;
    };
}

const TarStream = struct {
    pub const SeekableStream = std.io.SeekableStream(
        TarStream,
        asynk.File.SeekError,
        asynk.File.GetSeekPosError,
        TarStream.seekTo,
        TarStream.seekBy,
        TarStream.getPos,
        TarStream.getEndPos,
    );

    file: std.tar.Iterator(asynk.File.Reader).File,
    start: usize,

    pub fn init(file: std.tar.Iterator(asynk.File.Reader).File) !TarStream {
        return .{
            .file = file,
            .start = try file.parent_reader.context.getPos(),
        };
    }

    pub fn reader(file: TarStream) std.tar.Iterator(asynk.File.Reader).File.Reader {
        return file.file.reader();
    }

    pub fn seekTo(self: TarStream, offset: u64) !void {
        return self.file.parent_reader.context.seekTo(self.start + offset);
    }

    pub fn seekBy(self: TarStream, offset: i64) !void {
        return self.file.parent_reader.context.seekBy(offset);
    }

    pub fn getPos(self: TarStream) !u64 {
        return try self.file.parent_reader.context.getPos() - self.start;
    }

    pub fn getEndPos(self: TarStream) !u64 {
        return self.file.size;
    }

    pub fn seekableStream(self: TarStream) TarStream.SeekableStream {
        return .{ .context = self };
    }
};

test "Read pickle (zipped)" {
    // test file created with following python snippet:
    //
    // import torch
    // torch.manual_seed(0)
    // model = torch.nn.Conv2d(2, 2, 3, stride=2, padding=[2, 4], dtype=torch.float16)
    // tensor = torch.tensor([[2, 4, 3, 2]], dtype=torch.uint8)
    // torch.save({ "model": model, "tensor": tensor}, "simple.pt")
    const file = try asynk.File.open("zml/aio/torch/simple.pt", .{ .mode = .read_only });
    const mmap_file = try zml.aio.MemoryMappedFile.init(file);
    var store = try zml.aio.BufferStore.init(testing.allocator, &.{mmap_file});
    defer store.deinit();

    {
        var tmp_arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer tmp_arena.deinit();
        const tmp_alloc = tmp_arena.allocator();
        var torch_file = try File.init(tmp_alloc, mmap_file);
        // We don't close the file directly, it will be closed by the store.

        const ops = try torch_file.parsePickle(tmp_alloc);
        try std.testing.expectEqual(302, ops.len);

        const py_values = try eval.evaluate(tmp_alloc, ops, true);
        try torch_file.parseModel(py_values, &store);
    }

    // now we have freed the tmp_arena.
    // all data needed should have been copied into the store arena.
    try zml.testing.expectEqualShapes(
        zml.Shape.init(.{ 1, 4 }, .u8),
        store.get("tensor").?.shape(),
    );
    try zml.testing.expectEqualShapes(
        zml.Shape.init(.{ 2, 2, 3, 3 }, .f16),
        store.get("model.weight").?.shape(),
    );
    try zml.testing.expectEqualShapes(
        zml.Shape.init(.{2}, .f16),
        store.get("model.bias").?.shape(),
    );
}

fn isBadFilename(filename: []const u8) bool {
    if (filename.len == 0 or filename[0] == '/')
        return true;

    var it = std.mem.splitScalar(u8, filename, '/');
    while (it.next()) |part| {
        if (std.mem.eql(u8, part, ".."))
            return true;
    }

    return false;
}
