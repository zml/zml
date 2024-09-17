const asynk = @import("async");
const std = @import("std");
const zml = @import("../../zml.zig");

const utils = @import("utils.zig");
const PickleOp = @import("ops.zig").PickleOp;
const RawPickleOp = @import("ops.zig").RawPickleOp;

const Allocator = std.mem.Allocator;
const testing = std.testing;

pub const Decoder = struct {
    buffer_file: zml.aio.MemoryMappedFile,
    file_map: std.StringArrayHashMapUnmanaged(std.zip.Iterator(asynk.File.SeekableStream).Entry) = .{},
    tar_file: ?TarStream = null,
    ops: []PickleOp,
    is_zip_file: bool,
    zip_prefix: []const u8 = &[_]u8{},

    const magic = "PK\x03\x04";

    pub fn fromTarFile(allocator: Allocator, mapped: zml.aio.MemoryMappedFile, file: std.tar.Iterator(asynk.File.Reader).File) !Decoder {
        const tar_stream = try TarStream.init(file);
        const file_magic = try tar_stream.reader().readBytesNoEof(magic.len);
        try tar_stream.seekTo(0);
        var self: Decoder = .{
            .buffer_file = mapped,
            .tar_file = tar_stream,
            .ops = undefined,
            .is_zip_file = std.mem.eql(u8, &file_magic, magic),
        };
        if (!self.is_zip_file) {
            const reader = tar_stream.reader();
            self.ops = try parse(allocator, reader, try tar_stream.getEndPos());
        } else {
            self.ops = try self.parseOps(allocator, self.tar_file.?.seekableStream());
        }
        return self;
    }

    pub fn init(allocator: Allocator, file: asynk.File) !Decoder {
        const file_magic = try file.reader().readBytesNoEof(magic.len);
        try file.seekTo(0);
        var self: Decoder = .{
            .buffer_file = try zml.aio.MemoryMappedFile.init(file),
            .is_zip_file = std.mem.eql(u8, &file_magic, magic),
            .ops = undefined,
        };
        if (!self.is_zip_file) {
            const reader = self.buffer_file.file.reader();
            self.ops = try parse(allocator, reader, try reader.context.getEndPos());
        } else {
            self.ops = try self.parseOps(allocator, self.buffer_file.file.seekableStream());
        }
        return self;
    }

    pub fn deinit(self: *Decoder) void {
        self.buffer_file.deinit();
        self.* = undefined;
    }

    fn parseOps(self: *Decoder, allocator: Allocator, seekable_stream: asynk.File.SeekableStream) ![]PickleOp {
        // TODO(SuperAuguste): deflate using `std.compress.flate`'s `decompressor`
        // TODO(SuperAuguste): explore swapping in non-generic reader here instead of using switch(?)
        //                     not sure if that'd actually be beneficial in any way

        var iter = try std.zip.Iterator(asynk.File.SeekableStream).init(seekable_stream);
        var filename_buf: [std.fs.max_path_bytes]u8 = undefined;
        while (try iter.next()) |entry| {
            const filename = filename_buf[0..entry.filename_len];
            try seekable_stream.seekTo(entry.header_zip_offset + @sizeOf(std.zip.CentralDirectoryFileHeader));
            const len = try seekable_stream.context.reader().readAll(filename);
            if (len != filename.len) return error.ZipBadFileOffset;
            if (isBadFilename(filename)) return error.ZipBadFilename;
            std.mem.replaceScalar(u8, filename, '\\', '/'); // normalize path separators
            try self.file_map.put(allocator, try allocator.dupe(u8, filename), entry);
        }

        var file_iter = self.file_map.iterator();
        while (file_iter.next()) |e| {
            const entry = e.value_ptr.*;
            const filename = e.key_ptr.*;
            if (std.mem.indexOf(u8, filename, "data.pkl")) |idx| {
                self.zip_prefix = filename[0..idx];
                const local_data_header_offset: u64 = local_data_header_offset: {
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

                    if (@as(u16, @bitCast(local_header.flags)) != @as(u16, @bitCast(entry.flags)))
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
                try seekable_stream.seekTo(local_data_file_offset);

                switch (entry.compression_method) {
                    .store => {
                        return parse(allocator, seekable_stream.context.reader(), entry.uncompressed_size);
                    },
                    .deflate => {
                        // TODO(cryptodeal): handle decompress
                        @panic("TODO support use of `deflate`");
                    },
                    else => @panic("TODO support other modes of compression"),
                }
            }
        }

        std.log.err("Could not find file ending in `data.pkl` in archive", .{});
        return error.PickleNotFound;
    }

    fn parse(allocator: Allocator, reader: anytype, len: usize) ![]PickleOp {
        var results = std.ArrayList(PickleOp).init(allocator);
        errdefer results.deinit();
        outer: while (true) {
            const b = try reader.readByte();
            switch (@as(RawPickleOp, @enumFromInt(b))) {
                .mark => try results.append(.{ .mark = {} }),
                .stop => {
                    try results.append(.{ .stop = {} });
                    break :outer;
                },
                .pop => try results.append(.{ .pop = {} }),
                .pop_mark => try results.append(.{ .pop_mark = {} }),
                .dup => try results.append(.{ .dup = {} }),
                .float => {
                    const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf);
                    try results.append(.{ .float = buf });
                },
                .int => {
                    const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf);
                    try results.append(.{ .int = buf });
                },
                .binint => try results.append(.{ .binint = try reader.readInt(i32, .little) }),
                .binint1 => try results.append(.{ .binint1 = try reader.readByte() }),
                .long => {
                    const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf);
                    try results.append(.{ .long = buf });
                },
                .binint2 => try results.append(.{ .binint2 = try reader.readInt(u16, .little) }),
                .none => try results.append(.{ .none = {} }),
                .persid => {
                    const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf);
                    try results.append(.{ .persid = buf });
                },
                .binpersid => try results.append(.{ .binpersid = {} }),
                .reduce => try results.append(.{ .reduce = {} }),
                .string => {
                    const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf);
                    try results.append(.{ .string = buf });
                },
                .binstring => {
                    const str_len = try reader.readInt(u32, .little);
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .binstring = buf });
                },
                .short_binstring => {
                    const str_len = try reader.readByte();
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .short_binstring = buf });
                },
                .unicode => {
                    const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf);
                    try results.append(.{ .unicode = buf });
                },
                .binunicode => {
                    const str_len = try reader.readInt(u32, .little);
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .binunicode = buf });
                },
                .append => try results.append(.{ .append = {} }),
                .build => try results.append(.{ .build = {} }),
                .global => {
                    const buf0 = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf0);
                    const buf1 = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf1);
                    _ = (buf1.len + 1);
                    try results.append(.{ .global = .{ buf0, buf1 } });
                },
                .dict => try results.append(.{ .dict = {} }),
                .empty_dict => try results.append(.{ .empty_dict = {} }),
                .appends => try results.append(.{ .appends = {} }),
                .get => {
                    const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf);
                    try results.append(.{ .get = buf });
                },
                .binget => try results.append(.{ .binget = try reader.readByte() }),
                .inst => {
                    const buf0 = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf0);
                    const buf1 = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf1);
                    _ = (buf1.len + 1);
                    try results.append(.{ .inst = .{ buf0, buf1 } });
                },
                .long_binget => try results.append(.{ .long_binget = try reader.readInt(u32, .little) }),
                .list => try results.append(.{ .list = {} }),
                .empty_list => try results.append(.{ .empty_list = {} }),
                .obj => try results.append(.{ .obj = {} }),
                .put => {
                    const buf = try reader.readUntilDelimiterAlloc(allocator, '\n', len);
                    errdefer allocator.free(buf);
                    try results.append(.{ .put = buf });
                },
                .binput => {
                    try results.append(.{ .binput = try reader.readByte() });
                },
                .long_binput => {
                    try results.append(.{ .long_binput = try reader.readInt(u32, .little) });
                },
                .setitem => try results.append(.{ .setitem = {} }),
                .tuple => try results.append(.{ .tuple = {} }),
                .empty_tuple => try results.append(.{ .empty_tuple = {} }),
                .setitems => try results.append(.{ .setitems = {} }),
                .binfloat => try results.append(.{ .binfloat = @bitCast(try reader.readInt(u64, .big)) }),
                .proto => try results.append(.{ .proto = try reader.readByte() }),
                .newobj => try results.append(.{ .newobj = {} }),
                .ext1 => try results.append(.{ .ext1 = try reader.readByte() }),
                .ext2 => try results.append(.{ .ext2 = try reader.readInt(i16, .little) }),
                .ext4 => try results.append(.{ .ext4 = try reader.readInt(i32, .little) }),
                .tuple1 => try results.append(.{ .tuple1 = {} }),
                .tuple2 => try results.append(.{ .tuple2 = {} }),
                .tuple3 => try results.append(.{ .tuple3 = {} }),
                .newtrue => try results.append(.{ .newtrue = {} }),
                .newfalse => try results.append(.{ .newfalse = {} }),
                .long1 => {
                    const str_len = try reader.readByte();
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .long1 = buf });
                },
                .long4 => {
                    const str_len = try reader.readInt(u32, .little);
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .long4 = buf });
                },
                .binbytes => {
                    const str_len = try reader.readInt(u32, .little);
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .binbytes = buf });
                },
                .binbytes8 => {
                    const str_len = try reader.readInt(u64, .little);
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .binbytes8 = buf });
                },
                .short_binbytes => {
                    const str_len = try reader.readByte();
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .short_binbytes = buf });
                },
                .binunicode8 => {
                    const str_len = try reader.readInt(u64, .little);
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .binunicode8 = buf });
                },
                .short_binunicode => {
                    const str_len = try reader.readByte();
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .binunicode8 = buf });
                },
                .empty_set => try results.append(.{ .empty_set = {} }),
                .additems => try results.append(.{ .additems = {} }),
                .frozenset => try results.append(.{ .frozenset = {} }),
                .newobj_ex => try results.append(.{ .newobj_ex = {} }),
                .stack_global => try results.append(.{ .stack_global = {} }),
                .memoize => try results.append(.{ .memoize = {} }),
                .frame => try results.append(.{ .frame = try reader.readInt(u64, .little) }),
                .bytearray8 => {
                    const str_len = try reader.readInt(u64, .little);
                    const buf = try allocator.alloc(u8, str_len);
                    errdefer allocator.free(buf);
                    _ = try reader.read(buf);
                    try results.append(.{ .bytearray8 = buf });
                },
                .next_buffer => try results.append(.{ .next_buffer = {} }),
                .readonly_buffer => try results.append(.{ .readonly_buffer = {} }),
                else => {},
            }
        }
        return results.toOwnedSlice();
    }
};

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

test "Read pickle (simple)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const eval = @import("eval.zig");
    const file = try asynk.File.open("zml/aio/torch/simple_test.pickle", .{ .mode = .read_only });
    var data = try Decoder.init(allocator, file);
    defer data.deinit();
    var vals, var memo = try eval.evaluate(allocator, data.ops, true);
    defer vals.deinit();
    defer memo.deinit();

    try testing.expect(vals.stack.len == 2);
    // skip first value (frame)
    try testing.expect(vals.stack[1] == .seq);
    try testing.expect(vals.stack[1].seq[0] == .dict);
    const entries = vals.stack[1].seq[1][0].seq[1];
    try testing.expect(entries.len == 5);
    for (entries, 0..) |kv, i| {
        try testing.expect(kv == .seq);
        try testing.expect(kv.seq[0] == .kv_tuple);
        switch (i) {
            0 => {
                const key = kv.seq[1][0];
                try testing.expect(key == .string);
                try testing.expectEqualStrings("hello", key.string);
                const value = kv.seq[1][1];
                try testing.expect(value == .string);
                try testing.expectEqualStrings("world", value.string);
            },
            1 => {
                const key = kv.seq[1][0];
                try testing.expect(key == .string);
                try testing.expectEqualStrings("int", key.string);
                const value = kv.seq[1][1];
                try testing.expect(value == .int);
                try testing.expect(value.int == 1);
            },
            2 => {
                const key = kv.seq[1][0];
                try testing.expect(key == .string);
                try testing.expectEqualStrings("float", key.string);
                const value = kv.seq[1][1];
                try testing.expect(value == .float);
                try testing.expectEqual(@as(f64, 3.141592), value.float);
            },
            3 => {
                const key = kv.seq[1][0];
                try testing.expect(key == .string);
                try testing.expectEqualStrings("list", key.string);
                const value = kv.seq[1][1];
                try testing.expect(value == .seq);
                try testing.expect(value.seq[0] == .list);
                for (value.seq[1], 0..) |item, j| {
                    try testing.expect(item == .int);
                    try testing.expect(item.int == @as(i64, @intCast(j)));
                }
            },
            4 => {
                const key = kv.seq[1][0];
                try testing.expect(key == .string);
                try testing.expectEqualStrings("tuple", key.string);
                const value = kv.seq[1][1];
                try testing.expect(value == .seq);
                try testing.expect(value.seq[0] == .tuple);
                try testing.expect(value.seq[1][0] == .string);
                try testing.expectEqualStrings("a", value.seq[1][0].string);
                try testing.expect(value.seq[1][1] == .int);
                try testing.expect(value.seq[1][1].int == 10);
            },
            else => unreachable,
        }
    }
}

test "Read pickle (zipped)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const file = try asynk.File.open("zml/aio/torch/simple.pt", .{ .mode = .read_only });
    var data = try Decoder.init(allocator, file);
    defer data.deinit();
}

pub fn isBadFilename(filename: []const u8) bool {
    if (filename.len == 0 or filename[0] == '/')
        return true;

    var it = std.mem.splitScalar(u8, filename, '/');
    while (it.next()) |part| {
        if (std.mem.eql(u8, part, ".."))
            return true;
    }

    return false;
}
