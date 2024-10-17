const asynk = @import("async");
const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const zml = @import("../../zml.zig");
const pickle = @import("pickle.zig");

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(File);
}

pub const File = struct {
    buffer_file: zml.aio.MemoryMappedFile,
    file_map: std.StringArrayHashMapUnmanaged(FileEntry) = .{},
    tar_file: ?TarStream = null,
    ops: []const pickle.Op,
    is_zip_file: bool,
    zip_prefix: []const u8 = &[_]u8{},

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

    pub fn fromTarFile(allocator: Allocator, mapped: zml.aio.MemoryMappedFile, file: std.tar.Iterator(asynk.File.Reader).File) !File {
        const tar_stream = try TarStream.init(file);
        const file_magic = try tar_stream.reader().readBytesNoEof(magic.len);
        try tar_stream.seekTo(0);
        var self: File = .{
            .buffer_file = mapped,
            .tar_file = tar_stream,
            .ops = undefined,
            .is_zip_file = std.mem.eql(u8, &file_magic, magic),
        };
        if (!self.is_zip_file) {
            const reader = tar_stream.reader();
            self.ops = try pickle.parse(allocator, reader, try tar_stream.getEndPos());
        } else {
            self.ops = try self.parseOps(allocator, self.tar_file.?.seekableStream());
        }
        return self;
    }

    pub fn init(allocator: Allocator, file: asynk.File) !File {
        const file_magic = try file.reader().readBytesNoEof(magic.len);
        try file.seekTo(0);
        var self: File = .{
            .buffer_file = try zml.aio.MemoryMappedFile.init(file),
            .is_zip_file = std.mem.eql(u8, &file_magic, magic),
            .ops = undefined,
        };
        if (!self.is_zip_file) {
            const reader = self.buffer_file.file.reader();
            self.ops = try pickle.parse(allocator, reader, try reader.context.getEndPos());
        } else {
            self.ops = try self.parseOps(allocator, self.buffer_file.file.seekableStream());
        }
        return self;
    }

    pub fn deinit(self: *File) void {
        self.buffer_file.deinit();
        self.* = undefined;
    }

    fn parseOps(self: *File, allocator: Allocator, seekable_stream: anytype) ![]const pickle.Op {
        var iter = try std.zip.Iterator(@TypeOf(seekable_stream)).init(seekable_stream);
        var filename_buf: [std.fs.max_path_bytes]u8 = undefined;
        while (try iter.next()) |entry| {
            const filename = filename_buf[0..entry.filename_len];
            try seekable_stream.seekTo(entry.header_zip_offset + @sizeOf(std.zip.CentralDirectoryFileHeader));
            const len = try seekable_stream.context.reader().readAll(filename);
            if (len != filename.len) return error.ZipBadFileOffset;
            if (isBadFilename(filename)) return error.ZipBadFilename;
            std.mem.replaceScalar(u8, filename, '\\', '/'); // normalize path separators
            try self.file_map.put(allocator, try allocator.dupe(u8, filename), FileEntry.init(entry));
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
                try seekable_stream.seekTo(local_data_file_offset);

                switch (entry.compression_method) {
                    .store => {
                        return pickle.parse(allocator, seekable_stream.context.reader(), entry.uncompressed_size);
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

test "Read pickle (zipped)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const file = try asynk.File.open("zml/aio/torch/simple.pt", .{ .mode = .read_only });
    var data = try File.init(allocator, file);
    defer data.deinit();
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
