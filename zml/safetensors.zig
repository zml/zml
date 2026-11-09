const std = @import("std");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;

const log = std.log.scoped(.@"zml/safetensors");

// TODO: consider promoting this to the top level struct
// TODO: Feels like we are missing a mapping file -> tensors that would be helfpful for loading weights
pub const TensorRegistry = struct {
    arena: std.heap.ArenaAllocator,

    tensors: std.StringArrayHashMapUnmanaged(Tensor),

    /// Metadata found in `__metadata__` field of the safetensors header.
    /// This is rarely use and the information there is not standardized, so we let user code handle it.
    metadata: []const u8,

    // mutex: std.Io.Mutex = .init,

    pub fn init(allocator: std.mem.Allocator) TensorRegistry {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
            .metadata = "",
        };
    }

    pub fn deinit(self: *TensorRegistry) void {
        self.arena.deinit();
    }

    pub fn fromRepo(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !TensorRegistry {
        const file_type, const file = try resolveModelEntrypoint(io, repo);
        defer file.close(io);

        var registry: TensorRegistry = .init(allocator);
        switch (file_type) {
            .safetensors => try registry.parseSafetensors(io, allocator, file),
            .index => try registry.parseIndexJson(io, allocator, repo, file),
            .full_index => try registry.parseFullIndexJson(io, allocator, repo, file),
            .unknown => return error.InvalidPath,
        }
        return registry;
    }

    pub fn fromPath(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !TensorRegistry {
        const file_type: FileType = .resolve(path);
        switch (file_type) {
            .safetensors => {
                // open directly
                const file = try std.Io.Dir.openFile(.cwd(), io, path, .{});
                var registry: TensorRegistry = .init(allocator);
                try registry.parseSafetensors(io, allocator, file);
                return registry;
            },
            .full_index,
            .index,
            => {
                // resolve repo as parent folder then open
                const repo_path = std.fs.path.dirname(path) orelse return ModelPathResolutionError.InvalidPath;
                const repo = std.Io.Dir.openDir(.cwd(), io, repo_path, .{}) catch |e| {
                    return switch (e) {
                        error.FileNotFound => ModelPathResolutionError.FileNotFound,
                        else => ModelPathResolutionError.InvalidPath,
                    };
                };
                const file = try std.Io.Dir.openFile(.cwd(), io, path, .{});

                var registry: TensorRegistry = .init(allocator);
                try switch (file_type) {
                    .index => registry.parseIndexJson(io, allocator, repo, file),
                    .full_index => registry.parseFullIndexJson(io, allocator, repo, file),
                    .safetensors, .unknown => unreachable,
                };
                return registry;
            },
            .unknown => {
                // the path is possibly a folder, let's try that
                const repo = std.Io.Dir.openDir(.cwd(), io, path, .{}) catch |e| {
                    return switch (e) {
                        error.FileNotFound => ModelPathResolutionError.FileNotFound,
                        else => ModelPathResolutionError.InvalidPath,
                    };
                };
                return try fromRepo(allocator, io, repo);
            },
        }
    }

    /// Register a tensor, duplicate tensor name and uri.
    /// If you create a lot of tensors sharing the same uri,
    /// considere calling `registry.tensors` directly and avoid duplicated uris.
    pub fn registerTensor(self: *TensorRegistry, tensor: Tensor) !void {
        const allocator = self.arena.allocator();

        // self.mutex.lock();
        // defer self.mutex.unlock();

        var tensor_copy = tensor;

        tensor_copy.name = try allocator.dupe(u8, tensor.name);
        tensor_copy.file_uri = try allocator.dupe(u8, tensor.file_uri);

        try self.tensors.put(allocator, tensor_copy.name, tensor_copy);
    }

    pub fn reader(
        self: *TensorRegistry,
        io: std.Io,
        tensor_name: []const u8,
        buffer: []u8,
    ) !TensorReader {
        const tensor = self.tensors.get(tensor_name) orelse {
            log.err("Tensor {s} not found in registry", .{tensor_name});
            return error.TensorNotFound;
        };

        return try .init(io, tensor, buffer, .{});
    }

    pub fn totalBytes(self: *TensorRegistry) u64 {
        var total: u64 = 0;

        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            total += entry.value_ptr.byteSize();
        }

        return total;
    }

    /// Parse a given .safetensors file and create entries for each tensor
    pub fn parseSafetensors(
        self: *TensorRegistry,
        io: std.Io,
        scratch_allocator: std.mem.Allocator,
        file: std.Io.File,
    ) !void {
        var tmp_state = std.heap.ArenaAllocator.init(scratch_allocator);
        defer tmp_state.deinit();
        const tmp_arena = tmp_state.allocator();

        const self_arena = self.arena.allocator();
        var owned_file_name = try self_arena.alloc(u8, std.fs.max_path_bytes);
        const file_name_len = try file.realPath(io, owned_file_name);
        // Try to release extra memory, but we don't care if it fails
        _ = self_arena.resize(owned_file_name, file_name_len);
        owned_file_name = owned_file_name[0..file_name_len];

        const header_buf = try tmp_arena.alloc(u8, 64 * 1024);
        var file_reader: std.Io.File.Reader = file.reader(io, header_buf);

        const json_header_length: u64 = try file_reader.interface.takeInt(u64, .little);
        const BYTES_HEADER = 8;
        const data_start_offset = BYTES_HEADER + json_header_length;
        const json_data = try tmp_arena.alloc(u8, @intCast(json_header_length));

        try file_reader.interface.readSliceAll(json_data);
        const headers: std.json.Value = try std.json.parseFromSliceLeaky(
            std.json.Value,
            tmp_arena,
            json_data,
            .{},
        );

        try self.tensors.ensureUnusedCapacity(self_arena, headers.object.count());
        for (headers.object.keys(), headers.object.values()) |key, value| {
            if (std.mem.eql(u8, key, "__metadata__")) {
                // value belong to tmp arena, duplicate it into self arena.
                var owned_metadata: std.Io.Writer.Allocating = .init(self_arena);
                try std.json.fmt(value, .{ .whitespace = .minified }).format(&owned_metadata.writer);
                self.metadata = owned_metadata.written();
                continue;
            }

            const owned_name = try self_arena.dupe(u8, key);
            const tensor: Tensor = try .parse(value.object, owned_file_name, owned_name, data_start_offset);
            self.tensors.putAssumeCapacity(owned_name, tensor);
        }
    }

    pub fn parseIndexJson(
        self: *TensorRegistry,
        io: std.Io,
        gpa: std.mem.Allocator,
        repo: std.Io.Dir,
        index_json_file: std.Io.File,
    ) !void {
        var tmp_state: std.heap.ArenaAllocator = .init(gpa);
        defer tmp_state.deinit();
        const tmp = tmp_state.allocator();

        const file_reader_buf = try tmp.alloc(u8, 64 * 1024);
        var file_reader: std.Io.File.Reader = index_json_file.reader(io, file_reader_buf);
        var json_reader: std.json.Reader = .init(tmp, &file_reader.interface);
        const index = try std.json.parseFromTokenSourceLeaky(
            std.json.Value,
            tmp,
            &json_reader,
            .{ .allocate = .alloc_if_needed },
        );

        var safetensors_files: std.StringArrayHashMapUnmanaged(void) = .empty;
        try safetensors_files.ensureUnusedCapacity(tmp, 64);
        if (index.object.get("weight_map")) |wm| {
            for (wm.object.values()) |file_name| {
                try safetensors_files.put(tmp, file_name.string, {});
            }
        } else {
            log.warn("No weight_map attribute found in index", .{});
        }

        for (safetensors_files.keys()) |file_name| {
            const file = try repo.openFile(io, file_name, .{});
            defer file.close(io);
            try self.parseSafetensors(io, tmp, file);
        }
    }

    pub fn parseFullIndexJson(
        self: *TensorRegistry,
        io: std.Io,
        gpa: std.mem.Allocator,
        repo: std.Io.Dir,
        index_json_file: std.Io.File,
    ) !void {
        var tmp_state: std.heap.ArenaAllocator = .init(gpa);
        defer tmp_state.deinit();
        const tmp = tmp_state.allocator();

        const self_arena = self.arena.allocator();

        var repo_uri_buf = try tmp.alloc(u8, 1024);
        const repo_name = repo_uri_buf[0..try repo.realPath(io, repo_uri_buf)];

        // Note: this is relatively inefficient, we read the full file, copy most of it's content,
        // then reduce to TensorRegistry format.
        // We could update TensorRegistry as we stream the json.
        // Zig std json API are a bit awkward around streaming though.
        const file_reader_buf = try tmp.alloc(u8, 64 * 1024);
        var file_reader = index_json_file.reader(io, file_reader_buf);
        var json_reader: std.json.Reader = .init(tmp, &file_reader.interface);
        const index = try std.json.parseFromTokenSourceLeaky(
            std.json.Value,
            tmp,
            &json_reader,
            .{ .allocate = .alloc_if_needed },
        );

        var realpaths: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
        try realpaths.ensureUnusedCapacity(tmp, 64);
        const wm = index.object.get("weight_map") orelse return error.InvalidJson;

        try self.tensors.ensureUnusedCapacity(self_arena, wm.object.count());
        for (wm.object.keys(), wm.object.values()) |key, value| {
            const tensor_name = try self_arena.dupe(u8, key);
            const object = value.object;
            const filename = value.object.get("filename").?.string;
            const filename_entry = try realpaths.getOrPut(tmp, filename);
            const file_uri = if (filename_entry.found_existing)
                filename_entry.value_ptr.*
            else uri: {
                const uri = try std.fs.path.join(self_arena, &.{ repo_name, filename });
                filename_entry.value_ptr.* = uri;
                break :uri uri;
            };

            const tensor_info: Tensor = try .parse(object, file_uri, tensor_name, 0);
            self.tensors.putAssumeCapacity(tensor_name, tensor_info);
        }

        if (index.object.get("metadata")) |metadata| {
            // value belong to tmp arena, duplicate it into self arena.
            var owned_metadata: std.Io.Writer.Allocating = .init(self_arena);
            try std.json.fmt(metadata, .{ .whitespace = .minified }).format(&owned_metadata.writer);
            self.metadata = owned_metadata.written();
        }
    }

    pub fn saveAsFullIndexJson(
        self: *const TensorRegistry,
        io: std.Io,
        gpa: std.mem.Allocator,
        repo: std.Io.Dir,
        out: *std.Io.Writer,
    ) !void {
        var repo_uri_buf = try gpa.alloc(u8, std.fs.max_path_bytes);
        defer gpa.free(repo_uri_buf);
        const repo_name = repo_uri_buf[0..try repo.realPath(io, repo_uri_buf)];
        var total_size: u64 = 0;
        for (self.tensors.values()) |value| total_size += value.byteSize();

        try out.print(
            \\{{
            \\"metadata": {{"num_tensors":{d},"total_size":{d}}},
            \\"weight_map": {{
            \\
        , .{ self.tensors.count(), total_size });

        const n = self.tensors.count();
        for (0.., self.tensors.keys(), self.tensors.values()) |i, key, value| {
            std.debug.assert(std.mem.startsWith(u8, value.file_uri, repo_name));
            const local_name = value.file_uri[repo_name.len + 1 ..];
            const end_offset = value.offset + value.byteSize();
            const safetensors_dtype: []const u8 = dt: {
                for (safetensors_dtype_to_zml_dtype.keys(), safetensors_dtype_to_zml_dtype.values()) |st_dt, zml_dt| {
                    if (zml_dt == value.shape._dtype) break :dt st_dt;
                }
                // Note: we could bitcast to u8,
                // but reaching this panic implies the user didn't loaded this TensorRegistry
                // from a safetensors file, and manually created the dict.
                // In which case, they know what they are doing,
                // and can easily insert the bitcast themselves.
                std.debug.panic("Safetensors doesn't support data type {}", .{value.shape._dtype});
            };
            const comma = if (i + 1 == n) "" else ",";
            try out.print(
                \\  "{s}": {{"filename":"{s}","shape":{f},"data_offsets":[{d},{d}],"dtype":"{s}"}}{s}
                \\
            , .{ key, local_name, std.json.fmt(value.shape.dims(), .{ .whitespace = .minified }), value.offset, end_offset, safetensors_dtype, comma });
        }

        try out.writeAll(
            \\}
            \\}
            \\
        );
        try out.flush();
    }
};

pub const TensorReader = struct {
    tensor: Tensor,
    file: std.Io.File,
    file_reader: std.Io.File.Reader,
    alignment: ?std.mem.Alignment,
    remaining: u64,
    padding_remaining: u64,

    io: std.Io,
    interface: std.Io.Reader,

    pub const InitOpts = struct {
        alignment: ?std.mem.Alignment = null,
    };

    pub fn init(
        io: std.Io,
        tensor: Tensor,
        buffer: []u8,
        opts: InitOpts,
    ) !TensorReader {
        const file = try std.Io.Dir.openFile(.cwd(), io, tensor.file_uri, .{ .mode = .read_only });
        errdefer file.close(io);

        const offset, const padding_remaining = if (opts.alignment) |alignment| blk: {
            const aligned_offset = std.mem.alignBackward(u64, tensor.offset, alignment.toByteUnits());
            break :blk .{ aligned_offset, tensor.offset - aligned_offset };
        } else blk: {
            break :blk .{ tensor.offset, 0 };
        };

        var file_reader = file.reader(io, buffer);
        try file_reader.seekTo(offset);

        return .{
            .tensor = tensor,
            .file = file,
            .file_reader = file_reader,
            .alignment = opts.alignment,
            .remaining = tensor.byteSize(),
            .padding_remaining = padding_remaining,
            .io = io,
            .interface = .{
                .vtable = if (opts.alignment) |_| blk: {
                    break :blk &.{
                        .stream = streamAligned,
                        .discard = discardAligned,
                    };
                } else blk: {
                    break :blk &.{
                        .stream = stream,
                        .discard = discard,
                    };
                },
                .buffer = &.{},
                .seek = 0,
                .end = 0,
            },
        };
    }

    pub fn deinit(self: *TensorReader) void {
        self.file.close(self.io);
    }

    // Limit single reads to 1GB to avoid issues on macOS where pread returns
    // INT_MAX when the requested size is larger than 2GB
    const max_single_read_bytes: u64 = 1 << 30;

    fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) {
            return error.EndOfStream;
        }

        const combined_limit = limit.min(.limited64(self.remaining)).min(.limited64(max_single_read_bytes));
        const n = try self.file_reader.interface.stream(w, combined_limit);
        self.remaining -= n;
        return n;
    }

    fn discard(r: *std.Io.Reader, limit: std.Io.Limit) std.Io.Reader.Error!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) {
            return error.EndOfStream;
        }

        const combined_limit = limit.min(.limited64(self.remaining));
        const n = try self.file_reader.interface.discard(combined_limit);
        self.remaining -= n;
        return n;
    }

    fn streamAligned(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) {
            return error.EndOfStream;
        }

        const requested_bytes = @min(limit.minInt64(self.remaining), max_single_read_bytes);

        if (self.padding_remaining > 0) {
            _ = try self.file_reader.interface.discard(.limited64(self.padding_remaining));
            self.padding_remaining = 0;
        }

        const align_bytes = self.alignment.?.toByteUnits();
        const file_offset = self.file_reader.pos - std.mem.alignBackward(u64, self.file_reader.pos, align_bytes);
        const writer_addr = @intFromPtr(w.buffer.ptr) + w.end;
        const writer_offset = writer_addr - std.mem.alignBackward(usize, writer_addr, align_bytes);

        if (file_offset != writer_offset) {
            const skip = if (file_offset >= writer_offset) blk: {
                break :blk file_offset - writer_offset;
            } else blk: {
                break :blk align_bytes - writer_offset + file_offset;
            };

            w.advance(skip);
        }

        if (w.writableSliceGreedy(align_bytes)) |dest| {
            const phys_buf = dest.ptr - file_offset;
            const phys_pos = self.file_reader.pos - file_offset;
            const phys_size = std.mem.alignBackward(usize, @min(dest.len, requested_bytes) + file_offset, align_bytes);

            if (phys_size > file_offset) {
                const phys_read = self.file.readPositional(self.io, &.{phys_buf[0..phys_size]}, phys_pos) catch unreachable;
                if (phys_read > file_offset) {
                    const bytes_read = phys_read - file_offset;
                    w.advance(bytes_read);
                    self.file_reader.pos += bytes_read;
                    self.remaining -= bytes_read;
                    return bytes_read;
                }
            }
        } else |_| {}

        const bytes_read = try self.file_reader.interface.stream(w, .limited64(requested_bytes));
        self.remaining -= bytes_read;

        return bytes_read;
    }

    fn discardAligned(r: *std.Io.Reader, limit: std.Io.Limit) std.Io.Reader.Error!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) return 0;

        if (self.padding_remaining > 0) {
            _ = try self.file_reader.interface.discard(.limited64(self.padding_remaining));
            self.padding_remaining = 0;
        }

        const combined_limit = limit.min(.limited64(self.remaining));
        const n = try self.file_reader.interface.discard(combined_limit);
        self.remaining -= n;
        return n;
    }
};

pub const Tensor = struct {
    file_uri: []const u8,
    name: []const u8,
    shape: Shape,
    offset: u64,

    pub fn byteSize(self: Tensor) u64 {
        return self.shape.byteSize();
    }

    pub fn reader(self: Tensor, io: std.Io, buffer: []u8, opts: TensorReader.InitOpts) !TensorReader {
        return TensorReader.init(io, self, buffer, opts);
    }

    pub fn parse(object: std.json.ObjectMap, file_uri: []const u8, name: []const u8, base_offset: u64) !Tensor {
        const shape_field = object.get("shape").?.array;
        if (shape_field.items.len > Shape.MAX_RANK) {
            log.warn("Can't load tensor {s}, too many dims: {}", .{ name, shape_field.items.len });
            return error.TooManyDimensions;
        }

        const offset_field = object.get("data_offsets").?;
        const start: u64 = @intCast(offset_field.array.items[0].integer);
        const dtype = try stringToDtype(object.get("dtype").?.string);

        var dims: Shape.DimsArray = .empty;
        for (shape_field.items) |d| {
            dims.appendAssumeCapacity(d.integer);
        }

        return .{
            .file_uri = file_uri,
            .name = name,
            .shape = .init(dims.slice(), dtype),
            .offset = base_offset + start,
        };
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

pub const FileType = enum {
    full_index,
    index,
    safetensors,
    unknown,

    pub fn resolve(path: []const u8) FileType {
        if (std.mem.endsWith(u8, path, ".safetensors.index.json")) {
            return .index;
        } else if (std.mem.endsWith(u8, path, ".safetensors.full_index.json")) {
            return .full_index;
        } else if (std.mem.endsWith(u8, path, ".safetensors")) {
            return .safetensors;
        } else {
            return .unknown;
        }
    }
};

pub const ModelPathResolutionError = error{
    FileNotFound,
    InvalidPath,
} || std.mem.Allocator.Error;

pub fn resolveModelRepo(io: std.Io, path: []const u8) ModelPathResolutionError!std.Io.Dir {
    switch (FileType.resolve(path)) {
        .full_index, .index, .safetensors => {
            // The path is a file, let's open it's folder
            const dir_path = std.fs.path.dirname(path) orelse return ModelPathResolutionError.InvalidPath;

            const dir = std.Io.Dir.openDir(.cwd(), io, dir_path, .{}) catch |e| {
                return switch (e) {
                    error.FileNotFound => ModelPathResolutionError.FileNotFound,
                    else => ModelPathResolutionError.InvalidPath,
                };
            };

            return dir;
        },
        .unknown => {
            // The path is probably a directory, let's try to open it as such
            const dir = std.Io.Dir.openDir(.cwd(), io, path, .{}) catch |e| {
                return switch (e) {
                    error.FileNotFound => ModelPathResolutionError.FileNotFound,
                    else => ModelPathResolutionError.InvalidPath,
                };
            };
            return dir;
        },
    }
}

pub fn resolveModelEntrypoint(io: std.Io, repo: std.Io.Dir) ModelPathResolutionError!struct { FileType, std.Io.File } {
    const default_entrypoints: []const struct { FileType, []const u8 } = &.{
        .{ .full_index, "model.safetensors.full_index.json" },
        .{ .full_index, "consolidated.safetensors.full_index.json" },
        .{ .index, "model.safetensors.index.json" },
        .{ .index, "consolidated.safetensors.index.json" },
        .{ .safetensors, "model.safetensors" },
        .{ .safetensors, "consolidated.safetensors" },
    };

    for (default_entrypoints) |entrypoint| {
        const file_type, const file_name = entrypoint;
        const file = repo.openFile(io, file_name, .{ .mode = .read_only }) catch continue;

        return .{ file_type, file };
    }

    return error.FileNotFound;
}

const safetensors_dtype_to_zml_dtype = std.StaticStringMap(DataType).initComptime(.{
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

fn stringToDtype(safetensor_dtype: []const u8) !DataType {
    return safetensors_dtype_to_zml_dtype.get(safetensor_dtype) orelse {
        log.err("Unsupported safetensor data type: {s}", .{safetensor_dtype});
        return error.UnsupportedDataType;
    };
}
