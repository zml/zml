const std = @import("std");
const testing = std.testing;

const async = @import("async");
const stdx = @import("stdx");

const zml = @import("../../zml.zig");
const HostBuffer = zml.HostBuffer;
const eval = @import("eval.zig");
const pickle = @import("pickle.zig");
const py = @import("py.zig");

const log = std.log.scoped(.@"zml/aio");

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(File);
}

pub const File = struct {
    mmap_file: zml.aio.MemoryMappedFile,
    /// Map names to sub file
    file_map: std.StringArrayHashMapUnmanaged([]const u8) = .{},
    zip_prefix: []const u8,
    pickle_subfile: []const u8,

    const magic = "PK\x03\x04";

    pub fn init(allocator: std.mem.Allocator, mmap_file: zml.aio.MemoryMappedFile) !File {
        var pkl: []const u8 = mmap_file.data;
        var zip_prefix: []const u8 = &.{};
        var file_map: std.StringArrayHashMapUnmanaged([]const u8) = .{};
        if (std.mem.eql(u8, mmap_file.data[0..magic.len], magic)) {
            // We are dealing with a zip file.
            // Let's look for the `data.pkl` file and keep a map of all other files.
            // The other files will be the tensor storage and will be reference from `data.pkl`.
            var header_parsing_buffer: [4096]u8 = undefined;

            // std.zip requires on a std.fs.File and don't leverage std.Io.Reader directly.
            // So we use the synchronous API to parse the headers,
            // then we rely only on the memory map data to parse the pickle and load the buffers.
            // To mitigate this we use `async.launchBlocking` in `torch.open`.
            const raw_file: std.fs.File = .{ .handle = mmap_file.file._handle };
            var reader = raw_file.reader(&header_parsing_buffer);
            var it: std.zip.Iterator = try .init(&reader);

            while (try it.next()) |header| {
                if (header.filename_len == 0) {
                    continue;
                }
                if (header.compression_method != .store) {
                    return error.Unsupported;
                }

                const filename = mmap_file.data[header.header_zip_offset + @sizeOf(std.zip.CentralDirectoryFileHeader) ..][0..header.filename_len];

                var local_reader: std.Io.Reader = .fixed(mmap_file.data);
                local_reader.discardAll(header.file_offset) catch return error.InvalidZipFile;
                const local_header = local_reader.takeStruct(std.zip.LocalFileHeader, .little) catch return error.InvalidZipFile;
                local_reader.discardAll(local_header.filename_len) catch return error.InvalidZipFile;
                local_reader.discardAll(local_header.extra_len) catch return error.InvalidZipFile;

                // normalize path separators
                const file_content = mmap_file.data[local_reader.seek..][0..header.compressed_size];
                const my_filename: []u8 = try allocator.dupe(u8, filename);
                std.mem.replaceScalar(u8, my_filename, '\\', '/');

                try file_map.put(allocator, my_filename, file_content);

                if (std.mem.endsWith(u8, filename, "data.pkl")) {
                    pkl = file_content;
                    zip_prefix = filename[0 .. filename.len - "data.pkl".len];
                }
            }

            if (pkl.len == 0) {
                log.err("Could not find file ending in `data.pkl` in archive", .{});
                return error.PickleNotFound;
            }
        }

        return .{
            .mmap_file = mmap_file,
            .file_map = file_map,
            .pickle_subfile = pkl,
            .zip_prefix = zip_prefix,
        };
    }

    pub fn close(self: *File) void {
        self.mmap_file.deinit();
    }

    pub fn parsePickle(self: *File, allocator: std.mem.Allocator) ![]const pickle.Op {
        var reader: std.Io.Reader = .fixed(self.pickle_subfile);
        return try pickle.parse(allocator, &reader);
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
            try self.parseValue(allocator, store, .initBuffer(&prefix_buf), item);
        }
    }

    pub fn parseValue(self: File, allocator: std.mem.Allocator, store: *zml.aio.BufferStore, prefix: std.ArrayList(u8), v: py.Any) !void {
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
                                var values: std.ArrayList(ItemType) = .{};
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
                                        new_prefix.items.len += std.fmt.printInt(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
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
                                        new_prefix.items.len += std.fmt.printInt(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
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
                                        new_prefix.items.len += std.fmt.printInt(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
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
                                    new_prefix.items.len += std.fmt.printInt(new_prefix.unusedCapacitySlice(), int, 10, .lower, .{});
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

    fn parseTorchGlobal(self: File, allocator: std.mem.Allocator, store: *zml.aio.BufferStore, prefix: std.ArrayList(u8), v: py.Any) !bool {
        return switch (v) {
            .global => |object| {
                if (try self.parseTensor(allocator, object)) |host_buffer| {
                    const key = try allocator.dupe(u8, prefix.items);
                    const entry = try store.buffers.getOrPut(allocator, key);
                    if (entry.found_existing) {
                        log.warn("Duplicate key: {s}", .{prefix.items});
                        allocator.free(key);
                    }
                    entry.value_ptr.* = host_buffer;
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
        return self.file_map.get(filename) orelse {
            std.log.err("Could not find file ending in `{s}` in archive", .{filename});
            return error.TensorNotFound;
        };
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

test "Read pickle (zipped)" {
    // test file created with following python snippet:
    //
    // import torch
    // torch.manual_seed(0)
    // model = torch.nn.Conv2d(2, 2, 3, stride=2, padding=[2, 4], dtype=torch.float16)
    // tensor = torch.tensor([[2, 4, 3, 2]], dtype=torch.uint8)
    // torch.save({ "model": model, "tensor": tensor}, "simple.pt")
    const file = try async.File.open("zml/aio/torch/simple.pt", .{ .mode = .read_only });
    const mmap_file = try zml.aio.MemoryMappedFile.init(file);
    var store = try zml.aio.BufferStore.initWithFiles(testing.allocator, &.{mmap_file});
    defer store.deinit();

    {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        var torch_file = try File.init(arena.allocator(), mmap_file);
        // We don't close the file directly, it will be closed by the store.

        const ops = try torch_file.parsePickle(arena.allocator());
        try std.testing.expectEqual(302, ops.len);

        const py_values = try eval.evaluate(arena.allocator(), ops, true);
        try torch_file.parseModel(py_values, &store);
    }

    // now we have freed the arena.
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
