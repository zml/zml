const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.tools);

const truncated_shape_msg = "torch.Size([...])";
const truncated_values_msg = "[...]";
const unsupported_dtype_msg = "[unsupported dtype]";

pub const NumpyData = struct {
    shape: []usize,
    data: []align(8) u8,
    allocated_mem: []align(8) u8,
    dtype: []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *NumpyData) void {
        self.allocator.free(self.shape);
        self.allocator.free(self.allocated_mem);
        self.allocator.free(self.dtype);
    }

    pub fn items(self: NumpyData, comptime T: type) []T {
        const aligned_data = @as([]align(8) u8, @alignCast(self.data));
        return std.mem.bytesAsSlice(T, aligned_data);
    }

    pub fn toTensor(self: NumpyData) zml.Tensor {
        // Map Dtype
        // Supported: <f4 (f32), <i8 (i64), <i4 (i32)
        var dtype: zml.DataType = .f32;
        if (std.mem.indexOf(u8, self.dtype, "f4")) |_| {
            dtype = .f32;
        } else if (std.mem.indexOf(u8, self.dtype, "i8")) |_| {
            dtype = .i64;
        } else if (std.mem.indexOf(u8, self.dtype, "i4")) |_| {
            dtype = .i32;
        } else {
            log.warn("NumpyData.toTensor: Unknown dtype {s}, defaulting to f32", .{self.dtype});
        }

        switch (self.shape.len) {
            1 => return zml.Tensor.fromShape(zml.Shape.init(.{@as(i64, @intCast(self.shape[0]))}, dtype)),
            2 => return zml.Tensor.fromShape(zml.Shape.init(.{ @as(i64, @intCast(self.shape[0])), @as(i64, @intCast(self.shape[1])) }, dtype)),
            3 => return zml.Tensor.fromShape(zml.Shape.init(.{ @as(i64, @intCast(self.shape[0])), @as(i64, @intCast(self.shape[1])), @as(i64, @intCast(self.shape[2])) }, dtype)),
            4 => return zml.Tensor.fromShape(zml.Shape.init(.{ @as(i64, @intCast(self.shape[0])), @as(i64, @intCast(self.shape[1])), @as(i64, @intCast(self.shape[2])), @as(i64, @intCast(self.shape[3])) }, dtype)),
            5 => return zml.Tensor.fromShape(zml.Shape.init(.{ @as(i64, @intCast(self.shape[0])), @as(i64, @intCast(self.shape[1])), @as(i64, @intCast(self.shape[2])), @as(i64, @intCast(self.shape[3])), @as(i64, @intCast(self.shape[4])) }, dtype)),
            else => {
                log.err("NumpyData.toTensor: Unsupported rank {d}", .{self.shape.len});
                return zml.Tensor.fromShape(zml.Shape.init(.{}, dtype));
            },
        }
    }

    pub fn print(self: NumpyData, n: usize, label: []const u8) void {
        log.info("{s} shape: {any}, dtype: {s}", .{ label, self.shape, self.dtype });

        var buf: [4096]u8 = undefined;
        var pos: usize = 0;
        _ = std.fmt.bufPrint(buf[pos..], "[", .{}) catch {};
        pos += 1;

        if (std.mem.indexOf(u8, self.dtype, "f4")) |_| {
            const data = self.items(f32);
            for (0..@min(n, data.len)) |i| {
                if (std.fmt.bufPrint(buf[pos..], "{d:.4}, ", .{data[i]})) |s| {
                    pos += s.len;
                } else |_| break;
            }
        } else if (std.mem.indexOf(u8, self.dtype, "i8")) |_| {
            const data = self.items(i64);
            for (0..@min(n, data.len)) |i| {
                if (std.fmt.bufPrint(buf[pos..], "{d}, ", .{data[i]})) |s| {
                    pos += s.len;
                } else |_| break;
            }
        } else if (std.mem.indexOf(u8, self.dtype, "i4")) |_| {
            const data = self.items(i32);
            for (0..@min(n, data.len)) |i| {
                if (std.fmt.bufPrint(buf[pos..], "{d}, ", .{data[i]})) |s| {
                    pos += s.len;
                } else |_| break;
            }
        } else {
            _ = std.fmt.bufPrint(buf[pos..], "unsupported dtype content", .{}) catch {};
            pos += 25;
        }

        _ = std.fmt.bufPrint(buf[pos..], "]", .{}) catch {};
        pos += 1;
        log.info("{s} (first {d}): {s}", .{ label, n, buf[0..pos] });
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !NumpyData {
        return loadNumpy(allocator, path);
    }

    pub fn toBuffer(self: NumpyData, io: std.Io, platform: *const zml.Platform) !zml.Buffer {
        return zml.Buffer.fromBytes(io, platform, self.toTensor().shape(), self.data);
    }
};

pub fn printBuffer(allocator: std.mem.Allocator, io: std.Io, buf: zml.Buffer, n: usize, label: []const u8) !void {
    log.info("{s} Shape: {}", .{ label, buf.shape() });
    const slice = try zml.Slice.alloc(allocator, buf.shape());
    defer slice.free(allocator);
    try buf.toSlice(io, slice);

    var buffer: [4096]u8 = undefined;
    var pos: usize = 0;
    _ = std.fmt.bufPrint(buffer[pos..], "[", .{}) catch {};
    pos += 1;

    switch (buf.shape().dtype()) {
        .f32 => {
            const data = std.mem.bytesAsSlice(f32, slice.items(u8));
            for (0..@min(n, data.len)) |idx| {
                if (std.fmt.bufPrint(buffer[pos..], "{d:.4}, ", .{data[idx]})) |s| {
                    pos += s.len;
                } else |_| break;
            }
        },
        .i64 => {
            const data = std.mem.bytesAsSlice(i64, slice.items(u8));
            for (0..@min(n, data.len)) |idx| {
                if (std.fmt.bufPrint(buffer[pos..], "{d}, ", .{data[idx]})) |s| {
                    pos += s.len;
                } else |_| break;
            }
        },
        .i32 => {
            const data = std.mem.bytesAsSlice(i32, slice.items(u8));
            for (0..@min(n, data.len)) |idx| {
                if (std.fmt.bufPrint(buffer[pos..], "{d}, ", .{data[idx]})) |s| {
                    pos += s.len;
                } else |_| break;
            }
        },
        else => {
            _ = std.fmt.bufPrint(buffer[pos..], "unsupported dtype content", .{}) catch {};
            pos += 25;
        },
    }

    _ = std.fmt.bufPrint(buffer[pos..], "]", .{}) catch {};
    pos += 1;
    log.info("{s} (first {d}): {s}", .{ label, n, buffer[0..pos] });
}

pub const PrintFlattenOptions = struct {
    include_shape: bool = false,
};

pub fn printFlatten(
    allocator: std.mem.Allocator,
    io: std.Io,
    buf: zml.Buffer,
    n: usize,
    label: []const u8,
    opts: PrintFlattenOptions,
) !void {
    const shape = buf.shape();
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    try buf.toSlice(io, slice);

    var values_buf: [4096]u8 = undefined;
    const values = formatValueList(slice, shape.dtype(), n, &values_buf);

    if (opts.include_shape) {
        var shape_buf: [128]u8 = undefined;
        const shape_str = formatTorchShape(shape, &shape_buf);
        log.info("{s} {s} : {s}", .{ label, shape_str, values });
    } else {
        log.info("{s}: {s}", .{ label, values });
    }
}

fn formatTorchShape(shape: zml.Shape, buffer: []u8) []const u8 {
    var pos: usize = 0;
    if (std.fmt.bufPrint(buffer[pos..], "torch.Size([", .{})) |written| {
        pos += written.len;
    } else |_| {
        return truncated_shape_msg;
    }

    const dims = shape.dims();
    for (dims, 0..) |dim, idx| {
        if (idx != 0) {
            if (std.fmt.bufPrint(buffer[pos..], ", ", .{})) |written| {
                pos += written.len;
            } else |_| return truncated_shape_msg;
        }
        if (std.fmt.bufPrint(buffer[pos..], "{d}", .{dim})) |written| {
            pos += written.len;
        } else |_| return truncated_shape_msg;
    }

    if (std.fmt.bufPrint(buffer[pos..], "])", .{})) |written| {
        pos += written.len;
    } else |_| {
        return truncated_shape_msg;
    }

    return buffer[0..pos];
}

fn formatValueList(slice: zml.Slice, dtype: zml.DataType, limit: usize, buffer: []u8) []const u8 {
    var pos: usize = 0;
    if (buffer.len == 0) return truncated_values_msg;
    buffer[pos] = '[';
    pos += 1;

    switch (dtype) {
        .f32 => {
            const data = slice.constItems(f32);
            if (!appendValues("{d:.6}", f32, data, limit, buffer, &pos)) return truncated_values_msg;
        },
        .i64 => {
            const data = slice.constItems(i64);
            if (!appendValues("{d}", i64, data, limit, buffer, &pos)) return truncated_values_msg;
        },
        .i32 => {
            const data = slice.constItems(i32);
            if (!appendValues("{d}", i32, data, limit, buffer, &pos)) return truncated_values_msg;
        },
        .f16 => {
            const data = slice.constItems(f16);
            if (!appendValues("{d:.6}", f16, data, limit, buffer, &pos)) return truncated_values_msg;
        },
        .bf16 => {
            const data_u16 = slice.constItems(u16);
            var temp_f32: [128]f32 = undefined;
            const count = @min(limit, @min(data_u16.len, 128));
            for (data_u16[0..count], 0..) |v, i| {
                const u32_val = @as(u32, v) << 16;
                temp_f32[i] = @bitCast(u32_val);
            }
            if (!appendValues("{d:.6}", f32, temp_f32[0..count], limit, buffer, &pos)) return truncated_values_msg;
        },
        else => return unsupported_dtype_msg,
    }

    if (pos >= buffer.len) return truncated_values_msg;
    buffer[pos] = ']';
    pos += 1;
    return buffer[0..pos];
}

fn appendValues(
    comptime fmt: []const u8,
    comptime T: type,
    values: []const T,
    limit: usize,
    buffer: []u8,
    pos: *usize,
) bool {
    const max_items = @min(limit, values.len);
    for (0..max_items) |idx| {
        if (idx != 0) {
            if (std.fmt.bufPrint(buffer[pos.*..], ", ", .{})) |written| {
                pos.* += written.len;
            } else |_| return false;
        }
        if (std.fmt.bufPrint(buffer[pos.*..], fmt, .{values[idx]})) |written| {
            pos.* += written.len;
        } else |_| return false;
    }
    return true;
}

pub fn loadNumpy(allocator: std.mem.Allocator, path: []const u8) !NumpyData {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const file = std.c.fopen(path_z.ptr, "rb") orelse {
        log.err("Failed to open file: {s}", .{path});
        return error.FileNotFound;
    };
    defer _ = std.c.fclose(file);

    var reader = CFileReader{ .file = file };

    // 1. Check Magic "\x93NUMPY"
    var magic: [6]u8 = undefined;
    _ = try reader.readFull(&magic);
    if (!std.mem.eql(u8, &magic, "\x93NUMPY")) {
        log.err("Invalid Numpy magic: {s}", .{magic});
        return error.InvalidNumpyFile;
    }

    // 2. Version
    const major = try reader.readByte();
    const minor = try reader.readByte();

    if (major != 1) {
        log.warn("Numpy version {d}.{d} not fully tested, attempting parse...", .{ major, minor });
    }

    // 3. Header Length (little endian 2 bytes)
    const header_len = try reader.readInt(u16, .little);

    // 4. Read Header
    const header_buf = try allocator.alloc(u8, header_len);
    defer allocator.free(header_buf);
    _ = try reader.readFull(header_buf);

    // 5. Parse Header (simple text parsing for dictionary)

    // Find descr
    var dtype: []const u8 = "";
    if (std.mem.indexOf(u8, header_buf, "'descr'")) |idx| {
        var i = idx + 7;
        while (i < header_buf.len and header_buf[i] != ':') : (i += 1) {}
        while (i < header_buf.len and (header_buf[i] == ':' or header_buf[i] == ' ')) : (i += 1) {}
        if (i < header_buf.len and header_buf[i] == '\'') {
            i += 1;
            const start = i;
            while (i < header_buf.len and header_buf[i] != '\'') : (i += 1) {}
            dtype = try allocator.dupe(u8, header_buf[start..i]);
        }
    }

    // Find shape
    var shape_list = std.ArrayListUnmanaged(usize){};
    defer shape_list.deinit(allocator);

    if (std.mem.indexOf(u8, header_buf, "'shape'")) |idx| {
        var i = idx + 7;
        while (i < header_buf.len and header_buf[i] != ':') : (i += 1) {}
        while (i < header_buf.len and header_buf[i] != '(') : (i += 1) {}
        i += 1;

        while (i < header_buf.len and header_buf[i] != ')') {
            while (i < header_buf.len and header_buf[i] == ' ') : (i += 1) {}
            if (header_buf[i] == ')') break;

            const start = i;
            while (i < header_buf.len and std.ascii.isDigit(header_buf[i])) : (i += 1) {}
            if (start != i) {
                const num_str = header_buf[start..i];
                const dim = try std.fmt.parseInt(usize, num_str, 10);
                try shape_list.append(allocator, dim);
            }

            while (i < header_buf.len and (header_buf[i] == ',' or header_buf[i] == ' ')) : (i += 1) {}
        }
    }

    // 6. Read Data
    // Get current pos
    const header_end = ftell(file);
    // Get file size
    _ = fseek(file, 0, 2); // SEEK_END = 2
    const file_size = ftell(file);
    _ = fseek(file, header_end, 0); // SEEK_SET = 0

    const data_size = @as(usize, @intCast(file_size - header_end));

    // Allocate u64 buffer to ensure 8-byte alignment
    const u64_len = (data_size + 7) / 8;
    const u64_buf = try allocator.alloc(u64, u64_len);
    const u8_full_slice = std.mem.sliceAsBytes(u64_buf);

    // We take the exact size we need.
    // Since it comes from u64_buf, it is aligned to 8.
    const data = @as([]align(8) u8, @alignCast(u8_full_slice[0..data_size]));

    _ = try reader.readFull(data);

    return NumpyData{
        .shape = try shape_list.toOwnedSlice(allocator),
        .data = data,
        .allocated_mem = @as([]align(8) u8, @alignCast(u8_full_slice)),
        .dtype = dtype,
        .allocator = allocator,
    };
}

// Externs for libc functions likely missing from std.c in this enviroment
pub extern "c" fn ftell(stream: *std.c.FILE) c_long;
pub extern "c" fn fseek(stream: *std.c.FILE, offset: c_long, origin: c_int) c_int;

const CFileReader = struct {
    file: *std.c.FILE,

    pub fn readFull(self: CFileReader, buf: []u8) !usize {
        const n = std.c.fread(buf.ptr, 1, buf.len, self.file);
        if (n != buf.len) return error.IncompleteRead;
        return n;
    }

    pub fn readByte(self: CFileReader) !u8 {
        var b: u8 = undefined;
        if (std.c.fread(@ptrCast(&b), 1, 1, self.file) != 1) return error.IncompleteRead;
        return b;
    }

    pub fn readInt(self: CFileReader, comptime T: type, endian: std.builtin.Endian) !T {
        var bytes: [@sizeOf(T)]u8 = undefined;
        if (std.c.fread(@ptrCast(&bytes), 1, bytes.len, self.file) != bytes.len) return error.IncompleteRead;
        return std.mem.readInt(T, &bytes, endian);
    }
};

// How to use NumpyData
// // Load Embeds
// {
//     const filename = "/Users/kevin/zml/flux_klein_notebook_embeds.Numpy";

//     var Numpy = try tools.NumpyData.load(allocator, filename);
//     defer Numpy.deinit();

//     // Print using NumpyData.print (Host)
//     Numpy.print(20, "Embeds (NumpyData)");

//     // Print using zml.Buffer (Device/Platform)
//     const buffer = try Numpy.toBuffer(io, platform_auto);
//     defer buffer.deinit();
//     try tools.printBuffer(allocator, io, buffer, 20, "Embeds (Buffer)");

// }

// // Load Text IDs
// {
//     const filename = "/Users/kevin/zml/flux_klein_notebook_text_ids.Numpy";

//     var Numpy = try tools.NumpyData.load(allocator, filename);
//     defer Numpy.deinit();

//     // Print using NumpyData.print (Host)
//     Numpy.print(20, "TextIDs (NumpyData)");

//     // Print using zml.Buffer (Device/Platform)
//     const buffer = try Numpy.toBuffer(io, platform_auto);
//     defer buffer.deinit();
//     try tools.printBuffer(allocator, io, buffer, 20, "TextIDs (Buffer)");

// }

pub fn saveBufferToNumpy(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, buf: zml.Buffer, path: []const u8) !void {
    _ = platform;
    const shape = buf.shape();
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    try buf.toSlice(io, slice);

    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const file = std.c.fopen(path_z.ptr, "wb") orelse {
        log.err("Failed to open file for writing: {s}", .{path});
        return error.CountNotOpenFile;
    };
    defer _ = std.c.fclose(file);

    // 1. Magic
    _ = std.c.fwrite("\x93NUMPY", 1, 6, file);

    // 2. Version (1.0)
    const version = [2]u8{ 1, 0 };
    _ = std.c.fwrite(&version, 1, 2, file);

    // Determine effective dtype for header (bf16 -> f32)
    var effective_dtype = shape.dtype();
    if (effective_dtype == .bf16) effective_dtype = .f32;

    // 3. Prepare Header
    // "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 3, 128, 128), }"
    var header_list = try std.ArrayList(u8).initCapacity(allocator, 128);
    defer header_list.deinit(allocator);

    try header_list.appendSlice(allocator, "{'descr': '");
    switch (effective_dtype) {
        .f32 => try header_list.appendSlice(allocator, "<f4"),
        .i64 => try header_list.appendSlice(allocator, "<i8"),
        .i32 => try header_list.appendSlice(allocator, "<i4"),
        .f16 => try header_list.appendSlice(allocator, "<f2"),
        else => return error.UnsupportedDtype,
    }
    try header_list.appendSlice(allocator, "', 'fortran_order': False, 'shape': (");

    const dims = shape.dims();
    for (dims, 0..) |d, i| {
        if (i > 0) try header_list.appendSlice(allocator, ", ");
        var dim_buf: [32]u8 = undefined;
        const s = try std.fmt.bufPrint(&dim_buf, "{d}", .{d});
        try header_list.appendSlice(allocator, s);
    }
    if (dims.len == 1) try header_list.appendSlice(allocator, ","); // Single dim tuple needs trailing comma

    try header_list.appendSlice(allocator, "), }");

    // Padding
    while ((10 + header_list.items.len + 1) % 64 != 0) {
        try header_list.append(allocator, ' ');
    }
    try header_list.append(allocator, '\n');

    // 4. Write Header Length
    const header_len: u16 = @intCast(header_list.items.len);
    const header_len_le = std.mem.nativeToLittle(u16, header_len);
    const len_bytes = std.mem.asBytes(&header_len_le);
    _ = std.c.fwrite(len_bytes.ptr, 1, 2, file);

    // 5. Write Header
    _ = std.c.fwrite(header_list.items.ptr, 1, header_list.items.len, file);

    // 6. Write Data
    if (shape.dtype() == .bf16) {
        // Convert bf16 to f32
        const src = slice.items(u16);
        const f32_data = try allocator.alloc(f32, src.len);
        defer allocator.free(f32_data);
        for (src, 0..) |s, i| {
            const u32_val = @as(u32, s) << 16;
            f32_data[i] = @bitCast(u32_val);
        }
        _ = std.c.fwrite(std.mem.sliceAsBytes(f32_data).ptr, 1, f32_data.len * 4, file);
    } else {
        const data_bytes = slice.items(u8);
        _ = std.c.fwrite(data_bytes.ptr, 1, data_bytes.len, file);
    }

    log.info("Saved buffer to {s}", .{path});
}

pub fn assertBuffersEqual(allocator: std.mem.Allocator, io: std.Io, a: zml.Buffer, b: zml.Buffer, tolerance: f32) !void {
    const a_shape = a.shape();
    const b_shape = b.shape();

    if (a_shape.dtype() != b_shape.dtype()) {
        log.warn("Comparing buffers with different dtypes: {any} vs {any}", .{ a_shape.dtype(), b_shape.dtype() });
    }

    if (!a_shape.eqlDims(b_shape)) {
        log.err("Buffer shapes don't match: {any} vs {any}", .{ a_shape.dims(), b_shape.dims() });
        return error.ShapeMismatch;
    }

    // Get data from both buffers
    const a_slice = try a.toSliceAlloc(allocator, io);
    defer a_slice.free(allocator);
    const b_slice = try b.toSliceAlloc(allocator, io);
    defer b_slice.free(allocator);

    // Get element count from shape
    const count = a_shape.count();

    var max_diff: f64 = 0.0;
    var diff_count: usize = 0;

    for (0..count) |i| {
        // Get values as f64 for high precision comparison
        const va: f64 = getElementAsF64(a_slice, a_shape.dtype(), i);
        const vb: f64 = getElementAsF64(b_slice, b_shape.dtype(), i);

        const diff = @abs(va - vb);
        if (diff > @as(f64, tolerance)) {
            diff_count += 1;
            if (diff_count <= 5) {
                log.warn("Mismatch at index {}: {d} vs {d} (diff: {d})", .{ i, va, vb, diff });
            }
        }
        max_diff = @max(max_diff, diff);
    }

    if (diff_count > 0) {
        const message = try std.fmt.allocPrint(allocator, "Buffers differ: {} mismatches out of {} elements, max diff: {d}", .{ diff_count, count, max_diff });
        defer allocator.free(message);
        @panic(message);
    }

    log.info("Buffers match! Max diff: {d}", .{max_diff});
}

pub fn getElementAsF64(slice: zml.Slice, dtype: zml.DataType, idx: usize) f64 {
    return switch (dtype) {
        .f32 => @floatCast(slice.items(f32)[idx]),
        .f64 => slice.items(f64)[idx],
        .i32 => @floatFromInt(slice.items(i32)[idx]),
        .i64 => @floatFromInt(slice.items(i64)[idx]),
        .f16 => @floatCast(slice.items(f16)[idx]),
        .bf16 => blk: {
            const raw = slice.items(u16)[idx];
            const u32_val = @as(u32, raw) << 16;
            const f_val: f32 = @bitCast(u32_val);
            break :blk @floatCast(f_val);
        },
        else => 0.0,
    };
}

pub fn parseConfig(comptime TemplateConfig: type, allocator: std.mem.Allocator, io: std.Io, repo_dir: std.Io.Dir, options: struct { subfolder: ?[]const u8 = null, json_name: ?[]const u8 = null }) !std.json.Parsed(TemplateConfig) {
    const timer_start = std.Io.Clock.awake.now(io);
    const subfolder = options.subfolder orelse "";
    const json_name = options.json_name orelse "config.json";

    const config_sub_path = if (subfolder.len > 0)
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ subfolder, json_name })
    else
        try allocator.dupe(u8, json_name);
    defer allocator.free(config_sub_path);

    defer log.info("Loaded config {s} from {s} [{d}ms]", .{ @typeName(TemplateConfig), config_sub_path, timer_start.untilNow(io, .awake).toMilliseconds() });

    const parsed_config: std.json.Parsed(TemplateConfig) = label_parsing_block: {
        const config_json_file = try repo_dir.openFile(io, config_sub_path, .{});
        defer config_json_file.close(io);
        const stat = try config_json_file.stat(io);
        const file_size = stat.size;
        const config_json_buffer = try allocator.alloc(u8, @intCast(file_size));
        defer allocator.free(config_json_buffer);
        var config_reader = config_json_file.reader(io, config_json_buffer);
        var reader: std.json.Reader = .init(allocator, &config_reader.interface);
        defer reader.deinit();

        break :label_parsing_block try std.json.parseFromTokenSource(TemplateConfig, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();
    return parsed_config;
}
