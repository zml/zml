const asynk = @import("async");
const std = @import("std");
const utils = @import("../utils.zig");
const zml = @import("../../zml.zig");

const assert = std.debug.assert;
const log = std.log.scoped(.zml_io);

pub const GgufErrors = error{
    ValueTypeMismatch,
    InvalidGguf,
    UnsupportedGgufType,
    EndOfMetadata,
    OutOfMemory,
};

// Enums and structures
pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    deprecated_q4_2 = 4,
    deprecated_q4_3 = 5,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    // k-quantizations
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    i8 = 16,
    i16 = 17,
    i32 = 18,

    const MAX_KNOWN_ENUM = 18;

    pub fn canConvertQuant(self: TensorType) bool {
        return switch (self) {
            .q8_0, .q4_k, .q6_k, .q2_k, .q4_0, .q4_1 => true,
            else => false,
        };
    }

    pub fn toDtype(self: TensorType) ?zml.DataType {
        return switch (self) {
            .f32 => .f32,
            .f16 => .f16,
            .i8 => .i8,
            .i16 => .i16,
            .i32 => .i32,
            else => null,
        };
    }

    pub fn sizeOf(self: TensorType) usize {
        return self.toDtype().?.sizeOf();
    }

    /// Return the tensor type features
    pub fn getFeatures(t: TensorType) TensorTypeFeatures {
        return switch (t) {
            inline else => |val| @field(TENSOR_TYPE_FEATURES, @tagName(val)),
        };
    }
};

/// GGUF tensor type to features lookup table.
pub const TensorTypeFeatures = struct {
    items_per_block: u29,
    bytes_per_block: u29,

    pub fn alignment(features: TensorTypeFeatures) u8 {
        return std.math.log2_int(u29, features.bytes_per_block);
    }
};

pub const TENSOR_TYPE_FEATURES: std.enums.EnumFieldStruct(TensorType, TensorTypeFeatures, null) = .{
    .f32 = .{ .items_per_block = 1, .bytes_per_block = @sizeOf(f32) },
    .f16 = .{ .items_per_block = 1, .bytes_per_block = @sizeOf(f16) },
    .q4_0 = .{ .items_per_block = 32, .bytes_per_block = 18 },
    .q4_1 = .{ .items_per_block = 32, .bytes_per_block = 20 },
    .deprecated_q4_2 = .{ .items_per_block = 0, .bytes_per_block = 0 },
    .deprecated_q4_3 = .{ .items_per_block = 0, .bytes_per_block = 0 },
    .q5_0 = .{ .items_per_block = 32, .bytes_per_block = 22 },
    .q5_1 = .{ .items_per_block = 32, .bytes_per_block = 24 },
    .q8_0 = .{ .items_per_block = 32, .bytes_per_block = 34 },
    .q8_1 = .{ .items_per_block = 32, .bytes_per_block = 40 },
    .q2_k = .{ .items_per_block = 256, .bytes_per_block = 82 },
    .q3_k = .{ .items_per_block = 256, .bytes_per_block = 110 },
    .q4_k = .{ .items_per_block = 256, .bytes_per_block = 144 },
    .q5_k = .{ .items_per_block = 256, .bytes_per_block = 176 },
    .q6_k = .{ .items_per_block = 256, .bytes_per_block = 210 },
    .q8_k = .{ .items_per_block = 256, .bytes_per_block = 292 },
    .i8 = .{ .items_per_block = 1, .bytes_per_block = @sizeOf(i8) },
    .i16 = .{ .items_per_block = 1, .bytes_per_block = @sizeOf(i16) },
    .i32 = .{ .items_per_block = 1, .bytes_per_block = @sizeOf(i32) },
};

pub const GgufValueType = enum(u32) {
    // The value is a 8-bit unsigned integer.
    uint8 = 0,
    // The value is a 8-bit signed integer.
    int8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    uint16 = 2,
    // The value is a 16-bit signed little-endian integer.
    int16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    uint32 = 4,
    // The value is a 32-bit signed little-endian integer.
    int32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    float32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model
    // being invalid or the reader being buggy.
    bool = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    string = 8,
    // The value is an array of other values, with the length and type
    // prepended. Arrays can be nested, and the length of the array is the
    // number of elements in the array, not the number of bytes.
    array = 9,
    // The value is a 64-bit unsigned little-endian integer.
    uint64 = 10,
    // The value is a 64-bit signed little-endian integer.
    int64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    float64 = 12,
    // Special values used by the callbacks of gguf_do_with_value().
    array_start = 100,
    array_end = 101,

    // Allow other values in case GGUF add more types without us noticing
    _,

    pub fn sizeOf(self: GgufValueType) usize {
        return switch (self) {
            .uint8 => @sizeOf(u8),
            .int8 => @sizeOf(i8),
            .uint16 => @sizeOf(u16),
            .int16 => @sizeOf(i16),
            .uint32 => @sizeOf(u32),
            .int32 => @sizeOf(i32),
            .float32 => @sizeOf(f32),
            .bool => @sizeOf(bool),
            .uint64 => @sizeOf(u64),
            .int64 => @sizeOf(i64),
            .float64 => @sizeOf(f64),
            .string => @sizeOf([]u8),
            else => unreachable,
        };
    }

    pub fn arrayTypeCheck(self: GgufValueType, comptime T: type) !void {
        switch (self) {
            .string => if (T != []u8 and T != []const u8) return error.ValueTypeMismatch,
            .uint8 => if (T != u8) return error.ValueTypeMismatch,
            .int8 => if (T != i8) return error.ValueTypeMismatch,
            .uint16 => if (T != u16) return error.ValueTypeMismatch,
            .int16 => if (T != i16) return error.ValueTypeMismatch,
            .uint32 => if (T != u32) return error.ValueTypeMismatch,
            .int32 => if (T != i32) return error.ValueTypeMismatch,
            .float32 => if (T != f32) return error.ValueTypeMismatch,
            .bool => if (T != bool) return error.ValueTypeMismatch,
            .uint64 => if (T != u64) return error.ValueTypeMismatch,
            .int64 => if (T != i64) return error.ValueTypeMismatch,
            .float64 => if (T != f64) return error.ValueTypeMismatch,
            else => {},
        }
    }
};

pub const ValueType = enum {
    uint8,
    int8,
    uint16,
    int16,
    uint32,
    int32,
    float32,
    uint64,
    int64,
    float64,
    boolval,
    string,
    array,
};

// Union of possible values.
pub const GgufValue = union(ValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    uint64: u64,
    int64: i64,
    float64: f64,
    boolval: bool,
    string: []const u8,
    array: Array,

    pub const Array = struct {
        // Any value type is valid, including arrays.
        child: GgufValueType,
        // Number of elements, not bytes
        len: usize,
        data: []u8,
    };

    pub fn asLoaderValue(self: GgufValue) zml.aio.Value {
        return switch (self) {
            .array => |v| .{
                .array = .{
                    .item_type = switch (v.child) {
                        .bool => .boolval,
                        .uint8 => .uint8,
                        .int8 => .int8,
                        .uint16 => .uint16,
                        .int16 => .int16,
                        .uint32 => .uint32,
                        .int32 => .int32,
                        .float32 => .float32,
                        .uint64 => .uint64,
                        .int64 => .int64,
                        .float64 => .float64,
                        .string => .string,
                        // TODO: .array => .array,
                        else => unreachable,
                    },
                    .data = v.data,
                },
            },
            inline else => |v, tag| @unionInit(zml.aio.Value, @tagName(tag), v),
        };
    }
};

// Header
const GgufHeader = extern struct {
    // Magic number to announce that this is a GGUF file. Must be `GUFF`.
    magic: [4]u8,
    // The version of the format implemented.
    // Must be `3` for version described in this spec.
    version: u32,
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure
    // it is always present for loading the tensors.
    tensor_count: usize,
    // The number of metadata key-value pairs.
    metadata_kv_count: usize,

    pub fn validate(self: GgufHeader) !void {
        if (!std.mem.eql(u8, &self.magic, "GGUF")) {
            log.err("Invalid GGUF file: wrong header {s}", .{self.magic});
            return error.InvalidGguf;
        }
    }
};

// Key representation in this library API.
pub const GgufMetadataKv = struct {
    name: []const u8,
    type_: GgufValueType,
    val: GgufValue,
};

// Tensor representation in this library API.
const GGUF_TENSOR_MAX_DIM: usize = 8; // Future-proof: actual limit is 4.
pub const GgufTensorInfo = struct {
    name: []const u8,
    t: TensorType, // Tensor type (enum TensorType).
    rank: usize, // Number of dimensions of the tensor.
    dims: [GGUF_TENSOR_MAX_DIM]i64, // Dimensions (Eg. [512, 1024, 1, 1]).
    start: usize, // Offset from start of data section.
    byte_len: usize, // Total size in bytes.
    num_weights: usize, // Total number of parameters.

    pub inline fn shape(info: GgufTensorInfo) []const i64 {
        return info.dims[0..info.rank];
    }
};

// Return the value type name given the type ID.
fn getValueTypeName(t: u32) []const u8 {
    if (@as(usize, @intCast(t)) >= GGUF_VALUE_NAME.len) return "unknown";
    return GGUF_VALUE_NAME[@intCast(t)];
}

const GGUF_VALUE_NAME = [_][]const u8{
    "uint8",   "int8", "uint16", "int16", "uint32", "int32",
    "float32", "bool", "string", "array", "uint64", "int64",
    "float64",
};

/// GGUF file API
/// A memory-mapped view of a .gguf file.
/// Format used by GGML models: https://github.com/ggerganov/ggml/
pub const GgufFile = struct {
    header: GgufHeader, // GUFF file header info.
    size: usize, // Total file size.
    file: zml.aio.MemoryMappedFile,
    left_kv: usize, // Number of key-value pairs yet to read.
    left_tensors: usize, // Number of tensors yet to read.
    off: usize, // Offset of the next item to parse.
    alignment: usize = 32, // File data alignment. Default: 32 bytes.

    /// Open and memmap the given file.
    pub fn open(path: []const u8) !GgufFile {
        const file = try asynk.File.open(path, .{});
        const header = try file.reader().readStruct(GgufHeader);
        try header.validate();
        return .{
            .header = header,
            .size = (try file.stat()).size,
            .file = try zml.aio.MemoryMappedFile.init(file),
            .off = @sizeOf(GgufHeader),
            .left_kv = header.metadata_kv_count,
            .left_tensors = header.tensor_count,
        };
    }

    ///  Unmap the file memory and close the file handle.
    pub fn close(self: *GgufFile) void {
        self.file.deinit();
    }

    /// Set the context to read the first key-value entry in the GGUF
    /// file and then all the rest. Is used when creating a new context
    /// and also when you want to restart scanning the key-value
    /// items in the file.
    fn rewind(ctx: *GgufFile) void {
        ctx.off = @sizeOf(GgufHeader);
        ctx.left_kv = ctx.header.metadata_kv_count;
        ctx.left_tensors = ctx.header.tensor_count;
    }

    pub fn seek(self: *GgufFile, pos: usize) void {
        assert(pos < self.size);
        self.off = pos;
    }

    fn readInt(self: *GgufFile, comptime T: type) !T {
        if (self.off + @sizeOf(T) >= self.size) return error.InvalidGguf;
        const res = self.file.file.reader().readInt(T, .little);
        self.off += @sizeOf(T);
        return res;
    }

    fn readTensorType(self: *GgufFile) !TensorType {
        const raw = try self.readInt(u32);
        if (raw > TensorType.MAX_KNOWN_ENUM) {
            log.err("Unsupported GGUF tensor type: {d}", .{raw});
            return error.UnsupportedGgufType;
        }
        return @enumFromInt(raw);
    }

    fn readValueType(self: *GgufFile) !GgufValueType {
        const raw = try self.readInt(u32);
        const t: GgufValueType = @enumFromInt(raw);
        switch (t) {
            .uint8, .int8, .uint16, .int16, .uint32, .int32, .float32, .bool, .string, .array, .uint64, .int64, .float64, .array_start, .array_end => {},
            else => {
                log.err("Unsupported GGUF value type: {s}", .{@tagName(t)});
                return error.UnsupportedGgufType;
            },
        }
        return t;
    }

    pub fn readAlloc(self: *GgufFile, allocator: std.mem.Allocator, len: usize) ![]u8 {
        const data = try allocator.alloc(u8, len);
        const read = try self.file.file.reader().readAll(data);
        if (read != data.len) return error.InvalidGguf;
        self.off += len;
        return data;
    }

    pub fn skipBytes(self: *GgufFile, len: usize) !void {
        try self.file.file.seekBy(@intCast(len));
        self.off += len;
    }

    /// Read the len then the actual bytes.
    pub fn readString(self: *GgufFile, allocator: std.mem.Allocator) ![]u8 {
        const len: usize = try self.readInt(u64);
        return self.readAlloc(allocator, len);
    }

    pub fn skipString(self: *GgufFile) !void {
        const len: usize = try self.readInt(u64);
        return self.skipBytes(len);
    }

    fn readArrayHeader(self: *GgufFile, allocator: std.mem.Allocator) !GgufValue.Array {
        const child = try self.readValueType();
        const len: usize = try self.readInt(u64);
        const data = switch (child) {
            // Since strings have variable lenghts, we need to read them one by one
            .string => str: {
                var data = try allocator.alloc([]u8, len);
                for (0..len) |i| data[i] = try self.readString(allocator);
                break :str std.mem.sliceAsBytes(data);
            },
            else => try self.readAlloc(allocator, len * child.sizeOf()),
        };
        return .{
            .child = child,
            .len = len,
            .data = data,
        };
    }

    fn readTypedValue(self: *GgufFile, allocator: std.mem.Allocator, t: GgufValueType) !GgufValue {
        return switch (t) {
            .uint8 => .{ .uint8 = try self.readInt(u8) },
            .int8 => .{ .int8 = try self.readInt(i8) },
            .uint16 => .{ .uint16 = try self.readInt(u16) },
            .int16 => .{ .int16 = try self.readInt(i16) },
            .uint32 => .{ .uint32 = try self.readInt(u32) },
            .int32 => .{ .int32 = try self.readInt(i32) },
            .float32 => .{ .float32 = @bitCast(try self.readInt(u32)) },
            .bool => .{ .boolval = try self.readInt(u8) != 0 },
            .string => .{ .string = try self.readString(allocator) },
            .array => .{ .array = try self.readArrayHeader(allocator) },
            .uint64 => .{ .uint64 = try self.readInt(u64) },
            .int64 => .{ .int64 = try self.readInt(i64) },
            .float64 => .{ .float64 = @bitCast(try self.readInt(u64)) },
            else => error.UnsupportedGgufType,
        };
    }

    /// Parses the next metadata entry.
    /// Returns error.EndOfMetadata if there are no longer metadata to process in this GGUF file.
    pub fn readMetadata(self: *GgufFile, allocator: std.mem.Allocator) !GgufMetadataKv {
        if (self.left_kv == 0) return error.EndOfMetadata;
        self.left_kv -= 1;
        const name = try self.readString(allocator);
        const type_ = try self.readValueType();
        const val: GgufValue = try self.readTypedValue(allocator, type_);
        return .{ .name = name, .type_ = type_, .val = val };
    }

    // Set the data section offset. This function must be called exactly when
    // all the key-values are consumed, in the context of the first call of
    // ctx.getTensor(): this way we will be able to return tensor offsets
    // as absolute positions and pointers to the mmapped file.
    fn setDataOffset(self: *GgufFile) !void {
        const base_off = self.off;

        assert(self.left_kv == 0 and self.left_tensors == self.header.tensor_count);

        for (0..self.left_tensors) |_| try self.skipTensor();
        const padding: usize = getAlignmentPadding(self.alignment, self.off);
        self.file.data_offset = self.off + padding;

        try self.file.file.seekTo(base_off);
        self.off = base_off;
    }

    pub fn skipTensor(self: *GgufFile) !void {
        try self.skipString(); // Skip name
        const num_dim: u32 = try self.readInt(u32);
        // dimensions, type, and offset.
        try self.skipBytes(8 * num_dim + 4 + 8);
    }

    /// Parses the next tensor entry.
    /// Returns error.EndOfMetadata if there are no longer tensor metadata to process in this GGUF file.
    pub fn readTensorInfo(self: *GgufFile, allocator: std.mem.Allocator) !GgufTensorInfo {
        if (self.left_tensors == 0 or self.left_kv != 0) {
            return error.EndOfMetadata;
        }

        // We want to return tensor data with offsets relative to the start
        // of the file, so that the user of the API is able to access tensors
        // as it iterates over them. To do so, we need to perform a full
        // scan if this is the first tensor info we are reading.
        // TODO: explicitly set the data offset in
        if (self.file.data_offset == 0) try self.setDataOffset();
        self.left_tensors -= 1;
        const name = try self.readString(allocator);
        const num_dim = try self.readInt(u32);
        assert(@as(usize, @intCast(num_dim)) <= GGUF_TENSOR_MAX_DIM);
        // Read the dimentions; unused dimensions are left `undefined`.
        // Note: we reverse the order of the dimensions to match zml convention.
        var dims: [GGUF_TENSOR_MAX_DIM]i64 = undefined;
        var num_weights: usize = 1;
        for (0..num_dim) |j| {
            const d = try self.readInt(u64);
            dims[num_dim - 1 - j] = @intCast(d);
            num_weights *= d;
        }
        const t: TensorType = try self.readTensorType();
        const start = try self.readInt(u64);
        // To accurately calculate the bytes used by this tensor on the GGUF
        // file, we need to take into account that quantization methods store
        // tensors as block of N weights. So first of all we need to understand
        // the number of padding weights (since the last block may have just
        // fewer weights stored inside, but still requires to be stored to its full
        // length). Then we can do the math to see how many blocks we need, and
        // multiply by the block size to obtain the final total size.
        const tf = t.getFeatures();
        const byte_len: usize = (std.math.divCeil(usize, num_weights, tf.items_per_block) catch unreachable) * tf.bytes_per_block;
        return .{
            .name = name,
            .t = t,
            .rank = num_dim,
            .dims = dims,
            .start = start,
            .byte_len = byte_len,
            .num_weights = num_weights,
        };
    }
};

/// Given an offset or a length, returns the padding needed to align it to alignment.
fn getAlignmentPadding(alignment: usize, offset: usize) usize {
    return @rem((alignment - @rem(offset, alignment)), alignment);
}
