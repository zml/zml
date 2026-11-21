const std = @import("std");

const meta = @import("zml/meta");
const mlir = @import("mlir");
const pjrt = @import("pjrt");
const stablehlo = @import("mlir/dialects/stablehlo");

pub const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/pjrtx");

/// Where `Client.compile` writes what it hands to PJRT_Client_Compile, before compiling:
/// `<name>.mlirbc` (the StableHLO portable artifact, exact bytes), `<name>.mlir`
/// (the same module as text) and `<name>.compile_options.pb` (the serialized
/// CompileOptionsProto). A later program of the same name overwrites them.
pub const ProgramDump = struct {
    dir: []const u8,
    name: []const u8,

    fn write(self: ProgramDump, io: std.Io, allocator: std.mem.Allocator, module: *const mlir.Module, portable_artifact: []const u8, compile_options_pb: []const u8) !void {
        var dir = try std.Io.Dir.cwd().createDirPathOpen(io, self.dir, .{});
        defer dir.close(io);

        const bytecode_name = try std.fmt.allocPrint(allocator, "{s}.mlirbc", .{self.name});
        defer allocator.free(bytecode_name);
        try dir.writeFile(io, .{ .sub_path = bytecode_name, .data = portable_artifact });

        const options_name = try std.fmt.allocPrint(allocator, "{s}.compile_options.pb", .{self.name});
        defer allocator.free(options_name);
        try dir.writeFile(io, .{ .sub_path = options_name, .data = compile_options_pb });

        var text: std.Io.Writer.Allocating = .init(allocator);
        defer text.deinit();
        try text.writer.print("{f}\n", .{module.operation().fmt(.{})});
        const text_name = try std.fmt.allocPrint(allocator, "{s}.mlir", .{self.name});
        defer allocator.free(text_name);
        try dir.writeFile(io, .{ .sub_path = text_name, .data = text.written() });
    }
};

pub const Client = opaque {
    pub const CompileError = std.mem.Allocator.Error || error{InvalidMlirBytecodeVersion} || pjrt.ApiError;

    pub fn compile(client: *const pjrt.Client, api: *const pjrt.Api, allocator: std.mem.Allocator, io: std.Io, module: *const mlir.Module, compile_options_pb: []const u8, dump: ?ProgramDump) CompileError!*pjrt.LoadedExecutable {
        const stablehlo_version = blk: {
            if (api.stablehloCurrentVersion()) |requested_version| {
                break :blk stablehlo.smallerVersion(requested_version, stablehlo.currentVersion());
            }
            break :blk stablehlo.minimumVersion();
        };

        const cloned_op = module.operation().clone();
        var cloned_module = mlir.Module.fromOperation(cloned_op);
        defer cloned_module.deinit();

        var serialized_buffer: std.Io.Writer.Allocating = try .initCapacity(allocator, 4096);
        defer serialized_buffer.deinit();

        stablehlo.serializePortableArtifact2(cloned_module, stablehlo_version, &serialized_buffer.writer) catch |err| {
            log.err("failed to serialize to portable artifact: {}", .{err});
            return switch (err) {
                std.Io.Writer.Error.WriteFailed => error.OutOfMemory,
                else => |e| e,
            };
        };

        if (dump) |d| d.write(io, allocator, module, serialized_buffer.written(), compile_options_pb) catch |err| {
            log.warn("failed to dump the PJRT program {s} to {s}: {}", .{ d.name, d.dir, err });
        };

        return @ptrCast(try client.compile(api, io, .{
            .bytecode = serialized_buffer.written(),
            .bytecode_format = .mlir,
            .compile_options_pb = compile_options_pb,
        }));
    }
};

pub fn TensorToCustomCallBuffer(T: type) type {
    return meta.MapRestrict(Tensor, CustomCallBuffer).map(T);
}

pub fn ShapeToCustomCallBuffer(T: type) type {
    return meta.MapRestrict(Shape, CustomCallBuffer).map(T);
}

pub const CustomCallBuffer = struct {
    shape: Shape,
    ptr: *anyopaque,

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try writer.print("{f}@{*}", .{ self.shape, self.ptr });
    }

    pub fn fromPjrt(buffer: *const pjrt.ffi.Buffer) @This() {
        const dt = switch (buffer.dtype) {
            .invalid => @panic("Found an invalid pjrt buffer"),
            .token => @panic("Found an token pjrt buffer"),
            inline else => |tag| @field(DataType, @tagName(tag)),
        };
        return .{
            .ptr = buffer.data,
            .shape = .init(buffer.dims(), dt),
        };
    }

    // XLA:
    //   Align to 64-bytes, to mimic tsl::Allocator::kAllocatorAlignment.
    //
    //   Preferred XLA:CPU alignment for buffers. XLA:CPU itself aligns intermediate
    //   buffers to this boundary (slices inside the preallocated temp buffer).
    // https://github.com/openxla/xla/blob/fa2b7cfd909e8ffd343da1b7c5c341ace5e53aaa/xla/backends/cpu/alignment.h#L4
    pub const ALIGN = 64;

    // Assumes as host-side buffer
    pub fn asHostSlice(self: CustomCallBuffer) []align(ALIGN) u8 {
        const bytes: [*]align(ALIGN) u8 = @ptrCast(@alignCast(self.ptr));
        return bytes[0..self.shape.byteSize()];
    }

    // Assumes as host-side buffer
    pub fn asHostConstSlice(self: CustomCallBuffer) []align(ALIGN) const u8 {
        const bytes: [*]align(ALIGN) const u8 = @ptrCast(@alignCast(self.ptr));
        return bytes[0..self.shape.byteSize()];
    }

    // Assumes as host-side buffer
    pub fn asHostScalar(self: CustomCallBuffer, T: type) T {
        std.debug.assert(self.shape.byteSize() == @sizeOf(T));
        return std.mem.bytesAsValue(T, self.asHostConstSlice()[0..@sizeOf(T)]).*;
    }
};

pub fn bufferTypeFromDtype(dt: DataType) pjrt.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrt.BufferType, @tagName(tag)),
    };
}

pub fn dtypeFromBufferType(pjrt_type: pjrt.BufferType) DataType {
    return switch (pjrt_type) {
        .invalid => @panic("Found an invalid pjrt buffer"),
        inline else => |tag| @field(DataType, @tagName(tag)),
    };
}

test bufferTypeFromDtype {
    inline for (@typeInfo(DataType).@"enum".field_values) |field_value| {
        const dt: DataType = @fromBackingInt(@intCast(field_value));
        try std.testing.expectEqual(dt, dtypeFromBufferType(bufferTypeFromDtype(dt)));
    }

    inline for (@typeInfo(pjrt.BufferType).@"enum".field_values) |field_value| {
        const dt: pjrt.BufferType = @fromBackingInt(@intCast(field_value));
        if (dt == .invalid) continue;
        try std.testing.expectEqual(dt, bufferTypeFromDtype(dtypeFromBufferType(dt)));
    }
}
