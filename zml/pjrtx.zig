const std = @import("std");

const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const pjrt = @import("pjrt");

pub const DataType = @import("dtype.zig").DataType;

const log = std.log.scoped(.@"zml/pjrtx");

pub const Client = opaque {
    pub const CompileError = std.mem.Allocator.Error || error{InvalidMlirBytecodeVersion} || pjrt.ApiError;

    pub fn compile(client: *const pjrt.Client, api: *const pjrt.Api, allocator: std.mem.Allocator, io: std.Io, module: *const mlir.Module, compile_options_pb: []const u8) CompileError!*pjrt.LoadedExecutable {
        const stablehlo_version = blk: {
            if (api.stablehloCurrentVersion()) |requested_version| {
                break :blk dialects.stablehlo.smallerVersion(requested_version, dialects.stablehlo.currentVersion());
            }
            break :blk dialects.stablehlo.minimumVersion();
        };

        const cloned_op = module.operation().clone();
        var cloned_module = mlir.Module.fromOperation(cloned_op);
        defer cloned_module.deinit();

        var serialized_buffer: std.Io.Writer.Allocating = try .initCapacity(allocator, 4096);
        defer serialized_buffer.deinit();

        dialects.stablehlo.serializePortableArtifact2(cloned_module, stablehlo_version, &serialized_buffer.writer) catch |err| {
            log.err("failed to serialize to portable artifact: {}", .{err});
            return switch (err) {
                std.Io.Writer.Error.WriteFailed => error.OutOfMemory,
                else => |e| e,
            };
        };

        return @ptrCast(try client.compile(api, io, .{
            .bytecode = serialized_buffer.written(),
            .bytecode_format = .mlir,
            .compile_options_pb = compile_options_pb,
        }));
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
    inline for (@typeInfo(DataType).@"enum".fields) |field| {
        const dt: DataType = @enumFromInt(field.value);
        try std.testing.expectEqual(dt, dtypeFromBufferType(bufferTypeFromDtype(dt)));
    }

    inline for (@typeInfo(pjrt.BufferType).@"enum".fields) |field| {
        const dt: pjrt.BufferType = @enumFromInt(field.value);
        if (dt == .invalid) continue;
        try std.testing.expectEqual(dt, bufferTypeFromDtype(dtypeFromBufferType(dt)));
    }
}
