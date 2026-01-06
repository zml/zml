const std = @import("std");

const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const pjrt = @import("pjrt");
pub const ffi = pjrt.ffi;
pub const Api = pjrt.Api;
pub const ApiError = pjrt.ApiError;
pub const BufferType = pjrt.BufferType;
pub const ClientInitError = pjrt.ClientInitError;
pub const CompiledMemoryStats = pjrt.CompiledMemoryStats;
pub const Device = pjrt.Device;
pub const DeviceDescription = pjrt.DeviceDescription;
pub const Error = pjrt.Error;
pub const ErrorCode = pjrt.ErrorCode;
pub const Executable = pjrt.Executable;
pub const ExecuteContext = pjrt.ExecuteContext;
pub const ExecuteError = ApiError;
pub const GetCostAnalysisError = pjrt.GetCostAnalysisError;
pub const Memory = pjrt.Memory;
pub const MemoryStats = pjrt.MemoryStats;
pub const NamedValue = pjrt.NamedValue;
pub const SerializeResult = pjrt.SerializeResult;
pub const Stream = pjrt.Stream;
pub const ShapeSpec = pjrt.ShapeSpec;
pub const Buffer = pjrt.Buffer;
pub const Event = pjrt.Event;
pub const LoadedExecutable = pjrt.LoadedExecutable;
pub const AsyncHostToDeviceTransferManager = pjrt.AsyncHostToDeviceTransferManager;

const zml = struct {
    pub const DataType = @import("dtype.zig").DataType;
};

const log = std.log.scoped(.zml);

pub const CompileError = std.mem.Allocator.Error || error{InvalidMlirBytecodeVersion} || ApiError;

fn InnerMixin(comptime innerT: type) type {
    return struct {
        inline fn inner(self: anytype) if (@typeInfo(@TypeOf(self)).pointer.is_const) *const innerT else *innerT {
            return @ptrCast(self);
        }
    };
}

pub const Client = opaque {
    const inner = InnerMixin(pjrt.Client).inner;

    pub fn init(api: *const Api, options: []const NamedValue) ClientInitError!*Client {
        return @ptrCast(try pjrt.Client.init(api, options));
    }

    pub fn deinit(self: *Client, api: *const Api) void {
        self.inner().deinit(api);
    }

    pub fn getPlatformName(self: *const Client, api: *const Api) []const u8 {
        return self.inner().getPlatformName(api);
    }

    pub fn getDevices(self: *const Client, api: *const Api) []const *const Device {
        return self.inner().getDevices(api);
    }

    pub fn getAddressableDevices(self: *const Client, api: *const Api) []const *const Device {
        return self.inner().getAddressableDevices(api);
    }

    pub const BufferFromHostBufferArgs = pjrt.Client.BufferFromHostBufferArgs;
    pub fn bufferFromHostBuffer(self: *const Client, api: *const Api, args: BufferFromHostBufferArgs) ApiError!struct { *Buffer, ?*Event } {
        const buffer, const event_ = try self.inner().bufferFromHostBuffer(api, args);
        return .{ @ptrCast(buffer), @ptrCast(event_) };
    }

    pub fn deserializeAndLoad(self: *const Client, api: *const Api, io: std.Io, bytes: []const u8) ApiError!*LoadedExecutable {
        var future = io.async(pjrt.Client.deserializeAndLoad, .{ self.inner(), api, bytes });
        return @ptrCast(try future.await(io));
    }

    pub const CreateViewOfDeviceBufferArgs = pjrt.Client.CreateViewOfDeviceBufferArgs;
    pub fn createViewOfDeviceBuffer(self: *const Client, api: *const Api, args: CreateViewOfDeviceBufferArgs) ApiError!*Buffer {
        return @ptrCast(try self.inner().createViewOfDeviceBuffer(api, args));
    }

    pub fn compile(self: *const Client, api: *const Api, allocator: std.mem.Allocator, io: std.Io, module: *const mlir.Module, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        var bytecode: std.Io.Writer.Allocating = try .initCapacity(allocator, 4096);
        defer bytecode.deinit();
        module.operation().writeBytecode(.{ .desired_emit_version = 1 }, &bytecode.writer) catch |err| {
            log.err("failed to write module bytecode: {}", .{err});
            return switch (err) {
                error.WriteFailed => error.OutOfMemory,
                else => |e| e,
            };
        };

        const stablehlo_version = blk: {
            if (api.stablehloCurrentVersion()) |requested_version| {
                break :blk dialects.stablehlo.smallerVersion(requested_version, dialects.stablehlo.currentVersion());
            }
            break :blk dialects.stablehlo.minimumVersion();
        };

        var serialized_buffer: std.Io.Writer.Allocating = try .initCapacity(allocator, 4096);
        defer serialized_buffer.deinit();
        dialects.stablehlo.serializePortableArtifact(bytecode.written(), stablehlo_version, &serialized_buffer.writer) catch |err| {
            log.err("failed to serialize to portable artifact: {}", .{err});
            return switch (err) {
                std.Io.Writer.Error.WriteFailed => error.OutOfMemory,
                else => |e| e,
            };
        };

        return @ptrCast(try self.inner().compile(api, io, .{
            .bytecode = serialized_buffer.written(),
            .bytecode_format = .mlir,
            .compile_options_pb = compile_options_pb,
        }));
    }

    pub fn addressableMemories(self: *const Client, api: *const Api) []*const Memory {
        return self.inner().addressableMemories(api);
    }

    pub fn memoryByKind(self: *const Client, api: *const Api, kind: Memory.Kind) ?*const Memory {
        for (self.addressableMemories(api)) |mem| {
            if (mem.kind(api) == kind) {
                return mem;
            }
        }
        return null;
    }

    pub const CreateUninitializedBufferArgs = pjrt.Client.CreateUninitializedBufferArgs;

    pub fn createUnitializedBuffer(self: *const Client, api: *const Api, args: CreateUninitializedBufferArgs) ApiError!*Buffer {
        return @ptrCast(try self.inner().createUninitializedBuffer(api, args));
    }

    pub const CreateBuffersForAsyncHostToDeviceArgs = pjrt.Client.CreateBuffersForAsyncHostToDeviceArgs;

    pub fn createBuffersForAsyncHostToDevice(self: *const Client, api: *const Api, args: CreateBuffersForAsyncHostToDeviceArgs) ApiError!*AsyncHostToDeviceTransferManager {
        return @ptrCast(try self.inner().createBuffersForAsyncHostToDevice(api, args));
    }
};

pub fn bufferTypeFromDtype(dt: zml.DataType) pjrt.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrt.BufferType, @tagName(tag)),
    };
}

pub fn dtypeFromBufferType(pjrt_type: pjrt.BufferType) zml.DataType {
    return switch (pjrt_type) {
        .invalid => @panic("Found an invalid pjrt buffer"),
        inline else => |tag| @field(zml.DataType, @tagName(tag)),
    };
}

test bufferTypeFromDtype {
    inline for (@typeInfo(zml.DataType).@"enum".fields) |field| {
        const dt: zml.DataType = @enumFromInt(field.value);
        try std.testing.expectEqual(dt, dtypeFromBufferType(bufferTypeFromDtype(dt)));
    }

    inline for (@typeInfo(pjrt.BufferType).@"enum".fields) |field| {
        const dt: pjrt.BufferType = @enumFromInt(field.value);
        if (dt == .invalid) continue;
        try std.testing.expectEqual(dt, bufferTypeFromDtype(dtypeFromBufferType(dt)));
    }
}
