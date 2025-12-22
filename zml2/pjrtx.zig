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

    fn compileSync(self: *const Client, api: *const Api, allocator: std.mem.Allocator, module: *const mlir.Module, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        var bytecode: std.Io.Writer.Allocating = try .initCapacity(allocator, 4096);
        defer bytecode.deinit();
        module.operation().writeBytecode(.{ .desired_emit_version = 1 }, &bytecode.writer) catch |err| {
            log.err("failed to write module bytecode: {}", .{err});
            return switch (err) {
                error.WriteFailed => error.OutOfMemory,
                else => |e| e,
            };
        };

        var serialized_buffer: std.Io.Writer.Allocating = try .initCapacity(allocator, 4096);
        defer serialized_buffer.deinit();

        const stablehlo_version = blk: {
            if (api.stablehloCurrentVersion()) |requested_version| {
                break :blk dialects.stablehlo.smallerVersion(requested_version, dialects.stablehlo.currentVersion());
            }
            break :blk dialects.stablehlo.minimumVersion();
        };

        dialects.stablehlo.serializePortableArtifact(bytecode.written(), stablehlo_version, &serialized_buffer.writer) catch |err| {
            log.err("failed to serialize to portable artifact: {}", .{err});
            return switch (err) {
                std.Io.Writer.Error.WriteFailed => error.OutOfMemory,
                else => |e| e,
            };
        };

        return @ptrCast(try self.inner().compile(api, .{
            .bytecode = serialized_buffer.written(),
            .bytecode_format = .mlir,
            .compile_options_pb = compile_options_pb,
        }));
    }

    pub fn compile(self: *const Client, api: *const Api, allocator: std.mem.Allocator, io: std.Io, module: *const mlir.Module, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        var future = io.async(compileSync, .{ self, api, allocator, module, compile_options_pb });
        return future.await(io);
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

pub const Buffer = opaque {
    pub const inner = InnerMixin(pjrt.Buffer).inner;

    pub fn deinit(self: *Buffer, api: *const Api) void {
        self.inner().deinit(api);
    }

    pub fn getDevice(self: *const Buffer, api: *const Api) ApiError!*Device {
        return try self.inner().getDevice(api);
    }

    pub fn delete(self: *Buffer, api: *const Api) void {
        self.inner().delete(api);
    }

    pub fn isDeleted(self: *const Buffer, api: *const Api) bool {
        return self.inner().isDeleted(api);
    }

    pub fn isOnCpu(self: *const Buffer, api: *const Api) bool {
        return self.inner().isOnCpu(api);
    }

    pub fn memory(self: *const Buffer, api: *const Api) *const Memory {
        return self.inner().memory(api);
    }

    pub fn toHostBuffer(self: *const Buffer, api: *const Api, dst: []u8) ApiError!?*Event {
        return @ptrCast(try self.inner().toHostBuffer(api, dst));
    }

    pub fn getElementType(self: *const Buffer, api: *const Api) zml.DataType {
        return dtypeFromBufferType(self.inner().getElementType(api));
    }

    pub fn getDimensions(self: *const Buffer, api: *const Api) []const i64 {
        return self.inner().getDimensions(api);
    }

    pub fn getUnpaddedDimensions(self: *const Buffer, api: *const Api) ApiError![]const i64 {
        return try self.inner().getUnpaddedDimensions(api);
    }

    pub fn getOnDeviceSizeInBytes(self: *const Buffer, api: *const Api) ApiError!usize {
        return try self.inner().getOnDeviceSizeInBytes(api);
    }

    pub fn copyToDevice(self: *const Buffer, api: *const Api, device: *Device) ApiError!*Buffer {
        return @ptrCast(try self.inner().copyToDevice(api, device));
    }

    pub fn copyToMemory(self: *const Buffer, api: *const Api, memory_: *const Memory) ApiError!*Buffer {
        return @ptrCast(try self.inner().copyToMemory(api, memory_));
    }

    pub fn getReadyEvent(self: *const Buffer, api: *const Api) ?*Event {
        return @ptrCast(self.inner().getReadyEvent(api));
    }

    pub fn getOpaqueDeviceMemoryDataPointer(self: *const Buffer, api: *const Api) ApiError!*anyopaque {
        return try self.inner().getOpaqueDeviceMemoryDataPointer(api);
    }
};

pub const Event = opaque {
    pub const inner = InnerMixin(pjrt.Event).inner;

    pub fn deinit(self: *Event, api: *const Api) void {
        self.inner().deinit(api);
    }

    pub fn isReady(self: *const Event, api: *const Api) bool {
        return self.inner().isReady(api);
    }

    pub fn getEventError(self: *const Event, api: *const Api) ?*Error {
        return self.inner().getEventError(api);
    }

    pub fn awaitBlocking(self: *Event, api: *const Api) ApiError!void {
        if (self.isReady(api)) {
            return;
        }
        try self.inner().await(api);
    }

    pub fn await(self: *Event, api: *const Api, io: std.Io) ApiError!void {
        defer self.deinit(api);

        if (self.isReady(api)) {
            return;
        }

        var mutex: std.Io.Mutex = .init;
        var condvar: std.Io.Condition = .{};
        var done: bool = false;

        var ctx = struct {
            err: ?*pjrt.Error = null,
            mutex: *std.Io.Mutex,
            condvar: *std.Io.Condition,
            done: *bool,
            io: std.Io,
        }{ .mutex = &mutex, .condvar = &condvar, .done = &done, .io = io };

        try self.inner().onReady(api, &(struct {
            fn call(err: ?*pjrt.Error, user_arg: ?*anyopaque) callconv(.c) void {
                const ctx_: *@TypeOf(ctx) = @ptrCast(@alignCast(user_arg.?));
                ctx_.err = err;
                {
                    ctx_.mutex.lock(ctx_.io) catch unreachable;
                    defer ctx_.mutex.unlock(ctx_.io);
                    ctx_.done.* = true;
                }

                ctx_.condvar.signal(ctx_.io);
            }
        }.call), &ctx);
        mutex.lock(io) catch unreachable;
        while (!done) {
            ctx.condvar.wait(io, &mutex) catch unreachable;
        }

        if (ctx.err) |e| {
            defer e.deinit(api);
            const err_code = e.getCode(api).toApiError();
            log.err("{t} {s}", .{ err_code, e.getMessage(api) });
            return err_code;
        }
    }
};

pub const LoadedExecutable = opaque {
    const inner = InnerMixin(pjrt.LoadedExecutable).inner;

    pub fn deinit(self: *LoadedExecutable, api: *const Api) void {
        self.inner().deinit(api);
    }

    pub fn delete(self: *LoadedExecutable, api: *const Api) void {
        self.inner().delete(api);
    }

    pub fn isDeleted(self: *const LoadedExecutable, api: *const Api) bool {
        return self.inner().isDeleted(api);
    }

    pub fn getAddressableDevices(self: *const LoadedExecutable, api: *const Api) []const *Device {
        return self.inner().getAddressableDevices(api);
    }

    pub const ExecuteArgs = struct {
        arguments: []const [*]const *const Buffer,
        num_args: usize,
        results: []const [*]*Buffer,
        events: ?[]?*Event,
        non_donatable_input_indices: []const i64 = &.{},
        context: ?*ExecuteContext,
    };

    pub fn execute(self: *const LoadedExecutable, api: *const Api, args: ExecuteArgs) ExecuteError!void {
        try self.inner().execute(api, pjrt.LoadedExecutable.ExecuteArgs{
            .num_args = args.num_args,
            .arguments = @ptrCast(args.arguments),
            .results = @ptrCast(args.results),
            .events = @ptrCast(args.events),
            .non_donatable_input_indices = args.non_donatable_input_indices,
            .context = args.context,
        });
    }

    pub fn getExecutable(self: *LoadedExecutable, api: *const Api) ApiError!*Executable {
        return try self.inner().getExecutable(api);
    }
};

pub const AsyncHostToDeviceTransferManager = opaque {
    const inner = InnerMixin(pjrt.AsyncHostToDeviceTransferManager).inner;

    pub fn deinit(self: *AsyncHostToDeviceTransferManager, api: *const Api) void {
        self.inner().deinit(api);
    }

    pub fn transferData(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize, data: []const u8, offset: i64, is_last_transfer: bool) ApiError!*Event {
        return @ptrCast(try self.inner().transferData(api, buffer_index, data, offset, is_last_transfer));
    }

    pub fn retrieveBuffer(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize) ApiError!*Buffer {
        return @ptrCast(try self.inner().retrieveBuffer(api, buffer_index));
    }

    pub fn device(self: *AsyncHostToDeviceTransferManager, api: *const Api) ApiError!*Device {
        return @ptrCast(try self.inner().device(api));
    }

    pub fn bufferCount(self: *AsyncHostToDeviceTransferManager, api: *const Api) ApiError!usize {
        return self.inner().bufferCount(api);
    }

    pub fn bufferSize(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize) ApiError!usize {
        return self.inner().bufferSize(api, buffer_index);
    }

    pub fn setBufferError(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize, error_code: ErrorCode, error_message: []const u8) ApiError!void {
        return self.inner().setBufferError(api, buffer_index, @intFromEnum(error_code), error_message);
    }

    pub fn addMetadata(self: *AsyncHostToDeviceTransferManager, api: *const Api, transfer_metadata: []const NamedValue) ApiError!void {
        return self.inner().addMetadata(api, transfer_metadata);
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
