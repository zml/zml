const asynk = @import("async");
const builtin = @import("builtin");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const pjrt = @import("pjrt");
const std = @import("std");
const stdx = @import("stdx");
const c = @import("c");

const dtype = @import("dtype.zig");
const meta = @import("meta.zig");

const Target = @import("platform.zig").Target;

const log = std.log.scoped(.zml);

pub const Profiler = pjrt.Profiler;
pub const ApiError = pjrt.ApiError;
pub const ErrorCode = pjrt.ErrorCode;
pub const BufferType = pjrt.BufferType;
pub const Device = pjrt.Device;
pub const DeviceDescription = pjrt.DeviceDescription;
pub const Api = pjrt.Api;
pub const NamedValue = pjrt.NamedValue;
pub const ClientInitError = pjrt.ClientInitError;
pub const CompileError = std.mem.Allocator.Error || error{InvalidMlirBytecodeVersion} || ApiError;
pub const Error = pjrt.Error;
pub const GetCostAnalysisError = pjrt.GetCostAnalysisError;
pub const SerializeResult = pjrt.SerializeResult;
pub const Executable = pjrt.Executable;
pub const ExecuteError = ApiError;
pub const Memory = pjrt.Memory;

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
    pub fn bufferFromHostBuffer(self: *const Client, api: *const Api, args: BufferFromHostBufferArgs) ApiError!*Buffer {
        const buffer, const event_ = try asynk.callBlocking(pjrt.Client.bufferFromHostBuffer, .{ self.inner(), api, args });
        if (event_) |event__| {
            const event: *Event = @ptrCast(event__);
            try event.await_(api);
        }
        return @ptrCast(buffer);
    }

    pub fn deserializeAndLoad(self: *const Client, api: *const Api, bytes: []const u8) ApiError!*LoadedExecutable {
        return @ptrCast(try asynk.callBlocking(pjrt.Client.deserializeAndLoad, .{ self.inner(), api, bytes }));
    }

    pub const CreateViewOfDeviceBufferArgs = pjrt.Client.CreateViewOfDeviceBufferArgs;
    pub fn createViewOfDeviceBuffer(self: *const Client, api: *const Api, args: CreateViewOfDeviceBufferArgs) ApiError!*Buffer {
        return @ptrCast(try self.inner().createViewOfDeviceBuffer(api, args));
    }

    fn compileSync(self: *const Client, api: *const Api, allocator: std.mem.Allocator, module: mlir.Module, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        var bytecode = std.ArrayList(u8).init(allocator);
        defer bytecode.deinit();
        module.op().writeBytecodeWithConfig(bytecode.writer(), .{ .desiredEmitedVersion = 1 }) catch |err| {
            log.err("failed to write module bytecode: {}", .{err});
            return err;
        };

        var serialized_buffer = std.ArrayList(u8).init(allocator);
        defer serialized_buffer.deinit();

        const stablehlo_version = blk: {
            if (api.stablehloCurrentVersion()) |requested_version| {
                break :blk dialects.stablehlo.stablehloGetSmallerVersion(requested_version, dialects.stablehlo.getCurrentVersion());
            }
            break :blk dialects.stablehlo.getMinimumVersion();
        };

        dialects.stablehlo.serializePortableArtifact(bytecode.items, stablehlo_version, serialized_buffer.writer()) catch |err| {
            log.err("failed to serialize to portable artifact: {}", .{err});
            return err;
        };

        return @ptrCast(try self.inner().compile(api, .{
            .bytecode = serialized_buffer.items,
            .bytecode_format = .mlir,
            .compile_options_pb = compile_options_pb,
        }));
    }

    pub fn compile(self: *const Client, api: *const Api, allocator: std.mem.Allocator, module: mlir.Module, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        return try asynk.callBlocking(compileSync, .{ self, api, allocator, module, compile_options_pb });
    }

    /// Returns the Profiler for this API.
    /// Not all platform have a profiling api, for those the profiler object will do nothing.
    /// Platforms with known profiler extensions: cuda, xpu
    pub fn getProfiler(self: *const Client, api: *const Api, options: pjrt.Profiler.Options) pjrt.Profiler {
        return self.inner().getProfiler(api, options);
    }

    pub fn addressableMemories(self: *const Client, api: *const Api) []*const Memory {
        return self.inner().addressableMemories(api);
    }

    pub fn memoryByKind(self: *const Client, api: *const Api, kind: Memory.Kind) ?*Memory {
        for (self.addressableMemories(api)) |mem| {
            if (mem.kind(api) == kind) {
                return mem;
            }
        }
        return null;
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

    pub fn toHostBuffer(self: *const Buffer, api: *const Api, dst: []u8) ApiError!?*Event {
        return @ptrCast(try self.inner().toHostBuffer(api, dst));
    }

    pub fn getElementType(self: *const Buffer, api: *const Api) BufferType {
        return self.inner().getElementType(api);
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

    pub fn copyToDevice(self: *const Buffer, api: *const Api, device: Device) ApiError!*Buffer {
        return @ptrCast(self.inner().copyToDevice(api, device));
    }

    pub fn copyToMemory(self: *const Buffer, api: *const Api, memory: *const Memory) ApiError!*Buffer {
        return @ptrCast(self.inner().copyToMemory(api, memory));
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

    pub fn await_(self: *Event, api: *const Api) ApiError!void {
        defer self.deinit(api);

        if (self.isReady(api)) {
            return;
        }

        var ctx = struct {
            err: ?*pjrt.Error = null,
            event: asynk.threading.ResetEventSingle = .{},
        }{};

        try self.inner().onReady(api, &(struct {
            fn call(err: ?*pjrt.Error, user_arg: ?*anyopaque) callconv(.C) void {
                const ctx_: *@TypeOf(ctx) = @ptrCast(@alignCast(user_arg.?));
                ctx_.err = err;
                ctx_.event.set();
            }
        }.call), &ctx);
        ctx.event.wait();

        if (ctx.err) |e| {
            defer e.deinit(api);
            return e.getCode(api).toApiError();
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

    pub fn getAddressableDevices(self: *const LoadedExecutable, api: *const Api) []*const Device {
        return self.inner().getAddressableDevices(api);
    }

    pub const ExecuteArgs = struct {
        arguments: []const [*]const *const Buffer,
        num_args: usize,
        results: []const [*]*Buffer,
        events: []?*Event,
        non_donatable_input_indices: []const i64 = &.{},
    };

    pub fn execute(self: *const LoadedExecutable, api: *const Api, args: ExecuteArgs) ExecuteError!void {
        try asynk.callBlocking(pjrt.LoadedExecutable.execute, .{ self.inner(), api, pjrt.LoadedExecutable.ExecuteArgs{
            .num_args = args.num_args,
            .arguments = @ptrCast(args.arguments),
            .results = @ptrCast(args.results),
            .events = @ptrCast(args.events),
            .non_donatable_input_indices = args.non_donatable_input_indices,
        } });
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

    pub fn device(self: *AsyncHostToDeviceTransferManager, api: *const Api) *Device {
        return @ptrCast(self.inner().device(api));
    }

    pub fn bufferCount(self: *AsyncHostToDeviceTransferManager, api: *const Api) usize {
        return self.inner().bufferCount(api);
    }

    pub fn bufferSize(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize) usize {
        return self.inner().bufferSize(api, buffer_index);
    }

    pub fn setBufferError(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize, error_code: ErrorCode, error_message: []const u8) void {
        self.inner().setBufferError(api, buffer_index, error_code, error_message);
    }

    pub fn addMetadata(self: *AsyncHostToDeviceTransferManager, api: *const Api, transfer_metadata: []const NamedValue) void {
        return self.inner().addMetadata(api, transfer_metadata);
    }
};
