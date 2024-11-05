const builtin = @import("builtin");
const std = @import("std");

const asynk = @import("async");
const mlir = @import("mlir");

const pjrt = @import("pjrt");
const dtype = @import("dtype.zig");
const meta = @import("meta.zig");
const dialects = @import("mlir/dialects");

pub const Profiler = pjrt.Profiler;
pub const ApiError = pjrt.ApiError;
pub const ErrorCode = pjrt.ErrorCode;

const Target = @import("platform.zig").Target;

const log = std.log.scoped(.zml);

pub const Buffer = pjrt.Buffer;
pub const BufferType = pjrt.BufferType;
pub const Device = pjrt.Device;
pub const DeviceDescription = pjrt.DeviceDescription;
pub const Api = pjrt.Api;
pub const NamedValue = pjrt.NamedValue;
pub const ClientInitError = pjrt.ClientInitError;
pub const CompileError = std.mem.Allocator.Error || ApiError;
pub const Error = pjrt.Error;
pub const GetCostAnalysisError = pjrt.GetCostAnalysisError;
pub const SerializeResult = pjrt.SerializeResult;
pub const Executable = pjrt.Executable;
pub const ExecuteError = ApiError;

test {
    std.testing.refAllDecls(Client);
    std.testing.refAllDecls(Event);
    std.testing.refAllDecls(LoadedExecutable);
}

fn InnerMixin(comptime innerT: type) type {
    return struct {
        inline fn inner(self: anytype) if (@typeInfo(@TypeOf(self)).Pointer.is_const) *const innerT else *innerT {
            return @ptrCast(self);
        }
    };
}

pub const Client = opaque {
    const inner = InnerMixin(pjrt.Client).inner;

    pub fn init(api: *const Api, create_options: []const NamedValue) ClientInitError!*Client {
        return @ptrCast(try pjrt.Client.init(api, create_options));
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
    pub fn bufferFromHostBuffer(self: *const Client, api: *const Api, args: BufferFromHostBufferArgs) !*Buffer {
        const buffer, const event_ = try self.inner().bufferFromHostBuffer(api, args);
        if (event_) |event__| {
            const event: *Event = @ptrCast(event__);
            try event.await_(api);
        }
        return buffer;
    }

    pub fn deserializeAndLoad(self: *const Client, api: *const Api, bytes: []const u8) ApiError!*LoadedExecutable {
        return @ptrCast(try asynk.callBlocking(pjrt.Client.deserializeAndLoad, .{ self.inner(), api, bytes }));
    }

    pub const CreateViewOfDeviceBufferArgs = pjrt.Client.CreateViewOfDeviceBufferArgs;
    pub fn createViewOfDeviceBuffer(self: *const Client, api: *const Api, args: CreateViewOfDeviceBufferArgs) ApiError!*Buffer {
        var args_ = args;
        args_.on_delete_callback = args_.on_delete_callback orelse &(struct {
            fn call(_: ?*anyopaque, _: ?*anyopaque) callconv(.C) void {}
        }.call);
        const buf = try self.inner().createViewOfDeviceBuffer(api, args_);
        return @ptrCast(buf);
    }

    fn compileSync(self: *const Client, api: *const Api, allocator: std.mem.Allocator, module: mlir.Module, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        var bytecode = std.ArrayList(u8).init(allocator);
        defer bytecode.deinit();
        module.op().writeBytecodeWithConfig(bytecode.writer(), .{ .desiredEmitedVersion = 1 }) catch {
            std.debug.print("failed to write module bytecode\n", .{});
            unreachable;
        };

        var serialized_buffer = std.ArrayList(u8).init(allocator);
        defer serialized_buffer.deinit();
        dialects.stablehlo.serializePortableArtifact(bytecode.items, dialects.stablehlo.getMinimumVersion(), serialized_buffer.writer()) catch {
            std.debug.print("failed to serialize to portable artifact\n", .{});
            unreachable;
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

    pub fn await_(self: *Event, api: *const Api) !void {
        defer self.deinit(api);

        var ctx = struct {
            err: ?*pjrt.Error = null,
            notif: asynk.Notification,
        }{
            .notif = try asynk.Notification.init(),
        };
        defer ctx.notif.deinit();

        try self.inner().onReady(api, &(struct {
            fn call(err: ?*pjrt.Error, user_arg: ?*anyopaque) callconv(.C) void {
                const ctx_: *@TypeOf(ctx) = @ptrCast(@alignCast(user_arg.?));
                ctx_.err = err;
                ctx_.notif.notify() catch @panic("Unable to notify");
            }
        }.call), &ctx);

        try ctx.notif.wait();
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

    pub fn execute(self: *const LoadedExecutable, api: *const Api, args: struct {
        arguments: []const [*]const *const Buffer,
        num_args: usize,
        results: []const [*]*Buffer,
        events: []?*Event,
        non_donatable_input_indices: []const i64 = &.{},
    }) ExecuteError!void {
        try asynk.callBlocking(pjrt.LoadedExecutable.execute, .{ self.inner(), api, .{
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
