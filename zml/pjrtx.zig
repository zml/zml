const builtin = @import("builtin");
const std = @import("std");

const mlir = @import("mlir");
const pjrt = @import("pjrt");
const dtype = @import("dtype.zig");
const meta = @import("meta.zig");
const asynk = @import("async");

pub const Profiler = pjrt.Profiler;
pub const ApiError = pjrt.ApiError;
pub const ErrorCode = pjrt.ErrorCode;

const Target = @import("platform.zig").Target;

const log = std.log.scoped(.zml);

pub const Buffer = pjrt.Buffer;
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
        const event: *Event = @ptrCast(event_);
        try event.await_(api);
        return buffer;
    }

    pub fn deserializeAndLoad(self: *const Client, api: *const Api, bytes: []const u8) ApiError!*LoadedExecutable {
        return @ptrCast(try asynk.call(pjrt.Client.deserializeAndLoad, .{ self.inner(), api, bytes }));
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

    fn downgradeStableHLO(self: Client, operation: mlir.Operation) mlir.Operation {
        var cloned = operation.clone() catch unreachable;
        cloned.walk(.pre_order, .{ .api_version = self.getApiVersion() }, struct {
            const OpsStaticMap = std.StaticStringMap([]const [:0]const u8);
            const convertPre40Ops = OpsStaticMap.initComptime(.{
                .{ "stablehlo.broadcast", &.{"broadcast_sizes"} },
                .{ "stablehlo.dynamic_slice", &.{"slice_sizes"} },
                .{ "stablehlo.fft", &.{"fft_length"} },
                .{ "stablehlo.pad", &.{ "edge_padding_low", "edge_padding_high", "interior_padding" } },
                .{ "stablehlo.reverse", &.{"dimensions"} },
                .{ "stablehlo.slice", &.{ "start_indices", "limit_indices", "strides" } },
                .{ "stablehlo.transpose", &.{"permutation"} },
            });
            const convertOps = OpsStaticMap.initComptime(.{
                .{ "stablehlo.broadcast_in_dim", &.{"broadcast_dimensions"} },
                .{ "stablehlo.convolution", &.{ "window_strides", "rhs_dilation", "lhs_dilation", "window_reversal" } },
                .{ "stablehlo.dynamic_broadcast_in_dim", &.{ "broadcast_dimensions", "known_expanding_dimensions", "known_nonexpanding_dimensions" } },
                .{ "stablehlo.dynamic_convolution", &.{ "window_strides", "rhs_dilation", "lhs_dilation", "window_reversal" } },
                .{ "stablehlo.gather", &.{"slice_sizes"} },
                .{ "stablehlo.map", &.{"dimensions"} },
                .{ "stablehlo.reduce", &.{"dimensions"} },
                .{ "stablehlo.reduce_window", &.{ "window_dimensions", "window_strides", "base_dilations", "window_dilations" } },
                .{ "stablehlo.select_and_scatter", &.{ "window_dimensions", "window_strides" } },
            });

            fn convert(map: OpsStaticMap, op: mlir.Operation) void {
                if (map.get(op.name().str())) |attrs| {
                    for (attrs) |attr_name| {
                        if (op.getAttributeByName(attr_name)) |attr| {
                            if (attr.as(mlir.DenseArrayAttribute(.bool))) |attr_| {
                                op.setAttributeByName(attr_name, attr_.toElements().as(mlir.Attribute).?);
                            } else if (attr.as(mlir.DenseArrayAttribute(.i64))) |attr_| {
                                op.setAttributeByName(attr_name, attr_.toElements().as(mlir.Attribute).?);
                            }
                        }
                    }
                }
            }

            fn walk(wctx: anytype, op: mlir.Operation) mlir.Operation.WalkResult {
                // Keep in sync with https://github.com/openxla/xla/blob/a05ff095226aa2301903c2b475017b248d2c5ef3/xla/pjrt/mlir_to_hlo.cc#L101
                if (wctx.api_version.minor < 40) {
                    convert(convertPre40Ops, op);
                }
                convert(convertOps, op);

                return .advance;
            }
        }.walk);
        return cloned;
    }

    fn compileSync(self: *const Client, api: *const Api, allocator: std.mem.Allocator, module: mlir.Module, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();
        // Note: we may need to restore IR downgrade if we need to support old pjrt plugins.
        module.op().writeBytecode(buffer.writer());

        return @ptrCast(try self.inner().compile(api, .{
            .bytecode = buffer.items,
            .bytecode_format = .mlir,
            .compile_options_pb = compile_options_pb,
        }));
    }

    fn compileSync2(self: *const Client, api: *const Api, module: []const u8, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        return @ptrCast(try self.inner().compile(api, .{
            .bytecode = module,
            .bytecode_format = .mlir,
            .compile_options_pb = compile_options_pb,
        }));
    }

    pub fn compile(self: *const Client, api: *const Api, allocator: std.mem.Allocator, module: mlir.Module, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        return try asynk.call(compileSync, .{ self, api, allocator, module, compile_options_pb });
    }

    pub fn compile2(self: *const Client, api: *const Api, module: []const u8, compile_options_pb: []const u8) CompileError!*LoadedExecutable {
        return try asynk.call(compileSync2, .{ self, api, module, compile_options_pb });
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
        try self.inner().await_(api);

        var ctx = struct {
            err: ?*pjrt.Error = null,
            notif: asynk.Notification,
            ready: bool = false,
        }{
            .notif = try asynk.Notification.init(),
        };
        defer ctx.notif.deinit();

        try self.inner().onReady(api, &(struct {
            fn call(err: ?*pjrt.Error, user_arg: ?*anyopaque) callconv(.C) void {
                const ctx_: *@TypeOf(ctx) = @ptrCast(@alignCast(user_arg.?));
                ctx_.err = err;
                @atomicStore(bool, &ctx_.ready, true, .seq_cst);
                ctx_.notif.notify() catch @panic("Unable to notify");
            }
        }.call), &ctx);

        while (!ctx.ready) {
            try ctx.notif.wait();
        }
        if (ctx.err) |e| {
            defer e.deinit(api);
            return e.getCode(api).toApiError();
        }
    }
};

pub const LoadedExecutable = opaque {
    const inner = InnerMixin(pjrt.LoadedExecutable).inner;

    // pub fn deinit(self: *LoadedExecutable, api: *const Api) void {
    //     self.inner().deinit(api);
    // }

    pub fn delete(self: *LoadedExecutable, api: *const Api) void {
        self.inner().delete(api);
    }

    pub fn isDeleted(self: *const LoadedExecutable, api: *const Api) bool {
        return self.inner().isDeleted(api);
    }

    // TODO fix me
    // pub fn getAddressableDevices(self: *const LoadedExecutable, api: *const Api) []*const Device {
    //     return self.inner().getAddressableDevices(api);
    // }

    pub fn execute(self: *const LoadedExecutable, api: *const Api, args: struct {
        arguments: []const [*]const *const Buffer,
        num_args: usize,
        results: []const [*]*Buffer,
        events: []*Event,
        non_donatable_input_indices: []const i64 = &.{},
    }) ExecuteError!void {
        try self.inner().execute(api, .{
            .num_args = args.num_args,
            .arguments = @ptrCast(args.arguments),
            .results = @ptrCast(args.results),
            .events = @ptrCast(args.events),
            .non_donatable_input_indices = args.non_donatable_input_indices,
        });

        for (args.events) |event| {
            // TODO(Corentin): Maybe better handle the error here.
            event.await_(api) catch return error.Unknown;
        }
    }

    pub fn getExecutable(self: *LoadedExecutable, api: *const Api) ApiError!*Executable {
        return try self.inner().getExecutable(api);
    }
};
