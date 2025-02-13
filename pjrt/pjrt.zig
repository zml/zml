const builtin = @import("builtin");
const std = @import("std");
const stdx = @import("stdx");

const c = @import("c");

const log = std.log.scoped(.pjrt);

pub const Profiler = @import("profiler.zig").Profiler;

test {
    std.testing.refAllDecls(@This());
}

// We could calculate it like PJRT does, but it turns out that some of those
// were wrong in PJRT itself [1], which gets propagated to binary plugins. In
// order to mirror that, we just the value as computed by PJRT itself, through
// comptime reflection. We could make the argument to remove that one day since
// [1] has been fixed. The problem is that this problem could happen again in
// as the way PJRT does it is not very robust.
//
// 1. https://github.com/openxla/xla/issues/10032
fn pjrtStructSize(comptime T: type) usize {
    // unsafe on purpose, we want this to fail if that ever changes
    const typedef_name = comptime blk: {
        const needle = ".struct_";
        const idx = std.mem.indexOf(u8, @typeName(T), needle).?;
        break :blk @typeName(T)[idx + needle.len ..];
    };
    return @field(c, typedef_name ++ "_STRUCT_SIZE");
}

inline fn pjrtStruct(v: anytype) @TypeOf(v) {
    var ret = v;
    ret.struct_size = pjrtStructSize(@TypeOf(v));
    return ret;
}

pub const ApiError = error{
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
};

fn InnerMixin(comptime innerT: type) type {
    return struct {
        fn inner(self: anytype) *innerT {
            return @ptrCast(@constCast(@alignCast(self)));
        }
    };
}

pub const Api = struct {
    pub const Version = struct {
        major: i64,
        minor: i64,
    };

    const Funcs = std.meta.FieldEnum(c.PJRT_Api);

    inner: c.PJRT_Api,

    pub fn loadFrom(library: []const u8) !*const Api {
        var lib: std.DynLib = switch (builtin.os.tag) {
            .linux => blk: {
                const library_c = try std.posix.toPosixPath(library);
                break :blk .{
                    .inner = .{
                        .handle = c.dlopen(&library_c, c.RTLD_LAZY | c.RTLD_LOCAL | c.RTLD_NODELETE) orelse {
                            log.err("Unable to dlopen plugin: {s}", .{library});
                            return error.FileNotFound;
                        },
                    },
                };
            },
            else => try std.DynLib.open(library),
        };
        const DynGetPjrtApi = lib.lookup(*const fn () callconv(.C) *const Api, "GetPjrtApi") orelse {
            std.debug.panic("Unable to find GetPjrtApi symbol in library: {s}", .{library});
        };

        const api = DynGetPjrtApi();
        log.info("Loaded library: {s}", .{library});
        _ = api.call(.PJRT_Plugin_Initialize, .{}) catch unreachable;

        return api;
    }

    fn CallFnArgType(comptime func: Funcs) type {
        const fti = @typeInfo(std.meta.FieldType(c.PJRT_Api, func));
        const fn_ptr = @typeInfo(fti.optional.child);
        const fn_type_info = @typeInfo(fn_ptr.pointer.child);
        const arg_array_type_info = @typeInfo(fn_type_info.@"fn".params[0].type.?);
        return arg_array_type_info.pointer.child;
    }

    inline fn call(self: *const Api, comptime method: Funcs, arg: CallFnArgType(method)) ApiError!@TypeOf(arg) {
        var ret = pjrtStruct(arg);
        const fn_ptr = @field(&self.inner, @tagName(method)).?;
        const result = fn_ptr(&ret);
        if (@TypeOf(result) == void) {
            return ret;
        }

        if (result) |pjrt_c_error| {
            const pjrt_error: *Error = @ptrCast(pjrt_c_error);
            log.err("[{s}] {s}", .{ @tagName(method), pjrt_error.getMessage(self) });
            return pjrt_error.getCode(self).toApiError();
        }

        return ret;
    }

    pub fn lookupExtension(self: *const Api, comptime ExtensionT: type, ext_id: c_int) ?*const ExtensionT {
        var cur: [*c]const c.PJRT_Extension_Base = @alignCast(@ptrCast(self.inner.extension_start));
        while (cur != null) : (cur = cur.*.next) {
            if (cur.*.type == ext_id) {
                return @alignCast(@ptrCast(cur));
            }
        }

        return null;
    }

    pub inline fn version(self: *const Api) Version {
        return .{
            .major = @intCast(self.inner.pjrt_api_version.major_version),
            .minor = @intCast(self.inner.pjrt_api_version.minor_version),
        };
    }

    pub fn stablehloCurrentVersion(self: *const Api) ?[]const u8 {
        const state = struct {
            var buf: [32]u8 = undefined;
            var str: ?[:0]const u8 = null;
        };
        if (state.str) |str| {
            return str;
        }
        if (self.getPluginAttribute("stablehlo_current_version")) |v| {
            stdx.debug.assert(v.kind() == .int64list, "fetched attribute \"stablehlo_current_version\" from the plugin with type `{}`, expected `.int64list`", .{v.kind()});
            stdx.debug.assert(v.inner.value_size == 3, "expect version format to have 3 elements representing `major.minor.patch` format, got {} elements", .{v.inner.value_size});
            const value = v.inner.unnamed_0.int64_array_value[0..v.inner.value_size];
            state.str = std.fmt.bufPrintZ(&state.buf, "{d}.{d}.{d}", .{ value[0], value[1], value[2] }) catch unreachable;
        }
        return state.str;
    }

    pub fn customCallRegistry(api: *const Api) ?CustomCallRegistry {
        if (api.lookupExtension(c.PJRT_Gpu_Custom_Call, c.PJRT_Extension_Type_Gpu_Custom_Call)) |ext| {
            return .{ .inner = ext.custom_call.? };
        }
        // log.warn("No Custom Call registry found for platform: {}", .{self});
        return null;
    }

    fn getPluginAttribute(api: *const Api, key: []const u8) ?NamedValue {
        const attributes = api.getPluginAttributes();
        for (attributes) |attr| {
            if (std.mem.eql(u8, attr.name(), key)) {
                return attr;
            }
        }

        return null;
    }

    fn getPluginAttributes(api: *const Api) []const NamedValue {
        const ret = api.call(.PJRT_Plugin_Attributes, .{
            .extension_start = null,
        }) catch unreachable;

        if (ret.attributes == null) return &.{};

        return @ptrCast(ret.attributes[0..ret.num_attributes]);
    }
};

pub const ErrorCode = enum(c.PJRT_Error_Code) {
    cancelled = c.PJRT_Error_Code_CANCELLED,
    unknown = c.PJRT_Error_Code_UNKNOWN,
    invalid_argument = c.PJRT_Error_Code_INVALID_ARGUMENT,
    deadline_exceeded = c.PJRT_Error_Code_DEADLINE_EXCEEDED,
    not_found = c.PJRT_Error_Code_NOT_FOUND,
    already_exists = c.PJRT_Error_Code_ALREADY_EXISTS,
    permission_denied = c.PJRT_Error_Code_PERMISSION_DENIED,
    resource_exhausted = c.PJRT_Error_Code_RESOURCE_EXHAUSTED,
    failed_precondition = c.PJRT_Error_Code_FAILED_PRECONDITION,
    aborted = c.PJRT_Error_Code_ABORTED,
    out_of_range = c.PJRT_Error_Code_OUT_OF_RANGE,
    unimplemented = c.PJRT_Error_Code_UNIMPLEMENTED,
    internal = c.PJRT_Error_Code_INTERNAL,
    unavailable = c.PJRT_Error_Code_UNAVAILABLE,
    data_loss = c.PJRT_Error_Code_DATA_LOSS,
    unauthenticated = c.PJRT_Error_Code_UNAUTHENTICATED,

    pub fn toApiError(code: ErrorCode) ApiError {
        return switch (code) {
            .cancelled => ApiError.Cancelled,
            .unknown => ApiError.Unknown,
            .invalid_argument => ApiError.InvalidArgument,
            .deadline_exceeded => ApiError.DeadlineExceeded,
            .not_found => ApiError.NotFound,
            .already_exists => ApiError.AlreadyExists,
            .permission_denied => ApiError.PermissionDenied,
            .resource_exhausted => ApiError.ResourceExhausted,
            .failed_precondition => ApiError.FailedPrecondition,
            .aborted => ApiError.Aborted,
            .out_of_range => ApiError.OutOfRange,
            .unimplemented => ApiError.Unimplemented,
            .internal => ApiError.Internal,
            .unavailable => ApiError.Unavailable,
            .data_loss => ApiError.DataLoss,
            .unauthenticated => ApiError.Unauthenticated,
        };
    }
};

pub const Error = opaque {
    pub fn deinit(self: *Error, api: *const Api) void {
        _ = api.call(.PJRT_Error_Destroy, .{
            .@"error" = @ptrCast(self),
        }) catch unreachable;
    }

    pub fn getCode(self: *Error, api: *const Api) ErrorCode {
        const ret = api.call(.PJRT_Error_GetCode, .{
            .@"error" = @ptrCast(self),
        }) catch unreachable;
        return @enumFromInt(ret.code);
    }

    pub fn getMessage(self: *Error, api: *const Api) []const u8 {
        const ret = api.call(.PJRT_Error_Message, .{
            .@"error" = @ptrCast(self),
        }) catch unreachable;
        return ret.message[0..ret.message_size];
    }
};

pub const ClientInitError = error{LoadingFailed} || ApiError;

pub const ShapeSpec = extern struct {
    comptime {
        std.debug.assert(@sizeOf(ShapeSpec) == @sizeOf(c.PJRT_ShapeSpec));
    }

    inner: c.PJRT_ShapeSpec,

    pub fn init(dims_: []const usize, bt: BufferType) ShapeSpec {
        return .{
            .inner = pjrtStruct(c.PJRT_ShapeSpec{
                .dims = @ptrCast(@constCast(dims_.ptr)),
                .num_dims = dims.len,
                .buffer_type = @intFromEnum(bt),
            }),
        };
    }

    pub fn dims(self: ShapeSpec) []usize {
        return self.inner.dims[0..self.inner.num_dims];
    }

    pub fn bufferType(self: ShapeSpec) BufferType {
        return @enumFromInt(self.inner.buffer_type);
    }
};

pub const Client = opaque {
    const inner = InnerMixin(c.PJRT_Client).inner;

    pub const ProgramFormat = enum {
        hlo,
        mlir,
    };

    pub fn init(api: *const Api, create_options: []const NamedValue) ClientInitError!*Client {
        // log.info("Loaded PJRT runtime plugin: {s}", .{api.Platform});
        const ret = try api.call(.PJRT_Client_Create, .{
            .create_options = @ptrCast(create_options.ptr),
            .num_options = create_options.len,
            .kv_get_callback = null,
            .kv_put_callback = null,
            .kv_put_user_arg = null,
            .kv_get_user_arg = null,
        });
        return @ptrCast(ret.client.?);
    }

    pub fn deinit(self: *Client, api: *const Api) void {
        _ = api.call(.PJRT_Client_Destroy, .{
            .client = self.inner(),
        }) catch {};
    }

    pub fn getPlatformName(self: *const Client, api: *const Api) []const u8 {
        const ret = api.call(.PJRT_Client_PlatformName, .{
            .client = self.inner(),
        }) catch unreachable;
        return ret.platform_name[0..ret.platform_name_size];
    }

    pub fn getDevices(self: *const Client, api: *const Api) []const *Device {
        const ret = api.call(.PJRT_Client_Devices, .{
            .client = self.inner(),
        }) catch unreachable;
        return @ptrCast(ret.devices[0..ret.num_devices]);
    }

    pub fn getAddressableDevices(self: *const Client, api: *const Api) []const *Device {
        const ret = api.call(.PJRT_Client_AddressableDevices, .{
            .client = self.inner(),
        }) catch unreachable;
        return @ptrCast(ret.addressable_devices[0..ret.num_addressable_devices]);
    }

    pub const CompileArgs = struct {
        bytecode: []const u8,
        bytecode_format: ProgramFormat,
        compile_options_pb: []const u8,
    };

    pub fn compile(self: *const Client, api: *const Api, args: CompileArgs) ApiError!*LoadedExecutable {
        const bytecode_format_ = @tagName(args.bytecode_format);
        const ret = try api.call(.PJRT_Client_Compile, .{
            .program = &pjrtStruct(c.PJRT_Program{
                .code = @ptrCast(@constCast(args.bytecode.ptr)),
                .code_size = args.bytecode.len,
                .format = @ptrCast(@constCast(bytecode_format_.ptr)),
                .format_size = bytecode_format_.len,
            }),
            .compile_options = @ptrCast(@constCast(args.compile_options_pb.ptr)),
            .compile_options_size = args.compile_options_pb.len,
            .client = self.inner(),
        });
        return @ptrCast(ret.executable.?);
    }

    pub const BufferFromHostBufferArgs = struct {
        data: []const u8,
        buffer_type: BufferType,
        dims: []const i64,
        byte_strides: ?[]const i64,
        device: ?*const Device = null,
        host_buffer_semantics: HostBufferSemantics,
        memory: ?*const Memory = null,
    };

    pub fn bufferFromHostBuffer(self: *const Client, api: *const Api, args: BufferFromHostBufferArgs) ApiError!struct { *Buffer, ?*Event } {
        const ret = try api.call(.PJRT_Client_BufferFromHostBuffer, .{
            .client = self.inner(),
            .data = @ptrCast(@constCast(args.data.ptr)),
            .type = @intFromEnum(args.buffer_type),
            .dims = @ptrCast(@constCast(args.dims.ptr)),
            .num_dims = args.dims.len,
            .byte_strides = if (args.byte_strides) |bs| @ptrCast(@constCast(bs.ptr)) else null,
            .num_byte_strides = if (args.byte_strides) |bs| bs.len else 0,
            .host_buffer_semantics = @intFromEnum(args.host_buffer_semantics),
            .device = @ptrCast(@constCast(args.device)),
            .memory = @ptrCast(@constCast(args.memory)),
            .device_layout = null, // TODO
            .done_with_host_buffer = null,
            .buffer = null,
        });

        return .{
            @ptrCast(ret.buffer.?),
            @ptrCast(ret.done_with_host_buffer),
        };
    }

    /// Returns the Profiler for this API.
    /// Not all platform have a profiling api, for those the profiler object will do nothing.
    /// Platforms with known profiler extensions: cuda, xpu
    pub fn getProfiler(self: *const Client, api: *const Api, options: Profiler.Options) Profiler {
        if (api.version().minor >= 45) {
            if (api.lookupExtension(c.PJRT_Profiler_Extension, c.PJRT_Extension_Type_Profiler)) |ext| {
                return Profiler.init(ext.profiler_api.*, options);
            }
        }
        log.warn("No profiler found for platform: {}", .{self});
        return Profiler.init(null, null);
    }

    pub fn deserializeAndLoad(self: *const Client, api: *const Api, bytes: []const u8) ApiError!*LoadedExecutable {
        const ret = try api.call(.PJRT_Executable_DeserializeAndLoad, .{
            .client = self.inner(),
            .serialized_executable = bytes.ptr,
            .serialized_executable_size = bytes.len,
        });
        return @ptrCast(ret.loaded_executable.?);
    }

    pub const CreateViewOfDeviceBufferArgs = struct {
        data: []const u8,
        dims: []const i64,
        element_type: BufferType,
        layout: MemoryLayout,
        device: *const Device,
        on_delete_callback: *const fn (device_buffer_ptr: ?*anyopaque, ctx: ?*anyopaque) callconv(.C) void = &struct {
            fn call(_: ?*anyopaque, _: ?*anyopaque) callconv(.C) void {}
        }.call,
        on_delete_callback_arg: ?*anyopaque = null,
        stream: ?isize = null,
    };

    pub fn createViewOfDeviceBuffer(self: *const Client, api: *const Api, args: CreateViewOfDeviceBufferArgs) ApiError!*Buffer {
        const layout = args.layout.toCStruct();
        const ret = try api.call(.PJRT_Client_CreateViewOfDeviceBuffer, .{
            .client = self.inner(),
            .device_buffer_ptr = @ptrCast(@constCast(args.data.ptr)),
            .dims = args.dims.ptr,
            .num_dims = args.dims.len,
            .element_type = @intFromEnum(args.element_type),
            .layout = @ptrCast(@constCast(&layout)),
            .device = @ptrCast(@constCast(args.device)),
            .on_delete_callback = args.on_delete_callback,
            .on_delete_callback_arg = args.on_delete_callback_arg,
            .stream = if (args.stream) |stream| stream else 0,
        });
        return @ptrCast(ret.buffer.?);
    }

    pub fn addressableMemories(self: *const Client, api: *const Api) []*const Memory {
        const ret = api.call(.PJRT_Client_AddressableMemories, .{
            .client = self.inner(),
        }) catch unreachable;
        if (ret.addressable_memories) |memories| {
            return @constCast(@ptrCast(memories[0..ret.num_addressable_memories]));
        }
        return &.{};
    }

    pub fn dmaMap(self: *const Client, api: *const Api, data: []const u8) ApiError!*Buffer {
        const ret = try api.call(.PJRT_Client_DMA_Map, .{
            .client = self.inner(),
            .data = @ptrCast(@constCast(data.ptr)),
            .size = @intCast(data.len),
        });
        return @ptrCast(ret.buffer.?);
    }

    pub fn dmaUnmap(self: *const Client, api: *const Api, data: []const u8) void {
        _ = api.call(.PJRT_Client_DMA_Unmap, .{
            .client = self.inner(),
            .data = @ptrCast(@constCast(data.ptr)),
        }) catch unreachable;
    }

    pub const CreateBuffersForAsyncHostToDeviceArgs = struct {
        shape_specs: []const ShapeSpec,
        device_layouts: ?[]*const MemoryLayout = null,
        memory: *const Memory,
    };

    pub fn createBuffersForAsyncHostToDevice(self: *const Client, api: *const Api, args: CreateBuffersForAsyncHostToDeviceArgs) ApiError!*AsyncHostToDeviceTransferManager {
        const ret = try api.call(.PJRT_Client_CreateBuffersForAsyncHostToDevice, .{
            .client = self.inner(),
            .shape_specs = @ptrCast(args.shape_specs.ptr),
            .num_shape_specs = args.shape_specs.len,
            .device_layouts = if (args.device_layouts) |layouts| @ptrCast(@constCast(layouts.ptr)) else null,
            .num_device_layouts = if (args.device_layouts) |layouts| @intCast(layouts.len) else 0,
            .memory = @ptrCast(@constCast(args.memory)),
        });
        return @ptrCast(ret.transfer_manager.?);
    }
};

pub const Device = opaque {
    const inner = InnerMixin(c.PJRT_Device).inner;

    pub fn getDescription(self: *const Device, api: *const Api) *const DeviceDescription {
        const ret = api.call(.PJRT_Device_GetDescription, .{
            .device = self.inner(),
        }) catch unreachable;
        return @ptrCast(ret.device_description.?);
    }

    pub fn isAddressable(self: *const Device, api: *const Api) bool {
        const ret = api.call(.PJRT_Device_IsAddressable, .{
            .device = self.inner(),
        }) catch unreachable;
        return ret.is_addressable;
    }

    pub fn getLocalHardwareId(self: *const Device, api: *const Api) usize {
        const ret = api.call(.PJRT_Device_LocalHardwareId, .{
            .device = self.inner(),
        }) catch unreachable;
        return @intCast(ret.local_hardware_id);
    }
};

pub const DeviceDescription = opaque {
    const inner = InnerMixin(c.PJRT_DeviceDescription).inner;

    pub fn getId(self: *const DeviceDescription, api: *const Api) usize {
        const ret = api.call(.PJRT_DeviceDescription_Id, .{
            .device_description = self.inner(),
        }) catch unreachable;
        return @intCast(ret.id);
    }

    pub fn getProcessIndex(self: *const DeviceDescription, api: *const Api) usize {
        const ret = api.call(.PJRT_DeviceDescription_ProcessIndex, .{
            .device_description = self.inner(),
        }) catch unreachable;
        return @intCast(ret.process_index);
    }

    pub fn getKind(self: *const DeviceDescription, api: *const Api) []const u8 {
        const ret = api.call(.PJRT_DeviceDescription_Kind, .{
            .device_description = self.inner(),
        }) catch unreachable;
        return ret.device_kind[0..ret.device_kind_size];
    }

    pub fn debugString(self: *const DeviceDescription, api: *const Api) []const u8 {
        const ret = api.call(.PJRT_DeviceDescription_DebugString, .{
            .device_description = self.inner(),
        }) catch unreachable;
        return ret.debug_string[0..ret.debug_string_size];
    }

    pub fn toString(self: *const DeviceDescription, api: *const Api) []const u8 {
        const ret = api.call(.PJRT_DeviceDescription_ToString, .{
            .device_description = self.inner(),
        }) catch unreachable;
        return ret.to_string[0..ret.to_string_size];
    }
};

pub const GetCostAnalysisError = std.mem.Allocator.Error || ApiError;

pub const SerializeResult = struct {
    bytes: []const u8,
    handle: *anyopaque,
    deleter: *const fn (?*anyopaque) callconv(.C) void,

    pub fn deinit(self: *SerializeResult) void {
        self.deleter(self.handle);
        self.bytes = &.{};
        self.* = undefined;
    }
};

pub const Executable = opaque {
    const inner = InnerMixin(c.PJRT_Executable).inner;

    pub fn deinit(self: *Executable, api: *const Api) void {
        _ = api.call(.PJRT_Executable_Destroy, .{
            .executable = self.inner(),
        }) catch unreachable;
    }

    pub fn getCostAnalysis(self: *const Executable, api: *const Api) GetCostAnalysisError![]*const NamedValue {
        const ret = try api.call(.PJRT_Executable_GetCostAnalysis, .{
            .executable = self.inner(),
        });
        const values: [*]*const NamedValue = @ptrCast(ret.properties);
        return values[0..ret.num_properties];
    }

    pub fn serialize(self: *const Executable, api: *const Api) ApiError!SerializeResult {
        const ret = try api.call(.PJRT_Executable_Serialize, .{
            .executable = self.inner(),
        });

        return .{
            .bytes = ret.serialized_bytes[0..ret.serialized_bytes_size],
            .handle = ret.serialized_executable.?,
            .deleter = @ptrCast(ret.serialized_executable_deleter.?),
        };
    }
};

pub const LoadedExecutable = opaque {
    const inner = InnerMixin(c.PJRT_LoadedExecutable).inner;

    pub fn deinit(self: *LoadedExecutable, api: *const Api) void {
        _ = api.call(.PJRT_LoadedExecutable_Destroy, .{
            .executable = self.inner(),
        }) catch {};
        self.* = undefined;
    }

    pub fn delete(self: *LoadedExecutable, api: *const Api) void {
        _ = api.call(.PJRT_LoadedExecutable_Delete, .{
            .executable = self.inner(),
        }) catch unreachable;
    }

    pub fn isDeleted(self: *const LoadedExecutable, api: *const Api) bool {
        const ret = api.call(.PJRT_LoadedExecutable_IsDeleted, .{
            .executable = self.inner(),
        }) catch unreachable;
        return ret.is_deleted;
    }

    pub fn getAddressableDevices(self: *const LoadedExecutable, api: *const Api) []Device {
        const ret = api.call(.PJRT_LoadedExecutable_AddressableDevices, .{
            .executable = self.inner(),
        }) catch unreachable;
        return @ptrCast(ret.addressable_devices);
    }

    pub const ExecuteArgs = struct {
        num_args: usize,
        arguments: []const [*]const *const Buffer,
        results: []const [*]*Buffer,
        events: []?*Event,
        non_donatable_input_indices: []const i64 = &.{},
    };
    pub fn execute(self: *const LoadedExecutable, api: *const Api, args: ExecuteArgs) ApiError!void {
        var options = pjrtStruct(c.PJRT_ExecuteOptions{
            .send_callbacks = null,
            .recv_callbacks = null,
            .num_send_ops = 0,
            .num_recv_ops = 0,
            .launch_id = 0,
            .non_donatable_input_indices = @ptrCast(args.non_donatable_input_indices.ptr),
            .num_non_donatable_input_indices = args.non_donatable_input_indices.len,
        });
        _ = try api.call(.PJRT_LoadedExecutable_Execute, .{
            .executable = self.inner(),
            .options = @ptrCast(&options),
            .argument_lists = @ptrCast(args.arguments.ptr),
            .num_devices = @intCast(args.arguments.len),
            .num_args = args.num_args,
            .output_lists = @ptrCast(args.results.ptr),
            .device_complete_events = @ptrCast(args.events.ptr),
            .execute_device = null,
        });
    }

    pub fn getExecutable(self: *LoadedExecutable, api: *const Api) ApiError!*Executable {
        const ret = try api.call(.PJRT_LoadedExecutable_GetExecutable, .{
            .loaded_executable = self.inner(),
        });
        return @ptrCast(ret.executable.?);
    }
};

pub const BufferType = enum(c.PJRT_Buffer_Type) {
    INVALID = c.PJRT_Buffer_Type_INVALID,
    PRED = c.PJRT_Buffer_Type_PRED,
    S8 = c.PJRT_Buffer_Type_S8,
    S16 = c.PJRT_Buffer_Type_S16,
    S32 = c.PJRT_Buffer_Type_S32,
    S64 = c.PJRT_Buffer_Type_S64,
    U8 = c.PJRT_Buffer_Type_U8,
    U16 = c.PJRT_Buffer_Type_U16,
    U32 = c.PJRT_Buffer_Type_U32,
    U64 = c.PJRT_Buffer_Type_U64,
    F16 = c.PJRT_Buffer_Type_F16,
    F32 = c.PJRT_Buffer_Type_F32,
    F64 = c.PJRT_Buffer_Type_F64,
    BF16 = c.PJRT_Buffer_Type_BF16,
    C64 = c.PJRT_Buffer_Type_C64,
    C128 = c.PJRT_Buffer_Type_C128,
    F8E5M2 = c.PJRT_Buffer_Type_F8E5M2,
    F8E4M3FN = c.PJRT_Buffer_Type_F8E4M3FN,
    F8E4M3B11FNUZ = c.PJRT_Buffer_Type_F8E4M3B11FNUZ,
    F8E5M2FNUZ = c.PJRT_Buffer_Type_F8E5M2FNUZ,
    F8E4M3FNUZ = c.PJRT_Buffer_Type_F8E4M3FNUZ,
    S4 = c.PJRT_Buffer_Type_S4,
    U4 = c.PJRT_Buffer_Type_U4,
};

pub const MemoryLayoutType = enum(c.PJRT_Buffer_MemoryLayout_Type) {
    tiled = c.PJRT_Buffer_MemoryLayout_Type_Tiled,
    strides = c.PJRT_Buffer_MemoryLayout_Type_Strides,
};

pub const MemoryLayout = union(MemoryLayoutType) {
    pub const Type = MemoryLayoutType;

    pub const Tiled = struct {
        minor_to_major: []const i64,
        tile_dims: []const i64,
        tile_dims_sizes: []const usize,
    };

    pub const Strides = struct {
        byte_strides: []const i64,
    };

    tiled: Tiled,
    strides: Strides,

    fn toCStruct(self: MemoryLayout) c.PJRT_Buffer_MemoryLayout {
        return pjrtStruct(switch (self) {
            .tiled => |v| c.PJRT_Buffer_MemoryLayout{
                .type = c.PJRT_Buffer_MemoryLayout_Type_Tiled,
                .unnamed_0 = .{
                    .tiled = c.PJRT_Buffer_MemoryLayout_Tiled{
                        .minor_to_major = v.minor_to_major.ptr,
                        .minor_to_major_size = v.minor_to_major.len,
                        .tile_dims = v.tile_dims.ptr,
                        .tile_dim_sizes = v.tile_dims_sizes.ptr,
                        .num_tiles = v.tile_dims_sizes.len,
                    },
                },
            },
            .strides => |v| c.PJRT_Buffer_MemoryLayout{
                .type = c.PJRT_Buffer_MemoryLayout_Type_Strides,
                .unnamed_0 = .{
                    .strides = c.PJRT_Buffer_MemoryLayout_Strides{
                        .byte_strides = v.byte_strides.ptr,
                        .num_byte_strides = v.byte_strides.len,
                    },
                },
            },
        });
    }
};

pub const HostBufferSemantics = enum(c.PJRT_HostBufferSemantics) {
    ImmutableOnlyDuringCall = c.PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
    ImmutableUntilTransferCompletes = c.PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
    ImmutableZeroCopy = c.PJRT_HostBufferSemantics_kImmutableZeroCopy,
    MutableZeroCopy = c.PJRT_HostBufferSemantics_kMutableZeroCopy,
};

pub const Buffer = opaque {
    const inner = InnerMixin(c.PJRT_Buffer).inner;

    pub fn deinit(self: *Buffer, api: *const Api) void {
        _ = api.call(.PJRT_Buffer_Destroy, .{
            .buffer = self.inner(),
        }) catch unreachable;
    }

    pub fn getDevice(self: *const Buffer, api: *const Api) ApiError!*Device {
        const ret = try api.call(.PJRT_Buffer_Device, .{
            .buffer = self.inner(),
        });
        return @ptrCast(ret.device.?);
    }

    pub fn delete(self: *Buffer, api: *const Api) void {
        _ = api.call(.PJRT_Buffer_Delete, .{
            .buffer = self.inner(),
        }) catch unreachable;
    }

    pub fn isDeleted(self: *const Buffer, api: *const Api) bool {
        const ret = api.call(.PJRT_Buffer_IsDeleted, .{
            .buffer = self.inner(),
        }) catch unreachable;
        return ret.is_deleted;
    }

    pub fn isOnCpu(self: *const Buffer, api: *const Api) bool {
        const ret = api.call(.PJRT_Buffer_IsOnCpu, .{
            .buffer = self.inner(),
        }) catch unreachable;
        return ret.is_on_cpu;
    }

    pub fn toHostBuffer(self: *const Buffer, api: *const Api, dst: []u8) ApiError!?*Event {
        const ret = try api.call(.PJRT_Buffer_ToHostBuffer, .{
            .src = self.inner(),
            .dst = @ptrCast(dst.ptr),
            .dst_size = dst.len,
        });
        return @ptrCast(ret.event);
    }

    pub fn getElementType(self: *const Buffer, api: *const Api) BufferType {
        const ret = api.call(.PJRT_Buffer_ElementType, .{
            .buffer = self.inner(),
        }) catch unreachable;
        return @enumFromInt(ret.type);
    }

    pub fn getDimensions(self: *const Buffer, api: *const Api) []const i64 {
        const ret = api.call(.PJRT_Buffer_Dimensions, .{
            .buffer = self.inner(),
        }) catch unreachable;
        if (ret.num_dims == 0) {
            return &.{};
        }
        return ret.dims[0..ret.num_dims];
    }

    pub fn getUnpaddedDimensions(self: *const Buffer, api: *const Api) ApiError![]const i64 {
        const ret = try api.call(.PJRT_Buffer_UnpaddedDimensions, .{
            .buffer = self.inner(),
        });
        return ret.dims[0..ret.num_dims];
    }

    pub fn getOnDeviceSizeInBytes(self: *const Buffer, api: *const Api) ApiError!usize {
        const ret = try api.call(.PJRT_Buffer_OnDeviceSizeInBytes, .{
            .buffer = self.inner(),
        });
        return ret.on_device_size_in_bytes;
    }

    pub fn copyToDevice(self: *const Buffer, api: *const Api, device: Device) ApiError!Buffer {
        const ret = try api.call(.PJRT_Buffer_CopyToDevice, .{
            .buffer = self.inner(),
            .dst_device = device.inner,
        });
        return @ptrCast(ret.dst_buffer.?);
    }

    pub fn getReadyEvent(self: *const Buffer, api: *const Api) *Event {
        const ret = api.call(.PJRT_Buffer_ReadyEvent, .{
            .buffer = self.inner(),
        }) catch unreachable;
        return @ptrCast(ret.event.?);
    }

    pub fn getOpaqueDeviceMemoryDataPointer(self: *const Buffer, api: *const Api) ApiError!*anyopaque {
        const ret = try api.call(.PJRT_Buffer_OpaqueDeviceMemoryDataPointer, .{
            .buffer = self.inner(),
        });
        return ret.device_memory_ptr.?;
    }

    pub fn copyRawToHost(self: *const Buffer, api: *const Api, dst: []u8, offset: i64) ApiError!?*Event {
        const ret = try api.call(.PJRT_Buffer_CopyRawToHost, .{
            .buffer = self.inner(),
            .dst = @ptrCast(dst.ptr),
            .offset = offset,
            .transfer_size = @intCast(dst.len),
        });
        return @ptrCast(ret.event);
    }

    pub fn copyToMemory(self: *const Buffer, api: *const Api, dst_memory: *const Memory) ApiError!?*Buffer {
        const ret = try api.call(.PJRT_Buffer_CopyToMemory, .{
            .buffer = self.inner(),
            .dst_memory = @ptrCast(@constCast(dst_memory)),
        });
        return @ptrCast(ret.dst_buffer);
    }

    pub fn memory(self: *const Buffer, api: *const Api) *const Memory {
        const ret = api.call(.PJRT_Buffer_Memory, .{
            .buffer = self.inner(),
        }) catch unreachable;
        return @ptrCast(ret.memory);
    }

    pub fn increaseExternalReferenceCount(self: *const Buffer, api: *const Api) ApiError!void {
        _ = try api.call(.PJRT_Buffer_IncreaseExternalReferenceCount, .{
            .buffer = self.inner(),
        });
    }

    pub fn decreaseExternalReferenceCount(self: *const Buffer, api: *const Api) ApiError!void {
        _ = try api.call(.PJRT_Buffer_DecreaseExternalReferenceCount, .{
            .buffer = self.inner(),
        });
    }
};

pub const Event = opaque {
    const inner = InnerMixin(c.PJRT_Event).inner;

    pub fn deinit(self: *Event, api: *const Api) void {
        _ = api.call(.PJRT_Event_Destroy, .{
            .event = self.inner(),
        }) catch unreachable;
    }

    pub fn isReady(self: *const Event, api: *const Api) bool {
        const ret = api.call(.PJRT_Event_IsReady, .{
            .event = self.inner(),
        }) catch unreachable;
        return ret.is_ready;
    }

    pub fn getEventError(self: *const Event, api: *const Api) ?*Error {
        var args: Api.CallFnArgType(.PJRT_Event_Error) = .{ .event = self.inner() };
        args = pjrtStruct(args);
        const result: ?*c.PJRT_Error = api.inner.PJRT_Event_Error.?(&args);
        return @ptrCast(result);
    }

    pub fn await_(self: *const Event, api: *const Api) ApiError!void {
        _ = try api.call(.PJRT_Event_Await, .{
            .event = self.inner(),
        });
    }

    pub fn onReady(self: *Event, api: *const Api, func: *const fn (err: ?*Error, user_arg: ?*anyopaque) callconv(.C) void, user_arg: ?*anyopaque) ApiError!void {
        _ = try api.call(.PJRT_Event_OnReady, .{
            .event = self.inner(),
            .callback = @ptrCast(func),
            .user_arg = user_arg,
        });
    }
};

pub const Memory = opaque {
    pub const Kind = enum {
        device,
        pinned_host,
        unpinned_host,
    };

    const inner = InnerMixin(c.PJRT_Memory).inner;

    pub fn id(self: *const Memory, api: *const Api) usize {
        const ret = api.call(.PJRT_Memory_Id, .{
            .memory = self.inner(),
        }) catch unreachable;
        return @intCast(ret.id);
    }

    pub fn kind(self: *const Memory, api: *const Api) Kind {
        const ret = api.call(.PJRT_Memory_Kind, .{
            .memory = self.inner(),
        }) catch unreachable;
        const kind_ = ret.kind orelse unreachable;
        return std.meta.stringToEnum(Kind, kind_[0..ret.kind_size]) orelse unreachable;
    }

    pub fn kindId(self: *const Memory, api: *const Api) u32 {
        const ret = api.call(.PJRT_Memory_Kind_Id, .{
            .memory = self.inner(),
        }) catch unreachable;
        return @bitCast(ret.kind_id);
    }

    pub fn debugString(self: *const Memory, api: *const Api) []const u8 {
        const ret = api.call(.PJRT_Memory_DebugString, .{
            .memory = self.inner(),
        }) catch unreachable;
        if (ret.debug_string) |debug_string| {
            return debug_string[0..ret.debug_string_size];
        }
        return &.{};
    }

    pub fn toString(self: *const Memory, api: *const Api) []const u8 {
        const ret = api.call(.PJRT_Memory_ToString, .{
            .memory = self.inner(),
        }) catch unreachable;
        if (ret.to_string) |to_string| {
            return to_string[0..ret.to_string_size];
        }
        return &.{};
    }

    pub fn addressableByDevices(self: *const Memory, api: *const Api) []*Device {
        const ret = api.call(.PJRT_Memory_AddressableByDevices, .{
            .event = self.inner(),
        }) catch unreachable;
        if (ret.devices) |devices| {
            return devices[0..ret.num_devices];
        }
        return &.{};
    }
};

pub const AsyncHostToDeviceTransferManager = opaque {
    const inner = InnerMixin(c.PJRT_AsyncHostToDeviceTransferManager).inner;

    pub fn deinit(self: *AsyncHostToDeviceTransferManager, api: *const Api) void {
        _ = api.call(.PJRT_AsyncHostToDeviceTransferManager_Destroy, .{
            .transfer_manager = self.inner(),
        }) catch unreachable;
    }

    pub fn transferData(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize, data: []const u8, offset: i64, is_last_transfer: bool) ApiError!*Event {
        const ret = try api.call(.PJRT_AsyncHostToDeviceTransferManager_TransferData, .{
            .transfer_manager = self.inner(),
            .buffer_index = @intCast(buffer_index),
            .data = data.ptr,
            .offset = offset,
            .transfer_size = @intCast(data.len),
            .is_last_transfer = is_last_transfer,
        });
        return @ptrCast(ret.done_with_h2d_transfer.?);
    }

    pub fn retrieveBuffer(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize) ApiError!*Buffer {
        const ret = try api.call(.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer, .{
            .transfer_manager = self.inner(),
            .buffer_index = @intCast(buffer_index),
        });
        return @ptrCast(ret.buffer_out.?);
    }

    pub fn device(self: *AsyncHostToDeviceTransferManager, api: *const Api) ApiError!*Device {
        const ret = try api.call(.PJRT_AsyncHostToDeviceTransferManager_Device, .{
            .transfer_manager = self.inner(),
        });
        return @ptrCast(ret.device_out.?);
    }

    pub fn bufferCount(self: *AsyncHostToDeviceTransferManager, api: *const Api) ApiError!usize {
        const ret = try api.call(.PJRT_AsyncHostToDeviceTransferManager_BufferCount, .{
            .transfer_manager = self.inner(),
        });
        return ret.buffer_count;
    }

    pub fn bufferSize(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize) ApiError!usize {
        const ret = try api.call(.PJRT_AsyncHostToDeviceTransferManager_BufferSize, .{
            .transfer_manager = self.inner(),
            .buffer_index = @intCast(buffer_index),
        });
        return ret.buffer_size;
    }

    pub fn setBufferError(self: *AsyncHostToDeviceTransferManager, api: *const Api, buffer_index: usize, error_code: c.PJRT_Error_Code, error_message: []const u8) ApiError!void {
        _ = try api.call(.PJRT_AsyncHostToDeviceTransferManager_SetBufferError, .{
            .transfer_manager = self.inner(),
            .buffer_index = @intCast(buffer_index),
            .error_code = error_code,
            .error_message = error_message.ptr,
            .error_message_size = error_message.len,
        });
    }

    pub fn addMetadata(self: *AsyncHostToDeviceTransferManager, api: *const Api, transfer_metadata: []const NamedValue) ApiError!void {
        _ = try api.call(.PJRT_AsyncHostToDeviceTransferManager_AddMetadata, .{
            .transfer_manager = self.inner(),
            .transfer_metadata = transfer_metadata.ptr,
            .num_metadata = transfer_metadata.len,
        });
    }
};

pub const ExecutionContext = opaque {
    const inner = InnerMixin(c.PJRT_ExecutionContext).inner;

    pub fn init(api: *const Api) ApiError!*ExecutionContext {
        const ret = try api.call(.PJRT_ExecutionContext_Create, .{});
        return @ptrCast(ret.context.?);
    }

    pub fn deinit(self: *ExecutionContext, api: *const Api) void {
        _ = api.call(.PJRT_ExecutionContext_Destroy, .{
            .context = self.inner(),
        }) catch unreachable;
    }
};

pub const NamedValue = extern struct {
    comptime {
        std.debug.assert(@sizeOf(NamedValue) == @sizeOf(c.PJRT_NamedValue));
    }

    inner: c.PJRT_NamedValue,

    pub const Kind = enum(c.PJRT_NamedValue_Type) {
        string = c.PJRT_NamedValue_kString,
        int64 = c.PJRT_NamedValue_kInt64,
        int64list = c.PJRT_NamedValue_kInt64List,
        float = c.PJRT_NamedValue_kFloat,
        bool = c.PJRT_NamedValue_kBool,
    };

    pub fn kind(self: NamedValue) Kind {
        return @enumFromInt(self.inner.type);
    }

    pub fn name(self: NamedValue) []const u8 {
        return self.inner.name[0..self.inner.name_size];
    }

    pub fn from(name_: []const u8, value: anytype) NamedValue {
        return switch (@TypeOf(value)) {
            []u8, []const u8 => fromString(name_, value),
            i64 => fromInt64(name_, value),
            []i64, []const i64 => fromInt64List(name_, value),
            f32 => fromFloat(name_, value),
            bool => fromBool(name_, value),
            else => fromString(name_, @tagName(value)),
        };
    }

    pub fn fromString(name_: []const u8, value: []const u8) NamedValue {
        return .{ .inner = pjrtStruct(c.PJRT_NamedValue{
            .name = @ptrCast(@constCast(name_.ptr)),
            .name_size = name_.len,
            .type = c.PJRT_NamedValue_kString,
            .unnamed_0 = .{ .string_value = @ptrCast(@constCast(value.ptr)) },
            .value_size = value.len,
        }) };
    }

    pub fn fromInt64(name_: []const u8, value: i64) NamedValue {
        return .{ .inner = pjrtStruct(c.PJRT_NamedValue{
            .name = @ptrCast(@constCast(name_.ptr)),
            .name_size = name_.len,
            .type = c.PJRT_NamedValue_kInt64,
            .unnamed_0 = .{ .int64_value = value },
            .value_size = 1,
        }) };
    }

    pub fn fromInt64List(name_: []const u8, value: []const i64) NamedValue {
        return .{ .inner = pjrtStruct(c.PJRT_NamedValue{
            .name = @ptrCast(@constCast(name_.ptr)),
            .name_size = name_.len,
            .type = c.PJRT_NamedValue_kInt64List,
            .unnamed_0 = .{ .int64_array_value = @ptrCast(@constCast(value.ptr)) },
            .value_size = value.len,
        }) };
    }

    pub fn fromFloat(name_: []const u8, value: f32) NamedValue {
        return .{ .inner = pjrtStruct(c.PJRT_NamedValue{
            .name = @ptrCast(@constCast(name_.ptr)),
            .name_size = name_.len,
            .type = c.PJRT_NamedValue_kFloat,
            .unnamed_0 = .{ .float_value = value },
            .value_size = 1,
        }) };
    }

    pub fn fromBool(name_: []const u8, value: bool) NamedValue {
        return .{ .inner = pjrtStruct(c.PJRT_NamedValue{
            .name = @ptrCast(@constCast(name_.ptr)),
            .name_size = name_.len,
            .type = c.PJRT_NamedValue_kBool,
            .unnamed_0 = .{ .bool_value = value },
            .value_size = 1,
        }) };
    }

    pub fn format(
        self: NamedValue,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}{{ .name = {s},", .{ @typeName(NamedValue), self.inner.name[0..self.inner.name_size] });
        const u = self.inner.unnamed_0;
        switch (self.kind()) {
            .string => try writer.print(" .string = {s} ", .{u.string_value[0..self.inner.value_size]}),
            .int64 => try writer.print(" .int64 = {d} ", .{u.int64_value}),
            .int64list => try writer.print(" .int64list = {d} ", .{u.int64_array_value[0..self.inner.value_size]}),
            .float => try writer.print(" .float = {d} ", .{u.float_value}),
            .bool => try writer.print(" .bool = {} ", .{u.bool_value}),
        }
        try writer.writeAll("}");
    }
};

/// Custom call signature arguments are:
/// * a pointer to a platform specific stream handle
/// * a pointer to an unspecified list of platform specific buffer handle
/// * a context struct passed as a slice of bytes
pub const CustomCall = fn (*anyopaque, [*]*anyopaque, [*]const u8, usize) callconv(.C) void;

// todo : support all missing handlers available in GPU plugin extension: handler_instantiate, handler_prepare, handler_initialize
// introduced by https://github.com/openxla/xla/commit/ef85a7bcc308313492ebc50295a8a08b4e51b8f5
pub const CustomCallRegistry = extern struct {
    inner: *const c.PJRT_Gpu_Register_Custom_Call,

    pub fn register(self: *const CustomCallRegistry, api: *const Api, api_version: usize, name: []const u8, func: *const CustomCall) ApiError!void {
        var ret = pjrtStruct(c.PJRT_Gpu_Register_Custom_Call_Args{
            .function_name = name.ptr,
            .function_name_size = name.len,
            .api_version = @intCast(api_version),
            .handler_execute = @ptrCast(@constCast(func)),
        });
        const result = self.inner(&ret);
        if (result) |pjrt_c_error| {
            const pjrt_error: *Error = @ptrCast(pjrt_c_error);
            log.err("[GpuRegisterCustomCall] {s}", .{pjrt_error.getMessage(api)});
            return pjrt_error.getCode(api).toApiError();
        }
    }
};
