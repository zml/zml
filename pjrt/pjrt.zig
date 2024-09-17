const builtin = @import("builtin");
const std = @import("std");

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
        const fn_ptr = @typeInfo(fti.Optional.child);
        const fn_type_info = @typeInfo(fn_ptr.Pointer.child);
        const arg_array_type_info = @typeInfo(fn_type_info.Fn.params[0].type.?);
        return arg_array_type_info.Pointer.child;
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
            .cancelled => error.Cancelled,
            .unknown => error.Unknown,
            .invalid_argument => error.InvalidArgument,
            .deadline_exceeded => error.DeadlineExceeded,
            .not_found => error.NotFound,
            .already_exists => error.AlreadyExists,
            .permission_denied => error.PermissionDenied,
            .resource_exhausted => error.ResourceExhausted,
            .failed_precondition => error.FailedPrecondition,
            .aborted => error.Aborted,
            .out_of_range => error.OutOfRange,
            .unimplemented => error.Unimplemented,
            .internal => error.Internal,
            .unavailable => error.Unavailable,
            .data_loss => error.DataLoss,
            .unauthenticated => error.Unauthenticated,
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
        device: *const Device,
        host_buffer_semantics: HostBufferSemantics,
    };

    pub fn bufferFromHostBuffer(self: *const Client, api: *const Api, args: BufferFromHostBufferArgs) ApiError!struct { *Buffer, *Event } {
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
            .memory = null, // TODO
            .device_layout = null, // TODO
            .done_with_host_buffer = null,
            .buffer = null,
        });
        return .{
            @ptrCast(ret.buffer.?),
            @ptrCast(ret.done_with_host_buffer.?),
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
        return Profiler.init(null, options);
    }

    // pub fn getGpuCustomCallRegistry(self: *const Client, api: *const Api) ?*GpuCustomCallRegistry {
    //     if (api.lookupExtension(c.PJRT_Gpu_Custom_Call, c.PJRT_Extension_Type_Gpu_Custom_Call)) |ext| {
    //         return .{ .custom_call_register = ext.custom_call.? };
    //     }
    //     log.warn("No Gpu Custom Call registry found for platform: {}", .{self});
    //     return null;
    // }

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
        on_delete_callback: ?*const fn (device_buffer_ptr: ?*anyopaque, ctx: ?*anyopaque) callconv(.C) void = null,
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
};

// // pub const CustomCallSignature = *const fn (*anyopaque, **anyopaque, [*c]const u8, usize) callconv(.C) void;

// // pub const GpuCustomCallRegistry = struct {
// //     custom_call_register: *const c.PJRT_Gpu_Register_Custom_Call,

// //     pub fn registerCustomCall(self: GpuCustomCallRegistry, api: *const Api, api_version: usize, name: []const u8, func: CustomCallSignature) ApiError!void {
// //         var ret = pjrtStruct(c.PJRT_Gpu_Register_Custom_Call_Args{
// //             .function_name = name.ptr,
// //             .function_name_size = name.len,
// //             .api_version = @intCast(api_version),
// //             .custom_call_function = @ptrCast(@constCast(func)),
// //         });
// //         const result = self.custom_call_register(&ret);
// //         if (result) |pjrt_c_error| {
// //             const pjrt_error = .{ .inner = pjrt_c_error };
// //             log.err("{s}", .{pjrt_error.getMessage(api)});
// //             return pjrt_error.getCode().toApiError();
// //         }
// //     }
// // };

// // const OldPjrtExtension = extern struct {
// //     type: c.PJRT_Extension_Type,
// //     next: [*]OldPjrtExtension,
// // };

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

    pub fn execute(self: *const LoadedExecutable, api: *const Api, args: struct {
        num_args: usize,
        arguments: []const [*]const *const Buffer,
        results: []const [*]*Buffer,
        events: []*Event,
        non_donatable_input_indices: []const i64 = &.{},
    }) ApiError!void {
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
    Tiled = c.PJRT_Buffer_MemoryLayout_Type_Tiled,
    Strides = c.PJRT_Buffer_MemoryLayout_Type_Strides,
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

    Tiled: Tiled,
    Strides: Strides,

    fn toCStruct(self: MemoryLayout) c.PJRT_Buffer_MemoryLayout {
        return pjrtStruct(switch (self) {
            .Tiled => |v| c.PJRT_Buffer_MemoryLayout{
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
            .Strides => |v| c.PJRT_Buffer_MemoryLayout{
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

    pub fn toHostBuffer(self: *const Buffer, api: *const Api, dst: []u8) ApiError!*Event {
        const ret = try api.call(.PJRT_Buffer_ToHostBuffer, .{
            .src = self.inner(),
            .dst = @ptrCast(dst.ptr),
            .dst_size = dst.len,
        });
        return @ptrCast(ret.event.?);
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

    pub fn getEventError(self: *const Event, api: *const Api) ApiError!?*Error {
        const ret = try api.call(.PJRT_Event_Error, .{
            .event = self.inner(),
        });
        return @ptrCast(ret);
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
            else => unreachable,
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
