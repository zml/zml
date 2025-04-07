/// Bindings for PJRT custom call declaration / execution.
const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");

const pjrtStruct = @import("pjrt.zig").pjrtStruct;

const log = std.log.scoped(.pjrt);

pub const ApiVersion = extern struct {
    pub const major = c.XLA_FFI_API_MAJOR;
    pub const minor = c.XLA_FFI_API_MINOR;

    struct_size: usize,
    extension_start: ?*ExtensionBase,
    major_version: i32,
    minor_version: i32,
};

pub const ExtensionType = enum(c.XLA_FFI_Extension_Type) {
    metadata = c.XLA_FFI_Extension_Metadata,
};

pub const ExtensionBase = extern struct {
    struct_size: usize,
    type: ExtensionType,
    next: ?*ExtensionBase,
};

// Based of https://github.com/openxla/xla/blob/145f836bd5175dc5dd262f716a0c59af2b0297a0/xla/ffi/api/c_api.h#L449
pub const HandlerTraits = packed struct(u32) {
    /// Calls to FFI handler are safe to trace into the command buffer.
    /// It means that calls to FFI handler always launch exactly the same device operations (can depend on attribute values)
    /// that can be captured and then replayed.
    command_buffer_compatible: u1,

    __unassigned__: u31,
};

pub const Metadata = extern struct {
    struct_size: usize,
    api_version: ApiVersion,
    traits: HandlerTraits,
};

pub const MetadataExtension = extern struct {
    extension_base: ExtensionBase,
    metadata: ?*Metadata,
};

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

fn TransmuteMixin(comptime T: type, comptime InnerT: type) type {
    return struct {
        pub fn to(self: anytype) switch (@TypeOf(self)) {
            *T => *InnerT,
            *const T => *const InnerT,
            else => unreachable,
        } {
            return @ptrCast(@alignCast(self));
        }

        pub fn from(self: anytype) switch (@TypeOf(self)) {
            *InnerT => *T,
            *const InnerT => *const T,
            else => unreachable,
        } {
            return @ptrCast(@alignCast(self));
        }
    };
}

pub const Api = opaque {
    pub const inner = TransmuteMixin(Api, c.XLA_FFI_Api).to;

    pub fn getStream(self: *const Api, context: ?*ExecutionContext) ApiError!*anyopaque {
        var ret = pjrtStruct(c.XLA_FFI_Stream_Get_Args{
            .ctx = if (context) |ctx| ctx.inner() else null,
        });
        const result = self.inner().XLA_FFI_Stream_Get.?(&ret);

        if (result) |ffi_error| {
            const err = Error.fromInner(ffi_error);
            defer err.destroy(self);
            log.err("[Api.getStream] {s}", .{err.getMessage(self)});

            // TODO(Corentin): Retrieve error code from Error when implemented in XLA.
            return error.Unknown;
        }

        return ret.stream.?;
    }

    pub fn allocateDeviceMemory(self: *const Api, context: ?*ExecutionContext, size: usize, alignment: usize) ApiError!*anyopaque {
        var ret = pjrtStruct(c.XLA_FFI_DeviceMemory_Allocate_Args{
            .ctx = if (context) |ctx| ctx.inner() else null,
            .size = size,
            .alignment = alignment,
        });
        const result = self.inner().XLA_FFI_DeviceMemory_Allocate.?(&ret);

        if (result) |ffi_error| {
            const err = Error.fromInner(ffi_error);
            defer err.destroy(self);
            log.err("[Api.allocateDeviceMemory] {s}", .{err.getMessage(self)});

            // TODO(Corentin): Retrieve error code from Error when implemented in XLA.
            return error.Unknown;
        }

        return ret.data.?;
    }

    pub fn freeDeviceMemory(self: *const Api, context: ?*ExecutionContext, data: *anyopaque, size: usize) ApiError!void {
        var ret = pjrtStruct(c.XLA_FFI_DeviceMemory_Free_Args{
            .ctx = if (context) |ctx| ctx.inner() else null,
            .size = size,
            .data = data,
        });
        const result = self.inner().XLA_FFI_DeviceMemory_Free.?(&ret);

        if (result) |ffi_error| {
            const err = Error.fromInner(ffi_error);
            defer err.destroy(self);
            log.err("[Api.freeDeviceMemory] {s}", .{err.getMessage(self)});

            // TODO(Corentin): Retrieve error code from Error when implemented in XLA.
            return error.Unknown;
        }
    }

    // TODO(Corentin): Implement remaining methods if needed:
    // * XLA_FFI_ThreadPool_Schedule
    // * XLA_FFI_Handler_Register
    // * XLA_FFI_TypeId_Register
    // * XLA_FFI_State_Set
    // * XLA_FFI_State_Get
};

pub const ExecutionStage = enum(c.XLA_FFI_ExecutionStage) {
    instantiate = c.XLA_FFI_ExecutionStage_INSTANTIATE,
    prepare = c.XLA_FFI_ExecutionStage_PREPARE,
    initialize = c.XLA_FFI_ExecutionStage_INITIALIZE,
    execute = c.XLA_FFI_ExecutionStage_EXECUTE,
};

pub const ExecutionContext = opaque {
    pub const inner = TransmuteMixin(ExecutionContext, c.XLA_FFI_ExecutionContext).to;

    // pub fn attach(self: *ExecutionContext, api: *const Api, value: anytype) ApiError!void {
    //     // register type id ==> typeid
    //     const typename_ = "zml." ++ @typeName(@TypeOf(value));

    //     var ret = pjrtStruct(c.XLA_FFI_ExecutionContext_Register_Args{
    //         .ctx = self.inner(),
    //         .handler = @ptrCast(@alignCast(handler)),
    //     });
    //     const result = api.inner().XLA_FFI_ExecutionContext_Register.?(&ret);

    //     var ret = pjrtStruct(c.XLA_FFI_ExecutionContext_Register_Args{
    //         .ctx = self.inner(),
    //         .handler = @ptrCast(@alignCast(handler)),
    //     });
    //     const result = api.inner().XLA_FFI_ExecutionContext_Register.?(&ret);

    //     if (result) |ffi_error| {
    //         const err = Error.fromInner(ffi_error);
    //         defer err.destroy(api);
    //         log.err("[ExecutionContext.register] {s}", .{err.getMessage(api)});

    //         // TODO(Corentin): Retrieve error code from Error when implemented in XLA.
    //         return error.Unknown;
    //     }
    // }

    pub fn get(self: *ExecutionContext, api: *const Api, type_id: *TypeId) ApiError!*anyopaque {
        var ret = pjrtStruct(c.XLA_FFI_ExecutionContext_Get_Args{
            .ctx = self.inner(),
            .type_id = @ptrCast(@alignCast(type_id)),
        });
        const result = api.inner().XLA_FFI_ExecutionContext_Get.?(&ret);

        if (result) |ffi_error| {
            const err = Error.fromInner(ffi_error);
            defer err.destroy(api);
            log.err("[ExecutionContext.get] {s}", .{err.getMessage(api)});

            // TODO(Corentin): Retrieve error code from Error when implemented in XLA.
            return error.Unknown;
        }

        return ret.data.?;
    }

    // TODO getDeviceOrdinal()
};

const ByteSpan = extern struct {
    ptr: [*]const u8,
    len: usize,

    pub fn slice(self: ByteSpan) []const u8 {
        return self.ptr[0..self.len];
    }
};

pub const TypeId = extern struct {
    type_id: i64,
};

pub const DataType = enum(c.XLA_FFI_DataType) {
    invalid = c.XLA_FFI_DataType_INVALID,
    pred = c.XLA_FFI_DataType_PRED,
    s8 = c.XLA_FFI_DataType_S8,
    s16 = c.XLA_FFI_DataType_S16,
    s32 = c.XLA_FFI_DataType_S32,
    s64 = c.XLA_FFI_DataType_S64,
    u8 = c.XLA_FFI_DataType_U8,
    u16 = c.XLA_FFI_DataType_U16,
    u32 = c.XLA_FFI_DataType_U32,
    u64 = c.XLA_FFI_DataType_U64,
    f16 = c.XLA_FFI_DataType_F16,
    f32 = c.XLA_FFI_DataType_F32,
    f64 = c.XLA_FFI_DataType_F64,
    bf16 = c.XLA_FFI_DataType_BF16,
    c64 = c.XLA_FFI_DataType_C64,
    c128 = c.XLA_FFI_DataType_C128,
    token = c.XLA_FFI_DataType_TOKEN,
    f8e5m2 = c.XLA_FFI_DataType_F8E5M2,
    f8e3m4 = c.XLA_FFI_DataType_F8E3M4,
    f8e4m3 = c.XLA_FFI_DataType_F8E4M3,
    f8e4m3fn = c.XLA_FFI_DataType_F8E4M3FN,
    f8e4m3b11fnuz = c.XLA_FFI_DataType_F8E4M3B11FNUZ,
    f8e5m2fnuz = c.XLA_FFI_DataType_F8E5M2FNUZ,
    f8e4m3fnuz = c.XLA_FFI_DataType_F8E4M3FNUZ,
};

pub const Buffer = extern struct {
    struct_size: usize,
    extension_start: ?*c.XLA_FFI_Extension_Base,
    dtype: DataType,
    data: [*]u8,
    rank: u64,
    _dims: [*]const i64,

    pub fn dims(self: Buffer) []const i64 {
        return self._dims[0..self.rank];
    }

    pub fn format(
        buffer: Buffer,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("FfiBuffer({d}, .{s})@0x{x}", .{ buffer.dims(), @tagName(buffer.dtype), @intFromPtr(buffer.data) });
    }
};

pub const Args = extern struct {
    struct_size: usize,
    extension_start: ?*const c.XLA_FFI_Extension_Base,
    len: u64,
    types: [*]const Type,
    ptr: [*]*const Buffer,

    pub const Type = enum(c.XLA_FFI_ArgType) {
        buffer = c.XLA_FFI_ArgType_BUFFER,
    };

    pub fn get(self: Args, i: usize) *const Buffer {
        std.debug.assert(self.types[0..self.len][i] == .buffer);
        return self.ptr[0..self.len][i];
    }
};

pub const Rets = extern struct {
    struct_size: usize,
    extension_start: ?*const c.XLA_FFI_Extension_Base,
    len: u64,
    types: [*]const Type,
    ptr: [*]*const Buffer,

    pub const Type = enum(c.XLA_FFI_RetType) {
        buffer = c.XLA_FFI_RetType_BUFFER,
    };

    pub fn get(self: Rets, i: usize) *const Buffer {
        std.debug.assert(self.types[0..self.len][i] == .buffer);
        return self.ptr[0..self.len][i];
    }
};

pub const AttrType = enum(c.XLA_FFI_AttrType) {
    array = c.XLA_FFI_AttrType_ARRAY,
    dictionary = c.XLA_FFI_AttrType_DICTIONARY,
    scalar = c.XLA_FFI_AttrType_SCALAR,
    string = c.XLA_FFI_AttrType_STRING,
};

pub const Attrs = extern struct {
    struct_size: usize,
    extension_start: ?*ExtensionBase,
    len: u64,
    types: [*]const AttrType,
    names: [*]const *const ByteSpan,
    ptr: [*]const *const Attr,

    const Attr = extern union {
        scalar: Scalar,
        array: Array,
    };

    pub const Scalar = extern struct {
        dtype: DataType,
        value: *const anyopaque,

        pub fn get(self: Scalar, T: type) T {
            const ptr: *const T = @alignCast(@ptrCast(self.value));
            return ptr.*;
        }
    };

    pub const Array = extern struct {
        dtype: DataType,
        len: usize,
        data: [*]const u8,
    };

    pub fn getByIndex(self: Attrs, comptime attr_type: AttrType, index: usize) ?*const @FieldType(Attr, @tagName(attr_type)) {
        const attr = self.ptr[0..self.len][index];
        const actual_type = self.types[index];
        if (actual_type != attr_type) return null;
        return @ptrCast(attr);
    }

    pub fn getByName(self: Attrs, comptime attr_type: AttrType, name: []const u8) ?*const @FieldType(Attr, @tagName(attr_type)) {
        const names = self.names[0..self.len];
        for (0.., names) |i, attr_name| {
            if (std.mem.eql(u8, attr_name.slice(), name)) {
                return self.getByIndex(attr_type, i);
            }
        }

        return null;
    }
};

pub const CallFrame = extern struct {
    struct_size: usize,
    extension_start: ?*ExtensionBase,
    api: ?*const Api,
    ctx: ?*const ExecutionContext,
    stage: ExecutionStage,
    args: Args,
    results: Rets,
    attrs: Attrs,
    future: ?*Future,

    /// The registery mechanism will first call the custom call in registration mode,
    /// and expects us to indicate which version of XLA we have been compiled against.
    /// Returns true if we registered ourselves and if the caller custom call should return early.
    pub fn registeringHook(call_frame: *CallFrame) bool {
        if (call_frame.extension_start != null and call_frame.extension_start.?.type == .metadata) {
            const metadata_extension: *MetadataExtension = @fieldParentPtr("extension_base", call_frame.extension_start.?);
            metadata_extension.metadata.?.api_version.major_version = ApiVersion.major;
            metadata_extension.metadata.?.api_version.minor_version = ApiVersion.minor;
            return true;
        }
        return false;
    }
};

pub const Handler = fn (*CallFrame) callconv(.C) ?*Error;

pub const ErrorCode = enum(c.XLA_FFI_Error_Code) {
    cancelled = c.XLA_FFI_Error_Code_CANCELLED,
    unknown = c.XLA_FFI_Error_Code_UNKNOWN,
    invalid_argument = c.XLA_FFI_Error_Code_INVALID_ARGUMENT,
    deadline_exceeded = c.XLA_FFI_Error_Code_DEADLINE_EXCEEDED,
    not_found = c.XLA_FFI_Error_Code_NOT_FOUND,
    already_exists = c.XLA_FFI_Error_Code_ALREADY_EXISTS,
    permission_denied = c.XLA_FFI_Error_Code_PERMISSION_DENIED,
    resource_exhausted = c.XLA_FFI_Error_Code_RESOURCE_EXHAUSTED,
    failed_precondition = c.XLA_FFI_Error_Code_FAILED_PRECONDITION,
    aborted = c.XLA_FFI_Error_Code_ABORTED,
    out_of_range = c.XLA_FFI_Error_Code_OUT_OF_RANGE,
    unimplemented = c.XLA_FFI_Error_Code_UNIMPLEMENTED,
    internal = c.XLA_FFI_Error_Code_INTERNAL,
    unavailable = c.XLA_FFI_Error_Code_UNAVAILABLE,
    data_loss = c.XLA_FFI_Error_Code_DATA_LOSS,
    unauthenticated = c.XLA_FFI_Error_Code_UNAUTHENTICATED,

    pub fn toApiError(code: ErrorCode) ApiError {
        return switch (code) {
            .cancelled => error.Cancelled,
            .unknown => error.Unknown,
            .invalid_argument => error.InvalidArgument,
            .deadline_exceeded => error.DeadlineExceeded,
            .not_found => error.FfiNotFound,
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
    pub const inner = TransmuteMixin(Error, c.XLA_FFI_Error).to;
    pub const fromInner = TransmuteMixin(Error, c.XLA_FFI_Error).from;

    pub fn create(api: *const Api, error_code: ErrorCode, message: [:0]const u8) *Error {
        var ret = pjrtStruct(c.XLA_FFI_Error_Create_Args{
            .message = message.ptr,
            .errc = @intFromEnum(error_code),
        });
        return fromInner(api.inner().XLA_FFI_Error_Create.?(&ret).?);
    }

    pub fn destroy(err: *Error, api: *const Api) void {
        var ret = pjrtStruct(c.XLA_FFI_Error_Destroy_Args{ .@"error" = err.inner() });
        api.inner().XLA_FFI_Error_Destroy.?(&ret);
    }

    pub fn getMessage(err: *Error, api: *const Api) [:0]const u8 {
        var ret = pjrtStruct(c.XLA_FFI_Error_GetMessage_Args{
            .@"error" = err.inner(),
        });
        api.inner().XLA_FFI_Error_GetMessage.?(&ret);
        return std.mem.span(ret.message);
    }
};

pub const Future = opaque {
    pub const inner = TransmuteMixin(Future, c.XLA_FFI_Future).to;
    pub const fromInner = TransmuteMixin(Future, c.XLA_FFI_Future).from;

    pub fn create(api: *const Api) ApiError!*Future {
        var ret = pjrtStruct(c.XLA_FFI_Future_Create_Args{});
        const result = api.inner().XLA_FFI_Future_Create.?(&ret);

        if (result) |ffi_error| {
            const err = Error.fromInner(ffi_error);
            defer err.destroy(api);
            log.err("[Future.create] {s}", .{err.getMessage(api)});

            // TODO(Corentin): Retrieve error code from Error when implemented in XLA.
            return error.Unknown;
        }

        return fromInner(ret.future.?);
    }

    pub fn setAvailable(self: *Future, api: *const Api) ApiError!void {
        var ret = pjrtStruct(c.XLA_FFI_Future_SetAvailable_Args{
            .future = self.inner(),
        });

        const result = api.inner().XLA_FFI_Future_SetAvailable.?(&ret);

        if (result) |ffi_error| {
            const err = Error.fromInner(ffi_error);
            defer err.destroy(api);
            log.err("[Future.setAvailable] {s}", .{err.getMessage(api)});

            // TODO(Corentin): Retrieve error code from Error when implemented in XLA.
            return error.Unknown;
        }
    }

    pub fn setError(self: *Future, api: *const Api, err: *Error) ApiError!void {
        var ret = pjrtStruct(c.XLA_FFI_Future_SetError_Args{
            .future = self.inner(),
            .@"error" = err.inner(),
        });

        const result = api.inner().XLA_FFI_Future_SetError.?(&ret);

        if (result) |ffi_error| {
            const err2 = Error.fromInner(ffi_error);
            defer err2.destroy(api);
            log.err("[Future.setError] {s}", .{err2.getMessage(api)});

            // TODO(Corentin): Retrieve error code from Error when implemented in XLA.
            return error.Unknown;
        }
    }
};
