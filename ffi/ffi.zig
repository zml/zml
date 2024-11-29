const std = @import("std");
const stdx = @import("stdx");

pub const c = @import("c");

const log = std.log.scoped(.ffi);

fn ffiStructSize(comptime T: type) usize {
    // unsafe on purpose, we want this to fail if that ever changes
    const typedef_name = comptime blk: {
        const needle = ".struct_";
        const idx = std.mem.indexOf(u8, @typeName(T), needle).?;
        break :blk @typeName(T)[idx + needle.len ..];
    };
    return @field(c, typedef_name ++ "_STRUCT_SIZE");
}

pub inline fn ffiStruct(v: anytype) @TypeOf(v) {
    var ret = v;
    ret.struct_size = ffiStructSize(@TypeOf(v));
    return ret;
}

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

// TODO(Corentin): This is actually a bit set.
pub const HandlerTraits = u32;

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
        var ret = ffiStruct(c.XLA_FFI_Stream_Get_Args{
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
        var ret = ffiStruct(c.XLA_FFI_DeviceMemory_Allocate_Args{
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
        var ret = ffiStruct(c.XLA_FFI_DeviceMemory_Free_Args{
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

    pub fn get(self: *ExecutionContext, api: *const Api, type_id: *TypeId) ApiError!*anyopaque {
        var ret = ffiStruct(c.XLA_FFI_ExecutionContext_Get_Args{
            .ctx = self.inner(),
            .type_id = @ptrCast(@alignCast(type_id)),
        });
        const result = api.inner().XLA_FFI_ExecutionContext_Get.?(&ret);

        if (result) |ffi_error| {
            const err = Error.fromInner(ffi_error);
            defer err.destroy(self);
            log.err("[ExecutionContext.get] {s}", .{err.getMessage(self)});

            // TODO(Corentin): Retrieve error code from Error when implemented in XLA.
            return error.Unknown;
        }

        return ret.data.?;
    }
};

pub const TypeId = extern struct {
    type_id: i64,
};

pub const ArgType = enum(c.XLA_FFI_ArgType) {
    buffer = c.XLA_FFI_ArgType_BUFFER,
};

pub const RetType = enum(c.XLA_FFI_RetType) {
    buffer = c.XLA_FFI_RetType_BUFFER,
};

pub const Arg = union(ArgType) {
    buffer: Buffer,
};

pub const Ret = union(ArgType) {
    buffer: Buffer,
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
    data: ?*anyopaque,
    rank: i64,
    dims: [*]i64,
};

pub const Args = extern struct {
    struct_size: usize,
    extension_start: ?*c.XLA_FFI_Extension_Base,
    size: i64,
    types: [*]ArgType,
    args: [*]?*anyopaque,

    pub fn getArgAs(self: Args, comptime T: type, index: usize) *T {
        if (index > @as(usize, @intCast(self.size))) @panic("Index out of bound");
        return @ptrCast(@alignCast(self.args[index].?));
    }
};

pub const Rets = extern struct {
    struct_size: usize,
    extension_start: ?*c.XLA_FFI_Extension_Base,
    size: i64,
    types: [*]ArgType,
    rets: [*]?*anyopaque,

    pub fn getRetAs(self: Rets, comptime T: type, index: usize) *T {
        if (index > @as(usize, @intCast(self.size))) @panic("Index out of bound");
        return @ptrCast(@alignCast(self.rets[index].?));
    }
};

pub const AttrType = enum(c.XLA_FFI_AttrType) {
    array = c.XLA_FFI_AttrType_ARRAY,
    dictionary = c.XLA_FFI_AttrType_DICTIONARY,
    scalar = c.XLA_FFI_AttrType_SCALAR,
    string = c.XLA_FFI_AttrType_STRING,
};

pub const ByteSpan = extern struct {
    ptr: [*]const u8,
    len: usize,

    pub fn slice(self: ByteSpan) []const u8 {
        return self.ptr[0..self.len];
    }
};

pub const Scalar = extern struct {
    dtype: DataType,
    value: *anyopaque,
};

pub const Array = extern struct {
    dtype: DataType,
    size: usize,
    data: [*]anyopaque,
};

pub const Attrs = extern struct {
    struct_size: usize,
    extension_start: ?*ExtensionBase,
    size: i64,
    types: [*]AttrType,
    names: [*]*ByteSpan,
    attrs: [*]?*anyopaque,

    pub fn getAttrAs(self: Attrs, comptime T: type, index: usize) *T {
        if (index > @as(usize, @intCast(self.size))) @panic("Index out of bound");
        return @ptrCast(@alignCast(self.attrs[index].?));
    }

    pub fn getName(self: Attrs, index: usize) []const u8 {
        if (index > @as(usize, @intCast(self.size))) @panic("Index out of bound");
        const span = self.names[index].*;
        return span.ptr[0..span.len];
    }

    pub fn getAttrByNameAs(self: Attrs, comptime T: type, name: []const u8) ?*T {
        const index: usize = for (0..@intCast(self.size)) |index| {
            const attr_name = self.names[index].slice();
            if (std.mem.eql(u8, attr_name, name)) break index;
        } else return null;

        return self.getAttrAs(T, index);
    }
};

pub const CallFrame = extern struct {
    struct_size: usize,
    extension_start: ?*ExtensionBase,
    api: ?*const Api,
    ctx: ?*ExecutionContext,
    stage: ExecutionStage,
    args: Args,
    rets: Rets,
    attrs: Attrs,
    future: ?*Future,
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
    pub const inner = TransmuteMixin(Error, c.XLA_FFI_Error).to;
    pub const fromInner = TransmuteMixin(Error, c.XLA_FFI_Error).from;

    pub fn create(api: *const Api, error_code: ErrorCode, message: [:0]const u8) *Error {
        var ret = ffiStruct(c.XLA_FFI_Error_Create_Args{
            .message = message.ptr,
            .errc = @intFromEnum(error_code),
        });
        return fromInner(api.inner().XLA_FFI_Error_Create.?(&ret).?);
    }

    pub fn destroy(err: *Error, api: *const Api) void {
        var ret = ffiStruct(c.XLA_FFI_Error_Destroy_Args{ .@"error" = err.inner() });
        api.inner().XLA_FFI_Error_Destroy.?(&ret);
    }

    pub fn getMessage(err: *Error, api: *const Api) [:0]const u8 {
        var ret = ffiStruct(c.XLA_FFI_Error_GetMessage_Args{
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
        var ret = ffiStruct(c.XLA_FFI_Future_Create_Args{});
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
        var ret = ffiStruct(c.XLA_FFI_Future_SetAvailable_Args{
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
        var ret = ffiStruct(c.XLA_FFI_Future_SetError_Args{
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
