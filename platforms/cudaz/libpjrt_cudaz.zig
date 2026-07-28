const std = @import("std");

const c = @import("c");

const Handle = struct {
    tag: usize,
};

const Error = struct {
    base: c.PJRT_Error,
    code: c.PJRT_Error_Code,
    message: []const u8,
};

const platform_name = "cudaz";
const device_kind = "CUDA";
const device_debug_string = "cudaz device 0";
const device_string = "CudazDevice(id=0)";
const memory_kind = "device";
const memory_debug_string = "cudaz device memory 0";
const memory_string = "CudazDeviceMemory(id=0)";
const device_coords = [_]i64{0};

const error_vtable: c.PJRT_Error_FunctionTable = .{
    .struct_size = c.PJRT_Error_FunctionTable_STRUCT_SIZE,
    .instance_size = @sizeOf(Error),
    .extension_start = null,
    .destroy = &errorVtableDestroy,
    .message = &errorVtableMessage,
    .get_code = &errorVtableGetCode,
    .for_each_payload = &errorVtableForEachPayload,
};

var client: Handle = .{ .tag = 0 };
var device: Handle = .{ .tag = 0 };
var device_description: Handle = .{ .tag = 0 };
var memory: c.PJRT_Memory = .{ .vtable = null };
var addressable_devices: [1]?*c.PJRT_Device = .{@ptrCast(&device)};
var addressable_memories: [1][*c]c.PJRT_Memory = .{&memory};
var device_attributes: [1]c.PJRT_NamedValue = .{.{
    .struct_size = c.PJRT_NamedValue_STRUCT_SIZE,
    .extension_start = null,
    .name = "coords",
    .name_size = "coords".len,
    .type = c.PJRT_NamedValue_kInt64List,
    .unnamed_0 = .{ .int64_array_value = &device_coords },
    .value_size = device_coords.len,
}};
var compile_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_UNIMPLEMENTED,
    .message = "cudaz: MLIR compilation is not implemented",
};

fn errorFromPjrt(pjrt_error: anytype) *@TypeOf(compile_error) {
    return @ptrCast(@alignCast(@constCast(pjrt_error)));
}

fn errorVtableDestroy(_: [*c]c.PJRT_Error) callconv(.c) void {}

fn errorVtableMessage(
    pjrt_error: [*c]const c.PJRT_Error,
    message: [*c][*c]const u8,
    message_size: [*c]usize,
) callconv(.c) void {
    const err = errorFromPjrt(pjrt_error);
    message.* = err.message.ptr;
    message_size.* = err.message.len;
}

fn errorVtableGetCode(pjrt_error: [*c]const c.PJRT_Error) callconv(.c) c.PJRT_Error_Code {
    return errorFromPjrt(pjrt_error).code;
}

fn errorVtableForEachPayload(
    _: [*c]const c.PJRT_Error,
    _: c.PJRT_Error_PayloadVisitor,
    _: ?*anyopaque,
) callconv(.c) void {}

fn errorDestroy(_: [*c]c.PJRT_Error_Destroy_Args) callconv(.c) void {}

fn errorMessage(args: [*c]c.PJRT_Error_Message_Args) callconv(.c) void {
    const pjrt_error = errorFromPjrt(args.*.@"error");
    args.*.message = pjrt_error.message.ptr;
    args.*.message_size = pjrt_error.message.len;
}

fn errorGetCode(args: [*c]c.PJRT_Error_GetCode_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.code = errorFromPjrt(args.*.@"error").code;
    return null;
}

fn pluginInitialize(_: [*c]c.PJRT_Plugin_Initialize_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn pluginAttributes(args: [*c]c.PJRT_Plugin_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.attributes = null;
    args.*.num_attributes = 0;
    return null;
}

fn clientCreate(args: [*c]c.PJRT_Client_Create_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.client = @ptrCast(&client);
    return null;
}

fn clientDestroy(_: [*c]c.PJRT_Client_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn clientPlatformName(args: [*c]c.PJRT_Client_PlatformName_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.platform_name = platform_name.ptr;
    args.*.platform_name_size = platform_name.len;
    return null;
}

fn clientAddressableDevices(args: [*c]c.PJRT_Client_AddressableDevices_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.addressable_devices = @ptrCast(&addressable_devices);
    args.*.num_addressable_devices = addressable_devices.len;
    return null;
}

fn clientAddressableMemories(args: [*c]c.PJRT_Client_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.addressable_memories = &addressable_memories;
    args.*.num_addressable_memories = addressable_memories.len;
    return null;
}

fn clientCompile(_: [*c]c.PJRT_Client_Compile_Args) callconv(.c) ?*c.PJRT_Error {
    return @ptrCast(&compile_error);
}

fn deviceDescriptionId(args: [*c]c.PJRT_DeviceDescription_Id_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.id = 0;
    return null;
}

fn deviceDescriptionProcessIndex(args: [*c]c.PJRT_DeviceDescription_ProcessIndex_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.process_index = 0;
    return null;
}

fn deviceDescriptionAttributes(args: [*c]c.PJRT_DeviceDescription_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.attributes = &device_attributes;
    args.*.num_attributes = device_attributes.len;
    return null;
}

fn deviceDescriptionKind(args: [*c]c.PJRT_DeviceDescription_Kind_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.device_kind = device_kind.ptr;
    args.*.device_kind_size = device_kind.len;
    return null;
}

fn deviceDescriptionDebugString(args: [*c]c.PJRT_DeviceDescription_DebugString_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.debug_string = device_debug_string.ptr;
    args.*.debug_string_size = device_debug_string.len;
    return null;
}

fn deviceDescriptionToString(args: [*c]c.PJRT_DeviceDescription_ToString_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.to_string = device_string.ptr;
    args.*.to_string_size = device_string.len;
    return null;
}

fn deviceGetDescription(args: [*c]c.PJRT_Device_GetDescription_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.device_description = @ptrCast(&device_description);
    return null;
}

fn deviceLocalHardwareId(args: [*c]c.PJRT_Device_LocalHardwareId_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.local_hardware_id = 0;
    return null;
}

fn deviceAddressableMemories(args: [*c]c.PJRT_Device_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.memories = &addressable_memories;
    args.*.num_memories = addressable_memories.len;
    return null;
}

fn deviceDefaultMemory(args: [*c]c.PJRT_Device_DefaultMemory_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.memory = &memory;
    return null;
}

fn memoryId(args: [*c]c.PJRT_Memory_Id_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.id = 0;
    return null;
}

fn memoryKind(args: [*c]c.PJRT_Memory_Kind_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.kind = memory_kind.ptr;
    args.*.kind_size = memory_kind.len;
    return null;
}

fn memoryKindId(args: [*c]c.PJRT_Memory_Kind_Id_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.kind_id = 0;
    return null;
}

fn memoryDebugString(args: [*c]c.PJRT_Memory_DebugString_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.debug_string = memory_debug_string.ptr;
    args.*.debug_string_size = memory_debug_string.len;
    return null;
}

fn memoryToString(args: [*c]c.PJRT_Memory_ToString_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.to_string = memory_string.ptr;
    args.*.to_string_size = memory_string.len;
    return null;
}

fn memoryAddressableByDevices(args: [*c]c.PJRT_Memory_AddressableByDevices_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.devices = &addressable_devices;
    args.*.num_devices = addressable_devices.len;
    return null;
}

fn makeApi() c.PJRT_Api {
    var result = std.mem.zeroes(c.PJRT_Api);
    result.struct_size = c.PJRT_Api_STRUCT_SIZE;
    result.pjrt_api_version = .{
        .struct_size = c.PJRT_Api_Version_STRUCT_SIZE,
        .extension_start = null,
        .major_version = c.PJRT_API_MAJOR,
        .minor_version = c.PJRT_API_MINOR,
    };
    result.PJRT_Error_Destroy = &errorDestroy;
    result.PJRT_Error_Message = &errorMessage;
    result.PJRT_Error_GetCode = &errorGetCode;
    result.PJRT_Plugin_Initialize = &pluginInitialize;
    result.PJRT_Plugin_Attributes = &pluginAttributes;
    result.PJRT_Client_Create = &clientCreate;
    result.PJRT_Client_Destroy = &clientDestroy;
    result.PJRT_Client_PlatformName = &clientPlatformName;
    result.PJRT_Client_AddressableDevices = &clientAddressableDevices;
    result.PJRT_Client_AddressableMemories = &clientAddressableMemories;
    result.PJRT_Client_Compile = &clientCompile;
    result.PJRT_DeviceDescription_Id = &deviceDescriptionId;
    result.PJRT_DeviceDescription_ProcessIndex = &deviceDescriptionProcessIndex;
    result.PJRT_DeviceDescription_Attributes = &deviceDescriptionAttributes;
    result.PJRT_DeviceDescription_Kind = &deviceDescriptionKind;
    result.PJRT_DeviceDescription_DebugString = &deviceDescriptionDebugString;
    result.PJRT_DeviceDescription_ToString = &deviceDescriptionToString;
    result.PJRT_Device_GetDescription = &deviceGetDescription;
    result.PJRT_Device_LocalHardwareId = &deviceLocalHardwareId;
    result.PJRT_Device_AddressableMemories = &deviceAddressableMemories;
    result.PJRT_Device_DefaultMemory = &deviceDefaultMemory;
    result.PJRT_Memory_Id = &memoryId;
    result.PJRT_Memory_Kind = &memoryKind;
    result.PJRT_Memory_Kind_Id = &memoryKindId;
    result.PJRT_Memory_DebugString = &memoryDebugString;
    result.PJRT_Memory_ToString = &memoryToString;
    result.PJRT_Memory_AddressableByDevices = &memoryAddressableByDevices;
    return result;
}

const api = makeApi();

pub export fn GetPjrtApi() *const c.PJRT_Api {
    return &api;
}
