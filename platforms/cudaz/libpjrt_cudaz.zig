const std = @import("std");

const c = @import("c");
const cuda = @import("cuda.zig");

const Handle = struct {
    tag: usize,
};

const Error = struct {
    base: c.PJRT_Error,
    code: c.PJRT_Error_Code,
    message: []const u8,
};

const Memory = extern struct {
    base: c.PJRT_Memory,
    client: *Client,
};

const Client = struct {
    cuda: cuda.Client,
    memory: Memory,
    loaded_executable: Handle,
    addressable_devices: [1]?*c.PJRT_Device,
    addressable_memories: [1][*c]c.PJRT_Memory,

    fn create() !*Client {
        const self = try std.heap.page_allocator.create(Client);
        errdefer std.heap.page_allocator.destroy(self);

        self.cuda = try .init();
        self.memory = .{
            .base = .{ .vtable = null },
            .client = self,
        };
        self.loaded_executable = .{ .tag = 0 };
        self.addressable_devices = .{@ptrCast(&self.cuda.device)};
        self.addressable_memories = .{@ptrCast(&self.memory)};
        return self;
    }

    fn destroy(self: *Client) void {
        self.cuda.deinit();
        std.heap.page_allocator.destroy(self);
    }

    fn fromPjrt(pjrt_client: ?*c.PJRT_Client) *Client {
        return @ptrCast(@alignCast(pjrt_client.?));
    }

    fn fromDevice(pjrt_device: ?*c.PJRT_Device) *Client {
        const cuda_device: *cuda.Device = @ptrCast(@alignCast(pjrt_device.?));
        const cuda_client: *cuda.Client = @fieldParentPtr("device", cuda_device);
        return @fieldParentPtr("cuda", cuda_client);
    }
};

const Buffer = struct {
    client: *Client,
    allocation: cuda.Allocation,
    element_type: c.PJRT_Buffer_Type,
    dims: []i64,

    fn destroy(self: *Buffer) void {
        self.client.cuda.free(self.allocation);
        std.heap.page_allocator.free(self.dims);
        std.heap.page_allocator.destroy(self);
    }

    fn fromPjrt(pjrt_buffer: ?*c.PJRT_Buffer) *Buffer {
        return @ptrCast(@alignCast(pjrt_buffer.?));
    }
};

const TransferManager = struct {
    client: *Client,
    buffers: []*Buffer,
    retrieved: []bool,

    fn destroy(self: *TransferManager) void {
        for (self.buffers, self.retrieved) |buffer, was_retrieved| {
            if (!was_retrieved) buffer.destroy();
        }
        std.heap.page_allocator.free(self.retrieved);
        std.heap.page_allocator.free(self.buffers);
        std.heap.page_allocator.destroy(self);
    }

    fn fromPjrt(pjrt_manager: ?*c.PJRT_AsyncHostToDeviceTransferManager) *TransferManager {
        return @ptrCast(@alignCast(pjrt_manager.?));
    }
};

const Event = struct {
    tag: usize,

    fn fromPjrt(pjrt_event: ?*c.PJRT_Event) *Event {
        return @ptrCast(@alignCast(pjrt_event.?));
    }
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

var device_attributes: [1]c.PJRT_NamedValue = .{.{
    .struct_size = c.PJRT_NamedValue_STRUCT_SIZE,
    .extension_start = null,
    .name = "coords",
    .name_size = "coords".len,
    .type = c.PJRT_NamedValue_kInt64List,
    .unnamed_0 = .{ .int64_array_value = &device_coords },
    .value_size = device_coords.len,
}};
var invalid_argument_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_INVALID_ARGUMENT,
    .message = "cudaz: invalid PJRT argument",
};
var resource_exhausted_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_RESOURCE_EXHAUSTED,
    .message = "cudaz: host allocation failed",
};
var cuda_driver_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_UNAVAILABLE,
    .message = "cudaz: CUDA driver is unavailable",
};
var cuda_initialization_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_UNAVAILABLE,
    .message = "cudaz: CUDA initialization failed",
};
var cuda_device_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_UNAVAILABLE,
    .message = "cudaz: CUDA device 0 is unavailable",
};
var cuda_context_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_INTERNAL,
    .message = "cudaz: CUDA context operation failed",
};
var cuda_stream_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_INTERNAL,
    .message = "cudaz: CUDA stream operation failed",
};
var cuda_host_registration_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_RESOURCE_EXHAUSTED,
    .message = "cudaz: CUDA host-memory registration failed",
};
var cuda_allocation_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_RESOURCE_EXHAUSTED,
    .message = "cudaz: CUDA device allocation failed",
};
var cuda_copy_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_INTERNAL,
    .message = "cudaz: CUDA host-to-device copy failed",
};

fn errorFromPjrt(pjrt_error: anytype) *Error {
    return @ptrCast(@alignCast(@constCast(pjrt_error)));
}

fn errorToPjrt(err: anyerror) ?*c.PJRT_Error {
    const pjrt_error: *Error = switch (err) {
        error.OutOfMemory => &resource_exhausted_error,
        error.InvalidArgument => &invalid_argument_error,
        error.DriverUnavailable, error.MissingDriverSymbol => &cuda_driver_error,
        error.InitializationFailed => &cuda_initialization_error,
        error.DeviceUnavailable => &cuda_device_error,
        error.ContextFailed => &cuda_context_error,
        error.StreamFailed => &cuda_stream_error,
        error.HostRegistrationFailed => &cuda_host_registration_error,
        error.DeviceAllocationFailed => &cuda_allocation_error,
        error.HostToDeviceCopyFailed => &cuda_copy_error,
        else => &invalid_argument_error,
    };
    return @ptrCast(pjrt_error);
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
    const client = Client.create() catch |err| return errorToPjrt(err);
    args.*.client = @ptrCast(client);
    return null;
}

fn clientDestroy(args: [*c]c.PJRT_Client_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args.*.client) |pjrt_client| Client.fromPjrt(pjrt_client).destroy();
    return null;
}

fn clientPlatformName(args: [*c]c.PJRT_Client_PlatformName_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.platform_name = platform_name.ptr;
    args.*.platform_name_size = platform_name.len;
    return null;
}

fn clientAddressableDevices(args: [*c]c.PJRT_Client_AddressableDevices_Args) callconv(.c) ?*c.PJRT_Error {
    const client = Client.fromPjrt(args.*.client);
    args.*.addressable_devices = &client.addressable_devices;
    args.*.num_addressable_devices = client.addressable_devices.len;
    return null;
}

fn clientAddressableMemories(args: [*c]c.PJRT_Client_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
    const client = Client.fromPjrt(args.*.client);
    args.*.addressable_memories = &client.addressable_memories;
    args.*.num_addressable_memories = client.addressable_memories.len;
    return null;
}

fn clientCompile(args: [*c]c.PJRT_Client_Compile_Args) callconv(.c) ?*c.PJRT_Error {
    const client = Client.fromPjrt(args.*.client);
    args.*.executable = @ptrCast(&client.loaded_executable);
    return null;
}

fn clientDmaMap(args: [*c]c.PJRT_Client_DmaMap_Args) callconv(.c) ?*c.PJRT_Error {
    if (args.*.size > 0 and args.*.data == null) return errorToPjrt(error.InvalidArgument);
    const data: []u8 = if (args.*.size == 0)
        &.{}
    else
        @as([*]u8, @ptrCast(args.*.data.?))[0..args.*.size];
    Client.fromPjrt(args.*.client).cuda.registerHostMemory(data) catch |err| return errorToPjrt(err);
    return null;
}

fn clientDmaUnmap(args: [*c]c.PJRT_Client_DmaUnmap_Args) callconv(.c) ?*c.PJRT_Error {
    const data_ptr = args.*.data orelse return errorToPjrt(error.InvalidArgument);
    Client.fromPjrt(args.*.client).cuda.unregisterHostMemory(data_ptr) catch |err| return errorToPjrt(err);
    return null;
}

fn clientBufferFromHostBuffer(args: [*c]c.PJRT_Client_BufferFromHostBuffer_Args) callconv(.c) ?*c.PJRT_Error {
    if (args.*.num_dims > 0 and args.*.dims == null) return errorToPjrt(error.InvalidArgument);
    validateCompactByteStrides(
        args.*.type,
        args.*.dims,
        args.*.num_dims,
        args.*.byte_strides,
        args.*.num_byte_strides,
    ) catch |err| return errorToPjrt(err);

    const client = Client.fromPjrt(args.*.client);
    if (args.*.memory != null) {
        const memory: *Memory = @ptrCast(@alignCast(args.*.memory));
        if (memory.client != client) return errorToPjrt(error.InvalidArgument);
    } else if (args.*.device) |pjrt_device| {
        if (Client.fromDevice(pjrt_device) != client) return errorToPjrt(error.InvalidArgument);
    } else {
        return errorToPjrt(error.InvalidArgument);
    }

    const shape: c.PJRT_ShapeSpec = .{
        .struct_size = c.PJRT_ShapeSpec_STRUCT_SIZE,
        .extension_start = null,
        .dims = args.*.dims,
        .num_dims = args.*.num_dims,
        .element_type = args.*.type,
    };
    const buffer = createBuffer(client, &shape) catch |err| return errorToPjrt(err);

    if (buffer.allocation.size > 0 and args.*.data == null) {
        buffer.destroy();
        return errorToPjrt(error.InvalidArgument);
    }
    const data: []const u8 = if (buffer.allocation.size == 0)
        &.{}
    else
        @as([*]const u8, @ptrCast(args.*.data.?))[0..buffer.allocation.size];
    client.cuda.copyHostToDevice(buffer.allocation, 0, data) catch |err| {
        buffer.destroy();
        return errorToPjrt(err);
    };

    const event = createReadyEvent() catch {
        buffer.destroy();
        return errorToPjrt(error.OutOfMemory);
    };
    args.*.buffer = @ptrCast(buffer);
    args.*.done_with_host_buffer = @ptrCast(event);
    return null;
}

fn elementSize(element_type: c.PJRT_Buffer_Type) error{InvalidArgument}!usize {
    return switch (element_type) {
        c.PJRT_Buffer_Type_PRED,
        c.PJRT_Buffer_Type_S1,
        c.PJRT_Buffer_Type_U1,
        c.PJRT_Buffer_Type_S2,
        c.PJRT_Buffer_Type_U2,
        c.PJRT_Buffer_Type_S4,
        c.PJRT_Buffer_Type_U4,
        c.PJRT_Buffer_Type_S8,
        c.PJRT_Buffer_Type_U8,
        c.PJRT_Buffer_Type_F8E5M2,
        c.PJRT_Buffer_Type_F8E4M3FN,
        c.PJRT_Buffer_Type_F8E4M3B11FNUZ,
        c.PJRT_Buffer_Type_F8E5M2FNUZ,
        c.PJRT_Buffer_Type_F8E4M3FNUZ,
        c.PJRT_Buffer_Type_F8E4M3,
        c.PJRT_Buffer_Type_F8E3M4,
        c.PJRT_Buffer_Type_F8E8M0FNU,
        c.PJRT_Buffer_Type_F4E2M1FN,
        => 1,
        c.PJRT_Buffer_Type_S16,
        c.PJRT_Buffer_Type_U16,
        c.PJRT_Buffer_Type_F16,
        c.PJRT_Buffer_Type_BF16,
        => 2,
        c.PJRT_Buffer_Type_S32,
        c.PJRT_Buffer_Type_U32,
        c.PJRT_Buffer_Type_F32,
        => 4,
        c.PJRT_Buffer_Type_S64,
        c.PJRT_Buffer_Type_U64,
        c.PJRT_Buffer_Type_F64,
        c.PJRT_Buffer_Type_C64,
        => 8,
        c.PJRT_Buffer_Type_C128 => 16,
        else => error.InvalidArgument,
    };
}

fn shapeByteSize(shape: *const c.PJRT_ShapeSpec) error{InvalidArgument}!usize {
    if (shape.num_dims > 0 and shape.dims == null) return error.InvalidArgument;

    var element_count: usize = 1;
    if (shape.num_dims > 0) {
        for (shape.dims[0..shape.num_dims]) |dim| {
            if (dim < 0) return error.InvalidArgument;
            element_count = std.math.mul(usize, element_count, @intCast(dim)) catch return error.InvalidArgument;
        }
    }
    return std.math.mul(usize, element_count, try elementSize(shape.element_type)) catch error.InvalidArgument;
}

fn validateCompactByteStrides(
    element_type: c.PJRT_Buffer_Type,
    dims_ptr: [*c]const i64,
    num_dims: usize,
    strides_ptr: [*c]const i64,
    num_strides: usize,
) error{InvalidArgument}!void {
    if (num_strides == 0) return;
    if (num_strides != num_dims or dims_ptr == null or strides_ptr == null) return error.InvalidArgument;

    const dims = dims_ptr[0..num_dims];
    const strides = strides_ptr[0..num_strides];
    var expected_stride = try elementSize(element_type);
    var axis = num_dims;
    while (axis > 0) {
        axis -= 1;
        if (dims[axis] < 0 or strides[axis] < 0) return error.InvalidArgument;
        if (strides[axis] != expected_stride) return error.InvalidArgument;
        expected_stride = std.math.mul(usize, expected_stride, @intCast(dims[axis])) catch return error.InvalidArgument;
    }
}

test "shape byte size follows PJRT element type and dimensions" {
    const dims = [_]i64{ 3, 5 };
    const shape: c.PJRT_ShapeSpec = .{
        .struct_size = c.PJRT_ShapeSpec_STRUCT_SIZE,
        .extension_start = null,
        .dims = &dims,
        .num_dims = dims.len,
        .element_type = c.PJRT_Buffer_Type_F32,
    };
    try std.testing.expectEqual(60, shapeByteSize(&shape));
}

test "shape byte size handles scalars and rejects negative dimensions" {
    var shape: c.PJRT_ShapeSpec = .{
        .struct_size = c.PJRT_ShapeSpec_STRUCT_SIZE,
        .extension_start = null,
        .dims = null,
        .num_dims = 0,
        .element_type = c.PJRT_Buffer_Type_C128,
    };
    try std.testing.expectEqual(16, shapeByteSize(&shape));

    const invalid_dims = [_]i64{ 3, -1 };
    shape.dims = &invalid_dims;
    shape.num_dims = invalid_dims.len;
    try std.testing.expectError(error.InvalidArgument, shapeByteSize(&shape));
}

test "host byte strides must describe compact row-major data" {
    const dims = [_]i64{ 3, 5 };
    const compact_strides = [_]i64{ 20, 4 };
    try validateCompactByteStrides(
        c.PJRT_Buffer_Type_F32,
        &dims,
        dims.len,
        &compact_strides,
        compact_strides.len,
    );

    const padded_strides = [_]i64{ 24, 4 };
    try std.testing.expectError(
        error.InvalidArgument,
        validateCompactByteStrides(
            c.PJRT_Buffer_Type_F32,
            &dims,
            dims.len,
            &padded_strides,
            padded_strides.len,
        ),
    );
}

fn createBuffer(client: *Client, shape: *const c.PJRT_ShapeSpec) !*Buffer {
    const allocation = try client.cuda.allocate(try shapeByteSize(shape));
    errdefer client.cuda.free(allocation);

    const dims = if (shape.num_dims == 0)
        try std.heap.page_allocator.alloc(i64, 0)
    else
        try std.heap.page_allocator.dupe(i64, shape.dims[0..shape.num_dims]);
    errdefer std.heap.page_allocator.free(dims);

    const buffer = try std.heap.page_allocator.create(Buffer);
    buffer.* = .{
        .client = client,
        .allocation = allocation,
        .element_type = shape.element_type,
        .dims = dims,
    };
    return buffer;
}

fn createReadyEvent() !*Event {
    const event = try std.heap.page_allocator.create(Event);
    event.* = .{ .tag = 0 };
    return event;
}

fn createTransferManager(client: *Client, shapes: []const c.PJRT_ShapeSpec) !*TransferManager {
    if (shapes.len == 0) return error.InvalidArgument;

    const manager = try std.heap.page_allocator.create(TransferManager);
    errdefer std.heap.page_allocator.destroy(manager);

    const buffers = try std.heap.page_allocator.alloc(*Buffer, shapes.len);
    errdefer std.heap.page_allocator.free(buffers);

    const retrieved = try std.heap.page_allocator.alloc(bool, shapes.len);
    errdefer std.heap.page_allocator.free(retrieved);
    @memset(retrieved, false);

    var initialized: usize = 0;
    errdefer for (buffers[0..initialized]) |buffer| buffer.destroy();

    for (shapes, buffers) |*shape, *buffer| {
        buffer.* = try createBuffer(client, shape);
        initialized += 1;
    }

    manager.* = .{
        .client = client,
        .buffers = buffers,
        .retrieved = retrieved,
    };
    return manager;
}

fn clientCreateBuffersForAsyncHostToDevice(
    args: [*c]c.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
) callconv(.c) ?*c.PJRT_Error {
    if (args.*.num_shape_specs == 0 or args.*.shape_specs == null) return errorToPjrt(error.InvalidArgument);

    const client = Client.fromPjrt(args.*.client);
    const pjrt_memory = args.*.memory;
    if (pjrt_memory == null) return errorToPjrt(error.InvalidArgument);
    const memory: *Memory = @ptrCast(@alignCast(pjrt_memory));
    if (memory.client != client) return errorToPjrt(error.InvalidArgument);

    const manager = createTransferManager(client, args.*.shape_specs[0..args.*.num_shape_specs]) catch |err| return errorToPjrt(err);
    args.*.transfer_manager = @ptrCast(manager);
    return null;
}

fn transferManagerDestroy(
    args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args,
) callconv(.c) ?*c.PJRT_Error {
    if (args.*.transfer_manager) |pjrt_manager| TransferManager.fromPjrt(pjrt_manager).destroy();
    return null;
}

fn transferManagerRetrieveBuffer(
    args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args,
) callconv(.c) ?*c.PJRT_Error {
    const manager = TransferManager.fromPjrt(args.*.transfer_manager);
    if (args.*.buffer_index < 0) return errorToPjrt(error.InvalidArgument);
    const index: usize = @intCast(args.*.buffer_index);
    if (index >= manager.buffers.len) return errorToPjrt(error.InvalidArgument);
    if (manager.retrieved[index]) return errorToPjrt(error.InvalidArgument);

    manager.retrieved[index] = true;
    args.*.buffer_out = @ptrCast(manager.buffers[index]);
    return null;
}

fn transferManagerTransferData(
    args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args,
) callconv(.c) ?*c.PJRT_Error {
    const manager = TransferManager.fromPjrt(args.*.transfer_manager);
    if (args.*.buffer_index < 0 or args.*.offset < 0 or args.*.transfer_size < 0) return errorToPjrt(error.InvalidArgument);

    const index: usize = @intCast(args.*.buffer_index);
    const offset: usize = @intCast(args.*.offset);
    const transfer_size: usize = @intCast(args.*.transfer_size);
    if (index >= manager.buffers.len) return errorToPjrt(error.InvalidArgument);
    if (transfer_size > 0 and args.*.data == null) return errorToPjrt(error.InvalidArgument);

    const data: []const u8 = if (transfer_size == 0)
        &.{}
    else
        @as([*]const u8, @ptrCast(args.*.data.?))[0..transfer_size];
    manager.client.cuda.copyHostToDevice(manager.buffers[index].allocation, offset, data) catch |err| return errorToPjrt(err);

    const event = createReadyEvent() catch return errorToPjrt(error.OutOfMemory);
    args.*.done_with_h2d_transfer = @ptrCast(event);
    return null;
}

fn transferManagerDevice(
    args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_Device_Args,
) callconv(.c) ?*c.PJRT_Error {
    const manager = TransferManager.fromPjrt(args.*.transfer_manager);
    args.*.device_out = @ptrCast(&manager.client.cuda.device);
    return null;
}

fn transferManagerBufferCount(
    args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args,
) callconv(.c) ?*c.PJRT_Error {
    const manager = TransferManager.fromPjrt(args.*.transfer_manager);
    args.*.buffer_count = manager.buffers.len;
    return null;
}

fn transferManagerBufferSize(
    args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args,
) callconv(.c) ?*c.PJRT_Error {
    const manager = TransferManager.fromPjrt(args.*.transfer_manager);
    if (args.*.buffer_index < 0) return errorToPjrt(error.InvalidArgument);
    const index: usize = @intCast(args.*.buffer_index);
    if (index >= manager.buffers.len) return errorToPjrt(error.InvalidArgument);
    args.*.buffer_size = manager.buffers[index].allocation.size;
    return null;
}

fn eventDestroy(args: [*c]c.PJRT_Event_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args.*.event) |pjrt_event| std.heap.page_allocator.destroy(Event.fromPjrt(pjrt_event));
    return null;
}

fn eventIsReady(args: [*c]c.PJRT_Event_IsReady_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.is_ready = true;
    return null;
}

fn eventError(_: [*c]c.PJRT_Event_Error_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn eventAwait(_: [*c]c.PJRT_Event_Await_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn eventOnReady(args: [*c]c.PJRT_Event_OnReady_Args) callconv(.c) ?*c.PJRT_Error {
    const callback = args.*.callback orelse return errorToPjrt(error.InvalidArgument);
    callback(null, args.*.user_arg);
    return null;
}

fn deviceDescriptionId(args: [*c]c.PJRT_DeviceDescription_Id_Args) callconv(.c) ?*c.PJRT_Error {
    const device: *cuda.Device = @ptrCast(@alignCast(args.*.device_description.?));
    args.*.id = device.handle;
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
    args.*.device_description = @ptrCast(args.*.device.?);
    return null;
}

fn deviceLocalHardwareId(args: [*c]c.PJRT_Device_LocalHardwareId_Args) callconv(.c) ?*c.PJRT_Error {
    const device: *cuda.Device = @ptrCast(@alignCast(args.*.device.?));
    args.*.local_hardware_id = device.handle;
    return null;
}

fn deviceAddressableMemories(args: [*c]c.PJRT_Device_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
    const client = Client.fromDevice(args.*.device);
    args.*.memories = &client.addressable_memories;
    args.*.num_memories = client.addressable_memories.len;
    return null;
}

fn deviceDefaultMemory(args: [*c]c.PJRT_Device_DefaultMemory_Args) callconv(.c) ?*c.PJRT_Error {
    const client = Client.fromDevice(args.*.device);
    args.*.memory = @ptrCast(&client.memory);
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
    const memory: *Memory = @ptrCast(@alignCast(args.*.memory));
    args.*.devices = &memory.client.addressable_devices;
    args.*.num_devices = memory.client.addressable_devices.len;
    return null;
}

fn bufferDestroy(args: [*c]c.PJRT_Buffer_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args.*.buffer) |pjrt_buffer| Buffer.fromPjrt(pjrt_buffer).destroy();
    return null;
}

fn bufferElementType(args: [*c]c.PJRT_Buffer_ElementType_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.type = Buffer.fromPjrt(args.*.buffer).element_type;
    return null;
}

fn bufferDimensions(args: [*c]c.PJRT_Buffer_Dimensions_Args) callconv(.c) ?*c.PJRT_Error {
    const buffer = Buffer.fromPjrt(args.*.buffer);
    args.*.dims = buffer.dims.ptr;
    args.*.num_dims = buffer.dims.len;
    return null;
}

fn bufferOnDeviceSizeInBytes(args: [*c]c.PJRT_Buffer_OnDeviceSizeInBytes_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.on_device_size_in_bytes = Buffer.fromPjrt(args.*.buffer).allocation.size;
    return null;
}

fn bufferDevice(args: [*c]c.PJRT_Buffer_Device_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.device = @ptrCast(&Buffer.fromPjrt(args.*.buffer).client.cuda.device);
    return null;
}

fn bufferMemory(args: [*c]c.PJRT_Buffer_Memory_Args) callconv(.c) ?*c.PJRT_Error {
    args.*.memory = @ptrCast(&Buffer.fromPjrt(args.*.buffer).client.memory);
    return null;
}

fn bufferReadyEvent(args: [*c]c.PJRT_Buffer_ReadyEvent_Args) callconv(.c) ?*c.PJRT_Error {
    _ = Buffer.fromPjrt(args.*.buffer);
    const event = createReadyEvent() catch return errorToPjrt(error.OutOfMemory);
    args.*.event = @ptrCast(event);
    return null;
}

fn loadedExecutableDestroy(_: [*c]c.PJRT_LoadedExecutable_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
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
    result.PJRT_Event_Destroy = &eventDestroy;
    result.PJRT_Event_IsReady = &eventIsReady;
    result.PJRT_Event_Error = &eventError;
    result.PJRT_Event_Await = &eventAwait;
    result.PJRT_Event_OnReady = &eventOnReady;
    result.PJRT_Client_Create = &clientCreate;
    result.PJRT_Client_Destroy = &clientDestroy;
    result.PJRT_Client_PlatformName = &clientPlatformName;
    result.PJRT_Client_AddressableDevices = &clientAddressableDevices;
    result.PJRT_Client_AddressableMemories = &clientAddressableMemories;
    result.PJRT_Client_Compile = &clientCompile;
    result.PJRT_Client_BufferFromHostBuffer = &clientBufferFromHostBuffer;
    result.PJRT_Client_DmaMap = &clientDmaMap;
    result.PJRT_Client_DmaUnmap = &clientDmaUnmap;
    result.PJRT_Client_CreateBuffersForAsyncHostToDevice = &clientCreateBuffersForAsyncHostToDevice;
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
    result.PJRT_AsyncHostToDeviceTransferManager_Destroy = &transferManagerDestroy;
    result.PJRT_AsyncHostToDeviceTransferManager_TransferData = &transferManagerTransferData;
    result.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer = &transferManagerRetrieveBuffer;
    result.PJRT_AsyncHostToDeviceTransferManager_Device = &transferManagerDevice;
    result.PJRT_AsyncHostToDeviceTransferManager_BufferCount = &transferManagerBufferCount;
    result.PJRT_AsyncHostToDeviceTransferManager_BufferSize = &transferManagerBufferSize;
    result.PJRT_LoadedExecutable_Destroy = &loadedExecutableDestroy;
    result.PJRT_Buffer_Destroy = &bufferDestroy;
    result.PJRT_Buffer_ElementType = &bufferElementType;
    result.PJRT_Buffer_Dimensions = &bufferDimensions;
    result.PJRT_Buffer_OnDeviceSizeInBytes = &bufferOnDeviceSizeInBytes;
    result.PJRT_Buffer_Device = &bufferDevice;
    result.PJRT_Buffer_Memory = &bufferMemory;
    result.PJRT_Buffer_ReadyEvent = &bufferReadyEvent;
    return result;
}

const api = makeApi();

pub export fn GetPjrtApi() *const c.PJRT_Api {
    return &api;
}
