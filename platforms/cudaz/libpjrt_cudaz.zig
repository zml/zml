const std = @import("std");

const c = @import("c");
const compiler = @import("compiler.zig");
const cuda = @import("cuda.zig");

const log = std.log.scoped(.@"zml/platforms/cudaz");

const Error = struct {
    base: c.PJRT_Error,
    code: c.PJRT_Error_Code,
    message: []const u8,
};

const Executable = struct {
    ptx: [:0]u8,

    fn create(ptx: [:0]u8) !*Executable {
        const self = try std.heap.page_allocator.create(Executable);
        self.* = .{ .ptx = ptx };
        return self;
    }

    fn clone(self: *const Executable) !*Executable {
        const ptx = try std.heap.page_allocator.dupeZ(u8, self.ptx);
        errdefer std.heap.page_allocator.free(ptx);
        return create(ptx);
    }

    fn destroy(self: *Executable) void {
        std.heap.page_allocator.free(self.ptx);
        std.heap.page_allocator.destroy(self);
    }

    fn fromPjrt(pjrt_executable: ?*c.PJRT_Executable) *Executable {
        return @ptrCast(@alignCast(pjrt_executable.?));
    }
};

const LoadedExecutable = struct {
    client: *Client,
    executable: *Executable,
    kernel: cuda.Kernel,

    fn destroy(self: *LoadedExecutable) void {
        self.client.cuda.unloadKernel(self.kernel);
        self.executable.destroy();
        std.heap.page_allocator.destroy(self);
    }

    fn fromPjrt(pjrt_executable: ?*c.PJRT_LoadedExecutable) *LoadedExecutable {
        return @ptrCast(@alignCast(pjrt_executable.?));
    }
};

const Memory = extern struct {
    base: c.PJRT_Memory,
    client: *Client,
};

const Client = struct {
    cuda: cuda.Client,
    memory: Memory,
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
const executable_name = "cudaz Zig PTX kernel";
const ptx_format = "ptx";
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
var cuda_device_to_host_copy_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_INTERNAL,
    .message = "cudaz: CUDA device-to-host copy failed",
};
var compiler_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_INTERNAL,
    .message = "cudaz: Zig PTX compilation failed",
};
var cuda_module_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_INTERNAL,
    .message = "cudaz: CUDA PTX module loading failed",
};
var cuda_function_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_NOT_FOUND,
    .message = "cudaz: CUDA kernel function was not found",
};
var cuda_launch_error: Error = .{
    .base = .{ .vtable = &error_vtable },
    .code = c.PJRT_Error_Code_INTERNAL,
    .message = "cudaz: CUDA kernel launch failed",
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
        error.DeviceToHostCopyFailed => &cuda_device_to_host_copy_error,
        error.RunfilesUnavailable,
        error.ToolchainConfigUnavailable,
        error.ZigExecutableUnavailable,
        error.ZigLibUnavailable,
        error.KernelSourceUnavailable,
        error.ZigCompilationFailed,
        error.InvalidKernelIr,
        => &compiler_error,
        error.ModuleLoadFailed => &cuda_module_error,
        error.FunctionLookupFailed => &cuda_function_error,
        error.KernelLaunchFailed => &cuda_launch_error,
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

    const ptx = compiler.compile(std.heap.page_allocator) catch |err| {
        log.err("Zig PTX compilation failed: {s}", .{@errorName(err)});
        return if (err == error.OutOfMemory)
            errorToPjrt(error.OutOfMemory)
        else
            @ptrCast(&compiler_error);
    };
    const executable = Executable.create(ptx) catch {
        std.heap.page_allocator.free(ptx);
        return errorToPjrt(error.OutOfMemory);
    };
    const kernel = client.cuda.loadKernel(executable.ptx, "main") catch |err| {
        executable.destroy();
        return errorToPjrt(err);
    };
    const loaded_executable = std.heap.page_allocator.create(LoadedExecutable) catch {
        client.cuda.unloadKernel(kernel);
        executable.destroy();
        return errorToPjrt(error.OutOfMemory);
    };
    loaded_executable.* = .{
        .client = client,
        .executable = executable,
        .kernel = kernel,
    };
    args.*.executable = @ptrCast(loaded_executable);
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

fn validateHostLayout(buffer: *const Buffer, layout: *const c.PJRT_Buffer_MemoryLayout) !void {
    switch (layout.type) {
        c.PJRT_Buffer_MemoryLayout_Type_Tiled => {
            const tiled = layout.unnamed_0.tiled;
            if (tiled.num_tiles != 0 or tiled.minor_to_major_size != buffer.dims.len) {
                return error.InvalidArgument;
            }
            if (tiled.minor_to_major_size > 0 and tiled.minor_to_major == null) {
                return error.InvalidArgument;
            }
            for (tiled.minor_to_major[0..tiled.minor_to_major_size], 0..) |dimension, index| {
                if (dimension != buffer.dims.len - 1 - index) return error.InvalidArgument;
            }
        },
        c.PJRT_Buffer_MemoryLayout_Type_Strides => {
            const strides = layout.unnamed_0.strides;
            if (strides.num_byte_strides != buffer.dims.len) return error.InvalidArgument;
            try validateCompactByteStrides(
                buffer.element_type,
                buffer.dims.ptr,
                buffer.dims.len,
                strides.byte_strides,
                strides.num_byte_strides,
            );
        },
        else => return error.InvalidArgument,
    }
}

fn bufferToHostBuffer(args: [*c]c.PJRT_Buffer_ToHostBuffer_Args) callconv(.c) ?*c.PJRT_Error {
    const buffer = Buffer.fromPjrt(args.*.src);
    args.*.event = null;

    if (args.*.host_layout) |host_layout| {
        validateHostLayout(buffer, host_layout) catch |err| return errorToPjrt(err);
    }

    const required_size = buffer.allocation.size;
    if (args.*.dst == null) {
        args.*.dst_size = required_size;
        return null;
    }
    if (args.*.dst_size < required_size) return errorToPjrt(error.InvalidArgument);

    const destination = @as([*]u8, @ptrCast(args.*.dst.?))[0..required_size];
    buffer.client.cuda.copyDeviceToHost(destination, buffer.allocation, 0) catch |err| {
        return errorToPjrt(err);
    };

    const event = createReadyEvent() catch return errorToPjrt(error.OutOfMemory);
    args.*.event = @ptrCast(event);
    return null;
}

test "to host buffer size query does not access CUDA" {
    var dims = [_]i64{ 2, 3 };
    var buffer: Buffer = .{
        .client = undefined,
        .allocation = .{ .ptr = 0, .size = 24 },
        .element_type = c.PJRT_Buffer_Type_F32,
        .dims = &dims,
    };
    var args: c.PJRT_Buffer_ToHostBuffer_Args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = @ptrCast(&buffer);

    try std.testing.expect(bufferToHostBuffer(&args) == null);
    try std.testing.expectEqual(@as(usize, 24), args.dst_size);
    try std.testing.expect(args.event == null);
}

test "to host buffer accepts only compact source-equivalent layouts" {
    var dims = [_]i64{ 2, 3 };
    const buffer: Buffer = .{
        .client = undefined,
        .allocation = .{ .ptr = 0, .size = 24 },
        .element_type = c.PJRT_Buffer_Type_F32,
        .dims = &dims,
    };

    const minor_to_major = [_]i64{ 1, 0 };
    const tiled: c.PJRT_Buffer_MemoryLayout = .{
        .struct_size = c.PJRT_Buffer_MemoryLayout_STRUCT_SIZE,
        .extension_start = null,
        .unnamed_0 = .{ .tiled = .{
            .struct_size = c.PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE,
            .extension_start = null,
            .minor_to_major = &minor_to_major,
            .minor_to_major_size = minor_to_major.len,
            .tile_dims = null,
            .tile_dim_sizes = null,
            .num_tiles = 0,
        } },
        .type = c.PJRT_Buffer_MemoryLayout_Type_Tiled,
    };
    try validateHostLayout(&buffer, &tiled);

    const byte_strides = [_]i64{ 12, 4 };
    const strides: c.PJRT_Buffer_MemoryLayout = .{
        .struct_size = c.PJRT_Buffer_MemoryLayout_STRUCT_SIZE,
        .extension_start = null,
        .unnamed_0 = .{ .strides = .{
            .struct_size = c.PJRT_Buffer_MemoryLayout_Strides_STRUCT_SIZE,
            .extension_start = null,
            .byte_strides = &byte_strides,
            .num_byte_strides = byte_strides.len,
        } },
        .type = c.PJRT_Buffer_MemoryLayout_Type_Strides,
    };
    try validateHostLayout(&buffer, &strides);

    const transposed_minor_to_major = [_]i64{ 0, 1 };
    var transposed = tiled;
    transposed.unnamed_0.tiled.minor_to_major = &transposed_minor_to_major;
    try std.testing.expectError(error.InvalidArgument, validateHostLayout(&buffer, &transposed));
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

fn executableDestroy(args: [*c]c.PJRT_Executable_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args.*.executable) |pjrt_executable| Executable.fromPjrt(pjrt_executable).destroy();
    return null;
}

fn executableName(args: [*c]c.PJRT_Executable_Name_Args) callconv(.c) ?*c.PJRT_Error {
    _ = Executable.fromPjrt(args.*.executable);
    args.*.executable_name = executable_name.ptr;
    args.*.executable_name_size = executable_name.len;
    return null;
}

fn executableNumReplicas(args: [*c]c.PJRT_Executable_NumReplicas_Args) callconv(.c) ?*c.PJRT_Error {
    _ = Executable.fromPjrt(args.*.executable);
    args.*.num_replicas = 1;
    return null;
}

fn executableNumPartitions(args: [*c]c.PJRT_Executable_NumPartitions_Args) callconv(.c) ?*c.PJRT_Error {
    _ = Executable.fromPjrt(args.*.executable);
    args.*.num_partitions = 1;
    return null;
}

fn executableOptimizedProgram(args: [*c]c.PJRT_Executable_OptimizedProgram_Args) callconv(.c) ?*c.PJRT_Error {
    const executable = Executable.fromPjrt(args.*.executable);
    const program = args.*.program orelse return errorToPjrt(error.InvalidArgument);
    program.*.format = ptx_format.ptr;
    program.*.format_size = ptx_format.len;

    if (program.*.code == null) {
        program.*.code_size = executable.ptx.len;
        return null;
    }
    if (program.*.code_size < executable.ptx.len) return errorToPjrt(error.InvalidArgument);
    @memcpy(program.*.code[0..executable.ptx.len], executable.ptx);
    program.*.code_size = executable.ptx.len;
    return null;
}

fn loadedExecutableDestroy(args: [*c]c.PJRT_LoadedExecutable_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args.*.executable) |pjrt_executable| LoadedExecutable.fromPjrt(pjrt_executable).destroy();
    return null;
}

fn loadedExecutableGetExecutable(args: [*c]c.PJRT_LoadedExecutable_GetExecutable_Args) callconv(.c) ?*c.PJRT_Error {
    const loaded_executable = LoadedExecutable.fromPjrt(args.*.loaded_executable);
    const executable = loaded_executable.executable.clone() catch return errorToPjrt(error.OutOfMemory);
    args.*.executable = @ptrCast(executable);
    return null;
}

fn loadedExecutableAddressableDevices(
    args: [*c]c.PJRT_LoadedExecutable_AddressableDevices_Args,
) callconv(.c) ?*c.PJRT_Error {
    const loaded_executable = LoadedExecutable.fromPjrt(args.*.executable);
    args.*.addressable_devices = &loaded_executable.client.addressable_devices;
    args.*.num_addressable_devices = loaded_executable.client.addressable_devices.len;
    return null;
}

const KernelParameters = extern struct {
    buffers: cuda.DevicePtr,
    buffer_len: usize,
};

fn execute(
    loaded_executable: *LoadedExecutable,
    args: *c.PJRT_LoadedExecutable_Execute_Args,
) !void {
    if (args.num_devices != loaded_executable.client.addressable_devices.len) {
        return error.InvalidArgument;
    }
    if (args.execute_device) |execute_device| {
        const expected_device: *c.PJRT_Device = @ptrCast(&loaded_executable.client.cuda.device);
        if (execute_device != expected_device) return error.InvalidArgument;
    }

    const buffer_count = std.math.mul(usize, args.num_devices, args.num_args) catch
        return error.InvalidArgument;
    if (buffer_count > 0 and args.argument_lists == null) return error.InvalidArgument;

    const device_pointers = try std.heap.page_allocator.alloc(cuda.DevicePtr, buffer_count);
    defer std.heap.page_allocator.free(device_pointers);

    var buffer_index: usize = 0;
    for (0..args.num_devices) |device_index| {
        const argument_list = args.argument_lists[device_index];
        if (args.num_args > 0 and argument_list == null) return error.InvalidArgument;
        for (0..args.num_args) |argument_index| {
            const buffer = Buffer.fromPjrt(argument_list[argument_index]);
            if (buffer.client != loaded_executable.client) return error.InvalidArgument;
            device_pointers[buffer_index] = buffer.allocation.ptr;
            buffer_index += 1;
        }
    }

    const pointer_bytes = std.mem.sliceAsBytes(device_pointers);
    const device_pointer_table = try loaded_executable.client.cuda.allocate(pointer_bytes.len);
    defer loaded_executable.client.cuda.free(device_pointer_table);
    try loaded_executable.client.cuda.copyHostToDevice(device_pointer_table, 0, pointer_bytes);

    const parameters: KernelParameters = .{
        .buffers = device_pointer_table.ptr,
        .buffer_len = buffer_count,
    };
    try loaded_executable.client.cuda.launch(
        loaded_executable.kernel,
        std.mem.asBytes(&parameters),
    );

    if (args.device_complete_events != null) {
        var initialized_events: usize = 0;
        errdefer for (0..initialized_events) |device_index| {
            std.heap.page_allocator.destroy(Event.fromPjrt(args.device_complete_events[device_index]));
        };

        for (0..args.num_devices) |device_index| {
            const event = try createReadyEvent();
            args.device_complete_events[device_index] = @ptrCast(event);
            initialized_events += 1;
        }
    }
}

fn loadedExecutableExecute(
    args: [*c]c.PJRT_LoadedExecutable_Execute_Args,
) callconv(.c) ?*c.PJRT_Error {
    const loaded_executable = LoadedExecutable.fromPjrt(args.*.executable);
    execute(loaded_executable, args) catch |err| return errorToPjrt(err);
    return null;
}

test "kernel parameter buffer matches the PTX entry ABI" {
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(KernelParameters, "buffers"));
    try std.testing.expectEqual(@sizeOf(cuda.DevicePtr), @offsetOf(KernelParameters, "buffer_len"));
    try std.testing.expectEqual(
        @sizeOf(cuda.DevicePtr) + @sizeOf(usize),
        @sizeOf(KernelParameters),
    );
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
    result.PJRT_Executable_Destroy = &executableDestroy;
    result.PJRT_Executable_Name = &executableName;
    result.PJRT_Executable_NumReplicas = &executableNumReplicas;
    result.PJRT_Executable_NumPartitions = &executableNumPartitions;
    result.PJRT_Executable_OptimizedProgram = &executableOptimizedProgram;
    result.PJRT_LoadedExecutable_Destroy = &loadedExecutableDestroy;
    result.PJRT_LoadedExecutable_GetExecutable = &loadedExecutableGetExecutable;
    result.PJRT_LoadedExecutable_AddressableDevices = &loadedExecutableAddressableDevices;
    result.PJRT_LoadedExecutable_Execute = &loadedExecutableExecute;
    result.PJRT_Buffer_Destroy = &bufferDestroy;
    result.PJRT_Buffer_ElementType = &bufferElementType;
    result.PJRT_Buffer_Dimensions = &bufferDimensions;
    result.PJRT_Buffer_OnDeviceSizeInBytes = &bufferOnDeviceSizeInBytes;
    result.PJRT_Buffer_ToHostBuffer = &bufferToHostBuffer;
    result.PJRT_Buffer_Device = &bufferDevice;
    result.PJRT_Buffer_Memory = &bufferMemory;
    result.PJRT_Buffer_ReadyEvent = &bufferReadyEvent;
    return result;
}

const api = makeApi();

pub export fn GetPjrtApi() *const c.PJRT_Api {
    return &api;
}
