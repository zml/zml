const std = @import("std");

const asynk = @import("async");
const mlir = @import("mlir");
const pjrt = @import("pjrt");
const stablehlo = @import("mlir/dialects").stablehlo;
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const CompilationContext = @import("module.zig").CompilationContext;
const DataType = @import("dtype.zig").DataType;
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const mlirx = @import("mlirx.zig");
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/callback");

/// Inserts a user-defined callback into the computation graph.
/// The callback is defined with a struct, that store runtime information needed by the callback.
///
/// ❗Experimental API❗
///
/// ```zig
/// pub const MyCallback = struct {
///     // a unique type_id will be set by the PJRT plugin during registration.
///     pub var type_id: pjrt.ffi.TypeId = undefined;
///
///     pub const callback_config: zml.callback.Config = .{
///          // assumption this custom call makes about the input / output buffers
///     };
///
///     // Required, this will tell the callback in which env it runs.
///     platform: zml.Platform,
///     // data needed by the callback
///     my_data: []const u8,
///
///     // storage modified by the runtime to tell the callback where it should write its results.
///     // Normally the callback doesn't need to allocate as the input and output buffers are given.
///     results: [1]Buffer = undefined,
///
///     pub fn init(my_data: []const u8) !MyCallback {
///         return .{ .my_data = my_data };
///     }
///
///     pub fn call(callback: *MyCallback, input: Buffer) !void {
///         // Do something with `input` and `callback.my_data`, write the results inside `callback.results[0]`
///     }
/// };
/// ```
///
/// See eg the implementation of the `zml.callback.Print` callback, for a practical example.
///
/// Note calling this during the compilation of a module, isn't enough:
///
/// * backend need to be made aware of the callback, see `zml.Platform.registerCallback`
/// * executable need to know the specific data needed by `MyCallback`, see `zml.Exe.bind`
pub fn call(
    comptime Callback: type,
    inputs: TensorArgs(Callback),
    output_shapes: []const Shape,
) []Tensor {
    checkIsValidCallback(Callback);

    const ctx = CompilationContext.current();
    const allocator = ctx.allocator();
    const mlir_ctx = ctx.mlirCtx();
    const platform = ctx._platform;
    const pjrt_api = platform.pjrt_api;

    if (pjrt_api.ffi() == null) {
        stdx.debug.panic("Custom calls are not supported for target {s}", .{@tagName(platform.target)});
    }

    const output_tensors = allocator.alloc(Tensor, output_shapes.len) catch @panic("OOM");
    // Note: we don't always free output_tensor, because it's returned to the caller.
    // It's also why we allocate it first so that it doesn't fragment the arena.
    errdefer allocator.free(output_tensors);

    const output_types = allocator.alloc(mlir.Type, output_shapes.len) catch @panic("OOM");
    defer allocator.free(output_types);
    for (output_types, output_shapes) |*output_type, output_shape| {
        output_type.* = mlirx.tensorType(mlir_ctx, output_shape);
    }
    const input_values = allocator.alloc(mlir.Value, inputs.len) catch @panic("OOM");
    defer allocator.free(input_values);
    for (input_values, inputs) |*input_value, input_tensor| {
        input_value.* = input_tensor.value();
    }

    const target_name = "zml$" ++ @typeName(Callback);
    const op = stablehlo.custom_call(
        mlir_ctx,
        input_values,
        .{
            .call_target_name = target_name,
            .api_version = .typed_ffi,
            .backend_config = .dict(mlir_ctx, &.{}),
            .additional_attributes = &.{.{ "mhlo.frontend_attributes", .dict(mlir_ctx, &.{}) }},
            .has_side_effect = true,
            .output_operand_aliases = Callback.callback_config.output_operand_aliases,
        },
        output_types,
        mlir_ctx.location(@src()),
    );

    for (output_tensors, output_shapes, 0..) |*output_tensor, output_shape, i| {
        output_tensor.* = Tensor._result(output_shape, op.result(i));
    }
    return output_tensors;
}

/// Describe properties of a callback
///
/// * output_operand_aliases: the callback reuse input buffer to write the output
/// * copy_inputs_to_host_pinned: the callback need to work on host visible buffers
/// * traits: PJRT specified properties of the callback
pub const Config = struct {
    output_operand_aliases: []const i64 = &.{},
    copy_inputs_to_host_pinned: bool = false,
    // TODO: document precisely what `command_buffer_compatible` is doing and its limitations.
    traits: pjrt.ffi.HandlerTraits = .{ .command_buffer_compatible = false },
    // TODO: handle sharded inputs
};

/// Compile-time check that a callback has all informations we require.
pub fn checkIsValidCallback(Callback: type) void {
    stdx.debug.assertComptime(@hasDecl(Callback, "call"), "Expected callback {} to have a call method", .{Callback});
    const ArgsT = stdx.meta.FnArgs(Callback.call);
    inline for (@typeInfo(ArgsT).@"struct".fields[1..]) |field| {
        stdx.debug.assertComptime(field.type == Buffer, "Expected callback {}.call arguments to be of type zml.Buffer, got {}", .{ Callback, field.type });
    }

    stdx.debug.assertComptime(@hasDecl(Callback, "type_id") and @TypeOf(Callback.type_id) == pjrt.ffi.TypeId, "Expected callback {} to have a field `pub var type_id: pjrt.ffi.TypeId`", .{Callback});
    stdx.debug.assertComptime(@hasDecl(Callback, "callback_config") and @TypeOf(Callback.callback_config) == Config, "Expected callback {} to have a field `pub const callback_config: zml.CustomCallOptions`", .{Callback});
}

pub fn register(Callback: type, platform: Platform) pjrt.ApiError!void {
    checkIsValidCallback(Callback);

    const ffi = platform.pjrt_api.ffi() orelse return error.Unavailable;
    const target_name = "zml$" ++ @typeName(Callback);

    const proxy_cb = proxy(Callback);
    Callback.type_id = try ffi.registerTypeId(platform.pjrt_api, @typeName(Callback));
    try ffi.register(platform.pjrt_api, target_name, @tagName(platform.target), &proxy_cb, Callback.callback_config.traits);
    log.debug("Registered custom call {} with target name \"{s}\"", .{ Callback, target_name });
}

fn proxy(Callback: type) pjrt.ffi.Handler {
    return struct {
        pub fn cb(call_frame: *pjrt.ffi.CallFrame) callconv(.c) ?*pjrt.ffi.Error {
            return CallbackImpl(Callback, call_frame);
        }
    }.cb;
}

fn CallbackImpl(comptime Callback: type, call_frame: *pjrt.ffi.CallFrame) ?*pjrt.ffi.Error {
    if (call_frame.registeringHook()) return null;

    const opts = Callback.callback_config;

    const execution_context = call_frame.ctx;
    log.debug("Custom call {s} called !", .{@typeName(Callback)});
    const user_ctx_opaque = execution_context.getContext(Callback.type_id, call_frame.api) catch {
        log.err("{} user data was never given for current executable", .{Callback});
        return .create(call_frame.api, .failed_precondition, "failed to fetch user context" ++ @typeName(Callback));
    };
    const user_ctx: *Callback = @ptrCast(@alignCast(user_ctx_opaque));
    // We actually have one more constraint here, we force the Callback to have a platform field,
    // and to correctly set it.
    // Is this good ? We could also simplify this by registering ourselves the `Platform` type id.
    const platform: Platform = user_ctx.platform;

    // Hook to get a cuda stream in the callback.
    if (@hasField(Callback, "stream") and platform.target != .cpu) {
        const stream = call_frame.api.stream(execution_context);
        user_ctx.stream = stream;
    }

    var callback_args: std.meta.ArgsTuple(@TypeOf(Callback.call)) = undefined;
    callback_args[0] = user_ctx;

    inline for (1..callback_args.len, call_frame.args.buffers()) |i, ffi_buffer| {
        const shape = shapeFromFfi(ffi_buffer);
        var zml_buffer: Buffer = if (platform.target == .cpu)
            .asViewOfHostBuffer(platform, .fromBytes(shape, ffi_buffer.data[0..shape.byteSize()]))
        else
            .asViewOfDeviceBuffer(platform, shape, null, ffi_buffer.data);
        if (opts.copy_inputs_to_host_pinned and platform.target != .cpu) {
            log.debug("Copying argument {d} {f} {*} to host_pinned memory !", .{ i, zml_buffer, zml_buffer.opaqueDeviceMemoryDataPointer() });
            zml_buffer = zml_buffer.copyToMemory(platform, .host_pinned, .{ .wait = true }) catch |err| {
                log.err("Failed to copy input buffer {d} {f} {*} to host_pinned: {}", .{ i, zml_buffer, zml_buffer.opaqueDeviceMemoryDataPointer(), err });
                return .create(call_frame.api, .resource_exhausted, "host pinned OOM");
            };
            log.debug("--> {f} {*} ({})", .{ zml_buffer, zml_buffer.opaqueDeviceMemoryDataPointer(), @as(*const f32, @ptrCast(@alignCast(zml_buffer.opaqueDeviceMemoryDataPointer()))).* });
        }
        callback_args[i] = zml_buffer;
    }

    for (0..call_frame.results.len) |i| {
        const ffi_buffer = call_frame.results.buffers()[i];
        const ffi_buffer_shape = shapeFromFfi(ffi_buffer);

        if (platform.target == .cpu) {
            user_ctx.results[i] = Buffer.asViewOfHostBuffer(platform, HostBuffer.fromBytes(ffi_buffer_shape, ffi_buffer.data[0..ffi_buffer_shape.byteSize()]));
        } else {
            user_ctx.results[i] = Buffer.asViewOfDeviceBuffer(platform, shapeFromFfi(ffi_buffer), null, ffi_buffer.data);
        }
    }

    @call(.auto, Callback.call, callback_args) catch |err| {
        log.err("Callback {} failed with {}", .{ Callback, err });
        return .create(call_frame.api, .internal, "internal callback error");
    };

    return .ok;
}

/// Internal custom calls.
/// These are not meant to be used by users, but rather by the library itself.
pub const internal_callbacks = [_]type{
    Print,
};

pub fn registerInternalCallbacks(platform: Platform) !void {
    inline for (internal_callbacks) |Callback| {
        try register(Callback, platform);
        // log.debug("Registered internal custom call {s} with type_id {d}", .{ @typeName(Callback), Callback.type_id.type_id });
    }
}

/// Allocate user data data needed by the ZML provided custom calls.
pub fn bindInternalCallbacks(
    arena: std.mem.Allocator,
    platform: Platform,
    ffi: pjrt.Ffi,
    execute_context: *pjrt.ExecuteContext,
) (std.mem.Allocator.Error || pjrt.ApiError)!void {
    // Atm we don't have a mechanism to detect which ZML callbacks the executable needs,
    // so we always allocate.
    {
        // Print
        const print_ptr = try arena.create(Print);
        print_ptr.* = try .init(platform);
        try addUserData(Print, platform.pjrt_api, ffi, execute_context, print_ptr);
    }
}

pub fn addUserData(
    Callback: type,
    api: *const pjrt.Api,
    ffi: pjrt.Ffi,
    execute_context: *pjrt.ExecuteContext,
    user_data: *Callback,
) pjrt.ApiError!void {
    try ffi.addUserData(
        api,
        execute_context,
        .{ .type_id = Callback.type_id.type_id, .user_data = @ptrCast(user_data) },
    );
    log.debug("Bound {s}@{x} with type id {d} on {any}", .{ @typeName(Callback), @intFromPtr(user_data), Callback.type_id.type_id, execute_context });
}

/// The print callback
pub const Print = struct {
    // a unique type_id will be set by the PJRT plugin during registration.
    pub var type_id: pjrt.ffi.TypeId = undefined;

    pub const callback_config: Config = .{
        // Print callback pretends to modify the given input buffer, but just returns it unmodified.
        .output_operand_aliases = &.{0},
        // It also needs PJRT to copy the data on the host first so it can print it.
        .copy_inputs_to_host_pinned = true,
        // Print is fairly predictable and can be captured in an execution graph.
        .traits = .{ .command_buffer_compatible = false },
    };

    platform: Platform,
    results: [1]Buffer = undefined,

    pub fn init(platform: Platform) !Print {
        return .{ .platform = platform };
    }

    pub fn call(_: *Print, input: Buffer) !void {
        std.log.defaultLog(.info, .zml, "Device buffer: {f}: {d}", .{ input, input.asHostBuffer() });
    }
};

fn shapeFromFfi(ffi_buffer: *const pjrt.ffi.Buffer) Shape {
    const dt: DataType = switch (ffi_buffer.dtype) {
        .invalid => stdx.debug.panic("Invalid FFI dtype {any} used by {any}", .{ ffi_buffer.dtype, ffi_buffer }),
        .token, .f8e4m3, .f8e3m4 => stdx.debug.panic("Unsupported FFI dtype {any} used by {any}", .{ ffi_buffer.dtype, ffi_buffer }),
        inline else => |t| @field(DataType, @tagName(t)),
    };
    return Shape.init(ffi_buffer.dims(), dt);
}

fn TensorArgs(Callback: type) type {
    const ArgsT = stdx.meta.FnArgs(Callback.call);

    const args = @typeInfo(ArgsT).@"struct".fields;
    return [args.len - 1]Tensor;
}
