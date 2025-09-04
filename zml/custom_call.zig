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
const ffi = pjrtx.ffi;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/custom_call");

pub fn CustomCallInputsType(CustomOp: type) type {
    const ArgsT = stdx.meta.FnArgs(CustomOp.call);
    if (@typeInfo(ArgsT) != .@"struct") {
        @compileError("Expected struct type");
    }

    const args = @typeInfo(ArgsT).@"struct".fields;

    for (args[1..args.len]) |field| {
        if (field.type != Buffer) {
            @compileError("Expected all arguments in custom call `call` to be of type zml.Buffer");
        }
    }

    return [args.len - 1]Tensor;
}

pub const CustomCallOptions = struct {
    output_operand_aliases: []const i64 = &.{},
    copy_inputs_to_host_pinned: bool = false,
    handler_traits: pjrt.ffi.HandlerTraits = .{ .command_buffer_compatible = false },
};

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
///     pub const custom_call_options: CustomCallOptions = .{
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
/// See eg the implementation of the `zml.Print` callback, for a practical example.
///
/// Note calling this during the compilation of a module, isn't enough.
/// The generated executable will need to know the specific data needed by `MyCallback` in this executable.
/// This is done through `zm.Exe.bind`.
pub fn customCall(
    comptime CustomOp: type,
    inputs: CustomCallInputsType(CustomOp),
    output_shapes: []const Shape,
) []Tensor {
    const op_name = @typeName(CustomOp);
    stdx.debug.assertComptime(@hasDecl(CustomOp, "call"), "{s} must have a call method", .{op_name});
    stdx.debug.assertComptime(@hasDecl(CustomOp, "type_id") and @TypeOf(CustomOp.type_id) == pjrt.ffi.TypeId, "{s} must have a field `pub var type_id: pjrt.ffi.TypeId`", .{op_name});

    const ctx = CompilationContext.current();
    const allocator = ctx.allocator();
    const mlir_ctx = ctx.mlirCtx();
    const platform = ctx._platform;
    const pjrt_api = platform.pjrt_api;

    if (pjrt_api.ffi() == null) {
        stdx.debug.panic("Custom calls are not supported for target {s}", .{@tagName(platform.target)});
    }

    const target_name = "zml$" ++ op_name;

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

    const op = stablehlo.custom_call(
        mlir_ctx,
        input_values,
        .{
            .call_target_name = target_name,
            .api_version = .typed_ffi,
            .backend_config = .dict(mlir_ctx, &.{}),
            .additional_attributes = &.{.{ "mhlo.frontend_attributes", .dict(mlir_ctx, &.{}) }},
            .has_side_effect = true,
            .output_operand_aliases = CustomOp.custom_call_options.output_operand_aliases,
        },
        output_types,
        mlir_ctx.location(@src()),
    );

    for (output_tensors, output_shapes, 0..) |*output_tensor, output_shape, i| {
        output_tensor.* = Tensor._result(output_shape, op.result(i));
    }
    return output_tensors;
}

pub fn proxy(CustomOp: type) ffi.Handler {
    return struct {
        pub fn cb(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
            return customCallImpl(CustomOp, call_frame);
        }
    }.cb;
}

fn customCallImpl(comptime CustomOp: type, call_frame: *ffi.CallFrame) ?*ffi.Error {
    if (call_frame.registeringHook()) return null;

    const opts = CustomOp.custom_call_options;

    const execution_context = call_frame.ctx;
    log.info("Custom call {s} called !", .{@typeName(CustomOp)});
    const user_ctx_opaque = execution_context.getContext(CustomOp.type_id, call_frame.api) catch {
        return .create(call_frame.api, .internal, "failed to fetch user context for custom_call" ++ @typeName(CustomOp));
    };
    const user_ctx: *CustomOp = @ptrCast(@alignCast(user_ctx_opaque));
    // We actually have one more constraint here, we force the CustomOp to have a platform field,
    // and to correctly set it.
    // Is this good ? We could also simplify this by registering ourselves the `Platform` type id.
    const platform: Platform = user_ctx.platform;

    // Hook to get a cuda stream in the callback.
    if (@hasField(CustomOp, "stream") and platform.target != .cpu) {
        const stream = call_frame.api.stream(execution_context);
        user_ctx.stream = stream;
    }

    var callback_args: std.meta.ArgsTuple(@TypeOf(CustomOp.call)) = undefined;
    callback_args[0] = user_ctx;

    inline for (1..callback_args.len, call_frame.args.buffers()) |i, ffi_buffer| {
        const shape = getShape(ffi_buffer);
        var zml_buffer: Buffer = if (platform.target == .cpu)
            .asViewOfHostBuffer(platform, .fromBytes(shape, ffi_buffer.data[0..shape.byteSize()]))
        else
            .asViewOfDeviceBuffer(platform, shape, null, ffi_buffer.data);
        if (opts.copy_inputs_to_host_pinned and platform.target != .cpu) {
            log.info("Copying argument {} {f} {*} to host_pinned memory !", .{ i, zml_buffer, zml_buffer.opaqueDeviceMemoryDataPointer() });
            zml_buffer = zml_buffer.copyToMemory(platform, .host_pinned, .{ .wait = true }) catch unreachable;
            log.info("--> {f} {*} ({})", .{ zml_buffer, zml_buffer.opaqueDeviceMemoryDataPointer(), @as(*const f32, @ptrCast(@alignCast(zml_buffer.opaqueDeviceMemoryDataPointer()))).* });
        }
        callback_args[i] = zml_buffer;
    }

    for (0..call_frame.results.len) |i| {
        const ffi_buffer = call_frame.results.buffers()[i];
        const ffi_buffer_shape = getShape(ffi_buffer);

        if (platform.target == .cpu) {
            user_ctx.results[i] = Buffer.asViewOfHostBuffer(platform, HostBuffer.fromBytes(ffi_buffer_shape, ffi_buffer.data[0..ffi_buffer_shape.byteSize()]));
        } else {
            user_ctx.results[i] = Buffer.asViewOfDeviceBuffer(platform, getShape(ffi_buffer), null, ffi_buffer.data);
        }
    }

    @call(.auto, CustomOp.call, callback_args) catch |err| {
        stdx.debug.panic("Error while calling {s} call func: {any}\n", .{ @typeName(CustomOp), err });
    };

    return null;
}

pub fn getShape(ffi_buffer: *const ffi.Buffer) Shape {
    const dt: DataType = switch (ffi_buffer.dtype) {
        .invalid => stdx.debug.panic("Invalid FFI dtype {any} used by {any}", .{ ffi_buffer.dtype, ffi_buffer }),
        .token, .f8e4m3, .f8e3m4 => stdx.debug.panic("Unsupported FFI dtype {any} used by {any}", .{ ffi_buffer.dtype, ffi_buffer }),
        inline else => |t| @field(DataType, @tagName(t)),
    };
    return Shape.init(ffi_buffer.dims(), dt);
}

/// Internal custom calls.
/// These are not meant to be used by users, but rather by the library itself.
pub const internal_custom_calls = [_]type{
    Print,
};

pub fn registerInternalCustomCalls(platform: Platform) !void {
    inline for (internal_custom_calls) |custom_call_type| {
        try platform.registerCustomCall(custom_call_type);
        log.info("Registered internal custom call {s} with type_id {d}", .{ @typeName(custom_call_type), custom_call_type.type_id.type_id });
    }
}

/// The print callback
pub const Print = struct {
    // a unique type_id will be set by the PJRT plugin during registration.
    pub var type_id: pjrt.ffi.TypeId = undefined;

    pub const custom_call_options: CustomCallOptions = .{
        // Print callback pretends to modify the given input buffer, but just returns it unmodified.
        .output_operand_aliases = &.{0},
        // It also needs PJRT to copy the data on the host first so it can print it.
        .copy_inputs_to_host_pinned = true,
        // Print is fairly predictable and can be captured in an execution graph.
        .handler_traits = .{ .command_buffer_compatible = false },
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
