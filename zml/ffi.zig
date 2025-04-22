const std = @import("std");
const testing = std.testing;
const builtin = @import("builtin");

const asynk = @import("async");
const stdx = @import("stdx");

const context = @import("context.zig");
const dialect = struct {
    const stablehlo = @import("mlir/dialects").stablehlo;
};
const dtype = @import("dtype.zig");
const pjrt = @import("pjrtx.zig");
const buffer = @import("buffer.zig");
const hostbuffer = @import("hostbuffer.zig");
const mlir = @import("mlir.zig");
const module = @import("module.zig");
const ops = @import("ops.zig");
const platform = @import("platform.zig");
const tensor = @import("tensor.zig");
const shape = @import("shape.zig");
const cuda = context.cuda;
const Buffer = buffer.Buffer;
const HostBuffer = hostbuffer.HostBuffer;
const CompilationContext = module.CompilationContext;
const DataType = dtype.DataType;
const Platform = platform.Platform;
const Shape = shape.Shape;
const Tensor = tensor.Tensor;

const log = std.log.scoped(.@"zml/custom_call");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

// todo: check deallocs

pub fn CustomCallInputType(custom_op: type) type {
    const ArgsT = stdx.meta.FnArgs(custom_op.beforeCustomCall);
    if (@typeInfo(ArgsT) != .@"struct") {
        @compileError("Expected struct type");
    }
    // todo check typeof first input then others args
    return [@typeInfo(ArgsT).@"struct".fields.len]Tensor;
}

pub fn CustomCallOutputType(custom_op: type) type {
    const ReturnT = stdx.meta.FnResultNoError(custom_op.call);

    if (@typeInfo(ReturnT) != .@"struct") {
        @compileError("Expected struct type");
    }

    if (ReturnT == Buffer) {
        return Tensor;
    }
    return [@typeInfo(ReturnT).@"struct".fields.len]Tensor;
}

pub fn lenOutputBeforeCustomCall(custom_op: type) usize {
    const ReturnT = stdx.meta.FnResultNoError(custom_op.beforeCustomCall);

    if (@typeInfo(ReturnT) != .@"struct") {
        @compileError("Expected struct type");
    }

    if (ReturnT == Tensor) {
        return 1;
    }
    return @typeInfo(ReturnT).@"struct".fields.len;
}

pub fn custom_call(
    comptime custom_op: type,
    inputs: CustomCallInputType(custom_op),
) CustomCallOutputType(custom_op) {
    stdx.debug.assert(@hasDecl(custom_op, "call"), "custom_op must have a call method", .{});
    const op_name = @typeName(custom_op);

    const ctx = inputs[0].getContext();
    const mlir_ctx = ctx.mlirCtx();
    const platform_ = ctx._platform;
    const pjrt_api = platform_.pjrt_api;
    const ffi = pjrt_api.ffi().?;
    const registry = ffi.customCallRegistry().?;
    const ffi_func = proxy(custom_op);
    const target_name = "callback_" ++ op_name;

    registry.registerFfi(pjrt_api, target_name, @tagName(platform_.target), &ffi_func) catch unreachable;
    log.info("{s} / Registered custom call with target_name \"{s}\" with proxy func {*}", .{ op_name, target_name, &ffi_func });

    var custom_call_inputs: []mlir.Value = undefined;
    var res_shapes: []Shape = undefined;
    var res_types: []mlir.Type = undefined;

    if (@hasDecl(custom_op, "beforeCustomCall")) {
        const before_custom_call = custom_op.beforeCustomCall;
        if (@typeInfo(@TypeOf(before_custom_call)) != .@"fn") {
            stdx.debug.panic("beforeCustomCall must be a function");
        }

        var args: std.meta.ArgsTuple(@TypeOf(before_custom_call)) = undefined;
        inline for (0..args.len) |i| {
            args[i] = inputs[i];
        }

        const before_custom_call_outputs = ctx.callFunc(@typeName(custom_op), before_custom_call, args) catch |err| {
            stdx.debug.panic("Error in {any} beforeCustomCall func: {any}\n", .{ @typeName(custom_op), err });
        };
        var before_custom_call_outputs_array: []Tensor = undefined;

        if (@TypeOf(before_custom_call_outputs) == Tensor) {
            before_custom_call_outputs_array = stdx.stackSlice(8, Tensor, 1);
            before_custom_call_outputs_array[0] = before_custom_call_outputs;
        } else {
            before_custom_call_outputs_array = stdx.stackSlice(8, Tensor, @typeInfo(@TypeOf(before_custom_call_outputs)).@"struct".fields.len);
            inline for (@typeInfo(@TypeOf(before_custom_call_outputs)).@"struct".fields, 0..) |field, i| {
                before_custom_call_outputs_array[i] = @field(before_custom_call_outputs, field.name);
            }
        }

        custom_call_inputs = stdx.stackSlice(8, mlir.Value, lenOutputBeforeCustomCall(custom_op));
        res_shapes = stdx.stackSlice(8, Shape, before_custom_call_outputs_array.len);
        res_types = stdx.stackSlice(8, mlir.Type, before_custom_call_outputs_array.len);

        for (before_custom_call_outputs_array, 0..) |o, i| {
            res_shapes[i] = o.shape();
            custom_call_inputs[i] = o.value();
            res_types[i] = mlir.ext.RankedTensorType.fromShape(mlir_ctx, o.shape()).as(mlir.Type);
        }
    } else {
        // todo : check if args len != outputs len
        log.warn("{s} / No beforeCustomCall function found, we will expect that the return type of call func will be like args type", .{op_name});
        custom_call_inputs = stdx.stackSlice(8, mlir.Value, inputs.len);

        for (custom_call_inputs, inputs) |*input, tensor_| {
            input.* = tensor_.value();
        }

        res_shapes = stdx.stackSlice(8, Shape, inputs.len);
        res_types = stdx.stackSlice(8, mlir.Type, inputs.len);
        inline for (inputs, 0..) |input, i| {
            res_shapes[i] = input.shape();
            res_types[i] = mlir.ext.RankedTensorType.fromShape(mlir_ctx, input.shape()).as(mlir.Type);
        }
    }

    const frontend_attributes = mlir.Attribute.dict(mlir_ctx, &.{
        .{ "_xla_compute_type", .string(mlir_ctx, "host") },
        .{ "_xla_buffer_placement", .string(mlir_ctx, @tagName(Buffer.Memory.host_pinned.toPjrtMemory())) },
    });

    const op = dialect.stablehlo.custom_call(
        mlir_ctx,
        custom_call_inputs,
        .{
            .call_target_name = target_name,
            .api_version = .typed_ffi,
            .backend_config = mlir.Attribute.dict(mlir_ctx, &.{}),
            .additional_attributes = &.{.{ "mhlo.frontend_attributes", frontend_attributes }},
            .has_side_effect = true,
            .output_operand_aliases = &.{},
        },
        res_types,
        mlir_ctx.location(@src()),
    );

    if (CustomCallOutputType(custom_op) == Tensor) {
        return Tensor._result(res_shapes[0], op.result(0));
    }

    var custom_call_outputs: CustomCallOutputType(custom_op) = undefined;

    for (&custom_call_outputs, res_shapes, 0..) |*r, shape_, i| {
        r.* = Tensor._result(shape_, op.result(i));
    }

    return custom_call_outputs;
}

fn proxy(comptime custom_op: type) pjrt.ffi.Handler {
    const op_name = @typeName(custom_op);
    _ = op_name; // autofix

    return struct {
        const Self = @This();

        pub fn proxy(call_frame: *pjrt.ffi.CallFrame) callconv(.C) ?*pjrt.ffi.Error {
            if (call_frame.registeringHook()) return null;
            const execution_context = call_frame.ctx.?;
            const device_id = execution_context.getDeviceOrdinal(call_frame.api.?) catch unreachable;
            const user_ctx: *custom_op = pjrt.ffi.ExecutionContext.Context(custom_op).get(execution_context, call_frame.api.?) catch unreachable;

            const platform_: *const Platform = user_ctx._platform;
            const pjrt_api = platform_.pjrt_api;
            const pjrt_client = platform_.pjrt_client;

            const device = pjrt_client.getAddressableDevices(pjrt_api)[@as(u32, @intCast(device_id))];
            _ = device; // autofix
            const ffi_buffer = call_frame.args.buffers()[0];
            // const buffer_on_device = Buffer.asViewOfDeviceBuffer(platform_.*, getShape(ffi_buffer), null, ffi_buffer.data.asPtr());
            // const buffer_pinned = buffer_on_device.copyToMemory(platform_.*, .host_pinned) catch unreachable;
            // _ = buffer_pinned; // autofix

            const ctx = call_frame.ctx;
            const stream = call_frame.api.?.stream(@constCast(ctx));
            _ = stream; // autofix
            const host = HostBuffer.empty(std.heap.smp_allocator, getShape(ffi_buffer)) catch unreachable;
            cuda.memcpyToHostAsync(@constCast(host.data), ffi_buffer.data, null);
            _ = cuda.streamSynchronize(null);
            // log.info("{s} / Proxy call_frame {*} on device {} (ordinal: {d})", .{ op_name, call_frame, device, device_id });

            // var callback_args: std.meta.ArgsTuple(@TypeOf(custom_op.call)) = undefined;
            // callback_args[0] = user_ctx;

            // inline for (1..callback_args.len) |i| {
            //     const ffi_buffer = call_frame.args.buffers()[i - 1];
            //     log.info("{s} / FFI Buffer arg: {} ({d}/{d})", .{ op_name, ffi_buffer, i, call_frame.args.len });

            //     const buffer_on_device = Buffer.asViewOfDeviceBuffer(platform_.*, getShape(ffi_buffer), null, ffi_buffer.data.asPtr());
            //     const buffer_device_pjrt_buffer = buffer_on_device._shards.get(0).getOpaqueDeviceMemoryDataPointer(pjrt_api);
            //     log.info("{s} / ZML Buffer on device: {} {any} ({d}/{d})", .{ op_name, buffer_on_device, buffer_device_pjrt_buffer, i, call_frame.args.len });

            //     const buffer_pinned = buffer_on_device.copyToMemory(platform_.*, .host_pinned) catch unreachable;
            //     const buffer_pinned_pjrt_buffer = buffer_pinned._shards.get(0).getOpaqueDeviceMemoryDataPointer(pjrt_api);
            //     log.info("{s} / ZML Buffer pinned: {} {any} ({d}/{d})", .{ op_name, buffer_pinned, buffer_pinned_pjrt_buffer, i, call_frame.args.len });

            //     callback_args[i] = buffer_pinned;
            // }

            // const zml_results = @call(.auto, custom_op.call, callback_args) catch |err| {
            //     stdx.debug.panic("Error while calling {any} call func: {any}\n", .{ @typeName(custom_op), err });
            // };

            // var results_array: []Buffer = undefined;
            // if (@TypeOf(zml_results) == Buffer) {
            //     results_array = stdx.stackSlice(8, Buffer, 1);
            //     results_array[0] = zml_results;
            // } else {
            //     results_array = stdx.stackSlice(8, Buffer, @typeInfo(@TypeOf(zml_results)).@"struct".fields.len);
            //     inline for (@typeInfo(@TypeOf(zml_results)).@"struct".fields, 0..) |field, i| {
            //         results_array[i] = @field(zml_results, field.name);
            //     }
            // }

            // for (results_array, 0..) |zml_buffer, i| {
            //     const ffi_buffer = call_frame.results.buffers()[i];
            //     _ = ffi_buffer; // autofix
            //     log.info("{s} / FFI Buffer result ptr: {} ({d}/{d})", .{ op_name, call_frame.results.ptr[i], i + 1, call_frame.results.len });

            //     const zml_buffer_pjrt_buffer = zml_buffer._shards.get(0).getOpaqueDeviceMemoryDataPointer(pjrt_api) catch unreachable;
            //     log.info("{s} / ZML Buffer result: {} {any} ({d}/{d})", .{ op_name, zml_buffer, zml_buffer_pjrt_buffer, i + 1, call_frame.results.len });

            //     // cuda.memcpyToDeviceBlocking(ffi_buffer.data, zml_buffer.dataInMemory() catch unreachable);
            // }

            return null;
        }
    }.proxy;
}

fn getShape(ffi_buffer: *const pjrt.ffi.Buffer) Shape {
    const dt: DataType = switch (ffi_buffer.dtype) {
        .invalid => stdx.debug.panic("Invalid FFI dtype {any} used by {any}", .{ ffi_buffer.dtype, ffi_buffer }),
        .pred => .bool,
        .s8 => .i8,
        .s16 => .i16,
        .s32 => .i32,
        .s64 => .i64,
        .token, .f8e4m3, .f8e3m4 => stdx.debug.panic("Unsupported FFI dtype {any} used by {any}", .{ ffi_buffer.dtype, ffi_buffer }),
        inline else => |t| @field(DataType, @tagName(t)),
    };
    return Shape.init(ffi_buffer.dims(), dt);
}
