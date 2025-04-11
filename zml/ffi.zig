const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const builtin = @import("builtin");

const stdx = @import("stdx");

const dialect = struct {
    const stablehlo = @import("mlir/dialects").stablehlo;
};
const dtype = @import("dtype.zig");
const pjrt = @import("pjrtx.zig");
const buffer = @import("buffer.zig");
const mlir = @import("mlir.zig");
const module = @import("module.zig");
const ops = @import("ops.zig");
const tensor = @import("tensor.zig");
const shape = @import("shape.zig");
const Buffer = buffer.Buffer;
const CompilationContext = module.CompilationContext;
const DataType = dtype.DataType;
const Tensor = tensor.Tensor;
const Shape = shape.Shape;

const log = std.log.scoped(.@"zml/custom_call");

fn nbInputTensors(comptime args: type) u32 {
    if (@typeInfo(args) != .@"struct") {
        @compileError("Expected struct type");
    }
    return @typeInfo(args).@"struct".fields.len - 1;
}

fn nbOutputTensors(comptime args: type) u32 {
    if (@typeInfo(args) != .@"struct") {
        @compileError("Expected struct type");
    }
    return @typeInfo(args).@"struct".fields.len;
}

fn isCustomCallContainer(custom_op: type) bool {
    return @hasDecl(custom_op, "call");
}

fn hasBeforeCustomCallHook(custom_op: type) bool {
    return @hasDecl(custom_op, "beforeCustomCall");
}

pub fn CustomCallInputType(custom_op: type) type {
    return [nbInputTensors(stdx.meta.FnArgs(custom_op.call))]Tensor;
}

pub fn CustomCallOutputType(custom_op: type) type {
    return [nbOutputTensors(stdx.meta.FnReturnNoError(custom_op.call))]Tensor;
}

pub fn custom_call(
    comptime custom_op: type,
    inputs: CustomCallInputType(custom_op),
) CustomCallOutputType(custom_op) {
    stdx.debug.assert(isCustomCallContainer(custom_op), "custom_op must have a call method", .{});

    const ctx = inputs[0].getContext();
    const mlir_ctx = ctx.mlirCtx();
    const platform = ctx._platform;
    const pjrt_api = platform.pjrt_api;
    const ffi = pjrt_api.ffi().?;
    const registry = ffi.customCallRegistry().?;
    const ffi_func = proxy(custom_op);
    const target_name = "callback_" ++ @typeName(custom_op);
    registry.registerFfi(pjrt_api, target_name, @tagName(ctx._platform.target), &ffi_func) catch unreachable;
    log.info("Registered custom call {s} for {s} with proxy func {*}", .{ target_name, @typeName(custom_op), &ffi_func });

    const custom_call_inputs = stdx.stackSlice(8, mlir.Value, inputs.len);
    var res_shapes: []Shape = undefined;
    var res_types: []mlir.Type = undefined;

    if (hasBeforeCustomCallHook(custom_op)) {
        const before_custom_call = custom_op.beforeCustomCall;
        if (@typeInfo(@TypeOf(before_custom_call)) != .@"fn") {
            stdx.debug.panic("beforeCustomCall must be a function");
        }

        var args: std.meta.ArgsTuple(@TypeOf(before_custom_call)) = undefined;
        inline for (0..args.len) |i| {
            args[i] = inputs[i];
        }

        const before_custom_call_outputs = ctx.callFunc(@typeName(custom_op), before_custom_call, args) catch |err| {
            std.debug.panic("Error in {any} beforeCustomCall func: {any}\n", .{ @typeName(custom_op), err });
        };
        std.debug.print("beforeCustomCall outputs: {any}\n", .{before_custom_call_outputs});
        res_shapes = stdx.stackSlice(8, Shape, before_custom_call_outputs.len);
        res_types = stdx.stackSlice(8, mlir.Type, before_custom_call_outputs.len);
        inline for (before_custom_call_outputs, 0..) |o, i| {
            res_shapes[i] = o.shape();
            res_types[i] = mlir.ext.RankedTensorType.fromShape(mlir_ctx, o.shape()).as(mlir.Type);
        }

        inline for (before_custom_call_outputs, 0..) |output, i| {
            custom_call_inputs[i] = output.value();
        }
    } else {
        log.warn("No beforeCustomCall function found for {s}, expecting return type to be args type", .{@typeName(custom_op)});
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

    const op = dialect.stablehlo.custom_call(
        mlir_ctx,
        custom_call_inputs,
        .{
            .call_target_name = target_name,
            .api_version = .typed_ffi,
            .backend_config = mlir.Attribute.dict(mlir_ctx, &.{}),
            .has_side_effect = true,
            .output_operand_aliases = &.{},
        },
        res_types,
        mlir_ctx.location(@src()),
    );

    var custom_call_outputs: CustomCallOutputType(custom_op) = undefined;
    for (&custom_call_outputs, res_shapes, 0..) |*r, shape_, i| {
        r.* = Tensor._result(shape_, op.result(i));
    }

    return custom_call_outputs;
}

fn view_buf(pjrt_api: *const pjrt.Api, pjrt_client: *const pjrt.Client, buffer_desc: *const pjrt.ffi.Buffer) Buffer {
    const buffer_shape = getShape(buffer_desc);
    const device = pjrt_client.getAddressableDevices(pjrt_api)[0];

    const minor_to_major: [Shape.MAX_RANK]i64 = comptime blk: {
        var res: [Shape.MAX_RANK]i64 = undefined;
        for (0..Shape.MAX_RANK) |i| {
            res[i] = @intCast(Shape.MAX_RANK - i - 1);
        }
        break :blk res;
    };

    const pjrt_buffer = pjrt_client.createViewOfDeviceBuffer(pjrt_api, .{
        .data = buffer_desc.data,
        .element_type = buffer.bufferTypeFromDtype(buffer_shape.dtype()),
        .dims = buffer_shape.dims(),
        .device = device,
        .layout = .{
            .tiled = .{
                .minor_to_major = minor_to_major[Shape.MAX_RANK - buffer_shape.rank() ..],
                .tile_dims = &.{},
                .tile_dims_sizes = &.{},
            },
        },
        .stream = null,
    }) catch @panic("failed to createViewOfDeviceBuffer");

    var shards: Buffer.Shards = .{};
    shards.appendAssumeCapacity(pjrt_buffer);

    return .{
        ._api = pjrt_api,
        ._shape = buffer_shape,
        ._shards = shards,
    };
}

fn proxy(comptime custom_op: type) pjrt.ffi.Handler {
    return struct {
        const Self = @This();

        pub fn proxy(call_frame: *pjrt.ffi.CallFrame) callconv(.C) ?*pjrt.ffi.Error {
            if (call_frame.registeringHook()) return null;
            const execution_context = call_frame.ctx.?;
            const user_ctx: *custom_op = pjrt.ffi.ExecutionContext.Context(custom_op).get(execution_context, call_frame.api.?) catch unreachable;
            const platform = user_ctx._platform;
            const pjrt_api = platform.pjrt_api;
            const pjrt_client = platform.pjrt_client;

            const input_buffers = stdx.stackSlice(8, Buffer, call_frame.args.len);
            for (input_buffers, 0..) |*b, i| {
                b.* = view_buf(pjrt_api, pjrt_client, call_frame.args.get(i));
            }

            const output_buffers = stdx.stackSlice(8, Buffer, call_frame.results.len);
            for (output_buffers, 0..) |*b, i| {
                b.* = view_buf(pjrt_api, pjrt_client, call_frame.results.get(i));
            }

            var args: std.meta.ArgsTuple(@TypeOf(custom_op.call)) = undefined;
            args[0] = user_ctx;
            inline for (1..args.len) |i| {
                args[i] = input_buffers[i - 1];
            }

            const outputs = @call(.auto, custom_op.call, args) catch |err| {
                std.debug.print("Error on custom call {any}: {any}\n", .{ @typeName(custom_op), err });
                const ffi_error = pjrt.ffi.Error.create(call_frame.api.?, pjrt.ffi.ErrorCode.unknown, @errorName(err));
                return ffi_error;
            };

            if (@typeInfo(@TypeOf(outputs)) == .array) {
                for (outputs, 0..) |output, i| {
                    std.debug.print("output {any}: {any}\n", .{ i, output });
                }
            } else {
                std.debug.print("output: {any}\n", .{outputs});
            }

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
