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

pub const FFIBuffer = pjrt.ffi.Buffer;
pub const FFIRets = pjrt.ffi.Rets;
pub const FFIStream = pjrt.ffi.Stream;
// todo: check deallocs

pub fn CustomCallCompilationInputType(custom_op: type) type {
    const ArgsT = stdx.meta.FnArgs(custom_op.compile);
    if (@typeInfo(ArgsT) != .@"struct") {
        @compileError("Expected struct type");
    }
    // todo check typeof first input then others args
    return [@typeInfo(ArgsT).@"struct".fields.len]Tensor;
}

pub fn CustomCallCompilationOutputType(custom_op: type) type {
    const ReturnT = stdx.meta.FnResultNoError(custom_op.compile);

    if (@typeInfo(ReturnT) != .@"struct") {
        @compileError("Expected struct type");
    }

    if (ReturnT == Buffer) {
        return Tensor;
    }
    return [@typeInfo(ReturnT).@"struct".fields.len]Tensor;
}

pub fn lenOutputBeforeCustomCall(custom_op: type) usize {
    const ReturnT = stdx.meta.FnResultNoError(custom_op.compile);

    if (@typeInfo(ReturnT) != .@"struct") {
        @compileError("Expected struct type");
    }

    if (ReturnT == Tensor) {
        return 1;
    }
    return @typeInfo(ReturnT).@"struct".fields.len;
}

pub fn CustomCallInputsType(custom_op: type) type {
    const ArgsT = stdx.meta.FnArgs(custom_op.call);
    if (@typeInfo(ArgsT) != .@"struct") {
        @compileError("Expected struct type");
    }
    // todo check typeof first input then others args
    return [@typeInfo(ArgsT).@"struct".fields.len - 1]Tensor;
}

pub fn custom_call(
    comptime custom_op: type,
    inputs: CustomCallInputsType(custom_op),
    res_shapes_: []const Shape,
) []Tensor {
    stdx.debug.assert(@hasDecl(custom_op, "call"), "custom_op must have a call method", .{});
    const op_name = @typeName(custom_op);

    const ctx = CompilationContext.current();
    const allocator = ctx.allocator();
    const mlir_ctx = ctx.mlirCtx();
    const platform_ = ctx._platform;
    const pjrt_api = platform_.pjrt_api;
    const ffi = pjrt_api.ffi().?;
    const registry = ffi.customCallRegistry().?;
    const ffi_func = proxy(custom_op);
    const target_name = "callback_" ++ op_name;

    registry.registerFfi(pjrt_api, target_name, @tagName(platform_.target), &ffi_func, .{}) catch unreachable;
    log.info("{s} / Registered custom call with target_name \"{s}\" with proxy func {*}", .{ op_name, target_name, &ffi_func });

    // var custom_call_inputs = allocator.alloc(mlir.Value, 8) catch unreachable;
    // var res_shapes = allocator.alloc(Shape, 8) catch unreachable;
    // var res_types = allocator.alloc(mlir.Type, 8) catch unreachable;

    // if (@typeInfo(@TypeOf(custom_op.compile)) != .@"fn") {
    //     stdx.debug.panic("compile must be a function");
    // }

    // var args: std.meta.ArgsTuple(@TypeOf(custom_op.compile)) = undefined;
    // inline for (0..args.len) |i| {
    //     args[i] = inputs[i];
    // }

    // const compile_ouputs = ctx.callFunc(@typeName(custom_op), custom_op.compile, args) catch |err| {
    //     stdx.debug.panic("Error in {any} beforeCustomCall func: {any}\n", .{ @typeName(custom_op), err });
    // };
    // var compile_ouputs_array = allocator.alloc(Shape, 8) catch unreachable;

    // if (@TypeOf(compile_ouputs) == Shape) {
    //     compile_ouputs_array[0] = compile_ouputs;
    // } else {
    //     inline for (@typeInfo(@TypeOf(compile_ouputs)).@"struct".fields, 0..) |field, i| {
    //         compile_ouputs_array[i] = @field(compile_ouputs, field.name);
    //     }
    // }

    // for (compile_ouputs_array[0..res_shapes_.len]) |*rest_t| {
    //     log.warn("compile_ouputs_array: {any}", .{rest_t});
    // }

    // for (inputs, 0..) |t, i| {
    //     custom_call_inputs[i] = t.value();
    // }

    // for (compile_ouputs_array[0..res_shapes_.len], 0..) |sh, i| {
    //     res_shapes[i] = sh;
    //     res_types[i] = mlir.ext.RankedTensorType.fromShape(mlir_ctx, sh).as(mlir.Type);
    // }

    const custom_call_inputs = allocator.alloc(mlir.Value, 8) catch unreachable;
    for (inputs, 0..) |t, i| {
        custom_call_inputs[i] = t.value();
    }

    const res_types = allocator.alloc(mlir.Type, 8) catch unreachable;
    for (res_shapes_, 0..) |sh, i| {
        res_types[i] = mlir.ext.RankedTensorType.fromShape(mlir_ctx, sh).as(mlir.Type);
    }

    const frontend_attributes = mlir.Attribute.dict(mlir_ctx, &.{
        .{ "_xla_compute_type", .string(mlir_ctx, "host") },
        .{ "_xla_buffer_placement", .string(mlir_ctx, @tagName(Buffer.Memory.host_pinned.toPjrtMemory())) },
    });
    const op = dialect.stablehlo.custom_call(
        mlir_ctx,
        custom_call_inputs[0..inputs.len],
        .{
            .call_target_name = target_name,
            .api_version = .typed_ffi,
            .backend_config = mlir.Attribute.dict(mlir_ctx, &.{}),
            .additional_attributes = &.{.{ "mhlo.frontend_attributes", frontend_attributes }},
            .has_side_effect = true,
            .output_operand_aliases = &.{},
        },
        res_types[0..res_shapes_.len],
        mlir_ctx.location(@src()),
    );

    // if (CustomCallCompilationOutputType(custom_op) == Tensor) {
    //     return Tensor._result(res_shapes_[0], op.result(0));
    // }

    var custom_call_outputs = allocator.alloc(Tensor, 8) catch unreachable;

    for (custom_call_outputs[0..res_shapes_.len], res_shapes_, 0..) |*t, sh, i| {
        t.* = Tensor._result(sh, op.result(i));
    }

    return custom_call_outputs[0..res_shapes_.len];
}

fn proxy(comptime custom_op: type) pjrt.ffi.Handler {
    const op_name = @typeName(custom_op);
    _ = op_name; // autofix

    return struct {
        const Self = @This();

        pub fn proxy(call_frame: *pjrt.ffi.CallFrame) callconv(.C) ?*pjrt.ffi.Error {
            if (call_frame.registeringHook()) return null;
            const execution_context = call_frame.ctx.?;
            // const device_id = execution_context.getDeviceOrdinal(call_frame.api) catch unreachable;
            const user_ctx: *custom_op = pjrt.ffi.ExecutionContext.Context(custom_op).get(execution_context, call_frame.api) catch unreachable;
            // const platform_: Platform = user_ctx.platform;
            // const pjrt_api = platform_.pjrt_api;
            // const pjrt_client = platform_.pjrt_client;
            // const tracer = platform_.tracer;
            // _ = tracer; // autofix

            // const device = pjrt_client.getAddressableDevices(pjrt_api)[@as(u32, @intCast(device_id))];
            // _ = device; // autofix
            const ctx = call_frame.ctx;
            const stream = call_frame.api.stream(@constCast(ctx));

            var callback_args: std.meta.ArgsTuple(@TypeOf(custom_op.call)) = undefined;
            callback_args[0] = user_ctx;

            inline for (1..callback_args.len) |i| {
                const ffi_buffer = call_frame.args.buffers()[i - 1];
                callback_args[i] = ffi_buffer;
            }

            user_ctx.*.results = call_frame.results.buffers();
            user_ctx.*.stream = stream;

            @call(.auto, custom_op.call, callback_args) catch |err| {
                stdx.debug.panic("Error while calling {s} call func: {any}\n", .{ @typeName(custom_op), err });
            };

            return null;
        }
    }.proxy;
}

pub fn getShape(ffi_buffer: *const pjrt.ffi.Buffer) Shape {
    const dt: DataType = switch (ffi_buffer.dtype) {
        .invalid => stdx.debug.panic("Invalid FFI dtype {any} used by {any}", .{ ffi_buffer.dtype, ffi_buffer }),
        .pred => .bool,
        .i8 => .i8,
        .i16 => .i16,
        .i32 => .i32,
        .i64 => .i64,
        .token, .f8e4m3, .f8e3m4 => stdx.debug.panic("Unsupported FFI dtype {any} used by {any}", .{ ffi_buffer.dtype, ffi_buffer }),
        inline else => |t| @field(DataType, @tagName(t)),
    };
    return Shape.init(ffi_buffer.dims(), dt);
}
