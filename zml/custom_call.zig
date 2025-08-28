const std = @import("std");

const asynk = @import("async");
const mlir = @import("mlir");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const CompilationContext = @import("module.zig").CompilationContext;
const cuda = @import("context.zig").cuda;
const DataType = @import("dtype.zig").DataType;
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const mlirx = @import("mlirx.zig");
const pjrtx = @import("pjrtx.zig");
const ffi = pjrtx.ffi;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

const dialect = struct {
    const stablehlo = @import("mlir/dialects").stablehlo;
};
const log = std.log.scoped(.@"zml/custom_call");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

pub fn CustomCallInputsType(custom_op: type) type {
    const ArgsT = stdx.meta.FnArgs(custom_op.call);
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
    register_ffi_options: pjrt.FFI.RegisterOptions = .{
        .traits = @enumFromInt(0),
    },
};

pub fn custom_call(
    comptime custom_op: type,
    inputs: CustomCallInputsType(custom_op),
    res_shapes_: []const Shape,
    comptime opts: CustomCallOptions,
) []Tensor {
    stdx.debug.assert(@hasDecl(custom_op, "call"), "custom_op must have a call method", .{});
    const op_name = @typeName(custom_op);

    const ctx = CompilationContext.current();
    const allocator = ctx.allocator();
    const mlir_ctx = ctx.mlirCtx();
    const platform = ctx._platform;
    const pjrt_api = platform.pjrt_api;
    const ffi_ = pjrt_api.ffi();

    if (ffi_ == null) {
        stdx.debug.panic("Custom calls are not supported for target {s}", .{@tagName(platform.target)});
    }

    const ffi_func = customCallProxy(custom_op, opts);
    const target_name = "callback_" ++ op_name;

    ffi_.?.register(pjrt_api, target_name, @tagName(platform.target), &ffi_func, opts.register_ffi_options) catch unreachable;
    log.info("{s} / Registered custom call with target name \"{s}\" with proxy func {*}", .{ op_name, target_name, &ffi_func });

    const custom_call_inputs = allocator.alloc(mlir.Value, 8) catch unreachable;
    for (inputs, 0..) |t, i| {
        custom_call_inputs[i] = t.value();
    }

    const res_types = allocator.alloc(mlir.Type, 8) catch unreachable;
    for (res_shapes_, 0..) |sh, i| {
        res_types[i] = mlirx.tensorType(mlir_ctx, sh);
    }

    const frontend_attributes: mlir.Attribute = .dict(mlir_ctx, &.{});

    const op = dialect.stablehlo.custom_call(
        mlir_ctx,
        custom_call_inputs[0..inputs.len],
        .{
            .call_target_name = target_name,
            .api_version = .typed_ffi,
            .backend_config = mlir.Attribute.dict(mlir_ctx, &.{}),
            .additional_attributes = &.{.{ "mhlo.frontend_attributes", frontend_attributes }},
            .has_side_effect = true,
            .output_operand_aliases = opts.output_operand_aliases,
        },
        res_types[0..res_shapes_.len],
        mlir_ctx.location(@src()),
    );

    var custom_call_outputs = allocator.alloc(Tensor, 8) catch unreachable;

    for (custom_call_outputs[0..res_shapes_.len], res_shapes_, 0..) |*t, sh, i| {
        t.* = Tensor._result(sh, op.result(i));
    }

    return custom_call_outputs[0..res_shapes_.len];
}

fn customCallProxy(CustomOp: type, comptime opts: CustomCallOptions) ffi.Handler {
    return struct {
        pub fn cb(call_frame: *ffi.CallFrame) callconv(.c) ?*ffi.Error {
            return customCallImpl(CustomOp, opts, call_frame);
        }
    }.cb;
}

fn customCallImpl(comptime CustomOp: type, opts: CustomCallOptions, call_frame: *ffi.CallFrame) ?*ffi.Error {
    if (call_frame.registeringHook()) return null;

    const execution_context = call_frame.ctx;
    log.info("Custom call {s} called ! {*}", .{ @typeName(CustomOp), call_frame });
    const user_ctx: *CustomOp = execution_context.getContext(CustomOp, call_frame.api) catch {
        return .create(call_frame.api, .internal, "failed to fetch user context for custom_call" ++ @typeName(CustomOp));
    };
    const platform: Platform = user_ctx.platform;

    if (@hasField(CustomOp, "stream") and platform.target != .cpu) {
        const stream = call_frame.api.stream(execution_context);
        user_ctx.stream = stream;
    }

    var callback_args: std.meta.ArgsTuple(@TypeOf(CustomOp.call)) = undefined;
    callback_args[0] = user_ctx;

    inline for (1..callback_args.len) |i| {
        const ffi_buffer = call_frame.args.buffers()[i - 1];
        const ffi_buffer_shape = getShape(ffi_buffer);
        var zml_buffer: Buffer = undefined;

        // TODO reconder using PJRT apis at all from pjrt callbacks.
        if (platform.target == .cpu) {
            zml_buffer = Buffer.asViewOfHostBuffer(platform, HostBuffer.fromBytes(ffi_buffer_shape, ffi_buffer.data[0..ffi_buffer_shape.byteSize()]));
        } else {
            zml_buffer = Buffer.asViewOfDeviceBuffer(platform, getShape(ffi_buffer), null, ffi_buffer.data);
        }
        if (opts.copy_inputs_to_host_pinned and platform.target != .cpu) {
            zml_buffer = zml_buffer.copyToMemory(platform, .host_pinned, .{ .wait = true }) catch unreachable;
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

/// Internal custom calls.
/// These are not meant to be used by users, but rather by the library itself.
pub const custom_call_internal_types = [_]type{
    Print,
};

pub fn registerInternalCustomCalls(
    platform: Platform,
) !void {
    inline for (custom_call_internal_types) |custom_call_type| {
        try platform.registerFFIType(custom_call_type);
        log.info("Registered internal custom call {s} with type_id {d}", .{ @typeName(custom_call_type), custom_call_type.type_id });
    }
}

pub const Print = struct {
    pub var type_id: i64 = undefined;

    allocator: std.mem.Allocator,
    platform: Platform,
    stream: *pjrt.Stream = undefined,

    results: [1]Buffer = undefined,

    pub fn init(platform: Platform) !Print {
        return .{
            .allocator = std.heap.smp_allocator,
            .platform = platform,
        };
    }

    pub fn call(_: *Print, input: Buffer) !void {
        std.log.defaultLog(.info, .zml, "Device buffer: {f}: {d}", .{ input, input.asHostBuffer() });
    }
};
