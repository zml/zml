const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const builtin = @import("builtin");

const asynk = @import("asynk");
const stdx = @import("stdx");

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

const CallbackT = Buffer;

pub fn CustomCallInputType(custom_op: type) type {
    const ArgsT = stdx.meta.FnArgs(custom_op.call);
    if (@typeInfo(ArgsT) != .@"struct") {
        @compileError("Expected struct type");
    }
    return [@typeInfo(ArgsT).@"struct".fields.len - 1]Tensor;
}

pub fn CustomCallOutputType(custom_op: type) type {
    const ReturnT = stdx.meta.FnReturnNoError(custom_op.call);

    if (@typeInfo(ReturnT) != .@"struct") {
        @compileError("Expected struct type");
    }

    if (ReturnT == CallbackT) {
        return Tensor;
    }

    return [@typeInfo(ReturnT).@"struct".fields.len]Tensor;
}

pub fn custom_call(
    comptime custom_op: type,
    inputs: CustomCallInputType(custom_op),
) CustomCallOutputType(custom_op) {
    stdx.debug.assert(@hasDecl(custom_op, "call"), "custom_op must have a call method", .{});

    const ctx = inputs[0].getContext();
    const mlir_ctx = ctx.mlirCtx();
    const platform_ = ctx._platform;
    const pjrt_api = platform_.pjrt_api;
    const ffi = pjrt_api.ffi().?;
    const registry = ffi.customCallRegistry().?;
    const ffi_func = proxy(custom_op);
    const target_name = "callback_" ++ @typeName(custom_op);
    registry.registerFfi(pjrt_api, target_name, @tagName(ctx._platform.target), &ffi_func) catch unreachable;
    log.info("Registered custom call {s} for {s} with proxy func {*}", .{ target_name, @typeName(custom_op), &ffi_func });

    const custom_call_inputs = stdx.stackSlice(8, mlir.Value, inputs.len);
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
            std.debug.panic("Error in {any} beforeCustomCall func: {any}\n", .{ @typeName(custom_op), err });
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

        res_shapes = stdx.stackSlice(8, Shape, before_custom_call_outputs_array.len);
        res_types = stdx.stackSlice(8, mlir.Type, before_custom_call_outputs_array.len);

        for (before_custom_call_outputs_array, 0..) |o, i| {
            res_shapes[i] = o.shape();
            res_types[i] = mlir.ext.RankedTensorType.fromShape(mlir_ctx, o.shape()).as(mlir.Type);
        }

        for (before_custom_call_outputs_array, 0..) |output, i| {
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

    const frontend_attributes = mlir.Attribute.dict(mlir_ctx, &.{
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

    // if (CustomCallOutputType(custom_op) == void) {
    //     return;
    // }

    if (CustomCallOutputType(custom_op) == Tensor) {
        var ret = Tensor._result(res_shapes[0], op.result(0));
        ret._output_memory_kind = .host_pinned;
        return ret;
    }

    var custom_call_outputs: CustomCallOutputType(custom_op) = undefined;

    for (&custom_call_outputs, res_shapes, 0..) |*r, shape_, i| {
        r.* = Tensor._result(shape_, op.result(i));
    }

    return custom_call_outputs;
}

fn view_buf(
    platform_: *const Platform,
    device: *const pjrt.Device,
    buffer_desc: *const pjrt.ffi.Buffer,
) Buffer {
    const pjrt_api = platform_.pjrt_api;
    const pjrt_client = platform_.pjrt_client;
    const buffer_shape = getShape(buffer_desc);

    const kind: Buffer.Memory = .device;
    const memories = device.addressableMemories(pjrt_api);
    var selected_mem: *const pjrt.Memory = undefined;
    for (memories) |mem| {
        if (mem.kind(platform_.pjrt_api) == kind.toPjrtMemory()) {
            selected_mem = mem;
        }
    }

    const minor_to_major: [Shape.MAX_RANK]i64 = comptime blk: {
        var res: [Shape.MAX_RANK]i64 = undefined;
        for (0..Shape.MAX_RANK) |i| {
            res[i] = @intCast(Shape.MAX_RANK - i - 1);
        }
        break :blk res;
    };

    std.debug.print("view_buf: {any} {any} {any}\n", .{ buffer_desc, buffer_shape, selected_mem.kind(pjrt_api) });
    const pjrt_buffer = pjrt_client.createViewOfDeviceBuffer(pjrt_api, .{
        .data = buffer_desc.data,
        .element_type = buffer.bufferTypeFromDtype(buffer_shape.dtype()),
        .dims = buffer_shape.dims(),
        .memory = selected_mem,
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
            const device_id = execution_context.getDeviceOrdinal(call_frame.api.?) catch unreachable;
            const user_ctx: *custom_op = pjrt.ffi.ExecutionContext.Context(custom_op).get(execution_context, call_frame.api.?) catch unreachable;

            const platform_: *const Platform = user_ctx._platform;
            const pjrt_api = platform_.pjrt_api;
            const pjrt_client = platform_.pjrt_client;

            const device = pjrt_client.getAddressableDevices(pjrt_api)[@as(u32, @intCast(device_id))];
            log.info("Proxy call_frame {*} on device {} (ordinal: {d})", .{ call_frame, device, device_id });

            var args: std.meta.ArgsTuple(@TypeOf(custom_op.call)) = undefined;
            args[0] = user_ctx;
            inline for (1..args.len) |i| {
                const buffer_desc = call_frame.args.get(i - 1);

                log.info("FFI Buffer {} ({d}/{d})", .{ buffer_desc, i, call_frame.args.len });
                // log.info("Pouet >>>>>>>>>> {d}", .{buffer_desc.data[0..1]});
                const buffer_shape = getShape(buffer_desc);
                _ = buffer_shape; // autofix

                // const host_buffer = HostBuffer.fromBytes(buffer_shape, buffer_desc.data[0..buffer_shape.byteSize()]);
                // log.info("Host buffer {any}", .{host_buffer.items(f32)});
                const zml_buf = view_buf(platform_, device, buffer_desc);
                log.info(">> zml_buf {any}", .{zml_buf.getMemory().kind(pjrt_api)});
                // std.debug.print("zml_buf YAYA {any}: {any}\n", .{ i, zml_buf.getValueFromDataInMemory(f32) catch unreachable });
                // const zml_buf_mem = zml_buf.copyToMemory(platform_.*, .host_pinned) catch unreachable;
                // const event = zml_buf_mem._shards.get(0).getReadyEvent(pjrt_api);
                // _ = event; // autofix
                // std.debug.print(" >> is ready : {any}\n", .{event.?.isReady(pjrt_api)});
                // std.time.sleep(std.time.ns_per_s);
                // event.?.await_(pjrt_api) catch unreachable;
                // std.debug.print(" >> is ready2 : {any}\n", .{event.?.isReady(pjrt_api)});
                // const dd = zml_buf_mem.getValueFromDataInMemory(f32) catch unreachable;
                // std.debug.print("zml_buf YOYO {any}: {any}\n", .{ i, dd });
                // log.info("zml_buf mem {any}", .{zml_buf_mem.getMemory().kind(pjrt_api)});

                // defer zml_buf.deinit();

                // const buffer_shape = getShape(buffer_desc);
                // const b = HostBuffer.fromBytes(buffer_shape, buffer_desc.data[0..buffer_shape.byteSize()]);

                // const data_ptr: usize = @intFromPtr(b.data.ptr);
                // const page_off = data_ptr % std.heap.page_size_min;
                // const page_start: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(data_ptr - page_off);
                // std.posix.mprotect(page_start[0 .. b.data.len + page_off], std.posix.PROT.WRITE | std.posix.PROT.READ) catch |e| {
                //     log.err("Failed to protect memory of buffer {}: {}", .{ b, e });
                // };

                args[i] = zml_buf;
                // std.debug.print("input buffer {any}: {any}\n", .{ i, b.data });
            }

            const outputs = @call(.auto, custom_op.call, args) catch |err| {
                std.debug.print("Error on custom call {any}: {any}\n", .{ @typeName(custom_op), err });
                const ffi_error = pjrt.ffi.Error.create(call_frame.api.?, pjrt.ffi.ErrorCode.unknown, @errorName(err));
                return ffi_error;
            };

            std.debug.print("output: {any}\n", .{outputs});

            // const output_buffers = stdx.stackSlice(8, CallbackT, call_frame.results.len);
            // for (output_buffers, 0..) |*b, i| {
            //     b.* = view_buf2(pjrt_api, pjrt_client, call_frame.results.get(i));
            // }

            // const output_buffers = stdx.stackSlice(8, CallbackT, call_frame.results.len);
            // for (output_buffers, 0..) |*b, i| {
            //     const buffer_desc = call_frame.results.get(i);
            //     const buffer_shape = getShape(buffer_desc);
            //     b.* = HostBuffer.fromBytes(buffer_shape, buffer_desc.data[0..buffer_shape.byteSize()]);

            //     const data_ptr: usize = @intFromPtr(b.data.ptr);
            //     const page_off = data_ptr % std.heap.page_size_min;
            //     const page_start: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(data_ptr - page_off);
            //     std.posix.mprotect(page_start[0 .. b.data.len + page_off], std.posix.PROT.WRITE | std.posix.PROT.READ) catch |e| {
            //         log.err("Failed to protect memory of buffer {}: {}", .{ b, e });
            //     };

            //     std.debug.print("output buffer {any}: {any}\n", .{ i, b.data });
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
