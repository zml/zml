const std = @import("std");

const c = @import("c");
const pjrt = @import("pjrt");
const upb = @import("upb");

const log = std.log.scoped(.@"zml/ffi");

// Shared helpers for custom call registration:
// - PJRT FFI handler registration
// - Custom partitioner callback registration
// They are intentionally independent so custom calls can use either one.

pub const Header = pjrt.CustomPartitioner.VersionAndError;
pub const PartitionArgs = pjrt.CustomPartitioner.PartitionArgs;
pub const InferShardingFromOperandsArgs = pjrt.CustomPartitioner.InferShardingFromOperandsArgs;
pub const String = pjrt.CustomPartitioner.String;

pub fn noopDtor(_: [*c]pjrt.CustomPartitioner.Callbacks) callconv(.c) void {}

pub fn noopPropagateUserSharding(
    _: [*c]pjrt.CustomPartitioner.Callbacks,
    args_: [*c]pjrt.CustomPartitioner.PropagateUserShardingArgs,
) callconv(.c) void {
    if (args_ == null) return;
    initHeader(&args_[0].header);
}

fn noopCleanup(_: ?*anyopaque) callconv(.c) void {}

pub fn initHeader(header: *Header) void {
    header.* = .{
        .api_version = pjrt.CustomPartitioner.api_version,
        .data = null,
        .cleanup_fn = &noopCleanup,
        .has_error = false,
        .code = c.PJRT_Error_Code_OK,
        .error_msg = .{ .data = null, .size = 0 },
    };
}

pub fn setError(
    allocator: std.mem.Allocator,
    header: *Header,
    code: pjrt.ErrorCode,
    comptime fmt: []const u8,
    args: anytype,
) void {
    initHeader(header);

    const msg = std.fmt.allocPrint(allocator, fmt, args) catch {
        header.has_error = true;
        header.code = c.PJRT_Error_Code_INTERNAL;
        header.error_msg = .{ .data = "OOM", .size = 3 };
        return;
    };

    header.has_error = true;
    header.code = @intFromEnum(code);
    header.error_msg = .{ .data = msg.ptr, .size = msg.len };
}

pub fn setStaticError(header: *Header, code: pjrt.ErrorCode, msg: []const u8) void {
    initHeader(header);
    header.has_error = true;
    header.code = @intFromEnum(code);
    header.error_msg = .{ .data = msg.ptr, .size = msg.len };
}

pub fn inferResultShardingFromOperand(
    allocator: std.mem.Allocator,
    args: *InferShardingFromOperandsArgs,
    operand_index: usize,
    comptime context_name: []const u8,
) void {
    initHeader(&args.header);

    if (args.num_args <= operand_index or !args.op_args[operand_index].has_sharding) {
        args.has_result_sharding = false;
        args.result_sharding = .{ .data = null, .size = 0 };
        return;
    }

    const sharding = args.op_args[operand_index].sharding.data[0..args.op_args[operand_index].sharding.size];
    const sharding_copy = allocator.dupe(u8, sharding) catch {
        setError(allocator, &args.header, .internal, "{s}: failed duplicating infer_sharding result", .{context_name});
        return;
    };

    args.has_result_sharding = true;
    args.result_sharding = .{
        .data = sharding_copy.ptr,
        .size = sharding_copy.len,
    };
}

pub fn duplicateOperandShardings(
    allocator: std.mem.Allocator,
    header: *Header,
    args: *const PartitionArgs,
    comptime context_name: []const u8,
) ?[]String {
    const out_arg_shardings = allocator.alloc(String, args.num_args) catch {
        setError(allocator, header, .internal, "{s}: failed allocating args_sharding", .{context_name});
        return null;
    };
    for (0..args.num_args) |i| {
        const op_arg = args.op_args[i];
        if (!op_arg.has_sharding) {
            setError(allocator, header, .invalid_argument, "{s}: partition callback requires all operand shardings", .{context_name});
            return null;
        }
        const sharding = op_arg.sharding.data[0..op_arg.sharding.size];
        const sharding_copy = allocator.dupe(u8, sharding) catch {
            setError(allocator, header, .internal, "{s}: failed duplicating operand sharding", .{context_name});
            return null;
        };
        out_arg_shardings[i] = .{
            .data = sharding_copy.ptr,
            .size = sharding_copy.len,
        };
    }
    return out_arg_shardings;
}

pub fn duplicateSharding(
    allocator: std.mem.Allocator,
    header: *Header,
    sharding: []const u8,
    comptime context_name: []const u8,
    comptime err_label: []const u8,
) ?[]u8 {
    return allocator.dupe(u8, sharding) catch {
        setError(allocator, header, .internal, "{s}: failed duplicating {s}", .{ context_name, err_label });
        return null;
    };
}

fn xlaPrimitiveTypeToMlirType(element_type: i32) ?[]const u8 {
    return switch (element_type) {
        c.xla_PRED => "i1",
        c.xla_S8 => "i8",
        c.xla_S16 => "i16",
        c.xla_S32 => "i32",
        c.xla_S64 => "i64",
        c.xla_U8 => "ui8",
        c.xla_U16 => "ui16",
        c.xla_U32 => "ui32",
        c.xla_U64 => "ui64",
        c.xla_F16 => "f16",
        c.xla_F32 => "f32",
        c.xla_F64 => "f64",
        c.xla_BF16 => "bf16",
        c.xla_C64 => "complex<f32>",
        c.xla_C128 => "complex<f64>",
        c.xla_F8E5M2 => "f8E5M2",
        c.xla_F8E4M3FN => "f8E4M3FN",
        c.xla_F8E4M3B11FNUZ => "f8E4M3B11FNUZ",
        c.xla_F8E5M2FNUZ => "f8E5M2FNUZ",
        c.xla_F8E4M3FNUZ => "f8E4M3FNUZ",
        c.xla_F8E4M3 => "f8E4M3",
        c.xla_F8E3M4 => "f8E3M4",
        c.xla_F8E8M0FNU => "f8E8M0FNU",
        c.xla_F4E2M1FN => "f4E2M1FN",
        else => null,
    };
}

fn shardCountForAxis(op_sharding: ?*const c.xla_OpSharding, axis: usize, rank: usize) i64 {
    const sharding = op_sharding orelse return 1;

    const sharding_type = c.xla_OpSharding_type(sharding);
    if (sharding_type == c.xla_OpSharding_REPLICATED or sharding_type == c.xla_OpSharding_MAXIMAL) {
        return 1;
    }
    if (sharding_type != c.xla_OpSharding_OTHER) {
        return 1;
    }

    var tile_dims_len: usize = 0;
    const tile_dims = c.xla_OpSharding_tile_assignment_dimensions(sharding, &tile_dims_len) orelse return 1;
    if (tile_dims_len == 0) return 1;

    var logical_tile_dims_len = tile_dims_len;
    if (c.xla_OpSharding_replicate_on_last_tile_dim(sharding) and logical_tile_dims_len > rank) {
        logical_tile_dims_len -= 1;
    }
    if (axis >= logical_tile_dims_len) return 1;

    const parts = tile_dims[axis];
    return if (parts > 0) parts else 1;
}

fn tensorTypeFromShapeProto(allocator: std.mem.Allocator, shape: *const c.xla_ShapeProto, op_sharding: ?*const c.xla_OpSharding) ![]u8 {
    if (c.xla_ShapeProto_element_type(shape) == c.xla_TUPLE) return error.UnsupportedTupleShape;

    const element_type = xlaPrimitiveTypeToMlirType(c.xla_ShapeProto_element_type(shape)) orelse return error.UnsupportedElementType;

    var dims_len: usize = 0;
    const dims = c.xla_ShapeProto_dimensions(shape, &dims_len);

    var dyn_len: usize = 0;
    const dynamic_dims = c.xla_ShapeProto_is_dynamic_dimension(shape, &dyn_len);

    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();

    try out.writer.writeAll("tensor<");
    for (0..dims_len) |i| {
        const is_dynamic = if (i < dyn_len and dynamic_dims != null) dynamic_dims[i] else false;
        const global_dim = if (dims != null) dims[i] else 0;
        const local_dim = blk: {
            const parts = shardCountForAxis(op_sharding, i, dims_len);
            if (parts <= 1) break :blk global_dim;
            break :blk std.math.divCeil(i64, global_dim, parts) catch return error.UnsupportedSharding;
        };
        if (is_dynamic) {
            try out.writer.writeAll("?x");
        } else {
            try out.writer.print("{d}x", .{local_dim});
        }
    }
    try out.writer.writeAll(element_type);
    try out.writer.writeAll(">");

    return out.toOwnedSlice();
}

pub const PassthroughPartitionSpec = struct {
    module_name: []const u8 = "zml_ffi_partitioner",
    entry_name: []const u8 = "main",
    custom_call_target: []const u8,
    has_side_effect: bool = false,
    // Optional fallback when result sharding is not provided by the partitioner input.
    result_sharding_operand_index: ?usize = null,
};

/// Generic partition callback helper for "custom call passthrough" lowers:
/// builds a tiny MLIR module with one `stablehlo.custom_call` and duplicates
/// operand/result shardings for the partitioner output contract.
pub fn partitionAsPassthroughCustomCall(
    allocator: std.mem.Allocator,
    args: *PartitionArgs,
    spec: PassthroughPartitionSpec,
    comptime context_name: []const u8,
) void {
    initHeader(&args.header);

    const result_sharding = blk: {
        if (args.op_result.has_sharding) {
            break :blk args.op_result.sharding.data[0..args.op_result.sharding.size];
        }
        if (spec.result_sharding_operand_index) |operand_index| {
            if (operand_index < args.num_args and args.op_args[operand_index].has_sharding) {
                break :blk args.op_args[operand_index].sharding.data[0..args.op_args[operand_index].sharding.size];
            }
        }
        setError(allocator, &args.header, .invalid_argument, "{s}: partition callback requires result sharding", .{context_name});
        return;
    };

    var arena_alloc: upb.Allocator = .init(std.heap.c_allocator);
    const upb_arena = c.upb_Arena_Init(null, 0, arena_alloc.inner()) orelse {
        setError(allocator, &args.header, .internal, "{s}: failed to create upb arena", .{context_name});
        return;
    };
    defer c.upb_Arena_Free(upb_arena);

    const result_shape_proto = c.xla_ShapeProto_parse(args.op_result.shape.data, args.op_result.shape.size, upb_arena) orelse {
        setError(allocator, &args.header, .invalid_argument, "{s}: failed to parse result ShapeProto", .{context_name});
        return;
    };
    const result_op_sharding = c.xla_OpSharding_parse(result_sharding.ptr, result_sharding.len, upb_arena) orelse {
        setError(allocator, &args.header, .invalid_argument, "{s}: failed to parse result OpSharding", .{context_name});
        return;
    };

    const result_type = tensorTypeFromShapeProto(allocator, result_shape_proto, result_op_sharding) catch |err| {
        setError(allocator, &args.header, .invalid_argument, "{s}: unsupported result shape for partitioning: {}", .{ context_name, err });
        return;
    };

    const arg_types = allocator.alloc([]const u8, args.num_args) catch {
        setError(allocator, &args.header, .internal, "{s}: failed allocating operand type list", .{context_name});
        return;
    };
    for (0..args.num_args) |i| {
        const op_arg = args.op_args[i];
        const arg_shape_proto = c.xla_ShapeProto_parse(op_arg.shape.data, op_arg.shape.size, upb_arena) orelse {
            setError(allocator, &args.header, .invalid_argument, "{s}: failed to parse arg[{d}] ShapeProto", .{ context_name, i });
            return;
        };
        const arg_op_sharding = if (op_arg.has_sharding)
            c.xla_OpSharding_parse(op_arg.sharding.data, op_arg.sharding.size, upb_arena) orelse {
                setError(allocator, &args.header, .invalid_argument, "{s}: failed to parse arg[{d}] OpSharding", .{ context_name, i });
                return;
            }
        else
            null;
        arg_types[i] = tensorTypeFromShapeProto(allocator, arg_shape_proto, arg_op_sharding) catch |err| {
            setError(allocator, &args.header, .invalid_argument, "{s}: unsupported arg[{d}] shape for partitioning: {}", .{ context_name, i, err });
            return;
        };
    }

    const backend_config = if (args.backend_config.data) |ptr|
        ptr[0..args.backend_config.size]
    else
        "{}";

    var mlir_module_buf = std.Io.Writer.Allocating.init(allocator);
    defer mlir_module_buf.deinit();
    mlir_module_buf.writer.print("module @{s} {{\n", .{spec.module_name}) catch {
        setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
        return;
    };
    mlir_module_buf.writer.print("  func.func public @{s}(", .{spec.entry_name}) catch {
        setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
        return;
    };
    for (arg_types, 0..) |arg_type, i| {
        if (i != 0) {
            mlir_module_buf.writer.writeAll(", ") catch {
                setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
                return;
            };
        }
        mlir_module_buf.writer.print("%arg{d}: {s}", .{ i, arg_type }) catch {
            setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
            return;
        };
    }
    mlir_module_buf.writer.print(") -> ({s}) {{\n", .{result_type}) catch {
        setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
        return;
    };
    mlir_module_buf.writer.print("    %0 = stablehlo.custom_call @{s}(", .{spec.custom_call_target}) catch {
        setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
        return;
    };
    for (0..args.num_args) |i| {
        if (i != 0) {
            mlir_module_buf.writer.writeAll(", ") catch {
                setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
                return;
            };
        }
        mlir_module_buf.writer.print("%arg{d}", .{i}) catch {
            setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
            return;
        };
    }
    mlir_module_buf.writer.print(
        ") {{api_version = 4 : i32, backend_config = {s}, has_side_effect = {s}}} : (",
        .{ backend_config, if (spec.has_side_effect) "true" else "false" },
    ) catch {
        setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
        return;
    };
    for (arg_types, 0..) |arg_type, i| {
        if (i != 0) {
            mlir_module_buf.writer.writeAll(", ") catch {
                setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
                return;
            };
        }
        mlir_module_buf.writer.writeAll(arg_type) catch {
            setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
            return;
        };
    }
    mlir_module_buf.writer.print(") -> {s}\n", .{result_type}) catch {
        setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
        return;
    };
    mlir_module_buf.writer.print("    return %0 : {s}\n", .{result_type}) catch {
        setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
        return;
    };
    mlir_module_buf.writer.writeAll("  }\n}\n") catch {
        setError(allocator, &args.header, .internal, "{s}: failed to build partitioner mlir module", .{context_name});
        return;
    };
    const mlir_module = mlir_module_buf.toOwnedSlice() catch {
        setError(allocator, &args.header, .internal, "{s}: failed to finalize partitioner mlir module", .{context_name});
        return;
    };

    const out_arg_shardings = duplicateOperandShardings(allocator, &args.header, args, context_name) orelse return;
    const result_sharding_copy = duplicateSharding(allocator, &args.header, result_sharding, context_name, "result sharding") orelse return;

    args.mlir_module = .{
        .data = mlir_module.ptr,
        .size = mlir_module.len,
    };
    args.args_sharding = out_arg_shardings.ptr;
    args.result_sharding = .{
        .data = result_sharding_copy.ptr,
        .size = result_sharding_copy.len,
    };
}

/// Declarative registration of one custom call.
/// Any section can be omitted depending on backend capabilities.
pub const Registration = struct {
    name: []const u8,
    ffi: ?FfiRegistration = null,
    partitioner: ?PartitionerRegistration = null,
};

pub const FfiRegistration = struct {
    handler: *const pjrt.ffi.Handler,
    traits: pjrt.ffi.HandlerTraits = .{ .command_buffer_compatible = false },
    platform_name: ?[]const u8 = null,
};

pub const PartitionerRegistration = struct {
    callbacks: ?*pjrt.CustomPartitioner.Callbacks = null,
    batch_partitionable: bool = false,
};

pub fn register(api: *const pjrt.Api, client: *pjrt.Client, registration: Registration) void {
    if (registration.ffi) |ffi_registration| {
        const platform_name = ffi_registration.platform_name orelse client.platformName(api);
        if (api.ffi()) |ffi| {
            ffi.register(api, registration.name, platform_name, ffi_registration.handler, ffi_registration.traits) catch |err| {
                log.warn("Failed to register FFI custom call \"{s}\", error: {}", .{ registration.name, err });
            };
        }
    }

    if (registration.partitioner) |partitioner_registration| {
        if (api.customPartitioner()) |partitioner| {
            if (partitioner_registration.callbacks) |callbacks| {
                partitioner.register(api, registration.name, callbacks) catch |err| {
                    if (err != error.AlreadyExists) {
                        log.warn("Failed to register partitioner for \"{s}\", error: {}", .{ registration.name, err });
                    }
                };
            }
            if (partitioner_registration.batch_partitionable) {
                partitioner.registerBatchPartitionable(api, registration.name) catch |err| {
                    if (err != error.AlreadyExists) {
                        log.warn("Failed to register batch partitionable custom call \"{s}\", error: {}", .{ registration.name, err });
                    }
                };
            }
        }
    }
}

/// Registers multiple custom calls in one place.
pub fn registerAll(api: *const pjrt.Api, client: *pjrt.Client, registrations: []const Registration) void {
    for (registrations) |registration| {
        register(api, client, registration);
    }
}
