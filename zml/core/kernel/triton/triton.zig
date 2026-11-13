const std = @import("std");

const func = @import("mlir/dialects/func");
const mlir = @import("mlir");
const stablehlo = @import("mlir/dialects/stablehlo");
const stdx = @import("stdx");
const triton_builder = @import("dsl/triton/builder");
pub const Builder = triton_builder.Builder;
pub const Value = triton_builder.Value;
pub const DType = triton_builder.DType;
pub const FinishError = triton_builder.FinishError;

const Compiler = @import("../../Compiler.zig");
const DataType = @import("../../dtype.zig").DataType;
const ops = @import("../../ops.zig");
const Shape = @import("../../shape.zig").Shape;
const Tensor = @import("../../tensor.zig").Tensor;

/// Build a struct type with one field per name, all of type `FieldT`.
fn StructOf(comptime names: []const [:0]const u8, comptime FieldT: type) type {
    var name_slices: [names.len][]const u8 = undefined;
    for (names, 0..) |n, i| name_slices[i] = n;
    return @Struct(.auto, null, &name_slices, &@splat(FieldT), &@splat(.{}));
}

fn makeKernelContext(comptime registerDialects: fn (*mlir.DialectRegistry) void) std.mem.Allocator.Error!*mlir.Context {
    mlir.registerPasses("Transforms");

    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();

    registerDialects(registry);

    func.registerExtensions(registry);

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();

    return ctx;
}

fn resolveOutputOperandAliases(
    aliases_opt: anytype,
    operand_offset: i64,
) stdx.BoundedArray(stablehlo.CustomCallOpts.OutputOperandAlias, stablehlo.CustomCallOpts.MAX_RESULTS) {
    var out: stdx.BoundedArray(stablehlo.CustomCallOpts.OutputOperandAlias, stablehlo.CustomCallOpts.MAX_RESULTS) = .empty;

    if (aliases_opt) |a| {
        inline for (@typeInfo(@TypeOf(a)).@"struct".field_names, 0..) |field_name, output_index| {
            if (@field(a, field_name)) |operand| out.appendAssumeCapacity(.{
                .output_index = @intCast(output_index),
                .operand_index = @as(i64, @intCast(@backingInt(operand))) + operand_offset,
            });
        }
    }

    return out;
}

pub fn newContext() std.mem.Allocator.Error!*mlir.Context {
    return makeKernelContext(triton_builder.registerDialects);
}

pub fn from(dt: DataType) DType {
    return switch (dt) {
        .bool => .i1,
        .i8, .u8 => .i8,
        .i16, .u16 => .i16,
        .i32, .u32 => .i32,
        .i64, .u64 => .i64,
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .f8e4m3fn => .f8e4m3fn,
        .f8e4m3fnuz => .f8e4m3fnuz,
        .f8e5m2 => .f8e5m2,
        else => std.debug.panic("zml.kernel.triton.from: dtype {s} has no Triton equivalent", .{@tagName(dt)}),
    };
}

fn Spec(comptime Config: type) type {
    return struct {
        name: [:0]const u8,
        inputs: []const [:0]const u8,
        outputs: []const [:0]const u8,
        run: *const fn (*Builder, Config) FinishError!void,
    };
}

pub fn Kernel(
    comptime ConfigT: type,
    comptime spec: Spec(ConfigT),
) type {
    return struct {
        pub const name: [:0]const u8 = spec.name;
        pub const Config = ConfigT;
        pub const Inputs = StructOf(spec.inputs, Tensor);
        pub const Outputs = StructOf(spec.outputs, Shape);
        pub const Results = StructOf(spec.outputs, Tensor);

        pub const CallOpts = struct {
            cfg: ConfigT,
            grid: [3]i32,
            num_stages: i32,
            num_warps: i32,
            output_operand_aliases: ?ops.CustomCallOutputOperandAliases(Inputs, Outputs) = null,
            debug: bool = false,
            /// Bytes for Triton's implicit global scratch argument, across all CTAs.
            global_scratch_memory_size: i32 = 0,
        };

        pub fn emit(allocator: std.mem.Allocator, cfg: ConfigT) ![:0]const u8 {
            const ctx = try newContext();
            defer ctx.deinit();

            var b = try triton_builder.Builder.open(allocator, ctx, name);
            defer b.deinit();

            try spec.run(&b, cfg);

            return b.finish();
        }

        pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
            const cur = Compiler.current();

            const ttir = emit(cur.allocator, opts.cfg) catch |err|
                std.debug.panic("zml.kernel.triton.Kernel({s}).call: emit failed: {}", .{ name, err });
            defer cur.allocator.free(ttir);

            var inputs_arr: [spec.inputs.len]Tensor = undefined;
            inline for (spec.inputs, 0..) |fname, i| inputs_arr[i] = @field(inputs, fname);

            var outputs_arr: [spec.outputs.len]Shape = undefined;
            inline for (spec.outputs, 0..) |fname, i| outputs_arr[i] = @field(outputs, fname);

            const aliases = resolveOutputOperandAliases(opts.output_operand_aliases, 0);

            const call_opts: ops.TritonOps = .{
                .debug = opts.debug,
                .name = name,
                .ir = ttir,
                .grid = opts.grid,
                .num_stages = opts.num_stages,
                .num_warps = opts.num_warps,
                .output_operand_aliases = aliases.constSlice(),
                .is_tma_allowed = opts.global_scratch_memory_size > 0,
                .global_scratch_memory_size = opts.global_scratch_memory_size,
            };

            const triton_results = if (opts.global_scratch_memory_size > 0) blk: {
                // XLA maps the final custom-call result to Triton's implicit
                // scratch pointer; it must not appear in the TTIR signature.
                const extended = outputs_arr ++ [_]Shape{.init(.{opts.global_scratch_memory_size}, .u8)};
                const res = ops.triton(inputs_arr, extended, call_opts);
                break :blk res[0..spec.outputs.len].*;
            } else ops.triton(inputs_arr, outputs_arr, call_opts);

            var results: Results = undefined;
            inline for (spec.outputs, 0..) |fname, i| @field(results, fname) = triton_results[i];
            return results;
        }
    };
}
