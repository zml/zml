const std = @import("std");

const func = @import("mlir/dialects/func");
const metal_builder = @import("dsl/metal/builder");
const mlir = @import("mlir");
const stablehlo = @import("mlir/dialects/stablehlo");
const stdx = @import("stdx");
const tcl_builder = @import("dsl/tcl/builder");
pub const Builder = tcl_builder.Builder;
pub const TensorOperation = tcl_builder.TensorOperation;
pub const Value = tcl_builder.Value;
pub const Axis = tcl_builder.Axis;
pub const Expr = tcl_builder.Expr;
pub const Dram = tcl_builder.Dram;
pub const Context = tcl_builder.Context;
pub const DType = tcl_builder.DType;
pub const VeOp = tcl_builder.VeOp;
pub const ReduceMode = tcl_builder.ReduceMode;
pub const Predicate = tcl_builder.Predicate;
pub const Tactic = tcl_builder.Tactic;
pub const FinishError = tcl_builder.FinishError;

const Compiler = @import("../../Compiler.zig");
const DataType = @import("../../dtype.zig").DataType;
const mlirx = @import("../../mlirx.zig");
const ops = @import("../../ops.zig");
const platform_ = @import("../../platform.zig");
const Shape = @import("../../shape.zig").Shape;
const Tensor = @import("../../tensor.zig").Tensor;

test {
    _ = @import("test.zig");
}

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
    return makeKernelContext(tcl_builder.registerDialects);
}

pub fn from(dt: DataType) DType {
    return switch (dt) {
        .bool => .bool,
        .i4 => .i4,
        .i8 => .i8,
        .i16 => .i16,
        .i32 => .i32,
        .i64 => .i64,
        .u8 => .u8,
        .u16 => .u16,
        .u32 => .u32,
        .u64 => .u64,
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .f4e2m1 => .f4_e2,
        .f8e4m3fn => .f8_e4,
        .f8e5m2 => .f8_e5,
        else => std.debug.panic("zml.kernel.tcl.from: dtype {s} has no TCL equivalent", .{@tagName(dt)}),
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

/// A TCL function compiled by the Furiosa plugin as one native unit:
/// `declareArgs` declares the inputs in order, `ret` the outputs.
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
        };

        pub fn emit(allocator: std.mem.Allocator, cfg: ConfigT) ![:0]const u8 {
            return build(allocator, cfg, null, null, null);
        }

        /// `updated`, when given, receives the input each output updates
        /// in place (`Builder.update`), or null.
        fn build(allocator: std.mem.Allocator, cfg: ConfigT, inputs: ?[]const Shape, outputs: ?[]const Shape, updated: ?*[spec.outputs.len]?usize) ![:0]const u8 {
            const ctx = try newContext();
            defer ctx.deinit();

            var b = try Builder.open(allocator, ctx, name);
            defer b.deinit();
            try spec.run(&b, cfg);

            const results = b.results orelse std.debug.panic("zml.kernel.tcl.Kernel({s}): run did not call ret", .{name});
            if (b.args.len != spec.inputs.len or results.len != spec.outputs.len) {
                std.debug.panic("zml.kernel.tcl.Kernel({s}): {d} arguments and {d} results for {d} inputs and {d} outputs", .{ name, b.args.len, results.len, spec.inputs.len, spec.outputs.len });
            }

            if (inputs) |shapes| for (b.args, shapes, spec.inputs) |t, s, n| check(t, s, n);
            if (outputs) |shapes| for (results, shapes, spec.outputs) |t, s, n| check(t, s, n);
            if (updated) |u| for (u, results) |*dst, r| {
                dst.* = b.updatedArgument(r);
            };

            return b.finish();
        }

        fn check(t: tcl_builder.Tensor, s: Shape, field: []const u8) void {
            var same = from(s.dtype()) == t.dtype;
            if (t.shards > 1) {
                // A partition holds one chip's share of a chip-split DRAM tensor.
                var n: i64 = 1;
                for (t.axes) |a| n *= a.size;
                same = same and n == @as(i64, @intCast(s.count())) * t.shards;
            } else {
                same = same and s.rank() == t.axes.len;
                if (same) for (t.axes, 0..) |a, i| {
                    same = same and a.size == s.dim(i);
                };
            }

            if (!same) std.debug.panic("zml.kernel.tcl.Kernel({s}): {s} is {f}, the TCL function has {s} over {d} axes", .{ name, field, s, @tagName(t.dtype), t.axes.len });
        }

        pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
            const cur = Compiler.current();

            var inputs_arr: [spec.inputs.len]Tensor = undefined;
            var input_shapes: [spec.inputs.len]Shape = undefined;
            inline for (spec.inputs, 0..) |fname, i| {
                inputs_arr[i] = @field(inputs, fname);
                input_shapes[i] = inputs_arr[i].shape();
            }

            var outputs_arr: [spec.outputs.len]Shape = undefined;
            inline for (spec.outputs, 0..) |fname, i| outputs_arr[i] = @field(outputs, fname);

            var updated: [spec.outputs.len]?usize = undefined;
            const ir = build(cur.allocator, opts.cfg, &input_shapes, &outputs_arr, &updated) catch |err|
                std.debug.panic("zml.kernel.tcl.Kernel({s}).call: emit failed: {}", .{ name, err });
            defer cur.allocator.free(ir);

            var aliases: stdx.BoundedArray(stablehlo.CustomCallOpts.OutputOperandAlias, spec.outputs.len) = .empty;
            for (updated, 0..) |input, output| if (input) |i| aliases.appendAssumeCapacity(.{
                .output_index = @intCast(output),
                .operand_index = @intCast(i),
            });
            const tensor_results = ops.tcl(inputs_arr, outputs_arr, .{ .ir = ir, .output_operand_aliases = aliases.constSlice() });
            var results: Results = undefined;
            inline for (spec.outputs, 0..) |fname, i| {
                @field(results, fname) = tensor_results[i];
            }

            return results;
        }
    };
}
