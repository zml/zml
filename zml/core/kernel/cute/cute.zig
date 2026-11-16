const std = @import("std");

const cute_builder = @import("dsl/cute/builder");
pub const AlgebraExpr = cute_builder.AlgebraExpr;
pub const Builder = cute_builder.Builder;
pub const ArgSpec = cute_builder.ArgSpec;
pub const Value = cute_builder.Value;
pub const View = cute_builder.View;
pub const Atom = cute_builder.Atom;
pub const DType = cute_builder.DType;
pub const AlgebraToken = cute_builder.AlgebraToken;
pub const LayoutSpec = cute_builder.LayoutSpec;
pub const ConstrainedDynamic = cute_builder.ConstrainedDynamic;
pub const FinishError = cute_builder.FinishError;
pub const TmaConfig = cute_builder.TmaConfig;
pub const TmaFormat = cute_builder.TmaFormat;
pub const BlockScaledMmaConfig = cute_builder.BlockScaledMmaConfig;
pub const CtaGroup = cute_builder.CtaGroup;
pub const ctaCount = cute_builder.ctaCount;
pub const MBarrierScope = cute_builder.MBarrierScope;
pub const OperandMajorMode = cute_builder.OperandMajorMode;
pub const Tcgen05SmemSwizzle = cute_builder.Tcgen05SmemSwizzle;
pub const Tcgen05SmemDesc = cute_builder.Tcgen05SmemDesc;
pub const Tcgen05InstrDesc = cute_builder.Tcgen05InstrDesc;
pub const CopyBulkTensorTileG2SOp = cute_builder.CopyBulkTensorTileG2SOp;
pub const CopyBulkTensor2DGather4G2SOp = cute_builder.CopyBulkTensor2DGather4G2SOp;
pub const CopyBulkTensorTileS2GOp = cute_builder.CopyBulkTensorTileS2GOp;
const func = @import("mlir/dialects/func");
const mlir = @import("mlir");
const stablehlo = @import("mlir/dialects/stablehlo");
const stdx = @import("stdx");

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
    return makeKernelContext(cute_builder.registerDialects);
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
        .f4e2m1 => .f4e2m1fn,
        .f8e4m3fn => .f8e4m3fn,
        .f8e5m2 => .f8e5m2,
        .f8e8m0 => .f8e8m0fnu,
        else => std.debug.panic("zml.kernel.cute.from: dtype {s} has no CuTe equivalent", .{@tagName(dt)}),
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

        /// `kernel.launch(grid=, block=)` in the Python DSL; dynamic shared
        /// memory is what the kernel declares.
        pub const CallOpts = struct {
            cfg: ConfigT,
            grid: [3]i32,
            block: [3]i32,
            zeroed_outputs: []const i32 = &.{},
            output_operand_aliases: ?ops.CustomCallOutputOperandAliases(Inputs, Outputs) = null,
        };

        pub fn emit(allocator: std.mem.Allocator, cfg: ConfigT, block: [3]i32) ![:0]const u8 {
            const ctx = try newContext();
            defer ctx.deinit();

            var b = try cute_builder.Builder.open(allocator, ctx, name);
            defer b.deinit();

            try spec.run(&b, cfg);

            // XLA passes one pointer per input then per output; it does
            // not check the count against the kernel.
            const expected = spec.inputs.len + spec.outputs.len;
            if (b.args.len != expected) {
                std.debug.panic("zml.kernel.cute.Kernel({s}): declareArgs declared {d} arguments, inputs + outputs are {d}", .{ name, b.args.len, expected });
            }

            return b.finish(block);
        }

        pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
            const cur = Compiler.current();

            const ir = emit(cur.allocator, opts.cfg, opts.block) catch |err|
                std.debug.panic("zml.kernel.cute.Kernel({s}).call: emit failed: {}", .{ name, err });
            defer cur.allocator.free(ir);

            var inputs_arr: [spec.inputs.len]Tensor = undefined;
            inline for (spec.inputs, 0..) |fname, i| inputs_arr[i] = @field(inputs, fname);

            var outputs_arr: [spec.outputs.len]Shape = undefined;
            inline for (spec.outputs, 0..) |fname, i| outputs_arr[i] = @field(outputs, fname);

            const aliases = resolveOutputOperandAliases(opts.output_operand_aliases, 0);

            const tensor_results = ops.cute(inputs_arr, outputs_arr, .{
                .name = name,
                .ir = ir,
                .grid = opts.grid,
                .block = opts.block,
                .zeroed_outputs = opts.zeroed_outputs,
                .output_operand_aliases = aliases.constSlice(),
            });

            var results: Results = undefined;
            inline for (spec.outputs, 0..) |fname, i| @field(results, fname) = tensor_results[i];
            return results;
        }
    };
}

fn ProgramSpec(comptime Config: type) type {
    return struct {
        /// Public host entry selected by XLA. The callback may give the
        /// nested CUDA kernel a different (private) symbol.
        name: [:0]const u8,
        inputs: []const [:0]const u8,
        outputs: []const [:0]const u8,
        run: *const fn (*Builder, Config) FinishError!void,
    };
}

/// A complete CuTe program: one or more `cuda.kernel` operations inside
/// a `gpu.module`, followed by a public `func.func` which constructs the
/// launch-time CuTe objects and calls `cuda.launch_ex`.
///
/// This is the source-level equivalent of invoking a Python `@cute.jit`
/// object whose `__call__` builds TMA descriptors and launches an
/// `@cute.kernel`. XLA executes the generated host function on its CUDA
/// stream, so the custom call deliberately carries no external grid or
/// block dimensions.
pub fn Program(
    comptime ConfigT: type,
    comptime spec: ProgramSpec(ConfigT),
) type {
    return struct {
        pub const name: [:0]const u8 = spec.name;
        pub const Config = ConfigT;
        pub const Inputs = StructOf(spec.inputs, Tensor);
        pub const Outputs = StructOf(spec.outputs, Shape);
        pub const Results = StructOf(spec.outputs, Tensor);

        pub const CallOpts = struct {
            cfg: ConfigT,
            zeroed_outputs: []const i32 = &.{},
            scalars: []const i64 = &.{},
            symmetric_workspace: ?ops.CuteOps.SymmetricWorkspace = null,
            output_operand_aliases: ?ops.CustomCallOutputOperandAliases(Inputs, Outputs) = null,
        };

        pub fn emit(allocator: std.mem.Allocator, cfg: ConfigT) ![:0]const u8 {
            const ctx = try newContext();
            defer ctx.deinit();

            var b = try cute_builder.Builder.openProgram(allocator, ctx, name);
            defer b.deinit();
            try spec.run(&b, cfg);
            return b.finishProgram();
        }

        pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
            const cur = Compiler.current();
            const ir = emit(cur.allocator, opts.cfg) catch |err|
                std.debug.panic("zml.kernel.cute.Program({s}).call: emit failed: {}", .{ name, err });
            defer cur.allocator.free(ir);

            var inputs_arr: [spec.inputs.len]Tensor = undefined;
            inline for (spec.inputs, 0..) |fname, i| inputs_arr[i] = @field(inputs, fname);

            var outputs_arr: [spec.outputs.len]Shape = undefined;
            inline for (spec.outputs, 0..) |fname, i| outputs_arr[i] = @field(outputs, fname);

            const aliases = resolveOutputOperandAliases(opts.output_operand_aliases, 0);
            const tensor_results = ops.cute(inputs_arr, outputs_arr, .{
                .name = name,
                .ir = ir,
                .zeroed_outputs = opts.zeroed_outputs,
                .scalars = opts.scalars,
                .symmetric_workspace = opts.symmetric_workspace,
                .output_operand_aliases = aliases.constSlice(),
            });

            var results: Results = undefined;
            inline for (spec.outputs, 0..) |fname, i| @field(results, fname) = tensor_results[i];
            return results;
        }
    };
}
