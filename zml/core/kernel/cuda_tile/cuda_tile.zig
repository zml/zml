const std = @import("std");

const cuda_tile_builder = @import("dsl/cuda_tile/builder");
pub const Builder = cuda_tile_builder.Builder;
pub const Value = cuda_tile_builder.Value;
pub const DType = cuda_tile_builder.DType;
pub const FinishError = cuda_tile_builder.FinishError;
pub const EntryHint = cuda_tile_builder.EntryHint;
pub const Arch = cuda_tile_builder.Arch;
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
    return makeKernelContext(cuda_tile_builder.registerDialects);
}

pub fn from(dt: DataType) DType {
    return switch (dt) {
        .bool => .i1,
        .i4, .u4 => .i4,
        .i8, .u8 => .i8,
        .i16, .u16 => .i16,
        .i32, .u32 => .i32,
        .i64, .u64 => .i64,
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .f8e4m3fn => .f8e4m3fn,
        .f8e5m2 => .f8e5m2,
        .f8e8m0 => .f8e8m0fnu,
        .f4e2m1 => .f4e2m1fn,
        else => std.debug.panic("zml.kernel.cuda_tile.from: dtype {s} has no CUDA Tile IR equivalent", .{@tagName(dt)}),
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

        /// No `num_warps`/`num_stages`: warp and CTA tuning is
        /// `optimization_hints` inside the IR (`Opts.hints`).
        pub const CallOpts = struct {
            cfg: ConfigT,
            grid: [3]i32,
            /// Outputs XLA zeroes before the launch, by position in
            /// `spec.outputs`, ascending.
            zeroed_outputs: []const i32 = &.{},
            output_operand_aliases: ?ops.CustomCallOutputOperandAliases(Inputs, Outputs) = null,
        };

        pub fn emit(allocator: std.mem.Allocator, cfg: ConfigT) ![:0]const u8 {
            const ctx = try newContext();
            defer ctx.deinit();

            var b = try cuda_tile_builder.Builder.open(allocator, ctx, name);
            defer b.deinit();

            try spec.run(&b, cfg);

            return b.finish(&.{});
        }

        pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
            const cur = Compiler.current();

            const ir = emit(cur.allocator, opts.cfg) catch |err|
                std.debug.panic("zml.kernel.cuda_tile.Kernel({s}).call: emit failed: {}", .{ name, err });
            defer cur.allocator.free(ir);

            var inputs_arr: [spec.inputs.len]Tensor = undefined;
            inline for (spec.inputs, 0..) |fname, i| inputs_arr[i] = @field(inputs, fname);

            var outputs_arr: [spec.outputs.len]Shape = undefined;
            inline for (spec.outputs, 0..) |fname, i| outputs_arr[i] = @field(outputs, fname);

            const aliases = resolveOutputOperandAliases(opts.output_operand_aliases, 0);

            const tensor_results = ops.cudaTile(inputs_arr, outputs_arr, .{
                .name = name,
                .ir = ir,
                .grid = opts.grid,
                .zeroed_outputs = opts.zeroed_outputs,
                .output_operand_aliases = aliases.constSlice(),
            });

            var results: Results = undefined;
            inline for (spec.outputs, 0..) |fname, i| @field(results, fname) = tensor_results[i];
            return results;
        }
    };
}
