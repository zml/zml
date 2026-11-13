const std = @import("std");

const fly_builder = @import("dsl/fly/builder");
pub const Builder = fly_builder.Builder;
pub const Value = fly_builder.Value;
pub const DType = fly_builder.DType;
pub const FinishError = fly_builder.FinishError;
pub const TiledCopy = fly_builder.TiledCopy;
pub const TiledMma = fly_builder.TiledMma;
pub const Arch = fly_builder.Arch;
pub const MmaFlavor = fly_builder.MmaFlavor;
pub const fly_rocdl = fly_builder.fly_rocdl;
pub const layout = fly_builder.layout;
pub const Layout = fly_builder.Layout;
pub const Tile = fly_builder.Tile;
pub const IntTuple = fly_builder.IntTuple;
pub const L = fly_builder.L;
pub const rowMajor = fly_builder.rowMajor;
pub const colMajor = fly_builder.colMajor;
pub const ordered = fly_builder.ordered;
pub const tile = fly_builder.tile;
pub const it = fly_builder.it;
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

/// Threads per wavefront: 64 on CDNA, 32 on RDNA.
pub fn waveSize(p: *const platform_.Platform) i32 {
    const cc = platform_.rocm.computeCapability(p) orelse return 64;
    return switch (cc.architecture()) {
        .cdna1, .cdna2, .cdna3, .cdna4 => 64,
        .rdna2, .rdna3, .rdna3_5, .rdna4 => 32,
    };
}

/// The plugin sizes a block in wavefronts, so a thread count has to divide
/// evenly: a kernel's layouts are built for an exact number of threads and
/// spare ones would partition out of range.
fn warpsFor(p: *const platform_.Platform, threads: i32) i32 {
    const wave = waveSize(p);
    if (@rem(threads, wave) != 0) {
        std.debug.panic("zml.kernel.fly: {d} threads is not a multiple of the {d}-wide wavefront", .{ threads, wave });
    }
    return @divExact(threads, wave);
}

/// The matrix atom this device provides, or null when it has none: WMMA
/// arrived with RDNA3, so gfx1030 has no matrix instruction at all.
pub fn mmaFlavor(p: *const platform_.Platform) ?MmaFlavor {
    const cc = platform_.rocm.computeCapability(p) orelse return .cdna3_mfma;
    return switch (cc.architecture()) {
        .cdna1, .cdna2, .cdna3, .cdna4 => .cdna3_mfma,
        .rdna3, .rdna3_5 => .gfx11_wmma,
        .rdna4 => .gfx120x_wmma,
        .rdna2 => null,
    };
}
pub fn newContext() std.mem.Allocator.Error!*mlir.Context {
    return makeKernelContext(fly_builder.registerDialects);
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
        .f8e5m2fnuz => .f8e5m2fnuz,
        else => std.debug.panic("zml.kernel.fly.from: dtype {s} has no Fly equivalent", .{@tagName(dt)}),
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
        pub const arg_names = spec.inputs ++ spec.outputs;
        /// By name, as `!fly.memref` views.
        pub const Args = StructOf(arg_names, Value);

        pub const CallOpts = struct {
            cfg: ConfigT,
            grid: [3]i32,
            /// Threads per block. Converted to the wavefronts the plugin
            /// wants using the device's wave size, which is 64 on CDNA and
            /// 32 on RDNA, so a kernel states the thread count its layouts
            /// were built for and runs on both.
            threads: i32,
            /// > 0 sets `amdgpu-waves-per-eu = "N, N"`: a hard occupancy clamp.
            waves_per_eu: i32 = 0,
            /// Required when the body uses `fly.get_dyn_shared`.
            shared_mem_bytes: i64 = 0,
            /// Inputs first, then outputs; see `zeroedOutput`. A no-op
            /// inside command buffers.
            zeroed_args: []const i32 = &.{},
            output_operand_aliases: ?ops.CustomCallOutputOperandAliases(Inputs, Outputs) = null,
        };

        /// The arguments `run` was declared with.
        pub fn args(b: *Builder) Args {
            var out: Args = undefined;
            inline for (arg_names, 0..) |fname, i| @field(out, fname) = b.arg(i);
            return out;
        }

        /// `zeroed_args` index of an output.
        pub fn zeroedOutput(comptime f: std.meta.FieldEnum(Outputs)) i32 {
            return @intCast(spec.inputs.len + @backingInt(f));
        }

        /// `shapes` is inputs, then outputs.
        pub fn emit(allocator: std.mem.Allocator, cfg: ConfigT, shapes: []const Shape) ![:0]const u8 {
            std.debug.assert(shapes.len == arg_names.len);
            const ctx = try newContext();
            defer ctx.deinit();

            var b = try fly_builder.Builder.open(allocator, ctx, name);
            defer b.deinit();

            var specs: [arg_names.len]Builder.ArgSpec = undefined;
            inline for (arg_names, 0..) |fname, i| {
                const shape = shapes[i];
                specs[i] = .{
                    .name = fname,
                    .dtype = from(shape.dtype()),
                    .dims = if (shape.rank() > 0) shape.dims() else null,
                };
            }
            try b.declareArgsLow(&specs);

            try spec.run(&b, cfg);

            return b.finish();
        }

        pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
            const cur = Compiler.current();

            var inputs_arr: [spec.inputs.len]Tensor = undefined;
            inline for (spec.inputs, 0..) |fname, i| inputs_arr[i] = @field(inputs, fname);

            var outputs_arr: [spec.outputs.len]Shape = undefined;
            inline for (spec.outputs, 0..) |fname, i| outputs_arr[i] = @field(outputs, fname);

            var shapes: [arg_names.len]Shape = undefined;
            inline for (0..spec.inputs.len) |i| shapes[i] = inputs_arr[i].shape();
            inline for (0..spec.outputs.len) |i| shapes[spec.inputs.len + i] = outputs_arr[i];

            const ir = emit(cur.allocator, opts.cfg, &shapes) catch |err|
                std.debug.panic("zml.kernel.fly.Kernel({s}).call: emit failed: {}", .{ name, err });
            defer cur.allocator.free(ir);

            const aliases = resolveOutputOperandAliases(opts.output_operand_aliases, 0);

            const tensor_results = ops.fly(inputs_arr, outputs_arr, .{
                .name = name,
                .ir = ir,
                .grid = opts.grid,
                .num_warps = warpsFor(cur.platform, opts.threads),
                .waves_per_eu = opts.waves_per_eu,
                .shared_mem_bytes = opts.shared_mem_bytes,
                .zeroed_args = opts.zeroed_args,
                .output_operand_aliases = aliases.constSlice(),
            });

            var results: Results = undefined;
            inline for (spec.outputs, 0..) |fname, i| @field(results, fname) = tensor_results[i];
            return results;
        }
    };
}
