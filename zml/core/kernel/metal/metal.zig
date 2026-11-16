const std = @import("std");

const func = @import("mlir/dialects/func");
const metal_builder = @import("dsl/metal/builder");
pub const Builder = metal_builder.Builder;
pub const Value = metal_builder.Value;
pub const Type = metal_builder.Type;
pub const Struct = metal_builder.Struct;
pub const ArgSpec = metal_builder.ArgSpec;
pub const LoopOpts = metal_builder.LoopOpts;
pub const FinishError = metal_builder.FinishError;
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
    return makeKernelContext(metal_builder.registerDialects);
}

pub fn from(dt: DataType) Type {
    return switch (dt) {
        .bool => .bool,
        .i8 => .char,
        .i16 => .short,
        .i32 => .int,
        .i64 => .long,
        .u8 => .uchar,
        .u16 => .ushort,
        .u32 => .uint,
        .u64 => .ulong,
        .f16 => .half,
        .bf16 => .bfloat,
        .f32 => .float,
        else => std.debug.panic("zml.kernel.metal.from: dtype {s} has no MSL equivalent", .{@tagName(dt)}),
    };
}

/// The bytes of a `constant T &` argument: `.of(@as(i32, n))`, or a host
/// `extern struct` laid out like a `constant S &` argument.
pub const Constant = struct {
    buffer: [256]u8 = undefined,
    len: usize = 0,

    pub fn of(value: anytype) Constant {
        const raw = std.mem.asBytes(&value);
        var c: Constant = .{ .len = raw.len };
        @memcpy(c.buffer[0..raw.len], raw);
        return c;
    }

    pub fn bytes(self: *const Constant) []const u8 {
        return self.buffer[0..self.len];
    }
};

fn Spec(comptime Config: type) type {
    return struct {
        name: [:0]const u8,
        /// The arguments bound to input tensors.
        inputs: []const [:0]const u8,
        /// The arguments bound to output tensors; one named like an
        /// input updates it in place.
        outputs: []const [:0]const u8,
        /// The `constant` arguments given by value at the call.
        constants: []const [:0]const u8 = &.{},
        run: *const fn (*Builder, Config) FinishError!void,
        fast_math: bool = true,
    };
}

/// A Metal kernel: `run` declares its arguments with `declareArgs`, by
/// name, and builds its body.
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
        pub const Constants = StructOf(spec.constants, Constant);

        pub const CallOpts = struct {
            cfg: ConfigT,
            /// Threadgroups per grid.
            grid: [3]u32,
            /// Threads per threadgroup.
            threadgroup: [3]u32,
            /// Bytes of `[[threadgroup(0)]]` memory.
            threadgroup_memory: u64 = 0,
            constants: Constants = std.mem.zeroes(Constants),
        };

        pub fn emit(allocator: std.mem.Allocator, cfg: ConfigT) ![:0]const u8 {
            var arguments: std.ArrayList(metal_builder.Argument) = .empty;
            defer arguments.deinit(allocator);
            return build(allocator, cfg, &arguments);
        }

        fn build(allocator: std.mem.Allocator, cfg: ConfigT, arguments: *std.ArrayList(metal_builder.Argument)) ![:0]const u8 {
            const ctx = try newContext();
            defer ctx.deinit();
            var b = Builder.open(allocator, ctx, name, .{ .fast_math = spec.fast_math });
            defer b.deinit();
            try spec.run(&b, cfg);
            const ir = try b.finish();
            errdefer allocator.free(ir);
            try arguments.appendSlice(allocator, b.arguments.items);
            return ir;
        }

        fn indexOf(names: []const [:0]const u8, arg: []const u8) ?usize {
            for (names, 0..) |n, i| if (std.mem.eql(u8, n, arg)) return i;
            return null;
        }

        pub fn call(inputs: Inputs, outputs: Outputs, opts: CallOpts) Results {
            const cur = Compiler.current();
            const allocator = cur.allocator;

            var arguments: std.ArrayList(metal_builder.Argument) = .empty;
            defer arguments.deinit(allocator);
            const ir = build(allocator, opts.cfg, &arguments) catch |err|
                std.debug.panic("zml.kernel.metal.Kernel({s}).call: emit failed: {}", .{ name, err });
            defer allocator.free(ir);

            var inputs_arr: [spec.inputs.len]Tensor = undefined;
            inline for (spec.inputs, 0..) |field, i| inputs_arr[i] = @field(inputs, field);
            var outputs_arr: [spec.outputs.len]Shape = undefined;
            inline for (spec.outputs, 0..) |field, i| outputs_arr[i] = @field(outputs, field);
            var constants: [spec.constants.len][]const u8 = undefined;
            inline for (spec.constants, 0..) |field, i| constants[i] = @field(opts.constants, field).bytes();

            var bound: [spec.outputs.len]bool = @splat(false);
            var bound_inputs: [spec.inputs.len]bool = @splat(false);
            var bound_constants: [spec.constants.len]bool = @splat(false);
            var aliases: stdx.BoundedArray(stablehlo.CustomCallOpts.OutputOperandAlias, spec.outputs.len) = .empty;
            var bindings: std.ArrayList(i32) = .empty;
            defer bindings.deinit(allocator);
            for (arguments.items) |arg| {
                if (arg.kind == .threadgroup) {
                    // `threadgroup_memory` sizes `[[threadgroup(0)]]` alone.
                    if (arg.index != 0) std.debug.panic("zml.kernel.metal.Kernel({s}): threadgroup argument {s} is [[threadgroup({d})]], and only [[threadgroup(0)]] gets memory", .{ name, arg.name, arg.index });
                    continue;
                }
                std.debug.assert(arg.index == bindings.items.len);
                const input = indexOf(spec.inputs, arg.name);
                const output = indexOf(spec.outputs, arg.name);
                if (input) |i| bound_inputs[i] = true;
                if (output) |o| bound[o] = true;
                const binding: i32 = if (input) |i| blk: {
                    if (output) |o| aliases.appendAssumeCapacity(.{ .output_index = @intCast(o), .operand_index = @intCast(i) });
                    break :blk @intCast(i);
                } else if (output) |o|
                    @intCast(spec.inputs.len + o)
                else if (indexOf(spec.constants, arg.name)) |k| blk: {
                    if (constants[k].len != arg.size) std.debug.panic("zml.kernel.metal.Kernel({s}): constant {s} has {d} bytes, the argument {d}", .{ name, arg.name, constants[k].len, arg.size });
                    bound_constants[k] = true;
                    break :blk -1 - @as(i32, @intCast(k));
                } else std.debug.panic("zml.kernel.metal.Kernel({s}): argument {s} is no input, output or constant", .{ name, arg.name });
                bindings.append(allocator, binding) catch @panic("OOM");
            }
            for (bound, spec.outputs) |b, output| if (!b) std.debug.panic("zml.kernel.metal.Kernel({s}): output {s} is no argument", .{ name, output });
            for (bound_inputs, spec.inputs) |b, input| if (!b) std.debug.panic("zml.kernel.metal.Kernel({s}): input {s} is no argument", .{ name, input });
            for (bound_constants, spec.constants) |b, constant| if (!b) std.debug.panic("zml.kernel.metal.Kernel({s}): constant {s} is no argument", .{ name, constant });

            const tensor_results = ops.metal(inputs_arr, outputs_arr, .{
                .name = name,
                .ir = ir,
                .grid = opts.grid,
                .threadgroup = opts.threadgroup,
                .threadgroup_memory = opts.threadgroup_memory,
                .arguments = bindings.items,
                .constants = &constants,
                .output_operand_aliases = aliases.constSlice(),
            });
            var results: Results = undefined;
            inline for (spec.outputs, 0..) |field, i| @field(results, field) = tensor_results[i];
            return results;
        }
    };
}
