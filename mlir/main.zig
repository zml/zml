const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const TensorOriginKind = enum {
    argument,
    value,
};

const TensorOrigin = union(TensorOriginKind) {
    argument: void,
    value: usize,
};

var global_compilation_context: ?*CompilationContext = null;

pub const Tensor = struct {
    shape_: ?zml.Shape = null,
    tensor_origin: TensorOrigin = .{ .argument = {} },

    pub fn init(shape_: zml.Shape) Tensor {
        return .{ .shape_ = shape_ };
    }

    pub fn matmul(self: *const Tensor, other: *const Tensor) Tensor {
        const self_value = self.getValue();
        const other_value = other.getValue();

        std.debug.print("self: {} - other: {}\n", .{ self_value, other_value });
        return .{};
    }

    pub fn add(self: *const Tensor, other: *const Tensor) Tensor {
        _ = self; // autofix
        _ = other; // autofix
        return .{ .tensor_origin = .{ .value = 1 } };
    }

    pub fn relu(self: *const Tensor) Tensor {
        _ = self; // autofix
        return .{ .tensor_origin = .{ .value = 1 } };
    }

    pub fn flattenAll(self: *const Tensor) Tensor {
        _ = self; // autofix
        return .{ .tensor_origin = .{ .value = 1 } };
    }

    pub fn convert(self: *const Tensor, dtype: zml.DataType) Tensor {
        _ = self; // autofix
        _ = dtype; // autofix
        return .{ .tensor_origin = .{ .value = 1 } };
    }

    pub fn argMax(self: *const Tensor, dim: i64) struct { indices: Tensor } {
        _ = self; // autofix
        _ = dim; // autofix
        return .{ .indices = .{ .tensor_origin = .{ .value = 1 } } };
    }

    pub fn getValue(self: *const Tensor) usize {
        return switch (self.tensor_origin) {
            .argument => b: {
                const entry = global_compilation_context.?.mapping.get(self).?;
                break :b entry * 1000;
            },
            .value => |v| v,
        };
    }
};

/// Model definition
const Mnist = struct {
    fc1: Layer = .{},
    fc2: Layer = .{},

    const Layer = struct {
        weight: Tensor = .{},
        bias: Tensor = .{},

        pub fn forward(self: *const Layer, input: *const Tensor) Tensor {
            return self.weight.matmul(input).add(&self.bias).relu();
        }
    };

    pub fn init() Mnist {
        return .{};
    }

    /// just two linear layers + relu activation
    pub fn forward(self: *const Mnist, input: *const Tensor) Tensor {
        // std.log.info("Compiling for target: {s}", .{@tagName(input.getContext().target())});
        var x = input.flattenAll().convert(.f32);
        const layers: []const *const Layer = &.{ &self.fc1, &self.fc2 };
        for (layers) |layer| {
            x = layer.forward(&x);
        }
        return x.argMax(0).indices.convert(.u8);
    }
};

const dialects = @import("mlir/dialects");
const mlir = @import("mlir");

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    var mnist: Mnist = .init();
    _ = &mnist;

    // TODO(Corentin): Hydrate the model using the config/bufferstore

    const input: Tensor = .init(zml.Shape.init(.{ 28, 28 }, .u8));

    compileModel(allocator, Mnist.forward, &mnist, .{&input});
}

const CompilationContext = struct {
    allocator: std.mem.Allocator,

    mapping: std.AutoArrayHashMapUnmanaged(*const Tensor, usize),

    pub fn init(allocator: std.mem.Allocator) CompilationContext {
        return .{
            .allocator = allocator,
            .mapping = .{},
        };
    }

    pub fn deinit(self: *CompilationContext) void {
        self.mapping.deinit(self.allocator);
    }

    pub fn appendToMapping(self: *CompilationContext, tensor: *const Tensor, unique_id: usize) void {
        const gop = self.mapping.getOrPut(self.allocator, tensor) catch unreachable;
        std.debug.assert(gop.found_existing == false);

        gop.value_ptr.* = unique_id;
    }
};

const Context = struct {
    compilation_context: *CompilationContext,
    current_id: usize = 0,
};

fn compileModel(allocator: std.mem.Allocator, comptime func: anytype, model: stdx.meta.Head(stdx.meta.FnArgs(func)), args: stdx.meta.Tail(stdx.meta.FnArgs(func))) void {
    var compilation_context: CompilationContext = .init(allocator);
    defer compilation_context.deinit();

    global_compilation_context = &compilation_context;
    defer global_compilation_context = null;

    var context: Context = .{ .compilation_context = &compilation_context };
    zml.meta.visit(struct {
        fn cb(inner_context: *Context, tensor: *const Tensor) void {
            inner_context.compilation_context.appendToMapping(tensor, inner_context.current_id);
            inner_context.current_id += 1;
        }
    }.cb, &context, model);
    zml.meta.visit(struct {
        fn cb(inner_context: *Context, tensor: *const Tensor) void {
            inner_context.compilation_context.appendToMapping(tensor, inner_context.current_id);
            inner_context.current_id += 1;
        }
    }.cb, &context, &args);

    _ = @call(.auto, func, .{model} ++ args);
}
