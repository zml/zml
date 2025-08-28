const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");

const cuda = zml.context.cuda;

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

const log = std.log.scoped(.@"examples/custom_call");

fn add_op_host_func(data: *anyopaque) callconv(.c) void {
    const ctx = @as(*AddOp, @ptrCast(@alignCast(data)));

    const result = std.mem.bytesAsValue(f32, ctx.result.asHostBuffer().mutBytes());
    result.* = ctx.a.asHostBuffer().item(f32, 0) + ctx.b.asHostBuffer().item(f32, 0);
}

pub const AddOp = struct {
    pub var type_id: i64 = undefined;
    const Self = @This();

    allocator: std.mem.Allocator,
    platform: zml.Platform,

    a: zml.Buffer,
    b: zml.Buffer,
    result: zml.Buffer,

    results: [1]zml.Buffer = undefined,
    stream: *anyopaque = undefined,

    pub fn init(
        allocator: std.mem.Allocator,
        platform: zml.Platform,
        io_shape: zml.Shape,
    ) !AddOp {
        const host_buffer = try zml.HostBuffer.empty(allocator, io_shape);
        defer host_buffer.deinit(allocator);

        const a = try zml.Buffer.from(platform, host_buffer, .{ .memory = .host_pinned });
        const b = try zml.Buffer.from(platform, host_buffer, .{ .memory = .host_pinned });
        const result = try zml.Buffer.from(platform, host_buffer, .{ .memory = .host_pinned });

        return .{
            .a = a,
            .b = b,
            .result = result,
            .platform = platform,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.a.deinit();
        self.b.deinit();
        self.result.deinit();
    }

    pub fn call(self: *Self, a: zml.Buffer, b: zml.Buffer) !void {
        if (self.platform.target == .cuda) {
            cuda.memcpyToHostAsync(self.a.asHostBuffer().mutBytes(), a.opaqueDeviceMemoryDataPointer(), self.stream);
            cuda.memcpyToHostAsync(self.b.asHostBuffer().mutBytes(), b.opaqueDeviceMemoryDataPointer(), self.stream);

            _ = cuda.cuLaunchHostFunc(self.stream, @ptrCast(&add_op_host_func), @ptrCast(self));

            cuda.memcpyToDeviceAsync(self.results[0].opaqueDeviceMemoryDataPointer(), self.result.asHostBuffer().bytes(), self.stream);
            return;
        }

        if (self.platform.target == .cpu) {
            const result = std.mem.bytesAsValue(f32, self.results[0].asHostBuffer().mutBytes());
            result.* = a.asHostBuffer().item(f32, 0) + b.asHostBuffer().item(f32, 0);
            return;
        }

        return error.UnsupportedTarget;
    }
};

const Layer = struct {
    pub fn forward(_: Layer, a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        const result = zml.custom_call(AddOp, .{ a.print(), b.print() }, &[_]zml.Shape{a.shape()}, .{});
        return result[0];
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    const shape = zml.Shape.init(.{}, .f32);

    const buffers: zml.aio.BufferStore.Buffers = .{};
    const buffer_store: zml.aio.BufferStore = .{
        .arena = arena_state,
        .buffers = buffers,
    };

    const model_shapes = try zml.aio.populateModel(Layer, allocator, buffer_store);

    var compilation = try asynk.asyncc(zml.compileModel, .{ allocator, Layer.forward, model_shapes, .{ shape, shape }, platform });

    var model_weights = try zml.aio.loadModelBuffers(Layer, model_shapes, buffer_store, arena, platform);
    defer zml.aio.unloadBuffers(&model_weights);

    const compiled = try compilation.awaitt();

    var executable = compiled.prepare(model_weights);
    defer executable.deinit();

    executable = try executable.withExecutionContext();

    try platform.registerFFIType(AddOp);
    var add_op: AddOp = try .init(allocator, platform, shape);
    defer add_op.deinit();
    try executable.bind(AddOp, &add_op);

    var input_a = [1]f32{1.0};
    var input_buffer_a = try zml.Buffer.from(platform, zml.HostBuffer.fromSlice(shape, &input_a), .{});
    defer input_buffer_a.deinit();

    var input_b = [1]f32{1.0};
    var input_buffer_b = try zml.Buffer.from(platform, zml.HostBuffer.fromSlice(shape, &input_b), .{});
    defer input_buffer_b.deinit();

    var result: zml.Buffer = executable.call(.{ input_buffer_a, input_buffer_b });
    defer result.deinit();

    var cpu_result = try result.toHostAlloc(arena);

    log.warn(
        "\nThe result of {d} + {d} = {d}\n",
        .{ &input_a, &input_b, cpu_result.items(f32) },
    );
}
