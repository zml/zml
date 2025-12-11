const std = @import("std");
const zml = @import("zml");

pub const Model = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn init() !Model {
        return .{
            .weight = .init(.{ .m = 4096, .n = 4096 }, .f32),
            .bias = .init(.{ .m = 4096 }, .f32),
        };
    }

    pub fn deinit(self: *Model) void {
        _ = self;
    }

    pub fn loadBuffers(model: Model, allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform) !zml.Bufferized(Model) {
        const weight_slice: zml.Slice = try .alloc(allocator, model.weight.shape());
        defer weight_slice.free(allocator);
        @memset(weight_slice.items(f32), 1);

        const bias_slice: zml.Slice = try .alloc(allocator, model.bias.shape());
        defer bias_slice.free(allocator);
        @memset(bias_slice.items(f32), 2);

        const weight_buffer: zml.Buffer = try .fromBytes(platform, weight_slice.shape, weight_slice.data, io);
        errdefer weight_buffer.deinit();
        const bias_buffer: zml.Buffer = try .fromBytes(platform, bias_slice.shape, bias_slice.data, io);
        errdefer bias_buffer.deinit();

        return .{ .weight = weight_buffer, .bias = bias_buffer };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model)) void {
        self.weight.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: Model, input: zml.Tensor) zml.Tensor {
        return self.weight.dot(input, .n).add(self.bias);
    }
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    const io = threaded.io();

    zml.init();
    defer zml.deinit();

    var platform = try zml.Platform.init(.cpu, io, .{});
    defer platform.deinit();

    var model: Model = try .init();
    defer model.deinit();

    const input: zml.Tensor = .init(.{ .n = 4096 }, .f32);
    var exe = try zml.module.compileModel(allocator, io, Model.forward, model, .{input}, platform);
    defer exe.deinit();

    const slice: zml.Slice = try .alloc(allocator, input.shape());
    defer slice.free(allocator);
    @memset(slice.items(f32), 1);

    var model_buffers = try model.loadBuffers(allocator, io, platform);
    defer Model.unloadBuffers(&model_buffers);

    const input_buffer: zml.Buffer = try .fromBytes(platform, slice.shape, slice.data, io);
    defer input_buffer.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ model_buffers, input_buffer });
    exe.call(args, &results, io);

    const output = results.get(zml.Buffer);
    defer output.deinit();

    const output_slice = try output.toSliceAlloc(allocator, io);
    defer output_slice.free(allocator);

    std.log.info("Output: {any}", .{output_slice.items(f32)[0..10]});
}
