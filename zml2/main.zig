const std = @import("std");
const zml = @import("zml");

pub const Model = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn init() !Model {
        return .{
            .weight = .init(zml.Shape.init(.{ .m = 4096, .n = 4096 }, .f32)),
            .bias = .init(zml.Shape.init(.{ .m = 4096 }, .f32)),
        };
    }

    pub fn deinit(self: *Model) void {
        _ = self;
    }

    pub fn forward(self: Model, input: zml.Tensor) zml.Tensor {
        return self.weight.dot(input, .n).add(self.bias);
    }
};

pub fn main() !void {
    zml.init();
    defer zml.deinit();

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var platform = try zml.Platform.init(.cpu, .{});
    defer platform.deinit();

    var model: Model = try .init();
    defer model.deinit();

    var exe = try zml.module.compileModel(allocator, Model.forward, model, .{zml.Tensor.init(zml.Shape.init(.{ .n = 4096 }, .f32))}, platform);
    defer exe.deinit();
}
