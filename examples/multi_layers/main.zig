const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
};

/// Model definition
const MultiLayers = struct {
    fc1: Layer,
    fc2: Layer,
    fc3: Layer,

    const Layer = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn init_layer() Layer {
            return .{
                .weight = zml.Tensor.init(.{ 2, 2 }, .f16).withTags(.{ .a, .b }),
                .bias = zml.Tensor.init(.{2}, .f16).withTags(.{.a}),
            };
        }

        pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor {
            return self.weight.dot(x, .b).add(self.bias).withTags(.{.b});
        }
    };

    pub fn init() MultiLayers {
        return .{
            .fc1 = .init_layer(),
            .fc2 = .init_layer(),
            .fc3 = .init_layer(),
        };
    }

    pub fn load(
        self: *const MultiLayers,
        io: std.Io,
        platform: *const zml.Platform,
        sharding: []const zml.sharding.Sharding,
    ) !zml.Bufferized(MultiLayers) {
        const w1_slice: zml.Slice = .init(self.fc1.weight.shape(), std.mem.sliceAsBytes(&[4]f16{ 1.0, 0.0, 0.0, 1.0 }));
        const b1_slice: zml.Slice = .init(self.fc1.bias.shape(), std.mem.sliceAsBytes(&[2]f16{ 1.0, 1.0 }));
        const w2_slice: zml.Slice = .init(self.fc2.weight.shape(), std.mem.sliceAsBytes(&[4]f16{ 1.0, 0.0, 0.0, 2.0 }));
        const b2_slice: zml.Slice = .init(self.fc2.bias.shape(), std.mem.sliceAsBytes(&[2]f16{ 1.0, 1.0 }));
        const w3_slice: zml.Slice = .init(self.fc3.weight.shape(), std.mem.sliceAsBytes(&[4]f16{ 1.0, 0.0, 0.0, 3.0 }));
        const b3_slice: zml.Slice = .init(self.fc3.bias.shape(), std.mem.sliceAsBytes(&[2]f16{ 1.0, 1.0 }));
        return .{
            .fc1 = .{
                .weight = try zml.Buffer.fromSlice(io, platform, w1_slice, sharding[0]),
                .bias = try zml.Buffer.fromSlice(io, platform, b1_slice, sharding[0]),
            },
            .fc2 = .{
                .weight = try zml.Buffer.fromSlice(io, platform, w2_slice, sharding[0]),
                .bias = try zml.Buffer.fromSlice(io, platform, b2_slice, sharding[0]),
            },
            .fc3 = .{
                .weight = try zml.Buffer.fromSlice(io, platform, w3_slice, sharding[0]),
                .bias = try zml.Buffer.fromSlice(io, platform, b3_slice, sharding[0]),
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MultiLayers)) void {
        self.fc1.weight.deinit();
        self.fc1.bias.deinit();
        self.fc2.weight.deinit();
        self.fc2.bias.deinit();
        self.fc3.weight.deinit();
        self.fc3.bias.deinit();
    }

    /// just two linear layers + relu activation
    pub fn forward(self: MultiLayers, input: zml.Tensor) zml.Tensor {
        var x = input.flatten().convert(.f16).withTags(.{.b});
        const layers: []const Layer = &.{ self.fc1, self.fc2, self.fc3 };
        for (layers) |layer| {
            x = layer.forward(x);
        }
        return x;
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    // Init model
    const multi_model: MultiLayers = .init();

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    // Compile model
    const input: zml.Tensor = .init(.{2}, .f16);
    var exe = blk: {
        log.info("Compiling model....", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("✅ Compiled model [{f}]", .{start.untilNow(io, .awake)});
        break :blk try platform.compile(allocator, io, multi_model, .forward, .{input}, .{ .shardings = &.{replicated_sharding} });
    };
    defer exe.deinit();

    // Load buffers
    var multi_buffers: zml.Bufferized(MultiLayers) = blk: {
        log.info("Initializing weights....", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("✅ Initialized weights [{f}]", .{
            start.untilNow(io, .awake),
        });
        break :blk try multi_model.load(io, platform, &.{replicated_sharding});
    };
    defer MultiLayers.unloadBuffers(&multi_buffers);

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    const input_slice: zml.Slice = .init(input.shape(), std.mem.sliceAsBytes(&[2]f16{ 1.0, 1.0 }));
    var input_buffer: zml.Buffer = try .fromSlice(io, platform, input_slice, replicated_sharding);
    defer input_buffer.deinit();

    args.set(.{ multi_buffers, input_buffer });
    exe.call(args, &results);
    var result: zml.Buffer = results.get(zml.Buffer);
    defer result.deinit();

    // fetch the result buffer to CPU memory
    const result_slice = try result.toSliceAlloc(allocator, io);
    defer result_slice.free(allocator);

    std.debug.print(
        "\n\nThe image of {d} is {d}\n",
        .{ input_slice, result_slice },
    );
}
