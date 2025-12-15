const std = @import("std");
const stdx = @import("stdx");

const zml = @import("zml");

const log = std.log.scoped(.mnist);

pub const std_options: std.Options = .{
    .log_level = .info,
};

/// Model definition
const Mnist = struct {
    fc1: Layer,
    fc2: Layer,

    const Layer = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn forward(self: Layer, input: zml.Tensor) zml.Tensor {
            return self.weight.dot(input, .d).add(self.bias).relu().withTags(.{.d});
        }
    };

    pub fn init(registry: *const zml.safetensors.TensorRegistry) Mnist {
        const fc1_weight = b: {
            const desc = registry.tensors.get("fc1.weight").?;
            break :b zml.Tensor.fromShape(desc.shape).withTags(.{ .d_out, .d });
        };
        const fc1_bias: zml.Tensor = b: {
            const desc = registry.tensors.get("fc1.bias").?;
            break :b zml.Tensor.fromShape(desc.shape).withTags(.{.d_out});
        };
        const fc2_weight = b: {
            const desc = registry.tensors.get("fc2.weight").?;
            break :b zml.Tensor.fromShape(desc.shape).withTags(.{ .d_out, .d });
        };
        const fc2_bias: zml.Tensor = b: {
            const desc = registry.tensors.get("fc2.bias").?;
            break :b zml.Tensor.fromShape(desc.shape).withTags(.{.d_out});
        };
        return .{
            .fc1 = .{ .weight = fc1_weight, .bias = fc1_bias },
            .fc2 = .{ .weight = fc2_weight, .bias = fc2_bias },
        };
    }

    pub fn deinit(self: *const Mnist) void {
        _ = self; // autofix
    }

    pub fn loadBuffers(self: Mnist, allocator: std.mem.Allocator, io: std.Io, vfs: *zml.io.VFS, registry: *zml.safetensors.TensorRegistry, platform: zml.Platform) !zml.Bufferized(Mnist) {
        _ = self; // autofix
        const read_buffer = try allocator.alloc(u8, 4096);
        defer allocator.free(read_buffer);

        const fc1_weight: zml.Buffer = b: {
            const desc = registry.tensors.get("fc1.weight").?;
            const slice = try zml.Slice.alloc(allocator, desc.shape);
            defer slice.free(allocator);

            var writer = std.Io.Writer.fixed(slice.data);
            var reader = try registry.reader(io, vfs, "fc1.weight", read_buffer);

            _ = try reader.interface.streamRemaining(&writer);
            break :b try .fromSlice(io, platform, slice);
        };
        errdefer fc1_weight.deinit();
        const fc1_bias: zml.Buffer = b: {
            const desc = registry.tensors.get("fc1.bias").?;
            const slice = try zml.Slice.alloc(allocator, desc.shape);
            defer slice.free(allocator);

            var writer = std.Io.Writer.fixed(slice.data);
            var reader = try registry.reader(io, vfs, "fc1.bias", read_buffer);

            _ = try reader.interface.streamRemaining(&writer);
            break :b try .fromSlice(io, platform, slice);
        };
        errdefer fc1_bias.deinit();
        const fc2_weight: zml.Buffer = b: {
            const desc = registry.tensors.get("fc2.weight").?;
            const slice = try zml.Slice.alloc(allocator, desc.shape);
            defer slice.free(allocator);

            var writer = std.Io.Writer.fixed(slice.data);
            var reader = try registry.reader(io, vfs, "fc2.weight", read_buffer);

            _ = try reader.interface.streamRemaining(&writer);
            break :b try .fromSlice(io, platform, slice);
        };
        errdefer fc2_weight.deinit();
        const fc2_bias: zml.Buffer = b: {
            const desc = registry.tensors.get("fc2.bias").?;
            const slice = try zml.Slice.alloc(allocator, desc.shape);
            defer slice.free(allocator);

            var writer = std.Io.Writer.fixed(slice.data);
            var reader = try registry.reader(io, vfs, "fc2.bias", read_buffer);

            _ = try reader.interface.streamRemaining(&writer);
            break :b try .fromSlice(io, platform, slice);
        };
        errdefer fc2_bias.deinit();

        return .{
            .fc1 = .{ .weight = fc1_weight, .bias = fc1_bias },
            .fc2 = .{ .weight = fc2_weight, .bias = fc2_bias },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mnist)) void {
        self.fc1.weight.deinit();
        self.fc1.bias.deinit();
        self.fc2.weight.deinit();
        self.fc2.bias.deinit();
    }

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: zml.Tensor) zml.Tensor {
        // std.log.info("Compiling for target: {s}", .{@tagName(input.getContext().target())});
        var x = input.flatten().convert(.f32).withTags(.{.d});
        const layers: []const Layer = &.{ self.fc1, self.fc2 };
        for (layers) |layer| {
            x = layer.forward(x);
        }
        return x.argMax(0).indices.convert(.u8);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    var vfs_file: zml.io.VFS.File = .init(threaded.io());

    var vfs: zml.io.VFS = .init(allocator, threaded.io());
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());

    const io = vfs.io();

    zml.init();
    defer zml.deinit();

    // Auto-select platform
    const platform: zml.Platform = try .auto(threaded.io(), .{});

    // Parse program args
    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);
    const model_path = process_args[1];
    const t10kfilename = process_args[2];

    // Read model shapes.
    var registry = try zml.safetensors.parseFromPath(allocator, io, &vfs, model_path);
    defer registry.deinit();

    // Init model
    const mnist_model = Mnist.init(&registry);
    defer mnist_model.deinit();

    // Compile model
    log.info("Compiling model to MLIR....", .{});
    var timer = try stdx.time.Timer.start();
    const input: zml.Tensor = .init(.{ 28, 28 }, .u8);
    var exe = try platform.compileModel(allocator, io, Mnist.forward, mnist_model, .{input});
    defer exe.deinit();

    log.info("✅ Compiled model in {f}", .{timer.read()});

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    // Load buffers
    var mnist_buffers = try mnist_model.loadBuffers(allocator, io, &vfs, &registry, platform);
    defer Mnist.unloadBuffers(&mnist_buffers);
    log.info("✅ Weights transferred in {f}", .{timer.read()});

    log.info("Starting inference...", .{});

    // Load a random digit image from the dataset.
    const dataset = try vfs.openAbsoluteFile(io, t10kfilename, .{ .mode = .read_only });
    defer dataset.close(io);

    const now = std.Io.Clock.now(.awake, io) catch unreachable;
    var rng = std.Random.Xoshiro256.init(@intCast(now.toMilliseconds()));

    // inference - can be looped
    {
        const idx = rng.random().intRangeAtMost(u64, 0, 10000 - 1);
        var sample: [28 * 28]u8 align(16) = undefined;
        var reader = dataset.reader(io, &.{});
        try reader.seekTo(16 + (idx * 28 * 28));
        _ = try reader.interface.readSliceShort(&sample);

        var input_buffer: zml.Buffer = try .fromSlice(io, platform, zml.Slice.init(input.shape(), &sample));
        defer input_buffer.deinit();

        args.set(.{ mnist_buffers, input_buffer });

        printDigit(sample);
        exe.call(args, &results, io);

        var result: zml.Buffer = results.get(zml.Buffer);
        defer result.deinit();

        log.info(
            \\✅ RECOGNIZED DIGIT:
            \\                       +-------------+
            \\{s}
            \\                       +-------------+
            \\
        , .{digits[try result.getValue(u8, io)]});
    }
}

fn printDigit(digit: [28 * 28]u8) void {
    var buffer: [28][30][2]u8 = undefined;
    for (0..28) |y| {
        buffer[y][0] = .{ '|', ' ' };
        buffer[y][29] = .{ '|', '\n' };
        for (1..29) |x| {
            const idx = (y * 28) + (x - 1);
            const val = digit[idx];
            buffer[y][x] = blk: {
                if (val > 240) break :blk .{ '*', '*' };
                if (val > 225) break :blk .{ 'o', 'o' };
                if (val > 210) break :blk .{ '.', '.' };
                break :blk .{ ' ', ' ' };
            };
        }
    }

    log.info(
        \\
        \\     R E C O G N I Z I N G   I N P U T   I M A G E :
        \\+---------------------------------------------------------+
        \\{s}+---------------------------------------------------------+
        \\
    , .{std.mem.asBytes(&buffer)});
}

const digits = [_][]const u8{
    \\                       |     ###     |
    \\                       |    #   #    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    #   #    |
    \\                       |     ###     |
    ,
    \\                       |      #      |
    \\                       |     ##      |
    \\                       |    # #      |
    \\                       |      #      |
    \\                       |      #      |
    \\                       |      #      |
    \\                       |    #####    |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |         #   |
    \\                       |    #####    |
    \\                       |   #         |
    \\                       |   #         |
    \\                       |   #######   |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |         #   |
    \\                       |    #####    |
    \\                       |         #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
    \\                       |   #         |
    \\                       |   #    #    |
    \\                       |   #    #    |
    \\                       |   #    #    |
    \\                       |   #######   |
    \\                       |        #    |
    \\                       |        #    |
    ,
    \\                       |   #######   |
    \\                       |   #         |
    \\                       |   #         |
    \\                       |   ######    |
    \\                       |         #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |   #         |
    \\                       |   ######    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
    \\                       |   #######   |
    \\                       |   #    #    |
    \\                       |       #     |
    \\                       |      #      |
    \\                       |     #       |
    \\                       |     #       |
    \\                       |     #       |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    ######   |
    \\                       |         #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
};
