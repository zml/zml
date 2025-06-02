const std = @import("std");

const asynk = @import("async");
const zml = @import("zml");

const log = std.log.scoped(.mnist);

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

/// Model definition
const Mnist = struct {
    fc1: Layer,
    fc2: Layer,

    const Layer = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn forward(self: Layer, input: zml.Tensor) zml.Tensor {
            return self.weight.matmul(input).add(self.bias).relu();
        }
    };

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: zml.Tensor) zml.Tensor {
        // std.log.info("Compiling for target: {s}", .{@tagName(input.getContext().target())});
        var x = input.flattenAll().convert(.f32);
        const layers: []const Layer = &.{ self.fc1, self.fc2 };
        for (layers) |layer| {
            x = zml.call(layer, .forward, .{x});
        }
        return x.argMax(0).indices.convert(.u8);
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const allocator = std.heap.c_allocator;

    // // Create ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    // log.info("\n===========================\n==   ZML MNIST Example   ==\n===========================\n\n", .{});

    // // Auto-select platform
    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    // Parse program args
    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);
    const pt_model = process_args[1];
    const t10kfilename = process_args[2];

    // Memory arena dedicated to model shapes and weights
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Read model shapes.
    // Note this works because Mnist struct uses the same layer names as the pytorch model
    var buffer_store = try zml.aio.torch.open(allocator, pt_model);
    defer buffer_store.deinit();

    const mnist_model = try zml.aio.populateModel(Mnist, allocator, buffer_store);
    log.info("Reading model shapes from PyTorch file {s}...", .{pt_model});

    // Start compiling
    log.info("Compiling model to MLIR....", .{});
    var start_time = try std.time.Timer.start();
    var compilation = try asynk.asyncc(zml.compile, .{ allocator, Mnist.forward, .{}, .{zml.Shape.init(.{ 28, 28 }, .u8)}, buffer_store, platform });

    // While compiling, start loading weights on the platform
    var model_weights = try zml.aio.loadModelBuffers(Mnist, mnist_model, buffer_store, arena, platform);
    defer zml.aio.unloadBuffers(&model_weights);

    // Wait for end of compilation and end of weights loading.
    const compiled_mnist = try compilation.awaitt();
    log.info("✅ Compiled model in {d}ms", .{start_time.read() / std.time.ns_per_ms});

    const mnist = compiled_mnist.prepare(model_weights);
    defer mnist.deinit();
    log.info("✅ Weights transferred in {d}ms", .{start_time.read() / std.time.ns_per_ms});

    log.info("Starting inference...", .{});

    // Load a random digit image from the dataset.
    const dataset = try asynk.File.open(t10kfilename, .{ .mode = .read_only });
    defer dataset.close() catch unreachable;
    var rng = std.Random.Xoshiro256.init(@intCast(std.time.timestamp()));

    // inference - can be looped
    {
        const idx = rng.random().intRangeAtMost(u64, 0, 10000 - 1);
        var sample: [28 * 28]u8 align(16) = undefined;
        _ = try dataset.pread(&sample, 16 + (idx * 28 * 28));
        var input = try zml.Buffer.from(platform, zml.HostBuffer.fromBytes(zml.Shape.init(.{ 28, 28 }, .u8), &sample), .{});
        defer input.deinit();

        printDigit(sample);
        var result: zml.Buffer = mnist.call(.{input});
        defer result.deinit();

        log.info(
            \\✅ RECOGNIZED DIGIT:
            \\                       +-------------+
            \\{s}
            \\                       +-------------+
            \\
        , .{digits[try result.getValue(u8)]});
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
