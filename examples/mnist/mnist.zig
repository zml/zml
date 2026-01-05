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

        pub fn init(store: zml.io.TensorStore.View) Layer {
            return .{
                .weight = store.createTensorWithTags("weight", .{ .d_out, .d }),
                .bias = store.createTensorWithTags("bias", .{.d_out}),
            };
        }

        pub fn forward(self: Layer, input: zml.Tensor) zml.Tensor {
            return self.weight.dot(input, .d).add(self.bias).relu().withTags(.{.d});
        }
    };

    pub fn init(store: zml.io.TensorStore.View) Mnist {
        return .{
            .fc1 = .init(store.withPrefix("fc1")),
            .fc2 = .init(store.withPrefix("fc2")),
        };
    }

    pub fn loadBuffers(self: Mnist, allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, platform: zml.Platform) !zml.Bufferized(Mnist) {
        return zml.io.loadBuffersFromId(allocator, io, self, store, platform);
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
    const allocator = std.heap.c_allocator;
    //var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    //defer _ = gpa.deinit();

    //const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    // var vfs_file: zml.io.VFS.File = .init(allocator, threaded.io(), .{});

    // var vfs: zml.io.VFS = .init(allocator, threaded.io());
    // defer vfs.deinit();

    // try vfs.register("file", vfs_file.io());

    // const io = vfs.io();
    //
    const io = threaded.io();

    zml.init();
    defer zml.deinit();

    // Parse program args
    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);
    const model_path = process_args[1];
    const t10kfilename = process_args[2];

    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    defer registry.deinit();

    // Init model
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const mnist_model: Mnist = .init(store.view());

    // Auto-select platform
    const platform: zml.Platform = try .auto(io, .{});

    // Compile model
    const input: zml.Tensor = .init(.{ 28, 28 }, .u8);
    var exe = blk: {
        log.info("Compiling model....", .{});
        var timer = try std.time.Timer.start();
        defer log.info("✅ Compiled model in {D}", .{timer.read()});
        break :blk try platform.compile(allocator, io, mnist_model, .forward, .{input});
    };
    defer exe.deinit();

    // Load buffers
    var mnist_buffers = blk: {
        log.info("Transfering weights....", .{});
        var timer = try std.time.Timer.start();
        defer log.info("✅ Transferred weights in {D}", .{timer.read()});
        break :blk try mnist_model.loadBuffers(allocator, io, store.view(), platform);
    };
    defer Mnist.unloadBuffers(&mnist_buffers);

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    // Load a random digit image from the dataset.
    const dataset = try std.Io.Dir.openFile(.cwd(), io, t10kfilename, .{ .mode = .read_only });
    defer dataset.close(io);

    var rng: std.Random.DefaultPrng = blk: {
        const now = try std.Io.Clock.now(.awake, io);
        break :blk .init(@intCast(now.toMilliseconds()));
    };

    // inference - can be looped
    while (true) {
        const idx = rng.random().uintLessThan(u64, 10000);
        var sample: [28 * 28]u8 align(16) = undefined;
        _ = try dataset.readPositionalAll(io, &sample, 16 + (idx * 28 * 28));

        var input_buffer: zml.Buffer = try .fromSlice(io, platform, zml.Slice.init(input.shape(), &sample));
        defer input_buffer.deinit();

        printDigit(sample);

        args.set(.{ mnist_buffers, input_buffer });
        exe.call(args, &results);
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
