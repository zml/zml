const std = @import("std");
const log = std.log;

const zml = @import("zml");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Img = [28 * 28]u8;

/// Model definition
const Mnist = struct {
    fc1: Layer,
    fc2: Layer,

    const Layer = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn init(store: zml.io.TensorStore.View) Layer {
            return .{
                .weight = store.createReplicatedTensor("weight", .{ .d_out, .d }),
                .bias = store.createReplicatedTensor("bias", .{.d_out}),
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

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: zml.Tensor) zml.Tensor {
        var x = input.flatten().convert(.f32).withTags(.{.d});
        const layers: []const Layer = &.{ self.fc1, self.fc2 };
        const compiler = zml.Compiler.current();
        for (0.., layers) |i, layer| {
            compiler.pushLocationFmt(@src(), "layer[{d}]", .{i});
            defer compiler.popLocation();
            x = layer.forward(x);
        }
        return x.argMax(0).indices.convert(.u8);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;

    // Parse program args
    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const model_path = process_args[1];
    const t10kfilename = process_args[2];

    // Local paths. `zml.io.VFS.Remotes` adds hf://, s3://, gs:// and https://.
    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();
    const io = vfs.io();

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    // Returns at once: the loader calibrates while the model is compiled.
    var loader: zml.io.Loader = try .init(allocator, io, platform, .{ .checkpoint = model_path });
    defer loader.deinit();

    // Read model shapes. The store owns the registry.
    const registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    var store: zml.io.TensorStore = .init(allocator, registry, .{});
    defer store.deinit();

    // Init model
    const mnist_model: Mnist = .init(store.view());

    // // Compile model
    const input: zml.Tensor = .init(.{ 28, 28 }, .u8);
    var exe = blk: {
        log.info("Compiling model....", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("✅ Compiled model [{f}]", .{start.untilNow(io, .awake)});
        break :blk try platform.compile(allocator, io, mnist_model, .forward, .{input}, .{});
    };
    defer exe.deinit();

    // Load weights
    var mnist_buffers = try zml.mem.bufferize(allocator, &mnist_model);
    defer zml.mem.deinitBufferized(allocator, &mnist_buffers);
    try loader.start(&mnist_model, &mnist_buffers, &store, .none);
    try loader.wait(); // logs "Loaded weights"

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    // Load a random digit image from the dataset.
    const dataset = try std.Io.Dir.openFile(.cwd(), io, t10kfilename, .{ .mode = .read_only });
    defer dataset.close(io);

    var rng: std.Random.DefaultPrng = blk: {
        const now: std.Io.Timestamp = .now(io, .awake);
        break :blk .init(@intCast(now.toMilliseconds()));
    };

    // inference - can be looped
    const idx = rng.random().uintLessThan(u64, 10000);
    var sample: Img align(16) = undefined;
    _ = try dataset.readPositionalAll(io, &sample, 16 + (idx * @sizeOf(Img)));

    var input_buffer: zml.Buffer = try .fromSlice(io, platform, zml.Slice.init(input.shape(), &sample));
    defer input_buffer.deinit();

    printDigit(&sample);

    args.set(.{ mnist_buffers, input_buffer });
    try exe.call(io, args, &results, .{});
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

pub fn printDigit(digit_flat: *const Img) void {
    const digit: *const [28][28]u8 = @ptrCast(digit_flat);
    var buffer: [28][30][2]u8 = undefined;
    for (0..28) |y| {
        buffer[y][0] = .{ '|', ' ' };
        buffer[y][29] = .{ '|', '\n' };
        for (1..29) |x| {
            const val = digit[y][x - 1];
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

pub const digits = [_][]const u8{
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
