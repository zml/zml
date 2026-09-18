const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/compiler", .level = .debug },
    },
};

/// Model definition
const Mnist = struct {
    fc1: Layer,
    fc2: Layer,

    pub fn init(store: zml.io.TensorStore.View) Mnist {
        return .{
            // Layer 1 is sharded following it's output axis
            .fc1 = .{
                .weight = store.createTensor("fc1.weight", .{ .d_out, .d }, .{ .d_out = .model }),
                .bias = store.createTensor("fc1.bias", .{.d_out}, .{ .d_out = .model }),
            },
            // Layer 2 is sharded following it's input axis (and bias is fully replicated)
            .fc2 = .{
                .weight = store.createTensor("fc2.weight", .{ .d_out, .d }, .{ .d = .model }),
                .bias = store.createTensor("fc2.bias", .{.d_out}, .replicated),
            },
        };
    }

    pub fn load(
        self: *const Mnist,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        sharding: zml.Sharding,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Mnist) {
        var buffers = try zml.mem.bufferize(allocator, Mnist, self);
        errdefer zml.Buffer.deinitAll(Mnist, &buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        errdefer loader.deinit();

        try loader.load(io, Mnist, self, &buffers, store, &.{sharding}, .{});
        try loader.await(io);

        return buffers;
    }

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: zml.Tensor) zml.Tensor {
        var x = input.onMemory(.host_pinned).toMemory(.device).merge(.{ .d = .{ .x, .y } }).convert(.f32);
        const layers: []const Layer = &.{ self.fc1, self.fc2 };
        for (layers) |layer| {
            x = layer.forward(x);
        }
        return x.argMax(.d).indices.convert(.u8).withPartitioning(.{ .b = .data }).toMemory(.host_pinned);
    }

    const Layer = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn forward(self: Layer, input: zml.Tensor) zml.Tensor {
            log.warn("layer(input={f}, w={f}, b={f}) -> {f}", .{ input, self.weight, self.bias, input.dot(self.weight, .d) });
            const x = input.dot(self.weight, .d);
            return x.add(self.bias.broad(x.shape())).relu().rename(.{ .d_out = .d });
        }
    };
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    // Parse program args
    const process_args = try init.minimal.args.toSlice(arena.allocator());
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
    const target: zml.Target = try .selectFirstAcceleratorEnabled();
    const platform: *zml.Platform = try .init(allocator, io, target, .{
        .physical_mesh = if (target == .cpu) .{ .custom = zml.Sharding.PhysicalMesh.torus2x2 } else .auto,
    });
    defer platform.deinit(allocator, io);

    // Decide how to partition the devices of our hardware.
    // Here we use a mixed Data parallel / Model parallel
    const dp_mp = try platform.registerSharding("DP-MP", .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }));
    log.info("topology used: {f}", .{dp_mp});

    var profiler = try platform.profiler(allocator, io, .{
        .session_id = "mnist",
    });
    defer profiler.deinit();

    // // Compile model
    const bs: u32 = 4;
    const input: zml.Tensor = .withPartitioning(.init(.{ .b = bs, .x = 28, .y = 28 }, .u8), .{ .b = .data });
    var exe = exe: {
        log.info("Compiling model....", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("✅ Compiled model [{f}]", .{start.untilNow(io, .awake)});
        break :exe try platform.compile(
            allocator,
            io,
            mnist_model,
            .forward,
            .{input},
            .{
                .shardings = &.{dp_mp},
                .program_name = "mnist",
                .xla_dump_to = "/tmp/zml/mnist",
            },
        );
    };
    defer exe.deinit();

    // Load buffers
    var mnist_buffers = blk: {
        log.info("Transfering weights....", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("✅ Transferred weights [{f}]", .{
            start.untilNow(io, .awake),
        });
        break :blk try mnist_model.load(init.arena.allocator(), io, platform, dp_mp, &store);
    };
    defer zml.Buffer.deinitAll(Mnist, &mnist_buffers);

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
    const Img = [28][28]u8;
    var batch: []align(16) Img = try allocator.alignedAlloc(Img, .@"16", bs);
    defer allocator.free(batch);

    for (0..bs) |i| {
        const rand_id = rng.random().uintLessThan(u64, 10000);
        _ = try dataset.readPositionalAll(io, @ptrCast(&batch[i]), 16 + (rand_id * @sizeOf(Img)));
    }

    // This performs a host to host pinned memcpy.
    // TODO: show how to allocate a Host pinned buffer and write into it.
    var input_buffer: zml.Buffer = try .fromBytesOpts(io, platform, input.shape(), dp_mp, @ptrCast(batch), .{ .memory = .host_pinned });
    defer input_buffer.deinit();

    for (batch[0..]) |*sample| printDigit(sample);

    args.set(.{ mnist_buffers, input_buffer });

    try profiler.start();
    exe.call(args, &results);
    if (try profiler.stop()) |report| {
        log.info("Wrote profiler files to {s} and {s}", .{ report.protobuf_path, report.perfetto_path });
    }
    var recognized_digits_d: zml.Buffer = results.get(zml.Buffer);
    defer recognized_digits_d.deinit();

    const recognized_digits = try recognized_digits_d.getValue([bs]u8, io);
    for (0..bs) |i| {
        log.info(
            \\✅ Shard {d} RECOGNIZED DIGIT:
            \\                       +-------------+
            \\{s}
            \\                       +-------------+
            \\
        , .{ i, digits[recognized_digits[i]] });
    }
}

fn printDigit(digit: *const [28][28]u8) void {
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
