//! Data-parallel version of mnist.zig,
//! Separated to kept `mnist.zig` minimal

const std = @import("std");
const zml = @import("zml");

const base = @import("mnist.zig");
const log = std.log.scoped(.mnist);

pub const std_options: std.Options = .{
    .log_level = .info,
};

/// Model definition
const Mnist = struct {
    fc1: Layer,
    fc2: Layer,

    pub fn init(store: zml.io.TensorStore.View) Mnist {
        return .{
            // Layer 1 is sharded following it's output axis
            .fc1 = .{
                .weight = store.createTensor("fc1.weight", .{ .d_out, .d }, .dp_mp, .{ .d_out = .model }),
                .bias = store.createTensor("fc1.bias", .{.d_out}, .dp_mp, .{ .d_out = .model }),
            },
            // Layer 2 is sharded following it's input axis (and bias is fully replicated)
            .fc2 = .{
                .weight = store.createTensor("fc2.weight", .{ .d_out, .d }, .dp_mp, .{ .d = .model }),
                .bias = store.createReplicatedTensor("fc2.bias", .{.d_out}),
            },
        };
    }

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: zml.Tensor) zml.Tensor {
        var x = input.onMemory(.host_pinned).toMemory(.device).merge(.{ .d = .{ .x, .y } }).convert(.f32);
        const layers: []const Layer = &.{ self.fc1, self.fc2 };
        const compiler = zml.Compiler.current();
        for (0.., layers) |i, layer| {
            compiler.pushLocationFmt(@src(), "layer[{d}]", .{i});
            defer compiler.popLocation();
            x = layer.forward(x);
        }
        return x.argMax(.d).indices.convert(.u8).withPartitioning(input.shape()._sharding.mesh.?, .{ .b = .data }).toMemory(.host_pinned);
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

    // Decide how to partition the devices of our hardware.
    // Here we use a mixed Data parallel / Model parallel
    const dp_mp = try platform.registerMesh("dp_mp", try .init(
        .{ .data, .model },
        &.{ .low_bandwidth, .high_bandwidth },
        &.{ 2, -1 },
    ));
    log.info("topology used: {f}", .{dp_mp});

    // Returns at once: the loader calibrates while the model is compiled.
    var loader: zml.io.Loader = try .init(allocator, io, platform, .{ .checkpoint = model_path });
    defer loader.deinit();

    var profiler = try platform.profiler(allocator, io, .{ .session_id = "mnist" });
    defer profiler.deinit();

    // Read model shapes. The store owns the registry.
    const registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    var store: zml.io.TensorStore = .init(allocator, registry, .{ .meshes = &.{dp_mp} });
    defer store.deinit();
    const mnist_model: Mnist = .init(store.view());

    // // Compile model
    const bs: u32 = 4;
    const input: zml.Tensor = .withPartitioning(.init(.{ .b = bs, .x = 28, .y = 28 }, .u8), dp_mp, .{ .b = .data });
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
                .program_name = "mnist",
                .meshes = &.{dp_mp},
                .xla_dump_to = "/tmp/zml/mnist",
            },
        );
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
    var batch: []align(16) base.Img = try allocator.alignedAlloc(base.Img, .@"16", bs);
    defer allocator.free(batch);

    for (0..bs) |i| {
        const rand_id = rng.random().uintLessThan(u64, 10000);
        _ = try dataset.readPositionalAll(io, @ptrCast(&batch[i]), 16 + (rand_id * @sizeOf(base.Img)));
    }

    // This performs a host to host pinned memcpy.
    // TODO: show how to allocate a Host pinned buffer and write into it.
    var input_buffer: zml.Buffer = try .fromBytesOpts(io, platform, input.shape(), std.mem.sliceAsBytes(batch), .{ .memory = .host_pinned });
    defer input_buffer.deinit();

    for (batch[0..]) |*sample| base.printDigit(sample);

    args.set(.{ mnist_buffers, input_buffer });

    try profiler.start();
    try exe.call(io, args, &results, .{});
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
        , .{ i, base.digits[recognized_digits[i]] });
    }
}
