const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");

const async_ = asynk.async_;

const show_mlir = true;

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
        return x.argMax(0, .u8).indices;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try asynk.AsyncThread.main(allocator, asyncMain, .{});
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    std.debug.print("\n===========================\n==   ZML MNIST Example   ==\n===========================\n\n", .{});

    // Auto-select platform
    const platform = context.autoPlatform();
    {
        // List available targets
        std.debug.print("Available Platforms:\n", .{});
        const selected_prefix = "✅";
        const not_selected_prefix = "• ";
        const selected_postfix = "(AUTO-SELECTED)\n";
        const not_selected_postfix = "\n";
        for (zml.platform.available_targets) |target| {
            std.debug.print("  {s} {s} {s}", .{
                if (target == platform.target) selected_prefix else not_selected_prefix,
                @tagName(target),
                if (target == platform.target) selected_postfix else not_selected_postfix,
            });

            // now the platform's devices
            if (context.platforms.get(target)) |pfm| {
                for (pfm.getDevices(), 0..) |device, index| {
                    const deviceKind = device.getDescription(platform.pjrt_api).getKind(platform.pjrt_api);
                    std.debug.print("       ◦ #{d}: {s}\n", .{
                        index,
                        deviceKind,
                    });
                    // we only list 1 CPU device
                    if (target == .cpu) break;
                }
            }
        }
        std.debug.print("\n", .{});
    }

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
    std.debug.print("✅ Read model shapes from PyTorch file {s}\n", .{pt_model});

    // Start loading weights
    var model_weights = try zml.aio.loadModelBuffers(Mnist, mnist_model, buffer_store, arena, platform);
    defer zml.aio.unloadBuffers(&model_weights);

    // Start compiling
    const comp_start_time = std.time.milliTimestamp();
    if (show_mlir) {
        std.debug.print("\nCompiling model to MLIR....\n", .{});
        std.debug.print("-" ** 160 ++ "\n", .{});
    } else {
        std.debug.print("Compiling model to MLIR....\r", .{});
    }
    var compilation = try async_(zml.compile, .{ allocator, Mnist, .{}, .forward, .{zml.Shape.init(.{ 28, 28 }, .u8)}, buffer_store, platform });

    // Wait for end of compilation and end of weights loading.
    const compiled_mnist = try compilation.await_();
    const comp_end_time = std.time.milliTimestamp();
    if (show_mlir) std.debug.print("-" ** 160 ++ "\n", .{});
    std.debug.print("✅ Compiled MNIST model in {d} milliseconds! \n", .{comp_end_time - comp_start_time});

    // send weights to accelerator / GPU
    var mnist = try compiled_mnist.prepare(allocator, model_weights);
    defer mnist.deinit();
    std.debug.print("✅ Weights transferred, starting inference...\n\n", .{});

    // Load a random digit image from the dataset.
    const dataset = try asynk.File.open(t10kfilename, .{ .mode = .read_only });
    defer dataset.close() catch unreachable;
    var rng = std.Random.Xoshiro256.init(@intCast(std.time.timestamp()));

    // inference - can be looped
    {
        const idx = rng.random().intRangeAtMost(u64, 0, 10000 - 1);
        var sample: [28 * 28]u8 align(16) = undefined;
        _ = try dataset.pread(&sample, 16 + (idx * 28 * 28));
        var input = try zml.Buffer.from(platform, zml.HostBuffer.fromBytes(zml.Shape.init(.{ 28, 28 }, .u8), &sample));
        defer input.deinit();

        printDigit(sample);
        var result: zml.Buffer = mnist.call(.{input});
        defer result.deinit();

        std.debug.print("\n✅ RECOGNIZED DIGIT:\n", .{});
        std.debug.print("                       +-------------+\n", .{});
        std.debug.print("{s}\n", .{digits[try result.getValue(u8)]});
        std.debug.print("                       +-------------+\n\n", .{});
    }
}

fn printDigit(digit: [28 * 28]u8) void {
    var buffer: [28][30][2]u8 = undefined;
    std.debug.print("     R E C O G N I Z I N G   I N P U T   I M A G E :\n", .{});
    std.debug.print("+---------------------------------------------------------+\n", .{});
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
    std.fmt.format(asynk.StdOut().writer(), "{s}", .{std.mem.asBytes(&buffer)}) catch unreachable;
    std.debug.print("+---------------------------------------------------------+\n", .{});
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

pub const std_options = .{
    // Set the global log level to err
    .log_level = .err,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .pjrt, .level = .err },
        .{ .scope = .zml_module, .level = if (show_mlir) .debug else .err },
    },
};
