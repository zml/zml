const zml = @import("zml");
const Tensor = zml.Tensor;
const std = @import("std");
const async = @import("async");
const clap = @import("clap");
const log = std.log.scoped(.qwen);
const ShapeOf = zml.ShapeOf;

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(allocator: std.mem.Allocator, store: zml.aio.BufferStore) !Mlp {
        return try zml.aio.populateModelWithPrefix(Mlp, allocator, store, "model.language_model.layers.0.mlp");
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const proj = zml.call(self.up_proj, .forward, .{x});
        var output = zml.call(self.gate_proj, .forward, .{x});
        output = output.silu().mul(proj);
        const result = zml.call(self.down_proj, .forward, .{output});

        return result;
    }
};

const params = clap.parseParamsComptime(
    \\--help                      print this help
    \\--hf-model-path  <STRING>   path to the directory containing model weights, config and tokenizer
    \\--seq-len        <UINT>     sequence length (default: 512)
    \\--create-options <STRING>   platform creation options in ZON format, defaults to {}
);

fn bool_parser(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}

pub fn main() !void {
    try async.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const allocator = std.heap.c_allocator;

    const parsers = comptime .{
        .BOOL = bool_parser,
        .UINT = clap.parsers.int(u32, 0),
        .STRING = clap.parsers.string,
        .PATH = clap.parsers.string,
    };
    var diag: clap.Diagnostic = .{};
    var stderr_buffer: [1024]u8 = undefined;
    var stderr = std.fs.File.stderr().writer(&stderr_buffer);
    defer stderr.interface.flush() catch {};

    var cli = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        diag.report(&stderr.interface, err) catch {};
        stderr.interface.writeAll("usage: ") catch {};
        clap.usage(&stderr.interface, clap.Help, &params) catch {};
        stderr.interface.writeAll("\n") catch {};
        return;
    };
    defer cli.deinit();

    if (cli.args.help != 0) {
        clap.help(&stderr.interface, clap.Help, &params, .{}) catch {};
        return;
    }

    const hf_model_path = cli.args.@"hf-model-path" orelse {
        log.err("Missing --hf-model-path", .{});
        return;
    };

    const model_weights_path = b: {
        const simple_path = try std.fs.path.join(allocator, &.{ hf_model_path, "model.safetensors" });
        if (async.File.access(simple_path, .{})) {
            break :b simple_path;
        } else |_| {
            allocator.free(simple_path);
        }

        const sharded_path = try std.fs.path.join(allocator, &.{ hf_model_path, "model.safetensors.index.json" });
        break :b sharded_path;
    };
    defer allocator.free(model_weights_path);

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/mlp_capacity",
        .sharding_enabled = true,
    };

    const create_opts_zon = cli.args.@"create-options" orelse ".{}";
    const create_opts = std.zon.parse.fromSlice(zml.Platform.CreateOptions, allocator, @ptrCast(create_opts_zon), null, .{ .free_on_error = false }) catch |err| {
        log.err("Failed to parse --create-options as ZON ({}): {s}", .{ err, create_opts_zon });
        return err;
    };

    const platform = context.autoPlatform(create_opts).withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    var store = try zml.aio.detectFormatAndOpen(allocator, model_weights_path);
    defer store.deinit();

    var compiler_arena = std.heap.ArenaAllocator.init(allocator);
    defer compiler_arena.deinit();

    const mlp_tensors: Mlp = try Mlp.init(compiler_arena.allocator(), store);

    const up_proj_shape = mlp_tensors.up_proj.weight.shape();
    log.info("MLP shapes: {f}", .{up_proj_shape});

    const in_features = up_proj_shape.dim(1); // Second dimension is input features
    const batch_size: i64 = 32;

    const input_shape = zml.Shape.init(.{ batch_size, in_features }, up_proj_shape.dtype());

    // Create random input data on CPU
    var rng_state = std.Random.DefaultPrng.init(42);
    const random = rng_state.random();
    const data_size = input_shape.byteSize();
    const random_data = try compiler_arena.allocator().alloc(u8, data_size);
    defer compiler_arena.allocator().free(random_data);

    // Fill with random floats in [-1.0, 1.0] range
    const f32_data = std.mem.bytesAsSlice(f32, random_data);
    for (f32_data) |*val| {
        val.* = @floatCast((random.float(f64) * 2.0) - 1.0);
    }

    const input_host_buffer = zml.HostBuffer.fromBytes(input_shape, random_data);
    const input_buffer = try input_host_buffer.toDevice(platform);
    defer input_buffer.deinit();

    log.info("Input shape: {f}", .{input_shape});

    // const mlp_shapes = try zml.Tensor.shapesOf(mlp_tensors, compiler_arena.allocator());

    var mod = try async.async(zml.compileModel, .{
        allocator,
        Mlp.forward,
        mlp_tensors,
        .{
            input_shape,
        },
        platform,
    });

    log.info("\tLoading MLP weights from {s}...", .{model_weights_path});

    var mlp_buffers = try store.loadModelById(Mlp, compiler_arena.allocator(), mlp_tensors, platform);
    defer zml.aio.unloadBuffers(&mlp_buffers);

    var mlp_model = (try mod.await()).prepare(mlp_buffers);

    //Warmup call
    _ = mlp_model.call(.{input_buffer});

    var timer = try std.time.Timer.start();

    const output_buffer = mlp_model.call(.{input_buffer});
    const mlp_elapsed_ns = timer.read();
    const mlp_elapsed_ms = @as(f64, @floatFromInt(mlp_elapsed_ns)) / std.time.ns_per_ms;

    log.info("Temps d'exécution du MLP: {d:.3} ms", .{mlp_elapsed_ms});
    _ = output_buffer; // autofix
    log.info("Response of MLP", .{});
}
