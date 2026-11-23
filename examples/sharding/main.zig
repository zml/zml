const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.mesh);

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/compiler", .level = .debug },
    },
};

const CliMesh = enum {
    auto,
    mock,
};

const CliArgs = struct {
    partitioner: ?zml.Sharding.Partitioner = null,
    mesh: CliMesh = .auto,
    target: ?zml.Target = null,
    coordinator: ?std.Io.net.IpAddress = null,
    process_index: usize = 0,
    process_count: usize = 1,
    namespace: []const u8 = "zml-sharding",
    local_devices: std.ArrayList(i64) = .empty,
};

const DemoModel = struct {
    w: zml.Tensor, // {feature, hidden}
    b: zml.Tensor, // {hidden}

    pub fn init(w: zml.Tensor, b: zml.Tensor) DemoModel {
        return .{ .w = w, .b = b };
    }

    pub fn forward(self: DemoModel, input: zml.Tensor) zml.Tensor {
        const compiler = zml.Compiler.current();

        const x = input.convert(.f32);
        var y = x.dot(self.w, .feature);
        y = y.add(self.b.broad(y.shape()));
        y = y.withPartitioning(.dp_mp, .{ .batch = .data, .hidden = .model });
        y.print("dense_out");

        const gate = y.scale(0.01).sigmoid();

        const spec: zml.Sharding.Partitioning.Logical = .parse(.{ .batch = .data, .hidden = .model });

        return zml.ops.manualComputation(
            (struct {
                y: zml.Tensor,
                gate: zml.Tensor,

                fn body(ctx: @This(), output_sharded: zml.Shape) zml.Tensor {
                    const local_result = ctx.y.relu().mul(ctx.gate);
                    return local_result.reshape(output_sharded);
                }
            }).body,
            .{ .y = y, .gate = gate },
            y.shape(),
            compiler.mesh(.dp_mp),
            .{
                .in_specs = .{ .y = spec, .gate = spec },
                .out_specs = spec,
            },
        );
    }
};

fn usage() void {
    std.debug.print(
        \\Usage: sharding [--partitioner=shardy|gspmd] [--mesh=auto|mock] [--target=cuda|rocm|...]
        \\                [--coordinator=IP:PORT --process-index=N --process-count=N
        \\                 [--local-devices=0,1,...] [--namespace=NAME]]
        \\
        \\Simple sharding demo:
        \\  - explicit partitioning on data/model axes
        \\  - tensor print
        \\  - manualComputation with multiple inputs and one output
        \\
        \\Mesh modes:
        \\  - auto: physical_mesh=auto, inferred cpu device_count=2
        \\  - mock: physical_mesh=mock topology, inferred cpu device_count=9
        \\
        \\Multi-process (GPU only): start one process per rank with the same
        \\--coordinator (hosted by process 0), --process-count and --namespace.
        \\The data axis then spans processes and the model axis stays within one.
        \\Example, 2 processes x 4 GPUs on one 8-GPU host:
        \\  sharding --coordinator=127.0.0.1:8476 --process-count=2 --process-index=0 --local-devices=0,1,2,3
        \\  sharding --coordinator=127.0.0.1:8476 --process-count=2 --process-index=1 --local-devices=4,5,6,7
        \\
    , .{});
}

fn parseArgs(allocator: std.mem.Allocator, init: std.process.Init) !CliArgs {
    var args: CliArgs = .{};
    var it = init.minimal.args.iterate();
    _ = it.next(); // program name

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            usage();
            std.process.exit(0);
        } else if (std.mem.startsWith(u8, arg, "--partitioner=")) {
            const value = arg["--partitioner=".len..];
            if (std.mem.eql(u8, value, "shardy")) {
                args.partitioner = .shardy;
            } else if (std.mem.eql(u8, value, "gspmd")) {
                args.partitioner = .gspmd;
            } else {
                std.debug.print("error: unknown partitioner '{s}'\n\n", .{value});
                usage();
                return error.InvalidPartitioner;
            }
        } else if (std.mem.startsWith(u8, arg, "--mesh=")) {
            const value = arg["--mesh=".len..];
            if (std.mem.eql(u8, value, "auto")) {
                args.mesh = .auto;
            } else if (std.mem.eql(u8, value, "mock")) {
                args.mesh = .mock;
            } else {
                std.debug.print("error: unknown mesh mode '{s}'\n\n", .{value});
                usage();
                return error.InvalidMeshMode;
            }
        } else if (std.mem.startsWith(u8, arg, "--target=")) {
            args.target = std.meta.stringToEnum(zml.Target, arg["--target=".len..]) orelse return error.InvalidTarget;
        } else if (std.mem.startsWith(u8, arg, "--coordinator=")) {
            args.coordinator = try std.Io.net.IpAddress.parseLiteral(arg["--coordinator=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--process-index=")) {
            args.process_index = try std.fmt.parseInt(usize, arg["--process-index=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--process-count=")) {
            args.process_count = try std.fmt.parseInt(usize, arg["--process-count=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--namespace=")) {
            args.namespace = arg["--namespace=".len..];
        } else if (std.mem.startsWith(u8, arg, "--local-devices=")) {
            var ids = std.mem.splitScalar(u8, arg["--local-devices=".len..], ',');
            while (ids.next()) |id| try args.local_devices.append(allocator, try std.fmt.parseInt(i64, id, 10));
        } else {
            std.debug.print("error: unknown argument '{s}'\n\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }
    return args;
}

fn buildMockMesh(
    allocator: std.mem.Allocator,
    target: zml.Target,
    devices: []const zml.platform.Device,
) !zml.Sharding.PhysicalMesh {
    if (devices.len < 8) return error.NotEnoughDevicesForMockMesh;
    const topology: zml.Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[3]), .device(devices[1]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[2]), .device(devices[0]),
            }),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[4]), .device(devices[5]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[6]), .device(devices[7]),
            }),
        }),
    });
    return zml.Sharding.PhysicalMesh.fromTree(allocator, target, topology);
}

fn createSequenceBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    start: f32,
) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    for (slice.items(f32), 0..) |*e, i| {
        e.* = start + @as(f32, @floatFromInt(i));
    }

    return zml.Buffer.fromSlice(io, platform, slice);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args = try parseArgs(allocator, init);
    defer args.local_devices.deinit(allocator);

    var create_options: zml.platform.CreateOptions = switch (args.mesh) {
        .auto => .{},
        .mock => .{
            .cpu = .{ .device_count = 8 },
            .physical_mesh = .{ .custom = buildMockMesh },
        },
    };
    if (args.coordinator) |coordinator| {
        create_options.distributed = .{
            .coordinator_address = coordinator,
            .process_index = args.process_index,
            .process_count = args.process_count,
            .namespace = args.namespace,
            .local_device_ids = args.local_devices.items,
        };
    }

    // Multi-process platforms need an explicit target; default to CUDA.
    const target: ?zml.Target = args.target orelse if (args.coordinator != null) .cuda else null;
    var platform: *zml.Platform = if (target) |t|
        try .init(allocator, io, t, create_options)
    else
        try .auto(allocator, io, create_options);
    defer platform.deinit(allocator, io);

    log.info("\n{f}", .{platform.fmtVerbose()});

    var profiler_options: zml.Platform.ProfilerOptions = .defaults;
    profiler_options.repository_path = "/tmp/xprof";
    profiler_options.session_id = "profiling";

    var profiler = try platform.profiler(allocator, io, profiler_options);
    defer profiler.deinit();

    try profiler.start();
    defer {
        _ = profiler.stop() catch unreachable;
    }

    if (args.partitioner) |partitioner| {
        log.info("Partitioner: {s}", .{@tagName(partitioner)});
    } else {
        args.partitioner = .fromTarget(platform.target);
        log.info("Partitioner: {s} (default)", .{@tagName(args.partitioner.?)});
    }
    log.info("{f}", .{platform.physical_mesh});

    const mesh: *const zml.Mesh = try platform.registerMesh(
        "dp_mp",
        .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }),
    );

    log.info("{f}", .{mesh.logical});
    log.info("{f}", .{mesh});

    const input_shape = zml.Shape.init(.{ .batch = 16, .feature = 32 }, .f32)
        .withPartitioning(mesh, .{ .batch = .data, .feature = .replicated });
    const w_shape = zml.Shape.init(.{ .feature = 32, .hidden = 64 }, .f32)
        .withPartitioning(mesh, .{ .feature = .replicated, .hidden = .model });
    const b_shape = zml.Shape.init(.{ .hidden = 64 }, .f32)
        .withPartitioning(mesh, .{ .hidden = .model });

    const input: zml.Tensor = zml.Tensor.fromShape(input_shape);
    const w: zml.Tensor = zml.Tensor.fromShape(w_shape);
    const b: zml.Tensor = zml.Tensor.fromShape(b_shape);
    const model: DemoModel = .init(w, b);

    var exe = try platform.compile(
        allocator,
        io,
        model,
        .forward,
        .{input},
        .{
            .partitioner = args.partitioner,
            .meshes = &.{mesh},
        },
    );
    defer exe.deinit();

    var w_buf = try createSequenceBuffer(allocator, io, platform, w_shape, 0.0);
    defer w_buf.deinit();
    var b_buf = try createSequenceBuffer(allocator, io, platform, b_shape, 100.0);
    defer b_buf.deinit();
    var input_buf = try createSequenceBuffer(allocator, io, platform, input_shape, 1000.0);
    defer input_buf.deinit();

    log.info("input placement: {f}", .{input_buf});
    log.info("weight placement: {f}", .{w_buf});
    log.info("bias placement: {f}", .{b_buf});

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ w_buf, b_buf, input_buf });

    exe.call(io, exe_args, &exe_results, .{});
    var out = exe_results.get(zml.Buffer);
    defer out.deinit();

    const out_slice = out.toSliceAlloc(allocator, io) catch |err| switch (err) {
        error.GlobalReadRequiresGather => {
            // `dense_out` above is printed by each device of each process.
            log.info("output is split across processes, skipping host preview", .{});
            return;
        },
        else => return err,
    };
    defer out_slice.free(allocator);
    const out_items = out_slice.items(f32);
    const preview_len = @min(out_items.len, @as(usize, 16));
    log.info("output preview: {any}", .{out_items[0..preview_len]});
}
