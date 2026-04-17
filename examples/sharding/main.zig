const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.sharding);
pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/module", .level = .debug },
    },
};

const CliMesh = enum {
    auto,
    mock,
};

const CliArgs = struct {
    partitioner: ?zml.sharding.Partitioning.Partitioner = null,
    mesh: CliMesh = .auto,
};

const DemoModel = struct {
    w: zml.Tensor, // {feature, hidden}
    b: zml.Tensor, // {hidden}

    pub fn init(w: zml.Tensor, b: zml.Tensor) DemoModel {
        return .{ .w = w, .b = b };
    }

    pub fn forward(self: DemoModel, input: zml.Tensor) zml.Tensor {
        const x = input.convert(.f32);

        var y = x.dot(self.w, .feature);
        y = y.add(self.b.broad(y.shape()));
        y = y.withPartitioning(.{ .batch = .data, .hidden = .model });
        y.print("dense_out");

        const gate = y.scale(0.01).sigmoid();
        return zml.ops.manualComputation(
            .{ y, gate },
            y.shape(),
            {},
            (struct {
                fn body(_: void, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, output_sharded: zml.Shape) zml.Tensor {
                    const local_result = sharded_inputs[0].relu().mul(sharded_inputs[1]);
                    return local_result.reshape(output_sharded);
                }
            }).body,
        );
    }
};

fn usage() void {
    std.debug.print(
        \\Usage: sharding [--partitioner=shardy|gspmd] [--mesh=auto|mock]
        \\
        \\Simple sharding demo:
        \\  - explicit partitioning on data/model axes
        \\  - shard-local tensor print
        \\  - manualComputation with multiple inputs and one output
        \\
        \\Mesh modes:
        \\  - auto: physical_mesh=auto, inferred cpu device_count=2
        \\  - mock: physical_mesh=mock topology, inferred cpu device_count=9
        \\
    , .{});
}

fn parseArgs(init: std.process.Init) !CliArgs {
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
) !zml.sharding.PhysicalMesh {
    if (devices.len < 8) return error.NotEnoughDevicesForMockMesh;
    const topology: zml.sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
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
    return zml.sharding.PhysicalMesh.fromTree(allocator, target, topology);
}

fn createSequenceBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.sharding.Sharding,
    start: f32,
) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    for (slice.items(f32), 0..) |*e, i| {
        e.* = start + @as(f32, @floatFromInt(i));
    }

    return zml.Buffer.fromSlice(io, platform, slice, sharding);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args = try parseArgs(init);

    const create_options: zml.platform.CreateOptions = switch (args.mesh) {
        .auto => .{},
        .mock => .{
            .cpu = .{ .device_count = 8 },
            .physical_mesh = .{ .custom = buildMockMesh },
        },
    };

    var platform: *zml.Platform = try .auto(allocator, io, create_options);
    defer platform.deinit(allocator, io);

    log.info("\n{f}", .{platform.fmtVerbose()});

    var profiler_options: zml.Platform.ProfilerOptions = .defaults;
    profiler_options.repository_path = "/tmp/xprof";
    profiler_options.session_id = "profiling";

    var profiler = try platform.profiler(allocator, io, profiler_options);
    defer profiler.deinit();

    try profiler.start();
    defer {
        if ((profiler.stop() catch unreachable)) |profile| {
            log.info("Profile dumped: {s} and {s}", .{ profile.protobuf_path, profile.perfetto_path });
        }
    }

    if (args.partitioner) |partitioner| {
        log.info("Partitioner: {s}", .{@tagName(partitioner)});
    } else {
        args.partitioner = .fromTarget(platform.target);
        log.info("Partitioner: {s} (default)", .{@tagName(args.partitioner.?)});
    }
    log.info("{f}", .{platform.physical_mesh});

    const physical_mesh = platform.physical_mesh;
    const mesh: zml.sharding.LogicalMesh = try .init("demo_mesh", .{
        .data = .low_bandwidth,
        .model = .high_bandwidth,
    });
    const strategy: zml.sharding.Strategy = try .suggest(mesh, physical_mesh);
    const sharding: zml.sharding.Sharding = try .initFromStrategy(platform, mesh, strategy);

    log.info("{f}", .{mesh});
    log.info("{f}", .{sharding});

    const input_shape = zml.Shape.init(.{ .batch = 16, .feature = 32 }, .f32)
        .withPartitioning(.{ .batch = .data, .feature = .replicated });
    const w_shape = zml.Shape.init(.{ .feature = 32, .hidden = 64 }, .f32)
        .withPartitioning(.{ .feature = .replicated, .hidden = .model });
    const b_shape = zml.Shape.init(.{ .hidden = 64 }, .f32)
        .withPartitioning(.{ .hidden = .model });

    const input: zml.Tensor = .fromShape(input_shape);
    const w: zml.Tensor = .fromShape(w_shape);
    const b: zml.Tensor = .fromShape(b_shape);
    const model: DemoModel = .init(w, b);

    var exe = try platform.compile(
        allocator,
        io,
        model,
        .forward,
        .{input},
        .{
            .partitioner = args.partitioner,
            .shardings = &.{sharding},
        },
    );
    defer exe.deinit();

    var w_buf = try createSequenceBuffer(allocator, io, platform, w_shape, sharding, 0.0);
    defer w_buf.deinit();
    var b_buf = try createSequenceBuffer(allocator, io, platform, b_shape, sharding, 100.0);
    defer b_buf.deinit();
    var input_buf = try createSequenceBuffer(allocator, io, platform, input_shape, sharding, 1000.0);
    defer input_buf.deinit();

    log.info("input placement: {f}", .{input_buf.placement()});
    log.info("weight placement: {f}", .{w_buf.placement()});
    log.info("bias placement: {f}", .{b_buf.placement()});

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ w_buf, b_buf, input_buf });

    exe.call(exe_args, &exe_results);
    var out = exe_results.get(zml.Buffer);
    defer out.deinit();
    _ = try out.await(io);

    const out_slice = try out.toSliceAlloc(allocator, io);
    defer out_slice.free(allocator);
    const out_items = out_slice.items(f32);
    const preview_len = @min(out_items.len, @as(usize, 16));
    log.info("output preview: {any}", .{out_items[0..preview_len]});
}
