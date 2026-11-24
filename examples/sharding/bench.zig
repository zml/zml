//! Multi-host benchmark on the `dp_mp` mesh (`.data` spans processes, `.model` stays
//! within one):
//!
//! 1. Collectives: all-reduce and all-gather across `.data` for a sweep of buffer
//!    sizes. Each result is checked, then timed, and reported as NCCL-tests style
//!    algorithm / bus bandwidth.
//! 2. Model: a tensor-parallel MLP layer (all-reduce over `.model`) followed by a
//!    data-parallel weight-gradient all-reduce over `.data`, run for `--layers`.
//!
//! Compilation time is kept independent of the workload size: each collective is a
//! one-op program, and the model compiles a single layer that the host calls once
//! per layer instead of unrolling every layer into one program. Set
//! `ZML_AUTOTUNE_CACHE_DIR` to reuse XLA GEMM autotuning results across runs.

const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.bench);

pub const std_options: std.Options = .{ .log_level = .info };

const Mode = enum { all, collectives, model };

const CliArgs = struct {
    pub const help =
        \\Usage: bench [--mode=all|collectives|model] [--sizes-mib=1,16,256] [--iters=20] [--warmup=3]
        \\             [--layers=8] [--batch=256] [--d-model=4096] [--d-ff=16384]
        \\             [--target=cuda] [--coordinator=IP:PORT --process-index=N --process-count=N
        \\              [--local-devices=0,1,...] [--namespace=NAME]]
        \\
        \\Collectives run across the .data mesh axis (one shard per process); sizes are
        \\per-process buffer sizes. The model layer splits --batch over .data and
        \\--d-ff over .model, and all-reduces a d_ff x d_model bf16 gradient over .data.
        \\
    ;

    mode: Mode = .all,
    sizes_mib: []const u8 = "1,16,256",
    iters: u32 = 20,
    warmup: u32 = 3,
    layers: u32 = 8,
    batch: u32 = 256,
    d_model: u32 = 4096,
    d_ff: u32 = 16384,
    target: zml.Target = .cuda,
    coordinator: ?[]const u8 = null,
    process_index: usize = 0,
    process_count: usize = 1,
    local_devices: []const u8 = "0",
    namespace: []const u8 = "zml-sharding-bench",
};

const Collective = enum {
    all_reduce,
    all_gather,

    fn program(comptime self: Collective) fn (zml.Tensor) zml.Tensor {
        return switch (self) {
            .all_reduce => allReduceData,
            .all_gather => allGatherData,
        };
    }

    /// Ratio between bus bandwidth and algorithm bandwidth for `n` participants,
    /// as defined by nccl-tests.
    fn busFactor(self: Collective, n: u32) f64 {
        const nf: f64 = @floatFromInt(n);
        return switch (self) {
            .all_reduce => 2 * (nf - 1) / nf,
            .all_gather => (nf - 1) / nf,
        };
    }
};

/// Sums the `.shard` slices of every process: input `{shard = n, x}` split on
/// `.data`, output `{shard = 1, x}` replicated.
fn allReduceData(x: zml.Tensor) zml.Tensor {
    const output: zml.Shape = .init(.{ .shard = 1, .x = x.dim(.x) }, x.dtype());
    return zml.ops.manualComputation(
        struct {
            fn body(local: zml.Tensor, local_output: zml.Shape) zml.Tensor {
                const mesh = zml.Compiler.current().mesh(.dp_mp);
                return zml.ops.allReduceOnAxis(local, zml.Tensor.add, mesh, .data).reshape(local_output);
            }
        }.body,
        x,
        output,
        zml.Compiler.current().mesh(.dp_mp),
        .{ .in_specs = .parse(.{ .shard = .data }), .out_specs = .replicated },
    );
}

/// Concatenates the `.shard` slices of every process: input `{shard = n, x}` split
/// on `.data`, output the same shape replicated.
fn allGatherData(x: zml.Tensor) zml.Tensor {
    const output: zml.Shape = .init(.{ .shard = x.dim(.shard), .x = x.dim(.x) }, x.dtype());
    return zml.ops.manualComputation(
        struct {
            fn body(local: zml.Tensor, local_output: zml.Shape) zml.Tensor {
                const mesh = zml.Compiler.current().mesh(.dp_mp);
                return zml.ops.allGatherOnAxis(local, mesh, .data, .shard).reshape(local_output);
            }
        }.body,
        x,
        output,
        zml.Compiler.current().mesh(.dp_mp),
        .{ .in_specs = .parse(.{ .shard = .data }), .out_specs = .replicated },
    );
}

/// One MLP layer: tensor parallel over `.model`, data parallel over `.data`.
const Layer = struct {
    w_up: zml.Tensor, // {d, f}, f split on .model
    w_down: zml.Tensor, // {f, d}, f split on .model

    const Inputs = struct { x: zml.Tensor, w_up: zml.Tensor, w_down: zml.Tensor };
    const Shapes = struct { x: zml.Shape, grad: zml.Shape };
    pub const Outputs = struct { x: zml.Tensor, grad: zml.Tensor };
    pub const Buffers = struct { x: zml.Buffer, grad: zml.Buffer };

    /// Returns the next activations `{b, d}` (split on `.data`) and a stand-in
    /// gradient for `w_down` `{f, d}` (split on `.model`, summed over `.data`).
    pub fn forward(self: Layer, x: zml.Tensor) Outputs {
        const outputs: Shapes = .{
            .x = .init(.{ .b = x.dim(.b), .d = x.dim(.d) }, x.dtype()),
            .grad = .init(.{ .f = self.w_down.dim(.f), .d = self.w_down.dim(.d) }, x.dtype()),
        };
        return zml.ops.manualComputation(
            body,
            .{ .x = x, .w_up = self.w_up, .w_down = self.w_down },
            outputs,
            zml.Compiler.current().mesh(.dp_mp),
            .{
                .in_specs = .{
                    .x = .parse(.{ .b = .data }),
                    .w_up = .parse(.{ .f = .model }),
                    .w_down = .parse(.{ .f = .model }),
                },
                .out_specs = .{
                    .x = .parse(.{ .b = .data }),
                    .grad = .parse(.{ .f = .model }),
                },
            },
        );
    }

    fn body(in: Inputs, local_outputs: Shapes) Outputs {
        const mesh = zml.Compiler.current().mesh(.dp_mp);
        const h = in.x.dot(in.w_up, .d).gelu(); // {b, f_local}
        // Each .model shard holds a partial sum over its slice of d_ff.
        const y = zml.ops.allReduceOnAxis(h.dot(in.w_down, .f), zml.Tensor.add, mesh, .model); // {b, d}
        // Stand-in for dL/dw_down: reduce over the local batch, then over every data shard.
        const grad = zml.ops.allReduceOnAxis(h.dot(y, .b), zml.Tensor.add, mesh, .data); // {f_local, d}
        return .{
            .x = in.x.add(y).reshape(local_outputs.x),
            .grad = grad.reshape(local_outputs.grad),
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    var local_devices: std.ArrayList(i64) = .empty;
    defer local_devices.deinit(allocator);
    var create_options: zml.platform.CreateOptions = .{};
    if (args.coordinator) |coordinator| {
        var ids = std.mem.splitScalar(u8, args.local_devices, ',');
        while (ids.next()) |id| try local_devices.append(allocator, try std.fmt.parseInt(i64, id, 10));
        create_options.distributed = .{
            .coordinator_address = try .parseLiteral(coordinator),
            .process_index = args.process_index,
            .process_count = args.process_count,
            .namespace = args.namespace,
            .local_device_ids = local_devices.items,
        };
    }

    var platform: *zml.Platform = try .init(allocator, io, args.target, create_options);
    defer platform.deinit(allocator, io);

    const mesh: *const zml.Mesh = try platform.registerMesh(
        "dp_mp",
        .mesh(.{ .data = .low_bandwidth, .model = .high_bandwidth }),
    );
    const reporter = platform.processIndex() == 0;
    if (reporter) log.info("{d} processes, {d} global devices, mesh data={d} model={d}", .{
        platform.processCount(),
        platform.globalDevices().len,
        mesh.numPartitionsForLogicalAxis(.data),
        mesh.numPartitionsForLogicalAxis(.model),
    });

    if (args.mode != .model) try runCollectives(allocator, io, platform, mesh, args, reporter);
    if (args.mode != .collectives) try runModel(allocator, io, platform, mesh, args, reporter);

    try platform.barrier("bench-done");
}

fn runCollectives(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mesh: *const zml.Mesh,
    args: CliArgs,
    reporter: bool,
) !void {
    const shards = mesh.numPartitionsForLogicalAxis(.data);
    if (reporter) log.info("collectives over .data ({d} shards), {d} warmup + {d} timed iterations", .{ shards, args.warmup, args.iters });

    var sizes = std.mem.splitScalar(u8, args.sizes_mib, ',');
    while (sizes.next()) |size_text| {
        const size_mib = try std.fmt.parseInt(u64, size_text, 10);
        const bytes_per_shard = size_mib * 1024 * 1024;
        const input_shape = zml.Shape.init(.{ .shard = shards, .x = bytes_per_shard / @sizeOf(f32) }, .f32)
            .withPartitioning(mesh, .{ .shard = .data });

        var input = try createShardBuffer(allocator, io, platform, input_shape);
        defer input.deinit();

        inline for (.{ Collective.all_reduce, Collective.all_gather }) |collective| {
            const compile_start: std.Io.Timestamp = .now(io, .awake);
            var exe = try platform.compileFn(allocator, io, comptime collective.program(), .{zml.Tensor.fromShape(input_shape)}, .{ .meshes = &.{mesh} });
            defer exe.deinit();
            const compile_time = compile_start.untilNow(io, .awake);

            var exe_args = try exe.args(allocator);
            defer exe_args.deinit(allocator);
            var exe_results = try exe.results(allocator);
            defer exe_results.deinit(allocator);
            exe_args.set(.{input});

            try checkCollective(allocator, io, &exe, exe_args, &exe_results, collective, shards);

            const seconds = try timeCalls(io, platform, &exe, exe_args, &exe_results, args.warmup, args.iters);
            // nccl-tests convention: all-reduce moves the per-process buffer,
            // all-gather the full gathered buffer.
            const moved_bytes: f64 = @floatFromInt(switch (collective) {
                .all_reduce => bytes_per_shard,
                .all_gather => bytes_per_shard * shards,
            });
            const alg_gbps = moved_bytes * 8 / seconds / 1e9;
            if (reporter) log.info("{s:<10} {d:>6} MiB/shard  compile {d:>7.1} ms  time {d:>9.3} ms  algbw {d:>7.1} Gb/s  busbw {d:>7.1} Gb/s", .{
                @tagName(collective),
                size_mib,
                toMs(compile_time),
                seconds * 1e3,
                alg_gbps,
                alg_gbps * collective.busFactor(shards),
            });
        }
    }
}

/// Fills shard `s` of a `{shard, x}` f32 buffer with `s + 1`.
fn createShardBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    const per_shard: usize = @intCast(shape.dim(.x));
    for (slice.items(f32), 0..) |*e, i| e.* = @floatFromInt(i / per_shard + 1);
    return zml.Buffer.fromSlice(io, platform, slice);
}

/// Runs the collective once and checks every element of the replicated result.
fn checkCollective(
    allocator: std.mem.Allocator,
    io: std.Io,
    exe: *const zml.Exe,
    exe_args: zml.Exe.Arguments,
    exe_results: *zml.Exe.Results,
    collective: Collective,
    shards: u32,
) !void {
    exe.call(io, exe_args, exe_results, .{ .wait = true });
    var output = exe_results.get(zml.Buffer);
    defer output.deinit();

    const host = try output.toSliceAlloc(allocator, io);
    defer host.free(allocator);
    const values = host.items(f32);
    const per_shard = values.len / switch (collective) {
        .all_reduce => 1,
        .all_gather => shards,
    };
    for (values, 0..) |value, i| {
        const expected: f32 = switch (collective) {
            .all_reduce => @floatFromInt(shards * (shards + 1) / 2),
            .all_gather => @floatFromInt(i / per_shard + 1),
        };
        if (value != expected) {
            log.err("{s}: element {d} is {d}, expected {d}", .{ @tagName(collective), i, value, expected });
            return error.WrongCollectiveResult;
        }
    }
}

/// Returns the mean wall time of one call, after `warmup` untimed calls.
/// All processes start the timed loop together.
fn timeCalls(
    io: std.Io,
    platform: *zml.Platform,
    exe: *const zml.Exe,
    exe_args: zml.Exe.Arguments,
    exe_results: *zml.Exe.Results,
    warmup: u32,
    iters: u32,
) !f64 {
    for (0..warmup) |_| {
        exe.call(io, exe_args, exe_results, .{ .wait = true });
        var output = exe_results.get(zml.Buffer);
        output.deinit();
    }
    try platform.barrier("bench-timed-loop");

    const start: std.Io.Timestamp = .now(io, .awake);
    for (0..iters) |_| {
        exe.call(io, exe_args, exe_results, .{ .wait = true });
        var output = exe_results.get(zml.Buffer);
        output.deinit();
    }
    const elapsed_ns: f64 = @floatFromInt(start.untilNow(io, .awake).toNanoseconds());
    return elapsed_ns / 1e9 / @as(f64, @floatFromInt(iters));
}

fn runModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mesh: *const zml.Mesh,
    args: CliArgs,
    reporter: bool,
) !void {
    const data_shards = mesh.numPartitionsForLogicalAxis(.data);
    const model_shards = mesh.numPartitionsForLogicalAxis(.model);

    const x_shape = zml.Shape.init(.{ .b = args.batch, .d = args.d_model }, .bf16)
        .withPartitioning(mesh, .{ .b = .data });
    const w_up_shape = zml.Shape.init(.{ .d = args.d_model, .f = args.d_ff }, .bf16)
        .withPartitioning(mesh, .{ .f = .model });
    const w_down_shape = zml.Shape.init(.{ .f = args.d_ff, .d = args.d_model }, .bf16)
        .withPartitioning(mesh, .{ .f = .model });

    const layer: Layer = .{ .w_up = .fromShape(w_up_shape), .w_down = .fromShape(w_down_shape) };
    const compile_start: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compile(allocator, io, layer, .forward, .{zml.Tensor.fromShape(x_shape)}, .{ .meshes = &.{mesh} });
    defer exe.deinit();
    if (reporter) log.info("model: compiled one layer in {d:.1} ms (reused for all {d} layers)", .{
        toMs(compile_start.untilNow(io, .awake)),
        args.layers,
    });

    // All layers share one set of weights: compute and communication are the same
    // as with distinct weights, for a fraction of the memory and load time.
    var w_up = try createPatternBuffer(allocator, io, platform, w_up_shape, 1e-3);
    defer w_up.deinit();
    var w_down = try createPatternBuffer(allocator, io, platform, w_down_shape, 1e-3);
    defer w_down.deinit();
    var x = try createPatternBuffer(allocator, io, platform, x_shape, 1e-1);
    defer x.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    const step_ns = try timeSteps(io, platform, &exe, &exe_args, &exe_results, .{ .w_up = w_up, .w_down = w_down }, &x, args);

    const local_batch: f64 = @floatFromInt(args.batch / data_shards);
    const local_ff: f64 = @floatFromInt(args.d_ff / model_shards);
    const d_model: f64 = @floatFromInt(args.d_model);
    const layers: f64 = @floatFromInt(args.layers);
    const step_s = step_ns / 1e9;
    // Three GEMMs per layer: up projection, down projection, weight gradient.
    const flops = 3 * 2 * local_batch * d_model * local_ff * layers;
    const grad_bytes = local_ff * d_model * 2 * layers;
    if (reporter) log.info("model: {d} layers, batch {d} (local {d}), d_model {d}, d_ff {d} (local {d}): step {d:.3} ms, {d:.1} TFLOP/s per device, {d:.0} MiB gradient all-reduced over .data per step ({d:.1} Gb/s if the step were all communication)", .{
        args.layers,
        args.batch,
        args.batch / data_shards,
        args.d_model,
        args.d_ff,
        args.d_ff / model_shards,
        step_s * 1e3,
        flops / step_s / 1e12,
        grad_bytes / (1024 * 1024),
        grad_bytes * 8 / step_s / 1e9,
    });
}

/// Fills a bf16 buffer with a small repeating pattern centred on zero.
fn createPatternBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, scale: f32) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    for (slice.items(zml.floats.BFloat16), 0..) |*e, i| {
        const centred: f32 = @as(f32, @floatFromInt(i % 17)) - 8;
        e.* = .fromF32(centred * scale);
    }
    return zml.Buffer.fromSlice(io, platform, slice);
}

/// Returns the mean wall time of one step (`args.layers` layer calls) in nanoseconds.
/// Each step starts from `x` and leaves `x` unchanged.
fn timeSteps(
    io: std.Io,
    platform: *zml.Platform,
    exe: *const zml.Exe,
    exe_args: *zml.Exe.Arguments,
    exe_results: *zml.Exe.Results,
    weights: struct { w_up: zml.Buffer, w_down: zml.Buffer },
    x: *const zml.Buffer,
    args: CliArgs,
) !f64 {
    var elapsed_ns: f64 = 0;
    for (0..args.warmup + args.iters) |iter| {
        if (iter == args.warmup) try platform.barrier("bench-model-loop");
        const start: std.Io.Timestamp = .now(io, .awake);

        var activations: ?zml.Buffer = null;
        defer if (activations) |*a| a.deinit();
        for (0..args.layers) |_| {
            exe_args.set(.{ weights.w_up, weights.w_down, activations orelse x.* });
            exe.call(io, exe_args.*, exe_results, .{ .wait = true });
            var outputs = exe_results.get(Layer.Buffers);
            outputs.grad.deinit();
            if (activations) |*a| a.deinit();
            activations = outputs.x;
        }

        if (iter >= args.warmup) elapsed_ns += @floatFromInt(start.untilNow(io, .awake).toNanoseconds());
    }
    return elapsed_ns / @as(f64, @floatFromInt(args.iters));
}

fn toMs(duration: std.Io.Duration) f64 {
    return @as(f64, @floatFromInt(duration.toNanoseconds())) / 1e6;
}
