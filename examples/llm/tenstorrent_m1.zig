//! M1 runner of the Tenstorrent backend (DESIGN 14.2).
//!
//! Runs one Llama decoder layer (llama_layer.zig) with the real weights of
//! layer 0 of a checkpoint: a prefill of 128 tokens, then decode steps of one
//! token each carrying the KV cache. Every leg gets the same inputs: the
//! embedding rows of a fixed token sequence, the hidden states layer 0 sees.
//!
//! Legs, the reference always first:
//!   cpu            CPU backend, F32, one device: the reference
//!   cpu_bf16       CPU backend, BF16 as examples/llm runs, one device
//!   cpu_bf16_tp2   CPU backend, BF16, tensor-parallel over two devices
//!   fake1, fake2   the plugin's device-free runtime, one and two devices
//!   cards1, cards2 the cards of this host, one and two: run under the card lock
//! The CPU BF16 legs show what BF16 and the two-way split cost on their own.
//!
//! Per step it prints, for every leg, max |y - ref| / max |ref| against the
//! reference for the worst token of the step (each token against its own
//! maximum), and the distance between each two-device leg and its one-device
//! leg in BF16 ulps; the final K and V caches are judged the same way, per
//! position. A leg whose program the backend refuses (UNIMPLEMENTED) prints
//! the refusal verbatim and is not run; the process still exits 0. Exit 1
//! means a leg failed (any other error) or ran and broke a limit.

const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const zml = @import("zml");

const llama_layer = @import("llama_layer.zig");
const common = @import("models/common.zig");
const llama = @import("models/llama.zig");
const pjrt_error_log = @import("pjrt_error_log.zig");

const TransformerLayer = llama.model.TransformerLayer;
const KvCache = llama.model.KvCache;

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = pjrt_error_log.logFn,
};

const log = std.log.scoped(.tenstorrent_m1);

/// DESIGN 14.2, M1: every step within this of the CPU F32 reference.
const reference_limit = 2e-2;
/// DESIGN 14.2, M1: "TP=2 and 1-card outputs agree to BF16 rounding", read as
/// max |a - b| within this many BF16 units in the last place of max |ref|.
/// The CPU backend's own TP=2 and one-device BF16 runs differ by one.
const agreement_ulps = 2;

/// Device memory of a device-free leg: the layer's BF16 weights (436 MB on one
/// device) and its activations. The plugin's default is 16 MiB.
const device_free_arena_bytes = 1 << 30;

const Args = struct {
    model: []const u8,
    legs: []const u8 = "cpu_bf16,cpu_bf16_tp2,fake1,fake2",
    program: enum { layer, standin } = .layer,
    steps: u32 = 16,
    seed: u64 = 1,
    dump: ?[]const u8 = null,

    pub const help =
        \\ Use tenstorrent_m1 --model=<dir> [options]
        \\
        \\ Runs layer 0 of a Llama checkpoint, prefill of 128 tokens then decode steps, on the CPU in F32
        \\ (the reference) and on the legs named, and prints per step the relative error of each leg.
        \\
        \\ Options:
        \\   --model=<dir>       Llama checkpoint directory (config.json and safetensors)
        \\   --legs=<list>       Comma-separated legs besides the reference: cpu_bf16, cpu_bf16_tp2,
        \\                       fake1, fake2, cards1, cards2 (default: cpu_bf16,cpu_bf16_tp2,fake1,fake2).
        \\                       cards1 and cards2 open the cards: run them under the card lock.
        \\   --program=<name>    layer (default), or standin: elementwise arithmetic on the same
        \\                       arguments, for what the backend can compile before the layer
        \\   --steps=<n>         Decode steps after the prefill (default 16, at most 128)
        \\   --seed=<n>          Seed of the token sequence (default 1)
        \\   --dump=<dir>        Writes the programs handed to PJRT under <dir>/<leg>/, as the corpus does
        \\
    ;
};

const Where = enum { cpu, device_free, cards };

const Leg = struct {
    name: []const u8,
    where: Where,
    devices: u32,
    dtype: zml.DataType,
    /// Cluster descriptor of a device-free leg, in platforms/tenstorrent/fixtures.
    fixture: []const u8 = "",
    what: []const u8,
};

const reference_leg: Leg = .{ .name = "cpu", .where = .cpu, .devices = 1, .dtype = .f32, .what = "CPU backend, F32, 1 device (reference)" };

const known_legs = [_]Leg{
    .{ .name = "cpu_bf16", .where = .cpu, .devices = 1, .dtype = .bf16, .what = "CPU backend, BF16, 1 device" },
    .{ .name = "cpu_bf16_tp2", .where = .cpu, .devices = 2, .dtype = .bf16, .what = "CPU backend, BF16, TP=2" },
    .{ .name = "fake1", .where = .device_free, .devices = 1, .dtype = .bf16, .fixture = "blackhole_P150.yaml", .what = "Tenstorrent device-free, 1 device" },
    .{ .name = "fake2", .where = .device_free, .devices = 2, .dtype = .bf16, .fixture = "box_2xp150_capture.yaml", .what = "Tenstorrent device-free, TP=2" },
    .{ .name = "cards1", .where = .cards, .devices = 1, .dtype = .bf16, .what = "Tenstorrent cards, 1 card" },
    .{ .name = "cards2", .where = .cards, .devices = 2, .dtype = .bf16, .what = "Tenstorrent cards, TP=2" },
};

/// Each two-device leg and the one-device leg it must agree with.
const pairs = [_][2][]const u8{
    .{ "cpu_bf16_tp2", "cpu_bf16" },
    .{ "fake2", "fake1" },
    .{ "cards2", "cards1" },
};

const phases = [_]common.Phase{ .prefill, .decode };

const Failure = struct {
    stage: []const u8,
    err: anyerror,
    /// The plugin's message, verbatim, when PJRT gave one.
    message: []const u8,
};

const Results = struct {
    /// The layer's output after each step: [0] the prefill, then one per decode step.
    hidden: [][]f32,
    /// The cache after the last step, in [layer, k, h, hd] order whatever the target's layout.
    k: []f32,
    v: []f32,
    timings: Timings,
};

const Timings = struct {
    open_ms: f64 = 0,
    compile_ms: f64 = 0,
    load_ms: f64 = 0,
    prefill_ms: f64 = 0,
    decode_ms: f64 = 0,
};

const Outcome = union(enum) {
    ran: Results,
    /// The backend refused the prefill program, the decode program or both
    /// (compile errors from PJRT), indexed like `phases`.
    refused: [phases.len]?Failure,
    /// The platform could not open, or a run failed.
    failed: Failure,
    not_requested,
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    config: llama.Config,
    registry: *zml.safetensors.TensorRegistry,
    steps: u32,
    /// One embedding row per position: the prefill's rows, then one per decode step.
    rows_bf16: []const zml.floats.BFloat16,
    rows_f32: []const f32,
    /// See `Args.dump`.
    dump: ?[]const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(io, build_working_directory, .{});
        defer working_dir.close(io);
        try std.process.setCurrentDir(io, working_dir);
    }
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if (args.steps > llama_layer.cache_positions - llama_layer.prefill_tokens) {
        log.err("--steps={d}: the cache holds {d} positions after the prefill", .{ args.steps, llama_layer.cache_positions - llama_layer.prefill_tokens });
        return error.InvalidArgument;
    }

    var requested: [known_legs.len]bool = @splat(false);
    var names = std.mem.splitScalar(u8, args.legs, ',');
    while (names.next()) |name| {
        if (name.len == 0) continue;
        const index = legIndex(name) orelse {
            log.err("unknown leg {s}: cpu_bf16, cpu_bf16_tp2, fake1, fake2, cards1, cards2", .{name});
            return error.InvalidArgument;
        };
        requested[index] = true;
    }

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    defer repo.close(io);
    const parsed_config = try common.parseConfig(llama.Config, allocator, io, repo);
    defer parsed_config.deinit();
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    const positions = llama_layer.prefill_tokens + args.steps;
    const ids = try tokenIds(allocator, parsed_config.value, args.seed, positions);
    defer allocator.free(ids);
    const rows_bf16 = try embeddingRows(allocator, io, &registry, ids, parsed_config.value.hidden_size);
    defer allocator.free(rows_bf16);
    const rows_f32 = try allocator.alloc(f32, rows_bf16.len);
    defer allocator.free(rows_f32);
    for (rows_bf16, rows_f32) |b, *f| f.* = b.toF32();

    const ctx: Context = .{
        .allocator = allocator,
        .io = io,
        .config = parsed_config.value,
        .registry = &registry,
        .steps = args.steps,
        .rows_bf16 = rows_bf16,
        .rows_f32 = rows_f32,
        .dump = args.dump,
    };

    // The plugin needs a newer libstdc++ than the host's, which the CPU plugin
    // maps: load it first, through a device-free client.
    for (known_legs, requested) |leg, wanted| {
        if (!wanted or leg.where == .cpu) continue;
        const platform = try openDeviceFree(&ctx, "blackhole_P150.yaml", null);
        platform.deinit(allocator, io);
        break;
    }

    var report: Report = .{ .ctx = &ctx, .program = @tagName(args.program), .ids = ids, .model = args.model };
    defer report.deinit();
    switch (args.program) {
        inline else => |program| {
            const func = comptime switch (program) {
                .layer => llama_layer.forward,
                .standin => llama_layer.standIn,
            };
            report.reference = try runLeg(&ctx, func, reference_leg);
            for (known_legs, requested, &report.legs) |leg, wanted, *outcome| {
                if (!wanted) continue;
                outcome.* = try runLeg(&ctx, func, leg);
            }
        },
    }

    var stdout_buf: [4096]u8 = undefined;
    // Streaming: a positional writer would write from offset 0 of a redirected
    // stdout, over what the plugin's logs wrote there before.
    var stdout = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
    const ok = try report.write(&stdout.interface);
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn legIndex(name: []const u8) ?usize {
    for (known_legs, 0..) |leg, i| {
        if (std.mem.eql(u8, leg.name, name)) return i;
    }
    return null;
}

/// The beginning-of-text token, then a fixed pseudo-random sequence.
fn tokenIds(allocator: std.mem.Allocator, config: llama.Config, seed: u64, count: usize) ![]u32 {
    const ids = try allocator.alloc(u32, count);
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    ids[0] = config.bos_token_id;
    // Below the first special token of Llama 3 (128000).
    for (ids[1..]) |*id| id.* = random.uintLessThan(u32, 128000);
    return ids;
}

/// The checkpoint's embedding rows of `ids`: the inputs of layer 0.
fn embeddingRows(allocator: std.mem.Allocator, io: std.Io, registry: *zml.safetensors.TensorRegistry, ids: []const u32, hidden_size: u32) ![]zml.floats.BFloat16 {
    const entry = registry.tensors.get("model.embed_tokens.weight") orelse return error.MissingEmbedding;
    if (entry.shape.dtype() != .bf16 or entry.shape.rank() != 2 or entry.shape.dim(1) != hidden_size) {
        log.err("expected a BF16 embedding of {d} columns, got {f}", .{ hidden_size, entry.shape });
        return error.UnexpectedEmbedding;
    }
    const file = try std.Io.Dir.cwd().openFile(io, entry.file_uri, .{});
    defer file.close(io);

    const d: usize = hidden_size;
    const rows = try allocator.alloc(zml.floats.BFloat16, ids.len * d);
    errdefer allocator.free(rows);
    for (ids, 0..) |id, i| {
        if (id >= entry.shape.dim(0)) return error.TokenOutOfRange;
        const row = std.mem.sliceAsBytes(rows[i * d ..][0..d]);
        const read = try file.readPositionalAll(io, row, entry.offset + @as(u64, id) * row.len);
        if (read != row.len) return error.EndOfStream;
    }
    return rows;
}

fn fixturePath(buf: []u8, file: []const u8) ![]const u8 {
    const r = try zml.bazel.runfiles(bazel_builtin.current_repository);
    var name_buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "zml/platforms/tenstorrent/fixtures/{s}", .{file});
    return try r.rlocation(name, buf) orelse error.FileNotFound;
}

fn openDeviceFree(ctx: *const Context, fixture: []const u8, dump_dir: ?[]const u8) !*zml.Platform {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    return zml.Platform.init(ctx.allocator, ctx.io, .tenstorrent, .{
        .tenstorrent = .{
            .runtime = .fake,
            .cluster_descriptor = try fixturePath(&path_buf, fixture),
            .arena_bytes = device_free_arena_bytes,
        },
        .pjrt_program_dump_dir = dump_dir,
    });
}

fn openPlatform(ctx: *const Context, leg: Leg) !*zml.Platform {
    const allocator = ctx.allocator;
    const io = ctx.io;
    // The platform keeps its own copy.
    const dump_dir = if (ctx.dump) |dir| try std.Io.Dir.path.join(allocator, &.{ dir, leg.name }) else null;
    defer if (dump_dir) |dir| allocator.free(dir);
    return switch (leg.where) {
        .cpu => zml.Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = leg.devices }, .pjrt_program_dump_dir = dump_dir }),
        .device_free => openDeviceFree(ctx, leg.fixture, dump_dir),
        .cards => cards: {
            // The first cards of the default view, neighbours in it, alone:
            // a view of `leg.devices` cards whatever else this host has.
            var chips_buf: [zml.Platform.MAX_NUM_DEVICES * 21]u8 = undefined;
            var chips: std.Io.Writer = .fixed(&chips_buf);
            {
                const all: *zml.Platform = try .init(allocator, io, .tenstorrent, .{});
                defer all.deinit(allocator, io);
                if (all.devices.len < leg.devices) {
                    log.err("{s}: this host has {d} cards", .{ leg.name, all.devices.len });
                    return error.NotEnoughCards;
                }
                for (all.devices[0..leg.devices], 0..) |device, i| {
                    const info: zml.platform.tenstorrent.DeviceInfo = try .init(all, device);
                    try chips.print("{s}{d}", .{ if (i == 0) "" else ",", info.board_id });
                }
            }
            break :cards zml.Platform.init(allocator, io, .tenstorrent, .{
                .tenstorrent = .{ .visible_chips = chips.buffered() },
                .pjrt_program_dump_dir = dump_dir,
            });
        },
    };
}

fn failure(ctx: *const Context, stage: []const u8, err: anyerror) !Failure {
    return .{ .stage = stage, .err = err, .message = try ctx.allocator.dupe(u8, pjrt_error_log.get()) };
}

fn msSince(io: std.Io, from: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(from.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_ms;
}

/// Runs the prefill and the decode steps of `func` on one leg. What fails on
/// the leg (open, compile, load, run, read back) is its outcome; only running
/// out of host memory returns.
fn runLeg(ctx: *const Context, comptime func: anytype, leg: Leg) !Outcome {
    const allocator = ctx.allocator;
    const io = ctx.io;
    log.info("{s}: {s}", .{ leg.name, leg.what });

    pjrt_error_log.reset();
    const start: std.Io.Timestamp = .now(io, .awake);
    const platform = openPlatform(ctx, leg) catch |err| return .{ .failed = try failure(ctx, "open the platform", err) };
    defer platform.deinit(allocator, io);
    var stage: []const u8 = "count the devices";
    return runOnPlatform(ctx, func, leg, platform, msSince(io, start), &stage) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => .{ .failed = try failure(ctx, stage, err) },
    };
}

/// `runLeg` once the platform is open; `stage` names the step under way.
fn runOnPlatform(ctx: *const Context, comptime func: anytype, leg: Leg, platform: *zml.Platform, open_ms: f64, stage: *[]const u8) !Outcome {
    const allocator = ctx.allocator;
    const io = ctx.io;
    const Exe = zml.FnExe(func);
    var timings: Timings = .{ .open_ms = open_ms };

    if (platform.devices.len != leg.devices) {
        log.err("{s}: the platform serves {d} devices, the leg needs {d}", .{ leg.name, platform.devices.len, leg.devices });
        return error.UnexpectedDeviceCount;
    }

    stage.* = "describe the layer";
    const meshes: common.Meshes = try .init(platform);
    var store: zml.io.TensorStore = .fromRegistry(allocator, ctx.registry, &meshes.all());
    defer store.deinit();
    const layer: TransformerLayer = try .init(store.view().withPrefix("model").withPrefix("layers").withLayer(0), ctx.config);
    const backend: zml.attention.Backend = .auto(platform);

    var inputs: [phases.len]llama_layer.Input = undefined;
    var compiled: [phases.len]?Exe = @splat(null);
    defer for (&compiled) |*exe| if (exe.*) |*e| e.deinit();
    // Both programs are compiled even when one is refused: each refusal is reported.
    var refusals: [phases.len]?Failure = @splat(null);
    errdefer for (refusals) |refusal| if (refusal) |f| allocator.free(f.message);
    var start: std.Io.Timestamp = .now(io, .awake);
    for (phases, &inputs, &compiled, &refusals) |phase, *input, *exe, *refusal| {
        input.* = .init(layer, ctx.config, phase, leg.dtype, meshes, platform.target, backend);
        stage.* = if (phase == .prefill) "compile the prefill" else "compile the decode step";
        pjrt_error_log.reset();
        exe.* = Exe.compile(allocator, io, platform, .{
            .meshes = &meshes.all(),
            .program_name = llama_layer.programName(phase),
        }, .{input.*}) catch |err| b: {
            if (!isRefusal(leg.where, err)) return err;
            refusal.* = try failure(ctx, stage.*, err);
            break :b null;
        };
    }
    timings.compile_ms = msSince(io, start);
    for (refusals) |refusal| if (refusal != null) return .{ .refused = refusals };
    const exes: [phases.len]*const Exe = .{ &compiled[0].?, &compiled[1].? };

    stage.* = "load the weights";
    pjrt_error_log.reset();
    start = .now(io, .awake);
    var weights = try zml.mem.bufferize(allocator, TransformerLayer, &layer);
    {
        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();
        try loader.load(io, TransformerLayer, &layer, &weights, &store, .{});
        try loader.await(io);
    }
    defer TransformerLayer.unloadBuffers(&weights);
    timings.load_ms = msSince(io, start);

    // The cache starts at zero: positions the causal mask hides must not hold NaN bits.
    stage.* = "upload the cache and the positions";
    pjrt_error_log.reset();
    const kv_shape = inputs[0].kv_cache.k.shape();
    const zeros = try allocator.alloc(u8, kv_shape.byteSize());
    defer allocator.free(zeros);
    @memset(zeros, 0);
    var kv: KvCache.Buffer = .{ .k = try .fromBytes(io, platform, kv_shape, zeros), .v = undefined };
    {
        errdefer kv.k.deinit();
        kv.v = try .fromBytes(io, platform, kv_shape, zeros);
    }
    defer KvCache.deinitBuffer(&kv);

    // As examples/llm's Session, every position lives on the device before the first step.
    const step_count = 1 + ctx.steps;
    const token_index = try allocator.alloc(zml.Buffer, step_count);
    defer allocator.free(token_index);
    var positions_made: usize = 0;
    defer for (token_index[0..positions_made]) |*b| b.deinit();
    for (token_index, 0..) |*b, step| {
        const position: u32 = if (step == 0) 0 else llama_layer.prefill_tokens + @as(u32, @intCast(step - 1));
        b.* = try .scalar(io, platform, position, .u32);
        positions_made += 1;
    }

    var metadata = try inputs[0].attention_metadata.initBuffer(io, platform);
    defer zml.attention.Metadata.deinitBuffer(&metadata);

    var runners: [phases.len]Exe.Runner(.{.layer}) = undefined;
    var runners_made: usize = 0;
    defer for (runners[0..runners_made]) |*r| r.deinit(allocator);
    for (&runners, exes) |*r, exe| {
        r.* = try .init(exe, allocator, .{ .layer = weights });
        runners_made += 1;
    }

    const hidden = try allocator.alloc([]f32, step_count);
    var hidden_made: usize = 0;
    errdefer {
        for (hidden[0..hidden_made]) |h| allocator.free(h);
        allocator.free(hidden);
    }
    const d: usize = ctx.config.hidden_size;
    for (0..step_count) |step| {
        const phase_index: usize = if (step == 0) 0 else 1;
        const first_row: usize = if (step == 0) 0 else llama_layer.prefill_tokens + step - 1;
        const rows: usize = @intCast(llama_layer.tokens(phases[phase_index]));
        const bytes = switch (leg.dtype) {
            .f32 => std.mem.sliceAsBytes(ctx.rows_f32[first_row * d ..][0 .. rows * d]),
            .bf16 => std.mem.sliceAsBytes(ctx.rows_bf16[first_row * d ..][0 .. rows * d]),
            else => unreachable,
        };
        stage.* = if (step == 0) "run the prefill" else "run a decode step";
        pjrt_error_log.reset();
        // Donated to the output, which replaces it.
        var hidden_buffer: zml.Buffer = try .fromBytes(io, platform, inputs[phase_index].hidden.shape(), bytes);
        defer hidden_buffer.deinit();

        start = .now(io, .awake);
        try runners[phase_index].tryRun(io, .{
            .inputs = .{
                .hidden = hidden_buffer,
                .token_index = token_index[step],
                .kv_cache = kv,
                .attention_metadata = metadata,
            },
            .outputs = .{ .hidden = &hidden_buffer, .kv_cache = &kv },
        }, .{ .wait = true });
        const took = msSince(io, start);
        if (step == 0) timings.prefill_ms = took else timings.decode_ms += took / @as(f64, @floatFromInt(ctx.steps));

        stage.* = "read the results back";
        hidden[step] = try readF32(allocator, io, hidden_buffer);
        hidden_made += 1;
    }

    stage.* = "read the results back";
    const k = try readCanonicalKv(allocator, io, kv.k);
    errdefer allocator.free(k);
    const v = try readCanonicalKv(allocator, io, kv.v);
    log.info("{s}: ran (open {d:.0} ms, compile {d:.0} ms, load {d:.0} ms, prefill {d:.1} ms, decode {d:.2} ms per step)", .{
        leg.name, timings.open_ms, timings.compile_ms, timings.load_ms, timings.prefill_ms, timings.decode_ms,
    });
    return .{ .ran = .{ .hidden = hidden, .k = k, .v = v, .timings = timings } };
}

/// A refusal is the backend answering UNIMPLEMENTED for what it does not
/// lower yet; any other compile error, and any on the CPU, fails the leg.
fn isRefusal(where: Where, err: anyerror) bool {
    return where != .cpu and err == error.Unimplemented;
}

test "only the Tenstorrent backend's UNIMPLEMENTED is a refusal" {
    try std.testing.expect(isRefusal(.cards, error.Unimplemented));
    try std.testing.expect(isRefusal(.device_free, error.Unimplemented));
    try std.testing.expect(!isRefusal(.cpu, error.Unimplemented));
    try std.testing.expect(!isRefusal(.cards, error.InvalidArgument));
    try std.testing.expect(!isRefusal(.device_free, error.OutOfMemory));
}

fn readF32(allocator: std.mem.Allocator, io: std.Io, buffer: zml.Buffer) ![]f32 {
    const slice = try buffer.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const out = try allocator.alloc(f32, slice.shape.count());
    switch (slice.shape.dtype()) {
        .f32 => @memcpy(out, slice.constItems(f32)),
        .bf16 => for (slice.constItems(zml.floats.BFloat16), out) |b, *f| {
            f.* = b.toF32();
        },
        else => unreachable,
    }
    return out;
}

/// The cache in [layer, k, h, hd] order, from either layout (Tenstorrent stores heads first).
fn readCanonicalKv(allocator: std.mem.Allocator, io: std.Io, buffer: zml.Buffer) ![]f32 {
    const values = try readF32(allocator, io, buffer);
    defer allocator.free(values);
    const shape = buffer.shape();
    const strides = shape.computeElementStrides();
    const stride = [4]usize{
        @intCast(strides.get(shape.axis(.layer))),
        @intCast(strides.get(shape.axis(.k))),
        @intCast(strides.get(shape.axis(.h))),
        @intCast(strides.get(shape.axis(.hd))),
    };
    const out = try allocator.alloc(f32, values.len);
    var i: usize = 0;
    for (0..@intCast(shape.dim(.layer))) |l| {
        for (0..@intCast(shape.dim(.k))) |k| {
            for (0..@intCast(shape.dim(.h))) |h| {
                for (0..@intCast(shape.dim(.hd))) |hd| {
                    out[i] = values[l * stride[0] + k * stride[1] + h * stride[2] + hd * stride[3]];
                    i += 1;
                }
            }
        }
    }
    return out;
}

/// How far `got` is from `base`, one row of `width` values at a time (one
/// token's hidden state, or one cache position), each row against the largest
/// magnitude of the same row of `ref`: max |got - base| / max |ref|.
/// Elementwise relative error is meaningless where the reference is near
/// zero, which BF16 outputs always have. A whole-tensor maximum is too lenient
/// for the prefill: after layer 0 the beginning-of-text token's state is about
/// twenty times larger than the other tokens', which it would measure against
/// its own maximum.
const Distance = struct {
    /// Worst row of max |got - base| / max |ref|.
    rel: f64 = 0,
    /// Worst row of max |got - base| in BF16 units in the last place of max |ref|.
    ulps: f64 = 0,
    /// The row of `rel`.
    row: usize = 0,

    fn of(got: []const f32, base: []const f32, ref: []const f32, width: usize) Distance {
        std.debug.assert(got.len == ref.len and base.len == ref.len and ref.len % width == 0);
        var worst: Distance = .{};
        for (0..ref.len / width) |row| {
            var max_diff: f64 = 0;
            var max_ref: f64 = 0;
            var nan = false;
            for (got[row * width ..][0..width], base[row * width ..][0..width], ref[row * width ..][0..width]) |g, b, r| {
                if (std.math.isNan(g) or std.math.isNan(b)) nan = true;
                max_diff = @max(max_diff, @abs(@as(f64, g) - b));
                max_ref = @max(max_ref, @abs(@as(f64, r)));
            }
            const rel, const ulps = if (nan)
                .{ std.math.nan(f64), std.math.nan(f64) }
            else if (max_ref == 0)
                // A row the reference leaves at zero (a cache position never written) must stay zero.
                if (max_diff == 0) .{ 0, 0 } else .{ std.math.inf(f64), std.math.inf(f64) }
            else
                // BF16 keeps 8 significant bits: 2^(e - 7) for max_ref in [2^e, 2^(e+1)).
                .{ max_diff / max_ref, max_diff / @exp2(@floor(@log2(max_ref)) - 7) };
            if (worse(rel, worst.rel)) {
                worst.rel = rel;
                worst.row = row;
            }
            if (worse(ulps, worst.ulps)) worst.ulps = ulps;
        }
        return worst;
    }

    /// NaN is worse than any number, and stays.
    fn worse(a: f64, b: f64) bool {
        if (std.math.isNan(b)) return false;
        return std.math.isNan(a) or a > b;
    }

    fn within(self: Distance, limit: f64) bool {
        return self.rel <= limit;
    }

    fn agrees(self: Distance) bool {
        return self.ulps <= agreement_ulps;
    }

    fn maxWith(self: Distance, other: Distance) Distance {
        return .{
            .rel = if (worse(other.rel, self.rel)) other.rel else self.rel,
            .row = if (worse(other.rel, self.rel)) other.row else self.row,
            .ulps = if (worse(other.ulps, self.ulps)) other.ulps else self.ulps,
        };
    }
};

test {
    // The test binary links Bazel's runfiles initialiser, whose Zig half lives there.
    _ = zml.bazel;
}

test "Distance measures each token against its own row" {
    // Row 0 is large (as the beginning-of-text token), row 1 small and 10% off.
    const ref = [_]f32{ 32, -32, 0.5, -0.25 };
    const got = [_]f32{ 32, -32, 0.55, -0.25 };
    const d: Distance = .of(&got, &ref, &ref, 2);
    try std.testing.expectApproxEqRel(0.1, d.rel, 1e-6);
    try std.testing.expectEqual(1, d.row);
    // 0.05 in units of 2^-8, the BF16 ulp of 0.5.
    try std.testing.expectApproxEqRel(12.8, d.ulps, 1e-5);
    try std.testing.expect(!d.within(2e-2));
    // A whole-tensor maximum would have passed it: 0.05 / 32.
    try std.testing.expect(Distance.of(&got, &ref, &ref, 4).within(2e-2));
}

test "Distance: zero rows of the reference must stay zero, NaN fails" {
    const ref = [_]f32{ 1, 1, 0, 0 };
    try std.testing.expectEqual(0, Distance.of(&ref, &ref, &ref, 2).rel);
    const written = [_]f32{ 1, 1, 0, 1e-3 };
    try std.testing.expect(std.math.isInf(Distance.of(&written, &ref, &ref, 2).rel));
    const nan = [_]f32{ std.math.nan(f32), 1, 0, 0 };
    const d: Distance = .of(&nan, &ref, &ref, 2);
    try std.testing.expect(std.math.isNan(d.rel) and !d.within(2e-2) and !d.agrees());
    try std.testing.expect(std.math.isNan(d.maxWith(.{}).rel) and std.math.isNan(Distance.maxWith(.{}, d).rel));
}

const Report = struct {
    ctx: *const Context,
    program: []const u8,
    model: []const u8,
    ids: []const u32,
    reference: Outcome = .not_requested,
    legs: [known_legs.len]Outcome = @splat(.not_requested),

    fn deinit(self: *Report) void {
        freeOutcome(self.ctx.allocator, self.reference);
        for (self.legs) |outcome| freeOutcome(self.ctx.allocator, outcome);
    }

    fn freeOutcome(allocator: std.mem.Allocator, outcome: Outcome) void {
        switch (outcome) {
            .ran => |r| {
                for (r.hidden) |h| allocator.free(h);
                allocator.free(r.hidden);
                allocator.free(r.k);
                allocator.free(r.v);
            },
            .refused => |refusals| for (refusals) |refusal| if (refusal) |f| allocator.free(f.message),
            .failed => |f| allocator.free(f.message),
            .not_requested => {},
        }
    }

    fn results(self: *const Report, name: []const u8) ?Results {
        const outcome = self.legs[legIndex(name).?];
        return if (outcome == .ran) outcome.ran else null;
    }

    /// Writes the report; false when a leg that ran broke a limit or a leg failed.
    fn write(self: *const Report, w: *std.Io.Writer) !bool {
        var ok = true;
        const steps = self.ctx.steps;
        try w.print(
            \\
            \\== M1 runner: Llama decoder layer 0, program {s}, prefill of {d} tokens then {d} decode steps
            \\model {s}, tokens {d}, {d}, {d}, ... (seeded), KV cache of {d} positions zeroed first
            \\
        , .{ self.program, llama_layer.prefill_tokens, steps, self.model, self.ids[0], self.ids[1], self.ids[2], llama_layer.cache_positions });

        try w.print("\nlegs\n", .{});
        try writeOutcome(w, reference_leg, self.reference);
        for (known_legs, self.legs) |leg, outcome| {
            if (outcome == .not_requested) continue;
            try writeOutcome(w, leg, outcome);
            if (outcome == .failed) ok = false;
        }

        const ref = switch (self.reference) {
            .ran => |r| r,
            else => {
                try w.print("\nno reference: nothing to compare\n", .{});
                return false;
            },
        };

        const widths: Widths = .of(self.ctx.config);
        try w.print("\nmax |y - ref| / max |ref| against the reference, worst token of each step (limit {e})\n", .{reference_limit});
        try w.print("{s:<8} {s:>5} {s:>10}", .{ "step", "pos", "max|ref|" });
        for (known_legs, self.legs) |leg, outcome| if (outcome != .not_requested) try w.print(" {s:>12}", .{leg.name});
        try w.writeByte('\n');
        for (0..rowCount(steps)) |row| {
            try writeRowLabel(w, row, steps, maxAbs(output(ref, row, steps)));
            for (self.legs) |outcome| switch (outcome) {
                .not_requested => {},
                .ran => |r| try writeCell(w, compare(r, ref, ref, row, steps, widths), reference_limit),
                .refused => try w.print(" {s:>12}", .{"refused"}),
                .failed => try w.print(" {s:>12}", .{"failed"}),
            };
            try w.writeByte('\n');
        }

        try w.print("\nTP=2 against one device: max |a - b| in BF16 ulps of max |ref|, worst token of each step (limit {d})\n", .{agreement_ulps});
        try w.print("{s:<8} {s:>5}", .{ "step", "pos" });
        for (pairs) |pair| try w.print(" {s:>14}", .{pair[0]});
        try w.writeByte('\n');
        for (0..rowCount(steps)) |row| {
            try writeRowLabel(w, row, steps, null);
            for (pairs) |pair| {
                const a = self.results(pair[0]) orelse {
                    try w.print(" {s:>14}", .{"-"});
                    continue;
                };
                const b = self.results(pair[1]) orelse {
                    try w.print(" {s:>14}", .{"-"});
                    continue;
                };
                const dist = compare(a, b, ref, row, steps, widths);
                var cell_buf: [32]u8 = undefined;
                try w.print(" {s:>14}", .{try std.fmt.bufPrint(&cell_buf, "{d:.2}{s}", .{ dist.ulps, if (dist.agrees()) "" else "!" })});
            }
            try w.writeByte('\n');
        }

        try w.print("\nverdict (DESIGN 14.2, M1; every output: the prefill's and each decode step's hidden states, per token, and the final K and V caches, per position)\n", .{});
        for (known_legs, self.legs) |leg, outcome| {
            switch (outcome) {
                .not_requested => {},
                .refused => try w.print("  {s:<13} refused by the backend: no number\n", .{leg.name}),
                .failed => try w.print("  {s:<13} failed\n", .{leg.name}),
                .ran => |r| {
                    const worst, const at = worstRow(r, ref, ref, steps, widths);
                    const pass = worst.within(reference_limit);
                    if (!pass) ok = false;
                    try w.print("  {s:<13} {s} 2e-2 of the reference (worst {e:.2}: ", .{ leg.name, if (pass) "within" else "NOT within", worst.rel });
                    try writeWhere(w, at, steps, worst.row);
                    try w.print(")\n", .{});
                },
            }
        }
        for (pairs) |pair| {
            const a = self.results(pair[0]) orelse continue;
            const b = self.results(pair[1]) orelse continue;
            var worst: Distance = .{};
            for (0..rowCount(steps)) |row| worst = worst.maxWith(compare(a, b, ref, row, steps, widths));
            const pass = worst.agrees();
            if (!pass) ok = false;
            try w.print("  {s} vs {s}: {s} to {d} BF16 ulps (worst {d:.2})\n", .{ pair[0], pair[1], if (pass) "agree" else "DO NOT agree", agreement_ulps, worst.ulps });
        }
        if (std.mem.eql(u8, self.program, "layer")) {
            const cards_met = self.results("cards1") != null and self.results("cards2") != null and ok;
            try w.print("  M1 numeric criteria on the cards: {s}\n", .{if (cards_met) "MET" else "not met"});
        } else {
            try w.print("  M1 numeric criteria: not judged on the stand-in program\n", .{});
        }
        return ok;
    }

    /// Values per compared row: one token's hidden state, one cache position (every head).
    const Widths = struct {
        hidden: usize,
        kv: usize,

        fn of(config: llama.Config) Widths {
            const head_dim = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads);
            return .{ .hidden = config.hidden_size, .kv = @as(usize, config.num_key_value_heads) * head_dim };
        }
    };

    /// The compared outputs, in report order: the prefill, each decode step, then the K and V caches.
    fn rowCount(steps: u32) usize {
        return 1 + steps + 2;
    }

    fn output(r: Results, row: usize, steps: u32) []const f32 {
        if (row <= steps) return r.hidden[row];
        return if (row == steps + 1) r.k else r.v;
    }

    fn compare(got: Results, base: Results, ref: Results, row: usize, steps: u32, widths: Widths) Distance {
        const width = if (row <= steps) widths.hidden else widths.kv;
        return .of(output(got, row, steps), output(base, row, steps), output(ref, row, steps), width);
    }

    /// The worst output of a leg and its index in `rowCount` order.
    fn worstRow(got: Results, base: Results, ref: Results, steps: u32, widths: Widths) struct { Distance, usize } {
        var worst: Distance = .{};
        var at: usize = 0;
        for (0..rowCount(steps)) |row| {
            const dist = compare(got, base, ref, row, steps, widths);
            if (Distance.worse(dist.rel, worst.rel)) {
                worst = dist;
                at = row;
            }
        }
        return .{ worst, at };
    }

    fn writeRowLabel(w: *std.Io.Writer, row: usize, steps: u32, max_ref: ?f64) !void {
        if (row == 0) {
            var label_buf: [16]u8 = undefined;
            try w.print("{s:<8} {s:>5}", .{ "prefill", try std.fmt.bufPrint(&label_buf, "0-{d}", .{llama_layer.prefill_tokens - 1}) });
        } else if (row <= steps) {
            try w.print("{s:<8} {d:>5}", .{ "decode", llama_layer.prefill_tokens + row - 1 });
        } else {
            try w.print("{s:<8} {s:>5}", .{ if (row == steps + 1) "k cache" else "v cache", "end" });
        }
        if (max_ref) |m| try w.print(" {e:>10.3}", .{m});
    }

    fn writeWhere(w: *std.Io.Writer, row: usize, steps: u32, inner: usize) !void {
        if (row == 0) {
            try w.print("prefill, token {d}", .{inner});
        } else if (row <= steps) {
            try w.print("decode step {d}, position {d}", .{ row, llama_layer.prefill_tokens + row - 1 });
        } else {
            try w.print("{s} cache, position {d}", .{ if (row == steps + 1) "k" else "v", inner });
        }
    }

    fn maxAbs(values: []const f32) f64 {
        var m: f64 = 0;
        for (values) |v| m = @max(m, @abs(@as(f64, v)));
        return m;
    }

    fn writeCell(w: *std.Io.Writer, dist: Distance, limit: f64) !void {
        var cell_buf: [24]u8 = undefined;
        try w.print(" {s:>12}", .{try std.fmt.bufPrint(&cell_buf, "{e:.2}{s}", .{ dist.rel, if (dist.within(limit)) "" else "!" })});
    }

    fn writeOutcome(w: *std.Io.Writer, leg: Leg, outcome: Outcome) !void {
        try w.print("  {s:<13} {s:<38} ", .{ leg.name, leg.what });
        switch (outcome) {
            .not_requested => try w.print("not run\n", .{}),
            .ran => |r| try w.print("ran: open {d:.0} ms, compile {d:.0} ms, load {d:.0} ms, prefill {d:.1} ms, decode {d:.2} ms/step\n", .{
                r.timings.open_ms, r.timings.compile_ms, r.timings.load_ms, r.timings.prefill_ms, r.timings.decode_ms,
            }),
            .refused => |refusals| {
                try w.print("REFUSED\n", .{});
                for (refusals) |refusal| if (refusal) |f| try writeFailure(w, f);
            },
            .failed => |f| {
                try w.print("FAILED\n", .{});
                try writeFailure(w, f);
            },
        }
    }

    fn writeFailure(w: *std.Io.Writer, f: Failure) !void {
        try w.print("      at \"{s}\": {s}: {s}\n", .{ f.stage, @errorName(f.err), if (f.message.len > 0) f.message else "(no PJRT message)" });
    }
};
