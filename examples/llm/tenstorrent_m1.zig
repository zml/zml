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
//! reference, and the same distance between each two-device leg and its
//! one-device leg. A leg whose program the backend refuses prints the
//! refusal verbatim and is not run; the process still exits 0. Exit 1 means a
//! leg that ran broke a limit or failed.

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
    /// The backend refused a program (compile error from PJRT).
    refused: Failure,
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
    };

    // The plugin needs a newer libstdc++ than the host's, which the CPU plugin
    // maps: load it first, through a device-free client.
    for (known_legs, requested) |leg, wanted| {
        if (!wanted or leg.where == .cpu) continue;
        const platform = try openDeviceFree(&ctx, "blackhole_P150.yaml");
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
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
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

fn openDeviceFree(ctx: *const Context, fixture: []const u8) !*zml.Platform {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    return zml.Platform.init(ctx.allocator, ctx.io, .tenstorrent, .{ .tenstorrent = .{
        .runtime = .fake,
        .cluster_descriptor = try fixturePath(&path_buf, fixture),
        .arena_bytes = device_free_arena_bytes,
    } });
}

fn openPlatform(ctx: *const Context, leg: Leg) !*zml.Platform {
    const allocator = ctx.allocator;
    const io = ctx.io;
    return switch (leg.where) {
        .cpu => zml.Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = leg.devices } }),
        .device_free => openDeviceFree(ctx, leg.fixture),
        .cards => if (leg.devices == 1) one: {
            // The first card of the default view, alone.
            const board = all: {
                const platform: *zml.Platform = try .init(allocator, io, .tenstorrent, .{});
                defer platform.deinit(allocator, io);
                const info: zml.platform.tenstorrent.DeviceInfo = try .init(platform, platform.devices[0]);
                break :all info.board_id;
            };
            var chips_buf: [32]u8 = undefined;
            break :one zml.Platform.init(allocator, io, .tenstorrent, .{
                .tenstorrent = .{ .visible_chips = try std.fmt.bufPrint(&chips_buf, "{d}", .{board}) },
            });
        } else zml.Platform.init(allocator, io, .tenstorrent, .{}),
    };
}

fn failure(ctx: *const Context, stage: []const u8, err: anyerror) !Failure {
    return .{ .stage = stage, .err = err, .message = try ctx.allocator.dupe(u8, pjrt_error_log.get()) };
}

fn msSince(io: std.Io, from: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(from.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_ms;
}

/// Runs the prefill and the decode steps of `func` on one leg. Errors of the
/// leg itself (open, compile, run) are its outcome; only host errors return.
fn runLeg(ctx: *const Context, comptime func: anytype, leg: Leg) !Outcome {
    const allocator = ctx.allocator;
    const io = ctx.io;
    const Exe = zml.FnExe(func);
    log.info("{s}: {s}", .{ leg.name, leg.what });
    var timings: Timings = .{};

    pjrt_error_log.reset();
    var start: std.Io.Timestamp = .now(io, .awake);
    const platform = openPlatform(ctx, leg) catch |err| return .{ .failed = try failure(ctx, "open the platform", err) };
    defer platform.deinit(allocator, io);
    timings.open_ms = msSince(io, start);
    if (platform.devices.len != leg.devices) {
        log.err("{s}: the platform serves {d} devices, the leg needs {d}", .{ leg.name, platform.devices.len, leg.devices });
        return .{ .failed = try failure(ctx, "count the devices", error.UnexpectedDeviceCount) };
    }

    const meshes: common.Meshes = try .init(platform);
    var store: zml.io.TensorStore = .fromRegistry(allocator, ctx.registry, &meshes.all());
    defer store.deinit();
    const layer: TransformerLayer = try .init(store.view().withPrefix("model").withPrefix("layers").withLayer(0), ctx.config);
    const backend: zml.attention.Backend = .auto(platform);

    const phases = [_]common.Phase{ .prefill, .decode };
    var inputs: [phases.len]llama_layer.Input = undefined;
    var exes: [phases.len]Exe = undefined;
    var compiled: usize = 0;
    defer for (exes[0..compiled]) |*exe| exe.deinit();
    start = .now(io, .awake);
    for (phases, &inputs, &exes) |phase, *input, *exe| {
        input.* = .init(layer, ctx.config, phase, leg.dtype, meshes, platform.target, backend);
        pjrt_error_log.reset();
        exe.* = Exe.compile(allocator, io, platform, .{
            .meshes = &meshes.all(),
            .program_name = llama_layer.programName(phase),
        }, .{input.*}) catch |err| return .{ .refused = try failure(ctx, if (phase == .prefill) "compile the prefill" else "compile the decode step", err) };
        compiled += 1;
    }
    timings.compile_ms = msSince(io, start);

    start = .now(io, .awake);
    var weights = try zml.mem.bufferize(allocator, TransformerLayer, &layer);
    {
        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();
        pjrt_error_log.reset();
        loader.load(io, TransformerLayer, &layer, &weights, &store, .{}) catch |err| return .{ .failed = try failure(ctx, "load the weights", err) };
        loader.await(io) catch |err| return .{ .failed = try failure(ctx, "load the weights", err) };
    }
    defer TransformerLayer.unloadBuffers(&weights);
    timings.load_ms = msSince(io, start);

    // The cache starts at zero: positions the causal mask hides must not hold NaN bits.
    const kv_shape = inputs[0].kv_cache.k.shape();
    const zeros = try allocator.alloc(u8, kv_shape.byteSize());
    defer allocator.free(zeros);
    @memset(zeros, 0);
    var kv: KvCache.Buffer = .{
        .k = try .fromBytes(io, platform, kv_shape, zeros),
        .v = try .fromBytes(io, platform, kv_shape, zeros),
    };
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
    for (&runners, &exes) |*r, *exe| {
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
        // Donated to the output, which replaces it.
        var hidden_buffer: zml.Buffer = try .fromBytes(io, platform, inputs[phase_index].hidden.shape(), bytes);
        defer hidden_buffer.deinit();

        pjrt_error_log.reset();
        start = .now(io, .awake);
        runners[phase_index].tryRun(io, .{
            .inputs = .{
                .hidden = hidden_buffer,
                .token_index = token_index[step],
                .kv_cache = kv,
                .attention_metadata = metadata,
            },
            .outputs = .{ .hidden = &hidden_buffer, .kv_cache = &kv },
        }, .{ .wait = true }) catch |err| return .{ .failed = try failure(ctx, if (step == 0) "run the prefill" else "run a decode step", err) };
        const took = msSince(io, start);
        if (step == 0) timings.prefill_ms = took else timings.decode_ms += took / @as(f64, @floatFromInt(ctx.steps));

        hidden[step] = try readF32(allocator, io, hidden_buffer);
        hidden_made += 1;
    }

    const k = try readCanonicalKv(allocator, io, kv.k);
    errdefer allocator.free(k);
    const v = try readCanonicalKv(allocator, io, kv.v);
    log.info("{s}: ran (open {d:.0} ms, compile {d:.0} ms, load {d:.0} ms, prefill {d:.1} ms, decode {d:.2} ms per step)", .{
        leg.name, timings.open_ms, timings.compile_ms, timings.load_ms, timings.prefill_ms, timings.decode_ms,
    });
    return .{ .ran = .{ .hidden = hidden, .k = k, .v = v, .timings = timings } };
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

/// max |got - ref| over max |ref|: elementwise relative error is meaningless
/// where the reference is near zero, which BF16 outputs always have.
const Distance = struct {
    rel: f64,
    max_diff: f64,
    max_ref: f64,

    fn of(got: []const f32, base: []const f32, ref: []const f32) Distance {
        var max_diff: f64 = 0;
        var max_ref: f64 = 0;
        var nan = false;
        for (got, base, ref) |g, b, r| {
            if (std.math.isNan(g) or std.math.isNan(b)) nan = true;
            max_diff = @max(max_diff, @abs(@as(f64, g) - b));
            max_ref = @max(max_ref, @abs(@as(f64, r)));
        }
        if (nan) max_diff = std.math.nan(f64);
        return .{ .rel = max_diff / max_ref, .max_diff = max_diff, .max_ref = max_ref };
    }

    fn within(self: Distance, limit: f64) bool {
        return self.rel <= limit;
    }

    /// max_diff in BF16 units in the last place of max_ref: 2^(e - 7) for max_ref in [2^e, 2^(e+1)).
    fn ulps(self: Distance) f64 {
        return self.max_diff / @exp2(@floor(@log2(self.max_ref)) - 7);
    }
};

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
            .refused, .failed => |f| allocator.free(f.message),
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

        try w.print("\nmax |y - ref| / max |ref| against the reference, per step (limit {e})\n", .{reference_limit});
        try w.print("{s:<8} {s:>5} {s:>10}", .{ "step", "pos", "max|ref|" });
        for (known_legs, self.legs) |leg, outcome| if (outcome != .not_requested) try w.print(" {s:>12}", .{leg.name});
        try w.writeByte('\n');
        for (0..1 + steps) |step| {
            const position: usize = if (step == 0) 0 else llama_layer.prefill_tokens + step - 1;
            try w.print("{s:<8} {d:>5} {e:>10.3}", .{ if (step == 0) "prefill" else "decode", position, Distance.of(ref.hidden[step], ref.hidden[step], ref.hidden[step]).max_ref });
            for (self.legs) |outcome| switch (outcome) {
                .not_requested => {},
                .ran => |r| try writeCell(w, Distance.of(r.hidden[step], ref.hidden[step], ref.hidden[step]), reference_limit),
                .refused => try w.print(" {s:>12}", .{"refused"}),
                .failed => try w.print(" {s:>12}", .{"failed"}),
            };
            try w.writeByte('\n');
        }
        for ([_][]const u8{ "k cache", "v cache" }, 0..) |label, which| {
            try w.print("{s:<8} {s:>5} {e:>10.3}", .{ label, "end", Distance.of(kvOf(ref, which), kvOf(ref, which), kvOf(ref, which)).max_ref });
            for (self.legs) |outcome| switch (outcome) {
                .not_requested => {},
                .ran => |r| try writeCell(w, Distance.of(kvOf(r, which), kvOf(ref, which), kvOf(ref, which)), reference_limit),
                else => try w.print(" {s:>12}", .{"-"}),
            };
            try w.writeByte('\n');
        }

        try w.print("\nTP=2 against one device: max |a - b| in BF16 ulps of max |ref|, per step (limit {d})\n", .{agreement_ulps});
        try w.print("{s:<8} {s:>5}", .{ "step", "pos" });
        for (pairs) |pair| try w.print(" {s:>14}", .{pair[0]});
        try w.writeByte('\n');
        for (0..1 + steps) |step| {
            const position: usize = if (step == 0) 0 else llama_layer.prefill_tokens + step - 1;
            try w.print("{s:<8} {d:>5}", .{ if (step == 0) "prefill" else "decode", position });
            for (pairs) |pair| {
                const a = self.results(pair[0]) orelse {
                    try w.print(" {s:>14}", .{"-"});
                    continue;
                };
                const b = self.results(pair[1]) orelse {
                    try w.print(" {s:>14}", .{"-"});
                    continue;
                };
                const dist = Distance.of(a.hidden[step], b.hidden[step], ref.hidden[step]);
                var cell_buf: [32]u8 = undefined;
                try w.print(" {s:>14}", .{try std.fmt.bufPrint(&cell_buf, "{d:.2}{s}", .{ dist.ulps(), if (dist.ulps() <= agreement_ulps) "" else "!" })});
            }
            try w.writeByte('\n');
        }

        try w.print("\nverdict (DESIGN 14.2, M1)\n", .{});
        for (known_legs, self.legs) |leg, outcome| {
            switch (outcome) {
                .not_requested => {},
                .refused => try w.print("  {s:<13} refused by the backend: no number\n", .{leg.name}),
                .failed => try w.print("  {s:<13} failed\n", .{leg.name}),
                .ran => |r| {
                    var worst: Distance = .{ .rel = 0, .max_diff = 0, .max_ref = 0 };
                    var worst_step: usize = 0;
                    for (r.hidden, ref.hidden, 0..) |h, rh, step| {
                        const dist = Distance.of(h, rh, rh);
                        if (std.math.isNan(worst.rel)) break;
                        if (std.math.isNan(dist.rel) or dist.rel > worst.rel) {
                            worst = dist;
                            worst_step = step;
                        }
                    }
                    const pass = worst.within(reference_limit);
                    if (!pass) ok = false;
                    try w.print("  {s:<13} {s} 2e-2 of the reference at every step (worst {e:.2}, step {d})\n", .{ leg.name, if (pass) "within" else "NOT within", worst.rel, worst_step });
                },
            }
        }
        for (pairs) |pair| {
            const a = self.results(pair[0]) orelse continue;
            const b = self.results(pair[1]) orelse continue;
            var worst: f64 = 0;
            for (a.hidden, b.hidden, ref.hidden) |ah, bh, rh| {
                const ulps = Distance.of(ah, bh, rh).ulps();
                if (std.math.isNan(ulps) or ulps > worst) worst = ulps;
                if (std.math.isNan(worst)) break;
            }
            const pass = worst <= agreement_ulps;
            if (!pass) ok = false;
            try w.print("  {s} vs {s}: {s} to {d} BF16 ulps at every step (worst {d:.2})\n", .{ pair[0], pair[1], if (pass) "agree" else "DO NOT agree", agreement_ulps, worst });
        }
        if (std.mem.eql(u8, self.program, "layer")) {
            const cards_met = self.results("cards1") != null and self.results("cards2") != null and ok;
            try w.print("  M1 numeric criteria on the cards: {s}\n", .{if (cards_met) "MET" else "not met"});
        } else {
            try w.print("  M1 numeric criteria: not judged on the stand-in program\n", .{});
        }
        return ok;
    }

    fn kvOf(r: Results, which: usize) []const f32 {
        return if (which == 0) r.k else r.v;
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
            .refused, .failed => |f| try w.print("{s} at \"{s}\": {s}: {s}\n", .{
                if (outcome == .refused) "REFUSED" else "FAILED",
                f.stage,
                @errorName(f.err),
                if (f.message.len > 0) f.message else "(no PJRT message)",
            }),
        }
    }
};
