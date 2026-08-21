const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.triton_attention_autotune);
const triton_attention = zml.attention.triton;
const TuningConfig = triton_attention.TuningConfig;
const Parameters = triton_attention.paged.Parameters;
const AttentionOptions = zml.attention.paged_attention.AttentionOptions;

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = captureAutomaticResolverLog,
};

const automatic_selection_log_format =
    "unified attention selected candidate {d}: {any} ({t}; median {f}, MAD {f})";

const AutomaticResolverObservations = struct {
    tuned: std.atomic.Value(usize) = .init(0),
    cached: std.atomic.Value(usize) = .init(0),
    positive_tuned_timing: std.atomic.Value(bool) = .init(false),

    fn reset(self: *@This()) void {
        self.tuned.store(0, .seq_cst);
        self.cached.store(0, .seq_cst);
        self.positive_tuned_timing.store(false, .seq_cst);
    }
};

var automatic_resolver_observations: AutomaticResolverObservations = .{};

fn captureAutomaticResolverLog(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    if (comptime scope == .@"zml/attention/triton" and
        std.mem.eql(u8, format, automatic_selection_log_format))
    {
        const source = @tagName(args[2]);
        if (std.mem.eql(u8, source, "tuned")) {
            _ = automatic_resolver_observations.tuned.fetchAdd(1, .seq_cst);
            if (args[3].nanoseconds > 0) {
                automatic_resolver_observations.positive_tuned_timing.store(true, .seq_cst);
            }
        } else if (std.mem.eql(u8, source, "cache")) {
            _ = automatic_resolver_observations.cached.fetchAdd(1, .seq_cst);
        }
    }
    std.log.defaultLog(level, scope, format, args);
}

const Mode = enum {
    prefill,
    decode,
};

const Args = struct {
    mode: Mode = .decode,
    batch_size: usize = 1,
    query_tokens: ?usize = null,
    sequence_length: usize = 2048,
    confirmation_repetitions: usize = 20,
    automatic_platform_autotune: bool = false,

    pub const help =
        \\Usage: triton_attention_autotune [options]
        \\
        \\Autotune Triton unified paged attention with Llama-3.1-8B-like
        \\defaults (bf16, 32 query heads, 8 KV heads, head size 128,
        \\page size 16, and sequence length 2048).
        \\
        \\Options:
        \\  --mode=<prefill|decode>              Workload phase (default: decode)
        \\  --batch-size=<n>                     Number of sequences (default: 1)
        \\  --query-tokens=<n>                   Query tokens per sequence
        \\                                         (default: sequence length for
        \\                                         prefill, 1 for decode)
        \\  --sequence-length=<n>                KV sequence length (default: 2048)
        \\  --confirmation-repetitions=<n>       Runs in the final A/B check
        \\                                         (default: 20)
        \\  --automatic-platform-autotune        Compile the automatic attention
        \\                                         path twice and verify that it
        \\                                         tunes once, then uses the cache
        \\
    ;
};

const num_query_heads: usize = 32;
const num_kv_heads: usize = 8;
const head_dim: usize = 128;
const page_size: usize = 16;

const prefill_candidates = [_]TuningConfig{
    .automatic,
    .{ .two_d = .{ .block_m = 64, .tile_size = 16, .num_warps = 4, .num_stages = 1 } },
    .{ .two_d = .{ .block_m = 64, .tile_size = 32, .num_warps = 4, .num_stages = 1 } },
    .{ .two_d = .{ .block_m = 64, .tile_size = 64, .num_warps = 4, .num_stages = 1 } },
    .{ .two_d = .{ .block_m = 128, .tile_size = 32, .num_warps = 4, .num_stages = 1 } },
    .{ .two_d = .{ .block_m = 128, .tile_size = 64, .num_warps = 2, .num_stages = 1 } },
    .{ .two_d = .{ .block_m = 128, .tile_size = 64, .num_warps = 4, .num_stages = 2 } },
};

const decode_candidates = [_]TuningConfig{
    .automatic,
    .{ .three_d = .{
        .block_m = 16,
        .tile_size = 16,
        .num_segments_per_seq = 8,
        .attention_num_warps = 2,
        .attention_num_stages = 1,
        .reduce_num_warps = 1,
        .reduce_num_stages = 1,
    } },
    .{ .three_d = .{
        .block_m = 16,
        .tile_size = 16,
        .num_segments_per_seq = 32,
        .attention_num_warps = 2,
        .attention_num_stages = 1,
        .reduce_num_warps = 1,
        .reduce_num_stages = 1,
    } },
    .{ .three_d = .{
        .block_m = 16,
        .tile_size = 16,
        .num_segments_per_seq = 16,
        .attention_num_warps = 4,
        .attention_num_stages = 1,
        .reduce_num_warps = 1,
        .reduce_num_stages = 1,
    } },
    .{ .three_d = .{
        .block_m = 16,
        .tile_size = 16,
        .num_segments_per_seq = 16,
        .attention_num_warps = 2,
        .attention_num_stages = 1,
        .reduce_num_warps = 2,
        .reduce_num_stages = 1,
    } },
    .{ .two_d = .{ .block_m = 16, .tile_size = 16, .num_warps = 2, .num_stages = 2 } },
    .{ .two_d = .{ .block_m = 32, .tile_size = 16, .num_warps = 4, .num_stages = 2 } },
};

const max_tuned_tile_size: usize = blk: {
    var maximum = page_size;
    for (prefill_candidates ++ decode_candidates) |candidate| switch (candidate) {
        .automatic => {},
        .two_d => |config| maximum = @max(maximum, config.tile_size),
        .three_d => |config| maximum = @max(maximum, config.tile_size),
    };
    break :blk maximum;
};

const Program = struct {
    exe: zml.Exe,
    arguments: zml.Exe.Arguments,
    results: zml.Exe.Results,
    timer: ?zml.ExecutionTimer = null,
    validated: bool = false,

    fn deinit(self: *Program, allocator: std.mem.Allocator) void {
        self.results.deinit(allocator);
        self.arguments.deinit(allocator);
        self.exe.deinit();
    }
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shardings: []const zml.Sharding,
    parameters: Parameters,
    parameter_buffers: zml.Bufferized(Parameters),
    q_tensor: zml.Tensor,
    k_cache_tensor: zml.Tensor,
    v_cache_tensor: zml.Tensor,
    q_buffer: zml.Buffer,
    k_cache_buffer: zml.Buffer,
    v_cache_buffer: zml.Buffer,
    attention_options: AttentionOptions,
    reference: ?*const zml.Slice = null,
    compilation_index: usize = 0,

    fn compileCandidate(self: *Context, config: TuningConfig) !Program {
        return self.compileProgram(config, .device);
    }

    fn compileProgram(
        self: *Context,
        config: TuningConfig,
        execution_timing: zml.CompilationOptions.ExecutionTiming,
    ) !Program {
        config.validate(num_query_heads / num_kv_heads) catch return error.InvalidTuningConfig;

        var parameters = self.parameters;
        parameters.options_.tuning = config;

        const program_name = try std.fmt.allocPrint(
            self.allocator,
            "triton_unified_attention_autotune_{d}",
            .{self.compilation_index},
        );
        defer self.allocator.free(program_name);
        self.compilation_index += 1;

        var exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            triton_attention.paged.pagedAttention,
            .{
                parameters,
                self.q_tensor,
                self.k_cache_tensor,
                self.v_cache_tensor,
                self.attention_options,
            },
            .{
                .program_name = program_name,
                .shardings = self.shardings,
                .execution_timing = execution_timing,
            },
        );
        errdefer exe.deinit();

        var arguments = try exe.args(self.allocator);
        errdefer arguments.deinit(self.allocator);
        arguments.set(.{
            self.parameter_buffers,
            self.q_buffer,
            self.k_cache_buffer,
            self.v_cache_buffer,
        });

        const results = try exe.results(self.allocator);
        return .{
            .exe = exe,
            .arguments = arguments,
            .results = results,
        };
    }

    fn measure(self: *Context, program: *Program, repetitions: usize) !std.Io.Duration {
        const timer = if (program.timer) |*timer| timer else timer: {
            program.timer = try zml.ExecutionTimer.attach(&program.exe);
            break :timer &program.timer.?;
        };

        var measured_ns: i96 = 0;
        var remaining = repetitions;
        if (!program.validated) {
            try timer.reset();
            program.exe.tryCallOpts(
                self.io,
                program.arguments,
                &program.results,
                .{ .wait = true, .allow_input_donation = false },
            ) catch |err| {
                program.results.releaseBuffers();
                return err;
            };
            const duration = timer.read() catch |err| {
                program.results.releaseBuffers();
                return err;
            };
            var output = program.results.get(zml.Buffer);
            defer output.deinit();
            const reference = self.reference orelse return error.MissingReferenceOutput;
            try zml.testing.expectClose(self.io, reference.*, output, .{
                .absolute_tolerance = 1e-2,
                .relative_tolerance = 1e-2,
                .epsilon_relative = 1e-6,
            });
            program.validated = true;
            measured_ns = duration.nanoseconds;
            remaining -= 1;
        }

        if (remaining != 0) {
            const duration = try timer.measureCall(
                self.io,
                program.arguments,
                &program.results,
                remaining,
            );
            measured_ns = std.math.add(i96, measured_ns, duration.nanoseconds) catch
                return error.DurationOverflow;
        }
        return .fromNanoseconds(measured_ns);
    }

    fn deinitProgram(self: *Context, program: *Program) void {
        program.deinit(self.allocator);
    }

    fn makeReference(self: *Context, config: TuningConfig) !zml.Slice {
        var program = try self.compileProgram(config, .none);
        defer program.deinit(self.allocator);
        try program.exe.tryCallOpts(
            self.io,
            program.arguments,
            &program.results,
            .{ .wait = true, .allow_input_donation = false },
        );
        var output = program.results.get(zml.Buffer);
        defer output.deinit();
        return output.toSliceAlloc(self.allocator, self.io);
    }
};

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(10_000);
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    if (args.batch_size == 0 or args.sequence_length == 0 or args.confirmation_repetitions == 0) {
        zml.stdx.flags.fatal("batch size, sequence length, and confirmation repetitions must be non-zero", .{});
    }

    const query_tokens = args.query_tokens orelse switch (args.mode) {
        .prefill => args.sequence_length,
        .decode => 1,
    };
    if (query_tokens == 0 or query_tokens > args.sequence_length) {
        zml.stdx.flags.fatal(
            "query tokens must be in the range 1..sequence-length (got {d} for sequence length {d})",
            .{ query_tokens, args.sequence_length },
        );
    }
    if (args.mode == .decode and query_tokens != 1) {
        zml.stdx.flags.fatal("decode mode requires exactly one query token per sequence", .{});
    }

    const total_query_tokens = std.math.mul(usize, args.batch_size, query_tokens) catch
        zml.stdx.flags.fatal("batch-size * query-tokens overflows usize", .{});
    if (args.sequence_length > std.math.maxInt(i32) or total_query_tokens > std.math.maxInt(i32)) {
        zml.stdx.flags.fatal("sequence or query-token count exceeds the i32 metadata range", .{});
    }

    // The kernels fetch page-table lanes for a whole tile before masking tail
    // tokens, so every row needs valid page ids through the widest candidate.
    const padded_sequence_length = std.mem.alignForward(usize, args.sequence_length, max_tuned_tile_size);
    const max_num_pages = std.math.divCeil(usize, padded_sequence_length, page_size) catch unreachable;
    const num_pages = std.math.mul(usize, args.batch_size, max_num_pages) catch
        zml.stdx.flags.fatal("batch-size * page count overflows usize", .{});
    if (num_pages > std.math.maxInt(i32)) {
        zml.stdx.flags.fatal("physical page count exceeds the i32 block-table range", .{});
    }

    // The default mode drives the generic tuner explicitly below. The
    // integration mode instead enables the Platform-owned automatic resolver
    // and verifies its cache across two independent compilations.
    var platform: *zml.Platform = try .auto(allocator, io, .{
        .autotune = args.automatic_platform_autotune,
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda and platform.target != .rocm) {
        log.err("native Triton attention autotuning requires CUDA or ROCm; selected platform is {t}", .{platform.target});
        return error.UnsupportedPlatform;
    }
    if (args.automatic_platform_autotune and !platform.executionTimerAvailable()) {
        log.err("automatic Triton attention autotuning requires native execution timers", .{});
        return error.ExecutionTimerUnavailable;
    }
    if (platform.devices.len == 0 or
        platform.devices.len > num_kv_heads or
        num_kv_heads % platform.devices.len != 0)
    {
        log.err("cannot shard {d} KV heads over {d} devices", .{ num_kv_heads, platform.devices.len });
        return error.UnsupportedDeviceTopology;
    }

    const model_sharding = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth }));
    const shardings = [_]zml.Sharding{ platform.replicated_sharding, model_sharding };
    const tensor_partitioning = .{ .hkv = .model };

    const q_shape = zml.Shape.init(.{
        .b = total_query_tokens,
        .hkv = num_kv_heads,
        .hg = num_query_heads / num_kv_heads,
        .hd = head_dim,
    }, .bf16).withPartitioning(tensor_partitioning);
    const cache_shape = zml.Shape.init(.{
        .page = num_pages,
        .k_chunk = page_size,
        .hkv = num_kv_heads,
        .hd = head_dim,
    }, .bf16).withPartitioning(tensor_partitioning);

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();
    var q_buffer = try randomBf16Buffer(allocator, io, platform, q_shape, model_sharding, random);
    defer q_buffer.deinit();
    var k_cache_buffer = try randomBf16Buffer(allocator, io, platform, cache_shape, model_sharding, random);
    defer k_cache_buffer.deinit();
    var v_cache_buffer = try randomBf16Buffer(allocator, io, platform, cache_shape, model_sharding, random);
    defer v_cache_buffer.deinit();

    const options: triton_attention.paged.Options = .{
        .batch_size = args.batch_size,
        .max_num_pages = max_num_pages,
        .max_seqlen_q = query_tokens,
        .max_seqlen_k = args.sequence_length,
        .is_prefill = args.mode == .prefill,
        .tuning = .automatic,
    };
    const parameters: Parameters = .init(options);

    const block_table = try allocator.alloc(i32, num_pages);
    defer allocator.free(block_table);
    for (block_table, 0..) |*page, index| page.* = @intCast(index);

    const seq_lens = try allocator.alloc(i32, args.batch_size);
    defer allocator.free(seq_lens);
    @memset(seq_lens, @intCast(args.sequence_length));

    const query_start_len = try allocator.alloc(i32, args.batch_size + 1);
    defer allocator.free(query_start_len);
    for (query_start_len, 0..) |*start, index| {
        start.* = @intCast(index * query_tokens);
    }

    var block_table_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        parameters.block_table.shape(),
        .replicated,
        std.mem.sliceAsBytes(block_table),
    );
    defer block_table_buffer.deinit();
    var seq_lens_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        parameters.seq_lens.shape(),
        .replicated,
        std.mem.sliceAsBytes(seq_lens),
    );
    defer seq_lens_buffer.deinit();
    var query_start_len_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        parameters.query_start_len.shape(),
        .replicated,
        std.mem.sliceAsBytes(query_start_len),
    );
    defer query_start_len_buffer.deinit();
    const parameter_buffers: zml.Bufferized(Parameters) = .{
        .block_table = block_table_buffer,
        .seq_lens = seq_lens_buffer,
        .query_start_len = query_start_len_buffer,
    };

    var ctx: Context = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .shardings = &shardings,
        .parameters = parameters,
        .parameter_buffers = parameter_buffers,
        .q_tensor = .fromShape(q_shape),
        .k_cache_tensor = .fromShape(cache_shape),
        .v_cache_tensor = .fromShape(cache_shape),
        .q_buffer = q_buffer,
        .k_cache_buffer = k_cache_buffer,
        .v_cache_buffer = v_cache_buffer,
        .attention_options = .{ .is_causal = true },
    };

    const reference_config: TuningConfig = if (args.automatic_platform_autotune)
        switch (args.mode) {
            .prefill => prefill_candidates[1],
            .decode => decode_candidates[1],
        }
    else
        .automatic;
    var reference = try ctx.makeReference(reference_config);
    defer reference.free(allocator);
    ctx.reference = &reference;

    if (args.automatic_platform_autotune) {
        try verifyAutomaticPlatformAutotune(&ctx, reference);
        return;
    }

    const candidates: []const TuningConfig = switch (args.mode) {
        .prefill => &prefill_candidates,
        .decode => &decode_candidates,
    };
    log.info(
        "autotuning {s} on {t}/{d} devices: batch={d}, query tokens/sequence={d}, KV length={d}, candidates={d}",
        .{ @tagName(args.mode), platform.target, platform.devices.len, args.batch_size, query_tokens, args.sequence_length, candidates.len },
    );

    const result = try zml.autotune(
        allocator,
        &ctx,
        candidates,
        Context.compileCandidate,
        Context.measure,
        Context.deinitProgram,
        .{},
    );
    log.info(
        "selected candidate {d}: {any}; median={f}, MAD={f}, repetitions={d}, samples={d}, compiled={d}, rejected={d}",
        .{
            result.candidate_index,
            result.config,
            result.median,
            result.mad,
            result.repetitions,
            result.sample_count,
            result.compiled_count,
            result.rejected_count,
        },
    );

    const winner_is_automatic = switch (result.config) {
        .automatic => true,
        .two_d, .three_d => false,
    };
    if (winner_is_automatic) {
        log.info("automatic configuration won; no redundant A/B confirmation needed", .{});
    } else {
        var automatic_program = try ctx.compileProgram(.automatic, .device);
        defer automatic_program.deinit(allocator);
        var winner_program = try ctx.compileProgram(result.config, .device);
        defer winner_program.deinit(allocator);
        var automatic_timer = try zml.ExecutionTimer.attach(&automatic_program.exe);
        var winner_timer = try zml.ExecutionTimer.attach(&winner_program.exe);

        for (0..3) |round| {
            if (round % 2 == 0) {
                _ = try automatic_timer.measureCall(io, automatic_program.arguments, &automatic_program.results, 1);
                _ = try winner_timer.measureCall(io, winner_program.arguments, &winner_program.results, 1);
            } else {
                _ = try winner_timer.measureCall(io, winner_program.arguments, &winner_program.results, 1);
                _ = try automatic_timer.measureCall(io, automatic_program.arguments, &automatic_program.results, 1);
            }
        }

        const automatic_samples = try allocator.alloc(i96, args.confirmation_repetitions);
        defer allocator.free(automatic_samples);
        const winner_samples = try allocator.alloc(i96, args.confirmation_repetitions);
        defer allocator.free(winner_samples);
        const scratch = try allocator.alloc(i96, args.confirmation_repetitions);
        defer allocator.free(scratch);
        for (0..args.confirmation_repetitions) |round| {
            if (round % 2 == 0) {
                automatic_samples[round] = (try automatic_timer.measureCall(io, automatic_program.arguments, &automatic_program.results, 1)).nanoseconds;
                winner_samples[round] = (try winner_timer.measureCall(io, winner_program.arguments, &winner_program.results, 1)).nanoseconds;
            } else {
                winner_samples[round] = (try winner_timer.measureCall(io, winner_program.arguments, &winner_program.results, 1)).nanoseconds;
                automatic_samples[round] = (try automatic_timer.measureCall(io, automatic_program.arguments, &automatic_program.results, 1)).nanoseconds;
            }
        }
        const automatic_stats = timingStats(automatic_samples, scratch);
        const winner_stats = timingStats(winner_samples, scratch);
        const speedup = @as(f64, @floatFromInt(automatic_stats.median.nanoseconds)) /
            @as(f64, @floatFromInt(winner_stats.median.nanoseconds));
        log.info(
            "interleaved confirmation ({d} samples): automatic={f} (MAD {f}), winner={f} (MAD {f}), speedup={d:.3}x",
            .{
                args.confirmation_repetitions,
                automatic_stats.median,
                automatic_stats.mad,
                winner_stats.median,
                winner_stats.mad,
                speedup,
            },
        );
    }

    var production_program = try ctx.compileProgram(result.config, .none);
    defer production_program.deinit(allocator);
    try production_program.exe.tryCallOpts(
        io,
        production_program.arguments,
        &production_program.results,
        .{ .wait = true, .allow_input_donation = false },
    );
    var production_output = production_program.results.get(zml.Buffer);
    defer production_output.deinit();
    try zml.testing.expectClose(io, reference, production_output, .{
        .absolute_tolerance = 1e-2,
        .relative_tolerance = 1e-2,
        .epsilon_relative = 1e-6,
    });
    log.info("clean timing-disabled winner compiled, executed, and matched the reference", .{});
}

fn verifyAutomaticPlatformAutotune(ctx: *Context, reference: zml.Slice) !void {
    automatic_resolver_observations.reset();
    log.info(
        "verifying automatic Platform autotuning on {t}/{d} devices with two independent compilations",
        .{ ctx.platform.target, ctx.platform.devices.len },
    );

    try compileAutomaticAndCheck(ctx, reference);
    const tuned_after_first = automatic_resolver_observations.tuned.load(.seq_cst);
    const cached_after_first = automatic_resolver_observations.cached.load(.seq_cst);
    if (tuned_after_first != 1 or cached_after_first != 0) {
        log.err(
            "first automatic compilation should tune exactly once (observed tuned={d}, cache={d})",
            .{ tuned_after_first, cached_after_first },
        );
        return error.UnexpectedFirstAutotuneSource;
    }
    if (!automatic_resolver_observations.positive_tuned_timing.load(.seq_cst)) {
        log.err("automatic resolver did not report a positive native timing", .{});
        return error.NonPositiveAutotuneTiming;
    }

    try compileAutomaticAndCheck(ctx, reference);
    const tuned_after_second = automatic_resolver_observations.tuned.load(.seq_cst);
    const cached_after_second = automatic_resolver_observations.cached.load(.seq_cst);
    if (tuned_after_second != 1 or cached_after_second != 1) {
        log.err(
            "second automatic compilation should reuse the Platform cache (observed tuned={d}, cache={d})",
            .{ tuned_after_second, cached_after_second },
        );
        return error.AutotuneCacheNotReused;
    }

    log.info(
        "automatic resolver tuned once with positive native timing, the independent compilation hit the Platform cache, and both outputs matched",
        .{},
    );
}

fn compileAutomaticAndCheck(ctx: *Context, reference: zml.Slice) !void {
    var program = try ctx.compileProgram(.automatic, .none);
    defer program.deinit(ctx.allocator);
    try program.exe.tryCallOpts(
        ctx.io,
        program.arguments,
        &program.results,
        .{ .wait = true, .allow_input_donation = false },
    );
    var output = program.results.get(zml.Buffer);
    defer output.deinit();
    try zml.testing.expectClose(ctx.io, reference, output, .{
        .absolute_tolerance = 1e-2,
        .relative_tolerance = 1e-2,
        .epsilon_relative = 1e-6,
    });
}

fn randomBf16Buffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
    random: std.Random,
) !zml.Buffer {
    const host = try zml.Slice.alloc(allocator, shape);
    defer host.free(allocator);
    std.debug.assert(shape.dtype() == .bf16);
    for (host.items(zml.floats.BFloat16)) |*value| {
        value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    return .fromSlice(io, platform, host, sharding);
}

const TimingStats = struct {
    median: std.Io.Duration,
    mad: std.Io.Duration,
};

fn timingStats(samples: []const i96, scratch: []i96) TimingStats {
    std.debug.assert(samples.len != 0 and scratch.len >= samples.len);
    @memcpy(scratch[0..samples.len], samples);
    std.mem.sort(i96, scratch[0..samples.len], {}, comptime std.sort.asc(i96));
    const median = medianOfSorted(scratch[0..samples.len]);
    for (samples, 0..) |sample, index| scratch[index] = @intCast(@abs(sample - median));
    std.mem.sort(i96, scratch[0..samples.len], {}, comptime std.sort.asc(i96));
    return .{
        .median = .fromNanoseconds(median),
        .mad = .fromNanoseconds(medianOfSorted(scratch[0..samples.len])),
    };
}

fn medianOfSorted(values: []const i96) i96 {
    const middle = values.len / 2;
    if (values.len % 2 != 0) return values[middle];
    return values[middle - 1] + @divTrunc(values[middle] - values[middle - 1], 2);
}
