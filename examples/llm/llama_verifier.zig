const std = @import("std");
const zml = @import("zml");
const common = @import("models/common.zig");
const llama = @import("models/llama.zig");
const inference = llama.inference;

pub const std_options: std.Options = .{ .log_level = .info };
const log = std.log.scoped(.llama_verifier);
const bf16 = zml.DataType.bf16.toZigType();

const Args = struct {
    model: []const u8,
    prompt: []const u8 = "What is 7 plus 5? Explain briefly.",
    rows: u32 = 1,
    serial_attention: bool = false,
    seqlen: u32 = 128,
    offset: u32 = 0,
    window: u32 = 1,
    continuation: u32 = 16,
    repeats: u32 = 5,

    pub const help =
        \\Teacher-forced BF16 greedy Llama numerical experiment (single Tenstorrent device).
        \\--model=<local path> --prompt=<text> --rows=1|2 --seqlen=128
        \\--serial-attention (use native single-query decode attention per row)
        \\--offset=<cache position> (0: first decode position after prompt)
        \\--window=<consecutive verifier blocks> (default: 1, maximum: 8)
        \\--continuation=<serial steps after verification> (default: 16)
        \\--repeats=<isolated repeated verifier calls for latency> (default: 5)
        \\Oracle candidates come from serial reference; this reports block latency, not generation throughput.
    ;
};

const Snapshot = struct {
    k: zml.Slice,
    v: zml.Slice,

    fn read(allocator: std.mem.Allocator, io: std.Io, buffers: llama.model.KvCache.Buffer) !Snapshot {
        const k = try buffers.k.toSliceAlloc(allocator, io);
        errdefer k.free(allocator);
        return .{ .k = k, .v = try buffers.v.toSliceAlloc(allocator, io) };
    }

    fn deinit(self: Snapshot, allocator: std.mem.Allocator) void {
        self.k.free(allocator);
        self.v.free(allocator);
    }

    fn upload(self: Snapshot, io: std.Io, platform: *const zml.Platform) !llama.model.KvCache.Buffer {
        var k = try zml.Buffer.fromSlice(io, platform, self.k);
        errdefer k.deinit();
        return .{ .k = k, .v = try zml.Buffer.fromSlice(io, platform, self.v) };
    }
};

const VerifierRunner = inference.VerifierExe.Runner(.{.weights});

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if ((args.rows != 1 and args.rows != 2) or args.window == 0 or args.window > 8 or args.seqlen == 0) return error.InvalidArguments;

    const platform = try zml.Platform.init(allocator, io, .tenstorrent, .{ .tenstorrent = .{ .device_index = 0 } });
    defer platform.deinit(allocator, io);
    log.info("{f}", .{platform.fmtVerbose()});
    const backend: zml.attention.Backend = .{ .flash_attention = .tenstorrent };
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    defer repo.close(io);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    const meshes = try common.Meshes.init(platform);
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &meshes.all());
    defer store.deinit();
    var loaded = try llama.LoadedModel.init(allocator, io, repo, store.view(), .{ .sampling_strategy = .{ .topk = 1 } });
    defer loaded.deinit(allocator);
    if (loaded.inner.model.embed_tokens.weight.dtype() != .bf16) return error.VerifierRequiresBF16;

    var progress = std.Progress.start(io, .{ .root_name = "Teacher-forced verifier" });
    errdefer progress.end();
    var compiled = try loaded.compile(allocator, io, platform, backend, meshes, args.seqlen, &progress);
    defer compiled.deinit();
    var row_one = try inference.compileVerifier(allocator, io, platform, loaded.inner, compiled.params, 1, args.serial_attention);
    defer row_one.deinit();
    var wider = if (args.rows == 2) try inference.compileVerifier(allocator, io, platform, loaded.inner, compiled.params, 2, args.serial_attention) else null;
    defer if (wider) |*exe| exe.deinit();
    var weights = try loaded.loadBuffers(allocator, io, platform, &store, &progress);
    defer loaded.unloadBuffers(&weights, allocator);
    progress.end();
    var tokenizer = try loadTokenizer(allocator, io, repo);
    defer tokenizer.deinit();
    var session = try llama.Session.init(allocator, io, platform, tokenizer, &compiled, &weights);
    defer session.deinit();
    const prompt = try session.tokenizePrompt(allocator, args.prompt);
    defer allocator.free(prompt);
    if (prompt.len >= args.seqlen) return error.PromptTooLong;
    const first_position: u32 = @intCast(prompt.len);
    const offset = if (args.offset == 0) first_position else args.offset;
    if (offset < first_position or @as(u64, offset) + @as(u64, args.rows) * args.window > args.seqlen) return error.InvalidOffsetOrWindow;
    log.info("prompt_tokens={any}, first_position={d}, offset={d}, rows={d}, window={d}, serial_attention={}", .{ prompt, first_position, offset, args.rows, args.window, args.serial_attention });
    try session.runPrefill(prompt);
    var token = try scalarToken(io, platform, session.last_generated_token);
    defer token.deinit();
    var current = session.last_generated_token;
    var active = try zml.Buffer.scalar(io, platform, 1, .u32);
    defer active.deinit();
    var metadata = try compiled.params.attention_metadata.initBuffer(io, platform);
    defer zml.attention.Metadata.deinitBuffer(&metadata);
    for (first_position..offset) |position| {
        if (isEos(session.config, current)) return error.ReferenceReachedEosBeforeOffset;
        current = try serialStep(&session, &session.kv_cache_buffers, &token, position, &active, &metadata);
    }
    if (isEos(session.config, current)) return error.ReferenceReachedEosBeforeOffset;
    const initial_token = current;
    const initial = try Snapshot.read(allocator, io, session.kv_cache_buffers);
    defer initial.deinit(allocator);
    const block_steps = args.rows * args.window;
    var reference: std.ArrayList(u32) = .empty;
    defer reference.deinit(allocator);
    var snapshots: std.ArrayList(Snapshot) = .empty;
    defer {
        for (snapshots.items) |snapshot| snapshot.deinit(allocator);
        snapshots.deinit(allocator);
    }
    var first_step: ?Snapshot = null;
    defer if (first_step) |snapshot| snapshot.deinit(allocator);
    const maximum_steps = @min(@as(u64, block_steps) + args.continuation, args.seqlen - offset);
    for (0..@intCast(maximum_steps)) |step| {
        if (isEos(session.config, current)) break;
        current = try serialStep(&session, &session.kv_cache_buffers, &token, offset + step, &active, &metadata);
        try reference.append(allocator, current);
        if (step == 0) first_step = try Snapshot.read(allocator, io, session.kv_cache_buffers);
        if (step < block_steps and (step + 1) % args.rows == 0) try snapshots.append(allocator, try Snapshot.read(allocator, io, session.kv_cache_buffers));
    }
    if (reference.items.len < block_steps) return error.ReferenceReachedEosBeforeWindow;
    const serial_final = try Snapshot.read(allocator, io, session.kv_cache_buffers);
    defer serial_final.deinit(allocator);
    log.info("serial_reference_ids={any}", .{reference.items});

    // First gate: all-row head with one row versus the unchanged serial head.
    var one_runner = try VerifierRunner.init(&row_one, allocator, .{ .weights = weights });
    defer one_runner.deinit(allocator);
    var one_cache = try initial.upload(io, platform);
    defer llama.model.KvCache.deinitBuffer(&one_cache);
    var one_input = try scalarToken(io, platform, initial_token);
    defer one_input.deinit();
    var one_result = runVerifier(&one_runner, io, &one_cache, one_input, session.token_index_buffers[offset], active, metadata);
    defer one_result.deinit();
    const one_prediction = try one_result.getValue(u32, io);
    const one_snapshot = try Snapshot.read(allocator, io, one_cache);
    defer one_snapshot.deinit(allocator);
    const one_exact = compareCache(first_step.?, one_snapshot, "all-row s1 parity");
    log.info("all_row_s1 serial={d}, verifier={d}, cache_exact={}", .{ reference.items[0], one_prediction, one_exact });
    if (one_prediction != reference.items[0] or !one_exact) return error.AllRowHeadParityFailed;

    var runner = try VerifierRunner.init(if (wider) |*exe| exe else &row_one, allocator, .{ .weights = weights });
    defer runner.deinit(allocator);
    var verifier_cache = try initial.upload(io, platform);
    defer llama.model.KvCache.deinitBuffer(&verifier_cache);
    var exact = true;
    var predicted: [2]u32 = undefined;
    var last_input: ?zml.Buffer = null;
    defer if (last_input) |*buffer| buffer.deinit();
    var last_position: usize = offset;
    for (0..args.window) |block| {
        const start = block * args.rows;
        var inputs: [2]u32 = undefined;
        for (inputs[0..args.rows], 0..) |*input, row| input.* = if (start + row == 0) initial_token else reference.items[start + row - 1];
        if (last_input) |*buffer| buffer.deinit();
        last_input = try zml.Buffer.fromBytes(io, platform, .init(.{ .s = args.rows }, .u32), std.mem.sliceAsBytes(inputs[0..args.rows]));
        last_position = offset + start;
        var output = runVerifier(&runner, io, &verifier_cache, last_input.?, session.token_index_buffers[last_position], active, metadata);
        defer output.deinit();
        try output.toSlice(io, .init(output.shape(), std.mem.sliceAsBytes(predicted[0..args.rows])));
        for (predicted[0..args.rows], 0..) |prediction, row| {
            if (prediction != reference.items[start + row]) {
                log.err("token mismatch block={d} position={d} serial={d} verifier={d}", .{ block, last_position + row, reference.items[start + row], prediction });
                exact = false;
            }
        }
        const actual = try Snapshot.read(allocator, io, verifier_cache);
        defer actual.deinit(allocator);
        const cache_exact = compareCache(snapshots.items[block], actual, "verifier block");
        exact = cache_exact and exact;
        log.info("block={d}, position={d}, inputs={any}, predictions={any}, cache_exact={}", .{ block, last_position, inputs[0..args.rows], predicted[0..args.rows], cache_exact });
    }

    // Repeat the final teacher-forced block at its fixed position. Upload,
    // prediction readback, acceptance, and cache diagnostics are outside timing.
    if (args.repeats > 0) {
        var warm = runVerifier(&runner, io, &verifier_cache, last_input.?, session.token_index_buffers[last_position], active, metadata);
        warm.deinit();
        var last_repeated: ?zml.Buffer = null;
        defer if (last_repeated) |*output| output.deinit();
        const start: std.Io.Timestamp = .now(io, .awake);
        for (0..args.repeats) |iteration| {
            var output = runVerifier(&runner, io, &verifier_cache, last_input.?, session.token_index_buffers[last_position], active, metadata);
            if (iteration + 1 == args.repeats) last_repeated = output else output.deinit();
        }
        const elapsed = start.untilNow(io, .awake);
        log.info("teacher_forced_upper_bound rows={d}, calls={d}, total={f}, mean_ns_per_block={d}; excludes upload/readback/acceptance; not generation throughput", .{ args.rows, args.repeats, elapsed, @divTrunc(elapsed.toNanoseconds(), args.repeats) });
        const repeated_output = last_repeated.?;
        try repeated_output.toSlice(io, .init(repeated_output.shape(), std.mem.sliceAsBytes(predicted[0..args.rows])));
        const expected_start = block_steps - args.rows;
        for (predicted[0..args.rows], 0..) |prediction, row| {
            if (prediction != reference.items[expected_start + row]) {
                log.err("repeated token mismatch position={d} serial={d} verifier={d}", .{ last_position + row, reference.items[expected_start + row], prediction });
                exact = false;
            }
        }
        log.info("last_repeated_predictions={any}", .{predicted[0..args.rows]});
        const after_repeats = try Snapshot.read(allocator, io, verifier_cache);
        defer after_repeats.deinit(allocator);
        exact = compareCache(snapshots.items[snapshots.items.len - 1], after_repeats, "repeated verifier block") and exact;
    }
    var continuation_token = try scalarToken(io, platform, predicted[args.rows - 1]);
    defer continuation_token.deinit();
    var continued = predicted[args.rows - 1];
    for (reference.items[block_steps..], 0..) |expected, step| {
        if (isEos(session.config, continued)) {
            log.err("verifier continuation reached EOS early at step={d}", .{step});
            exact = false;
            break;
        }
        continued = try serialStep(&session, &verifier_cache, &continuation_token, offset + block_steps + step, &active, &metadata);
        if (continued != expected) {
            log.err("continuation mismatch step={d} serial={d} verifier={d}", .{ step, expected, continued });
            exact = false;
        }
    }
    const final = try Snapshot.read(allocator, io, verifier_cache);
    defer final.deinit(allocator);
    exact = compareCache(serial_final, final, "serial continuation") and exact;
    log.info("exact_serial_gate={}, continuation_steps={d}", .{ exact, reference.items.len - block_steps });
    if (!exact) return error.VerifierSerialEquivalenceFailed;
}

fn serialStep(session: *llama.Session, cache: *llama.model.KvCache.Buffer, token: *zml.Buffer, position: usize, active: *zml.Buffer, metadata: *const zml.Bufferized(zml.attention.Metadata)) !u32 {
    inference.run(&session.decode, .{
        .io = session.io,
        .tokens_buf = token,
        .next_token_buf = token,
        .token_index_buf = &session.token_index_buffers[position],
        .active_length_buf = active,
        .kv_cache_buffers = cache,
        .rng_buffers = &session.rng_buffers,
        .attention_metadata_buffers = metadata,
    });
    return token.getValue(u32, session.io);
}

fn runVerifier(runner: *VerifierRunner, io: std.Io, cache: *llama.model.KvCache.Buffer, tokens: zml.Buffer, position: zml.Buffer, active: zml.Buffer, metadata: zml.Bufferized(zml.attention.Metadata)) zml.Buffer {
    var output: zml.Buffer = undefined;
    runner.run(io, .{
        .inputs = .{ .tokens = tokens, .token_index = position, .active_length = active, .kv_cache = cache.*, .rng = null, .attention_metadata = metadata },
        .outputs = .{ .tokens = &output, .kv_cache = cache },
    }, .{ .wait = true });
    return output;
}

fn scalarToken(io: std.Io, platform: *const zml.Platform, token: u32) !zml.Buffer {
    return zml.Buffer.fromBytes(io, platform, .init(.{ .s = 1 }, .u32), std.mem.asBytes(&token));
}

fn isEos(config: *const llama.Config, token: u32) bool {
    return switch (config.eos_token_id) {
        .int => |id| token == id,
        .ints => |ids| std.mem.indexOfScalar(u32, ids, token) != null,
    };
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try repo.openFile(io, "tokenizer.json", .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);
    return .fromBytes(allocator, bytes);
}

fn compareCache(expected: Snapshot, actual: Snapshot, label: []const u8) bool {
    var exact = true;
    inline for (.{ "k", "v" }) |field| {
        const want = @field(expected, field);
        const got = @field(actual, field);
        const a = want.items(bf16);
        const b = got.items(bf16);
        var different: usize = 0;
        var first: usize = 0;
        var max_error: f32 = 0;
        for (a, b, 0..) |left, right, i| {
            if (@as(u16, @bitCast(left)) != @as(u16, @bitCast(right))) {
                if (different == 0) first = i;
                different += 1;
                const error_abs = @abs(left.toF32() - right.toF32());
                if (!std.math.isFinite(error_abs)) max_error = std.math.inf(f32) else max_error = @max(max_error, error_abs);
            }
        }
        if (different != 0) {
            exact = false;
            var coordinates: [zml.Shape.MAX_RANK]usize = @splat(0);
            var remaining = first;
            var axis: usize = want.shape.rank();
            while (axis > 0) {
                axis -= 1;
                const dim: usize = @intCast(want.shape.dims()[axis]);
                coordinates[axis] = remaining % dim;
                remaining /= dim;
            }
            log.warn("{s}: {s} cache differs elements={d}, first_layer={d}, first_row={d}, first_head={d}, first_hd={d}, max_abs={d}, first_expected={d}, first_actual={d}", .{ label, field, different, coordinates[want.shape.axis(.layer)], coordinates[want.shape.axis(.k)], coordinates[want.shape.axis(.h)], coordinates[want.shape.axis(.hd)], max_error, a[first].toF32(), b[first].toF32() });
        }
    }
    return exact;
}
