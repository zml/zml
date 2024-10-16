const std = @import("std");

const zml = @import("zml");
const meta = zml.meta;
const asynk = @import("async");
const flags = @import("tigerbeetle/flags");
const llama_mod = @import("llamalib");

const async_ = asynk.async_;

const LlamaLM = llama_mod.LlamaLM;
const Llama = llama_mod.Llama;
const KvCache = llama_mod.KvCache;
const TransformerLayer = llama_mod.TransformerLayer;
const SelfAttn = llama_mod.SelfAttn;
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const log = std.log.scoped(.llama);

// set this to false to disable the verbose logging
const show_mlir = true;

pub const std_options = .{
    .log_level = .err,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .pjrt, .level = if (show_mlir) .debug else .err },
        .{ .scope = .zml_module, .level = if (show_mlir) .debug else .err },
        .{ .scope = .zml, .level = if (show_mlir) .debug else .err },
        .{ .scope = .llama, .level = if (show_mlir) .debug else .info },
    },
};

pub const MlpModule = struct {
    mlp: llama_mod.Mlp,

    pub fn forward(self: MlpModule, x: zml.Tensor) zml.Tensor {
        return zml.call(self.mlp, .forward, .{x});
    }
};

pub const SelfAttnModule = struct {
    self_attn: llama_mod.SelfAttn,

    pub fn forward(self: SelfAttnModule, x: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache) struct { zml.Tensor, KvCache } {
        const y, const updated_kv_cache = self.self_attn.forward(x, token_index, kv_cache);
        return .{ y, updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain, .{});
}

const CliArgs = struct {
    pub const help =
        \\ llama --model=llama3.7B.safetensors --tokenizer=vocab.json --num_layers=2
    ;
    model: []const u8,
    layer_start: u8 = 0,
    num_layers: ?u8 = null,
    seq_len: u32 = 256,
    topk: u32 = 2,
    temperature: u32 = 1,
    num_heads: ?i64 = null,
    num_kv_heads: ?i64 = null,
    rope_freq_base: ?i64 = null,
    prompt: ?[]const u8 = null,
    test_activations: ?[]const u8 = null,
    seed: ?u128 = null,
    iterations: usize = 1,
};

pub fn asyncMain() !void {
    log.info("   LLama was compiled with {}", .{@import("builtin").mode});

    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const tmp = try std.fs.openDirAbsolute("/tmp", .{});
    try tmp.makePath("zml/sharding/cache");
    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/sharding",
        .sharding_enabled = true,
    };

    const platform = context.autoPlatform().withCompilationOptions(compilation_options);
    {
        // List available targets
        std.debug.print("\nSupported Platforms:\n", .{});
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

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);
    const model_file = cli_args.model;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    log.info("Model file: {s}", .{model_file});

    var ts = try zml.aio.detectFormatAndOpen(allocator, model_file);
    defer ts.deinit();

    var llama: LlamaLM = try zml.aio.populateModel(LlamaLM, model_arena, ts);
    const num_heads: i64 = cli_args.num_heads orelse ts.metadata("num_heads", .int64) orelse @panic("--num_heads is required for this model");
    const num_kv_heads: i64 = cli_args.num_kv_heads orelse ts.metadata("num_kv_heads", .int64) orelse num_heads;

    const rope_impl = if (ts.metadata("rope_impl", .string)) |val|
        std.meta.stringToEnum(zml.nn.RopeOpts.Implementation, val).?
    else
        .sequential;

    const llama_options: llama_mod.LlamaOptions = .{
        .max_seq_len = cli_args.seq_len,
        .num_kv_heads = num_kv_heads,
        .num_heads = num_heads,
        .gen_opts = .{
            .topk = cli_args.topk,
            .temperature = @floatFromInt(cli_args.temperature),
        },
        .rms_norm_eps = @floatCast(ts.metadata("rms_norm_eps", .float64) orelse 1e-5),
        .rope_opts = .{
            .impl = rope_impl,
            .freq_base = @floatCast(ts.metadata("rope_freq_base", .float64) orelse @as(f32, @floatFromInt(cli_args.rope_freq_base orelse 10_000))),
        },
    };
    log.info("✅ Parsed llama config: {}", .{llama_options});
    llama.init(llama_options);

    try runSelfAttnBenchmark(cli_args, &llama, llama_options, ts, model_arena, platform);
}

fn runMlpBenchmark(cli_args: CliArgs, llama: *LlamaLM, llama_options: llama_mod.LlamaOptions, ts: zml.aio.BufferStore, arena: std.mem.Allocator, platform: zml.Platform) !void {
    const dims = llama.model.shape();
    const S: usize = @intCast(cli_args.seq_len);
    const D: usize = @intCast(dims.nh * dims.hd);
    const H: usize = @intCast(llama.model.layers[0].mlp.up_proj.weight.dim(0));
    const dtype = llama.lm_head.weight.dtype();

    log.info(
        \\
        \\S: {}
        \\D: {}
        \\H: {}
        \\
    , .{ S, D, H });

    const x_shape = zml.Shape.init(.{ .s = cli_args.seq_len, .d = dims.nh * dims.hd }, dtype);

    var compilation = try async_(zml.compileModel, .{ arena, MlpModule{ .mlp = llama.model.layers[0].mlp }, .forward, .{x_shape}, platform });

    var llama_weights = try zml.aio.loadBuffers(LlamaLM, .{llama_options}, ts, arena, platform);
    defer zml.aio.unloadBuffers(&llama_weights);

    const compiled = try compilation.await_();
    var executable = try compiled.prepare(arena, .{ .mlp = llama_weights.model.layers[0].mlp });
    defer executable.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var x_buffer = try createRandomBuffer(arena, platform, x_shape, random);
    defer x_buffer.deinit();
    std.debug.print("\nRunning benchmark....\n", .{});

    // Ignore first run
    {
        var result: zml.Buffer = executable.call(.{x_buffer});
        defer result.deinit();
    }

    // call our executable module
    var elapsed_ns: u64 = 0;
    var timer = try std.time.Timer.start();
    for (0..cli_args.iterations) |_| {
        timer.reset();
        var result: zml.Buffer = executable.call(.{x_buffer});
        elapsed_ns += timer.lap();
        defer result.deinit();
    }
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const avg_elapsed_ms = elapsed_ms / @as(f64, @floatFromInt(cli_args.iterations));

    std.debug.print("\n✅ Benchmark done!\n\n", .{});

    std.debug.print(
        \\  x_shape: {}
        \\  Datatype: {s}
        \\  Iterations: {d}
        \\  Total elapsed: {d:.3}ms (avg: {d:.3}ms/it)
        \\
        \\
    , .{ x_shape, @tagName(x_shape.dtype()), cli_args.iterations, elapsed_ms, avg_elapsed_ms });
}

fn runSelfAttnBenchmark(cli_args: CliArgs, llama: *LlamaLM, llama_options: llama_mod.LlamaOptions, ts: zml.aio.BufferStore, arena: std.mem.Allocator, platform: zml.Platform) !void {
    const dims = llama.model.shape();
    const dtype = llama.lm_head.weight.dtype();
    const x_shape = zml.Shape.init(.{ .s = 1, .d = dims.nh * dims.hd }, dtype);
    const token_idx_shape = zml.Shape.init(.{}, .i32);
    const kv_shape = zml.Shape.init(.{ .layer = llama.model.layers.len, .h = dims.nkvh, .k = dims.s, .hd = dims.hd }, dtype).withSharding(.{.h});
    // needs to be optional
    const kv_cache_shape: ShapeOf(KvCache) = KvCache.initShape(kv_shape);

    var compilation = try async_(zml.compileModel, .{ arena, SelfAttnModule{ .self_attn = llama.model.layers[0].self_attn }, .forward, .{ x_shape, token_idx_shape, kv_cache_shape }, platform });

    var llama_weights = try zml.aio.loadBuffers(LlamaLM, .{llama_options}, ts, arena, platform);
    defer zml.aio.unloadBuffers(&llama_weights);

    const compiled = try compilation.await_();
    var executable = try compiled.prepare(arena, .{ .self_attn = llama_weights.model.layers[0].self_attn });
    defer executable.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var x_buffer = try createRandomBuffer(arena, platform, x_shape, random);
    defer x_buffer.deinit();
    var token_index_buffer: zml.Buffer = try zml.Buffer.fromSlice(platform, .{}, &[1]i32{1});
    defer token_index_buffer.deinit();
    var k_cache_buffer = try createRandomBuffer(arena, platform, kv_shape, random);
    defer k_cache_buffer.deinit();
    var v_cache_buffer = try createRandomBuffer(arena, platform, kv_shape, random);
    defer v_cache_buffer.deinit();
    var layer_index_buffer = try zml.Buffer.fromSlice(platform, .{}, &[1]i32{0});
    defer layer_index_buffer.deinit();
    var updated_kv_cache: zml.Bufferized(KvCache) = .{ .k = k_cache_buffer, .v = v_cache_buffer, .layer_index = layer_index_buffer };
    std.debug.print("\nRunning benchmark....\n", .{});

    // Ignore first run
    {
        var x: zml.Buffer, updated_kv_cache = executable.call(.{ x_buffer, token_index_buffer, updated_kv_cache });
        defer x.deinit();
    }

    // call our executable module
    var elapsed_ns: u64 = 0;
    var timer = try std.time.Timer.start();
    for (0..cli_args.iterations) |_| {
        timer.reset();
        var x: zml.Buffer, updated_kv_cache = executable.call(.{ x_buffer, token_index_buffer, updated_kv_cache });
        elapsed_ns += timer.lap();
        defer x.deinit();
    }
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const avg_elapsed_ms = elapsed_ms / @as(f64, @floatFromInt(cli_args.iterations));

    std.debug.print("\n✅ Benchmark done!\n\n", .{});

    std.debug.print(
        \\  x_shape: {}
        \\  Datatype: {s}
        \\  Iterations: {d}
        \\  Total elapsed: {d:.3}ms (avg: {d:.3}ms/it)
        \\
        \\
    , .{ x_shape, @tagName(x_shape.dtype()), cli_args.iterations, elapsed_ms, avg_elapsed_ms });
}

fn computeFloatingOpCount(S: usize, D: usize, H: usize) usize {
    const up_proj_flops = 2 * S * D * H;
    const gate_proj_flops = 2 * S * D * H;
    const silu_flops = 4 * S * H;
    const up_gate_mul_flops = S * H;
    const down_proj_flops = 2 * S * H * D;
    return up_proj_flops + gate_proj_flops + silu_flops + up_gate_mul_flops + down_proj_flops;
}

fn createRandomBuffer(allocator: std.mem.Allocator, platform: zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
    const data = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(data);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = random.int(ZigType);
                },
                .float => {
                    const value = random.float(f64);
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = if (ZigType == f64)
                        value
                    else if (ZigType == f32)
                        @floatCast(value)
                    else if (ZigType == f16)
                        @floatCast(value)
                    else
                        @bitCast(random.int(std.meta.Int(.unsigned, @bitSizeOf(ZigType))));
                },
                .complex => unreachable,
            }
        },
    }

    var host_buffer = zml.HostBuffer.fromBytes(shape, data);
    errdefer host_buffer.deinit(allocator);
    return zml.Buffer.from(platform, host_buffer);
}
