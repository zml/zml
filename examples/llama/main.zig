const asynk = @import("async");
const clap = @import("clap");
const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const llama = @import("llama.zig");

const LlamaLM = llama.LlamaLM;
const Llama = llama.Llama;
const KvCache = llama.KvCache;
const TransformerLayer = llama.TransformerLayer;
const SelfAttn = llama.SelfAttn;
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const log = std.log.scoped(.llama);

// set this to false to disable the verbose logging
const show_mlir = false;

pub const std_options = .{
    .log_level = .warn,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .@"zml/module", .level = if (show_mlir) .debug else .warn },
        .{ .scope = .llama, .level = .info },
    },
};

// pub fn generateText(
//     mod_llama: LlamaLM,
//     mod_prefill: zml.ModuleExe(LlamaLM.forwardPrefill),
//     mod: zml.ModuleExe(LlamaLM.forward),
//     tokenizer: zml.tokenizer.Tokenizer,
//     allocator: std.mem.Allocator,
//     opts: CliArgs,
// ) ![]const u8 {
//     const prompt_tok = tokenizer.encode(allocator, opts.prompt, .{}) catch unreachable;
//     const batch_size = opts.batch_size;
//     log.info("✅\tTokenized prompt: {d}", .{prompt_tok.len});
//     const dims = llama.model.shape();
//     const max_seq_len = dims.s;
//     const token_buffer = try allocator.alloc(i32, @intCast(batch_size * max_seq_len));
//     @memset(token_buffer, 0);
//     for (0..batch_size) |b| {
//         const seq = token_buffer[b * max_seq_len ..];
//         for (0..prompt_tok.len) |i| {
//             seq[i] = @intCast(prompt_tok[i]);
//         }
//     }
//
//     const tracer_buffer = try allocator.alloc(u8, @intCast(batch_size * max_seq_len));
//     defer allocator.free(token_buffer);
//     defer allocator.free(tracer_buffer);
//     defer allocator.free(prompt_tok);
//     var output = std.ArrayList(u8).init(allocator);
//     defer output.deinit();
//
//     var tokens = try zml.Buffer.fromSlice(mod.platform(), .{ batch_size, max_seq_len }, token_buffer);
//     var token_index = try zml.Buffer.constant(mod.platform(), zml.Shape.init(.{batch_size}, .i32), prompt_tok.len - 1);
//
//     var profiler = if (opts.profile != null) mod.platform().getProfiler(null) else undefined;
//     const seed = opts.seed orelse @as(u128, @bitCast(std.time.nanoTimestamp()));
//     var rng = try zml.Tensor.Rng.init(mod.platform(), seed);
//     if (opts.profile != null) profiler.start();
//     const prefill_start = std.time.microTimestamp();
//     tokens, token_index, var kv_cache, rng = mod_prefill.call(.{ tokens, token_index, rng });
//     const prefill_end = std.time.microTimestamp();
//     defer kv_cache.k.deinit();
//     defer kv_cache.v.deinit();
//     defer kv_cache.layer_index.deinit();
//
//     const tracer = zml.tools.Tracer.init("ai.zml.models.llama");
//     var decode_progress = prompt_tok.len;
//     const output_tokens_len = max_seq_len - prompt_tok.len - 1;
//
//     const start = std.time.microTimestamp();
//     const output_freq: u8 = 1;
//     for (0..output_tokens_len) |i| {
//         //_ = i;
//         const frame_id = tracer.frameStart(try std.fmt.bufPrintZ(tracer_buffer, "Generate token {}/{}", .{ i + 1, output_tokens_len }));
//         tokens, token_index, kv_cache, rng = mod.call(.{ tokens, token_index, kv_cache, rng });
//         if (!opts.benchmark and batch_size == 1 and (i + 1) % output_freq == 0) {
//             const n = output.items.len;
//             _ = try tokens.toHost(std.mem.sliceAsBytes(token_buffer));
//             try tokenizer.decodeWithOpts(&output, @ptrCast(token_buffer[decode_progress..][0..output_freq]), .{});
//             decode_progress += output_freq;
//             // std.debug.print("{s}", .{output.items[n..]});
//             tracer.frameEnd(frame_id, try std.fmt.bufPrintZ(tracer_buffer, "Decoded token {}/{} : {s}", .{ i + 1, output_tokens_len, output.items[n..] }));
//         } else {
//             tracer.frameEnd(frame_id, try std.fmt.bufPrintZ(tracer_buffer, "Generated token {}/{}", .{ i + 1, output_tokens_len }));
//         }
//     }
//     // std.debug.print("\n", .{});
//
//     const end = std.time.microTimestamp();
//     if (opts.profile) |profile_file| {
//         profiler.stop();
//         try profiler.dumpAsJsonTo(allocator, std.fs.cwd(), profile_file);
//     }
//
//     {
//         const duration = stdx.math.divFloat(f64, prefill_end - prefill_start, std.time.us_per_s);
//         const speed = @as(f64, @floatFromInt(batch_size * prompt_tok.len)) / duration;
//         log.info("✅ Prefilled {d} tokens in {:.3}s: {d:.3}tok/s", .{ batch_size * prompt_tok.len, duration, speed });
//     }
//
//     {
//         const duration = stdx.math.divFloat(f64, end - start, std.time.us_per_s);
//         const speed = @as(f64, @floatFromInt(batch_size * output_tokens_len)) / duration;
//         log.info("✅ Generated {d} tokens in {:.3}s: {d:.3}tok/s", .{ batch_size * output_tokens_len, duration, speed });
//     }
//
//     {
//         const duration = stdx.math.divFloat(f64, end - prefill_start, std.time.us_per_s);
//         const speed = @as(f64, @floatFromInt(batch_size * output_tokens_len)) / duration;
//         log.info("✅ Observed throughput from user perspective (including prefilling latency) num_tokeks={d} duration={d}s {d:.3}tok/s", .{ batch_size * output_tokens_len, duration, speed });
//     }
//
//     if (opts.benchmark) std.process.exit(0);
//
//     _ = try tokens.toHost(std.mem.sliceAsBytes(token_buffer));
//     for (0..@min(batch_size, 4)) |b| {
//         output.clearRetainingCapacity();
//         const seq = token_buffer[b * max_seq_len ..][0..max_seq_len];
//         const end_index = std.mem.indexOfScalar(i32, seq, 128001) orelse max_seq_len;
//         try tokenizer.decodeWithOpts(&output, @ptrCast(seq[0..end_index]), .{});
//         std.debug.print("Generation: {}\n{s}\n", .{ b, output.items });
//     }
//
//     return output.toOwnedSlice();
// }

const params = clap.parseParamsComptime(
    \\--help                    print this help
    \\--prompt <STRING>         the prompt
    \\--model-config <PATH>     config.json path
    \\--model-name <STRING>     model name
    \\--model-weights <PATH>    model weights path
    \\--model-tokenizer <PATH>  tokenizer path
    \\--seq-len <UINT>          sequence length
);

pub fn bool_parser(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    log.info("   LLama was compiled with {}", .{@import("builtin").mode});

    const allocator = std.heap.c_allocator;

    const parsers = comptime .{
        .BOOL = bool_parser,
        .UINT = clap.parsers.int(usize, 0),
        .STRING = clap.parsers.string,
        .PATH = clap.parsers.string,
    };
    var diag: clap.Diagnostic = .{};
    const stderr = std.io.getStdErr().writer();
    var res = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        diag.report(stderr, err) catch {};
        stderr.print("usage: ", .{}) catch {};
        clap.usage(stderr, clap.Help, &params) catch {};
        stderr.print("\n", .{}) catch {};
        return;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        clap.help(std.io.getStdErr().writer(), clap.Help, &params, .{}) catch {};
        return;
    }

    const config = blk: {
        if (res.args.@"model-config") |config_json_path| {
            var config_json_file = try asynk.File.open(config_json_path, .{ .mode = .read_only });
            defer config_json_file.close() catch unreachable;
            var reader = std.json.reader(allocator, config_json_file.reader());
            defer reader.deinit();
            const config_obj = try std.json.parseFromTokenSourceLeaky(llama.LlamaLM.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
            break :blk config_obj;
        } else {
            log.err("Missing --model-config", .{});
            return;
        }
    };

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/llama",
        .sharding_enabled = true,
    };

    // TODO: add --create-opts <PATH> json string/file with creation options
    // const create_opts = try std.json.parseFromSlice(zml.Platform.CreateOptions, allocator, cli_args.create_options, .{});
    // const platform = context.autoPlatform(create_opts.value).withCompilationOptions(compilation_options);
    // create_opts.deinit();
    const platform = context.autoPlatform(.{}).withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    var ts = try zml.aio.detectFormatAndOpen(allocator, res.args.@"model-weights".?);
    defer ts.deinit();

    var model_arena = std.heap.ArenaAllocator.init(allocator);
    var model_instance = try zml.aio.populateModel(llama.LlamaLM, model_arena.allocator(), ts);
    model_instance = model_instance; // autofix

    const llama_options: llama.LlamaLM.Options = .{
        .max_seq_len = @intCast(res.args.@"seq-len" orelse 256),
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };
    model_instance.init(config, llama_options);

    const dims = model_instance.model.shape();
    const dtype = model_instance.model.embed_tokens.weight.dtype();

    const batch_size = 1;

    // const tokens_shape = zml.Shape.init(.{ .b = batch_size, .s = llama_options.chunked_prefill_size }, .u32);
    const tokens_shape_prefill = zml.Shape.init(.{ .b = batch_size, .s = llama_options.max_seq_len }, .u32);
    const tokens_shape = zml.Shape.init(.{ .b = batch_size, .s = 1 }, .u32);
    const token_idx_shape = zml.Shape.init(.{ .b = batch_size }, .u32);

    // const kv_shape = zml.Shape.init(.{ .layer = model_instance.model.layers.len, .b = bs, .h = dims.nkvh, .k = dims.s, .hd = dims.hd }, dtype).withSharding(.{.h});
    const kv_shape = zml.Shape.init(.{ .layer = model_instance.model.layers.len, .b = batch_size, .k = dims.s, .h = dims.nkvh, .hd = dims.hd }, dtype).withSharding(.{.h});

    // needs to be optional
    const kv_cache_shape: zml.ShapeOf(llama.KvCache) = llama.KvCache.initShape(kv_shape);
    const rng_shape = zml.Tensor.Rng.shape();

    var start = try std.time.Timer.start();
    var fut_mod_prefill = try asynk.asyncc(zml.compile, .{ allocator, llama.LlamaLM.forward, .{ config, llama_options }, .{ tokens_shape_prefill, token_idx_shape, kv_cache_shape, rng_shape }, ts, platform });

    var fut_mod = try asynk.asyncc(zml.compile, .{ allocator, llama.LlamaLM.forward, .{ config, llama_options }, .{ tokens_shape, token_idx_shape, kv_cache_shape, rng_shape }, ts, platform });

    log.info("\tLoading Llama weights from {?s}...", .{res.args.@"model-weights"});
    var llama_weights = try zml.aio.loadBuffers(llama.LlamaLM, .{ config, llama_options }, ts, model_arena.allocator(), platform);
    defer zml.aio.unloadBuffers(&llama_weights);
    log.info("✅\tLoaded weights in {}", .{std.fmt.fmtDuration(start.read())});

    var llama_module_prefill = (try fut_mod_prefill.awaitt()).prepare(llama_weights);
    defer llama_module_prefill.deinit();
    var llama_module = (try fut_mod.awaitt()).prepare(llama_weights);
    defer llama_module.deinit();
    log.info("✅\tCompiled model in {}", .{std.fmt.fmtDuration(start.read())});

    log.info("Creating KvCache", .{});
    const kv_cache = try llama.KvCache.initBuffer(kv_shape, platform);
    _ = kv_cache; // autofix

    log.info("✅\tPrompt: {s}", .{res.args.prompt});

    // const story = try generateText(llama, llama_module_prefill, llama_module, tokenizer, allocator, cli_args);
    // defer allocator.free(story);
}
