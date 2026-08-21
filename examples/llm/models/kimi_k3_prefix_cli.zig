const std = @import("std");

const c = @import("c");

const zml = @import("zml");
const common = @import("common.zig");
const chat_template = @import("kimi_k3/chat_template.zig");
const inference = @import("kimi_k3/inference.zig");
const model = @import("kimi_k3/model.zig");
const session_impl = @import("kimi_k3/session.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const layer_count: usize = 4;
const maximum_context: usize = 4096;
const maximum_generation: usize = 32;

const Args = struct {
    weights: []const u8,
    tokenizer: []const u8,
    prompt: []const u8,
    max_new_tokens: usize = 1,
    context_limit: usize = 512,

    pub const help =
        \\Use kimi_k3_prefix_cli --weights=<S4-directory> --tokenizer=<tokenizer.json> --prompt=<text> [options]
        \\
        \\Run deterministic greedy decoding through the diagnostic four-layer
        \\Kimi K3 prefix on NVIDIA CUDA GPU 0.
        \\
        \\Options:
        \\  --max-new-tokens=<1..32>  Maximum generated tokens (default: 1)
        \\  --context-limit=<1..4096>  Prompt plus generation bound (default: 512)
        \\
        \\WARNING: Four layers are a development diagnostic, not reliable full-model answers.
        \\
    ;
};

fn enableDeterministicCuda(allocator: std.mem.Allocator) !void {
    const deterministic_flag = "--xla_gpu_deterministic_ops=true";
    if (std.c.getenv("XLA_FLAGS")) |existing_z| {
        const existing = std.mem.span(existing_z);
        if (std.mem.indexOf(u8, existing, deterministic_flag) == null) {
            const combined = try std.fmt.allocPrintSentinel(
                allocator,
                "{s} {s}",
                .{ existing, deterministic_flag },
                0,
            );
            defer allocator.free(combined);
            if (c.setenv("XLA_FLAGS", combined.ptr, 1) != 0) return error.KimiK3DeterministicXlaSetupFailed;
        }
    } else if (c.setenv("XLA_FLAGS", deterministic_flag, 1) != 0) {
        return error.KimiK3DeterministicXlaSetupFailed;
    }
    if (c.setenv("TF_DETERMINISTIC_OPS", "1", 0) != 0) return error.KimiK3DeterministicCudaSetupFailed;
}

fn elapsedUs(io: std.Io, started: i96) i96 {
    return @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - started, 1000);
}

fn validateArgs(args: Args) !void {
    if (std.mem.trim(u8, args.prompt, " \t\r\n").len == 0) return error.EmptyKimiK3Prompt;
    if (!std.unicode.utf8ValidateSlice(args.prompt)) return error.InvalidKimiK3PromptUtf8;
    if (args.max_new_tokens == 0 or args.max_new_tokens > maximum_generation) {
        return error.InvalidKimiK3GenerationLimit;
    }
    if (args.context_limit == 0 or args.context_limit > maximum_context) {
        return error.InvalidKimiK3ContextLimit;
    }
}

fn initPrefixModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo: std.Io.Dir,
    store: zml.io.TensorStore.View,
) !model.LoadedModel {
    const parsed = try common.parseConfig(model.Config, allocator, io, repo);
    errdefer parsed.deinit();
    const inner = try model.Model.initSelected(
        allocator,
        store,
        parsed.value,
        .{ .max_seq_len = @intCast(maximum_context) },
        .{ .layer_limit = layer_count },
    );
    return .{ .inner = inner, .parsed_config = parsed };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    try validateArgs(args);

    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, args.tokenizer);
    defer tokenizer.deinit();
    const prompt_tokens = try chat_template.tokenizePrompt(allocator, tokenizer, args.prompt);
    defer allocator.free(prompt_tokens);
    const sequence_capacity = std.math.add(usize, prompt_tokens.len, args.max_new_tokens) catch {
        return error.KimiK3ContextCapacityOverflow;
    };
    if (sequence_capacity > args.context_limit) return error.KimiK3PromptExceedsContextLimit;

    try enableDeterministicCuda(allocator);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.90 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;
    if (platform.devices.len != 1) return error.KimiK3PrefixCliRequiresOneVisibleGpu;

    const repo = try zml.safetensors.resolveModelRepo(io, args.weights);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    var loaded_model = try initPrefixModel(allocator, io, repo, store.view());
    defer loaded_model.deinit(allocator);
    const shardings: common.Shardings = try .init(platform);
    var progress = std.Progress.start(io, .{ .root_name = "Kimi K3 four-layer diagnostic" });
    defer progress.end();

    const compile_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var compiled: inference.CompiledModel = try loaded_model.compile(
        allocator,
        io,
        platform,
        .vanilla,
        shardings,
        sequence_capacity,
        &progress,
    );
    defer compiled.deinit();
    const compile_us = elapsedUs(io, compile_started);

    const load_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var buffers = try loaded_model.loadPrefixBuffers(allocator, io, platform, &store, &progress, shardings);
    defer loaded_model.unloadBuffers(&buffers, allocator);
    const load_us = elapsedUs(io, load_started);

    var session = try session_impl.Session.init(allocator, io, platform, tokenizer, &compiled, &buffers);
    const resident_load_stats = buffers.load_stats.*;
    defer session.deinit();
    var history = try std.ArrayList(u32).initCapacity(allocator, sequence_capacity);
    defer history.deinit(allocator);
    try history.appendSlice(allocator, prompt_tokens);

    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    const stdout = &stdout_file.interface;
    try stdout.writeAll(
        "KIMI_K3_PREFIX_DIAGNOSTIC_WARNING layers=4 reliable_answer=false " ++
            "message=This_is_not_full_model_Kimi_K3_inference\n",
    );
    try stdout.print(
        "KIMI_K3_PREFIX_PROMPT tokens={} capacity={} max_new_tokens={} ids={any}\n",
        .{ prompt_tokens.len, sequence_capacity, args.max_new_tokens, prompt_tokens },
    );
    try stdout.flush();

    const prompt_started = std.Io.Clock.now(.real, io).toNanoseconds();
    try session.runPrefill(prompt_tokens);
    const prefill_us = elapsedUs(io, prompt_started);
    var decoded: std.Io.Writer.Allocating = .init(allocator);
    defer decoded.deinit();
    const decode_started = std.Io.Clock.now(.real, io).toNanoseconds();
    try session.runDecode(&history, &decoded.writer);
    const decode_us = elapsedUs(io, decode_started);
    if (!std.meta.eql(resident_load_stats, buffers.load_stats.*)) return error.KimiK3ResidentWeightsReloaded;

    const generated = history.items[prompt_tokens.len..];
    const eos = tokenizer.tokenId("<|end_of_msg|>") orelse return error.KimiK3MissingEosToken;
    const stopped: []const u8 = if (generated.len < args.max_new_tokens and session.last_generated_token == eos)
        "eos"
    else
        "limit";
    try stdout.print(
        "KIMI_K3_PREFIX_GENERATED tokens={} ids={any} stopped={s}\n",
        .{ generated.len, generated, stopped },
    );
    try stdout.writeAll("KIMI_K3_PREFIX_TEXT_BEGIN\n");
    try stdout.writeAll(decoded.written());
    try stdout.writeAll("\nKIMI_K3_PREFIX_TEXT_END\n");
    try stdout.print(
        "KIMI_K3_PREFIX_CLI_PASS backend=cuda device=0 layers=4 scope=diagnostic_prefix deterministic_ops=true " ++
            "prompt_tokens={} generated_tokens={} compile_us={} load_us={} prefill_us={} decode_us={} " ++
            "resident_layer_loads={} payload_reads={} payload_bytes={} steady_state_reloads=0\n",
        .{
            prompt_tokens.len,
            generated.len,
            compile_us,
            load_us,
            prefill_us,
            decode_us,
            resident_load_stats.layer_loads,
            resident_load_stats.payload_reads,
            resident_load_stats.payload_bytes,
        },
    );
    try stdout.flush();
}
