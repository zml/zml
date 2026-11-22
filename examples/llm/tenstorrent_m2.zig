//! M2 runner of the Tenstorrent backend (DESIGN 14.2): the whole of
//! examples/llm's Llama on the cards, judged at the token level against ZML's
//! CPU platform running the same BF16 model on the same prompts.
//!
//! Modes (one platform per process):
//!   reference  greedy generation on the platform (the CPU), recording at
//!              every position the top logits; written to --reference.
//!   compare    teacher forcing on the platform: the reference's tokens are
//!              fed back (the prompt through the prefill, the continuation
//!              one decode step at a time) and the argmax at every position
//!              is compared with the reference's. DESIGN 14.0's whole-model
//!              criterion: on the cards, exit 1 when fewer positions agree
//!              (the prompt's and the decode steps') than the CPU platform's
//!              own N-partition run agrees with its 1-partition reference
//!              (`--floor`: that run's record, teacher-forced on the same
//!              reference), or when a disagreement is not a near-tie
//!              (`rounding_ulps` in both runs). On the CPU the run measures
//!              that floor: only its disagreements are judged.
//!   generate   free-running generation with the programs examples/llm
//!              compiles (prefill and decode, its sampler), timed per step;
//!              with --reference, the tokens are compared with the
//!              reference's continuation.
//!
//! reference and compare run the example's embedding, decoder layers and LM
//! head (inference.Forward.layers and LmHead.project) with the logits as the
//! program's output instead of a sampled token: the prefill returns the
//! logits of every position, a decode step those of its one token. generate
//! runs inference.Forward.forward itself, as examples/llm's session does.

const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const common = @import("models/common.zig");
const llama = @import("models/llama.zig");
const pjrt_error_log = @import("pjrt_error_log.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = pjrt_error_log.logFn,
};

const log = std.log.scoped(.tenstorrent_m2);

const KvCache = llama.model.KvCache;

/// A disagreement is put down to BF16 rounding when, in both runs, the two tokens' logits are
/// within this many BF16 ulps (of the larger) of each other; anything wider is a bug to find.
const rounding_ulps = 2;
/// Logits kept per position.
const top_count = 8;

const Mode = enum { reference, compare, generate };

const Args = struct {
    model: []const u8,
    mode: Mode = .generate,
    platform: enum { cpu, cards } = .cards,
    /// Cards of a `cards` platform: the first ones of the default view.
    cards: u32 = 1,
    seqlen: u32 = 2048,
    /// Positions generated (reference, generate) or fed back (compare) after each prompt.
    tokens: u32 = 64,
    /// How many of the built-in prompts to run, from the first.
    prompts: u32 = prompt_texts.len,
    /// generate: this prompt instead of the built-in ones.
    prompt: ?[]const u8 = null,
    reference: ?[]const u8 = null,
    /// compare: also writes this run's records, in the reference's format.
    out: ?[]const u8 = null,
    /// compare on the cards: the record of the CPU platform on N partitions teacher-forced on the
    /// same reference (`--mode=compare --platform=cpu --cards=N --out=<file>`), whose agreement
    /// with the reference is the floor (DESIGN 14.0).
    floor: ?[]const u8 = null,
    /// generate: examples/llm's sampler (1: greedy argmax; examples/llm's default is 4).
    topk: u32 = 1,
    /// generate: runs of every prompt (timings are reported per run).
    runs: u32 = 1,
    /// Kernel JIT cache of the cards (default: the platform's).
    cache_dir: ?[]const u8 = null,
    /// Who picks the weights' device layouts: `executable` as the Llama example
    /// compiles (a matmul's weight in the order the matmul reads), or `default`
    /// (the layout of its shape), to measure the difference.
    weight_layouts: zml.Compiler.WeightLayouts = .executable,
    /// tt_dram_reserve_bytes in MiB (default: the plugin's).
    reserve_mib: ?u32 = null,

    pub const help =
        \\ Use tenstorrent_m2 --model=<dir> [options]
        \\
        \\ Runs examples/llm's Llama, prefill and decode, and judges it at the token level (DESIGN 14.2, M2).
        \\
        \\ Options:
        \\   --model=<dir>          Llama checkpoint directory (config.json, tokenizer.json, safetensors)
        \\   --mode=<mode>          reference: greedy generation with every position's top logits, written to
        \\                          --reference; compare: teacher forcing of --reference's tokens, top-1 agreement;
        \\                          generate: free-running generation with examples/llm's own programs (default)
        \\   --platform=<cpu|cards> Where to run (default: cards; run under the card lock)
        \\   --cards=<n>            Cards of the run, tensor-parallel over all of them (default 1)
        \\   --seqlen=<n>           Sequence length of the programs, a multiple of 32 (default 2048, as examples/llm)
        \\   --tokens=<n>           Positions generated or fed back after each prompt (default 64)
        \\   --prompts=<n>          Built-in prompts to run, from the first (default: all)
        \\   --prompt=<text>        generate: this prompt instead of the built-in ones
        \\   --reference=<file>     reference: output; compare: input; generate: tokens to compare with
        \\   --out=<file>           compare: also writes this run's records
        \\   --floor=<file>         compare on the cards: the CPU's N-partition record on the same reference (written by
        \\                          --mode=compare --platform=cpu --cards=N --out=<file>), the agreement floor
        \\   --topk=<n>             generate: examples/llm's sampler, 1 is greedy (default 1)
        \\   --runs=<n>             generate: runs of each prompt (default 1)
        \\   --cache-dir=<dir>      Kernel JIT cache of the cards
        \\   --weight-layouts=<executable|default>  Weights in the matmul's layout (default, as examples/llm) or the shape's
        \\   --reserve-mib=<n>      DRAM per card left to TTNN (tt_dram_reserve_bytes), in MiB
        \\
    ;
};

/// Prompts of different lengths: a question, an instruction, and two texts
/// to work on (about 30, 50, 380 and 1,100 tokens with the chat template).
const prompt_texts = [_][]const u8{
    "What is the capital of France?",
    "Explain, step by step, how a large language model produces the answer to a question. Keep it under two hundred words.",
    passage_printing ++ "\n\nSummarize this passage in three sentences.",
    passage_printing ++ "\n\n" ++ passage_memory ++ "\n\nGive these two passages a common title, then list the five ideas they share.",
};

const passage_printing =
    \\Before the printing press, every book in Europe was copied by hand. A monastery scriptorium could spend
    \\a year on a single Bible, and the copies drifted apart as scribes made their own mistakes and corrected
    \\those of others. Paper, which had reached Europe from China through the Islamic world, was becoming cheap
    \\in the fourteenth century, and a growing class of merchants, lawyers and students wanted books they could
    \\own. The bottleneck was labour, not material.
    \\
    \\Around 1440 in Mainz, Johannes Gutenberg combined several existing techniques into one system. He cast
    \\letters in a mould that could be adjusted for the width of each character, so that thousands of identical
    \\pieces of type could be made quickly from a hard alloy of lead, tin and antimony. He adapted the screw
    \\press used for wine and paper, and he developed an oil-based ink that would stick to metal. None of these
    \\ideas was entirely new; what mattered was that they worked together, and that a page, once set, could be
    \\printed hundreds of times with the same text.
    \\
    \\The consequences were fast and uneven. By 1500 presses operated in more than two hundred European towns,
    \\and perhaps twenty million volumes had been printed. Prices fell, standard editions appeared, and errors,
    \\once corrected in the type, stayed corrected. Scholars could compare the same page in different cities.
    \\Printers also discovered that pamphlets, calendars and grammars sold better than heavy theological works,
    \\and the book trade became a business that followed its readers rather than its patrons.
;

const passage_memory =
    \\A modern processor can perform an arithmetic operation in well under a nanosecond, but fetching a value
    \\from main memory takes a hundred nanoseconds or more. Computer architects have spent fifty years hiding
    \\that gap. The usual answer is a hierarchy: small and fast memories close to the arithmetic units, larger
    \\and slower ones further away. Registers hold the values in use right now, caches hold recently used data
    \\in the hope that it will be needed again, and main memory holds everything else.
    \\
    \\A hierarchy only helps when programs behave predictably. Most do: a loop touches the same variables again
    \\and again, and an array is read from one end to the other. Hardware exploits the first habit by keeping
    \\recent data and the second by fetching whole lines of neighbouring bytes, or by prefetching the next lines
    \\before they are requested. Software helps by arranging data so that what is used together is stored
    \\together, and by splitting large problems into blocks that fit in the faster levels.
    \\
    \\Accelerators for machine learning push the same ideas further. Instead of transparent caches they often
    \\give the programmer explicit scratchpad memories and engines that move data between levels, because the
    \\access pattern of a matrix multiplication is known in advance and can be scheduled exactly. The cost is
    \\complexity: a compiler or a library must decide which tile of which matrix sits where, and when it moves.
    \\The reward is that the arithmetic units are kept busy instead of waiting, which is where most of the
    \\speed of these machines comes from. The same reasoning explains why reading the weights of a large model
    \\from memory, rather than computing with them, usually sets the speed at which it generates text.
;

// ---------------------------------------------------------------------------
// The programs.

/// The arguments of the logits programs: those of inference.Forward without
/// the sampler's (active length, random state).
const LogitsInput = struct {
    weights: llama.model.Model,
    tokens: Tensor,
    token_index: Tensor,
    kv_cache: KvCache,
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,
};

const LogitsOutput = struct {
    /// `{.s, .voc}`: the prediction of the next token at every input position, replicated.
    logits: Tensor,
    kv_cache: KvCache,
};

/// examples/llm's embedding, decoder layers and LM head, without the sampler.
fn forwardLogits(input: LogitsInput) LogitsOutput {
    const hidden, const kv_cache = llama.inference.Forward.layers(.{
        .weights = input.weights,
        .tokens = input.tokens,
        .token_index = input.token_index,
        // Neither is read before the LM head.
        .active_length = undefined,
        .rng = undefined,
        .kv_cache = input.kv_cache,
        .attention_metadata = input.attention_metadata,
        .attention_parameters = input.attention_parameters,
    });
    const lm_head: llama.model.LmHead = .init(input.weights);
    const logits = lm_head.project(hidden.withPartialTags(.{ .s, .d }));
    return .{
        .logits = logits.withPartitioning(.model, .{ .s = .replicated, .voc = .replicated }),
        .kv_cache = kv_cache.reuseBuffer(input.kv_cache),
    };
}

const LogitsExe = zml.FnExe(forwardLogits);
const LogitsRunner = LogitsExe.Runner(.{.weights});

// ---------------------------------------------------------------------------
// Records.

/// One position: the logits predicting the next token, highest first (ties: lowest id first).
const Top = struct {
    ids: [top_count]u32,
    values: [top_count]f32,
};

const PromptRecord = struct {
    text: []const u8,
    /// The prompt's tokens, then the continuation.
    tokens: []const u32,
    prompt_len: u32,
    /// One per position fed: the prompt's positions (the prefill), then one per decode step.
    tops: []const Top,
};

const Record = struct {
    model: []const u8,
    platform: []const u8,
    seqlen: u32,
    prompts: []const PromptRecord,
};

/// The `top_count` largest values of `row`.
fn topOf(row: []const f32) Top {
    var top: Top = .{ .ids = @splat(0), .values = @splat(-std.math.inf(f32)) };
    var filled: usize = 0;
    for (row, 0..) |value, i| {
        if (std.math.isNan(value)) continue;
        if (filled == top_count and value <= top.values[top_count - 1]) continue;
        // Insertion: strictly greater values move ahead, so equal ones keep the lower id first.
        var at: usize = if (filled < top_count) filled else top_count - 1;
        while (at > 0 and value > top.values[at - 1]) : (at -= 1) {
            top.values[at] = top.values[at - 1];
            top.ids[at] = top.ids[at - 1];
        }
        top.values[at] = value;
        top.ids[at] = @intCast(i);
        if (filled < top_count) filled += 1;
    }
    return top;
}

test {
    // The runfiles initializer every Bazel Zig binary links calls zml.bazel's
    // export, which only a reference brings into a test build.
    _ = zml.bazel.runfiles;
}

test topOf {
    var row: [20]f32 = @splat(0);
    row[3] = 5;
    row[7] = 5;
    row[11] = 9;
    row[2] = -1;
    const top = topOf(&row);
    try std.testing.expectEqual(11, top.ids[0]);
    try std.testing.expectEqual(3, top.ids[1]);
    try std.testing.expectEqual(7, top.ids[2]);
    try std.testing.expectEqual(0, top.ids[3]);
    try std.testing.expectEqual(1, top.ids[4]);
    try std.testing.expectEqual(@as(f32, 0), top.values[7]);
}

/// One BF16 unit in the last place at the magnitude of `x`.
fn bf16Ulp(x: f32) f32 {
    const magnitude = @abs(x);
    if (magnitude == 0 or !std.math.isFinite(magnitude)) return std.math.floatMin(f32);
    const exponent = std.math.floor(std.math.log2(magnitude));
    return std.math.exp2(exponent - 7);
}

test bf16Ulp {
    try std.testing.expectEqual(@as(f32, 0.125), bf16Ulp(17.5));
    try std.testing.expectEqual(@as(f32, 0.0625), bf16Ulp(-12));
    try std.testing.expectEqual(@as(f32, 1.0 / 128.0), bf16Ulp(1));
}

/// The value of `id` among a position's top logits, if it is there.
fn valueOf(top: Top, id: u32) ?f32 {
    for (top.ids, top.values) |i, v| if (i == id) return v;
    return null;
}

// ---------------------------------------------------------------------------
// The run.

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    args: Args,
    platform: *zml.Platform,
    config: llama.Config,
    tokenizer: *zml.tokenizer.Tokenizer,
    /// Positions on the device: token_index[p] holds p, as examples/llm's session keeps them.
    token_index: []zml.Buffer,
    kv: zml.Bufferized(KvCache),
    metadata: zml.Bufferized(zml.attention.Metadata),
    params: llama.inference.CompilationParameters,
    timings: Timings = .{},
};

const Timings = struct {
    open_ms: f64 = 0,
    compile_ms: [2]f64 = @splat(0),
    load_ms: f64 = 0,
    /// Bytes of the checkpoint's tensors.
    weight_bytes: u64 = 0,
    bytes_after_load: u64 = 0,
    bytes_after_run: u64 = 0,
    peak_bytes: ?u64 = null,
    bytes_limit: ?u64 = null,

    fn recordMemory(self: *Timings, platform: *const zml.Platform, after: enum { load, run }) void {
        for (platform.devices) |device| {
            const stats = device.memoryStats();
            switch (after) {
                .load => self.bytes_after_load = @max(self.bytes_after_load, stats.bytes_in_use),
                .run => self.bytes_after_run = @max(self.bytes_after_run, stats.bytes_in_use),
            }
            if (stats.peak_bytes_in_use) |peak| self.peak_bytes = @max(self.peak_bytes orelse 0, peak);
            if (stats.bytes_limit) |limit| self.bytes_limit = @max(self.bytes_limit orelse 0, limit);
        }
    }
};

fn msSince(io: std.Io, from: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(from.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_ms;
}

fn mib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 20);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(io, build_working_directory, .{});
        defer working_dir.close(io);
        try std.process.setCurrentDir(io, working_dir);
    }
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if (args.seqlen % 32 != 0) {
        log.err("--seqlen={d}: Tenstorrent tiles positions 32 at a time", .{args.seqlen});
        return error.InvalidArgument;
    }
    if (args.mode != .generate and args.reference == null) {
        log.err("--mode={s} needs --reference=<file>", .{@tagName(args.mode)});
        return error.InvalidArgument;
    }

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    defer repo.close(io);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var tokenizer = try loadTokenizer(allocator, io, repo);
    defer tokenizer.deinit();

    const open_start: std.Io.Timestamp = .now(io, .awake);
    const platform = try openPlatform(allocator, io, args);
    defer platform.deinit(allocator, io);
    const open_ms = msSince(io, open_start);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const meshes: common.Meshes = try .init(platform);
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &meshes.all());
    defer store.deinit();

    const generation: common.GenerationOptions = .{ .sampling_strategy = .{ .topk = args.topk } };
    var loaded: llama.LoadedModel = try .init(allocator, io, repo, store.view(), generation);
    defer loaded.deinit(allocator);
    const config = loaded.parsed_config.value;

    const backend: zml.attention.Backend = .auto(platform);
    var params: llama.inference.CompilationParameters = .init(loaded.inner, config, args.seqlen, backend, meshes, platform.target);
    params.weight_layouts = args.weight_layouts;

    var ctx: Context = .{
        .allocator = allocator,
        .io = io,
        .args = args,
        .platform = platform,
        .config = config,
        .tokenizer = &tokenizer,
        .token_index = undefined,
        .kv = undefined,
        .metadata = undefined,
        .params = params,
        .timings = .{ .open_ms = open_ms },
    };
    for (registry.tensors.values()) |tensor| ctx.timings.weight_bytes += tensor.shape.byteSize();

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
    const stdout = &stdout_writer.interface;

    const ok = switch (args.mode) {
        .reference, .compare => try runLogits(&ctx, &loaded, &store, stdout),
        .generate => try runGenerate(&ctx, &loaded, &store, stdout),
    };
    try stdout.flush();
    if (!ok) std.process.exit(1);
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const bytes = try dir.readFileAlloc(io, "tokenizer.json", allocator, .limited(64 << 20));
    defer allocator.free(bytes);
    return try .fromBytes(allocator, bytes);
}

fn openPlatform(allocator: std.mem.Allocator, io: std.Io, args: Args) !*zml.Platform {
    switch (args.platform) {
        .cpu => return zml.Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = args.cards } }),
        .cards => {
            // The first cards of the default view, alone: a view of `args.cards`
            // cards whatever else this host has.
            var chips_buf: [zml.Platform.MAX_NUM_DEVICES * 21]u8 = undefined;
            var chips: std.Io.Writer = .fixed(&chips_buf);
            {
                const all: *zml.Platform = try .init(allocator, io, .tenstorrent, .{
                    .tenstorrent = .{ .cache_dir = args.cache_dir },
                });
                defer all.deinit(allocator, io);
                if (all.devices.len < args.cards) {
                    log.err("this host has {d} cards, --cards={d}", .{ all.devices.len, args.cards });
                    return error.NotEnoughCards;
                }
                for (all.devices[0..args.cards], 0..) |device, i| {
                    const info: zml.platform.tenstorrent.DeviceInfo = try .init(all, device);
                    try chips.print("{s}{d}", .{ if (i == 0) "" else ",", info.board_id });
                }
            }
            return zml.Platform.init(allocator, io, .tenstorrent, .{
                .tenstorrent = .{
                    .visible_chips = chips.buffered(),
                    .cache_dir = args.cache_dir,
                    .dram_reserve_bytes = if (args.reserve_mib) |m| @as(i64, m) << 20 else null,
                },
            });
        },
    }
}

/// The chat template of examples/llm's session (llama/session.zig, tokenizePrompt).
fn tokenizePrompt(ctx: *const Context, allocator: std.mem.Allocator, prompt: []const u8) ![]u32 {
    var encoder = try ctx.tokenizer.encoder();
    defer encoder.deinit();
    const start_header = ctx.tokenizer.tokenId("<|start_header_id|>") orelse return error.NoSuchToken;
    const end_header = ctx.tokenizer.tokenId("<|end_header_id|>") orelse return error.NoSuchToken;
    const eot = ctx.tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;
    const newline = ctx.tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

    var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
    errdefer tokens.deinit();
    const w: *std.Io.Writer = &tokens.writer;
    try encoder.appendTokens(w, &.{ ctx.config.bos_token_id, start_header });
    try encoder.encode(w, "user");
    try encoder.appendTokens(w, &.{ end_header, newline });
    try encoder.encode(w, prompt);
    try encoder.appendTokens(w, &.{ eot, newline, start_header });
    try encoder.encode(w, "assistant");
    try encoder.appendTokens(w, &.{ end_header, newline });
    return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
}

fn isEos(config: llama.Config, token: u32) bool {
    return switch (config.eos_token_id) {
        .int => |eos| token == eos,
        .ints => |list| std.mem.indexOfScalar(u32, list, token) != null,
    };
}

/// Weights, cache, metadata and the device positions, after the programs compiled; each
/// weight in the layout `exes` take it in (`zml.Exe.commonModelLayouts`).
fn loadState(ctx: *Context, loaded: *const llama.LoadedModel, store: *zml.io.TensorStore, exes: []const *const zml.Exe) !llama.Buffers {
    var progress: std.Progress.Node = .none;
    var arena: std.heap.ArenaAllocator = .init(ctx.allocator);
    defer arena.deinit();
    const layouts = try zml.Exe.commonModelLayouts(arena.allocator(), exes, &loaded.inner);
    const start: std.Io.Timestamp = .now(ctx.io, .awake);
    const buffers = try common.loadBuffers(loaded, ctx.allocator, ctx.io, ctx.platform, store, &progress, layouts);
    ctx.timings.load_ms = msSince(ctx.io, start);
    ctx.kv = try ctx.params.kv_cache.initBuffer(ctx.io, ctx.platform);
    ctx.metadata = try ctx.params.attention_metadata.initBuffer(ctx.io, ctx.platform);
    ctx.token_index = try ctx.allocator.alloc(zml.Buffer, ctx.args.seqlen);
    for (ctx.token_index, 0..) |*b, i| b.* = try .scalar(ctx.io, ctx.platform, i, .u32);
    ctx.timings.recordMemory(ctx.platform, .load);
    return buffers;
}

fn unloadState(ctx: *Context) void {
    for (ctx.token_index) |*b| b.deinit();
    ctx.allocator.free(ctx.token_index);
    KvCache.deinitBuffer(&ctx.kv);
    zml.attention.Metadata.deinitBuffer(&ctx.metadata);
}

/// The first --prompts built-in prompts, or --prompt (held in `one`).
fn promptList(ctx: *const Context, one: *[1][]const u8) []const []const u8 {
    if (ctx.args.prompt) |p| {
        one[0] = p;
        return one;
    }
    return prompt_texts[0..@min(ctx.args.prompts, prompt_texts.len)];
}

// ---------------------------------------------------------------------------
// reference and compare.

fn runLogits(ctx: *Context, loaded: *const llama.LoadedModel, store: *zml.io.TensorStore, stdout: *std.Io.Writer) !bool {
    const allocator = ctx.allocator;
    const io = ctx.io;

    var reference: ?std.json.Parsed(Record) = null;
    defer if (reference) |*r| r.deinit();
    if (ctx.args.mode == .compare) {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ctx.args.reference.?, allocator, .limited(1 << 30));
        defer allocator.free(bytes);
        reference = try std.json.parseFromSlice(Record, allocator, bytes, .{ .allocate = .alloc_always });
        if (reference.?.value.seqlen != ctx.args.seqlen) {
            log.err("the reference ran with --seqlen={d}", .{reference.?.value.seqlen});
            return error.InvalidArgument;
        }
    }

    var exes: [2]LogitsExe = undefined;
    for (&exes, [_]common.Phase{ .prefill, .decode }, 0..) |*exe, phase, i| {
        const tokens: i64 = if (phase == .prefill) ctx.args.seqlen else 1;
        const start: std.Io.Timestamp = .now(io, .awake);
        exe.* = try LogitsExe.compile(allocator, io, ctx.platform, .{
            .meshes = &ctx.params.meshes.all(),
            .program_name = phase.programName("llama", "logits"),
            .weight_layouts = ctx.params.weight_layouts,
        }, .{.{
            .weights = loaded.inner,
            .tokens = Tensor.init(.{ .s = tokens }, .u32),
            .token_index = ctx.params.token_index,
            .kv_cache = ctx.params.kv_cache,
            .attention_metadata = ctx.params.attention_metadata,
            .attention_parameters = if (phase == .prefill) ctx.params.prefill_attention_parameters else ctx.params.decode_attention_parameters,
        }});
        ctx.timings.compile_ms[i] = msSince(io, start);
        log.info("compiled the {s} logits program in {d:.0} ms (PJRT compile, Load, warm-up)", .{ @tagName(phase), ctx.timings.compile_ms[i] });
    }
    defer for (&exes) |*exe| exe.deinit();

    var weights = try loadState(ctx, loaded, store, &.{ &exes[0].raw, &exes[1].raw });
    defer loaded.unloadBuffers(&weights, allocator);
    defer unloadState(ctx);
    log.info("loaded the weights in {d:.0} ms", .{ctx.timings.load_ms});

    var runners: [2]LogitsRunner = undefined;
    for (&runners, &exes) |*r, *exe| r.* = try .init(exe, allocator, .{ .weights = weights });
    defer for (&runners) |*r| r.deinit(allocator);

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    var one: [1][]const u8 = undefined;
    const texts: []const []const u8 = if (reference) |r| blk: {
        const list = try arena.allocator().alloc([]const u8, r.value.prompts.len);
        for (list, r.value.prompts) |*t, p| t.* = p.text;
        break :blk list;
    } else promptList(ctx, &one);

    var records: std.ArrayList(PromptRecord) = .empty;
    var prefill_ms: std.ArrayList(f64) = .empty;
    defer prefill_ms.deinit(allocator);
    var decode_ms: std.ArrayList(f64) = .empty;
    defer decode_ms.deinit(allocator);
    for (texts, 0..) |text, p| {
        const ref: ?PromptRecord = if (reference) |r| r.value.prompts[p] else null;
        const prompt: []const u32 = if (ref) |r| r.tokens[0..r.prompt_len] else try tokenizePrompt(ctx, arena.allocator(), text);
        const continuation: usize = if (ref) |r| r.tokens.len - r.prompt_len - 1 else ctx.args.tokens;
        if (prompt.len + continuation + 1 > ctx.args.seqlen) {
            log.err("prompt {d}: {d} + {d} tokens do not fit --seqlen={d}", .{ p, prompt.len, continuation, ctx.args.seqlen });
            return error.PromptTooLong;
        }
        const tokens = try arena.allocator().alloc(u32, prompt.len + continuation + 1);
        @memcpy(tokens[0..prompt.len], prompt);
        const tops = try arena.allocator().alloc(Top, prompt.len + continuation);

        // The prompt, through the prefill: one prediction per position.
        {
            const slice: zml.Slice = try .alloc(allocator, .init(.{ctx.args.seqlen}, .u32));
            defer slice.free(allocator);
            @memset(slice.items(u32), 0);
            @memcpy(slice.items(u32)[0..prompt.len], prompt);
            var tokens_buffer: zml.Buffer = try .fromSlice(io, ctx.platform, slice);
            defer tokens_buffer.deinit();
            var logits: zml.Buffer = undefined;
            const start: std.Io.Timestamp = .now(io, .awake);
            try runners[0].tryRun(io, .{
                .inputs = .{
                    .tokens = tokens_buffer,
                    .token_index = ctx.token_index[0],
                    .kv_cache = ctx.kv,
                    .attention_metadata = ctx.metadata,
                },
                .outputs = .{ .logits = &logits, .kv_cache = &ctx.kv },
            }, .{ .wait = true });
            try prefill_ms.append(allocator, msSince(io, start));
            defer logits.deinit();
            try readTops(allocator, io, logits, tops[0..prompt.len]);
        }
        tokens[prompt.len] = if (ref) |r| r.tokens[prompt.len] else tops[prompt.len - 1].ids[0];

        // The continuation, one decode step per position.
        var token_buffer: zml.Buffer = undefined;
        for (0..continuation) |step| {
            const position = prompt.len + step;
            token_buffer = try .fromBytes(io, ctx.platform, .init(.{ .s = 1 }, .u32), std.mem.asBytes(&tokens[position]));
            defer token_buffer.deinit();
            var logits: zml.Buffer = undefined;
            const start: std.Io.Timestamp = .now(io, .awake);
            try runners[1].tryRun(io, .{
                .inputs = .{
                    .tokens = token_buffer,
                    .token_index = ctx.token_index[position],
                    .kv_cache = ctx.kv,
                    .attention_metadata = ctx.metadata,
                },
                .outputs = .{ .logits = &logits, .kv_cache = &ctx.kv },
            }, .{ .wait = true });
            try decode_ms.append(allocator, msSince(io, start));
            defer logits.deinit();
            try readTops(allocator, io, logits, tops[position .. position + 1]);
            tokens[position + 1] = if (ref) |r| r.tokens[position + 1] else tops[position].ids[0];
        }
        log.info("prompt {d}: {d} prompt positions, {d} decode steps", .{ p, prompt.len, continuation });
        try records.append(arena.allocator(), .{
            .text = text,
            .tokens = tokens,
            .prompt_len = @intCast(prompt.len),
            .tops = tops,
        });
    }
    ctx.timings.recordMemory(ctx.platform, .run);

    const record: Record = .{
        .model = ctx.args.model,
        .platform = platformName(ctx),
        .seqlen = ctx.args.seqlen,
        .prompts = records.items,
    };
    const path = if (ctx.args.mode == .reference) ctx.args.reference else ctx.args.out;
    if (path) |file| {
        const json = try std.json.Stringify.valueAlloc(allocator, record, .{});
        defer allocator.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = json });
        log.info("wrote {s}", .{file});
    }

    try writeTimings(ctx, stdout, prefill_ms.items, decode_ms.items);
    if (reference) |r| return try writeComparison(ctx, stdout, r.value, record);
    try writeGenerations(ctx, stdout, record);
    return true;
}

fn platformName(ctx: *const Context) []const u8 {
    return switch (ctx.args.platform) {
        .cpu => if (ctx.args.cards == 1) "cpu" else "cpu (several devices)",
        .cards => if (ctx.args.cards == 1) "cards1" else if (ctx.args.cards == 2) "cards2" else "cards",
    };
}

/// The top logits of each row of `logits` `{.s, .voc}`, for the first `tops.len` rows.
fn readTops(allocator: std.mem.Allocator, io: std.Io, logits: zml.Buffer, tops: []Top) !void {
    const slice = try logits.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    // {.s, .voc}
    const voc: usize = @intCast(slice.shape.dim(slice.shape.rank() - 1));
    const row = try allocator.alloc(f32, voc);
    defer allocator.free(row);
    for (tops, 0..) |*top, r| {
        switch (slice.shape.dtype()) {
            .bf16 => for (slice.constItems(zml.floats.BFloat16)[r * voc ..][0..voc], row) |b, *f| {
                f.* = b.toF32();
            },
            .f32 => @memcpy(row, slice.constItems(f32)[r * voc ..][0..voc]),
            else => return error.UnexpectedLogitsType,
        }
        top.* = topOf(row);
    }
}

fn writeTimings(ctx: *const Context, w: *std.Io.Writer, prefill_ms: []const f64, decode_ms: []f64) !void {
    const t = ctx.timings;
    try w.print("\n{s}: open {d:.0} ms, compile + Load {d:.0} ms (prefill) and {d:.0} ms (decode), weights {d:.0} MiB in {d:.0} ms ({d:.2} GB/s)\n", .{
        platformName(ctx),                                           t.open_ms, t.compile_ms[0], t.compile_ms[1], mib(t.weight_bytes), t.load_ms,
        @as(f64, @floatFromInt(t.weight_bytes)) / (t.load_ms * 1e6),
    });
    try w.print("device memory in use (largest device): {d:.1} MiB after load, {d:.1} MiB after the run", .{ mib(t.bytes_after_load), mib(t.bytes_after_run) });
    if (t.bytes_limit) |limit| try w.print(", limit {d:.1} MiB", .{mib(limit)});
    try w.writeAll("\n");
    if (prefill_ms.len > 0) {
        try w.writeAll("prefill launches (ms):");
        for (prefill_ms) |ms| try w.print(" {d:.1}", .{ms});
        try w.writeAll("\n");
    }
    if (decode_ms.len > 0) {
        const s = summarize(decode_ms);
        try w.print("decode steps, launch to logits on the host: median {d:.2} ms, mean {d:.2}, min {d:.2}, max {d:.2} ({d} steps)\n", .{ s.median, s.mean, s.min, s.max, decode_ms.len });
    }
}

const Stats = struct { median: f64, mean: f64, min: f64, max: f64 };

fn summarize(values: []f64) Stats {
    if (values.len == 0) return .{ .median = 0, .mean = 0, .min = 0, .max = 0 };
    std.mem.sort(f64, values, {}, std.sort.asc(f64));
    var sum: f64 = 0;
    for (values) |v| sum += v;
    const n = values.len;
    const median = if (n % 2 == 1) values[n / 2] else (values[n / 2 - 1] + values[n / 2]) / 2;
    return .{ .median = median, .mean = sum / @as(f64, @floatFromInt(n)), .min = values[0], .max = values[n - 1] };
}

test summarize {
    var v = [_]f64{ 4, 1, 3, 2 };
    const s = summarize(&v);
    try std.testing.expectEqual(2.5, s.median);
    try std.testing.expectEqual(2.5, s.mean);
    try std.testing.expectEqual(1, s.min);
    try std.testing.expectEqual(4, s.max);
}

fn writeGenerations(ctx: *const Context, w: *std.Io.Writer, record: Record) !void {
    for (record.prompts, 0..) |p, i| {
        try w.print("\nprompt {d} ({d} tokens), greedy continuation of {d} tokens:\n", .{ i, p.prompt_len, p.tokens.len - p.prompt_len });
        try writeText(ctx, w, p.tokens[p.prompt_len..]);
        try w.writeAll("\n");
    }
}

fn writeText(ctx: *const Context, w: *std.Io.Writer, tokens: []const u32) !void {
    var decoder = try ctx.tokenizer.decoder();
    defer decoder.deinit();
    var buf: [256]u8 = undefined;
    for (tokens) |t| try w.writeAll(try decoder.feedOne(t, &buf));
    try w.writeAll(try decoder.finalize(&buf));
}

fn writeToken(ctx: *const Context, w: *std.Io.Writer, token: u32) !void {
    var decoder = try ctx.tokenizer.decoder();
    defer decoder.deinit();
    var buf: [256]u8 = undefined;
    var text: std.Io.Writer.Allocating = .init(ctx.allocator);
    defer text.deinit();
    try text.writer.writeAll(try decoder.feedOne(token, &buf));
    try text.writer.writeAll(try decoder.finalize(&buf));
    try w.print("{d} {f}", .{ token, std.zig.fmtString(text.written()) });
}

/// Teacher-forced agreement of `got` with `ref`: agreeing and compared positions, the prompt's
/// (prefill) and the continuation's (decode), and how many disagreements BF16 rounding explains.
const Verdict = struct {
    prompt: [2]usize = .{ 0, 0 },
    decode: [2]usize = .{ 0, 0 },
    disagreements: usize = 0,
    /// Disagreements whose two tokens are within `rounding_ulps` of each other in both runs.
    within_rounding: usize = 0,

    fn all(self: Verdict) [2]usize {
        return .{ self.prompt[0] + self.decode[0], self.prompt[1] + self.decode[1] };
    }

    /// Agreeing positions over compared ones (0 when none was compared).
    fn rate(self: Verdict) f64 {
        const a = self.all();
        if (a[1] == 0) return 0;
        return @as(f64, @floatFromInt(a[0])) / @as(f64, @floatFromInt(a[1]));
    }

    /// DESIGN 14.0, the whole-model criterion (decided under the owner's delegation): top-1
    /// agreement over every teacher-forced position, the prompt's included, at least `floor` (the
    /// CPU platform's own N-partition agreement with its 1-partition reference on the same
    /// prompts; a fixed bar is below BF16's own noise), and every disagreement a near-tie.
    fn met(self: Verdict, floor: f64) bool {
        if (self.all()[1] == 0) return false;
        return self.rate() >= floor and self.within_rounding == self.disagreements;
    }
};

/// How far `other_id` trails the top-1 of `top`, in BF16 ulps of the top-1's logit (null when
/// `other_id` is outside `top`'s top 8).
fn marginUlps(top: Top, other_id: u32) ?f32 {
    const other = valueOf(top, other_id) orelse return null;
    return (top.values[0] - other) / bf16Ulp(top.values[0]);
}

/// Whether a disagreement is a near-tie in both runs.
fn withinRounding(ref: Top, got: Top) bool {
    const in_ref = marginUlps(ref, got.ids[0]) orelse return false;
    const in_got = marginUlps(got, ref.ids[0]) orelse return false;
    return in_ref <= rounding_ulps and in_got <= rounding_ulps;
}

fn judge(ref: Record, got: Record) Verdict {
    var verdict: Verdict = .{};
    for (ref.prompts, got.prompts) |r, g| {
        for (r.tops, g.tops, 0..) |rt, gt, position| {
            const counts = if (position < r.prompt_len) &verdict.prompt else &verdict.decode;
            counts[1] += 1;
            if (rt.ids[0] == gt.ids[0]) {
                counts[0] += 1;
            } else {
                verdict.disagreements += 1;
                if (withinRounding(rt, gt)) verdict.within_rounding += 1;
            }
        }
    }
    return verdict;
}

fn testTop(first: u32, first_value: f32, second: u32, second_value: f32) Top {
    var top: Top = .{ .ids = .{ 0, 0, 100, 101, 102, 103, 104, 105 }, .values = .{ 0, 0, 10, 9, 8, 7, 6, 5 } };
    top.ids[0] = first;
    top.values[0] = first_value;
    top.ids[1] = second;
    top.values[1] = second_value;
    return top;
}

test judge {
    const allocator = std.testing.allocator;
    const agree = testTop(1, 20, 2, 18);
    // Token 2 leads token 1 by one BF16 ulp here (0.125 at 20) where the reference ties them.
    const ref_tie = testTop(1, 20, 2, 20);
    const got_tie = testTop(2, 20.125, 1, 20);
    // Token 2 leads by 8 ulps: no rounding explains it.
    const got_far = testTop(2, 21, 1, 20);

    const ref_tops = try allocator.alloc(Top, 200);
    defer allocator.free(ref_tops);
    const got_tops = try allocator.alloc(Top, 200);
    defer allocator.free(got_tops);
    @memset(ref_tops, agree);
    @memset(got_tops, agree);
    var ref_prompts = [_]PromptRecord{.{ .text = "", .tokens = &.{}, .prompt_len = 10, .tops = ref_tops[0..20] }};
    var got_prompts = [_]PromptRecord{.{ .text = "", .tokens = &.{}, .prompt_len = 10, .tops = got_tops[0..20] }};
    const ref: Record = .{ .model = "", .platform = "cpu", .seqlen = 32, .prompts = &ref_prompts };
    const got: Record = .{ .model = "", .platform = "cards1", .seqlen = 32, .prompts = &got_prompts };

    // Every decode position agrees and one prompt position of ten does not: 95% of all
    // positions, below a CPU floor of 99% although the decode positions alone are at 100%, at
    // the floor of a CPU that disagreed as often.
    ref_tops[3] = ref_tie;
    got_tops[3] = got_tie;
    var verdict = judge(ref, got);
    try std.testing.expectEqual([2]usize{ 9, 10 }, verdict.prompt);
    try std.testing.expectEqual([2]usize{ 10, 10 }, verdict.decode);
    try std.testing.expectEqual(1, verdict.within_rounding);
    try std.testing.expectEqual(0.95, verdict.rate());
    try std.testing.expect(!verdict.met(0.99));
    try std.testing.expect(verdict.met(0.95));

    // One near-tie in 200 positions: 99.5%, explained by rounding; above a 99% floor, below a
    // CPU that agreed everywhere.
    ref_prompts[0].tops = ref_tops;
    got_prompts[0].tops = got_tops;
    verdict = judge(ref, got);
    try std.testing.expectEqual(1, verdict.disagreements);
    try std.testing.expect(verdict.met(0.99));
    try std.testing.expect(!verdict.met(1.0));

    // The same rate, but the disagreement is not a near-tie in this run.
    got_tops[3] = got_far;
    verdict = judge(ref, got);
    try std.testing.expectEqual(0, verdict.within_rounding);
    try std.testing.expect(!verdict.met(0.99));

    // Outside the other run's top 8: not explained either.
    got_tops[3] = testTop(200, 20.125, 201, 20);
    try std.testing.expect(!judge(ref, got).met(0.99));
}

/// Agreement of `got` (teacher-forced on the reference's tokens) with `ref`, position by position.
fn writeComparison(ctx: *const Context, w: *std.Io.Writer, ref: Record, got: Record) !bool {
    var arena: std.heap.ArenaAllocator = .init(ctx.allocator);
    defer arena.deinit();
    const scratch = arena.allocator();
    try w.print("\nteacher-forced top-1 agreement of {s} with {s} (DESIGN 14.0 whole-model criterion)\n", .{ got.platform, ref.platform });
    try w.writeAll("prompt  prompt positions (prefill)   decode positions   worst |logit difference| on the reference's top-8, in BF16 ulps\n");
    var total_prefill: [2]usize = .{ 0, 0 };
    var total_decode: [2]usize = .{ 0, 0 };
    var disagreements: std.ArrayList(struct { prompt: usize, position: usize }) = .empty;
    defer disagreements.deinit(ctx.allocator);
    for (ref.prompts, got.prompts, 0..) |r, g, p| {
        var agree: [2]usize = .{ 0, 0 };
        var count: [2]usize = .{ 0, 0 };
        var worst_ulps: f64 = 0;
        for (r.tops, g.tops, 0..) |rt, gt, position| {
            const kind: usize = if (position < r.prompt_len) 0 else 1;
            count[kind] += 1;
            if (rt.ids[0] == gt.ids[0]) {
                agree[kind] += 1;
            } else {
                try disagreements.append(ctx.allocator, .{ .prompt = p, .position = position });
            }
            for (rt.ids, rt.values) |id, value| {
                const other = valueOf(gt, id) orelse continue;
                worst_ulps = @max(worst_ulps, @abs(other - value) / bf16Ulp(value));
            }
        }
        try w.print("{d:>6}  {d:>5}/{d:<5} {d:>6.2}%       {d:>4}/{d:<4} {d:>6.2}%   {d:.1}\n", .{
            p,        agree[0], count[0],                    percent(agree[0], count[0]),
            agree[1], count[1], percent(agree[1], count[1]), worst_ulps,
        });
        total_prefill[0] += agree[0];
        total_prefill[1] += count[0];
        total_decode[0] += agree[1];
        total_decode[1] += count[1];
    }
    try w.print("   all  {d:>5}/{d:<5} {d:>6.2}%       {d:>4}/{d:<4} {d:>6.2}%\n", .{
        total_prefill[0], total_prefill[1], percent(total_prefill[0], total_prefill[1]),
        total_decode[0],  total_decode[1],  percent(total_decode[0], total_decode[1]),
    });

    if (disagreements.items.len > 0) {
        try w.writeAll("\ndisagreements: the reference's top-1 and this run's, each with both runs' logits for it (- when outside the other's top 8);\n");
        try w.writeAll("margin: the gap between the two tokens' logits, in BF16 ulps of the larger, in each run\n");
        for (disagreements.items) |d| {
            const rt = ref.prompts[d.prompt].tops[d.position];
            const gt = got.prompts[d.prompt].tops[d.position];
            try w.print("  prompt {d} position {d} ({s}): reference ", .{ d.prompt, d.position, if (d.position < ref.prompts[d.prompt].prompt_len) "prefill" else "decode" });
            try writeToken(ctx, w, rt.ids[0]);
            try w.print(" at {d:.4} (here {s}), here ", .{ rt.values[0], try fmtValue(scratch, valueOf(gt, rt.ids[0])) });
            try writeToken(ctx, w, gt.ids[0]);
            try w.print(" at {d:.4} (reference {s}); margin: reference {s}, here {s}{s}\n", .{
                gt.values[0],
                try fmtValue(scratch, valueOf(rt, gt.ids[0])),
                try fmtMargin(scratch, rt.values[0], valueOf(rt, gt.ids[0])),
                try fmtMargin(scratch, gt.values[0], valueOf(gt, rt.ids[0])),
                if (withinRounding(rt, gt)) "" else "  <- not within BF16 rounding",
            });
        }
    }

    const verdict = judge(ref, got);
    std.debug.assert(verdict.prompt[0] == total_prefill[0] and verdict.decode[0] == total_decode[0]);
    const all = verdict.all();
    try w.print("\ndisagreements within BF16 rounding (at most {d} ulps apart in both runs): {d} of {d}\n", .{ rounding_ulps, verdict.within_rounding, verdict.disagreements });
    // The floor: the CPU's own N-partition agreement with its 1-partition reference.
    var floor_rate: f64 = 0;
    var floor_source: []const u8 = "this run is the CPU's: it measures the floor";
    if (ctx.args.platform == .cards) {
        const file = ctx.args.floor orelse {
            try w.print("M2 token criterion on {s}: NOT JUDGED ({d:.2}% of all {d} teacher-forced positions): the criterion's floor is the CPU platform's own N-partition agreement with the reference; pass --floor=<record of --mode=compare --platform=cpu --cards=N>\n", .{ got.platform, percent(all[0], all[1]), all[1] });
            return false;
        };
        const bytes = try std.Io.Dir.cwd().readFileAlloc(ctx.io, file, ctx.allocator, .limited(1 << 30));
        defer ctx.allocator.free(bytes);
        const parsed = try std.json.parseFromSlice(Record, ctx.allocator, bytes, .{ .allocate = .alloc_always });
        defer parsed.deinit();
        if (parsed.value.prompts.len != ref.prompts.len) {
            log.err("--floor={s} has {d} prompts, the reference {d}", .{ file, parsed.value.prompts.len, ref.prompts.len });
            return error.InvalidArgument;
        }
        const floor = judge(ref, parsed.value);
        floor_rate = floor.rate();
        floor_source = try std.fmt.allocPrint(scratch, "floor {d:.2}%: {s} on the same reference, {d} of {d} positions", .{ floor_rate * 100, parsed.value.platform, floor.all()[0], floor.all()[1] });
    }
    const met = verdict.met(floor_rate);
    try w.print("M2 token criterion on {s}: {s} ({d:.2}% of all {d} teacher-forced positions; {d:.2}% of {d} decode positions; {d:.2}% of {d} prompt positions; {s})\n", .{
        got.platform,                                if (met) "MET" else "NOT MET",
        percent(all[0], all[1]),                     all[1],
        percent(total_decode[0], total_decode[1]),   total_decode[1],
        percent(total_prefill[0], total_prefill[1]), total_prefill[1],
        floor_source,
    });
    return met;
}

fn percent(a: usize, b: usize) f64 {
    if (b == 0) return 0;
    return 100 * @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(b));
}

fn fmtValue(allocator: std.mem.Allocator, value: ?f32) ![]const u8 {
    if (value) |v| return std.fmt.allocPrint(allocator, "{d:.4}", .{v});
    return "-";
}

fn fmtMargin(allocator: std.mem.Allocator, top: f32, other: ?f32) ![]const u8 {
    if (other) |o| return std.fmt.allocPrint(allocator, "{d:.1}", .{(top - o) / bf16Ulp(top)});
    return "-";
}

// ---------------------------------------------------------------------------
// generate.

fn runGenerate(ctx: *Context, loaded: *const llama.LoadedModel, store: *zml.io.TensorStore, stdout: *std.Io.Writer) !bool {
    const allocator = ctx.allocator;
    const io = ctx.io;

    var reference: ?std.json.Parsed(Record) = null;
    defer if (reference) |*r| r.deinit();
    if (ctx.args.reference) |file| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, file, allocator, .limited(1 << 30));
        defer allocator.free(bytes);
        reference = try std.json.parseFromSlice(Record, allocator, bytes, .{ .allocate = .alloc_always });
    }

    // examples/llm's own programs, compiled as its LoadedModel.compile does.
    var progress: std.Progress.Node = .none;
    var compiled: llama.inference.CompiledModel = undefined;
    {
        var start: std.Io.Timestamp = .now(io, .awake);
        const prefill = try llama.inference.compileKernel(allocator, io, ctx.platform, loaded.inner, ctx.params, ctx.args.seqlen, ctx.params.prefill_attention_parameters, .prefill, &progress);
        ctx.timings.compile_ms[0] = msSince(io, start);
        start = .now(io, .awake);
        const decode = try llama.inference.compileKernel(allocator, io, ctx.platform, loaded.inner, ctx.params, 1, ctx.params.decode_attention_parameters, .decode, &progress);
        ctx.timings.compile_ms[1] = msSince(io, start);
        compiled = .{ .loaded_model = loaded, .prefill = prefill, .decode = decode, .params = ctx.params };
    }
    defer compiled.deinit();

    var weights = try loadState(ctx, loaded, store, &.{ &compiled.prefill.raw, &compiled.decode.raw });
    defer loaded.unloadBuffers(&weights, allocator);
    defer unloadState(ctx);
    var prefill_runner: llama.inference.KernelRunner = try .init(allocator, &compiled.prefill, &weights);
    defer prefill_runner.deinit(allocator);
    var decode_runner: llama.inference.KernelRunner = try .init(allocator, &compiled.decode, &weights);
    defer decode_runner.deinit(allocator);
    var rng = try zml.Tensor.Rng.init().initBuffer(io, ctx.platform, 1);
    defer zml.Tensor.Rng.deinitBuffer(&rng);

    var no_steps: [0]f64 = .{};
    try writeTimings(ctx, stdout, &.{}, &no_steps);
    try stdout.flush();

    var all_decode: std.ArrayList(f64) = .empty;
    defer all_decode.deinit(allocator);
    var one: [1][]const u8 = undefined;
    for (promptList(ctx, &one), 0..) |text, p| {
        const prompt = try tokenizePrompt(ctx, allocator, text);
        defer allocator.free(prompt);
        for (0..ctx.args.runs) |run| {
            var generated: std.ArrayList(u32) = .empty;
            defer generated.deinit(allocator);
            var decode_ms: std.ArrayList(f64) = .empty;
            defer decode_ms.deinit(allocator);
            const ttft = try generateOnce(ctx, &prefill_runner, &decode_runner, &rng, prompt, &generated, &decode_ms);
            try all_decode.appendSlice(allocator, decode_ms.items);
            const s = summarize(decode_ms.items);
            try stdout.print("\nprompt {d} run {d}: {d} prompt tokens; time to first token {d:.1} ms; {d} decode steps, launch to token on the host: median {d:.2} ms, mean {d:.2}, min {d:.2}, max {d:.2} ({d:.2} tokens/s at the median)\n", .{
                p, run, prompt.len, ttft, decode_ms.items.len, s.median, s.mean, s.min, s.max, if (s.median > 0) 1000 / s.median else 0,
            });
            if (run == 0) {
                try writeText(ctx, stdout, generated.items);
                try stdout.writeAll("\n");
                if (reference) |r| if (p < r.value.prompts.len) {
                    const ref = r.value.prompts[p];
                    const ref_tokens = ref.tokens[ref.prompt_len..];
                    var same: usize = 0;
                    while (same < @min(ref_tokens.len, generated.items.len) and ref_tokens[same] == generated.items[same]) same += 1;
                    try stdout.print("the first {d} of {d} tokens equal the reference's ({s}) greedy continuation\n", .{ same, @min(ref_tokens.len, generated.items.len), r.value.platform });
                };
            }
            try stdout.flush();
        }
    }
    ctx.timings.recordMemory(ctx.platform, .run);
    const s = summarize(all_decode.items);
    try stdout.print("\nall decode steps: median {d:.2} ms ({d} steps); device memory in use after the runs {d:.1} MiB\n", .{ s.median, all_decode.items.len, mib(ctx.timings.bytes_after_run) });
    return true;
}

/// One prompt as examples/llm's session runs it (llama/session.zig): the
/// prefill of the whole prompt, then one decode step per token, each token
/// read back on the host. Returns the time to the first token, in ms.
fn generateOnce(
    ctx: *Context,
    prefill: *llama.inference.KernelRunner,
    decode: *llama.inference.KernelRunner,
    rng: *zml.Bufferized(zml.Tensor.Rng),
    prompt: []const u32,
    generated: *std.ArrayList(u32),
    decode_ms: *std.ArrayList(f64),
) !f64 {
    const allocator = ctx.allocator;
    const io = ctx.io;
    if (prompt.len + ctx.args.tokens + 1 > ctx.args.seqlen) return error.PromptTooLong;

    const slice: zml.Slice = try .alloc(allocator, .init(.{ctx.args.seqlen}, .u32));
    defer slice.free(allocator);
    @memset(slice.items(u32), 0);
    @memcpy(slice.items(u32)[0..prompt.len], prompt);

    const start: std.Io.Timestamp = .now(io, .awake);
    var tokens_buffer: zml.Buffer = try .fromSlice(io, ctx.platform, slice);
    defer tokens_buffer.deinit();
    var active_length: zml.Buffer = try .scalar(io, ctx.platform, prompt.len, .u32);
    defer active_length.deinit();
    var next: zml.Buffer = undefined;
    try prefill.forward.tryRun(io, .{
        .inputs = .{
            .tokens = tokens_buffer,
            .token_index = ctx.token_index[0],
            .active_length = active_length,
            .kv_cache = ctx.kv,
            .rng = rng.*,
            .attention_metadata = ctx.metadata,
        },
        .outputs = .{ .tokens = &next, .kv_cache = &ctx.kv, .rng = rng },
    }, .{});
    var token = try next.getValue(u32, io);
    next.deinit();
    const ttft = msSince(io, start);

    var current: zml.Buffer = try .fromBytes(io, ctx.platform, .init(.{ .s = 1 }, .u32), std.mem.asBytes(&token));
    defer current.deinit();
    var one: zml.Buffer = try .scalar(io, ctx.platform, 1, .u32);
    defer one.deinit();
    while (true) {
        try generated.append(allocator, token);
        if (isEos(ctx.config, token) or generated.items.len > ctx.args.tokens) break;
        // The token just generated is fed at its own position.
        const position = prompt.len + generated.items.len - 1;
        const step_start: std.Io.Timestamp = .now(io, .awake);
        try decode.forward.tryRun(io, .{
            .inputs = .{
                .tokens = current,
                .token_index = ctx.token_index[position],
                .active_length = one,
                .kv_cache = ctx.kv,
                .rng = rng.*,
                .attention_metadata = ctx.metadata,
            },
            .outputs = .{ .tokens = &current, .kv_cache = &ctx.kv, .rng = rng },
        }, .{});
        token = try current.getValue(u32, io);
        try decode_ms.append(allocator, msSince(io, step_start));
    }
    return ttft;
}
