const std = @import("std");
const zml = @import("zml");

const stdx = zml.stdx;
const log = std.log;

const graph = @import("graph.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const inference = @import("inference.zig");
const algebra = @import("algebra.zig");
const save_load = @import("saveload.zig");
const tokens = @import("tokens.zig");
const sampling = @import("sampling.zig");
const quantization = @import("quantization.zig");
const productq = @import("productq.zig");

const Llm_handler = llm_.Llm_handler;
const LmHeadMatrix = algebra.LmHeadMatrix;
const Graph = graph.Graph;
const ProductQuantizer = productq.ProductQuantizer;
const ProductQuantizerFastScan = productq.ProductQuantizerFastScan;

const Sampler = sampling.Sampler;
const SamplingResult = sampling.SamplingResult;
const SamplingReference = sampling.SamplingReference;
const AngularSampler = sampling.AngularSampler;
const TruncateSampler = sampling.TruncateSampler;
const GraphSampler = sampling.GraphSampler;

const Int8Sampler = sampling.Int8Sampler;
const Int8x4Sampler = sampling.Int8x4Sampler;
const Int4Sampler = sampling.Int4Sampler;
const QJL1Sampler = sampling.QJL1Sampler;
const QJL2Sampler = sampling.QJL2Sampler;
const QJL2x1Sampler = sampling.QJL2x1Sampler;
const QJLNx1Sampler = sampling.QJLNx1Sampler;

const QuantizationInt8 = quantization.QuantizationInt8;
const QuantizationInt4 = quantization.QuantizationInt4;
const QuantizationQJL1 = quantization.QuantizationQJL1;
const QuantizationQJL2 = quantization.QuantizationQJL2;

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Zml_handler = struct {
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    platform: *zml.Platform,
    uris: Uri_handler,
    io: std.Io,
    local_io: std.Io,
    progress: std.Progress.Node,
    args: Args,
    timers: Timing_handler,
    mem: MemoryChecker,

    pub fn fromInit(init: std.process.Init, io: std.Io) !Zml_handler {
        if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
            var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
            defer working_dir.close(init.io);
            try std.process.setCurrentDir(init.io, working_dir);
        }
        const platform = try zml.Platform.auto(init.gpa, io, .{});
        errdefer platform.deinit(init.gpa);

        const args = stdx.flags.parse(init.minimal.args, Args);

        return .{
            .allocator = init.gpa,
            .arena = init.arena,
            .platform = platform,
            .uris = .init(),
            .io = io,
            .local_io = init.io,
            .progress = std.Progress.start(io, .{}),
            .args = args,
            .timers = .{},
            .mem = .{ .platform = platform },
        };
    }

    pub fn deinit(self: *Zml_handler) void {
        self.progress.end();
        self.platform.deinit(self.allocator, self.io);
    }

    pub fn tic(self: *Zml_handler, target: *Timing_handler.Field_timer) void {
        target.init = std.Io.Timestamp.now(self.io, .awake);
    }

    pub fn toc(self: *Zml_handler, target: *Timing_handler.Field_timer) void {
        const end = std.Io.Timestamp.now(self.io, .awake);
        const start: std.Io.Timestamp = target.init;
        const duration = std.Io.Timestamp.durationTo(start, end);
        target.nanoseconds += duration.nanoseconds;
    }

    pub fn reset(_: *Zml_handler, target: *Timing_handler.Field_timer) void {
        target.nanoseconds = 0;
    }

    pub fn log(_: *Zml_handler, target: *Timing_handler.Field_timer, total: usize) void {
        std.log.info("Sampling      : {d:>6.2}s", .{@as(f64, @floatFromInt(target.nanoseconds)) / 1e9});
        std.log.info("ms per sample : {d:>6.2}ms", .{@as(f64, @floatFromInt(target.nanoseconds)) / (1e6 * @as(f64, @floatFromInt(total)))});
    }
};

pub const Uri_handler = struct {
    llama: []const u8,
    qwen: []const u8,
    checkpoint: []const u8,

    pub fn init() Uri_handler {
        return .{
            .llama = "file://examples//detokenization//models//llama",
            .qwen = "file://examples//detokenization//models//qwen",
            .checkpoint = "file://examples//detokenization//checkpoints",
        };
    }
};

pub const Shardings = struct {
    model: zml.Sharding,
    experts: zml.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        return .{
            .model = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth })),
            .experts = try platform.registerSharding("experts", .mesh(.{ .experts = .high_bandwidth })),
        };
    }

    pub fn all(self: Shardings) [2]zml.Sharding {
        return .{ self.model, self.experts };
    }
};

pub const Timing_handler = struct {
    pub const Field_timer = struct {
        nanoseconds: i96 = 0,
        init: std.Io.Timestamp = std.Io.Timestamp.zero,
    };

    similarity_matrix: Field_timer = .{},
    junk_rows: Field_timer = .{},

    knn_graph: Field_timer = .{},
    nsw_graph: Field_timer = .{},

    greedy_search: Field_timer = .{},
    prune_pool_fwd: Field_timer = .{},
    prune_pool_bwd: Field_timer = .{},

    insert_node: Field_timer = .{},

    sampling: Field_timer = .{},

    embed_search_q: Field_timer = .{},
    embed_dot_q: Field_timer = .{},
    embed_search: Field_timer = .{},
    embed_dot: Field_timer = .{},

    quant_dot: Field_timer = .{},
    quant_vec: Field_timer = .{},
    quant_slice_dot: Field_timer = .{},
    quant_walsh: Field_timer = .{},
    pq_rotate: Field_timer = .{},
    pq_lut: Field_timer = .{},
    pq_score: Field_timer = .{},
    pq_top_k: Field_timer = .{},
    quant_5p_prefix_top: Field_timer = .{},
    quant_5p_prefix_dot: Field_timer = .{},
    quant_5p_quantize: Field_timer = .{},
    quant_5p_micro: Field_timer = .{},
    quant_5p_prune: Field_timer = .{},
    quant_5p_half: Field_timer = .{},
    quant_5p_full_1bit: Field_timer = .{},
    quant_5p_2bit: Field_timer = .{},
    quant_5p_dense_top8: Field_timer = .{},
    quant_5p_dense_final: Field_timer = .{},

    trc_sparse_256: Field_timer = .{},
    trc_top_256: Field_timer = .{},
    trc_dense_256: Field_timer = .{},

    prefill: Field_timer = .{},
    decode: Field_timer = .{},

    pub fn print(self: Timing_handler) void {
        std.log.info("Sim matrix    : {d:>6.2}s", .{@as(f64, @floatFromInt(self.similarity_matrix.nanoseconds)) / 1e9});
        std.log.info("Junk rows     : {d:>6.2}s", .{@as(f64, @floatFromInt(self.junk_rows.nanoseconds)) / 1e9});

        std.log.info("kNN graph ini : {d:>6.2}s", .{@as(f64, @floatFromInt(self.knn_graph.nanoseconds)) / 1e9});
        std.log.info("NSW graph ext : {d:>6.2}s", .{@as(f64, @floatFromInt(self.nsw_graph.nanoseconds)) / 1e9});
        std.log.info("Greedy search : {d:>6.2}s", .{@as(f64, @floatFromInt(self.greedy_search.nanoseconds)) / 1e9});
        std.log.info("Pruning fwd   : {d:>6.2}s", .{@as(f64, @floatFromInt(self.prune_pool_fwd.nanoseconds)) / 1e9});
        std.log.info("Pruning bwd   : {d:>6.2}s", .{@as(f64, @floatFromInt(self.prune_pool_bwd.nanoseconds)) / 1e9});
        std.log.info("Insert node   : {d:>6.2}s", .{@as(f64, @floatFromInt(self.insert_node.nanoseconds)) / 1e9});

        std.log.info("Quant dot     : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_dot.nanoseconds)) / 1e9});
        std.log.info("Quant vec     : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_vec.nanoseconds)) / 1e9});
        std.log.info("Bit slice dot : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_slice_dot.nanoseconds)) / 1e9});
        std.log.info("Quant walsh   : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_walsh.nanoseconds)) / 1e9});

        std.log.info("PQ rotate     : {d:>6.2}s", .{@as(f64, @floatFromInt(self.pq_rotate.nanoseconds)) / 1e9});
        std.log.info("PQ LUT        : {d:>6.2}s", .{@as(f64, @floatFromInt(self.pq_lut.nanoseconds)) / 1e9});
        std.log.info("PQ score      : {d:>6.2}s", .{@as(f64, @floatFromInt(self.pq_score.nanoseconds)) / 1e9});
        std.log.info("PQ topK       : {d:>6.2}s", .{@as(f64, @floatFromInt(self.pq_top_k.nanoseconds)) / 1e9});

        std.log.info("5p prefix top : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_prefix_top.nanoseconds)) / 1e9});
        std.log.info("5p prefix dot : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_prefix_dot.nanoseconds)) / 1e9});
        std.log.info("5p quantize   : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_quantize.nanoseconds)) / 1e9});
        std.log.info("5p micro 1bit : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_micro.nanoseconds)) / 1e9});
        std.log.info("5p prune      : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_prune.nanoseconds)) / 1e9});
        std.log.info("5p half 1bit  : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_half.nanoseconds)) / 1e9});
        std.log.info("5p full 1bit  : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_full_1bit.nanoseconds)) / 1e9});
        std.log.info("5p 2bit       : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_2bit.nanoseconds)) / 1e9});
        std.log.info("5p dense topK : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_dense_top8.nanoseconds)) / 1e9});
        std.log.info("5p dense fin  : {d:>6.2}s", .{@as(f64, @floatFromInt(self.quant_5p_dense_final.nanoseconds)) / 1e9});

        std.log.info("Trc sparse    : {d:>6.2}s", .{@as(f64, @floatFromInt(self.trc_sparse_256.nanoseconds)) / 1e9});
        std.log.info("Trc top 256   : {d:>6.2}s", .{@as(f64, @floatFromInt(self.trc_top_256.nanoseconds)) / 1e9});
        std.log.info("Trc dense 256 : {d:>6.2}s", .{@as(f64, @floatFromInt(self.trc_dense_256.nanoseconds)) / 1e9});

        std.log.info("LLM prefill   : {d:>6.2}s", .{@as(f64, @floatFromInt(self.prefill.nanoseconds)) / 1e9});
        std.log.info("LLM decode    : {d:>6.2}s", .{@as(f64, @floatFromInt(self.decode.nanoseconds)) / 1e9});
    }
};

pub const MemoryChecker = struct {
    platform: *zml.Platform,
    bytes_before: i64 = 0,
    bytes_after: i64 = 0,

    pub fn start(self: *MemoryChecker, id: usize) void {
        self.bytes_before = @intCast(self.platform.devices[id].memoryStats().bytes_in_use);
    }

    pub fn check(self: *MemoryChecker, id: usize) void {
        self.bytes_after = @intCast(self.platform.devices[id].memoryStats().bytes_in_use);
        const leaked = @abs(self.bytes_after - self.bytes_before);
        if (leaked != 0) {
            std.log.info("memory usage: before={d} after={d} leaked={d}", .{ self.bytes_before, self.bytes_after, leaked });
        }
    }
};

const Args = struct {
    pub const help =
        \\ Use detokenization [options]
        \\
    ;
};

pub fn printZmlLogo(io: std.Io) !void {
    const LOGO =
        \\
        \\
        \\ ███████╗███╗   ███╗██╗
        \\ ╚══███╔╝████╗ ████║██║
        \\   ███╔╝ ██╔████╔██║██║
        \\  ███╔╝  ██║╚██╔╝██║██║  .ai
        \\ ███████╗██║ ╚═╝ ██║███████╗
        \\ ╚══════╝╚═╝     ╚═╝╚══════╝
        \\
        \\
        \\
    ;
    var writer = std.Io.File.stdout().writer(io, &.{});
    try writer.interface.writeAll(LOGO);
    try writer.interface.flush();
}

pub fn main(init: std.process.Init) !void {
    var http_client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer http_client.deinit();

    var vfs_file: zml.io.VFS.File = .init(init.gpa, init.io, .{});
    defer vfs_file.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(init.gpa, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(init.gpa, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    var zml_handler: Zml_handler = try .fromInit(init, io);
    defer zml_handler.deinit();

    try printZmlLogo(zml_handler.io);

    try runTestsSampling(&zml_handler, zml_handler.uris.llama);
    try runTestsLlm(&zml_handler, zml_handler.uris.llama);
    
    try runTestsSampling(&zml_handler, zml_handler.uris.qwen);
    try runTestsLlm(&zml_handler, zml_handler.uris.qwen);

    zml_handler.timers.print();
}

const bench_qjl = true;
const bench_int = true;
const bench_pq = true;
const bench_misc = true;

pub fn runTestsSampling(zml_handler: *Zml_handler, path: []const u8) !void {
    const alloc = zml_handler.allocator;
    std.log.info("***** Get lm_head", .{});
    var lm_head = try algebra.getLmHead(zml_handler, path);
    defer LmHeadMatrix.deinit(&lm_head, alloc);

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, path);
    var tokenizer = try Llm_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    std.log.info("***** Init sampler", .{});
    var sampler: Sampler = try .init(zml_handler, &lm_head, tokenizer);
    defer Sampler.deinit(&sampler);

    const ref_baseline: SamplingReference = if (true) blk: {
        std.log.info("***** Dense sampling reference", .{});
        const computed_ref = try computeSamplingReference(zml_handler, &sampler, "Dense");
        //try exportSamplingReference(zml_handler, "reference.safetensors", computed_ref);
        break :blk computed_ref;
    } else blk: {
        std.log.info("***** Loading sampling reference", .{});
        break :blk try loadSamplingReference(zml_handler, "reference.safetensors");
    };
    defer alloc.free(ref_baseline.ref);

    var references: std.ArrayList(SamplingReference) = .empty;
    defer {
        for (references.items) |reference| {
            alloc.free(reference.ref);
        }
        references.deinit(alloc);
    }

    var comparisons: std.ArrayList(SamplingComparison) = .empty;
    defer {
        for (comparisons.items) |comparison| {
            alloc.free(comparison.tvd_values);
        }
        comparisons.deinit(alloc);
    }

    try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_baseline));

    if (bench_qjl) {
        const ref_qjl1 = try testQuantizedSampling(zml_handler, &lm_head, QuantizationQJL1, QJL1Sampler, "QJL1");
        try references.append(alloc, ref_qjl1);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_qjl1));
        const ref_qjl2x1 = try testQuantizedSampling(zml_handler, &lm_head, QuantizationQJL1, QJL2x1Sampler, "QJL2x1");
        try references.append(alloc, ref_qjl2x1);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_qjl2x1));
        const ref_qjl2 = try testQuantizedSampling(zml_handler, &lm_head, QuantizationQJL2, QJL2Sampler, "QJL2");
        try references.append(alloc, ref_qjl2);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_qjl2));
        const ref_qjlNx1 = try testQuantizedSampling(zml_handler, &lm_head, QuantizationQJL1, QJLNx1Sampler, "QJLNx1");
        try references.append(alloc, ref_qjlNx1);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_qjlNx1));
    }
    if (bench_int) {
        const ref_int4 = try testQuantizedSampling(zml_handler, &lm_head, QuantizationInt4, Int4Sampler, "Int4");
        try references.append(alloc, ref_int4);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_int4));
        const ref_int8x4 = try testQuantizedSampling(zml_handler, &lm_head, QuantizationInt4, Int8x4Sampler, "Int8x4");
        try references.append(alloc, ref_int8x4);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_int8x4));
        const ref_int8 = try testQuantizedSampling(zml_handler, &lm_head, QuantizationInt8, Int8Sampler, "Int8");
        try references.append(alloc, ref_int8);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_int8));
    }
    if (bench_pq) {
        const ref_pq_van = try testPQSampling(zml_handler, &lm_head, ProductQuantizer, .vanilla, "vanilla PQ");
        try references.append(alloc, ref_pq_van);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_pq_van));
        const ref_pq_ani = try testPQSampling(zml_handler, &lm_head, ProductQuantizer, .anisotropic, "aniso PQ");
        try references.append(alloc, ref_pq_ani);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_pq_ani));
        const ref_pqfc_van = try testPQSampling(zml_handler, &lm_head, ProductQuantizerFastScan, .vanilla, "vanilla PQ-FS");
        try references.append(alloc, ref_pqfc_van);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_pqfc_van));
        const ref_pqfc_ani = try testPQSampling(zml_handler, &lm_head, ProductQuantizerFastScan, .anisotropic, "aniso PQ-FS");
        try references.append(alloc, ref_pqfc_ani);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_pqfc_ani));
    }
    if (bench_misc) {
        const ref_trunc = try testSampling(zml_handler, &lm_head, TruncateSampler, "Truncated");
        try references.append(alloc, ref_trunc);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_trunc));
        const ref_angu = try testSampling(zml_handler, &lm_head, AngularSampler, "Angular");
        try references.append(alloc, ref_angu);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_angu));
        const ref_graph = try testSampling(zml_handler, &lm_head, GraphSampler, "Graph");
        try references.append(alloc, ref_graph);
        try comparisons.append(alloc, try compareSampling(alloc, ref_baseline, ref_graph));
    }
    printComparisonSummary(comparisons.items);
}

pub fn testPQSampling(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, Quantizer: type, option: anytype, label: []const u8) !SamplingReference {
    std.log.info("***** Init {s} quantizer", .{label});
    var quant: Quantizer = try .init(zml_handler, lm_head, option);
    defer Quantizer.deinit(&quant);
    try quant.buildCodebook();
    std.log.info("***** Test {s} sampling", .{label});
    return try computeSamplingReference(zml_handler, &quant, label);
}

pub fn testQuantizedSampling(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, Quantizer: type, Sampling: type, label: []const u8) !SamplingReference {
    std.log.info("***** Init {s} quantizer", .{label});
    var quant: Quantizer = try .init(zml_handler, lm_head);
    defer Quantizer.deinit(&quant);
    try quant.quantize();
    std.log.info("***** Init {s} sampler", .{label});
    var sampler: Sampling = try .init(zml_handler, lm_head, &quant);
    defer sampler.deinit();
    std.log.info("***** Test {s} sampling", .{label});
    return try computeSamplingReference(zml_handler, &sampler, label);
}

pub fn testSampling(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, Sampling: type, label: []const u8) !SamplingReference {
    std.log.info("***** Init {s} sampler", .{label});
    var sampler: Sampling = try .init(zml_handler, lm_head);
    defer sampler.deinit();
    std.log.info("***** Test {s} sampling", .{label});
    return try computeSamplingReference(zml_handler, &sampler, label);
}


pub fn runTestsLlm(zml_handler: *Zml_handler, path: []const u8) !void {
    const alloc = zml_handler.allocator;
    std.log.info("***** Get lm_head", .{});
    var lm_head = try algebra.getLmHead(zml_handler, path);
    defer LmHeadMatrix.deinit(&lm_head, alloc);

    std.log.info("***** Init LLM handler", .{});
    var llm = try Llm_handler.init(zml_handler, path);
    defer llm.deinit(zml_handler.allocator);

    std.log.info("***** Tokenize prompt1", .{});
    const prompt1 = try inference.tokenizePrompt(zml_handler, llm.tokenizer, "Write a python script that computes the n-th prime number");
    defer zml_handler.allocator.free(prompt1);

    std.log.info("***** Tokenize prompt2", .{});
    const prompt2 = try inference.tokenizePrompt(zml_handler, llm.tokenizer, "Write a story about a pirate and his cat");
    defer zml_handler.allocator.free(prompt2);

    try testLlmGPU(zml_handler, &llm, prompt1);
    try testLlmGPU(zml_handler, &llm, prompt2);

    if (bench_qjl) {
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationQJL1, QJL1Sampler, prompt1, "QJL1");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationQJL1, QJL1Sampler, prompt2, "QJL1");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationQJL1, QJL2x1Sampler, prompt1, "QJL2x1");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationQJL1, QJL2x1Sampler, prompt2, "QJL2x1");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationQJL1, QJLNx1Sampler, prompt1, "QJLNx1");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationQJL1, QJLNx1Sampler, prompt2, "QJLNx1");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationQJL2, QJL2Sampler, prompt1, "QJL2");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationQJL2, QJL2Sampler, prompt2, "QJL2");
    }
    if (bench_int) {
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationInt4, Int4Sampler, prompt1, "Int4");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationInt4, Int4Sampler, prompt2, "Int4");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationInt4, Int8x4Sampler, prompt1, "Int8x4");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationInt4, Int8x4Sampler, prompt2, "Int8x4");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationInt8, Int8Sampler, prompt1, "Int8");
        try testQuantizedLlm(zml_handler, &llm, &lm_head, QuantizationInt8, Int8Sampler, prompt2, "Int8");
    }
    if (bench_pq) {
        try testPQLlm(zml_handler, &llm, &lm_head, ProductQuantizer, .vanilla, prompt1, "PQ vanilla");
        try testPQLlm(zml_handler, &llm, &lm_head, ProductQuantizer, .vanilla, prompt2, "PQ vanilla");
        try testPQLlm(zml_handler, &llm, &lm_head, ProductQuantizer, .anisotropic, prompt1, "PQ aniso");
        try testPQLlm(zml_handler, &llm, &lm_head, ProductQuantizer, .anisotropic, prompt2, "PQ aniso");
        try testPQLlm(zml_handler, &llm, &lm_head, ProductQuantizerFastScan, .vanilla, prompt1, "PQ-FS vanilla");
        try testPQLlm(zml_handler, &llm, &lm_head, ProductQuantizerFastScan, .vanilla, prompt2, "PQ-FS vanilla");
        try testPQLlm(zml_handler, &llm, &lm_head, ProductQuantizerFastScan, .anisotropic, prompt1, "PQ-FS aniso");
        try testPQLlm(zml_handler, &llm, &lm_head, ProductQuantizerFastScan, .anisotropic, prompt2, "PQ-FS aniso");
    }
    if (bench_misc) {
        try testLlm(zml_handler, &llm, &lm_head, TruncateSampler, prompt1, "Truncated");
        try testLlm(zml_handler, &llm, &lm_head, TruncateSampler, prompt2, "Truncated");
        try testLlm(zml_handler, &llm, &lm_head, AngularSampler, prompt1, "Angular");
        try testLlm(zml_handler, &llm, &lm_head, AngularSampler, prompt2, "Angular");
        try testLlm(zml_handler, &llm, &lm_head, GraphSampler, prompt1, "Graph");
        try testLlm(zml_handler, &llm, &lm_head, GraphSampler, prompt2, "Graph");
    }
}

pub fn testLlmGPU(zml_handler: *Zml_handler, llm: *Llm_handler, prompt: []const u32) !void {
    std.log.info("***** Generate text GPU sampling", .{});
    zml_handler.mem.start(0);
    const generated_text_gpu = try inference.generateTextGPUSampling(zml_handler, llm, prompt);
    defer zml_handler.allocator.free(generated_text_gpu);
    zml_handler.mem.check(0);

    try llm.resetKvCache(zml_handler);
}

pub fn testPQLlm(zml_handler: *Zml_handler, llm: *Llm_handler, lm_head: *LmHeadMatrix, Quantizer: type, option: anytype, prompt: []const u32, label: []const u8) !void {
    std.log.info("***** Init {s} quantizer", .{label});
    var quant: Quantizer = try .init(zml_handler, lm_head, option);
    defer Quantizer.deinit(&quant);
    try quant.buildCodebook();

    std.log.info("***** Generate text CPU with {s} sampling", .{label});
    zml_handler.mem.start(0);
    const generated_text_cpu = try inference.generateTextCPUSampling(zml_handler, llm, &quant, prompt);
    defer zml_handler.allocator.free(generated_text_cpu);
    zml_handler.mem.check(0);

    try llm.resetKvCache(zml_handler);
}

pub fn testQuantizedLlm(zml_handler: *Zml_handler, llm: *Llm_handler, lm_head: *LmHeadMatrix, Quantizer: type, Sampling: type, prompt: []const u32, label: []const u8) !void {
    std.log.info("***** Init {s} quantizer", .{label});
    var quant: Quantizer = try .init(zml_handler, lm_head);
    defer Quantizer.deinit(&quant);
    try quant.quantize();

    std.log.info("***** Init {s} sampler", .{label});
    var sampler: Sampling = try .init(zml_handler, lm_head, &quant);
    defer sampler.deinit();

    std.log.info("***** Generate text CPU with {s} sampling", .{label});
    zml_handler.mem.start(0);
    const generated_text_cpu = try inference.generateTextCPUSampling(zml_handler, llm, &sampler, prompt);
    defer zml_handler.allocator.free(generated_text_cpu);
    zml_handler.mem.check(0);

    try llm.resetKvCache(zml_handler);
}

pub fn testLlm(zml_handler: *Zml_handler, llm: *Llm_handler, lm_head: *LmHeadMatrix, Sampling: type, prompt: []const u32, label: []const u8) !void {
    std.log.info("***** Init {s} sampler", .{label});
    var sampler: Sampling = try .init(zml_handler, lm_head);
    defer sampler.deinit();

    std.log.info("***** Generate text CPU with {s} sampling", .{label});
    zml_handler.mem.start(0);
    const generated_text_cpu = try inference.generateTextCPUSampling(zml_handler, llm, &sampler, prompt);
    defer zml_handler.allocator.free(generated_text_cpu);
    zml_handler.mem.check(0);

    try llm.resetKvCache(zml_handler);
}


pub fn computeSamplingReference(zml_handler: *Zml_handler, sampler: anytype, label: []const u8) !SamplingReference {
    var total_embeds: usize = 0;
    const small_bench = true;
    const big_bench = true;

    if (small_bench) {
        const tasks_id = [5]u8{ 0, 1, 2, 3, 4 };
        for (tasks_id) |task_id| {
            const task = switch (task_id) {
                0 => "coding",
                1 => "history",
                2 => "math",
                3 => "story",
                4 => "translate",
                else => return error.InvalidTask,
            };
            const embed_slice = try save_load.getSlice(zml_handler, "llama_embeds.safetensors", task, .f32, true);
            defer embed_slice.free(zml_handler.allocator);
            const n: usize = @intCast(embed_slice.shape.dims()[0]);
            total_embeds += n;
        }
    }

    if (big_bench) {
        const tasks_id = [3]u8{ 0, 1, 2 };
        for (tasks_id) |task_id| {
            const task = switch (task_id) {
                0 => "biblio",
                1 => "longcode",
                2 => "longtranslate",
                else => return error.InvalidTask,
            };
            const embed_slice = try save_load.getSlice(zml_handler, "llama_embeds2.safetensors", task, .f32, true);
            defer embed_slice.free(zml_handler.allocator);
            const n: usize = @intCast(embed_slice.shape.dims()[0]);
            total_embeds += n;
        }
    }

    var ref: SamplingReference = .{
        .ref = try zml_handler.allocator.alloc(SamplingResult, total_embeds),
        .label = label,
    };
    var nb: usize = 0;

    zml_handler.reset(&zml_handler.timers.sampling);

    if (small_bench) {
        const tasks_id = [5]u8{ 0, 1, 2, 3, 4 };
        for (tasks_id) |task_id| {
            const task = switch (task_id) {
                0 => "coding",
                1 => "history",
                2 => "math",
                3 => "story",
                4 => "translate",
                else => return error.InvalidTask,
            };
            const embed_slice = try save_load.getSlice(zml_handler, "llama_embeds.safetensors", task, .f32, true);
            defer embed_slice.free(zml_handler.allocator);
            const n: usize = @intCast(embed_slice.shape.dims()[0]);
            const d: usize = @intCast(embed_slice.shape.dims()[1]);
            std.log.info("sampling task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
            for (0..n) |embed_index| {
                const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];
                zml_handler.tic(&zml_handler.timers.sampling);
                ref.ref[nb] = sampler.sample(embed);
                zml_handler.toc(&zml_handler.timers.sampling);
                nb += 1;
            }
        }
    }

    if (big_bench) {
        const tasks_id = [3]u8{ 0, 1, 2 };
        for (tasks_id) |task_id| {
            const task = switch (task_id) {
                0 => "biblio",
                1 => "longcode",
                2 => "longtranslate",
                else => return error.InvalidTask,
            };
            const embed_slice = try save_load.getSlice(zml_handler, "llama_embeds2.safetensors", task, .f32, true);
            defer embed_slice.free(zml_handler.allocator);
            const n: usize = @intCast(embed_slice.shape.dims()[0]);
            const d: usize = @intCast(embed_slice.shape.dims()[1]);
            std.log.info("sampling task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
            for (0..n) |embed_index| {
                const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];
                zml_handler.tic(&zml_handler.timers.sampling);
                ref.ref[nb] = sampler.sample(embed);
                zml_handler.toc(&zml_handler.timers.sampling);
                nb += 1;
            }
        }
    }

    ref.ms_per_sample = @as(f64, @floatFromInt(zml_handler.timers.sampling.nanoseconds)) / (1e6 * @as(f64, @floatFromInt(nb)));
    std.log.info("Embed search: total={d}", .{nb});
    zml_handler.log(&zml_handler.timers.sampling, nb);

    return ref;
}

pub fn exportSamplingReference(zml_handler: *Zml_handler, file_name: []const u8, ref: SamplingReference) !void {
    const allocator = zml_handler.allocator;
    const n = ref.ref.len;
    const k = sampling.sampling_top_k;
    const nk = n * k;

    const rows_byte_len: u64 = @as(u64, @intCast(nk)) * @sizeOf(u64);
    const logits_byte_len: u64 = @as(u64, @intCast(nk)) * @sizeOf(f32);
    const probas_byte_len: u64 = @as(u64, @intCast(nk)) * @sizeOf(f32);
    const upper_bounds_byte_len: u64 = @as(u64, @intCast(nk)) * @sizeOf(f32);
    const nbs_byte_len: u64 = @as(u64, @intCast(n)) * @sizeOf(u64);

    var offset: u64 = 0;
    const rows_start = offset;
    offset += rows_byte_len;
    const rows_end = offset;
    const logits_start = offset;
    offset += logits_byte_len;
    const logits_end = offset;
    const probas_start = offset;
    offset += probas_byte_len;
    const probas_end = offset;
    const upper_bounds_start = offset;
    offset += upper_bounds_byte_len;
    const upper_bounds_end = offset;
    const nbs_start = offset;
    offset += nbs_byte_len;
    const nbs_end = offset;

    const header = try std.fmt.allocPrint(allocator,
        \\{{"rows":{{"dtype":"U64","shape":[{d},{d}],"data_offsets":[{d},{d}]}},"logits":{{"dtype":"F32","shape":[{d},{d}],"data_offsets":[{d},{d}]}},"probas":{{"dtype":"F32","shape":[{d},{d}],"data_offsets":[{d},{d}]}},"upper_bounds":{{"dtype":"F32","shape":[{d},{d}],"data_offsets":[{d},{d}]}},"nb":{{"dtype":"U64","shape":[{d}],"data_offsets":[{d},{d}]}}}}
    , .{
        n, k,         rows_start,         rows_end,
        n, k,         logits_start,       logits_end,
        n, k,         probas_start,       probas_end,
        n, k,         upper_bounds_start, upper_bounds_end,
        n, nbs_start, nbs_end,
    });
    defer allocator.free(header);

    const padded_header_len = std.mem.alignForward(usize, header.len, 8);
    const padded_header = try allocator.alloc(u8, padded_header_len);
    defer allocator.free(padded_header);
    @memcpy(padded_header[0..header.len], header);
    @memset(padded_header[header.len..], ' ');

    var header_len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_len_bytes, @intCast(padded_header.len), .little);

    const checkpoint_path = localPathFromFileUri(zml_handler.uris.checkpoint) orelse return error.NonLocalCheckpointPath;
    const output_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ checkpoint_path, file_name });
    defer allocator.free(output_path);

    var file = try std.Io.Dir.createFile(.cwd(), zml_handler.local_io, output_path, .{ .truncate = true });
    defer file.close(zml_handler.local_io);

    try file.writeStreamingAll(zml_handler.local_io, &header_len_bytes);
    try file.writeStreamingAll(zml_handler.local_io, padded_header);

    const chunk_len = 4096;
    const rows_buf = try allocator.alloc(u64, @min(chunk_len, nk));
    defer allocator.free(rows_buf);
    var flat_index: usize = 0;
    while (flat_index < nk) {
        const count = @min(rows_buf.len, nk - flat_index);
        for (rows_buf[0..count], 0..) |*row, i| {
            const index = flat_index + i;
            row.* = @intCast(ref.ref[index / k].candidates[index % k].row);
        }
        try file.writeStreamingAll(zml_handler.local_io, std.mem.sliceAsBytes(rows_buf[0..count]));
        flat_index += count;
    }

    const f32_buf = try allocator.alloc(f32, @min(chunk_len, nk));
    defer allocator.free(f32_buf);
    flat_index = 0;
    while (flat_index < nk) {
        const count = @min(f32_buf.len, nk - flat_index);
        for (f32_buf[0..count], 0..) |*logit, i| {
            const index = flat_index + i;
            logit.* = ref.ref[index / k].candidates[index % k].logit;
        }
        try file.writeStreamingAll(zml_handler.local_io, std.mem.sliceAsBytes(f32_buf[0..count]));
        flat_index += count;
    }

    flat_index = 0;
    while (flat_index < nk) {
        const count = @min(f32_buf.len, nk - flat_index);
        for (f32_buf[0..count], 0..) |*proba, i| {
            const index = flat_index + i;
            proba.* = ref.ref[index / k].candidates[index % k].proba;
        }
        try file.writeStreamingAll(zml_handler.local_io, std.mem.sliceAsBytes(f32_buf[0..count]));
        flat_index += count;
    }

    flat_index = 0;
    while (flat_index < nk) {
        const count = @min(f32_buf.len, nk - flat_index);
        for (f32_buf[0..count], 0..) |*upper_bound, i| {
            const index = flat_index + i;
            upper_bound.* = ref.ref[index / k].candidates[index % k].upper_bound;
        }
        try file.writeStreamingAll(zml_handler.local_io, std.mem.sliceAsBytes(f32_buf[0..count]));
        flat_index += count;
    }

    const nbs_buf = try allocator.alloc(u64, @min(chunk_len, n));
    defer allocator.free(nbs_buf);
    var sample_index: usize = 0;
    while (sample_index < n) {
        const count = @min(nbs_buf.len, n - sample_index);
        for (nbs_buf[0..count], 0..) |*nb, i| {
            nb.* = @intCast(ref.ref[sample_index + i].nb);
        }
        try file.writeStreamingAll(zml_handler.local_io, std.mem.sliceAsBytes(nbs_buf[0..count]));
        sample_index += count;
    }

    std.log.info("Exported sampling reference: file={s} samples={d} top_k={d} bytes={d}", .{ file_name, n, k, offset });
}

fn localPathFromFileUri(uri: []const u8) ?[]const u8 {
    const file_scheme = "file://";
    if (!std.mem.startsWith(u8, uri, file_scheme)) return null;
    return uri[file_scheme.len..];
}

pub fn loadSamplingReference(zml_handler: *Zml_handler, file_name: []const u8) !SamplingReference {
    const allocator = zml_handler.allocator;
    const k = sampling.sampling_top_k;

    const rows = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.checkpoint, file_name, "rows");
    defer rows.free(allocator);
    const logits = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.checkpoint, file_name, "logits");
    defer logits.free(allocator);
    const probas = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.checkpoint, file_name, "probas");
    defer probas.free(allocator);
    const upper_bounds = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.checkpoint, file_name, "upper_bounds");
    defer upper_bounds.free(allocator);
    const nbs = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.checkpoint, file_name, "nb");
    defer nbs.free(allocator);

    if (rows.shape.rank() != 2 or rows.shape.dim(1) != k or rows.shape.dtype() != .u64) return error.InvalidSamplingReference;
    const n: usize = @intCast(rows.shape.dim(0));
    if (logits.shape.rank() != 2 or logits.shape.dim(0) != n or logits.shape.dim(1) != k or logits.shape.dtype() != .f32) return error.InvalidSamplingReference;
    if (probas.shape.rank() != 2 or probas.shape.dim(0) != n or probas.shape.dim(1) != k or probas.shape.dtype() != .f32) return error.InvalidSamplingReference;
    if (upper_bounds.shape.rank() != 2 or upper_bounds.shape.dim(0) != n or upper_bounds.shape.dim(1) != k or upper_bounds.shape.dtype() != .f32) return error.InvalidSamplingReference;
    if (nbs.shape.rank() != 1 or nbs.shape.dim(0) != n or nbs.shape.dtype() != .u64) return error.InvalidSamplingReference;

    const rows_items = rows.constItems(u64);
    const logits_items = logits.constItems(f32);
    const probas_items = probas.constItems(f32);
    const upper_bounds_items = upper_bounds.constItems(f32);
    const nbs_items = nbs.constItems(u64);

    const ref: SamplingReference = .{
        .ref = try allocator.alloc(SamplingResult, n),
        .label = "Reference",
    };
    errdefer allocator.free(ref.ref);

    for (ref.ref, 0..) |*sample, sample_index| {
        sample.nb = @intCast(nbs_items[sample_index]);
        for (0..k) |rank| {
            const index = sample_index * k + rank;
            sample.candidates[rank] = .{
                .row = @intCast(rows_items[index]),
                .logit = logits_items[index],
                .proba = probas_items[index],
                .upper_bound = upper_bounds_items[index],
            };
        }
    }

    std.log.info("Loaded sampling reference: file={s} samples={d} top_k={d}", .{ file_name, n, k });
    return ref;
}


const SamplingComparison = struct {
    label: []const u8,
    total: usize,
    ms_per_sample: f64,
    top1_found_percent: f64,
    top1_found_count: usize,
    topp_bucket_cases: [topp_bucket_count]usize,
    topp_bucket_all_found_count: [topp_bucket_count]usize,
    topp_bucket_all_found_percent: [topp_bucket_count]f64,
    topp_bucket_found_token_count: [topp_bucket_count]usize,
    topp_bucket_token_count: [topp_bucket_count]usize,
    topp_bucket_found_percent: [topp_bucket_count]f64,
    total_topp_found_percent: f64,
    total_topp_found_token_count: usize,
    total_topp_token_count: usize,
    average_found_topp_mass: f64,
    tvd_values: []f64,
    tvd_p50: f64,
    tvd_p90: f64,
    tvd_p99: f64,
    full_fail_percent: f64,
    full_fail_count: usize,
    proba_fail_percent: f64,
    proba_fail_count: usize,
};

const topp_bucket_count = 4;
const topp_bucket_labels = [topp_bucket_count][]const u8{ "top-p=1", "top-p=2", "top-p=3-7", "top-p>=8" };

pub fn compareSampling(allocator: std.mem.Allocator, ref: SamplingReference, other: SamplingReference) !SamplingComparison {
    const total = @min(ref.ref.len, other.ref.len);
    if (ref.ref.len != other.ref.len) {
        std.log.warn("Compare sampling references with different lengths: ref={d} other={d} compared={d}", .{ ref.ref.len, other.ref.len, total });
    }
    if (total == 0) {
        std.log.warn("Compare sampling references: no samples", .{});
        const tvd_values = try allocator.alloc(f64, 0);
        return .{
            .label = other.label,
            .total = 0,
            .ms_per_sample = other.ms_per_sample,
            .top1_found_percent = 0.0,
            .top1_found_count = 0,
            .topp_bucket_cases = [_]usize{0} ** topp_bucket_count,
            .topp_bucket_all_found_count = [_]usize{0} ** topp_bucket_count,
            .topp_bucket_all_found_percent = [_]f64{0.0} ** topp_bucket_count,
            .topp_bucket_found_token_count = [_]usize{0} ** topp_bucket_count,
            .topp_bucket_token_count = [_]usize{0} ** topp_bucket_count,
            .topp_bucket_found_percent = [_]f64{0.0} ** topp_bucket_count,
            .total_topp_found_percent = 0.0,
            .total_topp_found_token_count = 0,
            .total_topp_token_count = 0,
            .average_found_topp_mass = 0.0,
            .tvd_values = tvd_values,
            .tvd_p50 = 0.0,
            .tvd_p90 = 0.0,
            .tvd_p99 = 0.0,
            .full_fail_percent = 0.0,
            .full_fail_count = 0,
            .proba_fail_percent = 0.0,
            .proba_fail_count = 0,
        };
    }

    const tvd_values = try allocator.alloc(f64, total);
    errdefer allocator.free(tvd_values);

    var top1_found_count: usize = 0;

    var topp_bucket_cases = [_]usize{0} ** topp_bucket_count;
    var topp_bucket_all_found_count = [_]usize{0} ** topp_bucket_count;
    var topp_bucket_found_token_count = [_]usize{0} ** topp_bucket_count;
    var topp_bucket_token_count = [_]usize{0} ** topp_bucket_count;

    var total_topp_token_count: usize = 0;
    var total_topp_found_token_count: usize = 0;
    var total_found_topp_mass: f64 = 0.0;

    var full_fail_count: usize = 0;

    var proba_success_count: usize = 0;
    var proba_fail_count: usize = 0;

    for (0..total) |sample_index| {
        const ref_sample = ref.ref[sample_index];
        const other_sample = other.ref[sample_index];
        const topp_count = samplingTopPCount(ref_sample);
        const other_topp_count = samplingTopPCount(other_sample);

        const ref_top1 = ref_sample.candidates[0];
        if (samplingContainsToken(other_sample, ref_top1.row)) {
            top1_found_count += 1;
        }

        const threshold = 0.5 * ref_top1.proba;
        var max_other_ref_proba: f32 = 0.0;
        for (other_sample.candidates) |candidate| {
            max_other_ref_proba = @max(max_other_ref_proba, samplingProbabilityForToken(ref_sample, candidate.row));
        }
        if (max_other_ref_proba >= threshold) {
            proba_success_count += 1;
        } else {
            proba_fail_count += 1;
        }

        var found_topp_count: usize = 0;
        var found_topp_mass: f64 = 0.0;
        for (ref_sample.candidates[0..topp_count]) |candidate| {
            if (samplingContainsToken(other_sample, candidate.row)) {
                found_topp_count += 1;
                found_topp_mass += candidate.proba;
            }
        }
        const tvd = samplingTvd(ref_sample, topp_count, other_sample, other_topp_count);
        tvd_values[sample_index] = tvd;

        total_topp_token_count += topp_count;
        total_topp_found_token_count += found_topp_count;
        total_found_topp_mass += found_topp_mass;
        if (found_topp_count == 0) {
            full_fail_count += 1;
        }

        const bucket = toppBucketIndex(topp_count);
        topp_bucket_cases[bucket] += 1;
        topp_bucket_token_count[bucket] += topp_count;
        topp_bucket_found_token_count[bucket] += found_topp_count;
        if (found_topp_count == topp_count) {
            topp_bucket_all_found_count[bucket] += 1;
        }
    }

    const inv_total = 1.0 / @as(f64, @floatFromInt(total));
    const top1_found_percent = 100.0 * @as(f64, @floatFromInt(top1_found_count)) * inv_total;
    var topp_bucket_all_found_percent: [topp_bucket_count]f64 = undefined;
    var topp_bucket_found_percent: [topp_bucket_count]f64 = undefined;
    for (0..topp_bucket_count) |bucket| {
        topp_bucket_all_found_percent[bucket] = percent(topp_bucket_all_found_count[bucket], topp_bucket_cases[bucket]);
        topp_bucket_found_percent[bucket] = percent(topp_bucket_found_token_count[bucket], topp_bucket_token_count[bucket]);
    }
    const total_topp_found_percent = percent(total_topp_found_token_count, total_topp_token_count);
    const average_found_topp_mass = total_found_topp_mass * inv_total;
    std.mem.sort(f64, tvd_values, {}, std.sort.asc(f64));
    const tvd_p50 = percentileSortedOrZero(tvd_values, 0.50);
    const tvd_p90 = percentileSortedOrZero(tvd_values, 0.90);
    const tvd_p99 = percentileSortedOrZero(tvd_values, 0.99);
    const full_fail_percent = 100.0 * @as(f64, @floatFromInt(full_fail_count)) * inv_total;
    const proba_fail_percent = 100.0 * @as(f64, @floatFromInt(proba_fail_count)) * inv_total;

    const comparison: SamplingComparison = .{
        .label = other.label,
        .total = total,
        .ms_per_sample = other.ms_per_sample,
        .top1_found_percent = top1_found_percent,
        .top1_found_count = top1_found_count,
        .topp_bucket_cases = topp_bucket_cases,
        .topp_bucket_all_found_count = topp_bucket_all_found_count,
        .topp_bucket_all_found_percent = topp_bucket_all_found_percent,
        .topp_bucket_found_token_count = topp_bucket_found_token_count,
        .topp_bucket_token_count = topp_bucket_token_count,
        .topp_bucket_found_percent = topp_bucket_found_percent,
        .total_topp_found_percent = total_topp_found_percent,
        .total_topp_found_token_count = total_topp_found_token_count,
        .total_topp_token_count = total_topp_token_count,
        .average_found_topp_mass = average_found_topp_mass,
        .tvd_values = tvd_values,
        .tvd_p50 = tvd_p50,
        .tvd_p90 = tvd_p90,
        .tvd_p99 = tvd_p99,
        .full_fail_percent = full_fail_percent,
        .full_fail_count = full_fail_count,
        .proba_fail_percent = proba_fail_percent,
        .proba_fail_count = proba_fail_count,
    };

    std.log.info("Results for {s}", .{comparison.label});
    std.log.info("Total samplings    : {d}", .{comparison.total});
    std.log.info("Found top1 token   : {d:.4}% ({d:>5}/{d:>5})", .{ top1_found_percent, top1_found_count, total });
    for (0..topp_bucket_count) |bucket| {
        std.log.info("{s:>9} all found: {d:.4}% ({d:>5}/{d:>5}); tokens found: {d:.4}% ({d:>5}/{d:>5})", .{
            topp_bucket_labels[bucket],
            topp_bucket_all_found_percent[bucket],
            topp_bucket_all_found_count[bucket],
            topp_bucket_cases[bucket],
            topp_bucket_found_percent[bucket],
            topp_bucket_found_token_count[bucket],
            topp_bucket_token_count[bucket],
        });
    }
    std.log.info("top-p all found    : {d:.4}% ({d:>5}/{d:>5})", .{ total_topp_found_percent, total_topp_found_token_count, total_topp_token_count });
    std.log.info("top-p mass proba   : {d:.6}", .{average_found_topp_mass});
    std.log.info("top-p proba loss   : p50={d:.6} p90={d:.6} p99={d:.6}", .{ tvd_p50, tvd_p90, tvd_p99 });
    std.log.info("full fail          : {d}/{d} ({d:.4}%)", .{ full_fail_count, total, full_fail_percent });
    std.log.info("probability fail   : {d}/{d} ({d:.4}%)", .{ proba_fail_count, total, proba_fail_percent });

    return comparison;
}

pub fn printComparisonSummary(comparisons: []const SamplingComparison) void {
    if (comparisons.len == 0) return;

    std.log.info("", .{});
    std.log.info("Sampling comparison summary", .{});
    std.log.info(
        "{s:<14} {s:>9} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10}",
        .{ "sampler", "ms/search", "top1 %", "p1 all", "p2 all", "p3-7 all", "p8+ all", "top-p %", "mass", "full fail", "loss p50", "loss p90", "loss p99" },
    );
    for (comparisons) |comparison| {
        std.log.info(
            "{s:<14} {d:>9.3} {d:>10.4} {d:>10.4} {d:>10.4} {d:>10.4} {d:>10.4} {d:>10.4} {d:>10.6} {d:>10.4} {d:>10.6} {d:>10.6} {d:>10.6}",
            .{
                comparison.label,
                comparison.ms_per_sample,
                comparison.top1_found_percent,
                comparison.topp_bucket_all_found_percent[0],
                comparison.topp_bucket_all_found_percent[1],
                comparison.topp_bucket_all_found_percent[2],
                comparison.topp_bucket_all_found_percent[3],
                comparison.total_topp_found_percent,
                comparison.average_found_topp_mass,
                comparison.full_fail_percent,
                comparison.tvd_p50,
                comparison.tvd_p90,
                comparison.tvd_p99,
            },
        );
    }
}

fn toppBucketIndex(topp_count: usize) usize {
    return switch (topp_count) {
        1 => 0,
        2 => 1,
        3...7 => 2,
        else => 3,
    };
}

fn samplingTopPCount(sample: SamplingResult) usize {
    return @min(sample.nb + 1, sample.candidates.len);
}

fn samplingTvd(ref_sample: SamplingResult, ref_topp_count: usize, other_sample: SamplingResult, other_topp_count: usize) f64 {
    var max_loss: f64 = 0.0;
    for (ref_sample.candidates[0..ref_topp_count]) |candidate| {
        const ref_proba: f64 = candidate.proba;
        const other_proba: f64 = samplingTopPProbabilityForToken(other_sample, other_topp_count, candidate.row);
        max_loss = @max(max_loss, ref_proba - other_proba);
    }
    return max_loss;
}

fn samplingTopPProbabilityForToken(sample: SamplingResult, topp_count: usize, token: usize) f32 {
    for (sample.candidates[0..topp_count]) |candidate| {
        if (candidate.row == token) return candidate.proba;
    }
    return 0.0;
}

fn percentileSortedOrZero(values: []const f64, q: f64) f64 {
    if (values.len == 0) return 0.0;
    const scaled = q * @as(f64, @floatFromInt(values.len - 1));
    const index: usize = @intFromFloat(@round(scaled));
    return values[index];
}

fn samplingContainsToken(sample: SamplingResult, token: usize) bool {
    for (sample.candidates) |candidate| {
        if (candidate.row == token) return true;
    }
    return false;
}

fn samplingProbabilityForToken(sample: SamplingResult, token: usize) f32 {
    for (sample.candidates) |candidate| {
        if (candidate.row == token) return candidate.proba;
    }
    return 0.0;
}

fn percent(numerator: usize, denominator: usize) f64 {
    if (denominator == 0) return 0.0;
    return 100.0 * @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator));
}
