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

    try runLlm(&zml_handler);
    //try runTestsSampling(&zml_handler);

    zml_handler.timers.print();
}

pub fn runTestsSampling(zml_handler: *Zml_handler) !void {
    std.log.info("***** Get lm_head", .{});
    var lm_head = try algebra.getLmHead(zml_handler);
    defer LmHeadMatrix.deinit(&lm_head, zml_handler.allocator);

    std.log.info("***** Init sampler", .{});
    var sampler: Sampler = try .init(zml_handler, &lm_head);
    defer Sampler.deinit(&sampler);

    const ref: SamplingReference = if (false) blk: {
        std.log.info("***** Dense sampling reference", .{});
        const computed_ref = try computeSamplingReference(zml_handler, &sampler);
        //try exportSamplingReference(zml_handler, "reference.safetensors", computed_ref);
        break :blk computed_ref;
    } else blk: {
        std.log.info("***** Loading sampling reference", .{});
        break :blk try loadSamplingReference(zml_handler, "reference.safetensors");
    };
    defer zml_handler.allocator.free(ref.ref);

    if (false) {
        std.log.info("***** Init QJL 1 bit quantizer", .{});
        var quant_qjl1: QuantizationQJL1 = try .init(zml_handler, &lm_head);
        defer QuantizationQJL1.deinit(&quant_qjl1);
        try quant_qjl1.quantize();

        std.log.info("***** Init QJL 1 bit sampler", .{});
        var qjl1_sampler: QJL1Sampler = try .init(zml_handler, &lm_head, &quant_qjl1);
        defer qjl1_sampler.deinit();

        std.log.info("***** Test QJL 1 bit sampling", .{});
        const ref_qjl1 = try computeSamplingReference(zml_handler, &qjl1_sampler);
        defer zml_handler.allocator.free(ref_qjl1.ref);
        compareSampling(ref, ref_qjl1);

        std.log.info("***** Init QJL 2x1 bit sampler", .{});
        var qjl2x1_sampler: QJL2x1Sampler = try .init(zml_handler, &lm_head, &quant_qjl1);
        defer qjl2x1_sampler.deinit();

        std.log.info("***** Test QJL 2x1 bit sampling", .{});
        const ref_qjl2x1 = try computeSamplingReference(zml_handler, &qjl2x1_sampler);
        defer zml_handler.allocator.free(ref_qjl2x1.ref);
        compareSampling(ref, ref_qjl2x1);

        std.log.info("***** Init QJL Nx1 bit sampler", .{});
        var qjlNx1_sampler: QJLNx1Sampler = try .init(zml_handler, &lm_head, &quant_qjl1);
        defer qjlNx1_sampler.deinit();

        std.log.info("***** Test QJL Nx1 bit sampling", .{});
        const ref_qjlNx1 = try computeSamplingReference(zml_handler, &qjlNx1_sampler);
        defer zml_handler.allocator.free(ref_qjlNx1.ref);
        compareSampling(ref, ref_qjlNx1);

        std.log.info("***** Init QJL 2 bit quantizer", .{});
        var quant_qjl2: QuantizationQJL2 = try .init(zml_handler, &lm_head);
        defer QuantizationQJL2.deinit(&quant_qjl2);
        try quant_qjl2.quantize();

        std.log.info("***** Init QJL 2 bit sampler", .{});
        var qjl2_sampler: QJL2Sampler = try .init(zml_handler, &lm_head, &quant_qjl2);
        defer qjl2_sampler.deinit();

        std.log.info("***** Test QJL 2 bit sampling", .{});
        const ref_qjl2 = try computeSamplingReference(zml_handler, &qjl2_sampler);
        defer zml_handler.allocator.free(ref_qjl2.ref);
        compareSampling(ref, ref_qjl2);

        std.log.info("***** Init int4 quantizer", .{});
        var quant_int4: QuantizationInt4 = try .init(zml_handler, &lm_head);
        defer QuantizationInt4.deinit(&quant_int4);
        try quant_int4.quantize();

        std.log.info("***** Init int4 sampler", .{});
        var int4_sampler: Int4Sampler = try .init(zml_handler, &lm_head, &quant_int4);
        defer int4_sampler.deinit();

        std.log.info("***** Test int4 sampling", .{});
        const ref_int4 = try computeSamplingReference(zml_handler, &int4_sampler);
        defer zml_handler.allocator.free(ref_int4.ref);
        compareSampling(ref, ref_int4);

        std.log.info("***** Init int8x4 sampler", .{});
        var int8x4_sampler: Int8x4Sampler = try .init(zml_handler, &lm_head, &quant_int4);
        defer int8x4_sampler.deinit();

        std.log.info("***** Test int8x4 sampling", .{});
        const ref_int8x4 = try computeSamplingReference(zml_handler, &int8x4_sampler);
        defer zml_handler.allocator.free(ref_int8x4.ref);
        compareSampling(ref, ref_int8x4);

        std.log.info("***** Init int8 quantizer", .{});
        var quant_int8: QuantizationInt8 = try .init(zml_handler, &lm_head);
        defer QuantizationInt8.deinit(&quant_int8);
        try quant_int8.quantize();

        std.log.info("***** Init int8 sampler", .{});
        var int8_sampler: Int8Sampler = try .init(zml_handler, &lm_head, &quant_int8);
        defer int8_sampler.deinit();

        std.log.info("***** Test int8 sampling", .{});
        const ref_int8 = try computeSamplingReference(zml_handler, &int8_sampler);
        defer zml_handler.allocator.free(ref_int8.ref);
        compareSampling(ref, ref_int8);

        std.log.info("***** Init graph sampler", .{});
        var graph_sampler: GraphSampler = try .init(zml_handler, &lm_head);
        defer graph_sampler.deinit();

        std.log.info("***** Test graph sampling", .{});
        const ref_graph = try computeSamplingReference(zml_handler, &graph_sampler);
        defer zml_handler.allocator.free(ref_graph.ref);
        compareSampling(ref, ref_graph);

        std.log.info("***** Init truncated sampler", .{});
        var truncated_sampler: TruncateSampler = try .init(zml_handler, &lm_head);
        defer truncated_sampler.deinit();

        std.log.info("***** Test truncated sampling", .{});
        const ref_tr = try computeSamplingReference(zml_handler, &truncated_sampler);
        defer zml_handler.allocator.free(ref_tr.ref);
        compareSampling(ref, ref_tr);

        std.log.info("***** Init vanilla FastScan PQ Quantizer", .{});
        var pq_iso_fc = try ProductQuantizerFastScan.init(zml_handler, &lm_head, .vanilla);
        defer pq_iso_fc.deinit();
        try pq_iso_fc.buildCodebook();

        std.log.info("***** Test vanilla FastScan PQ sampling", .{});
        const ref_pq_iso_fc = try computeSamplingReference(zml_handler, &pq_iso_fc);
        defer zml_handler.allocator.free(ref_pq_iso_fc.ref);
        compareSampling(ref, ref_pq_iso_fc);

        std.log.info("***** Init anisotropic FastScan PQ Quantizer", .{});
        var pq_aniso_fc = try ProductQuantizerFastScan.init(zml_handler, &lm_head, .anisotropic);
        defer pq_aniso_fc.deinit();
        try pq_aniso_fc.buildCodebook();

        std.log.info("***** Test anisotropic FastScan PQ sampling", .{});
        const ref_pq_aniso_fc = try computeSamplingReference(zml_handler, &pq_aniso_fc);
        defer zml_handler.allocator.free(ref_pq_aniso_fc.ref);
        compareSampling(ref, ref_pq_aniso_fc);

        std.log.info("***** Init vanilla PQ Quantizer", .{});
        var pq_iso = try ProductQuantizer.init(zml_handler, &lm_head, .vanilla);
        defer pq_iso.deinit();
        try pq_iso.buildCodebook();

        std.log.info("***** Test vanilla PQ sampling", .{});
        const ref_pq_iso = try computeSamplingReference(zml_handler, &pq_iso);
        defer zml_handler.allocator.free(ref_pq_iso.ref);
        compareSampling(ref, ref_pq_iso);

        std.log.info("***** Init anisotropic PQ Quantizer", .{});
        var pq_aniso = try ProductQuantizer.init(zml_handler, &lm_head, .anisotropic);
        defer pq_aniso.deinit();
        try pq_aniso.buildCodebook();

        std.log.info("***** Test anisotropic PQ sampling", .{});
        const ref_pq_aniso = try computeSamplingReference(zml_handler, &pq_aniso);
        defer zml_handler.allocator.free(ref_pq_aniso.ref);
        compareSampling(ref, ref_pq_aniso);

        std.log.info("***** Init angular sampler", .{});
        var sampler_ang: AngularSampler = try .init(zml_handler, &lm_head);
        defer AngularSampler.deinit(&sampler_ang);

        std.log.info("***** Test angular sampling", .{});
        const ref_ang = try computeSamplingReference(zml_handler, &sampler_ang);
        defer zml_handler.allocator.free(ref_ang.ref);
        compareSampling(ref, ref_ang);
    }
}

pub fn runLlm(zml_handler: *Zml_handler) !void {
    std.log.info("***** Get lm_head", .{});
    var lm_head = try algebra.getLmHead(zml_handler);
    defer LmHeadMatrix.deinit(&lm_head, zml_handler.allocator);

    std.log.info("***** Init QJL 1 bit quantizer", .{});
    var quantizer: QuantizationQJL1 = try .init(zml_handler, &lm_head);
    defer QuantizationQJL1.deinit(&quantizer);
    try quantizer.quantize();

    std.log.info("***** Init QJL 1 bit sampler", .{});
    var sampler: QJL1Sampler = try .init(zml_handler, &lm_head, &quantizer);
    defer sampler.deinit();

    std.log.info("***** Init LLM handler", .{});
    var llm = try llm_.Llm_handler.init(zml_handler);
    defer llm.deinit(zml_handler.allocator);

    std.log.info("***** Tokenize prompt", .{});
    const inspi_tokens = try inference.tokenizePrompt(zml_handler, llm.tokenizer);
    defer zml_handler.allocator.free(inspi_tokens);

    std.log.info("***** Generate text CPU sampling", .{});
    zml_handler.mem.start(0);
    const generated_text_cpu = try inference.generateTextCPUSampling(zml_handler, &llm, &sampler, inspi_tokens);
    defer zml_handler.allocator.free(generated_text_cpu);
    zml_handler.mem.check(0);

    std.log.info("***** Generate text GPU sampling", .{});
    zml_handler.mem.start(0);
    const generated_text_gpu = try inference.generateTextGPUSampling(zml_handler, &llm, inspi_tokens);
    defer zml_handler.allocator.free(generated_text_gpu);
    zml_handler.mem.check(0);
}


pub fn computeSamplingReference(zml_handler: *Zml_handler, sampler: anytype) !SamplingReference {
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
            const embed_slice = try save_load.getSlice(zml_handler, "qwen_embeds.safetensors", task, .f32, true);
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
            const embed_slice = try save_load.getSlice(zml_handler, "qwen_embeds2.safetensors", task, .f32, true);
            defer embed_slice.free(zml_handler.allocator);
            const n: usize = @intCast(embed_slice.shape.dims()[0]);
            total_embeds += n;
        }
    }

    const ref: SamplingReference = .{ .ref = try zml_handler.allocator.alloc(SamplingResult, total_embeds) };
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
            const embed_slice = try save_load.getSlice(zml_handler, "qwen_embeds.safetensors", task, .f32, true);
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
            const embed_slice = try save_load.getSlice(zml_handler, "qwen_embeds2.safetensors", task, .f32, true);
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

    const ref: SamplingReference = .{ .ref = try allocator.alloc(SamplingResult, n) };
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


pub fn compareSampling(ref: SamplingReference, other: SamplingReference) void {
    const total = @min(ref.ref.len, other.ref.len);
    if (ref.ref.len != other.ref.len) {
        std.log.warn("Compare sampling references with different lengths: ref={d} other={d} compared={d}", .{ ref.ref.len, other.ref.len, total });
    }
    if (total == 0) {
        std.log.warn("Compare sampling references: no samples", .{});
        return;
    }

    var top1_found_count: usize = 0;

    var small_topp_cases: usize = 0;
    var small_topp_all_found_count: usize = 0;

    var large_topp_cases: usize = 0;
    var large_topp_token_count: usize = 0;
    var large_topp_found_token_count: usize = 0;

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

        total_topp_token_count += topp_count;
        total_topp_found_token_count += found_topp_count;
        total_found_topp_mass += found_topp_mass;
        if (found_topp_count == 0) {
            full_fail_count += 1;
        }

        if (topp_count < 6) {
            small_topp_cases += 1;
            if (found_topp_count == topp_count) {
                small_topp_all_found_count += 1;
            }
        } else {
            large_topp_cases += 1;
            large_topp_token_count += topp_count;
            large_topp_found_token_count += found_topp_count;
        }
    }

    const inv_total = 1.0 / @as(f64, @floatFromInt(total));
    const top1_found_percent = 100.0 * @as(f64, @floatFromInt(top1_found_count)) * inv_total;
    const small_all_found_percent = percent(small_topp_all_found_count, small_topp_cases);
    const large_found_percent = percent(large_topp_found_token_count, large_topp_token_count);
    const total_topp_found_percent = percent(total_topp_found_token_count, total_topp_token_count);
    const average_found_topp_mass = total_found_topp_mass * inv_total;
    const full_fail_percent = 100.0 * @as(f64, @floatFromInt(full_fail_count)) * inv_total;
    const proba_success_percent = 100.0 * @as(f64, @floatFromInt(proba_success_count)) * inv_total;
    const proba_fail_percent = 100.0 * @as(f64, @floatFromInt(proba_fail_count)) * inv_total;

    std.log.info("Compare sampling: samples={d}", .{total});
    std.log.info("ref top1 found in other candidates: {d}/{d} ({d:.4}%)", .{ top1_found_count, total, top1_found_percent });
    std.log.info("top-p size < 6 cases: {d}; all top-p tokens found: {d}/{d} ({d:.4}%)", .{ small_topp_cases, small_topp_all_found_count, small_topp_cases, small_all_found_percent });
    std.log.info("top-p size > 5 cases: {d}; top-p tokens found: {d}/{d} ({d:.4}%)", .{ large_topp_cases, large_topp_found_token_count, large_topp_token_count, large_found_percent });
    std.log.info("all top-p tokens found: {d}/{d} ({d:.4}%)", .{ total_topp_found_token_count, total_topp_token_count, total_topp_found_percent });
    std.log.info("average found top-p probability mass: {d:.6}", .{average_found_topp_mass});
    std.log.info("full fail frequency: {d}/{d} ({d:.4}%)", .{ full_fail_count, total, full_fail_percent });
    std.log.info("probability success frequency: {d}/{d} ({d:.4}%)", .{ proba_success_count, total, proba_success_percent });
    std.log.info("probability fail frequency: {d}/{d} ({d:.4}%)", .{ proba_fail_count, total, proba_fail_percent });
}

fn samplingTopPCount(sample: SamplingResult) usize {
    return @min(sample.nb + 1, sample.candidates.len);
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
