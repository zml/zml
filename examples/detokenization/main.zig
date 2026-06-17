const std = @import("std");
const zml = @import("zml");

const stdx = zml.stdx;
const log = std.log;

const graph = @import("graph.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const svd_ = @import("svd.zig");
const inference = @import("inference.zig");
const algebra = @import("algebra.zig");
const save_load = @import("saveload.zig");
const tokens = @import("tokens.zig");

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
    prune_pool: Field_timer = .{},

    prefill: Field_timer = .{},
    decode: Field_timer = .{},

    pub fn print(self: Timing_handler) void {
        std.log.info("Sim matrix    : {d:>6.2}s", .{@as(f64, @floatFromInt(self.similarity_matrix.nanoseconds)) / 1e9});
        std.log.info("Junk rows     : {d:>6.2}s", .{@as(f64, @floatFromInt(self.junk_rows.nanoseconds)) / 1e9});
        std.log.info("kNN graph ini : {d:>6.2}s", .{@as(f64, @floatFromInt(self.knn_graph.nanoseconds)) / 1e9});
        std.log.info("NSW graph ext : {d:>6.2}s", .{@as(f64, @floatFromInt(self.nsw_graph.nanoseconds)) / 1e9});
        std.log.info("Greedy search : {d:>6.2}s", .{@as(f64, @floatFromInt(self.greedy_search.nanoseconds)) / 1e9});
        std.log.info("Pruning pool  : {d:>6.2}s", .{@as(f64, @floatFromInt(self.prune_pool.nanoseconds)) / 1e9});
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

    //try runLlm(&zml_handler);
    try runTests(&zml_handler);

    zml_handler.timers.print();
}

pub fn runLlm(zml_handler: *Zml_handler) !void {
    var llm = try llm_.Llm_handler.init(zml_handler);
    defer llm.deinit(zml_handler.allocator);

    const inspi_tokens = try inference.tokenizePrompt(zml_handler, llm.tokenizer);
    defer zml_handler.allocator.free(inspi_tokens);

    zml_handler.mem.start(0);
    const inspi_result = try inference.generateText(zml_handler, &llm, inspi_tokens);
    defer zml_handler.allocator.free(inspi_result);
    zml_handler.mem.check(0);
}

pub fn runTests(zml_handler: *Zml_handler) !void {
    var model_handler = try model_.Model_handler.init(zml_handler);
    defer model_handler.deinit(zml_handler.allocator);
    defer model_handler.unloadBuffers();

    std.log.info("Compute SVD", .{});
    const u, const diag = try algebra.loadSvd(zml_handler, &model_handler);
    defer u.free(zml_handler.allocator);
    defer diag.free(zml_handler.allocator);
    for (diag.constItems(f64), 0..) |s, i| {
        if (i > 15) {
            std.log.info("...", .{});
            break;
        }
        std.log.info("singular value: {d}", .{@sqrt(s)});
    }
    
    std.log.info("Get lm_head_rotated", .{});
    const lm_head_rotated, const lm_head_rotated_tr = try algebra.getLmHeadRotated(zml_handler, &model_handler, u);
    defer lm_head_rotated.free(zml_handler.allocator);
    defer lm_head_rotated_tr.free(zml_handler.allocator);

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
    var tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();
    
    std.log.info("Init SVD sampler", .{});
    var svd_sampler: svd_.SvdSampler = try .init(zml_handler.allocator, lm_head_rotated, lm_head_rotated_tr, tokenizer);
    defer svd_sampler.deinit();

    //try testTokenSvdSearch(zml_handler, lm_head_rotated, &svd_sampler);
    try testEmbedSvdSearch(zml_handler, &svd_sampler, u);
    
    //try analyzeTopRows(zml_handler, &model_handler);

    zml_handler.tic(&zml_handler.timers.similarity_matrix);
    //var similarity_matrix = try computeSimilarityMatrix(zml_handler, &model_handler; true);
    var similarity_matrix = try algebra.loadSimilarityMatrix(zml_handler, &model_handler, true);
    defer similarity_matrix.deinit(zml_handler.allocator);
    zml_handler.toc(&zml_handler.timers.similarity_matrix);

    //try testSimilarityMatrix(zml_handler, &model_handler, &similarity_matrix, true);

    std.log.info("Get lm_head", .{});
    const lm_head = try algebra.getLmHead(zml_handler, &model_handler);
    defer lm_head.free(zml_handler.allocator);

    std.log.info("Get lm_head_normalized", .{});
    const lm_head_normalized = try algebra.getLmHeadNormalized(zml_handler, &model_handler);
    defer lm_head_normalized.free(zml_handler.allocator);

    std.log.info("Get lm_head row norms", .{});
    const lm_head_row_norms = try algebra.getLmHeadRowNorms(zml_handler, &model_handler);
    defer lm_head_row_norms.free(zml_handler.allocator);

    std.log.info("Get junk rows", .{});
    zml_handler.tic(&zml_handler.timers.junk_rows);
    const junk_rows = try algebra.getJunkRows(zml_handler, &model_handler);
    defer zml_handler.allocator.free(junk_rows);
    zml_handler.toc(&zml_handler.timers.junk_rows);
    std.log.info("Found {d} junk rows", .{junk_rows.len});

    std.log.info("Get medoid", .{});
    const medoid = 0; //try getMedoid(zml_handler, &model_handler, junk_rows);
    std.log.info("Medoid: {d}", .{medoid});

    std.log.info("Get MRT insertion order", .{});
    const mrt_order = try algebra.getMrtOrder(zml_handler, &model_handler);
    defer zml_handler.allocator.free(mrt_order);

    std.log.info("Init MRT", .{});
    const mrt_params: graph.GraphParams = .{ .k_max = 1024 };
    var mrt: graph.Graph = try .init(zml_handler, lm_head, lm_head_normalized, &similarity_matrix, lm_head_row_norms, junk_rows, medoid, mrt_params);
    defer mrt.deinit();
    try mrt.makeMrt(mrt_order);
    std.log.info("Exact MRT : nb edges: {d}", .{mrt.nbEdges()});

    std.log.info("Init graph", .{});
    const graph_params: graph.GraphParams = .{};
    //var g: graph.Graph = try .init(zml_handler, lm_head, lm_head_normalized, &similarity_matrix, lm_head_row_norms, junk_rows, medoid, graph_params);
    var g: graph.Graph = try .fromFile(zml_handler, lm_head, lm_head_normalized, &similarity_matrix, lm_head_row_norms, "nsw-knn-16.safetensors", graph_params);
    defer g.deinit();

    zml_handler.tic(&zml_handler.timers.knn_graph);
    //g.setNearestNeighbors();
    zml_handler.toc(&zml_handler.timers.knn_graph);

    std.log.info("Exact kNN : nb edges: {d}", .{g.nbEdges()});

    zml_handler.tic(&zml_handler.timers.nsw_graph);
    //g.extendToNsw();
    zml_handler.toc(&zml_handler.timers.nsw_graph);

    std.log.info("NSW extension : nb edges: {d}", .{g.nbEdges()});
    //g.testNswExtention();

    try testEmbedGraphSearch(zml_handler, &model_handler, &mrt, false);
}

pub fn run(zml_handler: *Zml_handler) !void {
    var model_handler = try model_.Model_handler.init(zml_handler);
    defer model_handler.deinit(zml_handler.allocator);
    defer model_handler.unloadBuffers();

    zml_handler.tic(&zml_handler.timers.similarity_matrix);
    var similarity_matrix = try algebra.computeSimilarityMatrix(zml_handler, &model_handler);
    defer similarity_matrix.deinit(zml_handler.allocator);
    zml_handler.toc(&zml_handler.timers.similarity_matrix);

    std.log.info("Get lm_head", .{});
    const lm_head = try algebra.getLmHead(zml_handler, &model_handler);
    defer lm_head.free(zml_handler.allocator);

    std.log.info("Get lm_head_normalized", .{});
    const lm_head_normalized = try algebra.getLmHeadNormalized(zml_handler, &model_handler);
    defer lm_head_normalized.free(zml_handler.allocator);

    std.log.info("Get lm_head row norms", .{});
    const lm_head_row_norms = try algebra.getLmHeadRowNorms(zml_handler, &model_handler);
    defer lm_head_row_norms.free(zml_handler.allocator);

    std.log.info("Get junk rows", .{});
    zml_handler.tic(&zml_handler.timers.junk_rows);
    const junk_rows = try algebra.getJunkRows(zml_handler, &model_handler);
    defer zml_handler.allocator.free(junk_rows);
    zml_handler.toc(&zml_handler.timers.junk_rows);
    std.log.info("Found {d} junk rows", .{junk_rows.len});

    std.log.info("Get medoid", .{});
    const medoid = try algebra.getMedoid(zml_handler, &model_handler, junk_rows);
    std.log.info("Medoid: {d}", .{medoid});

    std.log.info("Init graph", .{});
    const graph_params: graph.GraphParams = .{};
    var g: graph.Graph = try .init(zml_handler, lm_head, lm_head_normalized, &similarity_matrix, lm_head_row_norms, junk_rows, medoid, graph_params);
    defer g.deinit();

    zml_handler.tic(&zml_handler.timers.knn_graph);
    g.setNearestNeighbors();
    zml_handler.toc(&zml_handler.timers.knn_graph);
    std.log.info("Exact kNN : nb edges: {d}", .{g.nbEdges()});

    zml_handler.tic(&zml_handler.timers.nsw_graph);
    g.extendToNsw();
    zml_handler.toc(&zml_handler.timers.nsw_graph);
    std.log.info("NSW extension : nb edges: {d}", .{g.nbEdges()});

    var llm = try llm_.Llm_handler.init(zml_handler);
    defer llm.deinit(zml_handler.allocator);

    const inspi_tokens = try inference.tokenizePrompt(zml_handler, llm.tokenizer);
    defer zml_handler.allocator.free(inspi_tokens);

    zml_handler.mem.start(0);
    const inspi_result = try inference.generateTextGraph(zml_handler, &llm, &model_handler, &g, inspi_tokens);
    defer zml_handler.allocator.free(inspi_result);
    zml_handler.mem.check(0);
}


pub fn testTokenGraphSearch(lm_head: zml.Slice, g: *graph.Graph) void {
    std.log.info("Test token graph search", .{});
    const n: usize = @intCast(lm_head.shape.dim(.voc));
    const d: usize = @intCast(lm_head.shape.dim(.d));
    std.debug.assert(n == g.n);
    std.debug.assert(d == g.dim);

    const rows = lm_head.constItems(f32);
    var exact_first_count: usize = 0;
    var non_junk_count: usize = 0;
    var junk_count: usize = 0;
    var total_visited: usize = 0;
    var min_visited: usize = std.math.maxInt(usize);
    var max_visited: usize = 0;

    for (0..n) |row_id| {
        const query = rows[row_id * d ..][0..d];
        g.greedySearch(query);
        const nb_visited = g.nb_visited;
        total_visited += nb_visited;
        min_visited = @min(min_visited, nb_visited);
        max_visited = @max(max_visited, nb_visited);

        if (g.is_junk[row_id]) {
            junk_count += 1;
        } else {
            non_junk_count += 1;
            if (g.L > 0 and g.visited[0].node == row_id) {
                exact_first_count += 1;
            }
        }

        if (row_id == 0 or (row_id + 1) % 10000 == 0 or row_id + 1 == n) {
            std.log.info("Token graph search row {d}/{d}", .{ row_id + 1, n });
        }
    }

    const avg_visited = @as(f64, @floatFromInt(total_visited)) / @as(f64, @floatFromInt(n));
    const exact_rate = if (non_junk_count == 0) 0.0 else @as(f64, @floatFromInt(exact_first_count)) / @as(f64, @floatFromInt(non_junk_count));
    std.log.info(
        "Token graph search: total={d} non_junk={d} junk={d} exact_first={d}/{d} ({d:.4}%) nb_visited min={d} max={d} avg={d:.2}",
        .{
            n,
            non_junk_count,
            junk_count,
            exact_first_count,
            non_junk_count,
            100.0 * exact_rate,
            min_visited,
            max_visited,
            avg_visited,
        },
    );
}

pub fn testTokenSvdSearch(_: *Zml_handler, lm_head_rot: zml.Slice, svd: *svd_.SvdSampler) !void {
    std.log.info("Test token SVD search", .{});
    const n: usize = @intCast(lm_head_rot.shape.dim(.voc));
    const d: usize = @intCast(lm_head_rot.shape.dim(.d));

    const rows = lm_head_rot.constItems(f32);
    var rand = std.Random.DefaultPrng.init(1);
    const rng = std.Random.Xoshiro256.random(&rand);

    for (0..n) |row_id| {
        const query = rows[row_id * d ..][0..d];
        const real_tok = try svd.sampleFull(query, rng);
        const safe_tok = 99;//try svd.sampleSafe(query, rng);
        const unsafe_tok = try svd.sampleUnsafe(query, rng);
        
        std.log.info("SVD sample: query={d} real={d} safe={d} unsafe={d}", .{ row_id, real_tok, safe_tok, unsafe_tok });
    }
}


pub fn testEmbedGraphSearch(zml_handler: *Zml_handler, model_handler: *model_.Model_handler, g: *graph.Graph, log_each_search: bool) !void {
    const top_k = 16;
    std.log.info("Test embed graph search", .{});

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
    var tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    const embeds_repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.checkpoint);
    var registry: zml.safetensors.TensorRegistry = try .fromRepoFile(zml_handler.allocator, zml_handler.io, embeds_repo, "qwen_embeds.safetensors");
    defer registry.deinit();

    const d = g.dim;
    const vocab_size = g.n;
    const embed_slice = try zml.Slice.alloc(zml_handler.allocator, .init(.{ .s = 1, .d = d }, .f32));
    defer embed_slice.free(zml_handler.allocator);
    const embed_items = embed_slice.items(f32);

    const prob_by_token = try zml_handler.allocator.alloc(f32, vocab_size);
    defer zml_handler.allocator.free(prob_by_token);

    var total_count: usize = 0;
    var found_top1_count: usize = 0;
    var graph_first_is_top1_count: usize = 0;
    var total_real_top16_mass: f64 = 0.0;
    var total_graph_top16_mass: f64 = 0.0;
    var total_mass_ratio: f64 = 0.0;
    var min_mass_ratio: f64 = std.math.inf(f64);
    var max_mass_ratio: f64 = 0.0;
    var total_visited: usize = 0;
    var min_visited: usize = std.math.maxInt(usize);
    var max_visited: usize = 0;

    var registry_it = registry.iterator();
    while (registry_it.next()) |entry| {
        const task_name = entry.key_ptr.*;
        const task_embeds = try save_load.loadSafetensorSliceFromRegistry(zml_handler, &registry, task_name);
        defer task_embeds.free(zml_handler.allocator);

        const embed_count = embeddingCount(task_embeds, d) orelse return error.InvalidEmbeddingShape;
        std.log.info("Test embed graph search task={s} embeddings={d} shape={f}", .{ task_name, embed_count, task_embeds.shape });
        for (0..embed_count) |embed_index| {
            copyEmbedding(task_embeds, d, embed_index, embed_items);

            var embed_buffer = try zml.Buffer.fromSlice(zml_handler.io, zml_handler.platform, embed_slice, .replicated);
            defer embed_buffer.deinit();

            model_handler.exes.score_args.set(.{ model_handler.model_buffers, embed_buffer });
            model_handler.exes.score_exe.call(model_handler.exes.score_args, &model_handler.exes.score_results);

            var sorted_probas_buffer: zml.Buffer = undefined;
            var sorted_indices_buffer: zml.Buffer = undefined;
            var sorted_similarities_buffer: zml.Buffer = undefined;
            model_handler.exes.score_results.fill(.{ &sorted_probas_buffer, &sorted_indices_buffer, &sorted_similarities_buffer });
            defer sorted_probas_buffer.deinit();
            defer sorted_indices_buffer.deinit();
            defer sorted_similarities_buffer.deinit();

            const sorted_probas_slice = try sorted_probas_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
            defer sorted_probas_slice.free(zml_handler.allocator);
            const sorted_indices_slice = try sorted_indices_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
            defer sorted_indices_slice.free(zml_handler.allocator);
            const sorted_similarities_slice = try sorted_similarities_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
            defer sorted_similarities_slice.free(zml_handler.allocator);

            const sorted_probas = sorted_probas_slice.constItems(f32);
            const sorted_indices = sorted_indices_slice.constItems(i32);
            const sorted_similarities = sorted_similarities_slice.constItems(f32);
            std.debug.assert(sorted_probas.len == vocab_size);
            std.debug.assert(sorted_indices.len == vocab_size);
            std.debug.assert(sorted_similarities.len == vocab_size);

            for (sorted_indices, sorted_probas) |token_id, proba| {
                prob_by_token[@intCast(token_id)] = proba;
            }

            if (log_each_search) {
                std.log.info("Embed graph search task={s} index={d}", .{ task_name, embed_index });
                try g.greedySearchWithLog(embed_items, tokenizer);
            } else {
                g.greedySearch(embed_items);
            }

            const nb_visited = g.nb_visited;
            total_visited += nb_visited;
            min_visited = @min(min_visited, nb_visited);
            max_visited = @max(max_visited, nb_visited);

            const real_top1: usize = @intCast(sorted_indices[0]);
            const graph_top_count = @min(g.L, top_k);
            var real_top16_mass: f64 = 0.0;
            for (0..@min(top_k, sorted_probas.len)) |i| {
                real_top16_mass += sorted_probas[i];
            }
            var graph_top16_mass: f64 = 0.0;
            var found_top1 = false;
            for (0..graph_top_count) |i| {
                const node = g.visited[i].node;
                graph_top16_mass += prob_by_token[node];
                if (node == real_top1) found_top1 = true;
            }
            if (found_top1) found_top1_count += 1;
            if (g.L > 0 and g.visited[0].node == real_top1) graph_first_is_top1_count += 1;
            const mass_ratio = graph_top16_mass / real_top16_mass;
            total_real_top16_mass += real_top16_mass;
            total_graph_top16_mass += graph_top16_mass;
            total_mass_ratio += mass_ratio;
            min_mass_ratio = @min(min_mass_ratio, mass_ratio);
            max_mass_ratio = @max(max_mass_ratio, mass_ratio);
            total_count += 1;

            if (log_each_search) {
                try logEmbedGraphSampling(tokenizer, embed_items, g, sorted_probas, sorted_indices, sorted_similarities, prob_by_token, top_k);
                std.log.info("Embed graph search stats task={s} index={d} real_top16_mass={d:.8} graph_top16_mass={d:.8} ratio={d:.8} found_top1={}", .{
                    task_name,
                    embed_index,
                    real_top16_mass,
                    graph_top16_mass,
                    mass_ratio,
                    found_top1,
                });
            }
        }
    }

    const inv_total = if (total_count == 0) 0.0 else 1.0 / @as(f64, @floatFromInt(total_count));
    std.log.info(
        "Embed graph search: total={d} found_top1_in_graph_top16={d}/{d} ({d:.4}%) graph_first_is_top1={d}/{d} ({d:.4}%)",
        .{
            total_count,
            found_top1_count,
            total_count,
            100.0 * @as(f64, @floatFromInt(found_top1_count)) * inv_total,
            graph_first_is_top1_count,
            total_count,
            100.0 * @as(f64, @floatFromInt(graph_first_is_top1_count)) * inv_total,
        },
    );
    std.log.info(
        "Embed graph search top16 mass: real_avg={d:.8} graph_avg={d:.8} ratio_min={d:.8} ratio_max={d:.8} ratio_avg={d:.8}",
        .{
            total_real_top16_mass * inv_total,
            total_graph_top16_mass * inv_total,
            if (total_count == 0) 0.0 else min_mass_ratio,
            max_mass_ratio,
            total_mass_ratio * inv_total,
        },
    );
    std.log.info(
        "Embed graph search nb_visited: min={d} max={d} avg={d:.2}",
        .{
            if (total_count == 0) 0 else min_visited,
            max_visited,
            @as(f64, @floatFromInt(total_visited)) * inv_total,
        },
    );
}

pub fn testEmbedSvdSearch(zml_handler: *Zml_handler, svd: *svd_.SvdSampler, u: zml.Slice) !void {
    std.log.info("Test embed SVD search", .{});
    const d = svd.d;
    std.debug.assert(u.dtype() == .f32);
    std.debug.assert(u.shape.count() == d * d);

    const embeds_repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.checkpoint);
    var registry: zml.safetensors.TensorRegistry = try .fromRepoFile(zml_handler.allocator, zml_handler.io, embeds_repo, "qwen_embeds.safetensors");
    defer registry.deinit();

    const embed = try zml_handler.allocator.alloc(f32, d);
    defer zml_handler.allocator.free(embed);
    const rot_embed = try zml_handler.allocator.alloc(f32, d);
    defer zml_handler.allocator.free(rot_embed);

    var rand = std.Random.DefaultPrng.init(1);
    const rng = std.Random.Xoshiro256.random(&rand);

    var total_count: usize = 0;
    var registry_it = registry.iterator();
    while (registry_it.next()) |entry| {
        const task_name = entry.key_ptr.*;
        const task_embeds = try save_load.loadSafetensorSliceFromRegistry(zml_handler, &registry, task_name);
        defer task_embeds.free(zml_handler.allocator);

        const embed_count = embeddingCount(task_embeds, d) orelse return error.InvalidEmbeddingShape;
        std.log.info("Test embed SVD search task={s} embeddings={d} shape={f}", .{ task_name, embed_count, task_embeds.shape });
        for (0..embed_count) |embed_index| {
            copyEmbedding(task_embeds, d, embed_index, embed);
            rotateEmbedding(u, embed, rot_embed);
            std.log.info("\n\n\n", .{});
            std.log.info("#################", .{});
            std.log.info("#### new emb ####", .{});
            std.log.info("#################", .{});
            const real_tok = try svd.sampleFull(rot_embed, rng);
            const safe_tok = 99; //try svd.sampleSafe(rot_embed, rng);
            const unsafe_tok = try svd.sampleUnsafe(rot_embed, rng);
            const trunc_tok = 99;//try svd.sampleTruncated(rot_embed, rng, 256);

            std.log.info("SVD embed sample: task={s} index={d} real={d} safe={d} unsafe={d} trunc={d}", .{ task_name, embed_index, real_tok, safe_tok, unsafe_tok, trunc_tok });
            total_count += 1;
        }
    }
    std.log.info("Embed SVD search: total={d}", .{total_count});
}


fn embeddingCount(embeds: zml.Slice, d: usize) ?usize {
    const dims = embeds.shape.dims();
    return switch (embeds.shape.rank()) {
        1 => if (@as(usize, @intCast(dims[0])) == d) 1 else null,
        2 => if (@as(usize, @intCast(dims[0])) == d) @intCast(dims[1]) else if (@as(usize, @intCast(dims[1])) == d) @intCast(dims[0]) else null,
        else => null,
    };
}

fn copyEmbedding(embeds: zml.Slice, d: usize, embed_index: usize, out: []f32) void {
    std.debug.assert(out.len == d);
    const dims = embeds.shape.dims();
    const rank = embeds.shape.rank();
    const dim_is_first = switch (rank) {
        1 => true,
        2 => @as(usize, @intCast(dims[0])) == d,
        else => unreachable,
    };
    const embed_count: usize = switch (rank) {
        1 => 1,
        2 => if (dim_is_first) @intCast(dims[1]) else @intCast(dims[0]),
        else => unreachable,
    };
    std.debug.assert(embed_index < embed_count);

    switch (embeds.dtype()) {
        .f32 => copyEmbeddingTyped(f32, embeds.constItems(f32), rank, dim_is_first, d, embed_count, embed_index, out),
        .bf16 => copyEmbeddingTyped(zml.floats.BFloat16, embeds.constItems(zml.floats.BFloat16), rank, dim_is_first, d, embed_count, embed_index, out),
        .f16 => copyEmbeddingTyped(f16, embeds.constItems(f16), rank, dim_is_first, d, embed_count, embed_index, out),
        else => unreachable,
    }
}

fn rotateEmbedding(u: zml.Slice, embed: []const f32, rot_embed: []f32) void {
    const d = embed.len;
    std.debug.assert(rot_embed.len == d);
    std.debug.assert(u.shape.count() == d * d);

    const u_items = u.constItems(f32);
    for (0..d) |col| {
        var value: f32 = 0.0;
        for (0..d) |row| {
            value += u_items[row * d + col] * embed[row];
        }
        rot_embed[col] = value;
    }
}

fn copyEmbeddingTyped(comptime T: type, items: []const T, rank: u4, dim_is_first: bool, d: usize, embed_count: usize, embed_index: usize, out: []f32) void {
    switch (rank) {
        1 => {
            for (0..d) |i| out[i] = scalarToF32(items[i]);
        },
        2 => {
            if (dim_is_first) {
                for (0..d) |i| out[i] = scalarToF32(items[i * embed_count + embed_index]);
            } else {
                const start = embed_index * d;
                for (0..d) |i| out[i] = scalarToF32(items[start + i]);
            }
        },
        else => unreachable,
    }
}

fn scalarToF32(value: anytype) f32 {
    const T = @TypeOf(value);
    return if (T == zml.floats.BFloat16) value.toF32() else @floatCast(value);
}

fn logEmbedGraphSampling(tokenizer: zml.tokenizer.Tokenizer, embedding: []const f32, g: *graph.Graph, sorted_probas: []const f32, sorted_indices: []const i32, sorted_similarities: []const f32, prob_by_token: []const f32, top_k: usize) !void {
    const row_norms = g.lm_head_row_norms.constItems(f32);
    const nb_real = @min(sorted_probas.len, top_k);
    std.log.info("Real sampling distribution (top {d})", .{nb_real});
    printEmbedSamplingHeader();
    for (0..nb_real) |i| {
        const token_id: usize = @intCast(sorted_indices[i]);
        try printEmbedSamplingRow(tokenizer, i + 1, token_id, g.is_junk[token_id], sorted_probas[i], row_norms[token_id], sorted_similarities[i]);
    }

    const embedding_norm = embeddingNorm(embedding);
    const nb_graph = @min(g.L, top_k);
    std.log.info("Graph sampling distribution over {d} visited nodes (top {d}, full probabilities)", .{ g.L, nb_graph });
    printEmbedSamplingHeader();
    for (0..nb_graph) |i| {
        const token_id = g.visited[i].node;
        try printEmbedSamplingRow(tokenizer, i + 1, token_id, g.is_junk[token_id], prob_by_token[token_id], row_norms[token_id], g.visited[i].similarity / embedding_norm);
    }
}

fn printEmbedSamplingHeader() void {
    std.log.info("{s:>6}  {s:>10}  {s:>7}  {s:>14}  {s:>14}  {s:>14}  {s}", .{ "rank", "token_id", "is_junk", "proba", "row_norm", "similarity", "token" });
    std.log.info("{s:>6}  {s:>10}  {s:>7}  {s:>14}  {s:>14}  {s:>14}  {s}", .{ "------", "----------", "-------", "--------------", "--------------", "--------------", "-----" });
}

fn printEmbedSamplingRow(tokenizer: zml.tokenizer.Tokenizer, rank: usize, token_id: usize, is_junk: bool, proba: f32, row_norm: f32, similarity: f32) !void {
    var decoded_buf: [512]u8 = undefined;
    const decoded = try tokens.decodeToken(tokenizer, @intCast(token_id), &decoded_buf);
    var escaped_buf: [512]u8 = undefined;
    const escaped = tokens.escapeTokenText(decoded, &escaped_buf);
    std.log.info("{d:>6}  {d:>10}  {d:>7}  {d:>14.8}  {d:>14.6}  {d:>14.8}  {s}", .{ rank, token_id, @intFromBool(is_junk), proba, row_norm, similarity, escaped });
}

fn embeddingNorm(embedding: []const f32) f32 {
    var norm2: f32 = 0.0;
    for (embedding) |x| norm2 += x * x;
    return @sqrt(norm2);
}
