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
const sampling = @import("sampling.zig");

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
    nb_tictoc: usize = 0,
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
        self.nb_tictoc += 1;
        target.init = std.Io.Timestamp.now(self.io, .awake);
    }

    pub fn toc(self: *Zml_handler, target: *Timing_handler.Field_timer) void {
        self.nb_tictoc += 1;
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
    prune_pool_fwd: Field_timer = .{},
    prune_pool_bwd: Field_timer = .{},

    insert_node: Field_timer = .{},

    graph_search_tot: Field_timer = .{},

    embed_search: Field_timer = .{},
    embed_dot: Field_timer = .{},

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
        std.log.info("Graph search  : {d:>6.2}s", .{@as(f64, @floatFromInt(self.graph_search_tot.nanoseconds)) / 1e9});
        std.log.info("Embed search  : {d:>6.2}s", .{@as(f64, @floatFromInt(self.embed_search.nanoseconds)) / 1e9});
        std.log.info("Embed dot     : {d:>6.2}s", .{@as(f64, @floatFromInt(self.embed_dot.nanoseconds)) / 1e9});
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
    try runTestsGraph(&zml_handler);

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

pub fn runTestsSvd(zml_handler: *Zml_handler) !void {
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
    var svd_sampler: svd_.SvdSampler = try .init(zml_handler.allocator, lm_head_rotated, lm_head_rotated_tr, tokenizer, false);
    defer svd_sampler.deinit();

    //try testTokenSvdSearch(zml_handler, lm_head_rotated, &svd_sampler);
    try testEmbedSvdSearch(zml_handler, &svd_sampler, u);
}

pub fn runTestsGraph(zml_handler: *Zml_handler) !void {
    var model_handler = try model_.Model_handler.init(zml_handler);
    defer model_handler.deinit(zml_handler.allocator);
    defer model_handler.unloadBuffers();

    zml_handler.tic(&zml_handler.timers.similarity_matrix);
    //var similarity_matrix = try algebra.computeSimilarityMatrix(zml_handler, &model_handler, false);
    var similarity_matrix = try algebra.loadSimilarityMatrix(zml_handler, &model_handler, false);
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

    std.log.info("Init sampler", .{});
    var sampler: sampling.Sampler = try .init(zml_handler, lm_head);
    defer sampling.Sampler.deinit(&sampler);

    std.log.info("Init graph", .{});
    const graph_params: graph.GraphParams = .{};
    var g: graph.Graph = try .init(zml_handler, lm_head, lm_head_normalized, &similarity_matrix, lm_head_row_norms, junk_rows, medoid, graph_params);
    //var g: graph.Graph = try .fromFile(zml_handler, lm_head, lm_head_normalized, &similarity_matrix, lm_head_row_norms, "nsw-knn-16.safetensors", graph_params);
    defer g.deinit();

    //const active_parents = try g.coarsify(zml_handler, 0.5);
    //defer zml_handler.allocator.free(active_parents);
    //try testEmbedCoarseSearch(zml_handler, &sampler, active_parents);

    zml_handler.tic(&zml_handler.timers.knn_graph);
    g.setNearestNeighbors(256);
    zml_handler.toc(&zml_handler.timers.knn_graph);

    try g.pruneNeighbors(1.5);

    zml_handler.tic(&zml_handler.timers.nsw_graph);
    try g.extendToNsw();
    //try g.extendNswSparseQueries();
    zml_handler.toc(&zml_handler.timers.nsw_graph);

    //try g.testNswExtention(&sampler);
    //try g.testNwsExtensionSparse();

    //g.consolidateNswNearest();
    //g.consolidateNswPrune();
    
    try g.testNswExtention(&sampler);
    //try g.testNwsExtensionSparse();

    zml_handler.tic(&zml_handler.timers.graph_search_tot);
    try testEmbedGraphSearch(zml_handler, &g, &sampler);
    zml_handler.toc(&zml_handler.timers.graph_search_tot);
}

pub fn runTestsMrt(zml_handler: *Zml_handler) !void {
    var model_handler = try model_.Model_handler.init(zml_handler);
    defer model_handler.deinit(zml_handler.allocator);
    defer model_handler.unloadBuffers();

    zml_handler.tic(&zml_handler.timers.similarity_matrix);
    //var similarity_matrix = try algebra.computeSimilarityMatrix(zml_handler, &model_handler, true);
    var similarity_matrix = try algebra.loadSimilarityMatrix(zml_handler, &model_handler, true);
    defer similarity_matrix.deinit(zml_handler.allocator);
    zml_handler.toc(&zml_handler.timers.similarity_matrix);

    //try testSimilarityMatrix(zml_handler, &model_handler, &similarity_matrix, true);

    std.log.info("Get lm_head", .{});
    const lm_head, const lm_head_translation = try algebra.getLmHead(zml_handler, &model_handler);
    defer lm_head.free(zml_handler.allocator);
    defer lm_head_translation.free(zml_handler.allocator);

    std.log.info("Get lm_head_normalized", .{});
    const lm_head_normalized, const lm_head_normalized_translation = try algebra.getLmHeadNormalized(zml_handler, &model_handler);
    defer lm_head_normalized.free(zml_handler.allocator);
    defer lm_head_normalized_translation.free(zml_handler.allocator);

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
        const safe_tok = 99; //try svd.sampleSafe(query, rng);
        const unsafe_tok = try svd.sampleUnsafe(query, rng);

        std.log.info("SVD sample: query={d} real={d} safe={d} unsafe={d}", .{ row_id, real_tok, safe_tok, unsafe_tok });
    }
}

pub fn testEmbedGraphSearch(zml_handler: *Zml_handler, g: *graph.Graph, sampler: *sampling.Sampler) !void {
    std.log.info("Test embed graph search", .{});

    var total_count: usize = 0;
    var found_top1_count: usize = 0;
    var missed_top1_nsw_extension_missed_count: usize = 0;
    var missed_top1_in_visited_tail_count: usize = 0;

    var total_visited: usize = 0;
    var min_visited: usize = std.math.maxInt(usize);
    var max_visited: usize = 0;

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
        const top1 = switch (task_id) {
            0 => codingTop1(),
            1 => historyTop1(),
            2 => mathTop1(),
            3 => storyTop1(),
            4 => translateTop1(),
            else => return error.InvalidTask,
        };

        const embed_slice = try save_load.getSlice(zml_handler, "qwen_embeds.safetensors", task, .f32, true);
        defer embed_slice.free(zml_handler.allocator);

        const n: usize = @intCast(embed_slice.shape.dims()[0]);
        const d: usize = @intCast(embed_slice.shape.dims()[1]);
        std.debug.assert(d == g.dim);

        std.log.info("Test embed graph search task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        for (0..n) |embed_index| {
            const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];

            g.greedySearch(embed);

            const nb_visited = g.nb_visited;
            total_visited += nb_visited;
            min_visited = @min(min_visited, nb_visited);
            max_visited = @max(max_visited, nb_visited);

            total_count += 1;
            const top1_token = top1[embed_index];
            var found_top1 = false;
            for (0..g.L) |i| {
                if (g.visited[i].node == top1_token) found_top1 = true;
            }
            if (found_top1) {
                found_top1_count += 1;
            } else {
                if (g.nsw_extension_search_missed[top1_token]) missed_top1_nsw_extension_missed_count += 1;
                for (g.L..g.nb_visited) |i| {
                    if (g.visited[i].node == top1_token) {
                        missed_top1_in_visited_tail_count += 1;
                        break;
                    }
                }
                _ = sampler;
                //std.log.info("Missed top1, id {d} str {s}", .{ top1[embed_index], try tokens.tokenString(sampler.tokenizer, top1[embed_index], sampler.allocator) });
                //std.log.info("Found instead {d} str {s}", .{ g.visited[0].node, try tokens.tokenString(sampler.tokenizer, g.visited[0].node, sampler.allocator) });
            }
        }
    }

    const percent_found = 100.0 * @as(f64, @floatFromInt(found_top1_count)) / @as(f64, @floatFromInt(total_count));
    const average_visit = @as(f64, @floatFromInt(total_visited)) / @as(f64, @floatFromInt(total_count));
    std.log.info("Embed graph search: total={d} found_top1={d} ({d:.4}%)", .{ total_count, found_top1_count, percent_found });
    std.log.info(
        "Embed graph search misses: nsw_extension_missed={d} visited_tail={d}",
        .{ missed_top1_nsw_extension_missed_count, missed_top1_in_visited_tail_count },
    );
    std.log.info("Embed graph search nb_visited: min={d} max={d} avg={d:.2}", .{ min_visited, max_visited, average_visit });
}

pub fn testEmbedSvdSearch(zml_handler: *Zml_handler, svd: *svd_.SvdSampler, u: zml.Slice) !void {
    std.log.info("Test embed SVD search", .{});
    const d = svd.d;
    std.debug.assert(u.dtype() == .f32);
    std.debug.assert(u.shape.count() == d * d);

    const rot_embed = try zml_handler.allocator.alloc(f32, d);
    defer zml_handler.allocator.free(rot_embed);

    var rand = std.Random.DefaultPrng.init(1);
    const rng = std.Random.Xoshiro256.random(&rand);

    var total_count: usize = 0;
    const tasks = [_][]const u8{ "coding", "history", "math", "story", "translate" };

    for (tasks) |task| {
        const embed_slice = try save_load.getSlice(zml_handler, "qwen_embed.safetensors", task, .f32, true);
        defer embed_slice.free(zml_handler.allocator);
        const n: usize = @intCast(embed_slice.shape.dims()[0]);

        std.log.info("Test embed SVD search task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        for (0..n) |embed_index| {
            const embed = zml.Slice.init(.init(.{ .s = 1, .d = d }, .f32), embed_slice.constItems(u8)[4 * embed_index * d .. 4 * (embed_index + 1) * d]);
            rotateEmbedding(u, embed.constItems(f32), rot_embed);
            std.log.info("\n", .{});
            std.log.info("#################", .{});
            std.log.info("#### new emb ####", .{});
            std.log.info("#################", .{});
            const real_tok = try svd.sampleFull(rot_embed, rng);
            const safe_tok = 99; //try svd.sampleSafe(rot_embed, rng);
            const unsafe_tok = try svd.sampleUnsafe(rot_embed, rng);
            const trunc_tok = 99; //try svd.sampleTruncated(rot_embed, rng, 256);

            std.log.info("SVD embed sample: task={s} index={d} real={d} safe={d} unsafe={d} trunc={d}", .{ task, embed_index, real_tok, safe_tok, unsafe_tok, trunc_tok });
            total_count += 1;
        }
    }
    std.log.info("Embed SVD search: total={d}", .{total_count});
}

pub fn testEmbedCoarseSearch(zml_handler: *Zml_handler, sampler: *sampling.Sampler, active_parent: []usize) !void {
    std.log.info("\n###############", .{});
    std.log.info("Test embed coarse search", .{});
    const tasks = [_][]const u8{ "coding", "history", "math", "story", "translate" };
    for (tasks) |task| {
        const embed_slice = try save_load.getSlice(zml_handler, "qwen_embed.safetensors", task, .f32, true);
        defer embed_slice.free(zml_handler.allocator);
        const n: usize = @intCast(embed_slice.shape.dims()[0]);
        const d: usize = @intCast(embed_slice.shape.dims()[1]);
        std.log.info("Test coarse search task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        for (0..n) |embed_index| {
            const embed = zml.Slice.init(.init(.{ .s = 1, .d = d }, .f32), embed_slice.constItems(u8)[4 * embed_index * d .. 4 * (embed_index + 1) * d]);
            std.log.info("\n", .{});
            std.log.info("#################", .{});
            std.log.info("#### new emb ####", .{});
            std.log.info("#################", .{});
            try sampler.sample(embed.constItems(f32), active_parent);
            try sampler.sampleCoarse(embed.constItems(f32), active_parent);
            if (embed_index > 0) break;
        }
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

fn mathTop1() []const u32 {
    return &[_]u32{ 1654, 525, 2661, 279, 5109, 25, 2303, 334, 58, 17, 11, 220, 20, 11, 220, 16, 18, 11, 220, 17, 20, 11, 220, 20, 15, 78341, 2303, 1654, 646, 990, 1817, 1372, 3070, 266, 1429, 3055, 97219, 323, 582, 646, 990, 279, 19624, 25, 3070, 44662, 24196, 11, 24768, 11, 1683, 115, 11, 320, 11, 873, 334, 382, 7981, 5795, 374, 311, 1855, 458, 7493, 429, 66249, 311, 3070, 300, 3265, 438, 3204, 311, 220, 21, 21, 21, 334, 382, 44364, 14374, 14822, 220, 16, 25, 9735, 311, 16045, 264, 10601, 429, 5221, 3265, 311, 220, 21, 21, 21, 271, 10061, 748, 1430, 311, 1936, 458, 7493, 1667, 46444, 323, 5256, 11, 2474, 1846, 8376, 311, 633, 601, 12128, 311, 3460, 5109, 382, 10061, 748, 1430, 1447, 14085, 198, 7, 20, 15, 1124, 15136, 220, 16, 18, 8, 488, 320, 17, 20, 1124, 15136, 220, 20, 8, 488, 220, 17, 198, 14085, 271, 10061, 748, 12564, 1447, 12, 400, 220, 20, 15, 1124, 15136, 220, 16, 18, 284, 220, 21, 20, 15, 24437, 12, 400, 220, 17, 20, 1124, 15136, 220, 20, 284, 220, 16, 17, 20, 24437, 12, 400, 220, 21, 20, 15, 488, 220, 16, 17, 20, 284, 220, 22, 22, 20, 24437, 12, 400, 220, 22, 22, 20, 488, 220, 17, 284, 220, 22, 22, 22, 84500, 4792, 748, 3070, 22, 22, 22, 97219, 892, 374, 3070, 16, 16, 16, 916, 334, 220, 21, 21, 21, 382, 21453, 2441, 10601, 1447, 14085, 198, 7, 20, 15, 1124, 15136, 220, 16, 18, 8, 488, 320, 17, 20, 1124, 15136, 220, 17, 8, 488, 220, 20, 198, 14085, 271, 12, 400, 220, 20, 15, 1124, 15136, 220, 16, 18, 284, 220, 21, 20, 15, 24437, 12, 400, 220, 17, 20, 1124, 15136, 220, 17, 284, 220, 20, 15, 24437, 12, 400, 220, 21, 20, 15, 488, 220, 20, 15, 284, 220, 22, 15, 15, 24437, 12, 400, 220, 22, 15, 15, 488, 220, 20, 284, 220, 22, 15, 20, 84500, 23322, 2238, 1550, 382, 21453, 1447, 14085, 198, 7, 20, 15, 1124, 15136, 220, 16, 18, 8, 488, 320, 20, 1124, 15136, 220, 17, 20, 8, 488, 220, 17, 198, 14085, 271, 12, 400, 220, 20, 15, 1124, 15136, 220, 16, 18, 284, 220, 21, 20, 15, 24437, 12, 400, 220, 20, 1124, 15136, 220, 17, 20, 284, 220, 16, 17, 20, 24437, 12, 400, 220, 21, 20, 15, 488, 220, 16, 17, 20, 284, 220, 22, 22, 20, 24437, 12, 400, 220, 22, 22, 20, 488, 220, 17, 284, 220, 22, 22, 22, 84500, 19198, 438, 1573, 382, 21453, 1447, 14085, 198, 7, 20, 15, 1124, 15136, 220, 16, 18, 8, 488, 320, 17, 20, 1124, 15136, 220, 17, 8, 481, 220, 20, 198, 14085, 271, 12, 400, 220, 20, 15, 1124, 15136, 220, 16, 18, 284, 220, 21, 20, 15, 24437, 12, 400, 220, 17, 20, 1124, 15136, 220, 17, 284, 220, 20, 15, 24437, 12, 400, 220, 21, 20, 15, 488, 220, 20, 15, 284, 220, 22, 15, 15, 24437, 12, 400, 220, 22, 15, 15, 481, 220, 20, 284, 220, 21, 24, 20, 84500, 4792, 748, 3070, 21, 24, 20, 97219, 892, 374, 3070, 3243, 220, 22, 16, 3123, 334, 504, 220, 21, 21, 21, 382, 4792, 748, 264, 1661, 9144, 382, 21453, 311, 633, 1496, 12128, 382, 21453, 1447, 14085, 198, 7, 20, 15, 1124, 15136, 220, 16, 18, 8, 488, 320, 17, 20, 1124, 15136, 220, 17, 8, 481, 320, 20, 488, 220, 17, 340, 14085, 271, 12, 400, 220, 20, 15, 1124, 15136, 220, 16, 18, 284, 220, 21, 20, 15, 24437, 12, 400, 220, 17, 20, 1124, 15136, 220, 17, 284, 220, 20, 15, 24437, 12, 400, 220, 21, 20, 15, 488, 220, 20, 15, 284, 220, 22, 15, 15, 24437, 12, 400, 220, 20, 488, 220, 17, 284, 220, 22, 24437, 12, 400, 220, 22, 15, 15, 481, 220, 22, 284, 220, 21, 24, 18, 84500, 4792, 748, 3070, 21, 24, 18, 97219, 892, 374, 3070, 18, 18, 3123, 334, 504, 220, 21, 21, 21, 382, 21453, 1447, 14085, 198, 7, 20, 15, 1124, 15136, 220, 16, 18, 8, 488, 320, 17, 20, 1124, 15136, 220, 17, 8, 481, 320, 20, 1124, 15136, 220, 17, 340, 14085, 271, 12, 400, 220, 20, 15, 1124, 15136, 220, 16, 18, 284, 220, 21, 20, 15, 24437, 12, 400, 220, 17, 20, 1124, 15136, 220, 17, 284, 220, 20, 15, 24437, 12, 400, 220, 20, 1124, 15136, 220, 17, 284, 220, 16, 15, 24437, 12, 400, 220, 21, 20, 15, 488, 220, 20, 15, 284, 220, 22, 15, 15, 24437, 12, 400, 220, 22, 15, 15, 481, 220, 16, 15, 284, 220, 21, 24, 15, 84500, 4792, 748, 3070, 21, 24, 15, 97219, 892, 374, 3070, 18, 15, 3123, 334, 504, 220, 21, 21, 21, 382, 21453, 1447, 14085, 198, 7, 20, 15, 1124, 15136, 220, 16, 18, 8, 488, 320, 17, 20, 1124, 15136, 220, 17, 8, 481, 320, 20, 1124, 15136, 220, 17, 488, 220, 17, 340, 14085, 271, 12, 400, 220, 20, 15, 1124, 15136, 220, 16, 18, 284, 220, 21, 20, 15, 24437, 12, 400, 220, 17, 20, 1124, 15136, 220, 17, 284, 220, 20, 15, 24437, 12, 400, 220, 20, 1124, 15136, 220, 17, 284, 220, 16, 15, 24437, 12, 400, 220, 16, 15, 488, 220, 17, 284, 220, 16, 17, 24437, 12, 400, 220, 21, 20, 15, 488, 220, 20, 15, 284, 220, 22, 15, 15, 24437, 12, 400, 220, 22, 15, 15, 481, 220, 16, 17, 284, 220, 21, 23, 23, 84500, 4792, 748, 3070, 21, 23, 23, 97219, 892, 374, 3070, 17, 17, 3123, 334, 504, 220, 21, 21, 21, 382, 21453, 1447, 14085, 198, 7, 20, 15, 1124, 15136, 220, 16, 18, 8, 488, 320, 17, 20, 1124, 15136, 220, 17, 8, 481, 320, 20, 1124, 15136, 220, 17, 488, 220, 17, 488, 220, 20, 340, 14085, 271, 12, 400, 220, 20, 15, 1124, 15136, 220, 16, 18, 284, 220, 21, 20, 15, 24437, 12, 400, 220, 17, 20, 1124, 15136, 220, 17, 284, 220, 20, 15, 24437, 12, 400, 220, 20, 1124, 15136, 220, 17, 284, 220 };
}

fn codingTop1() []const u32 {
    return &[_]u32{ 8420, 594, 264, 13027, 729, 429, 57203, 279, 3070, 45, 7563, 10250, 1372, 334, 29720, 1667, 279, 3070, 50, 19155, 315, 9740, 266, 535, 12032, 288, 334, 323, 264, 8741, 5486, 311, 16045, 279, 8416, 6822, 369, 279, 74336, 1447, 73594, 12669, 198, 750, 55129, 38217, 1445, 982, 262, 421, 308, 2651, 220, 15, 510, 286, 4828, 15402, 445, 77, 1969, 387, 264, 6785, 7546, 5130, 262, 671, 72715, 8416, 8416, 6822, 369, 279, 55129, 10250, 198, 262, 671, 12091, 279, 56204, 25, 55129, 10250, 374, 2686, 1091, 308, 353, 320, 839, 308, 488, 1487, 1487, 308, 340, 262, 671, 369, 308, 2604, 220, 21, 198, 262, 421, 308, 621, 220, 16, 510, 286, 470, 220, 17, 198, 262, 4409, 308, 621, 220, 17, 510, 286, 470, 220, 18, 198, 262, 4409, 308, 621, 220, 18, 510, 286, 470, 220, 20, 198, 262, 4409, 308, 621, 220, 19, 510, 286, 470, 220, 22, 198, 262, 4409, 308, 621, 220, 20, 510, 286, 470, 220, 16, 16, 271, 262, 671, 72715, 8416, 6822, 198, 262, 8416, 19447, 284, 1932, 7, 16, 15, 11, 526, 1445, 353, 320, 10374, 1665, 1445, 8, 488, 6888, 1665, 37270, 1665, 1445, 593, 19235, 262, 671, 8495, 586, 315, 9740, 266, 535, 12032, 288, 198, 262, 74336, 284, 508, 2514, 60, 353, 320, 13574, 19447, 488, 220, 16, 340, 262, 74336, 58, 15, 60, 284, 74336, 58, 16, 60, 284, 3557, 198, 262, 369, 600, 304, 2088, 7, 17, 11, 526, 7, 13574, 19447, 3070, 220, 15, 13, 20, 8, 488, 220, 16, 982, 286, 421, 74336, 989, 10343, 310, 369, 502, 304, 2088, 1956, 353, 600, 11, 8416, 19447, 488, 220, 16, 11, 600, 982, 394, 74336, 3809, 60, 284, 3557, 271, 262, 671, 20513, 49433, 198, 262, 49433, 284, 508, 72, 369, 600, 11, 374, 38217, 304, 13252, 1141, 19155, 8, 421, 374, 38217, 2533, 262, 671, 1416, 582, 3207, 944, 1477, 3322, 49433, 11, 5263, 279, 8416, 6822, 323, 1430, 1549, 198, 262, 1393, 2422, 24974, 1733, 8, 366, 308, 510, 286, 8416, 19447, 11404, 220, 17, 198, 286, 74336, 284, 508, 2514, 60, 353, 320, 13574, 19447, 488, 220, 16, 340, 286, 74336, 58, 15, 60, 284, 74336, 58, 16, 60, 284, 3557, 198, 286, 369, 600, 304, 2088, 7, 17, 11, 526, 7, 13574, 19447, 3070, 220, 15, 13, 20, 8, 488, 220, 16, 982, 310, 421, 74336, 989, 10343, 394, 369, 502, 304, 2088, 1956, 353, 600, 11, 8416, 19447, 488, 220, 16, 11, 600, 982, 503, 74336, 3809, 60, 284, 3557, 198, 286, 49433, 284, 508, 72, 369, 600, 11, 374, 38217, 304, 13252, 1141, 19155, 8, 421, 374, 38217, 2533, 262, 470, 49433, 7669, 481, 220, 16, 921, 13874, 19324, 14374, 2585, 311, 5443, 510, 8078, 2704, 311, 1159, 1565, 10374, 63, 1447, 73594, 12669, 198, 474, 6888, 271, 2, 13383, 10431, 510, 1350, 1445, 339, 38217, 7, 16, 15, 593, 220, 671, 9258, 25, 220, 17, 24, 198, 1350, 1445, 339, 38217, 7, 16, 15, 15, 593, 671, 9258, 25, 220, 20, 19, 16, 198, 13874, 19324, 14374, 18068, 510, 12, 1096, 729, 5711, 279, 3070, 50, 19155, 315, 9740, 266, 535, 12032, 288, 334, 311, 1477, 678, 49433, 705, 311, 458, 12943, 8416, 6822, 624, 12, 1416, 279, 2856, 16045, 374, 2238, 3347, 11, 432, 39296, 279, 8416, 6822, 323, 16297, 1549, 624, 12, 1096, 374, 11050, 369, 2613, 311, 69251, 2750, 315, 1565, 77, 62338, 10061, 752, 1414, 421, 498, 4172, 1075, 264, 2319, 429, 5711, 264, 2155, 12111, 320, 4803, 13295, 1817, 1372, 369, 8860, 2719, 5961, 568 };
}

fn translateTop1() []const u32 {
    return &[_]u32{ 39814, 0, 5692, 594, 279, 14468, 315, 3070, 1, 40, 2948, 80676, 1, 334, 304, 3070, 16, 15, 15459, 97219, 8110, 553, 847, 4345, 9459, 389, 892, 14468, 374, 279, 3070, 1726, 5566, 13438, 334, 382, 44364, 14374, 220, 16, 13, 3070, 22574, 334, 2303, 334, 40, 2948, 80676, 334, 2303, 9, 18395, 17133, 21518, 44364, 14374, 220, 17, 13, 3070, 61797, 334, 2303, 334, 64725, 86100, 5141, 3594, 573, 86809, 334, 2303, 9, 86015, 745, 25, 330, 40, 2948, 279, 80676, 1, 1365, 330, 6091, 78, 1, 374, 264, 3746, 7493, 315, 2948, 11, 323, 330, 5612, 573, 86809, 1, 374, 75434, 42015, 44364, 14374, 220, 18, 13, 3070, 43197, 334, 2303, 334, 29754, 85707, 3541, 25298, 483, 2382, 334, 2303, 9, 9112, 25, 330, 29754, 85707, 1, 374, 264, 2699, 41787, 11, 714, 432, 594, 16626, 1483, 304, 21355, 8585, 13, 330, 47, 391, 483, 2382, 1, 374, 264, 6233, 3409, 369, 80676, 42015, 44364, 14374, 220, 19, 13, 3070, 69111, 334, 2303, 334, 42799, 86100, 512, 3041, 69, 5054, 334, 2303, 9, 25756, 75434, 323, 25777, 13, 330, 32887, 69, 5054, 1, 374, 264, 8413, 11, 10581, 52760, 3409, 429, 11367, 31253, 42015, 44364, 14374, 220, 20, 13, 3070, 32079, 334, 2303, 334, 40369, 91342, 5016, 4059, 465, 2718, 68, 334, 2303, 9, 16374, 323, 2118, 13, 330, 30124, 4059, 465, 2718, 68, 1, 374, 264, 16690, 11, 10581, 52760, 3409, 42015, 44364, 14374, 220, 21, 13, 3070, 7084, 768, 35454, 320, 67199, 32295, 2303, 334, 54118, 86100, 438, 31632, 749, 1149, 300, 334, 2303, 9, 95275, 323, 36705, 349, 13, 330, 33, 269, 749, 1149, 300, 1, 702, 264, 21700, 11, 17795, 5112, 42015, 44364, 14374, 220, 22, 13, 3070, 51466, 334, 2303, 334, 128976, 103250, 29412, 102271, 128600, 334, 2303, 6599, 76504, 30378, 10450, 521, 55661, 297, 432, 988, 632, 49363, 84, 8, 1365, 330, 40, 2948, 80676, 1189, 576, 3409, 330, 103250, 55661, 1, 320, 103250, 8, 374, 25777, 323, 75434, 42015, 44364, 14374, 220, 23, 13, 3070, 42, 45195, 334, 2303, 334, 60315, 16560, 73518, 70582, 18411, 132488, 33883, 334, 2303, 6599, 45, 2145, 359, 72691, 554, 360, 28047, 524, 4223, 68, 8, 1365, 330, 40, 2948, 80676, 1189, 576, 3409, 330, 60315, 70582, 1, 320, 77, 25084, 8, 374, 8413, 323, 10581, 52760, 42015, 44364, 14374, 220, 24, 13, 3070, 47707, 334, 2303, 334, 85391, 125389, 36806, 14062, 37622, 137858, 334, 2303, 6599, 62893, 14528, 392, 398, 84, 16584, 4953, 1225, 8, 1365, 330, 40, 2948, 80676, 1189, 330, 60332, 37622, 134863, 1, 320, 47722, 4953, 6642, 8, 374, 264, 10226, 11, 47316, 6704, 1352, 429, 11367, 31253, 42015, 44364, 14374, 220, 16, 15, 13, 3070, 6953, 68291, 334, 2303, 334, 69682, 126381, 124114, 123940, 32790, 47632, 334, 2303, 6599, 52, 5803, 19644, 452, 2220, 277, 988, 266, 8, 1365, 330, 40, 2948, 80676, 1189, 576, 3409, 330, 20931, 123940, 32790, 47632, 1, 320, 23559, 988, 266, 8, 374, 25777, 323, 75434, 42015, 44364, 14374, 11162, 234, 116, 3070, 47, 2122, 10251, 477, 38041, 30, 334, 2303, 334, 69111, 25, 330, 42799, 86100, 512, 3041, 69, 5054, 1, 334, 2303, 10234, 30, 2303, 12, 576, 4244, 525, 8413, 323, 10581, 52760, 13, 2303, 12, 330, 32887, 69, 5054, 1, 702, 264, 21700, 11, 4558, 17795, 5112, 13, 2303, 12, 576, 11652, 5944, 374, 25777, 323, 75434, 13, 2303, 12, 1084, 11074, 1075, 264, 326, 617, 6115, 476, 264, 23467, 32794, 504, 264, 32794, 382, 44364, 10061, 752, 1414, 421, 498, 4172, 1075, 36693, 304, 803, 15459, 476, 264, 2155, 1707, 0, 11162, 99, 233 };
}

fn storyTop1() []const u32 {
    return &[_]u32{ 334, 785, 7996, 95272, 315, 21316, 85656, 76924, 56177, 785, 9396, 572, 264, 17846, 315, 14961, 1212, 279, 17788, 4145, 11, 323, 279, 9956, 57266, 23594, 1172, 279, 33200, 1410, 6723, 13, 21316, 85656, 76924, 14638, 518, 279, 33765, 315, 279, 353, 14417, 328, 46335, 12314, 806, 9104, 291, 3579, 74548, 553, 279, 35966, 315, 264, 3175, 73165, 13, 1260, 1030, 75744, 279, 8094, 51740, 369, 26127, 1635, 11, 323, 1431, 11, 448, 279, 34074, 1101, 264, 34855, 3123, 11, 566, 6876, 419, 1035, 387, 806, 1537, 44540, 382, 785, 353, 14417, 328, 46335, 9, 572, 264, 19866, 8284, 11, 62871, 553, 279, 9396, 323, 279, 28813, 13, 1084, 1030, 902, 85005, 11, 902, 13627, 11, 323, 902, 2415, 13, 1084, 75744, 389, 1181, 1828, 11, 14764, 553, 279, 6815, 315, 279, 9788, 13, 85656, 1030, 1730, 432, 1635, 4134, 11, 84253, 304, 279, 30249, 1007, 279, 13648, 315, 17689, 11, 1181, 85005, 259, 21924, 323, 1181, 40198, 22290, 1151, 553, 882, 13, 1260, 1030, 4429, 432, 11, 537, 438, 264, 21882, 11, 714, 438, 264, 11222, 13, 362, 11222, 429, 1477, 279, 282, 2312, 353, 3872, 273, 315, 279, 94447, 12314, 1380, 279, 5558, 58849, 315, 279, 1879, 1033, 1053, 311, 10246, 382, 785, 13627, 1030, 1293, 2474, 58481, 11, 825, 553, 825, 11, 326, 3073, 553, 279, 8284, 748, 14888, 10963, 476, 5558, 311, 279, 9396, 13, 85656, 1030, 3635, 279, 1537, 883, 36506, 11, 264, 57129, 7071, 448, 264, 4746, 2480, 315, 7343, 323, 264, 13527, 2480, 315, 67925, 13, 1260, 1030, 3055, 1012, 264, 883, 315, 1657, 4780, 11, 714, 279, 9396, 1030, 4429, 1105, 678, 2293, 25235, 7403, 11, 806, 4438, 11, 806, 13627, 13, 4695, 11, 566, 75744, 7484, 11, 42831, 279, 69957, 315, 264, 2272, 3055, 12163, 382, 2121, 279, 353, 14417, 328, 46335, 9, 2770, 4490, 1526, 279, 16876, 11, 85656, 6476, 279, 8284, 23065, 23969, 1435, 13, 576, 9956, 28973, 11, 323, 279, 73165, 28347, 12336, 13, 1260, 6966, 705, 11, 323, 369, 279, 1156, 882, 304, 1635, 11, 566, 5485, 2494, 304, 279, 6010, 27996, 6319, 6083, 16062, 504, 279, 8600, 11, 1181, 264, 12455, 504, 279, 3267, 382, 2132, 572, 279, 353, 3872, 273, 315, 279, 94447, 9, 382, 36, 71829, 22993, 6924, 279, 13284, 11, 806, 6078, 24020, 8818, 279, 13458, 429, 88095, 304, 806, 4746, 13, 1260, 1030, 7391, 806, 2272, 42831, 18707, 11, 323, 1431, 11, 518, 279, 835, 11, 566, 1035, 5499, 5545, 1105, 13, 576, 9396, 1030, 4429, 4297, 504, 1435, 11, 714, 432, 1035, 537, 1896, 806, 1590, 11618, 382, 2354, 264, 5538, 11486, 11, 566, 9226, 806, 7743, 304, 264, 34855, 11, 438, 421, 12094, 311, 279, 9956, 5086, 382, 1, 10061, 279, 9396, 1896, 752, 11, 714, 1077, 752, 29403, 311, 279, 835, 2217, 3036, 448, 429, 11, 279, 353, 14417, 328, 46335, 9, 58481, 1119, 279, 8600, 11, 264, 19866, 4221, 53840, 11, 15331, 279, 1537, 883, 315, 279, 9396, 389, 806, 1590, 44540, 13 };
}

fn historyTop1() []const u32 {
    return &[_]u32{ 785, 3070, 17977, 2886, 409, 12095, 334, 320, 59604, 6804, 2886, 701, 892, 35413, 504, 3070, 27523, 220, 16, 23, 311, 3217, 220, 17, 23, 11, 220, 16, 23, 22, 16, 97219, 572, 264, 29091, 3033, 429, 26753, 21286, 12095, 1283, 279, 43843, 12, 3533, 15579, 5004, 323, 279, 18179, 315, 279, 10440, 8585, 20448, 13, 11445, 4948, 24154, 1033, 27155, 323, 3041, 86148, 11, 2176, 304, 9625, 323, 36445, 13, 5692, 525, 279, 1376, 4948, 24154, 1447, 44364, 14374, 220, 16, 13, 3070, 62078, 311, 279, 8585, 5429, 1019, 12, 576, 12095, 2886, 572, 264, 17855, 11, 2115, 28380, 3033, 429, 21992, 304, 279, 39596, 315, 279, 43843, 12, 3533, 15579, 5004, 323, 279, 18179, 315, 279, 10440, 20448, 624, 12, 1084, 572, 264, 2118, 8645, 311, 279, 3070, 36975, 5429, 97219, 892, 572, 2058, 304, 1181, 85517, 323, 1030, 537, 3602, 1012, 36302, 9555, 624, 12, 576, 6804, 2886, 16105, 311, 4211, 3070, 22386, 380, 323, 25542, 30243, 97219, 2670, 35919, 6731, 11, 11864, 21676, 5859, 11, 323, 279, 24737, 315, 8817, 323, 1584, 624, 12, 11445, 13885, 15251, 264, 3070, 13281, 938, 1438, 504, 279, 8606, 8585, 4948, 1973, 97219, 892, 1030, 29701, 553, 15332, 323, 62754, 380, 8437, 382, 44364, 14374, 220, 17, 13, 3070, 10048, 4011, 553, 279, 8585, 10212, 1019, 12, 576, 3070, 43197, 3033, 97219, 6069, 553, 2410, 337, 80806, 663, 4813, 11, 19334, 279, 6804, 2886, 438, 264, 5899, 311, 5313, 30326, 323, 279, 19753, 315, 279, 501, 34444, 624, 12, 576, 3070, 10048, 4011, 315, 279, 12095, 6804, 2886, 334, 320, 11109, 220, 17, 16, 4142, 17, 23, 11, 220, 16, 23, 22, 16, 8, 572, 825, 315, 279, 6543, 13438, 90058, 1429, 27760, 90058, 416, 304, 6481, 7513, 3840, 624, 12, 62194, 315, 56198, 2347, 1033, 15695, 11, 51842, 11, 476, 505, 2181, 13, 1096, 3070, 26331, 28474, 334, 1030, 264, 28769, 5421, 389, 8585, 4948, 7674, 11, 90015, 264, 3070, 32880, 7806, 657, 8679, 315, 17855, 2142, 334, 323, 2115, 28380, 19029, 382, 44364, 14374, 220, 18, 13, 3070, 641, 40016, 389, 68950, 323, 56110, 53285, 1019, 12, 576, 3070, 17977, 2886, 334, 6116, 264, 3070, 18785, 315, 29091, 50518, 334, 323, 14606, 2937, 68950, 68022, 11, 2670, 3070, 42, 48258, 27088, 334, 323, 3070, 37, 4487, 13851, 3285, 2010, 334, 624, 12, 27088, 6139, 41717, 911, 279, 6804, 2886, 304, 806, 975, 353, 785, 16398, 5004, 304, 9625, 9, 320, 16, 23, 22, 16, 701, 80578, 432, 438, 264, 3070, 2528, 315, 63117, 8821, 19525, 334, 323, 264, 3070, 649, 36019, 3766, 315, 40189, 16170, 334, 624, 12, 576, 3070, 17977, 2886, 334, 374, 3545, 21870, 438, 264, 3070, 10645, 3823, 311, 279, 8522, 22126, 315, 220, 16, 24, 16, 22, 334, 323, 279, 10000, 315, 40189, 323, 49215, 19029, 304, 279, 220, 17, 15, 339, 9294, 382, 44364, 14374, 220, 19, 13, 3070, 71503, 389, 8585, 30497, 20397, 1019, 12, 576, 3070, 17977, 2886, 334, 572, 13771, 66, 55778, 323, 54989, 334, 304, 279, 13922, 39596, 11, 323, 1181, 4938, 572, 3070, 32812, 398, 63700, 334, 504, 8585, 4948, 40502, 369, 10793, 624, 12, 4354, 11, 432, 572, 3070, 265, 21275, 553, 279, 2115, 334, 304, 279, 220, 17, 15, 339, 9294, 11, 5310, 2337, 279, 3070, 43197, 20861, 5429, 334, 323, 279, 3070, 43197, 35074, 5429, 97219, 323, 2937, 2337, 279, 3070, 43197, 22843, 5429, 334, 624, 12, 576, 3070, 17977, 2886, 334, 6116, 264, 3070, 18785, 315, 13643, 2348, 57921, 2142, 334, 323, 264, 3070, 2427, 315, 19760, 369, 71127, 19029, 334, 304, 9625, 323, 7797, 382, 44364, 14374, 220, 20, 13, 3070, 33646, 74940, 1019, 12, 576, 3070, 17977, 2886, 334, 14606, 3070, 95722, 658, 19029, 334, 3941, 4505, 323, 279, 1879, 11, 7945, 304, 3070, 44506, 97219, 3070, 50170, 97219, 323, 3070, 70503, 334, 624, 12, 1084, 20459, 279, 3070, 8831, 3147, 315, 264, 3238, 14800, 3033, 97219, 323, 27061, 279, 4401, 315, 3070, 22386, 380, 323, 49215, 9677, 334, 304, 279, 3309, 220, 16, 24, 339, 323, 4124, 220, 17, 15, 339, 23631, 624, 12, 576, 3070, 17977, 2886, 334, 1083, 1030, 264, 3070, 69429, 5421, 389, 279, 6489, 9327, 97219, 90015, 279, 4522, 429, 3070, 95722, 658, 2297, 334, 572, 3204, 1526, 5411, 705, 5963, 819, 382, 44364, 14374, 220, 21, 13, 3070, 77415, 315, 46632, 323, 3321, 11185, 1019, 12, 576, 3070, 26331, 28474, 315, 279, 56198, 2347, 334, 2115, 264, 3070, 32880, 22290, 389, 8585, 8232, 97219, 28720, 311, 264, 3070, 93774, 315, 4948, 71398, 334, 323, 3070, 69, 682, 315, 17855, 2142, 334, 624, 12, 576, 3070, 86763, 594, 27760, 2033, 334, 738, 279, 6804, 2886, 738, 264, 46791, 369, 3070, 3094, 20030, 2524, 334, 323, 3070, 2454, 9170, 334, 304, 279, 829, 315, 5313, 30326, 323, 1973, 382, 44364, 14374, 73877, 510, 785, 3070, 17977, 2886, 409, 12095, 334, 572, 264, 3070, 6658, 714, 5089, 4948, 9342, 334, 429, 28891, 279, 6350, 1973, 304, 9625, 323, 14606, 29091, 19029, 15245, 13, 11445, 3070, 74685, 24154, 334, 2924, 510, 12, 362, 3070, 57365, 311, 279, 8585, 5429, 334, 323, 279, 21269, 315, 264, 501, 4948, 1973, 624, 12, 362, 3070, 18785, 315, 40189, 323, 49215, 51705, 334, 429, 27061, 220, 17, 15, 339, 33357, 92474, 624, 12, 362, 3070, 39884, 315, 71398, 323, 9170, 334, 429, 26271, 8585, 4948, 7674, 369, 10793, 624, 12, 362, 3070, 2427, 315, 19760, 334, 369, 71127, 19029, 323, 264, 26528, 315, 279, 2355, 315, 5411, 705, 5963, 819, 382, 641, 12126, 11, 279, 3070, 17977, 2886, 409, 12095, 334, 572, 264, 59750, 4445, 304, 6481, 7513, 3840, 11, 448, 28769, 4948, 11, 3590, 11, 323, 41833, 15917, 13 };
}
