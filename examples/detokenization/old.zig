
pub fn runTestsQuantized(zml_handler: *Zml_handler) !void {
    var model_handler = try model_.Model_handler.init(zml_handler);
    defer model_handler.deinit(zml_handler.allocator);
    defer model_handler.unloadBuffers();

    std.log.info("Get lm_head", .{});
    var lm_head = try algebra.getLmHead(zml_handler);
    defer LmHeadMatrix.deinit(&lm_head, zml_handler.allocator);

    std.log.info("Init Quantizer", .{});
    var quantizer = try quantized.Quantized.init(zml_handler, &lm_head);
    defer quantizer.deinit();

    //try quantizer.quantizeLmHead();
    //try quantizer.sliceLmHead();
    //try quantizer.testQuantizeSlice();
    //try quantizer.testWalshHadamardRoundtrip();

    try quantizer.quantizeQjlLmHead();
    try quantizer.quantizeQjlLmHead2Bits();
    try quantizer.precomputePartialDotLmHeadBlocks();
    //try quantizer.testQjlReconstructionError();

    try testEmbedQuantizedSearch(zml_handler, &quantizer);
}

pub fn runTestsQuantizedPQ(zml_handler: *Zml_handler) !void {
    var model_handler = try model_.Model_handler.init(zml_handler);
    defer model_handler.deinit(zml_handler.allocator);
    defer model_handler.unloadBuffers();

    std.log.info("Get lm_head", .{});
    var lm_head = try algebra.getLmHead(zml_handler);
    defer LmHeadMatrix.deinit(&lm_head, zml_handler.allocator);

    std.log.info("Init Product Quantizer", .{});
    var quantizer = try ProductQuantizerFastScan.init(zml_handler, &lm_head, .anisotropic);
    defer quantizer.deinit();

    try quantizer.buildCodebook();

    try testEmbedQuantizedPQSearch(zml_handler, &quantizer);
}

pub fn runTestsGraph(zml_handler: *Zml_handler) !void {
    std.log.info("Get lm_head", .{});
    var lm_head = try algebra.getLmHead(zml_handler);
    defer LmHeadMatrix.deinit(&lm_head, zml_handler.allocator);

    std.log.info("Init sampler", .{});
    var sampler: Sampler = try .init(zml_handler, &lm_head);
    defer Sampler.deinit(&sampler);

    var g_knn_angu, var g_nsw_angu = try buildAngularGraphs(zml_handler, &lm_head, &sampler);
    defer g_knn_angu.deinit();
    defer g_nsw_angu.deinit();

    var g_knn_mips, var g_nsw_mips = try buildMipsGraphs(zml_handler, &lm_head, &sampler);
    defer g_knn_mips.deinit();
    defer g_nsw_mips.deinit();

    try testEmbedGraphSearch(zml_handler, &g_knn_angu, &sampler, "KNN-angular");
    try testEmbedGraphSearch(zml_handler, &g_knn_mips, &sampler, "KNN-mips");

    try testEmbedGraphSearch(zml_handler, &g_nsw_angu, &sampler, "NSW-angular");
    try testEmbedGraphSearch(zml_handler, &g_nsw_mips, &sampler, "NSW-mips");

    g_nsw_angu.params.search_budget = 512;
    g_knn_mips.params.search_budget = 2048;
    try testEmbedDualGraphSearch(zml_handler, &g_nsw_angu, &g_knn_mips, &sampler, "NSW-angular -> KNN-mips");

    g_nsw_mips.params.search_budget = 512;
    g_knn_angu.params.search_budget = 2048;
    try testEmbedDualGraphSearch(zml_handler, &g_nsw_mips, &g_knn_angu, &sampler, "NSW-mips -> KNN-angular");

    g_nsw_mips.params.search_budget = 512;
    g_nsw_angu.params.search_budget = 2048;
    try testEmbedDualGraphSearch(zml_handler, &g_nsw_mips, &g_nsw_angu, &sampler, "NSW-mips -> NSW-angular");

    g_knn_mips.params.search_budget = 512;
    g_nsw_angu.params.search_budget = 2048;
    try testEmbedDualGraphSearch(zml_handler, &g_knn_mips, &g_nsw_angu, &sampler, "KNN-mips -> NSW-angular");

    g_knn_angu.params.search_budget = 512;
    g_nsw_mips.params.search_budget = 2048;
    try testEmbedDualGraphSearch(zml_handler, &g_knn_angu, &g_nsw_mips, &sampler, "KNN-angular -> NSW-mips");

    g_knn_mips.params.search_budget = 512;
    g_knn_angu.params.search_budget = 2048;
    try testEmbedDualGraphSearch(zml_handler, &g_knn_mips, &g_knn_angu, &sampler, "KNN-mips -> KNN-angular");
}


pub fn buildAngularGraphs(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, sampler: *Sampler) !struct { Graph, Graph } {
    zml_handler.tic(&zml_handler.timers.similarity_matrix);
    //var similarity_matrix = try algebra.computeSimilarityMatrix(zml_handler, &model_handler, false);
    var similarity_matrix = try algebra.loadSimilarityMatrix(zml_handler, true);
    defer similarity_matrix.deinit(zml_handler.allocator);
    zml_handler.toc(&zml_handler.timers.similarity_matrix);

    const ang_params: graph.GraphParams = .{ .graph_type = .Angular };

    std.log.info("********** Init KNN-angular graph", .{});
    zml_handler.tic(&zml_handler.timers.knn_graph);
    var g_knn: Graph = try .init(zml_handler, lm_head, &similarity_matrix, ang_params);
    g_knn.consolidateNearestPrune();
    try g_knn.testNswExtention(sampler);
    zml_handler.toc(&zml_handler.timers.knn_graph);

    std.log.info("********** Init NSW-angular graph", .{});
    zml_handler.tic(&zml_handler.timers.nsw_graph);
    var g_nsw: Graph = try .init(zml_handler, lm_head, &similarity_matrix, ang_params);
    g_nsw.consolidateNearestPrune();
    try g_nsw.extendToNsw();
    try g_nsw.testNswExtention(sampler);
    try g_nsw.fixNswExtention();
    try g_nsw.testNswExtention(sampler);
    g_nsw.consolidateNearestPrune();
    try g_nsw.testNswExtention(sampler);
    zml_handler.toc(&zml_handler.timers.nsw_graph);

    return .{ g_knn, g_nsw };
}

pub fn buildMipsGraphs(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, sampler: *Sampler) !struct { Graph, Graph } {
    zml_handler.tic(&zml_handler.timers.similarity_matrix);
    //var similarity_matrix = try algebra.computeSimilarityMatrix(zml_handler, model_handler, false);
    var similarity_matrix = try algebra.loadSimilarityMatrix(zml_handler, false);
    defer similarity_matrix.deinit(zml_handler.allocator);
    zml_handler.toc(&zml_handler.timers.similarity_matrix);

    const mips_params: graph.GraphParams = .{ .graph_type = .Mips };

    std.log.info("********** Init KNN-mips graph", .{});
    zml_handler.tic(&zml_handler.timers.knn_graph);
    var g_knn: Graph = try .init(zml_handler, lm_head, &similarity_matrix, mips_params);
    g_knn.consolidateNearestPrune();
    try g_knn.testNswExtention(sampler);
    zml_handler.toc(&zml_handler.timers.knn_graph);

    std.log.info("********** Init NSW-mips graph", .{});
    zml_handler.tic(&zml_handler.timers.nsw_graph);
    var g_nsw: Graph = try .init(zml_handler, lm_head, &similarity_matrix, mips_params);
    g_nsw.consolidateNearestPrune();
    try g_nsw.extendToNsw();
    try g_nsw.testNswExtention(sampler);
    try g_nsw.fixNswExtention();
    try g_nsw.testNswExtention(sampler);
    g_nsw.consolidateNearestPrune();
    try g_nsw.testNswExtention(sampler);
    zml_handler.toc(&zml_handler.timers.nsw_graph);

    return .{ g_knn, g_nsw };
}

pub fn runTestsQuantizedGraph(zml_handler: *Zml_handler) !void {
    std.log.info("********** Get lm_head", .{});
    var lm_head = try algebra.getLmHead(zml_handler);
    defer LmHeadMatrix.deinit(&lm_head, zml_handler.allocator);

    std.log.info("********** Init sampler", .{});
    var sampler: Sampler = try .init(zml_handler, &lm_head);
    defer Sampler.deinit(&sampler);

    std.log.info("********** Init similarity matrix", .{});
    zml_handler.tic(&zml_handler.timers.similarity_matrix);
    //var similarity_matrix = try algebra.computeSimilarityMatrix(zml_handler, &model_handler, false);
    var similarity_matrix = try algebra.loadSimilarityMatrix(zml_handler, true);
    defer similarity_matrix.deinit(zml_handler.allocator);
    zml_handler.toc(&zml_handler.timers.similarity_matrix);

    std.log.info("********** Init NSW-angular graph", .{});
    var g_nsw: Graph = try .init(zml_handler, &lm_head, &similarity_matrix, .{ .graph_type = .Angular });
    g_nsw.consolidateNearestPrune();
    zml_handler.tic(&zml_handler.timers.nsw_graph);
    try g_nsw.extendToNsw();
    zml_handler.toc(&zml_handler.timers.nsw_graph);
    try g_nsw.fixNswExtention();
    g_nsw.consolidateNearestPrune();
    try g_nsw.testNswExtention(&sampler);

    std.log.info("********** Init Quantizer", .{});
    var quantizer = try Quantized.init(zml_handler, &lm_head);
    defer quantizer.deinit();
    //try quantizer.quantizeQjlLmHead();

    //try g_nsw.extendNswRandom(&quantizer);

    try testEmbedGraphSearch(zml_handler, &g_nsw, &sampler, "NSW angular");
    //try testEmbedGraphQuantizedSearch(zml_handler, &g_nsw, &quantizer);
    //try testEmbedGraphSearchFullTopK(zml_handler, &g_nsw, &quantizer);

    //try g_nsw.extendNswRandom(&quantizer);
    //try testEmbedGraphQuantizedSearch(zml_handler, &g_nsw, &quantizer);
}

pub fn testEmbedGraphSearch(zml_handler: *Zml_handler, g: *Graph, _: *Sampler, s: []const u8) !void {
    std.log.info("\n********** Test embed graph search with graph = {s}", .{s});

    var total_count: usize = 0;
    var found_top1_count: usize = 0;

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
        zml_handler.tic(&zml_handler.timers.graph_search_tot);
        for (0..n) |embed_index| {
            const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];

            zml_handler.timers.nb_detokenize += 1;
            g.greedySearchPrefetch(embed);

            const nb_visited = g.nb_scored;
            total_visited += nb_visited;
            min_visited = @min(min_visited, nb_visited);
            max_visited = @max(max_visited, nb_visited);

            total_count += 1;
            const top1_token = top1[embed_index];
            var found_top1 = false;
            for (0..g.L) |i| {
                if (g.visited[i].node == top1_token) found_top1 = true;
            }
            if (found_top1) found_top1_count += 1;
        }
        zml_handler.toc(&zml_handler.timers.graph_search_tot);
    }

    const percent_found = 100.0 * @as(f64, @floatFromInt(found_top1_count)) / @as(f64, @floatFromInt(total_count));
    const average_visit = @as(f64, @floatFromInt(total_visited)) / @as(f64, @floatFromInt(total_count));
    std.log.info("Embed graph search: total={d} found_top1={d} ({d:.4}%)", .{ total_count, found_top1_count, percent_found });
    std.log.info("Embed graph search nb_visited: min={d} max={d} avg={d:.2}", .{ min_visited, max_visited, average_visit });
}

fn logTop16MissDetails(tokenizer: zml.tokenizer.Tokenizer, g: *Graph, quantizer: *Quantized, embed: []const f32, dense_sample: quantized.DenseSample, found_top16_count: usize, found_mass: f32) !void {
    std.log.info("Low graph overlap with real top-16: found={d}/16 retained_mass={d:.6}", .{ found_top16_count, found_mass });
    std.log.info("Real dense top-16", .{});
    std.log.info("{s:>4} {s:>10} {s:>14} {s:>14} {s:>14}  {s}", .{ "rank", "token", "proba", "logit", "row_norm", "text" });
    std.log.info("{s:>4} {s:>10} {s:>14} {s:>14} {s:>14}  {s}", .{ "----", "----------", "--------------", "--------------", "--------------", "----" });
    for (dense_sample.rows, dense_sample.probas, dense_sample.logits, 0..) |token, proba, logit, rank| {
        const token_str = try tokens.tokenString(tokenizer, token, g.allocator);
        defer g.allocator.free(token_str);
        std.log.info("{d:>4} {d:>10} {d:>14.8} {d:>14.6} {d:>14.6}  {s}", .{ rank + 1, token, proba, logit, quantizer.lm_head.row_norms[token], token_str });
    }

    std.log.info("Graph top-16", .{});
    std.log.info("{s:>4} {s:>10} {s:>14} {s:>14} {s:>14}  {s}", .{ "rank", "token", "proba", "logit", "row_norm", "text" });
    std.log.info("{s:>4} {s:>10} {s:>14} {s:>14} {s:>14}  {s}", .{ "----", "----------", "--------------", "--------------", "--------------", "----" });
    const graph_top = @min(g.L, quantized.top_k_sliced);
    for (0..graph_top) |rank| {
        const token = g.visited[rank].node;
        const token_i: usize = @intCast(token);
        const row = quantizer.lm_head.data[token_i * quantizer.d .. (token_i + 1) * quantizer.d];
        const logit = quantizer.realLogit(embed, row);
        const proba = dense_sample.probaFromLogit(logit);
        const token_str = try tokens.tokenString(tokenizer, token, g.allocator);
        defer g.allocator.free(token_str);
        std.log.info("{d:>4} {d:>10} {d:>14.8} {d:>14.6} {d:>14.6}  {s}", .{ rank + 1, token, proba, logit, quantizer.lm_head.row_norms[token_i], token_str });
    }
}

pub fn testEmbedGraphSearchFullTopK(zml_handler: *Zml_handler, g: *Graph, quantizer: *Quantized) !void {
    const top_p: f32 = 0.9;
    const allocator = zml_handler.allocator;
    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
    var tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    var total_count: usize = 0;
    var total_found_mass: f64 = 0.0;
    var worst_found_mass: f32 = std.math.inf(f32);
    var total_top_p_found: usize = 0;
    var worst_top_p_found: usize = std.math.maxInt(usize);
    var total_top_p_size: usize = 0;
    var max_top_p_size: usize = 0;
    var found_top1_count: usize = 0;
    var found_top1_top2_count: usize = 0;
    var found_top1_top2_top3_count: usize = 0;
    var total_best_found_at: u64 = 0;
    var min_best_found_at: u32 = std.math.maxInt(u32);
    var max_best_found_at: u32 = 0;

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
        defer embed_slice.free(allocator);

        const n: usize = @intCast(embed_slice.shape.dims()[0]);
        const d: usize = @intCast(embed_slice.shape.dims()[1]);
        std.debug.assert(d == g.dim);
        std.debug.assert(d == quantizer.d);

        std.log.info("Test embed graph full top-k task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        zml_handler.tic(&zml_handler.timers.graph_search_tot);
        for (0..n) |embed_index| {
            const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];

            zml_handler.timers.nb_detokenize += 1;
            g.greedySearchPrefetch(embed, 0);

            const best_found_at = g.visited_at[g.visited[0].node];
            total_best_found_at += @intCast(best_found_at);
            min_best_found_at = @min(min_best_found_at, best_found_at);
            max_best_found_at = @max(max_best_found_at, best_found_at);

            const dense_sample = quantizer.sampleDense(embed);

            var found_mass: f32 = 0.0;
            var top_p_size: usize = 0;
            var top_p_found: usize = 0;
            var top_p_mass: f32 = 0.0;
            var found_real_top: [3]bool = .{ false, false, false };
            var found_top16_count: usize = 0;
            for (dense_sample.probas, dense_sample.rows, 0..) |proba, token, rank| {
                var found = false;
                for (0..g.L) |candidate_i| {
                    if (g.visited[candidate_i].node == token) {
                        found = true;
                        break;
                    }
                }

                if (rank < found_real_top.len) found_real_top[rank] = found;
                if (found) {
                    found_top16_count += 1;
                    found_mass += proba;
                }
                if (top_p_mass < top_p) {
                    top_p_mass += proba;
                    top_p_size = rank + 1;
                    if (found) top_p_found += 1;
                }
            }
            if (found_top16_count == 0 or found_mass < 0.5) {
                try logTop16MissDetails(tokenizer, g, quantizer, embed, dense_sample, found_top16_count, found_mass);
            }

            total_count += 1;
            total_found_mass += @as(f64, @floatCast(found_mass));
            worst_found_mass = @min(worst_found_mass, found_mass);
            total_top_p_found += top_p_found;
            worst_top_p_found = @min(worst_top_p_found, top_p_found);
            total_top_p_size += top_p_size;
            max_top_p_size = @max(max_top_p_size, top_p_size);
            if (found_real_top[0]) found_top1_count += 1;
            if (top_p_size < 2 or (found_real_top[0] and found_real_top[1])) found_top1_top2_count += 1;
            if (top_p_size < 3 or (found_real_top[0] and found_real_top[1] and found_real_top[2])) found_top1_top2_top3_count += 1;
        }
        zml_handler.toc(&zml_handler.timers.graph_search_tot);
    }

    const avg_found_mass = total_found_mass / @as(f64, @floatFromInt(total_count));
    const avg_top_p_found = @as(f64, @floatFromInt(total_top_p_found)) / @as(f64, @floatFromInt(total_count));
    const avg_top_p_size = @as(f64, @floatFromInt(total_top_p_size)) / @as(f64, @floatFromInt(total_count));
    const found_top1_percent = 100.0 * @as(f64, @floatFromInt(found_top1_count)) / @as(f64, @floatFromInt(total_count));
    const found_top2_percent = 100.0 * @as(f64, @floatFromInt(found_top1_top2_count)) / @as(f64, @floatFromInt(total_count));
    const found_top3_percent = 100.0 * @as(f64, @floatFromInt(found_top1_top2_top3_count)) / @as(f64, @floatFromInt(total_count));
    const avg_best_found_at = @as(f64, @floatFromInt(total_best_found_at)) / @as(f64, @floatFromInt(total_count));
    std.log.info("Embed graph full top16: total={d}", .{total_count});
    std.log.info("Real top-16 probability mass retained by graph search: avg={d:.6} worst={d:.6}", .{ avg_found_mass, worst_found_mass });
    std.log.info("Top-p ({d}) average size={d:.2}, max size={d}", .{ top_p, avg_top_p_size, max_top_p_size });
    std.log.info("Graph found in top-p ({d}): avg={d:.2} worst={d}", .{ top_p, avg_top_p_found, worst_top_p_found });
    std.log.info("Graph final best found_at: min={d} max={d} avg={d:.2}", .{ min_best_found_at, max_best_found_at, avg_best_found_at });
    std.log.info("Graph found dense top prefixes: top1={d}/{d} ({d:.4}%), top1+top2={d}/{d} ({d:.4}%), top1+top2+top3={d}/{d} ({d:.4}%)", .{
        found_top1_count,
        total_count,
        found_top1_percent,
        found_top1_top2_count,
        total_count,
        found_top2_percent,
        found_top1_top2_top3_count,
        total_count,
        found_top3_percent,
    });
}

pub fn testEmbedDualGraphSearch(zml_handler: *Zml_handler, g1: *Graph, g2: *Graph, sampler: *Sampler, s: []const u8) !void {
    std.log.info("\n********** Test embed dual graph search {s}", .{s});

    var total_count: usize = 0;
    var found_top1_count: usize = 0;

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
        std.debug.assert(d == g1.dim);

        std.log.info("Test embed graph search task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        zml_handler.tic(&zml_handler.timers.graph_search_tot);
        for (0..n) |embed_index| {
            const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];

            zml_handler.timers.nb_detokenize += 1;

            g1.greedySearch(embed);

            if (g1.params.graph_type == .Angular) {
                for (0..g1.L) |i| {
                    g1.visited[i].similarity *= g1.lm_head.row_norms[g1.visited[i].node];
                }
            } else {
                for (0..g1.L) |i| {
                    g1.visited[i].similarity /= g1.lm_head.row_norms[g1.visited[i].node];
                }
            }
            g2.initSearchPool(g1.visited[0..g1.L]);
            g2.greedySearchWS(embed);

            const nb_visited = g1.nb_scored + g2.nb_scored;
            total_visited += nb_visited;
            min_visited = @min(min_visited, nb_visited);
            max_visited = @max(max_visited, nb_visited);

            total_count += 1;
            const top1_token = top1[embed_index];
            var found_top1 = false;
            for (0..g1.L) |i| {
                if (g1.visited[i].node == top1_token) found_top1 = true;
            }
            for (0..g2.L) |i| {
                if (g2.visited[i].node == top1_token) found_top1 = true;
            }
            if (found_top1) {
                found_top1_count += 1;
            } else {
                _ = sampler;
                //std.log.info("Missed top1, id {d} str {s}", .{ top1[embed_index], try tokens.tokenString(sampler.tokenizer, top1[embed_index], sampler.allocator) });
                //std.log.info("Found instead {d} str {s}", .{ g.visited[0].node, try tokens.tokenString(sampler.tokenizer, g.visited[0].node, sampler.allocator) });
            }
        }
        zml_handler.toc(&zml_handler.timers.graph_search_tot);
    }

    const percent_found = 100.0 * @as(f64, @floatFromInt(found_top1_count)) / @as(f64, @floatFromInt(total_count));
    const average_visit = @as(f64, @floatFromInt(total_visited)) / @as(f64, @floatFromInt(total_count));
    std.log.info("Embed graph search: total={d} found_top1={d} ({d:.4}%)", .{ total_count, found_top1_count, percent_found });
    std.log.info("Embed graph search nb_visited: min={d} max={d} avg={d:.2}", .{ min_visited, max_visited, average_visit });
}

pub fn testEmbedQuantizedSearch(zml_handler: *Zml_handler, quantizer: *Quantized) !void {
    std.log.info("Test embed Quantized search", .{});

    const PartialDotStats = struct {
        label: []const u8,
        found_top1: usize = 0,
        missing_top16: usize = 0,
        total_dense_scored: usize = 0,
        total_partial_scored: usize = 0,
        total_pruned: usize = 0,
        total_pruned_by_phase: [quantized.partial_dot_block_count]usize = [_]usize{0} ** quantized.partial_dot_block_count,
        min_dense_scored: usize = std.math.maxInt(usize),
        max_dense_scored: usize = 0,
        min_partial_scored: usize = std.math.maxInt(usize),
        max_partial_scored: usize = 0,
        min_pruned: usize = std.math.maxInt(usize),
        max_pruned: usize = 0,

        fn record(self: *@This(), sample: quantized.TwoPhaseSample, top1_token: usize) void {
            self.total_dense_scored += sample.nb_scored;
            self.total_partial_scored += sample.nb_partial_dot_scored;
            self.total_pruned += sample.nb_partial_dot_pruned;
            for (sample.partial_dot_pruned_by_phase, 0..) |pruned, phase_i| {
                self.total_pruned_by_phase[phase_i] += pruned;
            }
            self.min_dense_scored = @min(self.min_dense_scored, sample.nb_scored);
            self.max_dense_scored = @max(self.max_dense_scored, sample.nb_scored);
            self.min_partial_scored = @min(self.min_partial_scored, sample.nb_partial_dot_scored);
            self.max_partial_scored = @max(self.max_partial_scored, sample.nb_partial_dot_scored);
            self.min_pruned = @min(self.min_pruned, sample.nb_partial_dot_pruned);
            self.max_pruned = @max(self.max_pruned, sample.nb_partial_dot_pruned);

            for (sample.rows) |tok| {
                if (tok == top1_token) {
                    self.found_top1 += 1;
                    return;
                }
            }
            self.missing_top16 += 1;
        }

        fn log(self: @This(), total_count: usize) void {
            const inv_total = 1.0 / @as(f64, @floatFromInt(total_count));
            const percent_found = 100.0 * @as(f64, @floatFromInt(self.found_top1)) * inv_total;
            const avg_dense_scored = @as(f64, @floatFromInt(self.total_dense_scored)) * inv_total;
            const avg_partial_scored = @as(f64, @floatFromInt(self.total_partial_scored)) * inv_total;
            const avg_pruned = @as(f64, @floatFromInt(self.total_pruned)) * inv_total;

            std.log.info("{s} found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ self.label, self.found_top1, percent_found, self.missing_top16 });
            std.log.info("{s} row-block scored: min={d} max={d} avg={d:.2}", .{ self.label, self.min_partial_scored, self.max_partial_scored, avg_partial_scored });
            std.log.info("{s} rows pruned: min={d} max={d} avg={d:.2}", .{ self.label, self.min_pruned, self.max_pruned, avg_pruned });
            std.log.info("{s} dense exact scored rows: min={d} max={d} avg={d:.2}", .{ self.label, self.min_dense_scored, self.max_dense_scored, avg_dense_scored });
            for (self.total_pruned_by_phase, 0..) |phase_pruned, phase_i| {
                std.log.info("{s} phase {d:>2} pruned: total={d} avg={d:.2}", .{ self.label, phase_i + 1, phase_pruned, @as(f64, @floatFromInt(phase_pruned)) * inv_total });
            }
        }
    };

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
    var tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    var total_count: usize = 0;
    var found_top1_1bit_count: usize = 0;
    var found_top1_2bits_count: usize = 0;
    var found_top1_3phases_count: usize = 0;
    var found_top1_4phases_count: usize = 0;
    var found_top1_5phases_count: usize = 0;
    var missing_top16_1bit_count: usize = 0;
    var missing_top16_2bits_count: usize = 0;
    var missing_top16_3phases_count: usize = 0;
    var missing_top16_4phases_count: usize = 0;
    var missing_top16_5phases_count: usize = 0;
    var total_dense_scored_1bit: usize = 0;
    var total_dense_scored_2bits: usize = 0;
    var total_dense_scored_3phases: usize = 0;
    var total_dense_scored_4phases: usize = 0;
    var total_dense_scored_5phases: usize = 0;
    var total_1bit_scored_5phases: usize = 0;
    var total_half_1bit_scored_5phases: usize = 0;
    var total_2bit_scored_3phases: usize = 0;
    var total_2bit_scored_4phases: usize = 0;
    var total_2bit_scored_5phases: usize = 0;
    var min_dense_scored_1bit: usize = std.math.maxInt(usize);
    var min_dense_scored_2bits: usize = std.math.maxInt(usize);
    var min_dense_scored_3phases: usize = std.math.maxInt(usize);
    var min_dense_scored_4phases: usize = std.math.maxInt(usize);
    var min_dense_scored_5phases: usize = std.math.maxInt(usize);
    var min_1bit_scored_5phases: usize = std.math.maxInt(usize);
    var min_half_1bit_scored_5phases: usize = std.math.maxInt(usize);
    var min_2bit_scored_3phases: usize = std.math.maxInt(usize);
    var min_2bit_scored_4phases: usize = std.math.maxInt(usize);
    var min_2bit_scored_5phases: usize = std.math.maxInt(usize);
    var max_dense_scored_1bit: usize = 0;
    var max_dense_scored_2bits: usize = 0;
    var max_dense_scored_3phases: usize = 0;
    var max_dense_scored_4phases: usize = 0;
    var max_dense_scored_5phases: usize = 0;
    var max_1bit_scored_5phases: usize = 0;
    var max_half_1bit_scored_5phases: usize = 0;
    var max_2bit_scored_3phases: usize = 0;
    var max_2bit_scored_4phases: usize = 0;
    var max_2bit_scored_5phases: usize = 0;
    var partial_dot_estimator_stats: PartialDotStats = .{ .label = "partial-dot estimator" };

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
        std.debug.assert(d == quantizer.d);

        std.log.info("Test quantized graph search task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        for (0..n) |embed_index| {
            const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];
            const top1_token = top1[embed_index];

            const sample_1bit = try quantizer.sample2Phase1Bit(embed);
            const sample_2bits = try quantizer.sample2Phase2Bits(embed);
            const sample_3phases = try quantizer.sample3Phases(embed);
            const sample_4phases = try quantizer.sample4Phases(embed);
            const sample_5phases = try quantizer.sample5PhasesOptimized(embed);

            zml_handler.tic(&zml_handler.timers.quant_search);
            const sample_partial_dot_estimator = try quantizer.sampleMultiPhasePartialDot(embed);
            zml_handler.toc(&zml_handler.timers.quant_search);

            if (sample_1bit.nb_scored > 93340) {
                try quantizer.sample2Phase1BitLog(embed, &tokenizer);
                try quantizer.sample2Phase2BitsLog(embed, &tokenizer);
                //return;
            }
            if (sample_2bits.nb_scored > 63) {
                std.log.info("Needed {d} re scores", .{sample_2bits.nb_scored});
                //try quantizer.sample2Phase1BitLog(embed, &tokenizer);
                //try quantizer.sample2Phase2BitsLog(embed, &tokenizer);
                //return;
            }

            zml_handler.timers.nb_detokenize += 1;
            total_count += 1;

            total_dense_scored_1bit += sample_1bit.nb_scored;
            total_dense_scored_2bits += sample_2bits.nb_scored;
            total_dense_scored_3phases += sample_3phases.nb_scored;
            total_dense_scored_4phases += sample_4phases.nb_scored;
            total_dense_scored_5phases += sample_5phases.nb_scored;
            total_1bit_scored_5phases += sample_5phases.nb_1bit_scored;
            total_half_1bit_scored_5phases += sample_5phases.nb_half_1bit_scored;
            total_2bit_scored_3phases += sample_3phases.nb_2bit_scored;
            total_2bit_scored_4phases += sample_4phases.nb_2bit_scored;
            total_2bit_scored_5phases += sample_5phases.nb_2bit_scored;
            partial_dot_estimator_stats.record(sample_partial_dot_estimator, top1_token);
            min_dense_scored_1bit = @min(min_dense_scored_1bit, sample_1bit.nb_scored);
            min_dense_scored_2bits = @min(min_dense_scored_2bits, sample_2bits.nb_scored);
            min_dense_scored_3phases = @min(min_dense_scored_3phases, sample_3phases.nb_scored);
            min_dense_scored_4phases = @min(min_dense_scored_4phases, sample_4phases.nb_scored);
            min_dense_scored_5phases = @min(min_dense_scored_5phases, sample_5phases.nb_scored);
            min_1bit_scored_5phases = @min(min_1bit_scored_5phases, sample_5phases.nb_1bit_scored);
            min_half_1bit_scored_5phases = @min(min_half_1bit_scored_5phases, sample_5phases.nb_half_1bit_scored);
            min_2bit_scored_3phases = @min(min_2bit_scored_3phases, sample_3phases.nb_2bit_scored);
            min_2bit_scored_4phases = @min(min_2bit_scored_4phases, sample_4phases.nb_2bit_scored);
            min_2bit_scored_5phases = @min(min_2bit_scored_5phases, sample_5phases.nb_2bit_scored);
            max_dense_scored_1bit = @max(max_dense_scored_1bit, sample_1bit.nb_scored);
            max_dense_scored_2bits = @max(max_dense_scored_2bits, sample_2bits.nb_scored);
            max_dense_scored_3phases = @max(max_dense_scored_3phases, sample_3phases.nb_scored);
            max_dense_scored_4phases = @max(max_dense_scored_4phases, sample_4phases.nb_scored);
            max_dense_scored_5phases = @max(max_dense_scored_5phases, sample_5phases.nb_scored);
            max_1bit_scored_5phases = @max(max_1bit_scored_5phases, sample_5phases.nb_1bit_scored);
            max_half_1bit_scored_5phases = @max(max_half_1bit_scored_5phases, sample_5phases.nb_half_1bit_scored);
            max_2bit_scored_3phases = @max(max_2bit_scored_3phases, sample_3phases.nb_2bit_scored);
            max_2bit_scored_4phases = @max(max_2bit_scored_4phases, sample_4phases.nb_2bit_scored);
            max_2bit_scored_5phases = @max(max_2bit_scored_5phases, sample_5phases.nb_2bit_scored);

            var found_top1_1bit = false;
            for (sample_1bit.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_1bit = true;
                    break;
                }
            }
            if (found_top1_1bit) {
                found_top1_1bit_count += 1;
            } else {
                missing_top16_1bit_count += 1;
            }

            var found_top1_2bits = false;
            for (sample_2bits.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_2bits = true;
                    break;
                }
            }
            if (found_top1_2bits) {
                found_top1_2bits_count += 1;
            } else {
                //try quantizer.sample2Phase2BitsLog(embed, &tokenizer);
                missing_top16_2bits_count += 1;
            }

            var found_top1_3phases = false;
            for (sample_3phases.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_3phases = true;
                    break;
                }
            }
            if (found_top1_3phases) {
                found_top1_3phases_count += 1;
            } else {
                missing_top16_3phases_count += 1;
            }

            var found_top1_4phases = false;
            for (sample_4phases.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_4phases = true;
                    break;
                }
            }
            if (found_top1_4phases) {
                found_top1_4phases_count += 1;
            } else {
                missing_top16_4phases_count += 1;
            }

            var found_top1_5phases = false;
            for (sample_5phases.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_5phases = true;
                    break;
                }
            }
            if (found_top1_5phases) {
                found_top1_5phases_count += 1;
            } else {
                missing_top16_5phases_count += 1;
            }
        }
    }

    const inv_total = 1.0 / @as(f64, @floatFromInt(total_count));
    const percent_found_1bit = 100.0 * @as(f64, @floatFromInt(found_top1_1bit_count)) * inv_total;
    const percent_found_2bits = 100.0 * @as(f64, @floatFromInt(found_top1_2bits_count)) * inv_total;
    const percent_found_3phases = 100.0 * @as(f64, @floatFromInt(found_top1_3phases_count)) * inv_total;
    const percent_found_4phases = 100.0 * @as(f64, @floatFromInt(found_top1_4phases_count)) * inv_total;
    const percent_found_5phases = 100.0 * @as(f64, @floatFromInt(found_top1_5phases_count)) * inv_total;
    const avg_dense_scored_1bit = @as(f64, @floatFromInt(total_dense_scored_1bit)) * inv_total;
    const avg_dense_scored_2bits = @as(f64, @floatFromInt(total_dense_scored_2bits)) * inv_total;
    const avg_dense_scored_3phases = @as(f64, @floatFromInt(total_dense_scored_3phases)) * inv_total;
    const avg_dense_scored_4phases = @as(f64, @floatFromInt(total_dense_scored_4phases)) * inv_total;
    const avg_dense_scored_5phases = @as(f64, @floatFromInt(total_dense_scored_5phases)) * inv_total;
    const avg_1bit_scored_5phases = @as(f64, @floatFromInt(total_1bit_scored_5phases)) * inv_total;
    const avg_half_1bit_scored_5phases = @as(f64, @floatFromInt(total_half_1bit_scored_5phases)) * inv_total;
    const avg_2bit_scored_3phases = @as(f64, @floatFromInt(total_2bit_scored_3phases)) * inv_total;
    const avg_2bit_scored_4phases = @as(f64, @floatFromInt(total_2bit_scored_4phases)) * inv_total;
    const avg_2bit_scored_5phases = @as(f64, @floatFromInt(total_2bit_scored_5phases)) * inv_total;
    std.log.info("Embed quantized search: total={d}", .{total_count});
    std.log.info("1-bit found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_1bit_count, percent_found_1bit, missing_top16_1bit_count });
    std.log.info("2-bit found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_2bits_count, percent_found_2bits, missing_top16_2bits_count });
    std.log.info("3-phase found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_3phases_count, percent_found_3phases, missing_top16_3phases_count });
    std.log.info("4-phase found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_4phases_count, percent_found_4phases, missing_top16_4phases_count });
    std.log.info("5-phase found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_5phases_count, percent_found_5phases, missing_top16_5phases_count });
    partial_dot_estimator_stats.log(total_count);
    std.log.info("1-bit dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_1bit, max_dense_scored_1bit, avg_dense_scored_1bit });
    std.log.info("2-bit dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_2bits, max_dense_scored_2bits, avg_dense_scored_2bits });
    std.log.info("3-phase 2-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_2bit_scored_3phases, max_2bit_scored_3phases, avg_2bit_scored_3phases });
    std.log.info("3-phase dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_3phases, max_dense_scored_3phases, avg_dense_scored_3phases });
    std.log.info("4-phase 2-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_2bit_scored_4phases, max_2bit_scored_4phases, avg_2bit_scored_4phases });
    std.log.info("4-phase dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_4phases, max_dense_scored_4phases, avg_dense_scored_4phases });
    std.log.info("5-phase half 1-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_half_1bit_scored_5phases, max_half_1bit_scored_5phases, avg_half_1bit_scored_5phases });
    std.log.info("5-phase full 1-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_1bit_scored_5phases, max_1bit_scored_5phases, avg_1bit_scored_5phases });
    std.log.info("5-phase 2-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_2bit_scored_5phases, max_2bit_scored_5phases, avg_2bit_scored_5phases });
    std.log.info("5-phase dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_5phases, max_dense_scored_5phases, avg_dense_scored_5phases });
}

pub fn testEmbedQuantizedPQSearch(zml_handler: *Zml_handler, quantizer: *ProductQuantizerFastScan) !void {
    std.log.info("Test embed ProductQuantized search", .{});

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
    var tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    var sampler: Sampler = try .init(zml_handler, quantizer.lm_head);
    defer Sampler.deinit(&sampler);

    var total_count: usize = 0;
    var found_top1_count: usize = 0;
    var missing_top16_count: usize = 0;
    var total_dense_scored: usize = 0;
    var total_pq_scored: usize = 0;
    var total_pruned: usize = 0;
    var min_dense_scored: usize = std.math.maxInt(usize);
    var max_dense_scored: usize = 0;
    var min_pq_scored: usize = std.math.maxInt(usize);
    var max_pq_scored: usize = 0;
    var min_pruned: usize = std.math.maxInt(usize);
    var max_pruned: usize = 0;

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

        std.log.info("Test product quantized sampling task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        for (0..n) |embed_index| {
            const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];
            const top1_token = top1[embed_index];

            zml_handler.timers.nb_detokenize += 1;
            zml_handler.tic(&zml_handler.timers.quant_search);
            //if (embed_index < 5) quantizer.sampleLog(&sampler, embed);
            const pq_sample = quantizer.sample(embed);
            zml_handler.toc(&zml_handler.timers.quant_search);

            total_count += 1;
            total_dense_scored += pq_sample.nb_dense_scored;
            total_pq_scored += pq_sample.nb_pq_scored;
            total_pruned += pq_sample.nb_pruned;
            min_dense_scored = @min(min_dense_scored, pq_sample.nb_dense_scored);
            max_dense_scored = @max(max_dense_scored, pq_sample.nb_dense_scored);
            min_pq_scored = @min(min_pq_scored, pq_sample.nb_pq_scored);
            max_pq_scored = @max(max_pq_scored, pq_sample.nb_pq_scored);
            min_pruned = @min(min_pruned, pq_sample.nb_pruned);
            max_pruned = @max(max_pruned, pq_sample.nb_pruned);

            var found_top1 = false;
            for (pq_sample.top_k) |logit| {
                if (logit.row == top1_token) {
                    found_top1 = true;
                    break;
                }
            }

            if (found_top1) {
                found_top1_count += 1;
            } else {
                missing_top16_count += 1;
            }
        }
    }

    const inv_total = 1.0 / @as(f64, @floatFromInt(total_count));
    const percent_found = 100.0 * @as(f64, @floatFromInt(found_top1_count)) * inv_total;
    const avg_dense_scored = @as(f64, @floatFromInt(total_dense_scored)) * inv_total;
    const avg_pq_scored = @as(f64, @floatFromInt(total_pq_scored)) * inv_total;
    const avg_pruned = @as(f64, @floatFromInt(total_pruned)) * inv_total;
    std.log.info("Embed product quantized search: total={d}", .{total_count});
    std.log.info("PQ prune found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_count, percent_found, missing_top16_count });
    std.log.info("PQ row-bucket scored: min={d} max={d} avg={d:.2}", .{ min_pq_scored, max_pq_scored, avg_pq_scored });
    std.log.info("PQ rows pruned: min={d} max={d} avg={d:.2}", .{ min_pruned, max_pruned, avg_pruned });
    std.log.info("PQ dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored, max_dense_scored, avg_dense_scored });
}

pub fn testEmbedGraphQuantizedSearch(zml_handler: *Zml_handler, g: *Graph, quantizer: *Quantized) !void {
    std.log.info("Test embed GraphQuantized search", .{});

    var total_count: usize = 0;
    var found_top1: [4]usize = [_]usize{0} ** 4;
    var found_top1_acc: [4]usize = [_]usize{0} ** 4;
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
        std.debug.assert(d == quantizer.d);

        std.log.info("Test quantized graph search task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        zml_handler.tic(&zml_handler.timers.graph_search_tot);
        for (0..n) |embed_index| {
            const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];

            zml_handler.timers.nb_detokenize += 1;
            const top1_token = top1[embed_index];
            var scored: usize = 0;
            var found_acc = false;
            for (0..4) |start| {
                const entry_point: u32 = @as(u32, @intCast(start)) * (g.n / 4);
                g.greedySearchPrefetch(embed, entry_point);
                scored += g.nb_scored;
                var found = false;
                for (0..g.L) |i| {
                    if (g.visited[i].node == top1_token) {
                        found = true;
                        break;
                    }
                }
                found_acc |= found;
                if (found) found_top1[start] += 1;
                if (found_acc) found_top1_acc[start] += 1;
            }

            const nb_visited = scored;
            total_visited += nb_visited;
            min_visited = @min(min_visited, nb_visited);
            max_visited = @max(max_visited, nb_visited);
            total_count += 1;

            //g.params.search_budget = 2048;
            //try g.greedySearchQuantized2x2(embed, quantizer);

            //g.params.search_budget = 1024;
            //g.quantizedCrossover(embed);
        }
        zml_handler.toc(&zml_handler.timers.graph_search_tot);
    }

    const average_visit = @as(f64, @floatFromInt(total_visited)) / @as(f64, @floatFromInt(total_count));
    std.log.info("Embed quantized search: total={d}", .{total_count});
    for (0..4) |start| {
        const percent_found = 100.0 * @as(f64, @floatFromInt(found_top1[start])) / @as(f64, @floatFromInt(total_count));
        std.log.info("Pass {d} found_top1={d} ({d:.4}%)", .{ start, found_top1[start], percent_found });
    }
    for (0..4) |start| {
        const percent_found = 100.0 * @as(f64, @floatFromInt(found_top1_acc[start])) / @as(f64, @floatFromInt(total_count));
        std.log.info("Pass up to {d} found_top1_acc={d} ({d:.4}%)", .{ start, found_top1_acc[start], percent_found });
    }
    std.log.info("Embed graph search nb_visited: min={d} max={d} avg={d:.2}", .{ min_visited, max_visited, average_visit });
}
