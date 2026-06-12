const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");

const log = std.log;
const Tokenizer = zml.tokenizer.Tokenizer;

pub const GraphParams = struct {
    k_max: usize = 16,
    search_budget: usize = 512,
    alpha: f32 = 1.15,
    vamana_passes: usize = 2,
    top_k: usize = 16,
    L: usize = 128,
};

pub const Graph = struct {
    pub const Candidate = struct {
        node: usize,
        similarity: f32,

        fn beforeThan(_: void, lhs: Candidate, rhs: Candidate) bool {
            return lhs.similarity > rhs.similarity or (lhs.similarity == rhs.similarity and lhs.node < rhs.node);
        }
    };

    zml_handler: *main.Zml_handler,
    allocator: std.mem.Allocator,
    io: std.Io,
    // dataset fields
    dim: usize,
    lm_head: zml.Slice,
    lm_head_normalized: zml.Slice,
    similarity_matrix: *main.SimilarityMatrix,
    lm_head_row_norms: zml.Slice,
    // graph fields
    // TODO: we can use smaller integer types
    n: usize,
    params: GraphParams,
    neighbors: []usize,
    nb_neighbors: []usize,
    is_junk: []bool,
    // TODO: for NSW extension, we could store similarities relative to neighbors
    // either in a separate array or as a Neighbor struct { .node, .similarity }
    // search fields
    medoid: usize,
    // the max number of candidates, each candidate is a scored node so
    // this can be set as the max_budget + k_max
    capacity: usize,
    // the active number of candidates, the pool is kept sorted in the
    // range 0..L : this is the active pool
    L: usize,
    // the pool is composed of all scored nodes during this search
    visited: []Candidate,
    is_visited: []bool,
    nb_visited: usize,
    // for each candidate in visited, tells if it's neighbors have been
    // added to the pool (when true, the node had been dealt with)
    is_expanded: []bool,
    is_search_done: bool,
    // for each node of the graph, tells if neighbors have been obtained
    // with the pruning heuristic. in this case no neighbor would be removed
    // if we run the pruning again
    are_neighbors_pruned: []bool,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: zml.Slice, lm_head_normalized: zml.Slice, matrix: *main.SimilarityMatrix, lm_head_row_norms: zml.Slice, junk_rows: []const usize, params: GraphParams) !Graph {
        std.debug.assert(matrix.n > 0);
        std.debug.assert(params.k_max > 0);
        std.debug.assert(params.L > 0);
        std.debug.assert(params.search_budget > 0);
        std.debug.assert(params.alpha >= 1.0);
        std.debug.assert(matrix.k >= params.k_max);
        std.debug.assert(params.L <= params.search_budget + params.k_max);
        std.debug.assert(params.search_budget + params.k_max <= matrix.n);
        std.debug.assert(params.k_max < matrix.n);
        std.debug.assert(lm_head_row_norms.constItems(f32).len == matrix.n);

        const allocator = zml_handler.allocator;

        const is_junk = try allocator.alloc(bool, matrix.n);
        errdefer allocator.free(is_junk);
        @memset(is_junk, false);
        for (junk_rows) |row| is_junk[row] = true;

        const medoid = try getMedoid(allocator, lm_head_normalized, matrix.n, matrix.d, is_junk);

        const neighbors = try allocator.alloc(usize, matrix.n * params.k_max);
        errdefer allocator.free(neighbors);

        const nb_neighbors = try allocator.alloc(usize, matrix.n);
        errdefer allocator.free(nb_neighbors);
        @memset(nb_neighbors, 0);

        const is_visited = try allocator.alloc(bool, matrix.n);
        errdefer allocator.free(is_visited);
        @memset(is_visited, false);

        const is_expanded = try allocator.alloc(bool, params.search_budget + params.k_max);
        errdefer allocator.free(is_expanded);
        @memset(is_expanded, false);

        const are_neighbors_pruned = try allocator.alloc(bool, matrix.n);
        errdefer allocator.free(are_neighbors_pruned);
        @memset(are_neighbors_pruned, false);

        return .{
            .zml_handler = zml_handler,
            .allocator = allocator,
            .io = zml_handler.io,
            .dim = matrix.d,
            .lm_head = lm_head,
            .lm_head_normalized = lm_head_normalized,
            .n = matrix.n,
            .params = params,
            .neighbors = neighbors,
            .nb_neighbors = nb_neighbors,
            .is_junk = is_junk,
            .similarity_matrix = matrix,
            .lm_head_row_norms = lm_head_row_norms,
            .medoid = medoid,
            .capacity = params.search_budget + params.k_max,
            .L = 0,
            .visited = try allocator.alloc(Candidate, params.search_budget + params.k_max),
            .is_visited = is_visited,
            .nb_visited = 0,
            .is_expanded = is_expanded,
            .is_search_done = false,
            .are_neighbors_pruned = are_neighbors_pruned,
        };
    }

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.neighbors);
        self.allocator.free(self.nb_neighbors);
        self.allocator.free(self.is_junk);
        self.allocator.free(self.visited);
        self.allocator.free(self.is_visited);
        self.allocator.free(self.is_expanded);
        self.allocator.free(self.are_neighbors_pruned);
    }

    pub fn getMedoid(allocator: std.mem.Allocator, lm_head_normalized: zml.Slice, n: usize, dim: usize, is_junk: []const bool) !usize {
        const rows = lm_head_normalized.constItems(f32);

        // compute normalized average using f64 for accurate accumulation
        const average = try allocator.alloc(f64, dim);
        defer allocator.free(average);
        @memset(average, 0.0);
        var nb_rows: usize = 0;
        for (0..n) |i| {
            if (is_junk[i]) continue;
            nb_rows += 1;
            const row_i = rows[i * dim ..][0..dim];
            for (0..dim) |j| {
                average[j] += @floatCast(row_i[j]);
            }
        }
        std.debug.assert(nb_rows > 0);
        const inv_n = 1.0 / @as(f64, @floatFromInt(nb_rows));
        var norm2: f64 = 0.0;
        for (0..dim) |i| {
            average[i] *= inv_n;
            norm2 += average[i] * average[i];
        }
        const inv_norm = 1.0 / @sqrt(norm2);
        const average_f32 = try allocator.alloc(f32, dim);
        defer allocator.free(average_f32);
        for (0..dim) |i| {
            average_f32[i] = @floatCast(average[i] * inv_norm);
        }

        // medoid is the row that is most similar to the average
        var best_row: usize = 0;
        var best_similarity: f32 = -std.math.inf(f32);
        for (0..n) |i| {
            if (is_junk[i]) continue;
            const row_i = rows[i * dim ..][0..dim];
            var row_i_similarity: f32 = 0.0;
            for (0..dim) |j| {
                row_i_similarity += row_i[j] * average_f32[j];
            }
            if (row_i_similarity > best_similarity) {
                best_similarity = row_i_similarity;
                best_row = i;
            }
        }
        return best_row;
    }

    // ------------------- Search functions ------------------ //

    pub fn greedySearchNode(self: *Graph, query: usize) void {
        std.debug.assert(query % 2 == 0);
        
        self.zml_handler.tic(&self.zml_handler.timers.greedy_search);
        std.debug.assert(!self.is_junk[query]);
        // initialize search at entry point
        self.initNodeSearch(query);

        while (self.nb_visited < self.params.search_budget) {

            // find best node of the active pool that has not been expanded yet
            const node = self.popCandidate();

            // if all nodes in active pool have been expanded, terminate the search
            if (self.is_search_done) break;

            // TODO: we can batch insert to avoid doing several revert insert passes
            const start_neigh = self.params.k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                if (self.is_visited[neighbor]) continue;
                self.addNodeCandidate(query, neighbor);
            }
        }
        self.cleanup();
        self.zml_handler.toc(&self.zml_handler.timers.greedy_search);
    }

    pub fn greedySearch(self: *Graph, query: []const f32) void {
        // initialize search at entry point
        self.initSearch(query);

        while (self.nb_visited < self.params.search_budget) {

            // find best node of the active pool that has not been expanded yet
            const node = self.popCandidate();

            // if all nodes in active pool have been expanded, terminate the search
            if (self.is_search_done) break;

            // TODO: switch from atomic neighbor extension to incremental extension
            // store nb_expanded_neighbors instead of a boolean flag, and expand only
            // ne neighbor at a time. If a neighbor at a small position has much better
            // similarity with query than the base node, we want to focus of this neighbor
            // neighbors instead of the remaining neighbors of the base node. This can save
            // a lot of distance computations.
            const start_neigh = self.params.k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                if (self.is_visited[neighbor]) continue;
                self.addCandidate(query, neighbor);
            }
        }

        self.cleanup();
    }

    
    pub fn greedySearchWithLog(self: *Graph, query: []const f32, tokenizer: Tokenizer) !void {
        self.initSearch(query);
        errdefer self.cleanup();

        const row_norms = self.lm_head_row_norms.constItems(f32);
        var iter: usize = 0;
        log.info("Greedy search trace", .{});
        log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s:>10}  {s}", .{ "iter", "node", "similarity", "row_norm", "visited", "token" });
        log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s:>10}  {s}", .{ "------", "----------", "--------------", "--------------", "----------", "-----" });

        while (self.nb_visited < self.params.search_budget) {
            const node = self.popCandidate();
            if (self.is_search_done) break;

            try self.logGreedySearchNode(tokenizer, iter, node, self.candidateSimilarity(node), row_norms[node]);
            iter += 1;

            const start_neigh = self.params.k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                if (self.is_visited[neighbor]) continue;
                self.addCandidate(query, neighbor);
            }
        }

        self.cleanup();
    }

    fn candidateSimilarity(self: *const Graph, node: usize) f32 {
        for (self.visited[0..self.nb_visited]) |candidate| {
            if (candidate.node == node) return candidate.similarity;
        }
        unreachable;
    }

    fn logGreedySearchNode(self: *const Graph, tokenizer: Tokenizer, iter: usize, node: usize, sim: f32, row_norm: f32) !void {
        var decoded_buf: [512]u8 = undefined;
        const decoded = try decodeToken(tokenizer, @intCast(node), &decoded_buf);
        var escaped_buf: [512]u8 = undefined;
        const escaped = escapeTokenText(decoded, &escaped_buf);
        log.info("{d:>6}  {d:>10}  {d:>14.8}  {d:>14.6}  {d:>10}  {s}", .{ iter, node, sim, row_norm, self.nb_visited, escaped });
    }

    fn decodeToken(tokenizer: Tokenizer, token_id: u32, out: []u8) ![]const u8 {
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();

        const chunk = try decoder.feedOne(token_id, out);
        const final_chunk = try decoder.finalize(out[chunk.len..]);
        return out[0 .. chunk.len + final_chunk.len];
    }

    fn escapeTokenText(text: []const u8, out: []u8) []const u8 {
        var len: usize = 0;
        for (text) |c| {
            const replacement = switch (c) {
                '\n' => "\\n",
                '\r' => "\\r",
                '\t' => "\\t",
                '\\' => "\\\\",
                else => null,
            };
            if (replacement) |rep| {
                if (len + rep.len > out.len) break;
                @memcpy(out[len..][0..rep.len], rep);
                len += rep.len;
            } else {
                if (len + 1 > out.len) break;
                out[len] = if (std.ascii.isControl(c)) '?' else c;
                len += 1;
            }
        }
        return out[0..len];
    }

    
    pub fn scoreQueryNode(self: *const Graph, query: []const f32, node: usize) f32 {
        std.debug.assert(!self.is_junk[node]);
        const rows = self.lm_head_normalized.constItems(f32);
        const row = rows[node * self.dim ..][0..self.dim];
        var dot: f32 = 0;
        for (0..self.dim) |i| {
            dot += query[i] * row[i];
        }
        return dot;
    }

    pub fn initNodeSearch(self: *Graph, query: usize) void {
        // at start, pool is empty
        std.debug.assert(!self.is_visited[self.medoid]);

        // score query against medoid
        const sim = self.similarity(self.medoid, query);

        // medoid is the first and only visited node
        self.is_visited[self.medoid] = true;
        self.visited[0] = .{ .node = self.medoid, .similarity = sim };
        self.nb_visited = 1;
        self.L = 1;
        self.is_search_done = false;
    }

    pub fn initSearch(self: *Graph, query: []const f32) void {
        // at start, pool is empty
        std.debug.assert(!self.is_visited[self.medoid]);

        // score query against medoid
        const sim = self.scoreQueryNode(query, self.medoid);

        // medoid is the first and only visited node
        self.is_visited[self.medoid] = true;
        self.visited[0] = .{ .node = self.medoid, .similarity = sim };
        self.nb_visited = 1;
        self.L = 1;
        self.is_search_done = false;
    }

    pub fn addNodeCandidate(self: *Graph, query: usize, node: usize) void {
        std.debug.assert(!self.is_junk[node]);
        std.debug.assert(!self.is_visited[node]);
        std.debug.assert(self.nb_visited > 0);
        const sim = self.similarity(node, query);
        self.insert(node, sim);
    }

    pub fn addCandidate(self: *Graph, query: []const f32, node: usize) void {
        std.debug.assert(!self.is_junk[node]);
        std.debug.assert(!self.is_visited[node]);
        std.debug.assert(self.nb_visited > 0);
        const sim = self.scoreQueryNode(query, node);
        self.insert(node, sim);
    }

    pub fn insert(self: *Graph, node: usize, sim: f32) void {
        std.debug.assert(!self.is_junk[node]);
        std.debug.assert(!self.is_visited[node]);
        std.debug.assert(self.nb_visited > 0);
        std.debug.assert(self.L > 0);
        // TODO: if we split pool management into visited and active pool, we can simplify
        // TODO: if we split the active pool into expanded and unexpanded, we can improve

        self.is_visited[node] = true;
        // this is the lowest score of the active pool
        const worse_L_score = self.visited[self.L - 1].similarity;

        if (worse_L_score > sim) {
            // if node has worse score, insert it directly at the end of the pool.
            // this handles both cases where
            // - active pool is full: the new node is in the pool but not in the active pool
            // - active pool is not full: the end the pool is the end of the active pool
            self.visited[self.nb_visited] = .{ .node = node, .similarity = sim };
            self.L = @min(self.L + 1, self.params.L);
            std.debug.assert(!self.is_expanded[self.nb_visited]);
        } else {
            // if the node is among best L nodes, there are two cases:
            // - the active pool is not full: we can do a reverse linear pass to
            //   find the exact insertion point and slide all worse pool elements
            //   1 position to the right
            // - the active pool is full: since the new node is among best L nodes,
            //   the current last node in the active pool will be ejected from it.
            //   We can move this last node to the end of the pool, making room for
            //   the new node to be inserted by the reverse pass exactly like in the
            //   first case.
            if (self.L == self.params.L) {
                // move worse pool element to the end of visited nodes
                self.visited[self.nb_visited] = self.visited[self.L - 1];
                self.is_expanded[self.nb_visited] = self.is_expanded[self.L - 1];
                self.L -= 1;
            }
            var i = self.L;
            while (i > 0 and sim > self.visited[i - 1].similarity) {
                self.visited[i] = self.visited[i - 1];
                self.is_expanded[i] = self.is_expanded[i - 1];
                i -= 1;
            }
            std.debug.assert(self.L < self.params.L);
            std.debug.assert(i < self.params.L);
            self.visited[i] = .{ .node = node, .similarity = sim };
            self.is_expanded[i] = false;
            self.L += 1;
        }
        self.nb_visited += 1;
    }

    pub fn popCandidate(self: *Graph) usize {
        // find the best unexpanded candidate in the active pool
        // since the pool is kept sorted, return the first found
        for (0..self.L) |i| {
            if (!self.is_expanded[i]) {
                self.is_expanded[i] = true;
                return self.visited[i].node;
            }
        }
        self.is_search_done = true;
        // return any visited node, the search is done
        return self.visited[0].node;
    }

    pub fn cleanup(self: *Graph) void {
        // TODO: if this is slow, we can use epoch based flags instead of booleans
        // we can store it on one byte, like a bool, and clean it only each 256 epochs
        // is_expanded can be left uncleaned, and initialized at insertion instead of this
        for (0..self.nb_visited) |i| {
            const node = self.visited[i].node;
            self.is_visited[node] = false;
            self.is_expanded[i] = false;
        }
    }

    // ------------- Local neighborhood functions -------------- //

    pub fn setRandomNeighbors(self: *Graph) void {
        log.info("Random neighbors", .{});
        var prng = std.Random.DefaultPrng.init(0);
        const random = prng.random();

        const selected = self.allocator.alloc(Candidate, self.n) catch @panic("OOM");
        defer self.allocator.free(selected);

        const is_selected = self.allocator.alloc(bool, self.n) catch @panic("OOM");
        defer self.allocator.free(is_selected);
        @memset(is_selected, false);

        var nb_selected: usize = 0;

        for (0..self.n) |i| {
            const start_neigh = self.params.k_max * i;
            nb_selected = 0;
            self.nb_neighbors[i] = 0;
            if (self.is_junk[i]) continue;
            is_selected[i] = true;

            // rejection method as k_max << n
            while (nb_selected < self.params.k_max) {
                const candidate = random.uintLessThan(usize, self.n);
                if (self.is_junk[candidate] or is_selected[candidate]) continue;
                // add valid neighbor
                is_selected[candidate] = true;
                selected[nb_selected] = .{ .node = candidate, .similarity = self.similarity(i, candidate) };
                nb_selected += 1;
            }

            std.mem.sort(Candidate, selected[0..nb_selected], {}, Candidate.beforeThan);

            for (0..nb_selected) |j| {
                const neigh = selected[j].node;
                self.neighbors[start_neigh + j] = neigh;
                is_selected[neigh] = false;
            }
            self.nb_neighbors[i] = nb_selected;
            is_selected[i] = false;

            if (i == 0 or (i + 1) % 10000 == 0 or i + 1 == self.n) {
                log.info("Random neighbors node {d}/{d}", .{ i + 1, self.n });
            }
        }
    }

    pub fn setNearestNeighbors(self: *Graph) void {
        log.info("Nearest neighbors", .{});
        for (0..self.n) |i| {
            const start_neigh = self.params.k_max * i;
            self.nb_neighbors[i] = 0;
            if (self.is_junk[i]) continue;
            for (0..self.similarity_matrix.k) |j| {
                const candidate = self.similarity_matrix.nearestNeighbor(i, j);
                if (self.is_junk[candidate]) continue;
                self.neighbors[start_neigh + self.nb_neighbors[i]] = candidate;
                self.nb_neighbors[i] += 1;
                if (self.nb_neighbors[i] == self.params.k_max) break;
            }
            if (i == 0 or (i + 1) % 10000 == 0 or i + 1 == self.n) {
                log.info("Nearest neighbors node {d}/{d}", .{ i + 1, self.n });
            }
        }
    }

    pub fn setNearestNeighborsLos(self: *Graph) void {
        log.info("Nearest neighbors LOS", .{});
        for (0..self.n) |i| {
            const start_neigh = self.params.k_max * i;
            self.nb_neighbors[i] = 0;
            if (self.is_junk[i]) continue;
            for (0..self.similarity_matrix.k) |j| {
                const candidate = self.similarity_matrix.nearestNeighbor(i, j);
                if (self.is_junk[candidate] or !self.lineOfSight(i, candidate)) continue;
                self.neighbors[start_neigh + self.nb_neighbors[i]] = candidate;
                self.nb_neighbors[i] += 1;
                if (self.nb_neighbors[i] == self.params.k_max) break;
            }
            if (i == 0 or (i + 1) % 5000 == 0 or i + 1 == self.n) {
                log.info("Nearest neighbors LOS node {d}/{d}", .{ i + 1, self.n });
            }
        }
    }

    pub fn pruneCandidates(self: *Graph, base: usize, candidates: []Candidate) void {
        std.debug.assert(!self.is_junk[base]);
        // TODO: split reverse/forward prune timers
        self.zml_handler.tic(&self.zml_handler.timers.prune_pool);
        // update current_node neighbors with LOS pruning of the candidates
        const start_neigh = base * self.params.k_max;
        self.nb_neighbors[base] = 0;
        var end_neigh = start_neigh;
        const inv_alpha_squared = 1.0 / (self.params.alpha * self.params.alpha);
        for (candidates) |candidate| {
            std.debug.assert(candidate.node != base);
            // The LOS heuristic decides if a candidate node can be added to base's neighbors
            // If any of base's neighbor is already close enough from candidate, then it's rejected,
            // as the routing base -> close_neighbor -> candidate is deemed sufficient
            // The "close enough" formula is alpha * dist(close_neighbor, candidate) <= dist(base, candidate)
            const threshold = 1.0 - (1.0 - candidate.similarity) * inv_alpha_squared;
            var pruned = false;
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                // for u and v norm 1 vectors, ||u - v||² = u² + v² - 2 <u,v> = 2 (1 - sim(u,v))
                // a||n - c|| <= ||b - c|| => a²||n - c||² <= ||b - c||²
                // 2a² (1 - sim(n,c)) <= 2 (1 - sim(b,c))
                // sim(n,c) >= 1 - (1 - sim(b,c)) / a²
                if (self.similarity(neighbor, candidate.node) >= threshold) {
                    pruned = true;
                    break;
                }
            }
            if (!pruned) {
                self.neighbors[end_neigh] = candidate.node;
                self.nb_neighbors[base] += 1;
                end_neigh += 1;
            }
            if (self.nb_neighbors[base] == self.params.k_max) break;
        }
        self.are_neighbors_pruned[base] = true;
        self.zml_handler.toc(&self.zml_handler.timers.prune_pool);
    }

    // ------------------- NSW main function -------------------- //

    pub fn extendToNsw(self: *Graph) void {
        var prng = std.Random.DefaultPrng.init(1);
        const random = prng.random();

        const order = self.allocator.alloc(usize, self.n) catch @panic("OOM");
        defer self.allocator.free(order);
        for (0..order.len) |i| order[i] = i;

        const candidates = self.allocator.alloc(Candidate, 2 * self.params.k_max + self.params.search_budget) catch @panic("OOM");
        defer self.allocator.free(candidates);

        const start = std.Io.Timestamp.now(self.io, .awake);
        const alpha = self.params.alpha;

        for (0..self.params.vamana_passes) |pass_i| {
            self.params.alpha = if (pass_i == 0) 1.0 else alpha;
            // random visit order
            var nb_swap = order.len - 1;
            while (nb_swap > 0) : (nb_swap -= 1) {
                const j = random.uintLessThan(usize, nb_swap + 1);
                std.mem.swap(usize, &order[nb_swap], &order[j]);
            }
            log.info("NSW pass {d}/{d}", .{ pass_i + 1, self.params.vamana_passes });
            for (0..order.len) |i| {
                // at this iteration, we will update current_node's neighbors and add current_node as a neighbor in candidate nodes
                const current_node = order[i];
                const start_neigh = self.params.k_max * current_node;
                var end_neigh = start_neigh + self.nb_neighbors[current_node];

                if (self.is_junk[current_node]) continue;
                var nb_candidates: usize = 0;

                // the candidates are current_node's neighbors and the visited nodes
                // since both lists are sorted and contain unique nodes, we can build
                // the sorted list of candidates in one linear forward pass
                self.greedySearchNode(current_node);
                // only first L positions are sorted
                std.mem.sort(Candidate, self.visited[self.L..self.nb_visited], {}, Candidate.beforeThan);
                var pos_in_neighbors: usize = start_neigh;
                var pos_in_visited: usize = 0;
                // if current_node was visited during the search, skip it in the visited pool
                // otherwise it will end up being a neighbor of itself
                // if it's visited, it will always be the best visited node and be in first place
                if (self.visited[0].node == current_node) pos_in_visited = 1;
                while (pos_in_neighbors < end_neigh and pos_in_visited < self.nb_visited) {
                    const neigh = self.neighbors[pos_in_neighbors];
                    const visit = self.visited[pos_in_visited].node;
                    const neigh_sim = self.similarity(neigh, current_node);
                    const visit_sim = self.visited[pos_in_visited].similarity;
                    if (neigh == visit) {
                        // there is a duplicate: skip it in the visited pool and iterate
                        candidates[nb_candidates] = .{ .node = neigh, .similarity = neigh_sim };
                        pos_in_visited += 1;
                        pos_in_neighbors += 1;
                        nb_candidates += 1;
                        continue;
                    }
                    if (neigh_sim > visit_sim) {
                        candidates[nb_candidates] = .{ .node = neigh, .similarity = neigh_sim };
                        pos_in_neighbors += 1;
                    } else {
                        candidates[nb_candidates] = .{ .node = visit, .similarity = visit_sim };
                        pos_in_visited += 1;
                    }
                    nb_candidates += 1;
                }
                // from here one of the two positions has reached the end, add remaining element from other one
                for (pos_in_neighbors..end_neigh) |j| {
                    const neigh = self.neighbors[j];
                    candidates[nb_candidates] = .{ .node = neigh, .similarity = self.similarity(neigh, current_node) };
                    nb_candidates += 1;
                }
                for (pos_in_visited..self.nb_visited) |j| {
                    const visit = self.visited[j].node;
                    candidates[nb_candidates] = .{ .node = visit, .similarity = self.visited[j].similarity };
                    nb_candidates += 1;
                }

                // forward prune on candidates
                self.pruneCandidates(current_node, candidates[0..nb_candidates]);

                // from there, we insert current_node into each of its neighbors
                end_neigh = start_neigh + self.nb_neighbors[current_node];
                for (start_neigh..end_neigh) |j| {
                    const neighbor = self.neighbors[j];
                    const start_neigh_neigh = self.params.k_max * neighbor;
                    const end_neigh_neigh = start_neigh_neigh + self.nb_neighbors[neighbor];
                    const sim = self.similarity(neighbor, current_node);

                    // if neighbor -> current_node exists, skip
                    if (self.hasNeighbor(neighbor, current_node)) continue;

                    // if candidate still has room, add current_node to its neighbors
                    if (self.nb_neighbors[neighbor] < self.params.k_max) {
                        // insert with reverse linear pass to keep neighbors sorted
                        var inser_pos = end_neigh_neigh;
                        while (inser_pos > start_neigh_neigh and sim > self.similarity(neighbor, self.neighbors[inser_pos - 1])) {
                            self.neighbors[inser_pos] = self.neighbors[inser_pos - 1];
                            inser_pos -= 1;
                        }
                        std.debug.assert(start_neigh_neigh <= inser_pos);
                        self.neighbors[inser_pos] = current_node;
                        self.nb_neighbors[neighbor] += 1;
                        continue;
                    }

                    // If neighbor row is already pruned and if current_node is further than the worse neighbor,
                    // then no room will be made by pruning and current node will never be added
                    const worse_neighbor_sim = self.similarity(neighbor, self.neighbors[end_neigh_neigh - 1]);
                    if (self.are_neighbors_pruned[neighbor] and worse_neighbor_sim > sim) continue;

                    // reverse candidates : neighbor's neighbors + current_node
                    nb_candidates = 0;
                    for (start_neigh_neigh..end_neigh_neigh) |k| {
                        const neigh_neigh = self.neighbors[k];
                        // since neighbors are unique, no need to test if already candidate
                        candidates[nb_candidates].node = neigh_neigh;
                        candidates[nb_candidates].similarity = self.similarity(neighbor, neigh_neigh);
                        nb_candidates += 1;
                    }
                    // since candidates are sorted, we can insert current_node at the right position
                    var inser_pos = nb_candidates;
                    while (inser_pos > 0 and sim > candidates[inser_pos - 1].similarity) {
                        candidates[inser_pos] = candidates[inser_pos - 1];
                        inser_pos -= 1;
                    }
                    candidates[inser_pos] = .{ .node = current_node, .similarity = sim };
                    nb_candidates += 1;

                    // reverse prune
                    self.pruneCandidates(neighbor, candidates[0..nb_candidates]);
                }

                if (i == 0 or (i + 1) % 1000 == 0 or i + 1 == self.n) self.logNsw(start, i);
            }
        }
    }

    pub fn logNsw(self: *Graph, start: std.Io.Timestamp, i: usize) void {
        const now = std.Io.Timestamp.now(self.io, .awake);
        const elapsed_duration = std.Io.Timestamp.durationTo(start, now);
        const elapsed_seconds = @as(f64, @floatFromInt(elapsed_duration.nanoseconds)) / 1e9;
        const eta_seconds = elapsed_seconds * @as(f64, @floatFromInt(self.n - i - 1)) / @as(f64, @floatFromInt(i + 1));
        log.info(
            "NSW node {d}/{d} elapsed={d:.2}s eta={d:.2}s",
            .{ i + 1, self.n, elapsed_seconds, eta_seconds },
        );
    }

    // ----------------------- NSW utils ------------------------ //

    pub fn hasNeighbor(self: *const Graph, node: usize, candidate: usize) bool {
        const start_neigh = self.params.k_max * node;
        const end_neigh = start_neigh + self.nb_neighbors[node];
        for (start_neigh..end_neigh) |i| {
            if (self.neighbors[i] == candidate) return true;
        }
        return false;
    }

    // ---------------------- Syntax utils ----------------------- //

    pub fn nbEdges(self: *const Graph) usize {
        var count: usize = 0;
        for (0..self.n) |i| {
            count += self.nb_neighbors[i];
        }
        return count;
    }

    pub fn similarity(self: *const Graph, a: usize, b: usize) f32 {
        return self.similarity_matrix.dist(a, b);
    }
};
