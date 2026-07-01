const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");
const algebra = @import("algebra.zig");
const save_load = @import("saveload.zig");
const llm = @import("llm.zig");
const tokens = @import("tokens.zig");
const sampling = @import("sampling.zig");

const log = std.log;
const Tokenizer = zml.tokenizer.Tokenizer;
const SimilarityMatrix = algebra.SimilarityMatrix;
const Zml_handler = main.Zml_handler;
const Field_timer = main.Timing_handler.Field_timer;

pub const GraphParams = struct {
    k_max: usize = 32,
    search_budget: usize = 4096,
    alpha: f32 = 1.0,
    vamana_passes: usize = 1,
    top_k: usize = 16,
    L: usize = 256,
    nb_entry_points: usize = 1,
    graph_type: GraphType = .Mips,
};

pub const GraphType = enum {
    Angular,
    Mips,
};

pub const Graph = struct {
    pub const Candidate = struct {
        node: usize,
        similarity: f32,

        fn beforeThan(_: void, lhs: Candidate, rhs: Candidate) bool {
            return lhs.similarity > rhs.similarity or (lhs.similarity == rhs.similarity and lhs.node < rhs.node);
        }
    };

    pub const LazyExpansion = struct {
        node: usize,
        neighbor: usize,
    };

    zml_handler: *Zml_handler,
    allocator: std.mem.Allocator,
    io: std.Io,
    // dataset fields
    dim: usize,
    lm_head: []f32,
    similarity_matrix: *SimilarityMatrix,
    // graph fields
    // TODO: we can use smaller integer types
    n: usize,
    params: GraphParams,
    neighbors: []usize,
    nb_neighbors: []usize,
    is_junk: []bool,
    medoid: usize,
    medoids: []usize,
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
    // for each candidate in visited, tells if its neighbors have been
    // added to the pool (when true, the node had been dealt with)
    is_expanded: []bool,
    // for each candidate in visited, tells how many neighbors have been
    // explored by lazy search
    nb_expanded_neighbors: []usize,
    is_search_done: bool,
    // for each node of the graph, tells if neighbors have been obtained
    // with the pruning heuristic. in this case no neighbor would be removed
    // if we run the pruning again
    are_neighbors_pruned: []bool,
    // for each node, tells if the node was not found by greedy node search
    // during the last call to testNswExtention
    nsw_extension_search_missed: []bool,
    // utils
    sim_access: usize = 0,

    pub fn init(zml_handler: *Zml_handler, lm_head: zml.Slice, matrix: *SimilarityMatrix, junk_rows: []const usize, medoid: usize, params: GraphParams) !Graph {
        std.debug.assert(matrix.n > 0);
        std.debug.assert(params.k_max > 0);
        std.debug.assert(params.L > 0);
        std.debug.assert(params.search_budget > 0);
        std.debug.assert(params.alpha >= 1.0);
        std.debug.assert(params.L <= params.search_budget + params.k_max);
        std.debug.assert(params.search_budget + params.k_max <= matrix.n);
        std.debug.assert(params.k_max < matrix.n);
        std.debug.assert(params.nb_entry_points > 0);
        std.debug.assert((params.nb_entry_points & (params.nb_entry_points - 1)) == 0);

        const allocator = zml_handler.allocator;

        const lm_head_items = lm_head.constItems(f32);
        std.debug.assert(lm_head_items.len == matrix.n * matrix.d);
        const lm_head_data = try allocator.dupe(f32, lm_head_items);
        errdefer allocator.free(lm_head_data);

        const is_junk = try allocator.alloc(bool, matrix.n);
        errdefer allocator.free(is_junk);
        @memset(is_junk, false);
        for (junk_rows) |row| is_junk[row] = true;

        std.debug.assert(medoid < matrix.n);
        std.debug.assert(!is_junk[medoid]);

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

        const nb_expanded_neighbors = try allocator.alloc(usize, params.search_budget + params.k_max);
        errdefer allocator.free(nb_expanded_neighbors);
        @memset(nb_expanded_neighbors, 0);

        const are_neighbors_pruned = try allocator.alloc(bool, matrix.n);
        errdefer allocator.free(are_neighbors_pruned);
        @memset(are_neighbors_pruned, false);

        const nsw_extension_search_missed = try allocator.alloc(bool, matrix.n);
        errdefer allocator.free(nsw_extension_search_missed);
        @memset(nsw_extension_search_missed, false);

        const medoids = try allocator.alloc(usize, params.nb_entry_points);
        errdefer allocator.free(medoids);
        @memset(medoids, matrix.n);

        const visited = try allocator.alloc(Candidate, params.search_budget + params.k_max);
        errdefer allocator.free(visited);

        return .{
            .zml_handler = zml_handler,
            .allocator = allocator,
            .io = zml_handler.io,
            .dim = matrix.d,
            .lm_head = lm_head_data,
            .n = matrix.n,
            .params = params,
            .neighbors = neighbors,
            .nb_neighbors = nb_neighbors,
            .is_junk = is_junk,
            .similarity_matrix = matrix,
            .medoid = medoid,
            .medoids = medoids,
            .capacity = params.search_budget + params.k_max,
            .L = 0,
            .visited = visited,
            .is_visited = is_visited,
            .nb_visited = 0,
            .is_expanded = is_expanded,
            .nb_expanded_neighbors = nb_expanded_neighbors,
            .is_search_done = false,
            .are_neighbors_pruned = are_neighbors_pruned,
            .nsw_extension_search_missed = nsw_extension_search_missed,
        };
    }

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.lm_head);
        self.allocator.free(self.neighbors);
        self.allocator.free(self.nb_neighbors);
        self.allocator.free(self.is_junk);
        self.allocator.free(self.visited);
        self.allocator.free(self.is_visited);
        self.allocator.free(self.is_expanded);
        self.allocator.free(self.nb_expanded_neighbors);
        self.allocator.free(self.are_neighbors_pruned);
        self.allocator.free(self.nsw_extension_search_missed);
        self.allocator.free(self.medoids);
    }

    // ------------------- Search functions ------------------ //

    pub fn greedySearchNode(self: *Graph, query: usize, pass: usize) void {
        self.zml_handler.tic(&self.zml_handler.timers.greedy_search);
        std.debug.assert(!self.is_junk[query]);
        // initialize search at entry point
        self.initNodeSearch(query, pass);

        while (self.nb_visited < self.params.search_budget) {

            // find best node of the active pool that has not been expanded yet
            const node = self.popCandidate();

            // if we found the query node, terminate the search
            if (node == query) break;

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
        self.zml_handler.tic(&self.zml_handler.timers.embed_search);
        // initialize search at entry point
        self.initSearch(query);

        while (self.nb_visited < self.params.search_budget) {

            // find best node of the active pool that has not been expanded yet
            const node = self.popCandidate();

            // if all nodes in active pool have been expanded, terminate the search
            if (self.is_search_done) break;

            const start_neigh = self.params.k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                if (self.is_visited[neighbor]) continue;
                self.addCandidate(query, neighbor);
            }
        }

        self.cleanup();
        self.zml_handler.toc(&self.zml_handler.timers.embed_search);
    }

    pub fn greedySearchNodeLazy(self: *Graph, query: usize) void {
        self.zml_handler.tic(&self.zml_handler.timers.greedy_search);
        std.debug.assert(!self.is_junk[query]);
        self.initNodeSearch(query);

        while (self.nb_visited < self.params.search_budget) {
            if (self.visited[0].node == query) break;

            const expansion = self.popCandidateLazy();
            if (self.is_search_done) break;

            if (expansion.node == query) break;

            self.addNodeCandidate(query, expansion.neighbor);
        }

        self.cleanup();
        self.zml_handler.toc(&self.zml_handler.timers.greedy_search);
    }

    pub fn greedySearchLazy(self: *Graph, query: []const f32) void {
        self.zml_handler.tic(&self.zml_handler.timers.embed_search);
        self.initSearch(query);

        while (self.nb_visited < self.params.search_budget) {
            const expansion = self.popCandidateLazy();
            if (self.is_search_done) break;

            self.addCandidate(query, expansion.neighbor);
        }

        self.cleanup();
        self.zml_handler.toc(&self.zml_handler.timers.embed_search);
    }

    
    pub fn scoreQueryNode(self: *const Graph, query: []const f32, node: usize) f32 {
        self.zml_handler.tic(&self.zml_handler.timers.embed_dot);
        std.debug.assert(!self.is_junk[node]);
        const rows = self.lm_head;
        const row = rows[node * self.dim ..][0..self.dim];

        const simd_len = 32;
        std.debug.assert(self.dim % simd_len == 0);
        const Vec = @Vector(simd_len, f32);
        var acc: Vec = @splat(0);

        var i: usize = 0;
        while (i + simd_len <= self.dim) : (i += simd_len) {
            const query_vec: Vec = query[i..][0..simd_len].*;
            const row_vec: Vec = row[i..][0..simd_len].*;
            acc = @mulAdd(Vec, query_vec, row_vec, acc);
        }
        const dot = @reduce(.Add, acc);
        self.zml_handler.toc(&self.zml_handler.timers.embed_dot);
        return dot;
    }

    
    pub fn initNodeSearch(self: *Graph, query: usize, pass: usize) void {
        // at start, pool is empty
        std.debug.assert(!self.is_visited[self.medoid]);

        const entry_point, const entry_sim = self.selectNodeEntryPoint(query, pass);

        // medoid is the first and only visited node
        self.is_visited[entry_point] = true;
        self.visited[0] = .{ .node = entry_point, .similarity = entry_sim };
        self.nb_expanded_neighbors[0] = 0;
        self.nb_visited = 1;
        self.L = 1;
        self.is_search_done = false;
    }

    pub fn initSearch(self: *Graph, query: []const f32) void {
        // at start, pool is empty
        std.debug.assert(!self.is_visited[self.medoid]);

        const entry_point, const entry_sim = self.selectQueryEntryPoint(query);

        // medoid is the first and only visited node
        self.is_visited[entry_point] = true;
        self.visited[0] = .{ .node = entry_point, .similarity = entry_sim };
        self.nb_expanded_neighbors[0] = 0;
        self.nb_visited = 1;
        self.L = 1;
        self.is_search_done = false;
    }

    
    fn selectNodeEntryPoint(self: *Graph, query: usize, pass: usize) struct { usize, f32 } {
        if (pass == 0) {
            // first pass: mostly randomized
            const entry_id = query % self.params.nb_entry_points;
            const entry_point = self.medoids[entry_id];
            const entry_sim = self.similarity(query, entry_point);
            std.debug.assert(entry_point < self.n);
            std.debug.assert(!self.is_visited[entry_point]);
            return .{ entry_point, entry_sim };
        } else {
            var entry_point = self.medoids[0];
            var entry_sim = self.similarity(query, entry_point);
            std.debug.assert(entry_point < self.n);
            std.debug.assert(!self.is_visited[entry_point]);
            for (self.medoids[1..self.params.nb_entry_points]) |node| {
                std.debug.assert(node < self.n);
                std.debug.assert(!self.is_visited[node]);
                const sim = self.similarity(query, node);
                if (sim > entry_sim) {
                    entry_point = node;
                    entry_sim = sim;
                }
            }
            return .{ entry_point, entry_sim };
        }
    }

    fn selectQueryEntryPoint(self: *Graph, query: []const f32) struct { usize, f32 } {
        var entry_point = self.medoids[0];
        std.debug.assert(entry_point < self.n);
        std.debug.assert(!self.is_visited[entry_point]);
        var entry_sim = self.scoreQueryNode(query, entry_point);

        for (self.medoids[1..self.params.nb_entry_points]) |node| {
            std.debug.assert(node < self.n);
            std.debug.assert(!self.is_visited[node]);
            const sim = self.scoreQueryNode(query, node);
            if (sim > entry_sim) {
                entry_point = node;
                entry_sim = sim;
            }
        }
        return .{ entry_point, entry_sim };
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
        self.zml_handler.tic(&self.zml_handler.timers.insert_node);

        std.debug.assert(!self.is_junk[node]);
        std.debug.assert(!self.is_visited[node]);
        std.debug.assert(self.nb_visited > 0);
        std.debug.assert(self.L > 0);
        // TODO: if we split pool management into visited and active pool, we can simplify
        // TODO: if we split the active pool into expanded and unexpanded, we can improve:
        // popCandidate becomes O(1) as it simply read first position in unexpanded pool

        self.is_visited[node] = true;
        // this is the lowest score of the active pool
        const worse_L_score = self.visited[self.L - 1].similarity;

        if (worse_L_score > sim) {
            // if node has worse score, insert it directly at the end of the pool.
            // this handles both cases where
            // - active pool is full: the new node is in the pool but not in the active pool
            // - active pool is not full: the end the pool is the end of the active pool
            self.visited[self.nb_visited] = .{ .node = node, .similarity = sim };
            self.nb_expanded_neighbors[self.nb_visited] = 0;
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
                self.nb_expanded_neighbors[self.nb_visited] = self.nb_expanded_neighbors[self.L - 1];
                self.L -= 1;
            }
            var i = self.L;
            while (i > 0 and sim > self.visited[i - 1].similarity) {
                self.visited[i] = self.visited[i - 1];
                self.is_expanded[i] = self.is_expanded[i - 1];
                self.nb_expanded_neighbors[i] = self.nb_expanded_neighbors[i - 1];
                i -= 1;
            }
            std.debug.assert(self.L < self.params.L);
            std.debug.assert(i < self.params.L);
            self.visited[i] = .{ .node = node, .similarity = sim };
            self.is_expanded[i] = false;
            self.nb_expanded_neighbors[i] = 0;
            self.L += 1;
        }
        self.nb_visited += 1;
        self.zml_handler.toc(&self.zml_handler.timers.insert_node);
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

    pub fn popCandidateLazy(self: *Graph) LazyExpansion {
        // find the best candidate in the active pool with remaining neighbors
        // since the pool is kept sorted, return the first found
        for (0..self.L) |i| {
            const node = self.visited[i].node;
            while (self.nb_expanded_neighbors[i] < self.nb_neighbors[node]) {
                const neighbor_pos = self.nb_expanded_neighbors[i];
                self.nb_expanded_neighbors[i] += 1;
                const neighbor = self.neighbors[node * self.params.k_max + neighbor_pos];
                if (self.is_visited[neighbor]) continue;
                return .{ .node = node, .neighbor = neighbor };
            }
        }
        self.is_search_done = true;
        return .{ .node = self.visited[0].node, .neighbor = self.visited[0].node };
    }

    pub fn cleanup(self: *Graph) void {
        for (0..self.nb_visited) |i| {
            const node = self.visited[i].node;
            self.is_visited[node] = false;
            self.is_expanded[i] = false;
            self.nb_expanded_neighbors[i] = 0;
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

    pub fn setNearestNeighbors(self: *Graph, k: usize) void {
        log.info("Nearest neighbors", .{});
        for (0..self.n) |i| {
            const start_neigh = self.params.k_max * i;
            self.nb_neighbors[i] = 0;
            if (self.is_junk[i]) continue;
            for (0..self.similarity_matrix.k) |j| {
                const candidate = self.similarity_matrix.nearestNeighbor(i, j);
                std.debug.assert(candidate != i);
                if (self.is_junk[candidate]) continue;
                self.neighbors[start_neigh + self.nb_neighbors[i]] = candidate;
                self.nb_neighbors[i] += 1;
                if (self.nb_neighbors[i] == k) break;
            }
            if (i == 0 or (i + 1) % 10000 == 0 or i + 1 == self.n) {
                log.info("Nearest neighbors node {d}/{d}", .{ i + 1, self.n });
            }
        }
        std.log.info("Exact kNN : nb edges: {d}", .{self.nbEdges()});
    }

    fn sortNodeNeighbors(self: *Graph, node: usize, scratch: []Candidate) void {
        std.debug.assert(scratch.len >= self.nb_neighbors[node]);

        const start_neigh = self.params.k_max * node;
        for (0..self.nb_neighbors[node]) |i| {
            const neighbor = self.neighbors[start_neigh + i];
            scratch[i] = .{ .node = neighbor, .similarity = self.similarity(node, neighbor) };
        }
        std.mem.sort(Candidate, scratch[0..self.nb_neighbors[node]], {}, Candidate.beforeThan);
        for (0..self.nb_neighbors[node]) |i| {
            self.neighbors[start_neigh + i] = scratch[i].node;
        }
    }

    fn insertNeighborSorted(self: *Graph, node: usize, candidate: usize) void {
        std.debug.assert(node != candidate);
        std.debug.assert(self.nb_neighbors[node] < self.params.k_max);

        const start_neigh = self.params.k_max * node;
        const end_neigh = start_neigh + self.nb_neighbors[node];
        const sim = self.similarity(node, candidate);
        var insert_pos = end_neigh;
        while (insert_pos > start_neigh and sim > self.similarity(node, self.neighbors[insert_pos - 1])) {
            self.neighbors[insert_pos] = self.neighbors[insert_pos - 1];
            insert_pos -= 1;
        }
        self.neighbors[insert_pos] = candidate;
        self.nb_neighbors[node] += 1;
    }

    fn isPrunedByCurrentNeighbors(self: *Graph, node: usize, candidate: usize) bool {
        std.debug.assert(node != candidate);

        const base_sim = self.similarity(node, candidate);
        const alpha = if (base_sim > 0.0) self.params.alpha else 1.0 / self.params.alpha;
        const start_neigh = self.params.k_max * node;
        const end_neigh = start_neigh + self.nb_neighbors[node];
        for (start_neigh..end_neigh) |i| {
            const neighbor = self.neighbors[i];
            if (self.similarity(neighbor, candidate) >= alpha * base_sim) return true;
        }
        return false;
    }

    pub fn consolidateNswNearest(self: *Graph) void {
        log.info("Consolidate NSW with nearest neighbors", .{});

        const scratch = self.allocator.alloc(Candidate, self.params.k_max) catch @panic("OOM");
        defer self.allocator.free(scratch);

        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            self.sortNodeNeighbors(node, scratch);

            for (0..self.similarity_matrix.k) |neigh_pos| {
                if (self.nb_neighbors[node] == self.params.k_max) break;
                const candidate = self.similarity_matrix.nearestNeighbor(node, neigh_pos);
                if (candidate == node or self.is_junk[candidate] or self.hasNeighbor(node, candidate)) continue;
                self.insertNeighborSorted(node, candidate);
            }
            std.debug.assert(self.nb_neighbors[node] == self.params.k_max);

            if (node == 0 or (node + 1) % 10000 == 0 or node + 1 == self.n) {
                log.info("Consolidate nearest node {d}/{d}", .{ node + 1, self.n });
            }
        }
        std.log.info("Consolidated nearest NSW: nb edges: {d}", .{self.nbEdges()});
    }

    pub fn consolidateNswPrune(self: *Graph) void {
        log.info("Consolidate NSW with pruned nearest neighbors", .{});

        const scratch = self.allocator.alloc(Candidate, self.params.k_max) catch @panic("OOM");
        defer self.allocator.free(scratch);

        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            self.sortNodeNeighbors(node, scratch);

            for (0..self.similarity_matrix.k) |neigh_pos| {
                if (self.nb_neighbors[node] == self.params.k_max) break;
                const candidate = self.similarity_matrix.nearestNeighbor(node, neigh_pos);
                if (candidate == node or self.is_junk[candidate] or self.hasNeighbor(node, candidate)) continue;
                if (self.isPrunedByCurrentNeighbors(node, candidate)) continue;
                self.insertNeighborSorted(node, candidate);
            }

            if (node == 0 or (node + 1) % 10000 == 0 or node + 1 == self.n) {
                log.info("Consolidate pruned node {d}/{d}", .{ node + 1, self.n });
            }
        }
        std.log.info("Consolidated pruned NSW: nb edges: {d}", .{self.nbEdges()});
    }
    
    pub fn pruneNeighbors(self: *Graph, alpha: f32) !void {
        log.info("Pruning neighbors with alpha={d}", .{alpha});
        std.log.info("Initial edges: {d}", .{self.nbEdges()});
        const alpha_checkpoint = self.params.alpha;
        self.params.alpha = alpha;
        const candidates = self.allocator.alloc(Candidate, self.params.k_max) catch @panic("OOM");
        defer self.allocator.free(candidates);
        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            const start_neigh = node * self.params.k_max;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                candidates[i - start_neigh] = .{ .node = neighbor, .similarity = self.similarity(node, neighbor) };
            }
            self.pruneCandidates(node, candidates, &self.zml_handler.timers.prune_pool_fwd);
        }
        std.log.info("Final edges: {d}", .{self.nbEdges()});
        self.params.alpha = alpha_checkpoint;
        try self.logDegreeCounts("Nodes by out-degree", self.nb_neighbors, self.params.k_max);
    }

    
    pub fn pruneCandidates(self: *Graph, base: usize, candidates: []Candidate, timer: *Field_timer) void {
        if (self.params.graph_type == .Mips) {
            self.pruneMips(base, candidates, timer);
        } else {
            self.pruneAngular(base, candidates, timer);
        }
    }
    
    pub fn pruneAngular(self: *Graph, base: usize, candidates: []Candidate, timer: *Field_timer) void {
        std.debug.assert(!self.is_junk[base]);
        self.zml_handler.tic(timer);
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
        self.zml_handler.toc(timer);
    }

    pub fn pruneMips(self: *Graph, base: usize, candidates: []Candidate, timer: *Field_timer) void {
        std.debug.assert(!self.is_junk[base]);
        self.zml_handler.tic(timer);
        // update current_node neighbors with LOS pruning of the candidates
        const start_neigh = base * self.params.k_max;
        self.nb_neighbors[base] = 0;
        var end_neigh = start_neigh;
        for (candidates) |candidate| {
            std.debug.assert(candidate.node != base);
            var pruned = false;
            const base_sim = self.similarity(base, candidate.node);
            // if base_sim is negative, we need to invert alpha so the rule is still a relaxation
            const alpha = if (base_sim > 0.0) self.params.alpha else 1.0 / self.params.alpha;
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                const neigh_sim = self.similarity(neighbor, candidate.node);
                if (neigh_sim >= alpha * base_sim) {
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
        self.zml_handler.toc(timer);
    }

    // ------------------- NSW main function -------------------- //

    pub fn extendToNsw(self: *Graph) !void {
        try self.benchSimilarity();
        var prng = std.Random.DefaultPrng.init(1);
        const random = prng.random();

        const order = self.allocator.alloc(usize, self.n) catch @panic("OOM");
        defer self.allocator.free(order);
        for (0..order.len) |i| order[i] = i;

        const candidates = self.allocator.alloc(Candidate, 2 * self.params.k_max + self.params.search_budget) catch @panic("OOM");
        defer self.allocator.free(candidates);

        self.sim_access = 0;
        self.zml_handler.nb_tictoc = 0;

        self.setEntryPoints();

        for (0..self.params.vamana_passes) |pass_i| {
            // when pass_i > 0, we increase alpha from 1.0 to the params.alpha value
            // this means the flags are_neighbors_pruned is invalidated
            @memset(self.are_neighbors_pruned, false);
            // random visit order
            var nb_swap: usize = order.len - 1;
            while (nb_swap > 0) : (nb_swap -= 1) {
                const j = random.uintLessThan(usize, nb_swap + 1);
                std.mem.swap(usize, &order[nb_swap], &order[j]);
            }
            log.info("NSW pass {d}/{d}, alpha: {d}", .{ pass_i + 1, self.params.vamana_passes, self.params.alpha });
            const start = std.Io.Timestamp.now(self.io, .awake);
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
                self.greedySearchNode(current_node, pass_i);
                if (self.visited[0].node == current_node) {
                    // if current_node was found, we could decide the connectedness is ok
                    // and continue. this reduces the pressure on nb_neighbors for each node,
                    // and leads to huge improvements in node search (eg: 25% -> 70% success),
                    // but for now it degrades query search (eg: 84% -> 76% success).
                    //continue;
                }
                //const nb_search_cand: usize = self.nb_visited;
                const nb_search_cand: usize = self.L;
                // only first L positions are sorted
                if (nb_search_cand > self.L) std.mem.sort(Candidate, self.visited[self.L..nb_search_cand], {}, Candidate.beforeThan);
                var pos_in_neighbors: usize = start_neigh;
                var pos_in_visited: usize = 0;

                while (pos_in_neighbors < end_neigh and pos_in_visited < nb_search_cand) {
                    // if current_node was visited during the search, skip it in the visited pool
                    // otherwise it will end up being a neighbor of itself
                    // note that on the not metric case, if current_node was visited,
                    // it might not be the best candidate (the first in self.visited)
                    if (self.visited[pos_in_visited].node == current_node) {
                        pos_in_visited += 1;
                        continue;
                    }
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
                for (pos_in_visited..nb_search_cand) |j| {
                    if (self.visited[j].node == current_node) continue;
                    const visit = self.visited[j].node;
                    candidates[nb_candidates] = .{ .node = visit, .similarity = self.visited[j].similarity };
                    nb_candidates += 1;
                }

                // forward prune on candidates
                self.pruneCandidates(current_node, candidates[0..nb_candidates], &self.zml_handler.timers.prune_pool_fwd);

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
                    // TODO: this heuristic is not exact, as commenting it changes the behavior. investigate.
                    //const worse_neighbor_sim = self.similarity(neighbor, self.neighbors[end_neigh_neigh - 1]);
                    //if (self.are_neighbors_pruned[neighbor] and worse_neighbor_sim > sim) continue;

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
                    self.pruneCandidates(neighbor, candidates[0..nb_candidates], &self.zml_handler.timers.prune_pool_bwd);
                }

                if (i == 0 or (i + 1) % 1000 == 0 or i + 1 == self.n) self.logNsw(start, i);
            }
            std.log.info("NSW extension pass {d} done, nb edges: {d}", .{ pass_i + 1, self.nbEdges() });
            self.params.alpha += 0.25;
        }
        std.log.info("sim_access: {}", .{self.sim_access});
        std.log.info("nb tic toc: {}", .{self.zml_handler.nb_tictoc});
    }

    pub fn fixNswExtention(self: *Graph, ) !void {
        log.info("Fix NSW extension", .{});
        const in_degrees = try self.allocator.alloc(usize, self.n);
        defer self.allocator.free(in_degrees);
        @memset(in_degrees, 0);

        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            const start_neigh = self.params.k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |neigh| {
                in_degrees[self.neighbors[neigh]] += 1;
            }
        }

        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            if (in_degrees[node] < 8) {
                for (0..self.similarity_matrix.k) |k| {
                    const neigh = self.similarity_matrix.nearestNeighbor(node, k);
                    if (self.nb_neighbors[neigh] == self.params.k_max) continue;
                    if (self.is_junk[neigh]) continue;
                    self.neighbors[neigh * self.params.k_max + self.nb_neighbors[neigh]] = node;
                    self.nb_neighbors[neigh] += 1;
                    in_degrees[node] += 1;
                    if (in_degrees[node] == 8) break;
                }
            }
        }
    }
    
    fn increasingF32(_: void, lhs: f32, rhs: f32) bool {
        return lhs < rhs;
    }

    pub fn setEntryPoints(self: *Graph) void {
        self.medoids[0] = 0;
        if (false) {
            const k = self.nbEntryHyperplanes();
            const rows = self.lm_head;
            const row_norms = self.lm_head_row_norms.constItems(f32);
    
            const values = self.allocator.alloc(f32, self.n) catch @panic("OOM");
            defer self.allocator.free(values);
    
            for (0..k) |coord| {
                for (0..self.n) |node| {
                    values[node] = rows[node * self.dim + coord];
                }
                std.mem.sort(f32, values, {}, increasingF32);
                if (self.n % 2 == 0) {
                    const hi = self.n / 2;
                    self.entry_medians[coord] = 0.5 * (values[hi - 1] + values[hi]);
                } else {
                    self.entry_medians[coord] = values[self.n / 2];
                }
                log.info("entry hyperplane coord {d}: median={d:.8}", .{ coord, self.entry_medians[coord] });
            }
    
            @memset(self.medoids, self.medoid);
            const best_norms = self.allocator.alloc(f32, self.params.nb_entry_points) catch @panic("OOM");
            defer self.allocator.free(best_norms);
            const bucket_counts = self.allocator.alloc(usize, self.params.nb_entry_points) catch @panic("OOM");
            defer self.allocator.free(bucket_counts);
            @memset(bucket_counts, 0);
            for (best_norms) |*best_norm| best_norm.* = -std.math.inf(f32);
    
            for (0..self.n) |node| {
                if (self.is_junk[node]) continue;
                const row = rows[node * self.dim ..][0..self.dim];
                const bucket = self.bucketForRow(row, k);
                bucket_counts[bucket] += 1;
                if (row_norms[node] > best_norms[bucket]) {
                    best_norms[bucket] = row_norms[node];
                    self.medoids[bucket] = node;
                }
            }
    
            var empty_buckets: usize = 0;
            for (self.medoids[0..self.params.nb_entry_points], 0..) |entry_point, bucket| {
                std.debug.assert(entry_point < self.n);
                std.debug.assert(!self.is_junk[entry_point]);
                if (bucket_counts[bucket] == 0) empty_buckets += 1;
            }
            log.info("Entry points from {d} median hyperplanes: buckets={d} empty={d}", .{ k, self.params.nb_entry_points, empty_buckets });
        }
    }

    pub fn testNswExtention(self: *Graph, sampler: *sampling.Sampler) !void {
        self.setEntryPoints();
        log.info("Test NSW extension", .{});
        @memset(self.nsw_extension_search_missed, false);

        const in_degrees = try self.allocator.alloc(usize, self.n);
        defer self.allocator.free(in_degrees);
        @memset(in_degrees, 0);

        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            const start_neigh = self.params.k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |neigh| {
                in_degrees[self.neighbors[neigh]] += 1;
            }
        }
        var min_in_degree = self.n;
        var max_in_degree: usize = 0;
        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            min_in_degree = @min(min_in_degree, in_degrees[node]);
            max_in_degree = @max(max_in_degree, in_degrees[node]);
        }
        log.info("Min in-degree: {}", .{min_in_degree});
        log.info("Max in-degree: {}", .{max_in_degree});

        //try self.logDegreeCounts("Nodes by in-degree", in_degrees, max_in_degree);
        for (0..self.n) |node| {
            if (self.is_junk[node] or in_degrees[node] <= 2000) continue;
            const token_str = try tokens.tokenString(sampler.tokenizer, @as(u32, @intCast(node)), self.allocator);
            log.info("Node {d} has in-degree {d}, token: {s}", .{ node, in_degrees[node], token_str });
            self.allocator.free(token_str);
        }
        //try self.logDegreeCounts("Nodes by out-degree", self.nb_neighbors, self.params.k_max);

        const hop_dist = try self.getHopDistance();
        defer self.allocator.free(hop_dist);
        var max_hops: usize = 0;
        for (hop_dist) |hops| {
            if (hops > max_hops) max_hops = hops;
        }
        log.info("Max hops: {}", .{max_hops});
        for (0..2) |hops| {
            for (0..self.n) |i| {
                if (hop_dist[i] == hops) {
                    std.log.info("Node {d} needs {d} hops, out-degree: {d}, tok: {s}", .{ i, hops, self.nb_neighbors[i], try tokens.tokenString(sampler.tokenizer, i, self.allocator) });
                }
            }
        }
        for (8..8) |hops| {
            for (0..self.n) |i| {
                if (hop_dist[i] == hops) {
                    std.log.info("Node {d} needs {d} hops, out-degree: {d}, tok: {s}", .{ i, hops, self.nb_neighbors[i], try tokens.tokenString(sampler.tokenizer, i, self.allocator) });
                }
            }
        }
        try self.logDegreeCounts("Nodes by min-hops", hop_dist, max_hops);

        var exact_first_count: usize = 0;
        var valid_count: usize = 0;
        var total_visited: usize = 0;
        var min_visited: usize = std.math.maxInt(usize);
        var max_visited: usize = 0;
        const entry_point_counts = try self.allocator.alloc(usize, self.params.nb_entry_points);
        defer self.allocator.free(entry_point_counts);
        @memset(entry_point_counts, 0);

        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            valid_count += 1;
            const entry_point, _ = self.selectNodeEntryPoint(node, 2);
            for (self.medoids[0..self.params.nb_entry_points], 0..) |medoid, entry_i| {
                if (medoid == entry_point) {
                    entry_point_counts[entry_i] += 1;
                    break;
                }
            }
            self.greedySearchNode(node, 2);
            const nb_visited = self.nb_visited;
            total_visited += nb_visited;
            min_visited = @min(min_visited, nb_visited);
            max_visited = @max(max_visited, nb_visited);
            var found = false;
            for (0..self.nb_visited) |i| {
                if (self.visited[i].node == node) {
                    exact_first_count += 1;
                    found = true;
                    break;
                }
            }
            if (!found) {
                self.nsw_extension_search_missed[node] = true;
                //std.log.info("Token {d}, in-degree {d}, out-degree {d}, hop distance {d}, not found {s}", .{ node, in_degrees[node], self.nb_neighbors[node], hop_dist[node], try tokens.tokenString(sampler.tokenizer, node, self.allocator) });
            }

            if (valid_count == 1 or valid_count % 10000 == 0) {
                const exact_rate = @as(f64, @floatFromInt(exact_first_count)) / @as(f64, @floatFromInt(valid_count));
                log.info("NSW extension test node {d}/{d}, success rate {d:.4}%", .{ node + 1, self.n, 100.0 * exact_rate });
            }
        }

        const avg_visited = @as(f64, @floatFromInt(total_visited)) / @as(f64, @floatFromInt(valid_count));
        const exact_rate = @as(f64, @floatFromInt(exact_first_count)) / @as(f64, @floatFromInt(valid_count));
        log.info("NSW extension entry-point starts", .{});
        for (entry_point_counts, 0..) |count, entry_i| {
            if (count == 0) continue;
            log.info("entry {d} node {d}: {d} searches", .{ entry_i, self.medoids[entry_i], count });
        }
        log.info(
            "NSW extension test: valid={d} exact_first={d}/{d} ({d:.4}%) nb_visited min={d} max={d} avg={d:.2}",
            .{
                valid_count,
                exact_first_count,
                valid_count,
                100.0 * exact_rate,
                if (valid_count == 0) 0 else min_visited,
                max_visited,
                avg_visited,
            },
        );
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

    fn logDegreeCounts(self: *Graph, title: []const u8, degrees: []const usize, max_degree: usize) !void {
        const counts = try self.allocator.alloc(usize, max_degree + 1);
        defer self.allocator.free(counts);
        @memset(counts, 0);

        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            counts[degrees[node]] += 1;
        }

        log.info("{s}", .{title});
        for (counts, 0..) |count, degree| {
            if (count == 0) continue;
            log.info("{d} nodes of degree {d}", .{ count, degree });
        }
    }

    pub fn hasNeighbor(self: *const Graph, node: usize, candidate: usize) bool {
        const start_neigh = self.params.k_max * node;
        const end_neigh = start_neigh + self.nb_neighbors[node];
        for (start_neigh..end_neigh) |i| {
            if (self.neighbors[i] == candidate) return true;
        }
        return false;
    }

    // ---------------------- MRT functions ---------------------- //

    pub fn makeMrt(self: *Graph, order: []const usize) !void {
        std.debug.assert(order.len == self.n);
        std.debug.assert(!self.is_junk[0]);
        for (1..self.n) |node| {
            if (self.is_junk[node]) continue;
            const p = self.monotonicSearch(0, node);
            if (self.nb_neighbors[p] == self.params.k_max) {
                log.info("Reached max neighbors for node {d} after inserting {d} edges", .{ p, node });
                continue;
            }
            self.neighbors[self.params.k_max * p + self.nb_neighbors[p]] = node;
            self.nb_neighbors[p] += 1;
            if (node % 10000 == 0) log.info("Inserted node {d}", .{node});
        }

        const parents = try self.allocator.alloc(usize, self.n);
        @memset(parents, self.n);
        defer self.allocator.free(parents);
        const ranks = try self.allocator.alloc(usize, self.n);
        @memset(ranks, 0);
        defer self.allocator.free(ranks);
        const costs = try self.allocator.alloc(usize, self.n);
        @memset(costs, 0);
        defer self.allocator.free(costs);
        const nb_descendants = try self.allocator.alloc(usize, self.n);
        @memset(nb_descendants, 0);
        defer self.allocator.free(nb_descendants);

        const nodes_by_rank = try self.allocator.alloc(usize, self.n);
        @memset(nodes_by_rank, 0);
        defer self.allocator.free(nodes_by_rank);
        const nodes_by_cost = try self.allocator.alloc(usize, self.n);
        @memset(nodes_by_cost, 0);
        defer self.allocator.free(nodes_by_cost);
        const nodes_by_degree = try self.allocator.alloc(usize, self.n);
        @memset(nodes_by_degree, 0);
        defer self.allocator.free(nodes_by_degree);

        var max_degree: usize = 0;
        var max_rank: usize = 0;
        var total_rank: usize = 0;
        var max_cost: usize = 0;
        var total_cost: usize = 0;
        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            max_degree = @max(max_degree, self.nb_neighbors[node]);
            max_rank = @max(max_rank, ranks[node]);
            total_rank += ranks[node];
            max_cost = @max(max_cost, costs[node]);
            total_cost += costs[node];
            nodes_by_rank[ranks[node]] += 1;
            nodes_by_cost[costs[node]] += 1;
            nodes_by_degree[self.nb_neighbors[node]] += 1;
            const start_neigh = self.params.k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |neigh_slot| {
                const neigh_node = self.neighbors[neigh_slot];
                std.debug.assert(ranks[neigh_node] == 0);
                ranks[neigh_node] = ranks[node] + 1;
                costs[neigh_node] = costs[node] + self.nb_neighbors[node];
                parents[neigh_node] = node;
            }
        }
        for (0..self.n) |rev| {
            const node = self.n - 1 - rev;
            if (self.is_junk[node] or node == 0) continue;
            nb_descendants[parents[node]] += 1 + nb_descendants[node];
        }
        const avg_rank = @as(f64, @floatFromInt(total_rank)) / @as(f64, @floatFromInt(self.n));
        const avg_cost = @as(f64, @floatFromInt(total_cost)) / @as(f64, @floatFromInt(self.n));
        log.info("Max degree: {d}", .{max_degree});
        log.info("Max rank: {d}, avg rank: {d:.2}", .{ max_rank, avg_rank });
        log.info("Max cost: {d}, avg cost: {d:.2}", .{ max_cost, avg_cost });

        log.info("", .{});
        log.info("Nodes by degree: {d}", .{max_degree});
        for (0..self.n) |deg| {
            if (nodes_by_degree[deg] > 0) {
                log.info("{d} nodes of degree {d}", .{ nodes_by_degree[deg], deg });
            }
        }
        log.info("", .{});
        log.info("Nodes by rank: {d}", .{max_rank});
        for (0..self.n) |rank| {
            if (nodes_by_rank[rank] > 0) {
                log.info("{d} nodes of rank {d}", .{ nodes_by_rank[rank], rank });
            }
        }
        log.info("", .{});
        log.info("Nodes by cost: {d}", .{max_cost});
        for (0..self.n) |cost| {
            if (nodes_by_cost[cost] > 0) {
                log.info("{d} nodes of cost {d}", .{ nodes_by_cost[cost], cost });
            }
        }
        log.info("", .{});
        log.info("Top of the tree", .{});
        for (0..3) |rank| {
            for (0..self.n) |node| {
                if (self.is_junk[node]) continue;
                if (ranks[node] == rank) {
                    std.log.info("Node {d} has rank {d} degree {d} and nb_descendants {d}", .{ node, rank, self.nb_neighbors[node], nb_descendants[node] });
                }
            }
        }
        log.info("", .{});
        log.info("Biggest hub", .{});
        for (0..self.n) |node| {
            if (self.is_junk[node]) continue;
            if (self.nb_neighbors[node] == max_degree) {
                std.log.info("Node {d} has rank {d} degree {d} and nb_descendants {d}", .{ node, ranks[node], self.nb_neighbors[node], nb_descendants[node] });
                var parent = node;
                while (true) {
                    parent = parents[parent];
                    std.log.info("Parent {d} has rank {d} degree {d} and nb_descendants {d}", .{ parent, ranks[parent], self.nb_neighbors[parent], nb_descendants[parent] });
                    if (parent == 0) break;
                }
                break;
            }
        }
    }

    pub fn monotonicSearch(self: *Graph, start: usize, target: usize) usize {
        std.debug.assert(!self.is_junk[start]);
        std.debug.assert(!self.is_junk[target]);
        var best_sim = self.similarity(start, target);
        var best_node = start;
        while (true) {
            if (self.nb_neighbors[best_node] == 0) break;
            const start_neigh = self.params.k_max * best_node;
            const end_neigh = start_neigh + self.nb_neighbors[best_node];
            var best_neigh_node = self.neighbors[start_neigh];
            var best_neigh_sim = self.similarity(best_neigh_node, target);
            for (start_neigh + 1..end_neigh) |neigh_slot| {
                const neigh_node = self.neighbors[neigh_slot];
                const sim = self.similarity(neigh_node, target);
                if (sim > best_neigh_sim) {
                    best_neigh_sim = sim;
                    best_neigh_node = neigh_node;
                }
            }
            if (best_sim > best_neigh_sim) break;
            best_sim = best_neigh_sim;
            best_node = best_neigh_node;
        }
        return best_node;
    }

    pub fn monotonicTrailSearch(self: *Graph, start: usize, target: usize) usize {
        std.debug.assert(!self.is_junk[start]);
        std.debug.assert(!self.is_junk[target]);
        var best_sim = self.similarity(start, target);
        var best_node = start;
        self.visited[0] = .{ .node = best_node, .similarity = best_sim };
        self.nb_visited = 1;
        while (true) {
            if (self.nb_neighbors[best_node] == 0) break;
            const start_neigh = self.params.k_max * best_node;
            const end_neigh = start_neigh + self.nb_neighbors[best_node];
            var best_neigh_node = self.neighbors[start_neigh];
            var best_neigh_sim = self.similarity(best_neigh_node, target);
            for (start_neigh + 1..end_neigh) |neigh_slot| {
                const neigh_node = self.neighbors[neigh_slot];
                const sim = self.similarity(neigh_node, target);
                if (sim > best_neigh_sim) {
                    best_neigh_sim = sim;
                    best_neigh_node = neigh_node;
                }
            }
            if (best_sim > best_neigh_sim) break;
            best_sim = best_neigh_sim;
            best_node = best_neigh_node;
            self.visited[self.nb_visited] = .{ .node = best_node, .similarity = best_sim };
            self.nb_visited += 1;
        }
        var best_ancestor: usize = 0;
        var min_neighbors = self.nb_neighbors[0];
        for (1..self.nb_visited) |i| {
            const node = self.visited[i].node;
            if (self.nb_neighbors[node] <= min_neighbors) {
                best_ancestor = i;
                min_neighbors = self.nb_neighbors[node];
            }
        }
        return best_ancestor;
    }

    // ------------------- Hierarchy functions ------------------- //

    pub fn getHopDistance(self: *Graph) ![]usize {
        const hop_dist = try self.allocator.alloc(usize, self.n);
        errdefer self.allocator.free(hop_dist);
        @memset(hop_dist, self.n);

        var queue: std.ArrayList(usize) = try .initCapacity(self.allocator, 0);
        defer queue.deinit(self.allocator);

        for (self.medoids[0..self.params.nb_entry_points]) |entry_point| {
            std.debug.assert(entry_point < self.n);
            if (hop_dist[entry_point] == 0) continue;
            hop_dist[entry_point] = 0;
            try queue.append(self.allocator, entry_point);
        }

        var queue_head: usize = 0;
        while (queue_head < queue.items.len) {
            const node = queue.items[queue_head];
            queue_head += 1;
            const start_neigh = node * self.params.k_max;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |pos| {
                const neighbor = self.neighbors[pos];
                if (hop_dist[neighbor] == self.n) {
                    hop_dist[neighbor] = hop_dist[node] + 1;
                    try queue.append(self.allocator, neighbor);
                }
            }
        }
        return hop_dist;
    }

    // ---------------------- Syntax utils ----------------------- //

    pub fn nbEdges(self: *const Graph) usize {
        var count: usize = 0;
        for (0..self.n) |i| {
            count += self.nb_neighbors[i];
        }
        return count;
    }

    pub inline fn similarity(self: *Graph, a: usize, b: usize) f32 {
        self.sim_access += 1;
        return self.similarity_matrix.dist(a, b);
    }

    pub fn benchSimilarity(self: *Graph) !void {
        const i = try self.allocator.alloc(usize, 10_000);
        const j = try self.allocator.alloc(usize, 10_000);
        defer self.allocator.free(i);
        defer self.allocator.free(j);
        const A_i: usize = 1_000_007;
        const B_i: usize = 123_789;
        const A_j: usize = 645_007;
        const B_j: usize = 456_123;
        i[0] = 0;
        j[0] = 0;
        for (1..10_000) |k| {
            i[k] = (A_i * i[k - 1] + B_i) % self.n;
            j[k] = (A_j * j[k - 1] + B_j) % self.n;
        }
        const start = std.Io.Timestamp.now(self.io, .awake);
        var sim: f32 = 0.0;
        for (0..10_000) |k| {
            sim += self.similarity(i[k], j[k]);
        }
        const end = std.Io.Timestamp.now(self.io, .awake);
        const duration = end.nanoseconds - start.nanoseconds;
        std.log.info("sim: {d:.3} in {d} ns", .{ sim, duration });
        std.log.info("Time per access: {d} us", .{@as(f32, @floatFromInt(duration)) / @as(f32, @floatFromInt(10_000 * 1_000))});
    }
};
