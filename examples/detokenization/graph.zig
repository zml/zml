const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const algebra = @import("algebra.zig");
const save_load = @import("saveload.zig");
const llm = @import("llm.zig");
const tokens = @import("tokens.zig");
const sampling = @import("sampling.zig");

const Tokenizer = zml.tokenizer.Tokenizer;
const SimilarityMatrix = algebra.SimilarityMatrix;
const LmHeadMatrix = algebra.LmHeadMatrix;
const Zml_handler = main.Zml_handler;
const Field_timer = main.Timing_handler.Field_timer;

pub const graph_k_max = 32;
pub const graph_L = 256;

pub const GraphParams = struct {
    search_budget: u32 = 2048,
    vamana_passes: u32 = 2,
    top_k: u32 = 16,
    graph_type: GraphType = .Mips,
};

pub const GraphType = enum {
    Angular,
    Mips,
};

pub const Graph = struct {
    pub const Candidate = struct {
        node: u32,
        similarity: f32,

        fn beforeThan(_: void, lhs: Candidate, rhs: Candidate) bool {
            return lhs.similarity > rhs.similarity or (lhs.similarity == rhs.similarity and lhs.node < rhs.node);
        }
    };

    pub const LazyExpansion = struct {
        node: u32,
        neighbor: u32,
    };

    zml_handler: *Zml_handler,
    allocator: std.mem.Allocator,
    io: std.Io,
    // dataset fields
    dim: u32,
    similarity_matrix: *SimilarityMatrix,
    lm_head: *LmHeadMatrix,
    // graph fields
    n: u32,
    params: GraphParams,
    neighbors: []u32,
    nb_neighbors: []u32,
    medoid: u32,
    // the active number of candidates, the pool is kept sorted in the
    // range 0..L : this is the active pool
    L: u32,
    // the pool is composed of at most L best nodes visited so far
    visited: []Candidate,
    nb_scored: u32,
    // generation based flags to avoid cleanup
    visited_generation: []u32,
    visited_at: []u32,
    generation: u32,
    // we keep track of the trail of expanded nodes
    expanded: []Candidate,
    nb_expanded: u32,
    // during one iteration of greedy search, the batch of visited neighbors
    batch: []Candidate,
    // for each candidate in visited, tells if its neighbors have been
    // added to the pool (when true, the node had been dealt with)
    is_expanded: []bool,
    is_search_done: bool,
    // for each node, tells if the node was not found by greedy node search
    // during the last call to testNswExtention
    nsw_extension_search_missed: []bool,
    
    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, matrix: *SimilarityMatrix, params: GraphParams) !Graph {
        std.debug.assert(matrix.n > 0);
        std.debug.assert(graph_k_max > 0);
        std.debug.assert(graph_L > 0);
        std.debug.assert(params.search_budget > 0);
        std.debug.assert(graph_L <= params.search_budget + graph_k_max);
        std.debug.assert(params.search_budget + graph_k_max <= matrix.n);
        std.debug.assert(graph_k_max < matrix.n);

        const allocator = zml_handler.allocator;

        const neighbors = try allocator.alloc(u32, matrix.n * graph_k_max);
        errdefer allocator.free(neighbors);

        const nb_neighbors = try allocator.alloc(u32, matrix.n);
        errdefer allocator.free(nb_neighbors);
        @memset(nb_neighbors, 0);

        const is_expanded = try allocator.alloc(bool, params.search_budget + graph_k_max);
        errdefer allocator.free(is_expanded);
        @memset(is_expanded, false);

        const nsw_extension_search_missed = try allocator.alloc(bool, matrix.n);
        errdefer allocator.free(nsw_extension_search_missed);
        @memset(nsw_extension_search_missed, false);

        const visited = try allocator.alloc(Candidate, graph_L + graph_k_max);
        errdefer allocator.free(visited);

        const expanded = try allocator.alloc(Candidate, params.search_budget);
        errdefer allocator.free(expanded);

        const batch = try allocator.alloc(Candidate, graph_k_max);
        errdefer allocator.free(batch);

        const visited_generation = try allocator.alloc(u32, matrix.n);
        errdefer allocator.free(visited_generation);
        @memset(visited_generation, 0);

        const visited_at = try allocator.alloc(u32, matrix.n);
        errdefer allocator.free(visited_at);
        @memset(visited_at, 0);

        return .{
            .zml_handler = zml_handler,
            .allocator = allocator,
            .io = zml_handler.io,
            .dim = @intCast(lm_head.d),
            .lm_head = lm_head,
            .n = @intCast(matrix.n),
            .params = params,
            .neighbors = neighbors,
            .nb_neighbors = nb_neighbors,
            .similarity_matrix = matrix,
            .medoid = 0,
            .L = 0,
            .visited = visited,
            .nb_scored = 0,
            .visited_generation = visited_generation,
            .visited_at = visited_at,
            .generation = 0,
            .expanded = expanded,
            .nb_expanded = 0,
            .batch = batch,
            .is_expanded = is_expanded,
            .is_search_done = false,
            .nsw_extension_search_missed = nsw_extension_search_missed,
        };
    }

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.neighbors);
        self.allocator.free(self.nb_neighbors);
        self.allocator.free(self.visited);
        self.allocator.free(self.visited_generation);
        self.allocator.free(self.visited_at);
        self.allocator.free(self.expanded);
        self.allocator.free(self.batch);
        self.allocator.free(self.is_expanded);
        self.allocator.free(self.nsw_extension_search_missed);
        
    }

    // ------------------- Search functions ------------------ //

    pub fn greedySearchNode(self: *Graph, query: u32) void {
        //self.zml_handler.tic(&self.zml_handler.timers.greedy_search);
        std.debug.assert(!self.lm_head.is_junk[query]);
        // initialize search at entry point
        self.initNodeSearch(query);

        self.nb_expanded = 0;
        
        var nb_scored = self.nb_scored;
        while (nb_scored < self.params.search_budget) {

            // find best node of the active pool that has not been expanded yet
            const node = self.popCandidate();

            // if all nodes in active pool have been expanded, terminate the search
            if (self.is_search_done) break;
            
            const start_neigh = graph_k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            var nb_batch: u32 = 0;
            var i: u32 = start_neigh;
            while (i < end_neigh) : (i += 1) {
                const neighbor = self.neighbors[i];
                if (self.visited_generation[neighbor] == self.generation) continue;
                std.debug.assert(!self.lm_head.is_junk[neighbor]);
                const sim = self.similarity(neighbor, query);
                nb_scored += 1;
                self.visited_generation[neighbor] = self.generation;
                self.visited_at[neighbor] = nb_scored;

                if (self.L == graph_L and self.visited[self.L-1].similarity >= sim) continue;
                // reverse linear pass to insert neighbor in the batch
                // the batch is kept sorted so that it can be inserted
                // efficiently in the pool of visited nodes
                var pos = nb_batch;
                while (pos > 0 and sim > self.batch[pos - 1].similarity) {
                    self.batch[pos] = self.batch[pos - 1];
                    pos -= 1;
                }
                self.batch[pos] = .{ .node = neighbor, .similarity = sim };
                nb_batch += 1;
            }
            self.insertBatch(nb_batch);
        }

        self.nb_scored = nb_scored;
        //self.zml_handler.toc(&self.zml_handler.timers.greedy_search);
    }

    pub fn greedySearch(self: *Graph, query: []const f32) void {
        self.zml_handler.tic(&self.zml_handler.timers.embed_search);
        self.nb_expanded = 0;
        self.initSearch(query);
        while (self.nb_scored < self.params.search_budget) {
            const node = self.popCandidate();
            if (self.is_search_done) break;
            const start_neigh = graph_k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            var nb_batch: u32 = 0;
            var i: u32 = start_neigh;
            while (i < end_neigh) : (i += 1) {
                const neighbor = self.neighbors[i];
                if (self.visited_generation[neighbor] == self.generation) continue;
                std.debug.assert(!self.lm_head.is_junk[neighbor]);
                const sim = self.scoreQueryNode(query, neighbor);
                self.nb_scored += 1;
                self.visited_generation[neighbor] = self.generation;

                if (self.L == graph_L and self.visited[self.L-1].similarity >= sim) continue;
                // reverse linear pass to insert neighbor in the batch
                // the batch is kept sorted so that it can be inserted
                // efficiently in the pool of visited nodes
                var pos = nb_batch;
                while (pos > 0 and sim > self.batch[pos - 1].similarity) {
                    self.batch[pos] = self.batch[pos - 1];
                    pos -= 1;
                }
                self.batch[pos] = .{ .node = neighbor, .similarity = sim };
                nb_batch += 1;
            }
            self.insertBatch(nb_batch);
        }
        self.zml_handler.toc(&self.zml_handler.timers.embed_search);
    }

    pub fn greedySearchPrefetch(self: *Graph, query: []const f32) void {
        self.zml_handler.tic(&self.zml_handler.timers.embed_search);
        self.nb_expanded = 0;
        self.initSearch(query);

        const rows = self.lm_head.data;
        const row_norms = self.lm_head.row_norms;
        const dim: usize = @intCast(self.dim);
        const graph_type = self.params.graph_type;
        const prefetch_distance: u32 = 4;
        const simd_len = 32;
        std.debug.assert(dim % simd_len == 0);
        const Vec = @Vector(simd_len, f32);

        while (self.nb_scored < self.params.search_budget) {
            const node = self.popCandidate();
            if (self.is_search_done) break;

            const start_neigh = graph_k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            var batch_ids: [graph_k_max]u32 = undefined;
            var nb_ids: u32 = 0;

            var neigh_i: u32 = start_neigh;
            while (neigh_i < end_neigh) : (neigh_i += 1) {
                const neighbor = self.neighbors[neigh_i];
                if (self.visited_generation[neighbor] == self.generation) continue;
                std.debug.assert(!self.lm_head.is_junk[neighbor]);

                self.visited_generation[neighbor] = self.generation;
                self.nb_scored += 1;
                self.visited_at[neighbor] = self.nb_scored;
                batch_ids[nb_ids] = neighbor;
                nb_ids += 1;
            }

            var pf_i: u32 = 0;
            while (pf_i < @min(nb_ids, prefetch_distance)) : (pf_i += 1) {
                const prefetch_node = batch_ids[pf_i];
                const row_start = @as(usize, @intCast(prefetch_node)) * dim;
                @prefetch(rows.ptr + row_start, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            var nb_batch: u32 = 0;
            var score_i: u32 = 0;
            while (score_i < nb_ids) : (score_i += 1) {
                const next_pf_i = score_i + prefetch_distance;
                if (next_pf_i < nb_ids) {
                    const prefetch_node = batch_ids[next_pf_i];
                    const row_start = @as(usize, @intCast(prefetch_node)) * dim;
                    @prefetch(rows.ptr + row_start, .{ .rw = .read, .locality = 3, .cache = .data });
                }

                const neighbor = batch_ids[score_i];
                const row_start = @as(usize, @intCast(neighbor)) * dim;
                const row = rows[row_start..][0..dim];

                var acc: Vec = @splat(0);
                var col: usize = 0;
                while (col + simd_len <= dim) : (col += simd_len) {
                    const query_vec: Vec = query[col..][0..simd_len].*;
                    const row_vec: Vec = row[col..][0..simd_len].*;
                    acc = @mulAdd(Vec, query_vec, row_vec, acc);
                }
                const dot = @reduce(.Add, acc);
                const scale = if (graph_type == .Mips) 1.0 else row_norms[neighbor];
                const sim = dot / scale;

                if (self.L == graph_L and self.visited[self.L - 1].similarity >= sim) continue;
                self.batch[nb_batch] = .{ .node = neighbor, .similarity = sim };
                nb_batch += 1;
            }

            var sort_i: u32 = 1;
            while (sort_i < nb_batch) : (sort_i += 1) {
                const candidate = self.batch[sort_i];
                var insert_pos = sort_i;
                while (insert_pos > 0 and candidate.similarity > self.batch[insert_pos - 1].similarity) {
                    self.batch[insert_pos] = self.batch[insert_pos - 1];
                    insert_pos -= 1;
                }
                self.batch[insert_pos] = candidate;
            }

            self.insertBatch(nb_batch);
        }
        self.zml_handler.toc(&self.zml_handler.timers.embed_search);
    }

    pub fn greedySearchWS(self: *Graph, query: []const f32) void {
        // this is called after a pool initialization with initSearchPool
        self.zml_handler.tic(&self.zml_handler.timers.embed_search);
        while (self.nb_scored < self.params.search_budget) {
            const node = self.popCandidate();
            if (self.is_search_done) break;
            const start_neigh = graph_k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            var i: u32 = start_neigh;
            while (i < end_neigh) : (i += 1) {
                const neighbor = self.neighbors[i];
                if (self.visited_generation[neighbor] == self.generation) continue;
                self.addCandidate(query, neighbor);
            }
        }
        self.zml_handler.toc(&self.zml_handler.timers.embed_search);
    }

    pub fn quantizedCrossover(self: *Graph, query: []const f32) void {
        self.generation += 1;
        const crossover_candidates: u32 = self.L;
        var i: u32 = 0;
        while (i < crossover_candidates) : (i += 1) {
            const node = self.visited[i].node;
            self.visited[i].similarity = self.scoreQueryNode(query, node);
            self.nb_scored += 1;
            self.visited_generation[node] = self.generation;
        }
        std.mem.sort(Candidate, self.visited[0..crossover_candidates], {}, Candidate.beforeThan);
        self.nb_scored = crossover_candidates;
        self.is_search_done = false;
        self.greedySearchWS(query);
    }


    pub fn scoreQueryNode(self: *const Graph, query: []const f32, node: u32) f32 {
        self.zml_handler.tic(&self.zml_handler.timers.embed_dot);
        std.debug.assert(!self.lm_head.is_junk[node]);
        const rows = self.lm_head.data;
        const row = rows[node * self.dim ..][0..self.dim];

        const simd_len = 32;
        std.debug.assert(self.dim % simd_len == 0);
        const Vec = @Vector(simd_len, f32);
        var acc: Vec = @splat(0);

        var i: u32 = 0;
        while (i + simd_len <= self.dim) : (i += simd_len) {
            const query_vec: Vec = query[i..][0..simd_len].*;
            const row_vec: Vec = row[i..][0..simd_len].*;
            acc = @mulAdd(Vec, query_vec, row_vec, acc);
        }
        const dot = @reduce(.Add, acc);
        const scale = if (self.params.graph_type == .Mips) 1.0 else self.lm_head.row_norms[node];
        self.zml_handler.toc(&self.zml_handler.timers.embed_dot);
        return dot / scale;
    }

    pub inline fn initNodeSearch(self: *Graph, query: u32) void {
        self.generation += 1;
        
        // at start, pool is empty
        std.debug.assert(self.visited_generation[self.medoid] != self.generation);

        const entry_point, const entry_sim = self.selectNodeEntryPoint(query);

        // medoid is the first and only visited node
        self.visited_generation[entry_point] = self.generation;
        self.visited_at[entry_point] = 1;
        self.visited[0] = .{ .node = entry_point, .similarity = entry_sim };
        self.is_expanded[0] = false;
        self.nb_scored = 1;
        self.L = 1;
        self.is_search_done = false;
    }

    pub inline fn initSearch(self: *Graph, query: []const f32) void {
        self.generation += 1;
        // at start, pool is empty
        std.debug.assert(self.visited_generation[self.medoid] != self.generation);

        const entry_point, const entry_sim = self.selectQueryEntryPoint(query);

        // medoid is the first and only visited node
        self.visited_generation[entry_point] = self.generation;
        self.visited_at[entry_point] = 1;
        self.visited[0] = .{ .node = entry_point, .similarity = entry_sim };
        self.is_expanded[0] = false;
        self.nb_scored = 1;
        self.L = 1;
        self.is_search_done = false;
    }

    pub inline fn initSearchPool(self: *Graph, pool: []const Candidate) void {
        self.generation += 1;
        const entry_point = pool[0].node;
        const entry_sim = pool[0].similarity;

        std.debug.assert(self.visited_generation[entry_point] != self.generation);

        self.visited_generation[entry_point] = self.generation;
        self.visited[0] = .{ .node = entry_point, .similarity = entry_sim };
        self.is_expanded[0] = false;
        self.nb_scored = 1;
        self.L = 1;
        self.is_search_done = false;

        var i: u32 = 1;
        while (i < pool.len) : (i += 1) {
            self.insert(pool[i].node, pool[i].similarity);
        }
    }

    pub inline fn selectNodeEntryPoint(self: *Graph, query: u32) struct { u32, f32 } {
        var entry_point = (query + @divFloor(self.n, 2)) % self.n;
        while (self.lm_head.is_junk[entry_point]) {
            const next = (entry_point + 5411) % self.n;
            entry_point = next;
        }
        const entry_sim = self.similarity(query, entry_point);
        return .{ entry_point, entry_sim };
    }

    pub inline fn selectQueryEntryPoint(self: *Graph, query: []const f32) struct { u32, f32 } {
        const entry_point = self.medoid;
        const entry_sim = self.scoreQueryNode(query, entry_point);
        self.nb_scored += 1;
        std.debug.assert(entry_point < self.n);
        std.debug.assert(self.visited_generation[entry_point] != self.generation);
        return .{ entry_point, entry_sim };
    }

    pub inline fn insert(self: *Graph, node: u32, sim: f32) void {
        //self.zml_handler.tic(&self.zml_handler.timers.insert_node);
        std.debug.assert(!self.lm_head.is_junk[node]);
        std.debug.assert(self.visited_generation[node] != self.generation);
        self.visited_generation[node] = self.generation;
        var insert_pos = self.L;
        while (insert_pos > 0 and sim > self.visited[insert_pos - 1].similarity) {
            self.visited[insert_pos] = self.visited[insert_pos - 1];
            self.is_expanded[insert_pos] = self.is_expanded[insert_pos - 1];
            insert_pos -= 1;
        }
        std.debug.assert(self.L < graph_L + 1);
        std.debug.assert(insert_pos < graph_L + 1);
        self.visited[insert_pos] = .{ .node = node, .similarity = sim };
        self.is_expanded[insert_pos] = false;
        self.L = @min(self.L + 1, graph_L);
        //self.zml_handler.toc(&self.zml_handler.timers.insert_node);
    }

    pub inline fn insertBatch(self: *Graph, nb_batch: u32) void {
        var pos_in_batch = nb_batch;
        var pos_in_pool = self.L;
        var insert_pos = self.L + nb_batch;
        while (pos_in_batch > 0 and pos_in_pool > 0) {
            if (self.batch[pos_in_batch - 1].similarity > self.visited[pos_in_pool - 1].similarity) {
                self.visited[insert_pos - 1] = self.visited[pos_in_pool - 1];
                self.is_expanded[insert_pos - 1] = self.is_expanded[pos_in_pool - 1];
                pos_in_pool -= 1;
            } else {
                self.visited[insert_pos - 1] = self.batch[pos_in_batch - 1];
                self.is_expanded[insert_pos - 1] = false;
                pos_in_batch -= 1;
            }
            insert_pos -= 1;
        }
        var i: u32 = 0;
        while (i < pos_in_batch) : (i += 1) {
            self.visited[i] = self.batch[i];
            self.is_expanded[i] = false;
        }
        self.L = @min(self.L + nb_batch, graph_L);
    }

    pub inline fn popCandidate(self: *Graph) u32 {
        // find the best unexpanded candidate in the active pool
        // since the pool is kept sorted, return the first found
        var i: u32 = 0;
        while (i < self.L) : (i += 1) {
            if (!self.is_expanded[i]) {
                std.debug.assert(self.nb_expanded < self.expanded.len);
                self.expanded[self.nb_expanded] = self.visited[i];
                self.nb_expanded += 1;
                self.is_expanded[i] = true;
                return self.visited[i].node;
            }
        }
        self.is_search_done = true;
        // return any visited node, the search is done
        return self.visited[0].node;
    }

    // ------------- Local neighborhood functions -------------- //

    pub fn setRandomNeighbors(self: *Graph) void {
        std.log.info("Random neighbors", .{});
        var prng = std.Random.DefaultPrng.init(0);
        const random = prng.random();

        const selected = self.allocator.alloc(Candidate, self.n) catch @panic("OOM");
        defer self.allocator.free(selected);

        const is_selected = self.allocator.alloc(bool, self.n) catch @panic("OOM");
        defer self.allocator.free(is_selected);
        @memset(is_selected, false);

        var nb_selected: u32 = 0;

        var i: u32 = 0;
        while (i < self.n) : (i += 1) {
            const start_neigh = graph_k_max * i;
            nb_selected = 0;
            self.nb_neighbors[i] = 0;
            if (self.lm_head.is_junk[i]) continue;
            is_selected[i] = true;

            // rejection method as k_max << n
            while (nb_selected < graph_k_max) {
                const candidate = random.uintLessThan(u32, self.n);
                if (self.lm_head.is_junk[candidate] or is_selected[candidate]) continue;
                // add valid neighbor
                is_selected[candidate] = true;
                selected[nb_selected] = .{ .node = candidate, .similarity = self.similarity(i, candidate) };
                nb_selected += 1;
            }

            std.mem.sort(Candidate, selected[0..nb_selected], {}, Candidate.beforeThan);

            var j: u32 = 0;
            while (j < nb_selected) : (j += 1) {
                const neigh = selected[j].node;
                self.neighbors[start_neigh + j] = neigh;
                is_selected[neigh] = false;
            }
            self.nb_neighbors[i] = nb_selected;
            is_selected[i] = false;

            if (i == 0 or (i + 1) % 10000 == 0 or i + 1 == self.n) {
                std.log.info("Random neighbors node {d}/{d}", .{ i + 1, self.n });
            }
        }
    }

    pub fn consolidateNearest(self: *Graph) void {
        std.log.info("Consolidate with nearest neighbors", .{});

        const scratch = self.allocator.alloc(Candidate, graph_k_max) catch @panic("OOM");
        defer self.allocator.free(scratch);

        const nb_edge_init = self.nbEdges();
        var nb_saturated: u32 = 0;
        var nb_valid: u32 = 0;
        var nb_newly_saturated: u32 = 0;

        var node: u32 = 0;
        while (node < self.n) : (node += 1) {
            if (self.lm_head.is_junk[node]) continue;
            nb_valid += 1;
            if (self.nb_neighbors[node] == graph_k_max) nb_saturated += 1;

            var neigh_pos: u32 = 0;
            while (neigh_pos < self.similarity_matrix.k) : (neigh_pos += 1) {
                if (self.nb_neighbors[node] == graph_k_max) break;
                const candidate: u32 = @intCast(self.similarity_matrix.nearestNeighbor(@intCast(node), @intCast(neigh_pos)));
                if (candidate == node or self.lm_head.is_junk[candidate] or self.hasNeighbor(node, candidate)) continue;
                self.insertNeighbor(node, candidate);
            }
            if (self.nb_neighbors[node] == graph_k_max) nb_newly_saturated += 1;

            if (node == 0 or (node + 1) % 10000 == 0 or node + 1 == self.n) {
                std.log.info("Consolidate nearest node {d}/{d}", .{ node + 1, self.n });
            }
        }
        std.log.info("Consolidated nearest: nb edges: {d} -> {d}", .{ nb_edge_init, self.nbEdges() });
        std.log.info("Consolidated nearest: nb saturated: {d} -> {d} (valid: {d})", .{ nb_saturated, nb_newly_saturated, nb_valid });
    }

    pub fn consolidateNearestPrune(self: *Graph) void {
        std.log.info("Consolidate with pruned nearest neighbors", .{});

        const scratch = self.allocator.alloc(Candidate, graph_k_max) catch @panic("OOM");
        defer self.allocator.free(scratch);

        const nb_edge_init = self.nbEdges();
        var nb_saturated: u32 = 0;
        var nb_valid: u32 = 0;
        var nb_newly_saturated: u32 = 0;

        var node: u32 = 0;
        while (node < self.n) : (node += 1) {
            if (self.lm_head.is_junk[node]) continue;
            nb_valid += 1;
            if (self.nb_neighbors[node] == graph_k_max) nb_saturated += 1;

            var neigh_pos: u32 = 0;
            while (neigh_pos < self.similarity_matrix.k) : (neigh_pos += 1) {
                if (self.nb_neighbors[node] == graph_k_max) break;
                const candidate: u32 = @intCast(self.similarity_matrix.nearestNeighbor(@intCast(node), @intCast(neigh_pos)));
                if (candidate == node or self.lm_head.is_junk[candidate] or self.hasNeighbor(node, candidate)) continue;
                if (self.isPrunedByCurrentNeighbors(node, candidate)) continue;
                self.insertNeighbor(node, candidate);
            }
            if (self.nb_neighbors[node] == graph_k_max) nb_newly_saturated += 1;

            if (node == 0 or (node + 1) % 10000 == 0 or node + 1 == self.n) {
                std.log.info("Consolidate pruned node {d}/{d}", .{ node + 1, self.n });
            }
        }
        std.log.info("Consolidated nearest prune: nb edges: {d} -> {d}", .{ nb_edge_init, self.nbEdges() });
        std.log.info("Consolidated nearest prune: nb saturated: {d} -> {d} (valid: {d})", .{ nb_saturated, nb_newly_saturated, nb_valid });
    }

    pub fn insertNeighbor(self: *Graph, node: u32, candidate: u32) void {
        std.debug.assert(node != candidate);
        std.debug.assert(self.nb_neighbors[node] < graph_k_max);

        const start_neigh = graph_k_max * node;
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

    pub inline fn pruneCandidates(self: *Graph, base: u32, candidates: []Candidate, _: *Field_timer) void {
        std.debug.assert(!self.lm_head.is_junk[base]);
        //self.zml_handler.tic(timer);
        // update current_node neighbors with LOS pruning of the candidates
        const start_neigh = base * graph_k_max;
        self.nb_neighbors[base] = 0;
        var end_neigh = start_neigh;
        for (candidates) |candidate| {
            std.debug.assert(candidate.node != base);
            // The LOS heuristic decides if a candidate node can be added to base's neighbors
            // If any of base's neighbor is already close enough from candidate, then it's rejected,
            // as the routing base -> close_neighbor -> candidate is deemed sufficient
            // The "close enough" formula is dot(close_neighbor, candidate) >= dot(base, candidate)
            // this works in both mips/angular settings because the similarity matrix is different in each case
            const threshold = candidate.similarity;
            var pruned = false;
            var i: u32 = start_neigh;
            while (i < end_neigh) : (i += 1) {
                const neighbor = self.neighbors[i];
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
            if (self.nb_neighbors[base] == graph_k_max) break;
        }
        //self.zml_handler.toc(timer);
    }

    pub fn isPrunedByCurrentNeighbors(self: *Graph, base: u32, candidate: u32) bool {
        std.debug.assert(base != candidate);

        const threshold = self.similarity(base, candidate);

        const start_neigh = graph_k_max * base;
        const end_neigh = start_neigh + self.nb_neighbors[base];
        var i: u32 = start_neigh;
        while (i < end_neigh) : (i += 1) {
            if (self.similarity(candidate, self.neighbors[i]) >= threshold) return true;
        }
        return false;
    }

    // ------------------- NSW main function -------------------- //

    pub fn extendToNsw(self: *Graph) !void {
        //try self.benchSimilarity();
        const candidates = self.allocator.alloc(Candidate, 2 * graph_k_max + self.params.search_budget) catch @panic("OOM");
        defer self.allocator.free(candidates);

        var pass_i: u32 = 0;
        while (pass_i < self.params.vamana_passes) : (pass_i += 1) {
            // random visit order
            std.log.info("NSW pass {d}/{d}", .{ pass_i + 1, self.params.vamana_passes });
            const start = std.Io.Timestamp.now(self.io, .awake);
            var i: u32 = 0;
            while (i < self.n) : (i += 1) {
                // at this iteration, we will update current_node's neighbors and add current_node as a neighbor in candidate nodes
                const current_node = self.n - (i + 1);
                const start_neigh = graph_k_max * current_node;
                var end_neigh = start_neigh + self.nb_neighbors[current_node];

                if (self.lm_head.is_junk[current_node]) continue;
                var nb_candidates: u32 = 0;

                // the candidates are current_node's neighbors and the visited nodes
                // since both lists are sorted and contain unique nodes, we can build
                // the sorted list of candidates in one linear forward pass
                self.greedySearchNode(current_node);

                const nb_cand = self.nb_expanded;
                std.mem.sort(Candidate, self.expanded[0..nb_cand], {}, Candidate.beforeThan);
                const cands = self.expanded[0..nb_cand];

                //const nb_cand = self.L;
                //const cands = self.visited[0..nb_cand];

                var pos_in_neighbors: u32 = start_neigh;
                var pos_in_visited: u32 = 0;
                while (pos_in_neighbors < end_neigh and pos_in_visited < nb_cand) {
                    // if current_node was visited during the search, skip it in the visited pool
                    // otherwise it will end up being a neighbor of itself
                    // note that on the not metric case, if current_node was visited,
                    // it might not be the best candidate (the first in self.visited)
                    if (cands[pos_in_visited].node == current_node) {
                        pos_in_visited += 1;
                        continue;
                    }
                    const neigh = self.neighbors[pos_in_neighbors];
                    const visit = cands[pos_in_visited].node;
                    const neigh_sim = self.similarity(neigh, current_node);
                    const visit_sim = cands[pos_in_visited].similarity;
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
                var j: u32 = pos_in_neighbors;
                while (j < end_neigh) : (j += 1) {
                    const neigh = self.neighbors[j];
                    candidates[nb_candidates] = .{ .node = neigh, .similarity = self.similarity(neigh, current_node) };
                    nb_candidates += 1;
                }
                j = pos_in_visited;
                while (j < nb_cand) : (j += 1) {
                    if (cands[j].node == current_node) continue;
                    const visit = cands[j].node;
                    candidates[nb_candidates] = .{ .node = visit, .similarity = cands[j].similarity };
                    nb_candidates += 1;
                }

                // forward prune on candidates
                self.pruneCandidates(current_node, candidates[0..nb_candidates], &self.zml_handler.timers.prune_pool_fwd);

                // from there, we insert current_node into each of its neighbors
                end_neigh = start_neigh + self.nb_neighbors[current_node];
                j = start_neigh;
                while (j < end_neigh) : (j += 1) {
                    const neighbor = self.neighbors[j];
                    const start_neigh_neigh = graph_k_max * neighbor;
                    const end_neigh_neigh = start_neigh_neigh + self.nb_neighbors[neighbor];
                    const sim = self.similarity(neighbor, current_node);

                    // if neighbor -> current_node exists, skip
                    if (self.hasNeighbor(neighbor, current_node)) continue;

                    // if candidate has no room and current_node would be at the end of the neighbors,
                    // we can skip. this is in theory only true if the next case leaves the neighbors pruned
                    const worse_neigh = neighbor * graph_k_max + self.nb_neighbors[neighbor] - 1;
                    const worse_neigh_sim = self.similarity(neighbor, self.neighbors[worse_neigh]);
                    if (self.nb_neighbors[neighbor] == graph_k_max and worse_neigh_sim >= sim) continue;

                    // if candidate still has room, add current_node to its neighbors
                    if (self.nb_neighbors[neighbor] < graph_k_max) {
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

                    // reverse candidates : neighbor's neighbors + current_node
                    nb_candidates = 0;
                    var k: u32 = start_neigh_neigh;
                    while (k < end_neigh_neigh) : (k += 1) {
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
        }
    }

    pub fn fixNswExtention(self: *Graph) !void {
        std.log.info("Fix NSW extension", .{});
        const in_degrees = try self.allocator.alloc(u32, self.n);
        defer self.allocator.free(in_degrees);
        @memset(in_degrees, 0);

        var node: u32 = 0;
        while (node < self.n) : (node += 1) {
            if (self.lm_head.is_junk[node]) continue;
            const start_neigh = graph_k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            var neigh: u32 = start_neigh;
            while (neigh < end_neigh) : (neigh += 1) {
                in_degrees[self.neighbors[neigh]] += 1;
            }
        }

        node = 0;
        while (node < self.n) : (node += 1) {
            if (self.lm_head.is_junk[node]) continue;
            if (in_degrees[node] < 8) {
                var k: u32 = 0;
                while (k < self.similarity_matrix.k) : (k += 1) {
                    const neigh: u32 = @intCast(self.similarity_matrix.nearestNeighbor(@intCast(node), @intCast(k)));
                    if (self.nb_neighbors[neigh] == graph_k_max) continue;
                    if (self.lm_head.is_junk[neigh]) continue;
                    self.neighbors[neigh * graph_k_max + self.nb_neighbors[neigh]] = @intCast(node);
                    self.nb_neighbors[neigh] += 1;
                    in_degrees[node] += 1;
                    if (in_degrees[node] == 8) break;
                }
            }
        }
    }
    
    pub fn testNswExtention(self: *Graph, sampler: *sampling.Sampler) !void {
        if (true) return;
        std.log.info("Test NSW extension", .{});
        @memset(self.nsw_extension_search_missed, false);

        const in_degrees = try self.allocator.alloc(u32, self.n);
        defer self.allocator.free(in_degrees);
        @memset(in_degrees, 0);

        var node: u32 = 0;
        while (node < self.n) : (node += 1) {
            if (self.lm_head.is_junk[node]) continue;
            const start_neigh = graph_k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            var neigh: u32 = start_neigh;
            while (neigh < end_neigh) : (neigh += 1) {
                in_degrees[self.neighbors[neigh]] += 1;
            }
        }
        var min_in_degree = self.n;
        var max_in_degree: u32 = 0;
        node = 0;
        while (node < self.n) : (node += 1) {
            if (self.lm_head.is_junk[node]) continue;
            min_in_degree = @min(min_in_degree, in_degrees[node]);
            max_in_degree = @max(max_in_degree, in_degrees[node]);
        }
        std.log.info("Min in-degree: {}", .{min_in_degree});
        std.log.info("Max in-degree: {}", .{max_in_degree});

        //try self.logDegreeCounts("Nodes by in-degree", in_degrees, max_in_degree);
        node = 0;
        while (node < self.n) : (node += 1) {
            if (self.lm_head.is_junk[node] or in_degrees[node] <= 2000) continue;
            const token_str = try tokens.tokenString(sampler.tokenizer, @as(u32, @intCast(node)), self.allocator);
            std.log.info("Node {d} has in-degree {d}, token: {s}", .{ node, in_degrees[node], token_str });
            self.allocator.free(token_str);
        }
        //try self.logDegreeCounts("Nodes by out-degree", self.nb_neighbors, graph_k_max);

        const hop_dist = try self.getHopDistance();
        defer self.allocator.free(hop_dist);
        var max_hops: u32 = 0;
        for (hop_dist) |hops| {
            if (hops > max_hops) max_hops = hops;
        }
        std.log.info("Max hops: {}", .{max_hops});
        var hops: u32 = 2;
        while (hops < 2) : (hops += 1) {
            var i: u32 = 0;
            while (i < self.n) : (i += 1) {
                if (hop_dist[i] == hops) {
                    std.log.info("Node {d} needs {d} hops, out-degree: {d}, tok: {s}", .{ i, hops, self.nb_neighbors[i], try tokens.tokenString(sampler.tokenizer, i, self.allocator) });
                }
            }
        }
        hops = 8;
        while (hops < 8) : (hops += 1) {
            var i: u32 = 0;
            while (i < self.n) : (i += 1) {
                if (hop_dist[i] == hops) {
                    std.log.info("Node {d} needs {d} hops, out-degree: {d}, tok: {s}", .{ i, hops, self.nb_neighbors[i], try tokens.tokenString(sampler.tokenizer, i, self.allocator) });
                }
            }
        }
        //try self.logDegreeCounts("Nodes by min-hops", hop_dist, max_hops);

        var exact_first_count: u32 = 0;
        var valid_count: u32 = 0;
        var total_visited: u32 = 0;
        var min_visited: u32 = std.math.maxInt(u32);
        var max_visited: u32 = 0;
        var total_best_found_at: u64 = 0;
        var min_best_found_at: u32 = std.math.maxInt(u32);
        var max_best_found_at: u32 = 0;

        node = 0;
        while (node < self.n) : (node += 1) {
            if (self.lm_head.is_junk[node]) continue;
            valid_count += 1;
            self.greedySearchNode(node);
            const nb_scored = self.nb_scored;
            total_visited += nb_scored;
            min_visited = @min(min_visited, nb_scored);
            max_visited = @max(max_visited, nb_scored);
            const best_found_at = self.visited_at[self.visited[0].node];
            total_best_found_at += @intCast(best_found_at);
            min_best_found_at = @min(min_best_found_at, best_found_at);
            max_best_found_at = @max(max_best_found_at, best_found_at);
            var found = false;
            var i: u32 = 0;
            while (i < self.L) : (i += 1) {
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
                std.log.info("NSW extension test node {d}/{d}, success rate {d:.4}%", .{ node + 1, self.n, 100.0 * exact_rate });
            }
        }

        const avg_visited = @as(f64, @floatFromInt(total_visited)) / @as(f64, @floatFromInt(valid_count));
        const avg_best_found_at = @as(f64, @floatFromInt(total_best_found_at)) / @as(f64, @floatFromInt(valid_count));
        const exact_rate = @as(f64, @floatFromInt(exact_first_count)) / @as(f64, @floatFromInt(valid_count));
        std.log.info("NSW extension entry-point starts", .{});
        std.log.info(
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
        std.log.info(
            "NSW extension final best found_at: min={d} max={d} avg={d:.2}",
            .{
                if (valid_count == 0) 0 else min_best_found_at,
                max_best_found_at,
                avg_best_found_at,
            },
        );
    }

    pub fn logNsw(self: *Graph, start: std.Io.Timestamp, i: u32) void {
        const now = std.Io.Timestamp.now(self.io, .awake);
        const elapsed_duration = std.Io.Timestamp.durationTo(start, now);
        const elapsed_seconds = @as(f64, @floatFromInt(elapsed_duration.nanoseconds)) / 1e9;
        const eta_seconds = elapsed_seconds * @as(f64, @floatFromInt(self.n - i - 1)) / @as(f64, @floatFromInt(i + 1));
        std.log.info(
            "NSW node {d}/{d} elapsed={d:.2}s eta={d:.2}s",
            .{ i + 1, self.n, elapsed_seconds, eta_seconds },
        );
    }

    pub fn logNswRandom(self: *Graph, start: std.Io.Timestamp, query_i: u32, nb_queries: u32, nb_added_edges: u32, nb_skipped_queries: u32, nb_missed: u32, nb_processed: u32) void {
        const now = std.Io.Timestamp.now(self.io, .awake);
        const elapsed_duration = std.Io.Timestamp.durationTo(start, now);
        const elapsed_seconds = @as(f64, @floatFromInt(elapsed_duration.nanoseconds)) / 1e9;
        const eta_seconds = elapsed_seconds * @as(f64, @floatFromInt(nb_queries - query_i - 1)) / @as(f64, @floatFromInt(query_i + 1));
        std.log.info(
            "NSW random row {d}/{d} processed={d} missed={d} added_edges={d} skipped_queries={d} edges={d} elapsed={d:.2}s eta={d:.2}s",
            .{ query_i + 1, nb_queries, nb_processed, nb_missed, nb_added_edges, nb_skipped_queries, self.nbEdges(), elapsed_seconds, eta_seconds },
        );
    }

    // ----------------------- NSW utils ------------------------ //

    fn logDegreeCounts(self: *Graph, title: []const u8, degrees: []const u32, max_degree: u32) !void {
        const counts = try self.allocator.alloc(u32, max_degree + 1);
        defer self.allocator.free(counts);
        @memset(counts, 0);

        var node: u32 = 0;
        while (node < self.n) : (node += 1) {
            if (self.lm_head.is_junk[node]) continue;
            counts[degrees[node]] += 1;
        }

        std.log.info("{s}", .{title});
        var degree: u32 = 0;
        while (degree < counts.len) : (degree += 1) {
            const count = counts[degree];
            if (count == 0) continue;
            std.log.info("{d} nodes of degree {d}", .{ count, degree });
        }
    }

    pub fn hasNeighbor(self: *const Graph, node: u32, candidate: u32) bool {
        const start_neigh = graph_k_max * node;
        const end_neigh = start_neigh + self.nb_neighbors[node];
        var i: u32 = start_neigh;
        while (i < end_neigh) : (i += 1) {
            if (self.neighbors[i] == candidate) return true;
        }
        return false;
    }

    // ------------------- Hierarchy functions ------------------- //
    
    pub fn getHopDistance(self: *Graph) ![]u32 {
        const hop_dist = try self.allocator.alloc(u32, self.n);
        errdefer self.allocator.free(hop_dist);
        @memset(hop_dist, self.n);

        var queue: std.ArrayList(u32) = try .initCapacity(self.allocator, 0);
        defer queue.deinit(self.allocator);

        hop_dist[self.medoid] = 0;
        try queue.append(self.allocator, self.medoid);

        var queue_head: u32 = 0;
        while (queue_head < queue.items.len) {
            const node = queue.items[queue_head];
            queue_head += 1;
            const start_neigh = node * graph_k_max;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            var pos: u32 = start_neigh;
            while (pos < end_neigh) : (pos += 1) {
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

    pub fn nbEdges(self: *const Graph) u32 {
        var count: u32 = 0;
        var i: u32 = 0;
        while (i < self.n) : (i += 1) {
            count += self.nb_neighbors[i];
        }
        return count;
    }

    pub inline fn similarity(self: *Graph, a: u32, b: u32) f32 {
        return self.similarity_matrix.dist(@intCast(a), @intCast(b));
    }

    pub fn benchSimilarity(self: *Graph) !void {
        const i = try self.allocator.alloc(u32, 10_000);
        const j = try self.allocator.alloc(u32, 10_000);
        defer self.allocator.free(i);
        defer self.allocator.free(j);
        const A_i: u32 = 1_000_007;
        const B_i: u32 = 123_789;
        const A_j: u32 = 645_007;
        const B_j: u32 = 456_123;
        i[0] = 0;
        j[0] = 0;
        var k: u32 = 1;
        while (k < 10_000) : (k += 1) {
            i[k] = (A_i * i[k - 1] + B_i) % self.n;
            j[k] = (A_j * j[k - 1] + B_j) % self.n;
        }
        const start = std.Io.Timestamp.now(self.io, .awake);
        var sim: f32 = 0.0;
        k = 0;
        while (k < 10_000) : (k += 1) {
            sim += self.similarity(i[k], j[k]);
        }
        const end = std.Io.Timestamp.now(self.io, .awake);
        const duration = end.nanoseconds - start.nanoseconds;
        std.log.info("sim: {d:.3} in {d} ns", .{ sim, duration });
        std.log.info("Time per access: {d} us", .{@as(f32, @floatFromInt(duration)) / @as(f32, @floatFromInt(10_000 * 1_000))});
    }
};
