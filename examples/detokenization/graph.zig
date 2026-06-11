const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");

const log = std.log;

pub const GraphParams = struct {
    k_max: usize = 16,
    search_budget: usize = 512,
    alpha: f16 = 1.25,
    vamana_passes: usize = 1,
    top_k: usize = 16,
    L: usize = 128,
};

pub const Graph = struct {

    pub const Candidate = struct {
        node: usize,
        similarity: f16,

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
    // graph fields
    // TODO: we can use smaller integer types
    n: usize,
    params: GraphParams,
    neighbors: []usize,
    nb_neighbors: []usize,
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
    
    pub fn init(zml_handler: *main.Zml_handler, lm_head: zml.Slice, lm_head_normalized: zml.Slice, matrix: *main.SimilarityMatrix, params: GraphParams) !Graph {
        std.debug.assert(matrix.n > 0);
        std.debug.assert(params.k_max > 0);
        std.debug.assert(params.L > 0);
        std.debug.assert(params.search_budget > 0);
        std.debug.assert(params.alpha >= 1.0);
        std.debug.assert(matrix.k >= params.k_max);
        std.debug.assert(params.L <= params.search_budget + params.k_max);
        std.debug.assert(params.search_budget + params.k_max <= matrix.n);
        std.debug.assert(params.k_max < matrix.n);

        const allocator = zml_handler.allocator;
        const medoid = try getMedoid(allocator, lm_head_normalized, matrix.n, matrix.d);

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
            .similarity_matrix = matrix,
            .medoid = medoid,
            .capacity = params.search_budget + params.k_max,
            .L = 0,
            .visited = try allocator.alloc(Candidate, params.search_budget + params.k_max),
            .is_visited = is_visited,
            .nb_visited = 0,
            .is_expanded = is_expanded,
            .is_search_done = false,
        };
    }

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.neighbors);
        self.allocator.free(self.nb_neighbors);
        self.allocator.free(self.visited);
        self.allocator.free(self.is_visited);
        self.allocator.free(self.is_expanded);
    }

    pub fn getMedoid(allocator: std.mem.Allocator, lm_head_normalized: zml.Slice, n: usize, dim: usize) !usize {
        const rows = lm_head_normalized.constItems(f16);

        // compute normalized average using f64 for accurate accumulation
        const average = try allocator.alloc(f64, dim);
        defer allocator.free(average);
        @memset(average, 0.0);
        for (0..n) |i| {
            const row_i = rows[i * dim ..][0..dim];
            for (0..dim) |j| {
                average[j] += @floatCast(row_i[j]);
            }
        }
        const inv_n = 1.0 / @as(f64, @floatFromInt(n));
        var norm2: f64 = 0.0;
        for (0..dim) |i| {
            average[i] *= inv_n;
            norm2 += average[i] * average[i];
        }
        const inv_norm = 1.0 / @sqrt(norm2);
        const average_f16 = try allocator.alloc(f16, dim);
        defer allocator.free(average_f16);
        for (0..dim) |i| {
            average_f16[i] = @floatCast(average[i] * inv_norm);
        }

        // medoid is the row that is most similar to the average
        var best_row: usize = 0;
        var best_similarity: f16 = -std.math.inf(f16);
        for (0..n) |i| {
            const row_i = rows[i * dim ..][0..dim];
            var row_i_similarity: f16 = 0.0;
            for (0..dim) |j| {
                row_i_similarity += row_i[j] * average_f16[j];
            }
            if (row_i_similarity > best_similarity) {
                best_similarity = row_i_similarity;
                best_row = i;
            }
        }
        return best_row;
    }
    
    // ------------------- Search functions ------------------ //

    pub fn greedySearchNode(self: *Graph, query: usize) []Candidate {
        // initialize search at entry point
        self.initNodeSearch(query);

        while (self.nb_visited < self.params.search_budget) {
            
            // find best node of the active pool that has not been expanded yet
            const node = self.popCandidate();

            // if all nodes in active pool have been expanded, terminate the search
            if (self.is_search_done) break;

            // otherwise, expand search to best node neighbors
            const start_neigh = self.params.k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                if (self.is_visited[neighbor]) continue;
                self.addNodeCandidate(query, neighbor);
            }
        }

        self.cleanup();
        return self.visited[0..self.nb_visited];
    }

    pub fn greedySearch(self: *Graph, query: []const f16) []Candidate {
        // initialize search at entry point
        self.initSearch(query);

        while (self.nb_visited < self.params.search_budget) {
            
            // find best node of the active pool that has not been expanded yet
            const node = self.popCandidate();

            // if all nodes in active pool have been expanded, terminate the search
            if (self.is_search_done) break;

            // otherwise, expand search to best node neighbors
            const start_neigh = self.params.k_max * node;
            const end_neigh = start_neigh + self.nb_neighbors[node];
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                if (self.is_visited[neighbor]) continue;
                self.addCandidate(query, neighbor);
            }
        }
        
        self.cleanup();
        return self.visited[0..self.nb_visited];
    }

    
    pub fn scoreQueryNode(self: *const Graph, query: []const f16, node: usize) f16 {
        const rows = self.lm_head_normalized.constItems(f16);
        const row = rows[node * self.dim ..][0..self.dim];
        var dot: f16 = 0;
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
    
    pub fn initSearch(self: *Graph, query: []const f16) void {
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
        std.debug.assert(!self.is_visited[node]);
        std.debug.assert(self.nb_visited > 0);
        const sim = self.similarity(node, query);
        self.insert(node, sim);
    }

    pub fn addCandidate(self: *Graph, query: []const f16, node: usize) void {
        std.debug.assert(!self.is_visited[node]);
        std.debug.assert(self.nb_visited > 0);
        const sim = self.scoreQueryNode(query, node);
        self.insert(node, sim);
    }
    
    pub fn insert(self: *Graph, node: usize, sim: f16) void {
        std.debug.assert(!self.is_visited[node]);
        std.debug.assert(self.nb_visited > 0);
        std.debug.assert(self.L > 0);
        // TODO: is we split pool management into visited and active pool, we can simplify
        
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
        for (0..self.L) |i| { // TODO: we can keep track of the first unexpanded node
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
        const is_selected = self.allocator.alloc(bool, self.n) catch @panic("OOM");
        defer self.allocator.free(is_selected);
        @memset(is_selected, false);

        for (0..self.n) |i| {
            const start_neigh = self.params.k_max * i;
            is_selected[i] = true;
            self.nb_neighbors[i] = 0;

            // rejection method as k_max << n
            while (self.nb_neighbors[i] < self.params.k_max) {
                const candidate = random.uintLessThan(usize, self.n);
                if (is_selected[candidate]) continue;
                // add valid neighbor
                is_selected[candidate] = true;
                self.neighbors[start_neigh + self.nb_neighbors[i]] = candidate;
                self.nb_neighbors[i] += 1;
            }

            // reset rejection utils
            is_selected[i] = false;
            for (0..self.nb_neighbors[i]) |j| {
                is_selected[self.neighbors[start_neigh + j]] = false;
            }
            if (i == 0 or (i + 1) % 10000 == 0 or i + 1 == self.n) {
                log.info("Random neighbors node {d}/{d}", .{ i + 1, self.n });
            }
        }
    }

    pub fn setNearestNeighbors(self: *Graph) void {
        log.info("Nearest neighbors", .{});
        for (0..self.n) |i| {
            const start_neigh = self.params.k_max * i;
            self.nb_neighbors[i] = self.params.k_max;
            for (0..self.params.k_max) |j| {
                self.neighbors[start_neigh + j] = self.similarity_matrix.nearestNeighbor(i, j);
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
            for (0..self.similarity_matrix.k) |j| {
                const candidate = self.similarity_matrix.nearestNeighbor(i, j);
                if (!self.lineOfSight(i, candidate)) continue;
                self.neighbors[start_neigh + self.nb_neighbors[i]] = candidate;
                self.nb_neighbors[i] += 1;
                if (self.nb_neighbors[i] == self.params.k_max) break;
            }
            if (i == 0 or (i + 1) % 5000 == 0 or i + 1 == self.n) {
                log.info("Nearest neighbors LOS node {d}/{d}", .{ i + 1, self.n });
            }
        }
    }

    pub fn pruneCandidates(self: *Graph, i: usize, candidates: []Candidate) void {
        // TODO: split reverse/forward prune timers
        self.zml_handler.tic(&self.zml_handler.timers.prune_pool);
        // we assume candidates similarities is already set
        std.mem.sort(Candidate, candidates, {}, Candidate.beforeThan);

        // update current_node neighbors with LOS pruning of the candidates
        const start_neigh = i * self.params.k_max;
        self.nb_neighbors[i] = 0;
        for (candidates) |candidate| {
            // TODO: pass candidates to los, to avoid recomputing similarity
            if (!self.lineOfSight(i, candidate.node)) continue;
            self.neighbors[start_neigh + self.nb_neighbors[i]] = candidate.node;
            self.nb_neighbors[i] += 1;
            if (self.nb_neighbors[i] == self.params.k_max) break;
        }
        self.zml_handler.toc(&self.zml_handler.timers.prune_pool);
    }
    
    // --------------------- LOS heuristic ---------------------- //

    // TODO: inline manually inside pruneCandidates (recomputes similarity and neighbor range)
    pub fn lineOfSight(self: *const Graph, base: usize, candidate: usize) bool {
        // The LOS heuristic decides if a candidate node can be added to base's neighbors
        // If any of base's neighbor is already close enough from candidate, then it's rejected,
        // as the routing base -> close_neighbor -> candidate is deemed sufficient
        // The "close enough" formula is alpha * dist(close_neighbor, candidate) <= dist(base, candidate)
        const threshold = 1.0 - (1.0 - self.similarity(base, candidate)) / (self.params.alpha * self.params.alpha);
        const start_neigh = self.params.k_max * base;
        const end_neigh = start_neigh + self.nb_neighbors[base];
        for (start_neigh..end_neigh) |i| {
            const neighbor = self.neighbors[i];
            // for u and v norm 1 vectors, ||u - v||² = u² + v² - 2 <u,v> = 2 (1 - sim(u,v))
            // a||n - c|| <= ||b - c|| => a²||n - c||² <= ||b - c||²
            // 2a² (1 - sim(n,c)) <= 2 (1 - sim(b,c))
            // sim(n,c) >= 1 - (1 - sim(b,c)) / a²
            if (self.similarity(neighbor, candidate) >= threshold) return false;
        }
        return true;
    }

    // ------------------- NSW main function -------------------- //

    pub fn extendToNsw(self: *Graph) void {
        var prng = std.Random.DefaultPrng.init(1);
        const random = prng.random();

        const order = self.allocator.alloc(usize, self.n) catch @panic("OOM");
        defer self.allocator.free(order);
        for (0..order.len) |i| order[i] = i;

        const is_candidate = self.allocator.alloc(bool, self.n) catch @panic("OOM");
        defer self.allocator.free(is_candidate);
        @memset(is_candidate, false);

        const candidates = self.allocator.alloc(Candidate, 2 * self.params.k_max + self.params.search_budget) catch @panic("OOM");
        defer self.allocator.free(candidates);

        const start = std.Io.Timestamp.now(self.io, .awake);

        for (0..self.params.vamana_passes) |pass_i| {
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
                var nb_candidates: usize = 0;

                // candidates are initialized from current node's neighbors
                const start_neigh = self.params.k_max * current_node;
                var end_neigh = start_neigh + self.nb_neighbors[current_node];
                for (start_neigh..end_neigh) |j| {
                    const candidate = self.neighbors[j];
                    // since neighbors are unique, no need to test if already candidate
                    is_candidate[candidate] = true;
                    candidates[nb_candidates].node = candidate;
                    candidates[nb_candidates].similarity = self.similarity(current_node, candidate);
                    nb_candidates += 1;
                }
                // add new candidates: all nodes visited during the greedy search medoid -> current_node
                // there might be intersection with the current neighbor, so we need a datastructure to filter those
                self.zml_handler.tic(&self.zml_handler.timers.greedy_search);
                const pool = self.greedySearchNode(current_node);
                self.zml_handler.toc(&self.zml_handler.timers.greedy_search);
                for (0..pool.len) |j| {
                    const candidate = pool[j].node;
                    if (is_candidate[candidate] or candidate == current_node) continue;
                    is_candidate[candidate] = true;
                    candidates[nb_candidates] = pool[j];
                    nb_candidates += 1;
                }
                // clear candidates unicity util
                for (0..nb_candidates) |j| is_candidate[candidates[j].node] = false;

                // forward prune on candidates
                self.pruneCandidates(current_node, candidates[0..nb_candidates]);
                
                // from there, we insert current_node into each of its neighbors
                end_neigh = start_neigh + self.nb_neighbors[current_node];
                for (start_neigh..end_neigh) |j| {
                    const neighbor = self.neighbors[j];
                    
                    // if neighbor -> current_node exists, skip
                    if (self.hasNeighbor(neighbor, current_node)) continue;

                    // if candidate still has room, add current_node to its neighbors
                    if (self.nb_neighbors[neighbor] < self.params.k_max) {
                        self.addNeighbor(neighbor, current_node);
                        continue;
                    }

                    // TODO: can add a fail fast heuristic
                    // - if current_node is further than any neighbor
                    // - if neighbor row is already pruned
                    // in this case, no room will be made by pruning and current node will never be added
                    
                    // reverse candidates : neighbor's neighbors + current_node
                    nb_candidates = 0;
                    const start_neigh_neigh = self.params.k_max * neighbor;
                    const end_neigh_neigh = start_neigh_neigh + self.nb_neighbors[neighbor];
                    
                    for (start_neigh_neigh..end_neigh_neigh) |k| {
                        const neigh_neigh = self.neighbors[k];
                        // since neighbors are unique, no need to test if already candidate
                        candidates[nb_candidates].node = neigh_neigh;
                        candidates[nb_candidates].similarity = self.similarity(neighbor, neigh_neigh);
                        nb_candidates += 1;
                    }
                    // TODO: if we maintain neighbors sorted by similarity, we can simply do a reverse linear pass
                    // to insert current_node at the right position in candidates, and avoid the sort in pruneCandidates
                    // we already tested that current_node is not a neighbor of neighbor
                    candidates[nb_candidates].node = current_node;
                    candidates[nb_candidates].similarity = self.similarity(neighbor, current_node);
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

    pub fn addNeighbor(self: *Graph, node: usize, new_neighbor: usize) void {
        std.debug.assert(new_neighbor != node);
        std.debug.assert(!self.hasNeighbor(node, new_neighbor));
        std.debug.assert(self.nb_neighbors[node] < self.params.k_max);
        
        self.neighbors[self.params.k_max * node + self.nb_neighbors[node]] = new_neighbor;
        self.nb_neighbors[node] += 1;
    }

    // ---------------------- Syntax utils ----------------------- //

    pub fn nbNodes(self: *const Graph) usize {
        var count: usize = 0;
        for (0..self.n) |i| {
            count += self.nb_neighbors[i];
        }
        return count;
    }

    pub fn similarity(self: *const Graph, a: usize, b: usize) f16 {
        return self.similarity_matrix.dist(a, b);
    }

};
