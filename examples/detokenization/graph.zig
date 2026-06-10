const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");

const log = std.log;

pub const GraphParams = struct {
    k_max: usize = 32,
    search_budget: usize = 256,
    alpha: f16 = 1.25,
    vamana_passes: usize = 2,
};

pub const Graph = struct {

    const Candidate = struct {
        node: usize,
        similarity: f16,

        fn beforeThan(_: void, lhs: Candidate, rhs: Candidate) bool {
            return lhs.similarity > rhs.similarity or (lhs.similarity == rhs.similarity and lhs.node < rhs.node);
        }
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    // dataset fields
    dim: usize,
    lm_head: zml.Slice,
    lm_head_normalized: zml.Slice,
    similarity_matrix: *main.SimilarityMatrix,
    // graph fields
    n: usize,
    params: GraphParams,
    neighbors: []usize,
    nb_neighbors: []usize,
    // search fields
    medoid: usize,
    visited_nodes: []usize, // this is the buffer for visited nodes, no allocation
    nb_visited_nodes: usize, // this is the range in visited_nodes that are currently in use
    is_node_visited: []bool, // tells if each node in the graph has been visited. incrementally updated: reset to false from visited_nodes at end of search, init to false
    scores: []f16, // scores for each visited node. unitialized, updated whenever visited_nodes is updated

    pub fn init(zml_handler: *main.Zml_handler, lm_head: zml.Slice, lm_head_normalized: zml.Slice, matrix: *main.SimilarityMatrix, params: GraphParams) !Graph {
        const allocator = zml_handler.allocator;
        const medoid = try getMedoid(allocator, lm_head_normalized, matrix.n, matrix.d);

        const neighbors = try allocator.alloc(usize, matrix.n * params.k_max);
        errdefer allocator.free(neighbors);

        const nb_neighbors = try allocator.alloc(usize, matrix.n);
        errdefer allocator.free(nb_neighbors);
        @memset(nb_neighbors, 0);

        const visited_nodes = try allocator.alloc(usize, params.search_budget);
        errdefer allocator.free(visited_nodes);

        const is_node_visited = try allocator.alloc(bool, matrix.n);
        errdefer allocator.free(is_node_visited);
        @memset(is_node_visited, false);

        const scores = try allocator.alloc(f16, params.search_budget);
        errdefer allocator.free(scores);

        return .{
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
            .visited_nodes = visited_nodes,
            .nb_visited_nodes = 0,
            .is_node_visited = is_node_visited,
            .scores = scores,
        };
    }

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.neighbors);
        self.allocator.free(self.nb_neighbors);
        self.allocator.free(self.visited_nodes);
        self.allocator.free(self.is_node_visited);
        self.allocator.free(self.scores);
    }

    // ------------------ High level functions ----------------- //

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

    pub fn greedySearch(self: *Graph, q: []const f16) struct { []usize, []f16 } {
        self.nb_visited_nodes = 0;
        const start_score = self.scoreQueryNode(q, self.medoid);
        self.addVisitedNode(self.medoid, start_score);

        var current_node = self.medoid;
        var current_score = start_score;
        while (self.nb_visited_nodes < self.params.search_budget) {
            var best_next_node = current_node;
            var best_next_score = current_score;
            var found_new_neighbor = false;

            const start_neigh = self.params.k_max * current_node;
            const end_neigh = start_neigh + self.nb_neighbors[current_node];
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                if (self.nb_visited_nodes >= self.params.search_budget) break;
                if (self.is_node_visited[neighbor]) continue;

                const score = self.scoreQueryNode(q, neighbor);
                self.addVisitedNode(neighbor, score);
                
                found_new_neighbor = true;
                if (score > best_next_score) {
                    best_next_score = score;
                    best_next_node = neighbor;
                }
            }

            if (!found_new_neighbor or best_next_node == current_node) break;
            current_node = best_next_node;
            current_score = best_next_score;
        }

        for (0..self.nb_visited_nodes) |i| {
            self.is_node_visited[self.visited_nodes[i]] = false;
        }

        return .{ self.visited_nodes[0..self.nb_visited_nodes], self.scores[0..self.nb_visited_nodes] };
    }

    pub fn greedySearchNode(self: *Graph, q: usize) struct { []usize, []f16 } {
        self.nb_visited_nodes = 0;
        const start_score = self.similarity(self.medoid, q);
        self.addVisitedNode(self.medoid, start_score);

        var current_node = self.medoid;
        var current_score = start_score;
        while (self.nb_visited_nodes < self.params.search_budget) {
            var best_next_node = current_node;
            var best_next_score = current_score;
            var found_new_neighbor = false;

            const start_neigh = self.params.k_max * current_node;
            const end_neigh = start_neigh + self.nb_neighbors[current_node];
            for (start_neigh..end_neigh) |i| {
                const neighbor = self.neighbors[i];
                if (self.nb_visited_nodes >= self.params.search_budget) break;
                if (self.is_node_visited[neighbor]) continue;

                const score = self.similarity(neighbor, q);
                self.addVisitedNode(neighbor, score);
                
                found_new_neighbor = true;
                if (score > best_next_score) {
                    best_next_score = score;
                    best_next_node = neighbor;
                }
            }

            if (!found_new_neighbor or best_next_node == current_node) break;
            current_node = best_next_node;
            current_score = best_next_score;
        }

        for (0..self.nb_visited_nodes) |i| {
            self.is_node_visited[self.visited_nodes[i]] = false;
        }

        return .{ self.visited_nodes[0..self.nb_visited_nodes], self.scores[0..self.nb_visited_nodes] };
    }
    
    pub fn addVisitedNode(self: *Graph, node: usize, score: f16) void {
        std.debug.assert(self.nb_visited_nodes < self.params.search_budget);
        std.debug.assert(!self.is_node_visited[node]);
        self.is_node_visited[node] = true;
        self.visited_nodes[self.nb_visited_nodes] = node;
        self.scores[self.nb_visited_nodes] = score;
        self.nb_visited_nodes += 1;
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

    pub fn pruneNeighbors(self: *Graph) void {
        log.info("Pruning neighbors LOS", .{});
        for (0..self.n) |i| {
            self.pruneNodeNeighbors(i);
            if (i == 0 or (i + 1) % 5000 == 0 or i + 1 == self.n) {
                log.info("Pruning neighbors LOS node {d}/{d}", .{ i + 1, self.n });
            }
        }
    }

    pub fn pruneNodeNeighbors(self: *Graph, i: usize) void {
        const start_neigh = self.params.k_max * i;
        const end_neigh = start_neigh + self.nb_neighbors[i];
        self.nb_neighbors[i] = 0;
        for (start_neigh..end_neigh) |j| {
            const candidate = self.neighbors[j];
            if (!self.lineOfSight(i, candidate)) continue;
            self.neighbors[start_neigh + self.nb_neighbors[i]] = candidate;
            self.nb_neighbors[i] += 1;
        }
    }

    // --------------------- LOS heuristic ---------------------- //

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
            // a||n - c|| <= ||b - c|| => a²||n - c|| <= ||b - c||
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

        const candidates = self.allocator.alloc(Candidate, self.params.k_max + self.params.search_budget) catch @panic("OOM");
        defer self.allocator.free(candidates);

        const reverse_candidates = self.allocator.alloc(Candidate, self.params.k_max + 1) catch @panic("OOM");
        defer self.allocator.free(reverse_candidates);

        const start = std.Io.Timestamp.now(self.io, .awake);
        const total_iterations = self.params.vamana_passes * self.n;

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
                const visited, const scores = self.greedySearchNode(current_node);
                for (0..visited.len) |j| {
                    const candidate = visited[j];
                    if (is_candidate[candidate] or candidate == current_node) continue;
                    is_candidate[candidate] = true;
                    candidates[nb_candidates].node = candidate;
                    candidates[nb_candidates].similarity = scores[j];
                    nb_candidates += 1;
                }

                // sort pool with decreasing similarity to candidate
                std.mem.sort(Candidate, candidates[0..nb_candidates], {}, Candidate.beforeThan);

                // update current_node neighbors with LOS pruning of the candidates
                self.nb_neighbors[current_node] = 0;
                for (0..nb_candidates) |j| {
                    const candidate = candidates[j].node;
                    if (!self.lineOfSight(current_node, candidate)) continue;
                    self.neighbors[start_neigh + self.nb_neighbors[current_node]] = candidate;
                    self.nb_neighbors[current_node] += 1;
                    if (self.nb_neighbors[current_node] == self.params.k_max) break;
                }
                
                // from there, we insert current_node into each of its neighbors 
                end_neigh = start_neigh + self.nb_neighbors[current_node];
                for (start_neigh..end_neigh) |j| {
                    const neighbor = self.neighbors[j];
                    
                    // if neighbor -> current_node exists, a routing is already possible, skip
                    if (self.hasNeighbor(neighbor, current_node)) continue;

                    // if candidate still has room, add current_node to its neighbors
                    if (self.nb_neighbors[neighbor] < self.params.k_max) {
                        self.addNeighbor(neighbor, current_node);
                        continue;
                    }
                    
                    // if neighbor is saturated, try to make room for current_node by pruning with LOS
                    self.pruneNodeNeighbors(neighbor);
                    if (self.nb_neighbors[neighbor] < self.params.k_max) {
                        if (self.lineOfSight(neighbor, current_node)) self.addNeighbor(neighbor, current_node);
                    }
                }
                // clear candidates pool
                for (0..nb_candidates) |j| {
                    is_candidate[candidates[j].node] = false;
                }

                if (i == 0 or (i + 1) % 1000 == 0 or i + 1 == self.n) {
                    const now = std.Io.Timestamp.now(self.io, .awake);
                    const elapsed_duration = std.Io.Timestamp.durationTo(start, now);
                    const elapsed_seconds = @as(f64, @floatFromInt(elapsed_duration.nanoseconds)) / 1e9;
                    const completed_iterations = pass_i * self.n + i + 1;
                    const eta_seconds = elapsed_seconds * @as(f64, @floatFromInt(total_iterations - completed_iterations)) / @as(f64, @floatFromInt(completed_iterations));
                    log.info(
                        "NSW pass {d}/{d} node {d}/{d} elapsed={d:.2}s eta={d:.2}s",
                        .{ pass_i + 1, self.params.vamana_passes, i + 1, self.n, elapsed_seconds, eta_seconds },
                    );
                }
            }
        }
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
