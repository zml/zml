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
    };

    allocator: std.mem.Allocator,
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
            if (i == 0 or (i + 1) % 1000 == 0 or i + 1 == self.n) {
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
            if (i == 0 or (i + 1) % 1000 == 0 or i + 1 == self.n) {
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
                if (!self.lineOfSight(i, candidate, self.nb_neighbors[i])) continue;
                self.neighbors[start_neigh + self.nb_neighbors[i]] = candidate;
                self.nb_neighbors[i] += 1;
                if (self.nb_neighbors[i] == self.params.k_max) break;
            }
            if (i == 0 or (i + 1) % 1000 == 0 or i + 1 == self.n) {
                log.info("Nearest neighbors LOS node {d}/{d}", .{ i + 1, self.n });
            }
        }
    }

    pub fn pruneNeighbors(self: *Graph) void {
        log.info("Pruning neighbors LOS", .{});
        const is_pruned = self.allocator.alloc(bool, self.params.k_max) catch @panic("OOM");
        defer self.allocator.free(is_pruned);
        @memset(is_pruned[0..self.params.k_max], false);

        for (0..self.n) |i| {
            const start_neigh = self.params.k_max * i;
            const nb_neighbors = self.nb_neighbors[i];
            for (0..nb_neighbors) |j| {
                const candidate = self.neighbors[start_neigh + j];
                is_pruned[j] = !self.lineOfSightPartial(i, candidate, j, is_pruned);
            }
            // slide the non-pruned neighbors to consolidate, reset the pruned flags
            var free_pos: usize = 0;
            for (0..nb_neighbors) |j| {
                self.neighbors[start_neigh + free_pos] = self.neighbors[start_neigh + j];
                if (is_pruned[j]) {
                    is_pruned[j] = false;
                } else {
                    free_pos += 1;
                }
            }
            self.nb_neighbors[i] = free_pos;
            if (i == 0 or (i + 1) % 1000 == 0 or i + 1 == self.n) {
                log.info("Pruning neighbors LOS node {d}/{d}", .{ i + 1, self.n });
            }
        }
    }

    // --------------------- LOS heuristic ---------------------- //

    fn lineOfSight(self: *const Graph, base: usize, candidate: usize, candidate_position: usize) bool {
        const base_candidate_distance = self.distance(base, candidate);
        const start_neigh = self.params.k_max * base;
        const end_neigh = start_neigh + candidate_position;
        for (start_neigh..end_neigh) |i| {
            const neighbor = self.neighbors[i];
            if (self.params.alpha * self.distance(neighbor, candidate) < base_candidate_distance) return false;
        }
        return true;
    }

    fn lineOfSightPartial(self: *const Graph, base: usize, candidate: usize, candidate_position: usize, is_pruned: []const bool) bool {
        const base_candidate_distance = self.distance(base, candidate);
        const start_neigh = self.params.k_max * base;
        const end_neigh = start_neigh + candidate_position;
        for (start_neigh..end_neigh) |i| {
            if (is_pruned[i - start_neigh]) continue;
            const neighbor = self.neighbors[i];
            if (self.params.alpha * self.distance(neighbor, candidate) < base_candidate_distance) return false;
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

        const candidate_flags = self.allocator.alloc(bool, self.n) catch @panic("OOM");
        defer self.allocator.free(candidate_flags);
        @memset(candidate_flags, false);

        const candidate_pool = self.allocator.alloc(usize, self.n) catch @panic("OOM");
        defer self.allocator.free(candidate_pool);

        const reverse_candidate_flags = self.allocator.alloc(bool, self.n) catch @panic("OOM");
        defer self.allocator.free(reverse_candidate_flags);
        @memset(reverse_candidate_flags, false);

        const reverse_candidate_pool = self.allocator.alloc(usize, self.n) catch @panic("OOM");
        defer self.allocator.free(reverse_candidate_pool);

        for (0..self.params.vamana_passes) |pass_i| {
            // random visit order
            var nb_swap = order.len - 1;
            while (nb_swap > 0) : (nb_swap -= 1) {
                const j = random.uintLessThan(usize, nb_swap + 1);
                std.mem.swap(usize, &order[nb_swap], &order[j]);
            }
            log.info("NSW pass {d}/{d}", .{ pass_i + 1, self.params.vamana_passes });
            for (0..order.len) |i| {
                const current_node = order[i];
                var pool_count: usize = 0;

                // candidates neighbors are initialized from current node's neighbors
                const start_neigh = self.params.k_max * current_node;
                const end_neigh = start_neigh + self.nb_neighbors[current_node];
                for (start_neigh..end_neigh) |j| {
                    markCandidate(candidate_flags, candidate_pool, &pool_count, self.neighbors[j]);
                }
                // add new candidates: all nodes visited during the greedy search medoid -> current_node
                const query = self.lm_head_normalized.items(f16)[current_node * self.dim..][0..self.dim];
                const visited, _ = self.greedySearch(query);
                for (0..visited.len) |j| {
                    markCandidate(candidate_flags, candidate_pool, &pool_count, visited[j]);
                }
                candidate_flags[current_node] = false;

                self.setNeighborsFromCandidateFlags(current_node, candidate_flags);

                const pruned_start_neigh = self.params.k_max * current_node;
                const pruned_end_neigh = pruned_start_neigh + self.nb_neighbors[current_node];
                for (pruned_start_neigh..pruned_end_neigh) |j| {
                    const neighbor = self.neighbors[j];
                    if (neighbor == current_node or self.hasNeighbor(neighbor, current_node)) continue;
                    if (self.nb_neighbors[neighbor] < self.params.k_max) {
                        self.addNeighborAssumeCapacity(neighbor, current_node);
                    } else {
                        var reverse_pool_count: usize = 0;
                        const reverse_start_neigh = self.params.k_max * neighbor;
                        const reverse_end_neigh = reverse_start_neigh + self.nb_neighbors[neighbor];
                        for (reverse_start_neigh..reverse_end_neigh) |k| {
                            markCandidate(reverse_candidate_flags, reverse_candidate_pool, &reverse_pool_count, self.neighbors[k]);
                        }
                        markCandidate(reverse_candidate_flags, reverse_candidate_pool, &reverse_pool_count, current_node);
                        reverse_candidate_flags[neighbor] = false;
                        self.setNeighborsFromCandidateFlags(neighbor, reverse_candidate_flags);
                        clearCandidates(reverse_candidate_flags, reverse_candidate_pool[0..reverse_pool_count]);
                    }
                }

                clearCandidates(candidate_flags, candidate_pool[0..pool_count]);

                if (i == 0 or (i + 1) % 1000 == 0 or i + 1 == self.n) {
                    log.info("NSW pass {d}/{d} node {d}/{d}", .{ pass_i + 1, self.params.vamana_passes, i + 1, self.n });
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

    pub fn addNeighborAssumeCapacity(self: *Graph, node: usize, candidate: usize) void {
        std.debug.assert(candidate != node);
        std.debug.assert(self.nb_neighbors[node] < self.params.k_max);
        const pos = self.nb_neighbors[node];
        self.neighbors[self.params.k_max * node + pos] = candidate;
        self.nb_neighbors[node] = pos + 1;
    }

    pub fn addVisitedNode(self: *Graph, node: usize, score: f16) void {
        std.debug.assert(self.nb_visited_nodes < self.params.search_budget);
        self.is_node_visited[node] = true;
        self.visited_nodes[self.nb_visited_nodes] = node;
        self.scores[self.nb_visited_nodes] = score;
        self.nb_visited_nodes += 1;
    }

    pub fn nextNearestCandidate(self: *const Graph, base: usize, previous: ?Candidate, candidate_flags: ?[]const bool) ?Candidate {
        var best: ?Candidate = null;
        for (0..self.n) |i| {
            if (i == base) continue;
            if (candidate_flags) |flags| {
                if (!flags[i]) continue;
            }
            const candidate: Candidate = .{ .node = i, .similarity = self.similarity(base, i) };
            if (!candidateComesAfter(candidate, previous)) continue;
            if (best == null or candidateIsBetter(candidate, best.?)) best = candidate;
        }
        return best;
    }

    pub fn candidateComesAfter(candidate: Candidate, previous: ?Candidate) bool {
        const prev = previous orelse return true;
        return candidate.similarity < prev.similarity or (candidate.similarity == prev.similarity and candidate.node > prev.node);
    }

    pub fn candidateIsBetter(candidate: Candidate, best: Candidate) bool {
        return candidate.similarity > best.similarity or (candidate.similarity == best.similarity and candidate.node < best.node);
    }

    pub fn candidateGreaterThan(_: void, lhs: Candidate, rhs: Candidate) bool {
        return lhs.similarity > rhs.similarity or (lhs.similarity == rhs.similarity and lhs.node < rhs.node);
    }

    pub fn setNeighborsFromCandidateFlags(self: *Graph, base: usize, candidate_flags: []const bool) void {
        self.nb_neighbors[base] = 0;
        var previous: ?Candidate = null;
        while (self.nb_neighbors[base] < self.params.k_max) {
            const candidate = self.nextNearestCandidate(base, previous, candidate_flags) orelse break;
            previous = candidate;
            if (self.lineOfSight(base, candidate.node, self.nb_neighbors[base])) {
                self.addNeighborAssumeCapacity(base, candidate.node);
            }
        }
    }

    pub fn markCandidate(flags: []bool, pool: []usize, pool_count: *usize, node: usize) void {
        if (flags[node]) return;
        flags[node] = true;
        pool[pool_count.*] = node;
        pool_count.* += 1;
    }

    pub fn clearCandidates(flags: []bool, pool: []const usize) void {
        for (0..pool.len) |i| flags[pool[i]] = false;
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

    pub fn distance(self: *const Graph, a: usize, b: usize) f16 {
        const sim = self.similarity(a, b);
        return @sqrt(@max(@as(f16, 0), @as(f16, 2) - @as(f16, 2) * sim));
    }
};
