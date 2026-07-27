const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const algebra = @import("algebra.zig");
const save_load = @import("saveload.zig");
const llm = @import("llm.zig");
const tokens = @import("tokens.zig");
const sampling = @import("sampling.zig");
const quantization = @import("quantization.zig");

const Tokenizer = zml.tokenizer.Tokenizer;
const SimilarityMatrix = algebra.SimilarityMatrix;
const LmHeadMatrix = algebra.LmHeadMatrix;
const Zml_handler = main.Zml_handler;
const Field_timer = main.Timing_handler.Field_timer;
const QuantizationInt8 = quantization.QuantizationInt8;
const QuantizationInt4 = quantization.QuantizationInt4;
const QuantizationQJL1 = quantization.QuantizationQJL1;
const QuantizationQJL2 = quantization.QuantizationQJL2;
const LoadedVectorQJL1Quarter = quantization.LoadedVectorQJL1Quarter;
const VectorQJL1 = quantization.VectorQJL1;
const VectorQJL1Half = quantization.VectorQJL1Half;
const VectorQJL1Quarter = quantization.VectorQJL1Quarter;
const VectorQJL2 = quantization.VectorQJL2;

pub const graph_k_max = 64;
pub const graph_L = 512;
pub const graph_construction_search_budget = 4096;

pub const GraphParams = struct {
    vamana_passes: u32 = 2,
    top_k: u32 = 16,
    graph_type: GraphType = .Mips,
};

pub const GraphType = enum {
    Angular,
    Mips,
};

pub const GraphScoreType = enum {
    Dense,
    Int8,
    Int8x4,
    QJL1Quarter,
    QJL1Half,
    QJL1,
    QJL2,
};

pub const DenseScorer = struct {
    pub const prefetch_distance: u32 = 4;

    lm_head: *LmHeadMatrix,
    graph_type: GraphType,
    query: []const f32 = &.{},

    pub fn init(lm_head: *LmHeadMatrix, graph_type: GraphType) DenseScorer {
        return .{
            .lm_head = lm_head,
            .graph_type = graph_type,
        };
    }

    pub fn prepare(self: *DenseScorer, query: []const f32) void {
        std.debug.assert(query.len == self.lm_head.d);
        self.query = query;
    }

    pub inline fn prefetch(self: *const DenseScorer, node: u32) void {
        const row_start = @as(usize, @intCast(node)) * self.lm_head.d;
        @prefetch(self.lm_head.data.ptr + row_start, .{
            .rw = .read,
            .locality = 0,
            .cache = .data,
        });
    }

    pub inline fn score(self: *const DenseScorer, node: u32) f32 {
        const dim = self.lm_head.d;
        const row_start = @as(usize, @intCast(node)) * dim;
        const row = self.lm_head.data[row_start..][0..dim];

        const simd_len = 32;
        std.debug.assert(dim % simd_len == 0);
        const Vec = @Vector(simd_len, f32);
        var acc: Vec = @splat(0);

        var col: usize = 0;
        while (col + simd_len <= dim) : (col += simd_len) {
            const query_vec: Vec = self.query[col..][0..simd_len].*;
            const row_vec: Vec = row[col..][0..simd_len].*;
            acc = @mulAdd(Vec, query_vec, row_vec, acc);
        }
        const dot = @reduce(.Add, acc);
        const scale = if (self.graph_type == .Mips) 1.0 else self.lm_head.row_norms[node];
        return dot / scale;
    }
};

pub const Int8Scorer = struct {
    pub const prefetch_distance: u32 = 4;

    quantizer: QuantizationInt8,
    graph_type: GraphType,
    quantized_query: []i8,
    query_quant_scale: f32 = 0.0,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, graph_type: GraphType) !Int8Scorer {
        var quantizer: QuantizationInt8 = try .init(zml_handler, lm_head);
        errdefer quantizer.deinit();

        const quantized_query = try zml_handler.allocator.alloc(i8, lm_head.d);
        errdefer zml_handler.allocator.free(quantized_query);

        try quantizer.quantize();
        return .{
            .quantizer = quantizer,
            .graph_type = graph_type,
            .quantized_query = quantized_query,
        };
    }

    pub fn deinit(self: *Int8Scorer) void {
        self.quantizer.allocator.free(self.quantized_query);
        self.quantizer.deinit();
    }

    pub fn prepare(self: *Int8Scorer, query: []const f32) void {
        const query_norm = quantization.normL2(query);
        self.query_quant_scale = QuantizationInt8.quantizeVector(
            query,
            self.quantizer.buffer,
            self.quantized_query,
            query_norm,
        );
    }

    pub inline fn prefetch(self: *const Int8Scorer, node: u32) void {
        const row_start = @as(usize, @intCast(node)) * self.quantizer.d;
        @prefetch(self.quantizer.lm_head_quantized.ptr + row_start, .{
            .rw = .read,
            .locality = 0,
            .cache = .data,
        });
    }

    pub inline fn score(self: *const Int8Scorer, node: u32) f32 {
        const row_start = @as(usize, @intCast(node)) * self.quantizer.d;
        const quantized_row = self.quantizer.lm_head_quantized[row_start..][0..self.quantizer.d];
        const quantized_score = QuantizationInt8.int8dot(self.quantized_query, quantized_row);
        const dot = @as(f32, @floatFromInt(quantized_score)) *
            self.quantizer.row_quant_scale[node] *
            self.query_quant_scale;
        const scale = if (self.graph_type == .Mips)
            1.0
        else
            self.quantizer.lm_head.row_norms[node];
        return dot / scale;
    }
};

pub const Int8x4Scorer = struct {
    pub const prefetch_distance: u32 = 4;

    quantizer: QuantizationInt4,
    graph_type: GraphType,
    quantized_query: []i8,
    query_quant_scale: f32 = 0.0,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, graph_type: GraphType) !Int8x4Scorer {
        var quantizer: QuantizationInt4 = try .init(zml_handler, lm_head);
        errdefer quantizer.deinit();

        const quantized_query = try zml_handler.allocator.alloc(i8, lm_head.d);
        errdefer zml_handler.allocator.free(quantized_query);

        try quantizer.quantize();
        return .{
            .quantizer = quantizer,
            .graph_type = graph_type,
            .quantized_query = quantized_query,
        };
    }

    pub fn deinit(self: *Int8x4Scorer) void {
        self.quantizer.allocator.free(self.quantized_query);
        self.quantizer.deinit();
    }

    pub fn prepare(self: *Int8x4Scorer, query: []const f32) void {
        const query_norm = quantization.normL2(query);
        self.query_quant_scale = QuantizationInt8.quantizeVector(
            query,
            self.quantizer.buffer,
            self.quantized_query,
            query_norm,
        );
    }

    pub inline fn prefetch(self: *const Int8x4Scorer, node: u32) void {
        const packed_d = self.quantizer.d / 2;
        const row_start = @as(usize, @intCast(node)) * packed_d;
        @prefetch(self.quantizer.lm_head_quantized.ptr + row_start, .{
            .rw = .read,
            .locality = 0,
            .cache = .data,
        });
    }

    pub inline fn score(self: *const Int8x4Scorer, node: u32) f32 {
        const packed_d = self.quantizer.d / 2;
        const row_start = @as(usize, @intCast(node)) * packed_d;
        const quantized_row = self.quantizer.lm_head_quantized[row_start..][0..packed_d];
        const quantized_score = QuantizationInt4.int8x4dot(self.quantized_query, quantized_row);
        const dot = @as(f32, @floatFromInt(quantized_score)) *
            self.quantizer.row_quant_scale[node] *
            self.query_quant_scale;
        const scale = if (self.graph_type == .Mips)
            1.0
        else
            self.quantizer.lm_head.row_norms[node];
        return dot / scale;
    }
};

pub const QJL1QuarterScorer = struct {
    pub const prefetch_distance: u32 = 16;

    quantizer: QuantizationQJL1,
    graph_type: GraphType,
    quantized_query: VectorQJL1,
    loaded_query: LoadedVectorQJL1Quarter,
    query_norm: f32 = 0.0,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, graph_type: GraphType) !QJL1QuarterScorer {
        var quantizer: QuantizationQJL1 = try .init(zml_handler, lm_head);
        errdefer quantizer.deinit();
        try quantizer.quantize();
        return .{
            .quantizer = quantizer,
            .graph_type = graph_type,
            .quantized_query = quantization.makeVectorQJL1(),
            .loaded_query = undefined,
        };
    }

    pub fn deinit(self: *QJL1QuarterScorer) void {
        self.quantizer.deinit();
    }

    pub fn prepare(self: *QJL1QuarterScorer, query: []const f32) void {
        self.query_norm = quantization.normL2(query);
        _ = QuantizationQJL1.quantizeVector(
            query,
            self.quantizer.buffer,
            &self.quantized_query,
            self.query_norm,
        );
        const query_quarter: *const VectorQJL1Quarter = @ptrCast(&self.quantized_query);
        self.loaded_query = QuantizationQJL1.loadVectorQJL1Quarter(query_quarter);
    }

    pub inline fn prefetch(self: *const QJL1QuarterScorer, node: u32) void {
        @prefetch(&self.quantizer.lm_head_quantized_quarter1[node], .{
            .rw = .read,
            .locality = 0,
            .cache = .data,
        });
    }

    pub inline fn score(self: *const QJL1QuarterScorer, node: u32) f32 {
        const quantized_row = &self.quantizer.lm_head_quantized_quarter1[node];
        const mismatches = QuantizationQJL1.popcountXorQuarterLoaded(self.loaded_query, quantized_row);
        const cosine = QuantizationQJL1.qjl_dot_lut_quarter[mismatches];
        const row_scale = if (self.graph_type == .Mips)
            self.quantizer.lm_head.row_norms[node]
        else
            1.0;
        return cosine * self.query_norm * row_scale;
    }
};

pub const QJL1HalfScorer = struct {
    pub const prefetch_distance: u32 = 16;

    quantizer: QuantizationQJL1,
    graph_type: GraphType,
    quantized_query: VectorQJL1,
    query_norm: f32 = 0.0,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, graph_type: GraphType) !QJL1HalfScorer {
        var quantizer: QuantizationQJL1 = try .init(zml_handler, lm_head);
        errdefer quantizer.deinit();
        try quantizer.quantize();
        return .{
            .quantizer = quantizer,
            .graph_type = graph_type,
            .quantized_query = quantization.makeVectorQJL1(),
        };
    }

    pub fn deinit(self: *QJL1HalfScorer) void {
        self.quantizer.deinit();
    }

    pub fn prepare(self: *QJL1HalfScorer, query: []const f32) void {
        self.query_norm = quantization.normL2(query);
        _ = QuantizationQJL1.quantizeVector(
            query,
            self.quantizer.buffer,
            &self.quantized_query,
            self.query_norm,
        );
    }

    pub inline fn prefetch(self: *const QJL1HalfScorer, node: u32) void {
        @prefetch(&self.quantizer.lm_head_quantized[node], .{
            .rw = .read,
            .locality = 0,
            .cache = .data,
        });
    }

    pub inline fn score(self: *const QJL1HalfScorer, node: u32) f32 {
        const query_half: *const VectorQJL1Half = @ptrCast(&self.quantized_query);
        const quantized_row: *const VectorQJL1Half = @ptrCast(&self.quantizer.lm_head_quantized[node]);
        const mismatches = QuantizationQJL1.popcountXorHalf(query_half, quantized_row);
        const cosine = QuantizationQJL1.qjl_dot_lut_half[mismatches];
        const row_scale = if (self.graph_type == .Mips)
            self.quantizer.lm_head.row_norms[node]
        else
            1.0;
        return cosine * self.query_norm * row_scale;
    }
};

pub const QJL1Scorer = struct {
    pub const prefetch_distance: u32 = 16;

    quantizer: QuantizationQJL1,
    graph_type: GraphType,
    quantized_query: VectorQJL1,
    query_norm: f32 = 0.0,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, graph_type: GraphType) !QJL1Scorer {
        var quantizer: QuantizationQJL1 = try .init(zml_handler, lm_head);
        errdefer quantizer.deinit();
        try quantizer.quantize();
        return .{
            .quantizer = quantizer,
            .graph_type = graph_type,
            .quantized_query = quantization.makeVectorQJL1(),
        };
    }

    pub fn deinit(self: *QJL1Scorer) void {
        self.quantizer.deinit();
    }

    pub fn prepare(self: *QJL1Scorer, query: []const f32) void {
        self.query_norm = quantization.normL2(query);
        _ = QuantizationQJL1.quantizeVector(
            query,
            self.quantizer.buffer,
            &self.quantized_query,
            self.query_norm,
        );
    }

    pub inline fn prefetch(self: *const QJL1Scorer, node: u32) void {
        @prefetch(&self.quantizer.lm_head_quantized[node], .{
            .rw = .read,
            .locality = 0,
            .cache = .data,
        });
    }

    pub inline fn score(self: *const QJL1Scorer, node: u32) f32 {
        const quantized_row = &self.quantizer.lm_head_quantized[node];
        const cosine = QuantizationQJL1.qjl1dot(&self.quantized_query, quantized_row);
        const row_scale = if (self.graph_type == .Mips)
            self.quantizer.lm_head.row_norms[node]
        else
            1.0;
        return cosine * self.query_norm * row_scale;
    }
};

pub const QJL2Scorer = struct {
    pub const prefetch_distance: u32 = 16;

    quantizer: QuantizationQJL2,
    graph_type: GraphType,
    quantized_query: VectorQJL2,
    query_scale: f32 = 0.0,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, graph_type: GraphType) !QJL2Scorer {
        var quantizer: QuantizationQJL2 = try .init(zml_handler, lm_head);
        errdefer quantizer.deinit();
        try quantizer.quantize();
        return .{
            .quantizer = quantizer,
            .graph_type = graph_type,
            .quantized_query = quantization.makeVectorQJL2(),
        };
    }

    pub fn deinit(self: *QJL2Scorer) void {
        self.quantizer.deinit();
    }

    pub fn prepare(self: *QJL2Scorer, query: []const f32) void {
        const query_norm = quantization.normL2(query);
        self.query_scale = QuantizationQJL2.quantizeVector(
            query,
            self.quantizer.buffer,
            &self.quantized_query,
            query_norm,
        );
    }

    pub inline fn prefetch(self: *const QJL2Scorer, node: u32) void {
        @prefetch(&self.quantizer.lm_head_quantized[node], .{
            .rw = .read,
            .locality = 0,
            .cache = .data,
        });
    }

    pub inline fn score(self: *const QJL2Scorer, node: u32) f32 {
        const quantized_row = &self.quantizer.lm_head_quantized[node];
        const dot_i32 = QuantizationQJL2.qjl2dot(&self.quantized_query, quantized_row);
        const dot = @as(f32, @floatFromInt(dot_i32)) *
            self.quantizer.row_quant_scale[node] *
            self.query_scale;
        const scale = if (self.graph_type == .Mips)
            1.0
        else
            self.quantizer.lm_head.row_norms[node];
        return dot / scale;
    }
};

pub const GraphSearchState = union(GraphScoreType) {
    Dense: DenseScorer,
    Int8: Int8Scorer,
    Int8x4: Int8x4Scorer,
    QJL1Quarter: QJL1QuarterScorer,
    QJL1Half: QJL1HalfScorer,
    QJL1: QJL1Scorer,
    QJL2: QJL2Scorer,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, graph_type: GraphType, score_type: GraphScoreType) !GraphSearchState {
        return switch (score_type) {
            .Dense => .{ .Dense = .init(lm_head, graph_type) },
            .Int8 => .{ .Int8 = try .init(zml_handler, lm_head, graph_type) },
            .Int8x4 => .{ .Int8x4 = try .init(zml_handler, lm_head, graph_type) },
            .QJL1Quarter => .{ .QJL1Quarter = try .init(zml_handler, lm_head, graph_type) },
            .QJL1Half => .{ .QJL1Half = try .init(zml_handler, lm_head, graph_type) },
            .QJL1 => .{ .QJL1 = try .init(zml_handler, lm_head, graph_type) },
            .QJL2 => .{ .QJL2 = try .init(zml_handler, lm_head, graph_type) },
        };
    }

    pub fn deinit(self: *GraphSearchState) void {
        switch (self.*) {
            .Dense => {},
            .Int8 => |*scorer| scorer.deinit(),
            .Int8x4 => |*scorer| scorer.deinit(),
            .QJL1Quarter => |*scorer| scorer.deinit(),
            .QJL1Half => |*scorer| scorer.deinit(),
            .QJL1 => |*scorer| scorer.deinit(),
            .QJL2 => |*scorer| scorer.deinit(),
        }
    }
};

pub const Graph = struct {
    pub const Candidate = struct {
        node: u32,
        similarity: f32,

        fn beforeThan(_: void, lhs: Candidate, rhs: Candidate) bool {
            return lhs.similarity > rhs.similarity or (lhs.similarity == rhs.similarity and lhs.node < rhs.node);
        }
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
    generation: u32,
    // during one iteration of greedy search, the batch of visited neighbors
    batch: []Candidate,
    // for each node in the pool, tells if its neighbors have been
    // added to the pool (when true, the node had been dealt with)
    is_expanded: []bool,
    is_search_done: bool,

    pub fn init(zml_handler: *Zml_handler, lm_head: *LmHeadMatrix, matrix: *SimilarityMatrix, params: GraphParams) !Graph {
        const allocator = zml_handler.allocator;

        const neighbors = try allocator.alloc(u32, matrix.n * graph_k_max);
        errdefer allocator.free(neighbors);

        const nb_neighbors = try allocator.alloc(u32, matrix.n);
        errdefer allocator.free(nb_neighbors);
        @memset(nb_neighbors, 0);

        const is_expanded = try allocator.alloc(bool, matrix.n);
        errdefer allocator.free(is_expanded);
        @memset(is_expanded, false);

        const visited = try allocator.alloc(Candidate, graph_L + graph_k_max);
        errdefer allocator.free(visited);

        const batch = try allocator.alloc(Candidate, graph_k_max);
        errdefer allocator.free(batch);

        const visited_generation = try allocator.alloc(u32, matrix.n);
        errdefer allocator.free(visited_generation);
        @memset(visited_generation, 0);

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
            .generation = 0,
            .batch = batch,
            .is_expanded = is_expanded,
            .is_search_done = false,
        };
    }

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.neighbors);
        self.allocator.free(self.nb_neighbors);
        self.allocator.free(self.visited);
        self.allocator.free(self.visited_generation);
        self.allocator.free(self.batch);
        self.allocator.free(self.is_expanded);
    }

    // ------------------- Search functions ------------------ //

    pub fn greedySearchNode(self: *Graph, query: u32, search_budget: u32) void {
        //self.zml_handler.tic(&self.zml_handler.timers.greedy_search);
        std.debug.assert(!self.lm_head.is_junk[query]);
        // initialize search at entry point
        self.initNodeSearch(query);

        var nb_scored = self.nb_scored;
        while (nb_scored < search_budget) {

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

                if (self.L == graph_L and self.visited[self.L - 1].similarity >= sim) continue;
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

    pub fn greedySearch(self: *Graph, search_state: anytype, search_budget: u32) void {
        self.initSearchWithState(search_state);
        const prefetch_distance = @TypeOf(search_state.*).prefetch_distance;

        while (self.nb_scored < search_budget) {
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
                batch_ids[nb_ids] = neighbor;
                nb_ids += 1;
            }

            var pf_i: u32 = 0;
            while (pf_i < @min(nb_ids, prefetch_distance)) : (pf_i += 1) {
                search_state.prefetch(batch_ids[pf_i]);
            }

            var nb_batch: u32 = 0;
            var score_i: u32 = 0;
            while (score_i < nb_ids) : (score_i += 1) {
                const next_pf_i = score_i + prefetch_distance;
                if (next_pf_i < nb_ids) {
                    search_state.prefetch(batch_ids[next_pf_i]);
                }

                const neighbor = batch_ids[score_i];
                const sim = search_state.score(neighbor);

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
    }

    pub fn rerankPool(self: *Graph, search_state: anytype) void {
        const prefetch_distance = @TypeOf(search_state.*).prefetch_distance;
        var candidate_i: u32 = 0;
        while (candidate_i < @min(self.L, prefetch_distance)) : (candidate_i += 1) {
            search_state.prefetch(self.visited[candidate_i].node);
        }
        candidate_i = 0;
        while (candidate_i < self.L) : (candidate_i += 1) {
            const prefetch_i = candidate_i + prefetch_distance;
            if (prefetch_i < self.L) {
                search_state.prefetch(self.visited[prefetch_i].node);
            }
            const candidate = &self.visited[candidate_i];
            candidate.similarity = search_state.score(candidate.node);
        }
        std.mem.sort(Candidate, self.visited[0..self.L], {}, Candidate.beforeThan);
    }

    pub fn rerankDensePoolAsMips(self: *Graph) void {
        if (self.params.graph_type == .Mips) return;

        for (self.visited[0..self.L]) |*candidate| {
            candidate.similarity *= self.lm_head.row_norms[candidate.node];
        }
        std.mem.sort(Candidate, self.visited[0..self.L], {}, Candidate.beforeThan);
    }

    pub fn scoreQueryNode(self: *const Graph, query: []const f32, node: u32) f32 {
        self.zml_handler.tic(&self.zml_handler.timers.embed_dot);
        std.debug.assert(!self.lm_head.is_junk[node]);
        var scorer: DenseScorer = .init(self.lm_head, self.params.graph_type);
        scorer.prepare(query);
        const score = scorer.score(node);
        self.zml_handler.toc(&self.zml_handler.timers.embed_dot);
        return score;
    }

    pub inline fn initNodeSearch(self: *Graph, query: u32) void {
        self.generation += 1;

        // at start, pool is empty
        std.debug.assert(self.visited_generation[self.medoid] != self.generation);

        const entry_point, const entry_sim = self.selectNodeEntryPoint(query);

        // medoid is the first and only visited node
        self.visited_generation[entry_point] = self.generation;
        self.visited[0] = .{ .node = entry_point, .similarity = entry_sim };
        self.is_expanded[0] = false;
        self.nb_scored = 1;
        self.L = 1;
        self.is_search_done = false;
    }

    pub inline fn initSearch(self: *Graph, query: []const f32) void {
        var scorer: DenseScorer = .init(self.lm_head, self.params.graph_type);
        scorer.prepare(query);
        self.initSearchWithState(&scorer);
    }

    pub inline fn initSearchWithState(self: *Graph, search_state: anytype) void {
        self.generation += 1;
        // at start, pool is empty
        std.debug.assert(self.visited_generation[self.medoid] != self.generation);

        const entry_point = self.medoid;
        const entry_sim = search_state.score(entry_point);

        // medoid is the first and only visited node
        self.visited_generation[entry_point] = self.generation;
        self.visited[0] = .{ .node = entry_point, .similarity = entry_sim };
        self.is_expanded[0] = false;
        self.nb_scored = 1;
        self.L = 1;
        self.is_search_done = false;
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
        const candidates = self.allocator.alloc(Candidate, self.n) catch @panic("OOM");
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
                self.greedySearchNode(current_node, graph_construction_search_budget);

                const nb_cand = self.L;
                const cands = self.visited[0..nb_cand];

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

    pub fn testNswExtention(self: *Graph, _: *sampling.Sampler) !void {
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
            std.log.info("Node {d} has in-degree {d}", .{ node, in_degrees[node] });
        }
        //try self.logDegreeCounts("Nodes by out-degree", self.nb_neighbors, graph_k_max);

        const hop_dist = try self.getHopDistance();
        defer self.allocator.free(hop_dist);
        var max_hops: u32 = 0;
        for (hop_dist) |hops| {
            if (hops > max_hops) max_hops = hops;
        }
        std.log.info("Max hops: {}", .{max_hops});
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
