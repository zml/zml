const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");

const log = std.log;

pub const Model_handler = struct {
    model: Model,
    exes: ModelExes,
    model_buffers: zml.Bufferized(Model),

    pub fn init(zml_handler: *main.Zml_handler) !Model_handler {
        std.log.info("Init store", .{});
        const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
        var registry: zml.safetensors.TensorRegistry = try .fromRepo(zml_handler.allocator, zml_handler.io, repo);
        defer registry.deinit();

        var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
        defer store.deinit();

        std.log.info("Initialize model", .{});
        const model: Model = .init(store.view());

        std.log.info("Compile model", .{});
        var exes = try compileModel(zml_handler, model);
        errdefer exes.deinit(zml_handler.allocator);

        std.log.info("Load model buffers", .{});
        var model_buffers = try model.load(zml_handler, &store);
        errdefer Model.unloadBuffers(&model_buffers);

        return .{
            .model = model,
            .exes = exes,
            .model_buffers = model_buffers,
        };
    }

    pub fn compileModel(zml_handler: *main.Zml_handler, model: Model) !ModelExes {
        const opts: zml.module.CompilationOptions = .{};
        const similarity_matrix_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .similarityMatrix,
            .{.init(.{}, .u32)},
            opts,
        );
        errdefer similarity_matrix_exe.deinit();

        const similarity_matrix_args = try similarity_matrix_exe.args(zml_handler.allocator);
        errdefer similarity_matrix_args.deinit(zml_handler.allocator);

        const similarity_matrix_results = try similarity_matrix_exe.results(zml_handler.allocator);
        errdefer similarity_matrix_results.deinit(zml_handler.allocator);

        const similarity_matrix_normalized_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .similarityMatrixNormalized,
            .{.init(.{}, .u32)},
            opts,
        );
        errdefer similarity_matrix_normalized_exe.deinit();

        const similarity_matrix_normalized_args = try similarity_matrix_normalized_exe.args(zml_handler.allocator);
        errdefer similarity_matrix_normalized_args.deinit(zml_handler.allocator);

        const similarity_matrix_normalized_results = try similarity_matrix_normalized_exe.results(zml_handler.allocator);
        errdefer similarity_matrix_normalized_results.deinit(zml_handler.allocator);

        const rotated_lm_head_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .rotatedLmHead,
            .{.init(.{ .d = model.shape().dim(.d), .eig = model.shape().dim(.d) }, .f32)},
            opts,
        );
        errdefer rotated_lm_head_exe.deinit();

        const rotated_lm_head_args = try rotated_lm_head_exe.args(zml_handler.allocator);
        errdefer rotated_lm_head_args.deinit(zml_handler.allocator);

        const rotated_lm_head_results = try rotated_lm_head_exe.results(zml_handler.allocator);
        errdefer rotated_lm_head_results.deinit(zml_handler.allocator);

        const sort_by_first_row_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .sortByFirstRow,
            .{},
            opts,
        );
        errdefer sort_by_first_row_exe.deinit();

        const sort_by_first_row_args = try sort_by_first_row_exe.args(zml_handler.allocator);
        errdefer sort_by_first_row_args.deinit(zml_handler.allocator);

        const sort_by_first_row_results = try sort_by_first_row_exe.results(zml_handler.allocator);
        errdefer sort_by_first_row_results.deinit(zml_handler.allocator);

        const score_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .scoreTokens,
            .{.init(.{ .s = 1, .d = model.shape().dim(.d) }, .f32)},
            opts,
        );
        errdefer score_exe.deinit();

        const score_args = try score_exe.args(zml_handler.allocator);
        errdefer score_args.deinit(zml_handler.allocator);

        const score_results = try score_exe.results(zml_handler.allocator);
        errdefer score_results.deinit(zml_handler.allocator);

        const top1_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .top1Token,
            .{.init(.{ .s = 1, .d = model.shape().dim(.d) }, .f32)},
            opts,
        );
        errdefer top1_exe.deinit();

        const top1_args = try top1_exe.args(zml_handler.allocator);
        errdefer top1_args.deinit(zml_handler.allocator);

        const top1_results = try top1_exe.results(zml_handler.allocator);
        errdefer top1_results.deinit(zml_handler.allocator);

        return .{
            .similarity_matrix_exe = similarity_matrix_exe,
            .similarity_matrix_args = similarity_matrix_args,
            .similarity_matrix_results = similarity_matrix_results,
            .similarity_matrix_normalized_exe = similarity_matrix_normalized_exe,
            .similarity_matrix_normalized_args = similarity_matrix_normalized_args,
            .similarity_matrix_normalized_results = similarity_matrix_normalized_results,
            .rotated_lm_head_exe = rotated_lm_head_exe,
            .rotated_lm_head_args = rotated_lm_head_args,
            .rotated_lm_head_results = rotated_lm_head_results,
            .sort_by_first_row_exe = sort_by_first_row_exe,
            .sort_by_first_row_args = sort_by_first_row_args,
            .sort_by_first_row_results = sort_by_first_row_results,
            .score_exe = score_exe,
            .score_args = score_args,
            .score_results = score_results,
            .top1_exe = top1_exe,
            .top1_args = top1_args,
            .top1_results = top1_results,
        };
    }

    pub fn unloadBuffers(self: *Model_handler) void {
        Model.unloadBuffers(&self.model_buffers);
    }

    pub fn deinit(self: *Model_handler, allocator: std.mem.Allocator) void {
        self.exes.deinit(allocator);
    }
};

pub const ModelExes = struct {
    similarity_matrix_exe: zml.Exe,
    similarity_matrix_args: zml.Exe.Arguments,
    similarity_matrix_results: zml.Exe.Results,
    
    similarity_matrix_normalized_exe: zml.Exe,
    similarity_matrix_normalized_args: zml.Exe.Arguments,
    similarity_matrix_normalized_results: zml.Exe.Results,
    
    rotated_lm_head_exe: zml.Exe,
    rotated_lm_head_args: zml.Exe.Arguments,
    rotated_lm_head_results: zml.Exe.Results,
    
    sort_by_first_row_exe: zml.Exe,
    sort_by_first_row_args: zml.Exe.Arguments,
    sort_by_first_row_results: zml.Exe.Results,
    
    score_exe: zml.Exe,
    score_args: zml.Exe.Arguments,
    score_results: zml.Exe.Results,
    
    top1_exe: zml.Exe,
    top1_args: zml.Exe.Arguments,
    top1_results: zml.Exe.Results,

    pub fn deinit(self: ModelExes, allocator: std.mem.Allocator) void {
        self.similarity_matrix_exe.deinit();
        self.similarity_matrix_args.deinit(allocator);
        self.similarity_matrix_results.deinit(allocator);
        self.similarity_matrix_normalized_exe.deinit();
        self.similarity_matrix_normalized_args.deinit(allocator);
        self.similarity_matrix_normalized_results.deinit(allocator);
        self.rotated_lm_head_exe.deinit();
        self.rotated_lm_head_args.deinit(allocator);
        self.rotated_lm_head_results.deinit(allocator);
        self.sort_by_first_row_exe.deinit();
        self.sort_by_first_row_args.deinit(allocator);
        self.sort_by_first_row_results.deinit(allocator);
        self.score_exe.deinit();
        self.score_args.deinit(allocator);
        self.score_results.deinit(allocator);
        self.top1_exe.deinit();
        self.top1_args.deinit(allocator);
        self.top1_results.deinit(allocator);
    }
};

pub const Model = struct {
    pub const row_batch_size: i64 = 4096;
    pub const row_k_neighbors: i64 = 256;
    pub const top_rows_count: i64 = 100;
    pub const score_top_k: i64 = 16;

    lm_head: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Model {
        const lm_head = blk: {
            if (store.hasKey("lm_head.weight")) {
                break :blk createLmHeadTensor(store, "lm_head");
            }
            if (store.hasKey("model.embed_tokens.weight")) {
                break :blk createLmHeadTensor(store, "model.embed_tokens");
            }
            if (store.hasKey("model.language_model.embed_tokens.weight")) {
                break :blk createLmHeadTensor(store, "model.language_model.embed_tokens");
            }
            break :blk createLmHeadTensor(store, "embed_tokens");
        };

        return .{ .lm_head = lm_head };
    }

    fn createLmHeadTensor(store: zml.io.TensorStore.View, prefix: []const u8) zml.Tensor {
        return store.withPrefix(prefix).createTensor(
            "weight",
            .{ .voc, .d },
            .{ .voc = .replicated, .d = .replicated },
        );
    }

    pub fn load(self: *const Model, zml_handler: *main.Zml_handler, store: *zml.io.TensorStore) !zml.Bufferized(Model) {
        var total_bytes: usize = 0;
        const now: std.Io.Timestamp = .now(zml_handler.io, .awake);
        defer {
            const took = now.untilNow(zml_handler.io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
            log.info("Loaded model [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }

        return zml.io.load(Model, self, zml_handler.allocator, zml_handler.io, zml_handler.platform, store, .{
            .parallelism = 1,
            .dma_chunks = 8,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = null,
            .total_bytes = &total_bytes,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model)) void {
        self.lm_head.deinit();
    }

    pub fn shape(self: Model) zml.Shape {
        return self.lm_head.shape();
    }

    pub fn rotatedLmHead(self: Model, u: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const lm_head = self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
        const u_rot = u.withTags(.{ .d, .eig }).convert(.f32);
        const m = lm_head.dot(u_rot, .d).rename(.{ .eig = .d });
        return .{ m, m.transpose(.{ .d, .voc }) };
    }

    pub fn sortByFirstRow(self: Model) zml.Tensor {
        const lm_head = self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
        const zero = zml.Tensor.scalar(@as(u32, 0), .u32);
        const first_row = lm_head.dynamicSlice1d(lm_head.axis(.voc), .{ .start = zero, .len = 1 }).squeeze(.voc);
        const scores = lm_head.dot(first_row, .d);
        return scores.sort(.voc, .{ .descending = false }).indices.convert(.u64);
    }

    pub fn scoreTokens(self: Model, embedding: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const lm_head = self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
        const logits = lm_head.dot(embedding, .d).squeeze(.s);
        const sorted = logits.softmax(.voc).sort(.voc, .{ .descending = true });
        const zero = zml.Tensor.scalar(@as(u32, 0), .u32);
        const top_probas = sorted.values.dynamicSlice1d(sorted.values.axis(.voc), .{ .start = zero, .len = score_top_k }).rename(.{ .voc = .rank });
        const top_indices = sorted.indices.dynamicSlice1d(sorted.indices.axis(.voc), .{ .start = zero, .len = score_top_k }).rename(.{ .voc = .rank }).convert(.u32);
        return .{ top_probas, top_indices };
    }

    pub fn top1Token(self: Model, embedding: zml.Tensor) zml.Tensor {
        const lm_head = self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
        const scores = lm_head.dot(embedding, .d).squeeze(.s);
        return scores.argMax(.voc).indices.convert(.u32);
    }

    
    pub fn similarityMatrix(self: Model, row_start: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const lm_head = self.lm_head.withTags(.{ .voc, .d });
        return similarityMatrixForRows(lm_head, row_start);
    }

    pub fn similarityMatrixNormalized(self: Model, row_start: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const lm_head = self.lm_head.withTags(.{ .voc, .d });
        const normalized = normalizeRows(lm_head);
        return similarityMatrixForRows(normalized, row_start);
    }

    fn similarityMatrixForRows(lm_head: zml.Tensor, row_start: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const batch_slice: zml.Tensor.DynSlice = .{ .start = row_start, .len = row_batch_size };

        const rows = lm_head.rename(.{ .voc = .row }).dynamicSlice1d(lm_head.axis(.voc), batch_slice);
        const cols = lm_head.rename(.{ .voc = .col });
        const similarity = rows.dot(cols, .d);

        // Exclude self-similarity from nearest-neighbor sorting.
        const row_ids = zml.Tensor.iota(similarity.shape(), .row).add(row_start.convert(.i32));
        const col_ids = zml.Tensor.iota(similarity.shape(), .col);
        const self_mask = row_ids.cmp(.EQ, col_ids);
        const minus_inf = zml.Tensor.scalar(-std.math.inf(f32), .bf16).broad(similarity.shape());
        const similarity_for_sort = self_mask.select(minus_inf, similarity);

        const nearest = similarity_for_sort.topK(.{ .nearest = .col }, row_k_neighbors, .{ .descending = true }).indices.convert(.i32);

        return .{ similarity, nearest };
    }

    fn normalizeRows(lm_head: zml.Tensor) zml.Tensor {
        const squared_norm = lm_head.mul(lm_head).sum(.d);
        const inv_norm = squared_norm.rsqrt();
        return lm_head.mul(inv_norm.broad(lm_head.shape()));
    }

};
