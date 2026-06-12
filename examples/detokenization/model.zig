const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");

const log = std.log;

pub const Model_handler = struct {
    model: Model,
    shardings: main.Shardings,
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
        const shardings: main.Shardings = try .init(zml_handler.platform);

        std.log.info("Compile model", .{});
        var exes = try compileModel(zml_handler, model, shardings);
        errdefer exes.deinit(zml_handler.allocator);

        std.log.info("Load model buffers", .{});
        const shardings_arr = shardings.all();
        var model_buffers = try model.load(zml_handler, &store, &shardings_arr);
        errdefer Model.unloadBuffers(&model_buffers);

        return .{
            .model = model,
            .shardings = shardings,
            .exes = exes,
            .model_buffers = model_buffers,
        };
    }

    pub fn compileModel(zml_handler: *main.Zml_handler, model: Model, shardings: main.Shardings) !ModelExes {
        const shardings_arr = shardings.all();
        const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };
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

        const get_lm_head_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .get_lm_head,
            .{},
            opts,
        );
        errdefer get_lm_head_exe.deinit();

        const get_lm_head_args = try get_lm_head_exe.args(zml_handler.allocator);
        errdefer get_lm_head_args.deinit(zml_handler.allocator);

        const get_lm_head_results = try get_lm_head_exe.results(zml_handler.allocator);
        errdefer get_lm_head_results.deinit(zml_handler.allocator);

        const get_lm_head_normalized_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .get_lm_head_normalized,
            .{},
            opts,
        );
        errdefer get_lm_head_normalized_exe.deinit();

        const get_lm_head_normalized_args = try get_lm_head_normalized_exe.args(zml_handler.allocator);
        errdefer get_lm_head_normalized_args.deinit(zml_handler.allocator);

        const get_lm_head_normalized_results = try get_lm_head_normalized_exe.results(zml_handler.allocator);
        errdefer get_lm_head_normalized_results.deinit(zml_handler.allocator);

        const get_lm_head_row_norms_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .get_lm_head_row_norms,
            .{},
            opts,
        );
        errdefer get_lm_head_row_norms_exe.deinit();

        const get_lm_head_row_norms_args = try get_lm_head_row_norms_exe.args(zml_handler.allocator);
        errdefer get_lm_head_row_norms_args.deinit(zml_handler.allocator);

        const get_lm_head_row_norms_results = try get_lm_head_row_norms_exe.results(zml_handler.allocator);
        errdefer get_lm_head_row_norms_results.deinit(zml_handler.allocator);

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

        const find_junk_rows_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .findJunkRows,
            .{},
            opts,
        );
        errdefer find_junk_rows_exe.deinit();

        const find_junk_rows_args = try find_junk_rows_exe.args(zml_handler.allocator);
        errdefer find_junk_rows_args.deinit(zml_handler.allocator);

        const find_junk_rows_results = try find_junk_rows_exe.results(zml_handler.allocator);
        errdefer find_junk_rows_results.deinit(zml_handler.allocator);

        const analyze_top_rows_exe = try zml_handler.platform.compile(
            zml_handler.allocator,
            zml_handler.io,
            model,
            .analyze_top_rows,
            .{},
            opts,
        );
        errdefer analyze_top_rows_exe.deinit();

        const analyze_top_rows_args = try analyze_top_rows_exe.args(zml_handler.allocator);
        errdefer analyze_top_rows_args.deinit(zml_handler.allocator);

        const analyze_top_rows_results = try analyze_top_rows_exe.results(zml_handler.allocator);

        return .{
            .similarity_matrix_exe = similarity_matrix_exe,
            .similarity_matrix_args = similarity_matrix_args,
            .similarity_matrix_results = similarity_matrix_results,
            .get_lm_head_exe = get_lm_head_exe,
            .get_lm_head_args = get_lm_head_args,
            .get_lm_head_results = get_lm_head_results,
            .get_lm_head_normalized_exe = get_lm_head_normalized_exe,
            .get_lm_head_normalized_args = get_lm_head_normalized_args,
            .get_lm_head_normalized_results = get_lm_head_normalized_results,
            .get_lm_head_row_norms_exe = get_lm_head_row_norms_exe,
            .get_lm_head_row_norms_args = get_lm_head_row_norms_args,
            .get_lm_head_row_norms_results = get_lm_head_row_norms_results,
            .score_exe = score_exe,
            .score_args = score_args,
            .score_results = score_results,
            .find_junk_rows_exe = find_junk_rows_exe,
            .find_junk_rows_args = find_junk_rows_args,
            .find_junk_rows_results = find_junk_rows_results,
            .analyze_top_rows_exe = analyze_top_rows_exe,
            .analyze_top_rows_args = analyze_top_rows_args,
            .analyze_top_rows_results = analyze_top_rows_results,
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
    get_lm_head_exe: zml.Exe,
    get_lm_head_args: zml.Exe.Arguments,
    get_lm_head_results: zml.Exe.Results,
    get_lm_head_normalized_exe: zml.Exe,
    get_lm_head_normalized_args: zml.Exe.Arguments,
    get_lm_head_normalized_results: zml.Exe.Results,
    get_lm_head_row_norms_exe: zml.Exe,
    get_lm_head_row_norms_args: zml.Exe.Arguments,
    get_lm_head_row_norms_results: zml.Exe.Results,
    score_exe: zml.Exe,
    score_args: zml.Exe.Arguments,
    score_results: zml.Exe.Results,
    find_junk_rows_exe: zml.Exe,
    find_junk_rows_args: zml.Exe.Arguments,
    find_junk_rows_results: zml.Exe.Results,
    analyze_top_rows_exe: zml.Exe,
    analyze_top_rows_args: zml.Exe.Arguments,
    analyze_top_rows_results: zml.Exe.Results,

    pub fn deinit(self: ModelExes, allocator: std.mem.Allocator) void {
        self.similarity_matrix_exe.deinit();
        self.similarity_matrix_args.deinit(allocator);
        self.similarity_matrix_results.deinit(allocator);
        self.get_lm_head_exe.deinit();
        self.get_lm_head_args.deinit(allocator);
        self.get_lm_head_results.deinit(allocator);
        self.get_lm_head_normalized_exe.deinit();
        self.get_lm_head_normalized_args.deinit(allocator);
        self.get_lm_head_normalized_results.deinit(allocator);
        self.get_lm_head_row_norms_exe.deinit();
        self.get_lm_head_row_norms_args.deinit(allocator);
        self.get_lm_head_row_norms_results.deinit(allocator);
        self.score_exe.deinit();
        self.score_args.deinit(allocator);
        self.score_results.deinit(allocator);
        self.find_junk_rows_exe.deinit();
        self.find_junk_rows_args.deinit(allocator);
        self.find_junk_rows_results.deinit(allocator);
        self.analyze_top_rows_exe.deinit();
        self.analyze_top_rows_args.deinit(allocator);
        self.analyze_top_rows_results.deinit(allocator);
    }
};

pub const Model = struct {
    pub const row_batch_size: i64 = 2048;
    pub const row_k_neighbors: i64 = 256;
    pub const top_rows_count: i64 = 100;

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
            .{ .voc = .model, .d = .replicated },
        );
    }

    pub fn load(self: *const Model, zml_handler: *main.Zml_handler, store: *zml.io.TensorStore, shardings: []const zml.Sharding) !zml.Bufferized(Model) {
        var total_bytes: usize = 0;
        const now: std.Io.Timestamp = .now(zml_handler.io, .awake);
        defer {
            const took = now.untilNow(zml_handler.io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
            log.info("Loaded model [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }

        return zml.io.load(Model, self, zml_handler.allocator, zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
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

    pub fn get_lm_head(self: Model) zml.Tensor {
        return self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
    }

    pub fn get_lm_head_normalized(self: Model) zml.Tensor {
        return normalizeRows(self.get_lm_head());
    }

    pub fn get_lm_head_row_norms(self: Model) zml.Tensor {
        const lm_head = self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
        return lm_head.mul(lm_head).sum(.d).squeeze(.d).sqrt();
    }

    pub fn scoreTokens(self: Model, embedding: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        const lm_head = self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
        const normalized_lm_head = normalizeRows(lm_head);
        const normalized_embedding = normalizeRows(embedding);
        const logits = lm_head.dot(embedding, .d).squeeze(.s);
        const similarities = normalized_lm_head.dot(normalized_embedding, .d).squeeze(.s);
        const sorted = logits.softmax(.voc).sort(.voc, .{ .descending = true });
        const sorted_similarity_indices = sorted.indices.rename(.{ .voc = .rank });
        const sorted_similarities = similarities.gather(.{ .voc = sorted_similarity_indices }, .{}).rename(.{ .rank = .voc });
        return .{ sorted.values, sorted.indices, sorted_similarities };
    }

    pub fn analyze_top_rows(self: Model) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        const lm_head = self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
        const row_norms = lm_head.mul(lm_head).sum(.d).squeeze(.d).sqrt();
        const top_norm_rows = row_norms.topK(.{ .top_norm = .voc }, top_rows_count, .{ .descending = true });
        const top_rows = lm_head.gather(.{ .voc = top_norm_rows.indices }, .{});

        const zero = zml.Tensor.scalar(@as(u32, 0), .u32);
        const highest_norm_row = top_rows.dynamicSlice1d(top_rows.axis(.top_norm), .{ .start = zero, .len = 1 }).squeeze(.top_norm);
        const highest_norm_row_scores = lm_head.dot(highest_norm_row, .d);
        const highest_norm_row_top = highest_norm_row_scores.topK(.{ .top_dot = .voc }, top_rows_count, .{ .descending = true });

        const average_top_row = top_rows.mean(.top_norm).squeeze(.top_norm);
        const average_top_row_scores = lm_head.dot(average_top_row, .d);
        const average_top_row_top = average_top_row_scores.topK(.{ .top_avg_dot = .voc }, top_rows_count, .{ .descending = true });

        const smallest_norm_rows = row_norms.topK(.{ .junk = .voc }, top_rows_count, .{ .descending = false });
        const junk_rows = lm_head.gather(.{ .voc = smallest_norm_rows.indices }, .{});
        const junk_direction = normalizeVector(junk_rows.mean(.junk).squeeze(.junk));
        const junk_direction_scores = normalizeRows(lm_head).dot(junk_direction, .d);
        const anti_junk_top = junk_direction_scores.topK(.{ .anti_junk = .voc }, top_rows_count, .{ .descending = false });

        return .{
            top_norm_rows.values,
            top_norm_rows.indices.convert(.u64),
            highest_norm_row_top.values,
            highest_norm_row_top.indices.convert(.u64),
            average_top_row_top.values,
            average_top_row_top.indices.convert(.u64),
            anti_junk_top.values,
            anti_junk_top.indices.convert(.u64),
        };
    }

    pub fn findJunkRows(self: Model) zml.Tensor {
        const junk_seed_rows = 100;
        const lm_head = self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
        const row_norm2 = lm_head.mul(lm_head).sum(.d).squeeze(.d);
        const smallest_norm_rows = row_norm2.topK(.{ .junk = .voc }, junk_seed_rows, .{ .descending = false });
        const rows = lm_head.gather(.{ .voc = smallest_norm_rows.indices }, .{});
        const junk_direction = normalizeVector(rows.mean(.junk).squeeze(.junk));
        const similarity = normalizeRows(lm_head).dot(junk_direction, .d);
        const is_junk = similarity.cmp(.GT, zml.Tensor.scalar(0.75, .f32));
        const row_ids = zml.Tensor.iota(similarity.shape(), .voc).convert(.u64);
        const sentinel = zml.Tensor.scalar(@as(u64, @intCast(lm_head.dim(.voc))), .u64);
        return is_junk.select(row_ids, sentinel);
    }

    pub fn similarityMatrix(self: Model, row_start: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const lm_head = self.lm_head.withTags(.{ .voc, .d }).convert(.f32);
        const batch_slice: zml.Tensor.DynSlice = .{ .start = row_start, .len = row_batch_size };
        const normalized = normalizeRows(lm_head);

        const rows = normalized.rename(.{ .voc = .row }).dynamicSlice1d(lm_head.axis(.voc), batch_slice);
        const cols = normalized.rename(.{ .voc = .col });
        const similarity = rows.dot(cols, .d);

        // penalize self-similarity by setting -1.0 on the diagonal
        const row_ids = zml.Tensor.iota(similarity.shape(), .row).add(row_start.convert(.i32));
        const col_ids = zml.Tensor.iota(similarity.shape(), .col);
        const self_mask = row_ids.cmp(.EQ, col_ids);
        const minus_one = zml.Tensor.scalar(-1.0, .f32).broad(similarity.shape());
        const similarity_for_sort = self_mask.select(minus_one, similarity);

        const nearest = similarity_for_sort.topK(.{ .nearest = .col }, row_k_neighbors, .{ .descending = true }).indices.convert(.u64);

        return .{ similarity, nearest };
    }

    fn normalizeRows(lm_head: zml.Tensor) zml.Tensor {
        const squared_norm = lm_head.mul(lm_head).sum(.d);
        const inv_norm = squared_norm.rsqrt();
        return lm_head.mul(inv_norm.broad(lm_head.shape()));
    }

    fn normalizeVector(vector: zml.Tensor) zml.Tensor {
        const squared_norm = vector.mul(vector).sum(.d);
        const inv_norm = squared_norm.rsqrt();
        return vector.mul(inv_norm.broad(vector.shape()));
    }
};
