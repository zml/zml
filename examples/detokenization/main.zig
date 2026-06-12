const std = @import("std");
const zml = @import("zml");

const stdx = zml.stdx;
const log = std.log;

const graph = @import("graph.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const inference = @import("inference.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Zml_handler = struct {
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    platform: *zml.Platform,
    uris: Uri_handler,
    io: std.Io,
    local_io: std.Io,
    progress: std.Progress.Node,
    args: Args,
    timers: Timing_handler,
    mem: MemoryChecker,

    pub fn fromInit(init: std.process.Init, io: std.Io) !Zml_handler {
        if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
            var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
            defer working_dir.close(init.io);
            try std.process.setCurrentDir(init.io, working_dir);
        }
        const platform = try zml.Platform.auto(init.gpa, io, .{});
        errdefer platform.deinit(init.gpa);

        const args = stdx.flags.parse(init.minimal.args, Args);

        return .{
            .allocator = init.gpa,
            .arena = init.arena,
            .platform = platform,
            .uris = .init(),
            .io = io,
            .local_io = init.io,
            .progress = std.Progress.start(io, .{}),
            .args = args,
            .timers = .{},
            .mem = .{ .platform = platform },
        };
    }

    pub fn deinit(self: *Zml_handler) void {
        self.progress.end();
        self.platform.deinit(self.allocator, self.io);
    }

    pub fn tic(self: *Zml_handler, target: *Timing_handler.Field_timer) void {
        target.init = std.Io.Timestamp.now(self.io, .awake);
    }

    pub fn toc(self: *Zml_handler, target: *Timing_handler.Field_timer) void {
        const end = std.Io.Timestamp.now(self.io, .awake);
        const start: std.Io.Timestamp = target.init;
        const duration = std.Io.Timestamp.durationTo(start, end);
        target.nanoseconds += duration.nanoseconds;
    }
};

pub const Uri_handler = struct {
    llama: []const u8,
    qwen: []const u8,

    pub fn init() Uri_handler {
        return .{
            .llama = "file://examples//detokenization//models//llama",
            .qwen = "file://examples//detokenization//models//qwen",
        };
    }
};

pub const Shardings = struct {
    model: zml.Sharding,
    experts: zml.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        return .{
            .model = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth })),
            .experts = try platform.registerSharding("experts", .mesh(.{ .experts = .high_bandwidth })),
        };
    }

    pub fn all(self: Shardings) [2]zml.Sharding {
        return .{ self.model, self.experts };
    }
};

pub const Timing_handler = struct {
    pub const Field_timer = struct {
        nanoseconds: i96 = 0,
        init: std.Io.Timestamp = std.Io.Timestamp.zero,
    };

    similarity_matrix: Field_timer = .{},
    junk_rows: Field_timer = .{},

    knn_graph: Field_timer = .{},
    nsw_graph: Field_timer = .{},

    greedy_search: Field_timer = .{},
    prune_pool: Field_timer = .{},

    prefill: Field_timer = .{},
    decode: Field_timer = .{},

    pub fn print(self: Timing_handler) void {
        std.log.info("Sim matrix    : {d:>6.2}s", .{@as(f64, @floatFromInt(self.similarity_matrix.nanoseconds)) / 1e9});
        std.log.info("Junk rows     : {d:>6.2}s", .{@as(f64, @floatFromInt(self.junk_rows.nanoseconds)) / 1e9});
        std.log.info("kNN graph ini : {d:>6.2}s", .{@as(f64, @floatFromInt(self.knn_graph.nanoseconds)) / 1e9});
        std.log.info("NSW graph ext : {d:>6.2}s", .{@as(f64, @floatFromInt(self.nsw_graph.nanoseconds)) / 1e9});
        std.log.info("Greedy search : {d:>6.2}s", .{@as(f64, @floatFromInt(self.greedy_search.nanoseconds)) / 1e9});
        std.log.info("Pruning pool  : {d:>6.2}s", .{@as(f64, @floatFromInt(self.prune_pool.nanoseconds)) / 1e9});
        std.log.info("LLM prefill   : {d:>6.2}s", .{@as(f64, @floatFromInt(self.prefill.nanoseconds)) / 1e9});
        std.log.info("LLM decode    : {d:>6.2}s", .{@as(f64, @floatFromInt(self.decode.nanoseconds)) / 1e9});
    }
};

pub const MemoryChecker = struct {
    platform: *zml.Platform,
    bytes_before: i64 = 0,
    bytes_after: i64 = 0,

    pub fn start(self: *MemoryChecker, id: usize) void {
        self.bytes_before = @intCast(self.platform.devices[id].memoryStats().bytes_in_use);
    }

    pub fn check(self: *MemoryChecker, id: usize) void {
        self.bytes_after = @intCast(self.platform.devices[id].memoryStats().bytes_in_use);
        const leaked = @abs(self.bytes_after - self.bytes_before);
        if (leaked != 0) {
            std.log.info("memory usage: before={d} after={d} leaked={d}", .{ self.bytes_before, self.bytes_after, leaked });
        }
    }
};

pub const SimilarityMatrix = struct {
    data: zml.Slice,
    nearest_neighbors: zml.Slice,
    row_offsets: []usize,
    n: usize,
    d: usize,
    k: usize,

    pub fn dist(self: *SimilarityMatrix, i: usize, j: usize) f32 {
        if (i == j) return 1.0; // TODO: can prob remove this
        const row = @min(i, j);
        const col = @max(i, j);
        return self.data.constItems(f32)[self.row_offsets[row] + col - row - 1];
    }

    pub fn initOffsets(self: *SimilarityMatrix) void {
        var offset: usize = 0;
        for (0..self.n) |row| {
            self.row_offsets[row] = offset;
            offset += self.n - row - 1;
        }
    }

    pub fn nearestNeighbor(self: *SimilarityMatrix, row: usize, pos: usize) usize {
        return self.nearest_neighbors.constItems(usize)[row * self.k + pos];
    }

    pub fn deinit(self: *SimilarityMatrix, allocator: std.mem.Allocator) void {
        self.data.free(allocator);
        self.nearest_neighbors.free(allocator);
        allocator.free(self.row_offsets);
    }
};

const Args = struct {
    pub const help =
        \\ Use detokenization [options]
        \\
    ;
};

pub fn printZmlLogo(io: std.Io) !void {
    const LOGO =
        \\
        \\
        \\ ███████╗███╗   ███╗██╗
        \\ ╚══███╔╝████╗ ████║██║
        \\   ███╔╝ ██╔████╔██║██║
        \\  ███╔╝  ██║╚██╔╝██║██║  .ai
        \\ ███████╗██║ ╚═╝ ██║███████╗
        \\ ╚══════╝╚═╝     ╚═╝╚══════╝
        \\
        \\
        \\
    ;
    var writer = std.Io.File.stdout().writer(io, &.{});
    try writer.interface.writeAll(LOGO);
    try writer.interface.flush();
}

pub fn main(init: std.process.Init) !void {
    var http_client: std.http.Client = .{ .allocator = init.gpa, .io = init.io };
    defer http_client.deinit();

    var vfs_file: zml.io.VFS.File = .init(init.gpa, init.io, .{});
    defer vfs_file.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(init.gpa, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(init.gpa, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    var zml_handler: Zml_handler = try .fromInit(init, io);
    defer zml_handler.deinit();

    try printZmlLogo(zml_handler.io);

    //try runLlm(&zml_handler);
    try runTests(&zml_handler);

    zml_handler.timers.print();
}


pub fn runLlm(zml_handler: *Zml_handler) !void {
    var llm = try llm_.Llm_handler.init(zml_handler);
    defer llm.deinit(zml_handler.allocator);

    const inspi_tokens = try inference.tokenizePrompt(zml_handler, llm.tokenizer);
    defer zml_handler.allocator.free(inspi_tokens);

    zml_handler.mem.start(0);
    const inspi_result = try inference.generateText(zml_handler, &llm, inspi_tokens);
    defer zml_handler.allocator.free(inspi_result);
    zml_handler.mem.check(0);
}

pub fn runTests(zml_handler: *Zml_handler) !void {
    var model_handler = try model_.Model_handler.init(zml_handler);
    defer model_handler.deinit(zml_handler.allocator);
    defer model_handler.unloadBuffers();

    std.log.info("aaa", .{});
    try analyzeTopRows(zml_handler, &model_handler);

    zml_handler.tic(&zml_handler.timers.similarity_matrix);
    var similarity_matrix = try computeSimilarityMatrix(zml_handler, &model_handler);
    defer similarity_matrix.deinit(zml_handler.allocator);
    zml_handler.toc(&zml_handler.timers.similarity_matrix);

    //try testSimilarityMatrix(zml_handler, &model_handler, &similarity_matrix);

    std.log.info("Get lm_head", .{});
    const lm_head = try getLmHead(zml_handler, &model_handler);
    defer lm_head.free(zml_handler.allocator);

    std.log.info("Get lm_head_normalized", .{});
    const lm_head_normalized = try getLmHeadNormalized(zml_handler, &model_handler);
    defer lm_head_normalized.free(zml_handler.allocator);

    std.log.info("Get lm_head row norms", .{});
    const lm_head_row_norms = try getLmHeadRowNorms(zml_handler, &model_handler);
    defer lm_head_row_norms.free(zml_handler.allocator);

    std.log.info("Get junk rows", .{});
    zml_handler.tic(&zml_handler.timers.junk_rows);
    const junk_rows = try getJunkRows(zml_handler, &model_handler);
    defer zml_handler.allocator.free(junk_rows);
    zml_handler.toc(&zml_handler.timers.junk_rows);
    std.log.info("Found {d} junk rows", .{junk_rows.len});

    std.log.info("Init graph", .{});
    const graph_params: graph.GraphParams = .{};
    var g: graph.Graph = try .init(zml_handler, lm_head, lm_head_normalized, &similarity_matrix, lm_head_row_norms, junk_rows, graph_params);
    defer g.deinit();

    zml_handler.tic(&zml_handler.timers.knn_graph);
    g.setRandomNeighbors();
    zml_handler.toc(&zml_handler.timers.knn_graph);

    std.log.info("Exact kNN : nb edges: {d}", .{g.nbEdges()});

    zml_handler.tic(&zml_handler.timers.nsw_graph);
    g.extendToNsw();
    zml_handler.toc(&zml_handler.timers.nsw_graph);

    std.log.info("NSW extension : nb edges: {d}", .{g.nbEdges()});

    var llm = try llm_.Llm_handler.init(zml_handler);
    defer llm.deinit(zml_handler.allocator);

    const inspi_tokens = try inference.tokenizePrompt(zml_handler, llm.tokenizer);
    defer zml_handler.allocator.free(inspi_tokens);

    zml_handler.mem.start(0);
    const inspi_result = try inference.generateTextGraph(zml_handler, &llm, &model_handler, &g, inspi_tokens);
    defer zml_handler.allocator.free(inspi_result);
    zml_handler.mem.check(0);
}

pub fn analyzeTopRows(zml_handler: *Zml_handler, model_handler: *model_.Model_handler) !void {
    std.log.info("Analyze top lm_head rows", .{});

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
    var tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    model_handler.exes.analyze_top_rows_args.set(.{model_handler.model_buffers});
    model_handler.exes.analyze_top_rows_exe.call(model_handler.exes.analyze_top_rows_args, &model_handler.exes.analyze_top_rows_results);

    var top_norm_values_buffer: zml.Buffer = undefined;
    var top_norm_indices_buffer: zml.Buffer = undefined;
    var top_dot_values_buffer: zml.Buffer = undefined;
    var top_dot_indices_buffer: zml.Buffer = undefined;
    var top_avg_dot_values_buffer: zml.Buffer = undefined;
    var top_avg_dot_indices_buffer: zml.Buffer = undefined;
    model_handler.exes.analyze_top_rows_results.fill(.{
        &top_norm_values_buffer,
        &top_norm_indices_buffer,
        &top_dot_values_buffer,
        &top_dot_indices_buffer,
        &top_avg_dot_values_buffer,
        &top_avg_dot_indices_buffer,
    });
    defer top_norm_values_buffer.deinit();
    defer top_norm_indices_buffer.deinit();
    defer top_dot_values_buffer.deinit();
    defer top_dot_indices_buffer.deinit();
    defer top_avg_dot_values_buffer.deinit();
    defer top_avg_dot_indices_buffer.deinit();

    const top_norm_values_slice = try top_norm_values_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_norm_values_slice.free(zml_handler.allocator);
    const top_norm_indices_slice = try top_norm_indices_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_norm_indices_slice.free(zml_handler.allocator);
    const top_dot_values_slice = try top_dot_values_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_dot_values_slice.free(zml_handler.allocator);
    const top_dot_indices_slice = try top_dot_indices_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_dot_indices_slice.free(zml_handler.allocator);
    const top_avg_dot_values_slice = try top_avg_dot_values_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_avg_dot_values_slice.free(zml_handler.allocator);
    const top_avg_dot_indices_slice = try top_avg_dot_indices_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_avg_dot_indices_slice.free(zml_handler.allocator);

    try printTopRowsSection(
        tokenizer,
        "Top 100 highest-norm lm_head rows",
        "row_norm",
        top_norm_indices_slice.constItems(u64),
        top_norm_values_slice.constItems(f32),
    );
    try printTopRowsSection(
        tokenizer,
        "Top 100 dot products with the highest-norm row",
        "dot",
        top_dot_indices_slice.constItems(u64),
        top_dot_values_slice.constItems(f32),
    );
    try printTopRowsSection(
        tokenizer,
        "Top 100 dot products with the average of top 100 highest-norm rows",
        "dot",
        top_avg_dot_indices_slice.constItems(u64),
        top_avg_dot_values_slice.constItems(f32),
    );
}

fn printTopRowsSection(tokenizer: zml.tokenizer.Tokenizer, title: []const u8, value_label: []const u8, indices: []const u64, values: []const f32) !void {
    std.debug.assert(indices.len == values.len);
    log.info("{s}", .{title});
    log.info("{s:>6}  {s:>10}  {s:>14}  {s}", .{ "rank", "token_id", value_label, "token" });
    log.info("{s:>6}  {s:>10}  {s:>14}  {s}", .{ "------", "----------", "--------------", "-----" });
    for (indices, values, 0..) |token_id, value, i| {
        var decoded_buf: [512]u8 = undefined;
        const decoded = try decodeToken(tokenizer, @intCast(token_id), &decoded_buf);
        var escaped_buf: [512]u8 = undefined;
        const escaped = escapeTokenText(decoded, &escaped_buf);
        log.info("{d:>6}  {d:>10}  {d:>14.6}  {s}", .{ i + 1, token_id, value, escaped });
    }
}

fn decodeToken(tokenizer: zml.tokenizer.Tokenizer, token_id: u32, out: []u8) ![]const u8 {
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

pub fn computeSimilarityMatrix(zml_handler: *Zml_handler, model_handler: *model_.Model_handler) !SimilarityMatrix {
    std.log.info("Compute similarity matrix", .{});
    const allocator = zml_handler.allocator;

    const lm_head_shape = model_handler.model.shape();
    const n: usize = @intCast(lm_head_shape.dim(.voc));
    const d: usize = @intCast(lm_head_shape.dim(.d));
    std.log.info("lm_head shape: {d} x {d}", .{ n, d });
    const triangular_len: usize = @divExact(n * (n - 1), 2);
    const batch_size: usize = @intCast(model_.Model.row_batch_size);
    const k: usize = @intCast(model_.Model.row_k_neighbors);

    const similarity_matrix_slice: zml.Slice = try .alloc(allocator, .init(.{ .data = triangular_len }, .f32));
    errdefer similarity_matrix_slice.free(allocator);

    const nearest_neighbors_slice: zml.Slice = try .alloc(allocator, .init(.{ .data = n * k }, .u64));
    errdefer nearest_neighbors_slice.free(allocator);

    const similarity_batch_slice: zml.Slice = try .alloc(allocator, .init(.{ .row = batch_size, .col = n }, .f32));
    defer similarity_batch_slice.free(allocator);

    const nearest_batch_slice: zml.Slice = try .alloc(allocator, .init(.{ .row = batch_size, .nearest = k }, .u64));
    defer nearest_batch_slice.free(allocator);

    var write_offset: usize = 0;
    var row_start: usize = 0;
    while (row_start < n) : (row_start += batch_size) {
        std.log.info("batch {d}/{d}", .{ @divExact(row_start, batch_size) + 1, @divFloor(n, batch_size) + 1 });
        const rows_to_copy = @min(batch_size, n - row_start);
        const kernel_row_start = if (row_start + batch_size <= n) row_start else n - batch_size;

        var row_start_buffer = try zml.Buffer.scalar(zml_handler.io, zml_handler.platform, @as(u32, @intCast(kernel_row_start)), .u32);
        defer row_start_buffer.deinit();

        model_handler.exes.similarity_matrix_args.set(.{ model_handler.model_buffers, row_start_buffer });
        model_handler.exes.similarity_matrix_exe.call(model_handler.exes.similarity_matrix_args, &model_handler.exes.similarity_matrix_results);
        var similarity_batch_buffer: zml.Buffer = undefined;
        var nearest_batch_buffer: zml.Buffer = undefined;
        model_handler.exes.similarity_matrix_results.fill(.{ &similarity_batch_buffer, &nearest_batch_buffer });
        defer similarity_batch_buffer.deinit();
        defer nearest_batch_buffer.deinit();

        try similarity_batch_buffer.toSlice(zml_handler.io, similarity_batch_slice);
        try nearest_batch_buffer.toSlice(zml_handler.io, nearest_batch_slice);

        const batch_items = similarity_batch_slice.items(f32);
        const output_items = similarity_matrix_slice.items(f32);
        const nearest_batch_items = nearest_batch_slice.items(usize);
        const nearest_items = nearest_neighbors_slice.items(usize);
        for (0..rows_to_copy) |local_row| {
            const global_row = row_start + local_row;
            const kernel_local_row = global_row - kernel_row_start;
            const len = n - global_row - 1;
            const src_start = kernel_local_row * n + global_row + 1;
            @memcpy(output_items[write_offset..][0..len], batch_items[src_start..][0..len]);
            write_offset += len;
            @memcpy(nearest_items[global_row * k ..][0..k], nearest_batch_items[kernel_local_row * k ..][0..k]);
        }
    }
    std.debug.assert(write_offset == triangular_len);

    var matrix: SimilarityMatrix = .{
        .data = similarity_matrix_slice,
        .nearest_neighbors = nearest_neighbors_slice,
        .row_offsets = try allocator.alloc(usize, n),
        .n = n,
        .d = d,
        .k = k,
    };
    matrix.initOffsets();
    return matrix;
}

pub fn getLmHead(zml_handler: *Zml_handler, model_handler: *model_.Model_handler) !zml.Slice {
    model_handler.exes.get_lm_head_args.set(.{model_handler.model_buffers});
    model_handler.exes.get_lm_head_exe.call(model_handler.exes.get_lm_head_args, &model_handler.exes.get_lm_head_results);
    var lm_head_buffer = model_handler.exes.get_lm_head_results.get(zml.Buffer);
    defer lm_head_buffer.deinit();
    return lm_head_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
}

pub fn getLmHeadNormalized(zml_handler: *Zml_handler, model_handler: *model_.Model_handler) !zml.Slice {
    model_handler.exes.get_lm_head_normalized_args.set(.{model_handler.model_buffers});
    model_handler.exes.get_lm_head_normalized_exe.call(model_handler.exes.get_lm_head_normalized_args, &model_handler.exes.get_lm_head_normalized_results);
    var lm_head_normalized_buffer = model_handler.exes.get_lm_head_normalized_results.get(zml.Buffer);
    defer lm_head_normalized_buffer.deinit();
    return lm_head_normalized_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
}

pub fn getLmHeadRowNorms(zml_handler: *Zml_handler, model_handler: *model_.Model_handler) !zml.Slice {
    model_handler.exes.get_lm_head_row_norms_args.set(.{model_handler.model_buffers});
    model_handler.exes.get_lm_head_row_norms_exe.call(model_handler.exes.get_lm_head_row_norms_args, &model_handler.exes.get_lm_head_row_norms_results);
    var lm_head_row_norms_buffer = model_handler.exes.get_lm_head_row_norms_results.get(zml.Buffer);
    defer lm_head_row_norms_buffer.deinit();
    return lm_head_row_norms_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
}

pub fn getJunkRows(zml_handler: *Zml_handler, model_handler: *model_.Model_handler) ![]usize {
    model_handler.exes.find_junk_rows_args.set(.{model_handler.model_buffers});
    model_handler.exes.find_junk_rows_exe.call(model_handler.exes.find_junk_rows_args, &model_handler.exes.find_junk_rows_results);
    var junk_rows_buffer = model_handler.exes.find_junk_rows_results.get(zml.Buffer);
    defer junk_rows_buffer.deinit();

    var junk_rows_slice = try junk_rows_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer junk_rows_slice.free(zml_handler.allocator);

    const sentinel: u64 = @intCast(model_handler.model.shape().dim(.voc));
    var junk_rows: std.ArrayList(usize) = try .initCapacity(zml_handler.allocator, 0);
    errdefer junk_rows.deinit(zml_handler.allocator);
    for (junk_rows_slice.constItems(u64)) |row| {
        if (row == sentinel) continue;
        try junk_rows.append(zml_handler.allocator, @intCast(row));
    }
    return junk_rows.toOwnedSlice(zml_handler.allocator);
}

pub fn testSimilarityMatrix(zml_handler: *Zml_handler, model_handler: *model_.Model_handler, similarity_matrix: *SimilarityMatrix) !void {
    std.log.info("Test similarity matrix", .{});

    const lm_head = try getLmHead(zml_handler, model_handler);
    defer lm_head.free(zml_handler.allocator);

    const n: usize = @intCast(lm_head.shape.dim(.voc));
    const d: usize = @intCast(lm_head.shape.dim(.d));
    std.debug.assert(n == similarity_matrix.n);
    std.debug.assert(d == similarity_matrix.d);

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();
    const lm_head_items = lm_head.items(f32);

    var max_abs_diff: f32 = 0;
    for (0..1000) |_| {
        const i = random.uintLessThan(usize, n);
        const j = random.uintLessThan(usize, n);

        const u = lm_head_items[i * d ..][0..d];
        const v = lm_head_items[j * d ..][0..d];

        var dot: f32 = 0;
        var u_norm2: f32 = 0;
        var v_norm2: f32 = 0;
        for (u, v) |u_value, v_value| {
            dot += u_value * v_value;
            u_norm2 += u_value * u_value;
            v_norm2 += v_value * v_value;
        }

        const expected = dot / (@sqrt(u_norm2) * @sqrt(v_norm2));
        const actual = similarity_matrix.dist(i, j);
        const abs_diff = @abs(expected - actual);
        max_abs_diff = @max(max_abs_diff, abs_diff);
        std.log.err("rows=({d},{d}) expected={d} actual={d} diff={d}", .{ i, j, expected, actual, abs_diff });
    }

    std.log.info("Similarity matrix test passed: 1000 random pairs, max_abs_diff={d}", .{max_abs_diff});
}


pub fn getSlice(zml_handler: *Zml_handler, tensor_name: []const u8, dtype: anytype) !zml.Slice {
    std.log.info("Getting slice {s}", .{tensor_name});

    std.log.info("Init store", .{});
    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(zml_handler.allocator, zml_handler.io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(zml_handler.allocator, &registry);
    defer store.deinit();

    std.log.info("Init extractor", .{});
    const model: TensorExtractor = .init(store.view(), tensor_name);
    const shardings: Shardings = try .init(zml_handler.platform);
    const shardings_arr = shardings.all();

    std.log.info("Compile extract", .{});
    const opts: zml.module.CompilationOptions = .{ .shardings = &shardings_arr };
    const extract_exe = try zml_handler.platform.compile(zml_handler.allocator, zml_handler.io, model, .forward, .{}, opts);
    defer extract_exe.deinit();

    std.log.info("Load weights", .{});
    var model_buffers = try model.load(zml_handler, &store, &shardings.all());
    defer TensorExtractor.unloadBuffers(&model_buffers);

    std.log.info("Init slice and buffer", .{});
    const slice: zml.Slice = try .alloc(zml_handler.allocator, .init(model.shape(), dtype));
    var buffer: zml.Buffer = try .fromSlice(zml_handler.io, zml_handler.platform, slice, shardings.all()[0]);
    defer buffer.deinit();

    var extract_args = try extract_exe.args(zml_handler.allocator);
    defer extract_args.deinit(zml_handler.allocator);
    var extract_results = try extract_exe.results(zml_handler.allocator);
    defer extract_results.deinit(zml_handler.allocator);

    std.log.info("Call extract", .{});
    extract_args.set(.{model_buffers});
    extract_exe.call(extract_args, &extract_results);
    extract_results.fill(.{&buffer});

    try buffer.toSlice(zml_handler.io, slice);
    std.log.info("Return slice", .{});
    return slice;
}

const TensorExtractor = struct {
    tensor: zml.Tensor,
    pub fn init(store: zml.io.TensorStore.View, tensor_name: []const u8) TensorExtractor {
        return .{
            .tensor = store.createTensor(tensor_name, .{ .voc, .d }, .{ .voc = .replicated, .d = .replicated }),
        };
    }
    pub fn load(self: *const TensorExtractor, zml_handler: *Zml_handler, store: *zml.io.TensorStore, shardings: []const zml.Sharding) !zml.Bufferized(TensorExtractor) {
        return zml.io.load(TensorExtractor, self, zml_handler.arena.allocator(), zml_handler.io, zml_handler.platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 128 * 1024 * 1024,
        });
    }
    pub fn unloadBuffers(self: *zml.Bufferized(TensorExtractor)) void {
        self.tensor.deinit();
    }
    pub fn shape(self: TensorExtractor) zml.Shape {
        return self.tensor.shape();
    }
    pub fn forward(self: TensorExtractor) zml.Tensor {
        return self.tensor.convert(.f16);
    }
};


pub fn printSafetensors(registry: zml.safetensors.TensorRegistry) !void {
    const tensors: zml.safetensors.Tensors = registry.tensors;
    const data = tensors.entries;
    for (0..data.len) |i| {
        const entry = data.get(i);
        const tensor: zml.safetensors.Tensor = tensors.get(entry.key).?;
        std.log.info("Tensor(name={s} shape={f} size={d})", .{
            tensor.name,
            tensor.shape,
            tensor.byteSize(),
        });
    }
}

pub fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}
