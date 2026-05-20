const std = @import("std");

pub const Dataset = enum {
    math500,
    sharegpt,
    alpaca,
    swe_bench_lite,
    generic_jsonl,
    generic_json,
};

pub const Sample = struct {
    id: []const u8,
    prompt: []const u8,
    source_dataset: Dataset,
    source_split: []const u8,

    pub fn deinit(self: *Sample, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.prompt);
        allocator.free(self.source_split);
        self.* = undefined;
    }
};

pub const LoadStats = struct {
    total_rows: usize = 0,
    valid_rows: usize = 0,
    selected_rows: usize = 0,
    skipped_malformed: usize = 0,
    skipped_empty: usize = 0,
    skipped_overlong: usize = 0,
};

pub const LoadedSamples = struct {
    samples: []Sample,
    stats: LoadStats,

    pub fn deinit(self: *LoadedSamples, allocator: std.mem.Allocator) void {
        for (self.samples) |*sample| sample.deinit(allocator);
        allocator.free(self.samples);
        self.* = undefined;
    }
};

pub const LoadOptions = struct {
    dataset: Dataset,
    split: ?[]const u8 = null,
    path: []const u8,
    samples: usize,
    seed: u64 = 0,
    max_prompt_bytes: ?usize = null,
    max_file_bytes: usize = 512 * 1024 * 1024,
};

pub const LoadError = error{
    UnsupportedParquetDataset,
    NotEnoughValidSamples,
};

pub fn parquetConversionMessage() []const u8 {
    return "Parquet datasets are not read by this benchmark yet. Convert parquet shards to normalized JSONL first, then pass that JSONL path.";
}

pub fn loadSamples(allocator: std.mem.Allocator, io: std.Io, opts: LoadOptions) !LoadedSamples {
    if (isParquetPath(opts.path)) return LoadError.UnsupportedParquetDataset;

    const data = try std.Io.Dir.cwd().readFileAlloc(io, opts.path, allocator, .limited(opts.max_file_bytes));
    defer allocator.free(data);

    return loadSamplesFromSlice(allocator, opts, data);
}

pub fn loadSamplesFromSlice(allocator: std.mem.Allocator, opts: LoadOptions, data: []const u8) !LoadedSamples {
    if (isParquetPath(opts.path)) return LoadError.UnsupportedParquetDataset;

    var valid: std.ArrayList(Sample) = .empty;
    errdefer deinitSampleList(allocator, &valid);

    var stats: LoadStats = .{};
    const split = effectiveSplit(opts);

    switch (opts.dataset) {
        .sharegpt, .generic_json => try loadJsonArrayLike(allocator, data, opts, split, &valid, &stats),
        .math500, .alpaca, .swe_bench_lite, .generic_jsonl => try loadJsonl(allocator, data, opts, split, &valid, &stats),
    }

    if (valid.items.len < opts.samples) {
        stats.selected_rows = valid.items.len;
        return LoadError.NotEnoughValidSamples;
    }

    var prng = std.Random.DefaultPrng.init(opts.seed);
    prng.random().shuffle(Sample, valid.items);

    const selected = try allocator.alloc(Sample, opts.samples);
    errdefer allocator.free(selected);
    @memcpy(selected, valid.items[0..opts.samples]);

    for (valid.items[opts.samples..]) |*sample| sample.deinit(allocator);
    valid.deinit(allocator);

    stats.selected_rows = selected.len;
    return .{
        .samples = selected,
        .stats = stats,
    };
}

fn loadJsonl(
    allocator: std.mem.Allocator,
    data: []const u8,
    opts: LoadOptions,
    split: []const u8,
    valid: *std.ArrayList(Sample),
    stats: *LoadStats,
) !void {
    var lines = std.mem.splitScalar(u8, data, '\n');
    var row_index: usize = 0;
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;

        stats.total_rows += 1;
        defer row_index += 1;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{ .allocate = .alloc_always }) catch {
            stats.skipped_malformed += 1;
            continue;
        };
        defer parsed.deinit();

        const sample = sampleFromValue(allocator, opts.dataset, split, parsed.value, row_index) catch |err| switch (err) {
            error.EmptyPrompt => {
                stats.skipped_empty += 1;
                continue;
            },
            error.MalformedRow => {
                stats.skipped_malformed += 1;
                continue;
            },
            else => return err,
        };
        errdefer {
            var owned = sample;
            owned.deinit(allocator);
        }

        if (isOverlong(sample.prompt, opts.max_prompt_bytes)) {
            var owned = sample;
            owned.deinit(allocator);
            stats.skipped_overlong += 1;
            continue;
        }

        try valid.append(allocator, sample);
        stats.valid_rows += 1;
    }
}

fn loadJsonArrayLike(
    allocator: std.mem.Allocator,
    data: []const u8,
    opts: LoadOptions,
    split: []const u8,
    valid: *std.ArrayList(Sample),
    stats: *LoadStats,
) !void {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, data, .{ .allocate = .alloc_always }) catch {
        stats.skipped_malformed += 1;
        return;
    };
    defer parsed.deinit();

    const rows = rowsFromJsonRoot(parsed.value) orelse {
        stats.skipped_malformed += 1;
        return;
    };

    for (rows, 0..) |row, row_index| {
        stats.total_rows += 1;

        const sample = sampleFromValue(allocator, opts.dataset, split, row, row_index) catch |err| switch (err) {
            error.EmptyPrompt => {
                stats.skipped_empty += 1;
                continue;
            },
            error.MalformedRow => {
                stats.skipped_malformed += 1;
                continue;
            },
            else => return err,
        };
        errdefer {
            var owned = sample;
            owned.deinit(allocator);
        }

        if (isOverlong(sample.prompt, opts.max_prompt_bytes)) {
            var owned = sample;
            owned.deinit(allocator);
            stats.skipped_overlong += 1;
            continue;
        }

        try valid.append(allocator, sample);
        stats.valid_rows += 1;
    }
}

const RowError = error{ EmptyPrompt, MalformedRow } || std.mem.Allocator.Error;

fn sampleFromValue(
    allocator: std.mem.Allocator,
    dataset: Dataset,
    split: []const u8,
    value: std.json.Value,
    row_index: usize,
) RowError!Sample {
    if (value != .object) return error.MalformedRow;

    const prompt = switch (dataset) {
        .math500 => try formatMathPrompt(allocator, value),
        .alpaca => try formatAlpacaPrompt(allocator, value),
        .swe_bench_lite => try formatSweBenchPrompt(allocator, value),
        .sharegpt => try formatShareGptPrompt(allocator, value),
        .generic_jsonl, .generic_json => try formatGenericPrompt(allocator, value),
    };
    errdefer allocator.free(prompt);

    const trimmed_prompt = std.mem.trim(u8, prompt, " \t\r\n");
    if (trimmed_prompt.len == 0) return error.EmptyPrompt;

    const id = try sampleId(allocator, value, row_index);
    errdefer allocator.free(id);

    const source_split = try allocator.dupe(u8, split);
    errdefer allocator.free(source_split);

    if (trimmed_prompt.len != prompt.len) {
        const owned_trimmed = try allocator.dupe(u8, trimmed_prompt);
        errdefer allocator.free(owned_trimmed);
        allocator.free(prompt);
        return .{
            .id = id,
            .prompt = owned_trimmed,
            .source_dataset = dataset,
            .source_split = source_split,
        };
    }

    return .{
        .id = id,
        .prompt = prompt,
        .source_dataset = dataset,
        .source_split = source_split,
    };
}

fn formatMathPrompt(allocator: std.mem.Allocator, value: std.json.Value) RowError![]const u8 {
    const problem = getStringField(value, "problem") orelse return error.MalformedRow;
    if (isBlank(problem)) return error.EmptyPrompt;
    return try std.fmt.allocPrint(
        allocator,
        "{s}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        .{std.mem.trim(u8, problem, " \t\r\n")},
    );
}

fn formatAlpacaPrompt(allocator: std.mem.Allocator, value: std.json.Value) RowError![]const u8 {
    const instruction = getStringField(value, "instruction") orelse return error.MalformedRow;
    if (isBlank(instruction)) return error.EmptyPrompt;

    const trimmed_instruction = std.mem.trim(u8, instruction, " \t\r\n");
    const input = getStringField(value, "input") orelse "";
    const trimmed_input = std.mem.trim(u8, input, " \t\r\n");
    if (trimmed_input.len == 0) return try allocator.dupe(u8, trimmed_instruction);

    return try std.fmt.allocPrint(allocator, "{s}\n\nInput:\n{s}", .{ trimmed_instruction, trimmed_input });
}

fn formatSweBenchPrompt(allocator: std.mem.Allocator, value: std.json.Value) RowError![]const u8 {
    const problem_statement = getStringField(value, "problem_statement") orelse return error.MalformedRow;
    if (isBlank(problem_statement)) return error.EmptyPrompt;
    return try std.fmt.allocPrint(
        allocator,
        "Problem Statement:\n{s}\nPlease fix the issue described above.",
        .{std.mem.trim(u8, problem_statement, " \t\r\n")},
    );
}

fn formatShareGptPrompt(allocator: std.mem.Allocator, value: std.json.Value) RowError![]const u8 {
    const conversations = getArrayField(value, "conversations") orelse
        getArrayField(value, "messages") orelse
        return error.MalformedRow;

    for (conversations) |turn| {
        if (turn != .object) {
            continue;
        }
        const role = getStringField(turn, "from") orelse getStringField(turn, "role") orelse "";
        if (!isUserRole(role)) continue;

        const text = getStringField(turn, "value") orelse getStringField(turn, "content") orelse "";
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        if (trimmed.len == 0) return error.EmptyPrompt;
        return try allocator.dupe(u8, trimmed);
    }

    return error.MalformedRow;
}

fn formatGenericPrompt(allocator: std.mem.Allocator, value: std.json.Value) RowError![]const u8 {
    const prompt = getStringField(value, "prompt") orelse
        getStringField(value, "text") orelse
        getStringField(value, "input") orelse
        return error.MalformedRow;

    const trimmed = std.mem.trim(u8, prompt, " \t\r\n");
    if (trimmed.len == 0) return error.EmptyPrompt;
    return try allocator.dupe(u8, trimmed);
}

fn sampleId(allocator: std.mem.Allocator, value: std.json.Value, row_index: usize) ![]const u8 {
    if (getStringField(value, "id") orelse
        getStringField(value, "sample_id") orelse
        getStringField(value, "instance_id") orelse
        getStringField(value, "question_id")) |id|
    {
        const trimmed = std.mem.trim(u8, id, " \t\r\n");
        if (trimmed.len != 0) return try allocator.dupe(u8, trimmed);
    }

    return try std.fmt.allocPrint(allocator, "{d}", .{row_index});
}

fn rowsFromJsonRoot(value: std.json.Value) ?[]std.json.Value {
    return switch (value) {
        .array => |array| array.items,
        .object => getArrayField(value, "data") orelse getArrayField(value, "samples") orelse getArrayField(value, "rows"),
        else => null,
    };
}

fn getStringField(value: std.json.Value, field: []const u8) ?[]const u8 {
    if (value != .object) return null;
    const field_value = value.object.get(field) orelse return null;
    return switch (field_value) {
        .string => |string| string,
        .number_string => |string| string,
        else => null,
    };
}

fn getArrayField(value: std.json.Value, field: []const u8) ?[]std.json.Value {
    if (value != .object) return null;
    const field_value = value.object.get(field) orelse return null;
    return switch (field_value) {
        .array => |array| array.items,
        else => null,
    };
}

fn effectiveSplit(opts: LoadOptions) []const u8 {
    if (opts.split) |split| return split;
    return switch (opts.dataset) {
        .math500, .swe_bench_lite => "test",
        .alpaca => "train",
        .sharegpt, .generic_jsonl, .generic_json => "default",
    };
}

fn isUserRole(role: []const u8) bool {
    return std.ascii.eqlIgnoreCase(role, "human") or std.ascii.eqlIgnoreCase(role, "user");
}

fn isBlank(value: []const u8) bool {
    return std.mem.trim(u8, value, " \t\r\n").len == 0;
}

fn isOverlong(prompt: []const u8, max_prompt_bytes: ?usize) bool {
    return if (max_prompt_bytes) |max| prompt.len > max else false;
}

fn isParquetPath(path: []const u8) bool {
    return std.ascii.eqlIgnoreCase(std.fs.path.extension(path), ".parquet");
}

fn deinitSampleList(allocator: std.mem.Allocator, list: *std.ArrayList(Sample)) void {
    for (list.items) |*sample| sample.deinit(allocator);
    list.deinit(allocator);
}

test "math500 jsonl prompt formatting and deterministic selection stats" {
    const data =
        \\{"id":"a","problem":"What is 1+1?"}
        \\{"id":"b","problem":"   "}
        \\{"id":"c","problem":"What is 2+2?"}
        \\
    ;

    var loaded = try loadSamplesFromSlice(std.testing.allocator, .{
        .dataset = .math500,
        .path = "test.jsonl",
        .samples = 2,
        .seed = 0,
    }, data);
    defer loaded.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3), loaded.stats.total_rows);
    try std.testing.expectEqual(@as(usize, 2), loaded.stats.valid_rows);
    try std.testing.expectEqual(@as(usize, 2), loaded.stats.selected_rows);
    try std.testing.expectEqual(@as(usize, 1), loaded.stats.skipped_empty);
    try std.testing.expect(std.mem.indexOf(u8, loaded.samples[0].prompt, "Please reason step by step") != null);
}

test "sharegpt json array extracts first user turn" {
    const data =
        \\[
        \\  {"id":"x","conversations":[{"from":"system","value":"ignore"},{"from":"human","value":"Hello there"},{"from":"gpt","value":"Hi"}]},
        \\  {"id":"empty","conversations":[{"from":"human","value":"  "}]}
        \\]
    ;

    var loaded = try loadSamplesFromSlice(std.testing.allocator, .{
        .dataset = .sharegpt,
        .path = "sharegpt.json",
        .samples = 1,
        .seed = 0,
        .max_prompt_bytes = 100,
    }, data);
    defer loaded.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), loaded.stats.total_rows);
    try std.testing.expectEqual(@as(usize, 1), loaded.stats.valid_rows);
    try std.testing.expectEqualStrings("Hello there", loaded.samples[0].prompt);
    try std.testing.expectEqual(@as(usize, 1), loaded.stats.skipped_empty);
}

test "alpaca and swe bench normalized jsonl formats" {
    const alpaca =
        \\{"id":"alp","instruction":"Summarize","input":"A long note"}
        \\
    ;
    var alpaca_loaded = try loadSamplesFromSlice(std.testing.allocator, .{
        .dataset = .alpaca,
        .path = "alpaca.jsonl",
        .samples = 1,
        .seed = 0,
    }, alpaca);
    defer alpaca_loaded.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("Summarize\n\nInput:\nA long note", alpaca_loaded.samples[0].prompt);
    try std.testing.expectEqualStrings("train", alpaca_loaded.samples[0].source_split);

    const swe =
        \\{"instance_id":"swe-1","problem_statement":"Tests fail on import."}
        \\
    ;
    var swe_loaded = try loadSamplesFromSlice(std.testing.allocator, .{
        .dataset = .swe_bench_lite,
        .path = "swe.jsonl",
        .samples = 1,
        .seed = 0,
    }, swe);
    defer swe_loaded.deinit(std.testing.allocator);
    try std.testing.expect(std.mem.startsWith(u8, swe_loaded.samples[0].prompt, "Problem Statement:\nTests fail"));
    try std.testing.expectEqualStrings("swe-1", swe_loaded.samples[0].id);
}

test "generic jsonl filters malformed empty and overlong rows" {
    const data =
        \\{"id":"ok","prompt":"short"}
        \\not json
        \\{"id":"empty","prompt":" "}
        \\{"id":"long","prompt":"this prompt is too long"}
        \\
    ;

    var loaded = try loadSamplesFromSlice(std.testing.allocator, .{
        .dataset = .generic_jsonl,
        .path = "generic.jsonl",
        .samples = 1,
        .seed = 0,
        .max_prompt_bytes = 10,
    }, data);
    defer loaded.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 4), loaded.stats.total_rows);
    try std.testing.expectEqual(@as(usize, 1), loaded.stats.valid_rows);
    try std.testing.expectEqual(@as(usize, 1), loaded.stats.skipped_malformed);
    try std.testing.expectEqual(@as(usize, 1), loaded.stats.skipped_empty);
    try std.testing.expectEqual(@as(usize, 1), loaded.stats.skipped_overlong);
    try std.testing.expectEqualStrings("short", loaded.samples[0].prompt);
}

test "parquet paths return localized unsupported error" {
    try std.testing.expectError(LoadError.UnsupportedParquetDataset, loadSamplesFromSlice(std.testing.allocator, .{
        .dataset = .alpaca,
        .path = "train.parquet",
        .samples = 1,
        .seed = 0,
    }, ""));
    try std.testing.expect(std.mem.indexOf(u8, parquetConversionMessage(), "normalized JSONL") != null);
}
