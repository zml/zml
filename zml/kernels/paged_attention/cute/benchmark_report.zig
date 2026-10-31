const std = @import("std");
const cases = @import("cases.zig");

pub const Stats = struct {
    min_us: f64,
    p10_us: f64,
    median_us: f64,
    p90_us: f64,
    max_us: f64,
    mean_us: f64,

    pub fn init(a: std.mem.Allocator, samples: []const f64) !Stats {
        if (samples.len == 0) return error.EmptySamples;
        const sorted = try a.dupe(f64, samples);
        defer a.free(sorted);
        std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
        var sum: f64 = 0;
        for (samples) |sample| sum += sample;
        return .{
            .min_us = sorted[0],
            .p10_us = sorted[(samples.len - 1) / 10],
            .median_us = if (samples.len % 2 == 0)
                (sorted[samples.len / 2 - 1] + sorted[samples.len / 2]) / 2
            else
                sorted[samples.len / 2],
            .p90_us = sorted[samples.len - 1 - samples.len / 10],
            .max_us = sorted[samples.len - 1],
            .mean_us = sum / @as(f64, @floatFromInt(samples.len)),
        };
    }
};

pub const Result = struct {
    case_index: usize,
    case: cases.Case,
    compile_ms: f64,
    latency: Stats,
    // Keep collection order so drift and outliers can be inspected after export.
    samples_us: []const f64,
};

pub const Report = struct {
    schema_version: u32 = 1,
    mode: []const u8 = "benchmark",
    timing: []const u8 = "synchronized_invocation_us",
    includes: []const []const u8 = &.{ "runtime", "output_allocation", "attention", "merge", "wait" },
    excludes: []const []const u8 = &.{ "compilation", "upload", "output_destruction" },
    cache_policy: []const u8 = "reused_inputs_no_flush",
    dtype: []const u8 = "bf16",
    page_size: usize = cases.page_size,
    seed: u64 = 42,
    devices: usize,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    // Each case describes heads per device, matching the existing benchmark.
    kv_heads_scope: []const u8 = "per_device",
    warmup: usize,
    results: []const Result,

    pub fn write(self: Report, a: std.mem.Allocator, io: std.Io, prefix: []const u8) !void {
        if (std.fs.path.dirname(prefix)) |parent| try std.Io.Dir.cwd().createDirPath(io, parent);
        const json = try std.json.Stringify.valueAlloc(a, self, .{ .whitespace = .indent_2 });
        defer a.free(json);
        const csv = try self.csvAlloc(a);
        defer a.free(csv);
        for ([_][]const u8{ ".json", ".csv" }, [_][]const u8{ json, csv }) |extension, contents| {
            const path = try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, extension });
            defer a.free(path);
            const file = try std.Io.Dir.cwd().createFile(io, path, .{});
            defer file.close(io);
            try file.writePositionalAll(io, contents, 0);
            std.debug.print("EXPORTED {s}\n", .{path});
        }
    }

    pub fn csvAlloc(self: Report, a: std.mem.Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(a);
        defer out.deinit();
        const w = &out.writer;
        try w.writeAll("case_index,name,head_dim,batch,kv_heads_per_device,group,prefill,chunk,kv_len,mask,window,sink,scale,devices,cc_major,cc_minor,dtype,page_size,seed,warmup,sample_count,compile_ms,timing,cache_policy,min_us,p10_us,median_us,p90_us,max_us,mean_us\n");
        for (self.results) |r| {
            const c = r.case;
            try w.print("{d},", .{r.case_index});
            try csvString(w, c.name);
            try w.print(",{d},{d},{d},{d},{d},{d},{d},{s},{d},{},{d},{d},{d},{d},{s},{d},{d},{d},{d},{d:.6},{s},{s},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6},{d:.6}\n", .{
                c.head_dim,                    c.batch,                       c.kv_heads,       c.group,           c.prefill,                                                     if (c.prefill > 0) c.chunk else 1,
                c.kv_len,                      @tagName(c.mask),              c.window,         c.sink,            c.scale orelse 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim))), self.devices,
                self.compute_capability_major, self.compute_capability_minor, self.dtype,       self.page_size,    self.seed,                                                     self.warmup,
                r.samples_us.len,              r.compile_ms,                  self.timing,      self.cache_policy, r.latency.min_us,                                              r.latency.p10_us,
                r.latency.median_us,           r.latency.p90_us,              r.latency.max_us, r.latency.mean_us,
            });
        }
        return try out.toOwnedSlice();
    }
};

fn csvString(w: *std.Io.Writer, value: []const u8) !void {
    try w.writeByte('"');
    for (value) |ch| {
        if (ch == '"') try w.writeByte('"');
        try w.writeByte(ch);
    }
    try w.writeByte('"');
}

test "latency statistics preserve chronological samples and handle even and odd counts" {
    const samples = [_]f64{ 30, 10, 40, 20 };
    const stats = try Stats.init(std.testing.allocator, &samples);
    try std.testing.expectEqual(@as(f64, 25), stats.median_us);
    try std.testing.expectEqual(@as(f64, 25), stats.mean_us);
    try std.testing.expectEqual(@as(f64, 10), stats.min_us);
    try std.testing.expectEqual(@as(f64, 40), stats.max_us);
    try std.testing.expectEqual(@as(f64, 30), samples[0]);
    const odd = try Stats.init(std.testing.allocator, &.{ 30, 10, 20 });
    try std.testing.expectEqual(@as(f64, 20), odd.median_us);
    const deciles = try Stats.init(std.testing.allocator, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
    try std.testing.expectEqual(@as(f64, 1), deciles.p10_us);
    try std.testing.expectEqual(@as(f64, 9), deciles.p90_us);
    try std.testing.expectError(error.EmptySamples, Stats.init(std.testing.allocator, &.{}));
}

test "JSON and CSV exports identify timing units, geometry, and cache assumptions" {
    const a = std.testing.allocator;
    const samples = [_]f64{ 30, 10, 20 };
    const results = [_]Result{.{
        .case_index = 7,
        .case = .{ .name = "case,\"quoted\"" },
        .compile_ms = 12.5,
        .latency = try Stats.init(a, &samples),
        .samples_us = &samples,
    }};
    const report: Report = .{ .devices = 1, .compute_capability_major = 10, .compute_capability_minor = 3, .warmup = 5, .results = &results };
    const json = try std.json.Stringify.valueAlloc(a, report, .{});
    defer a.free(json);
    const parsed = try std.json.parseFromSlice(Report, a, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("synchronized_invocation_us", parsed.value.timing);
    try std.testing.expectEqualStrings("reused_inputs_no_flush", parsed.value.cache_policy);
    try std.testing.expectEqualSlices(f64, &samples, parsed.value.results[0].samples_us);
    try std.testing.expectEqual(@as(usize, 7), parsed.value.results[0].case_index);
    const csv = try report.csvAlloc(a);
    defer a.free(csv);
    try std.testing.expect(std.mem.indexOf(u8, csv, "7,\"case,\"\"quoted\"\"\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, csv, "synchronized_invocation_us,reused_inputs_no_flush") != null);
    try std.testing.expect(std.mem.indexOf(u8, csv, "10.000000,10.000000,20.000000,30.000000,30.000000,20.000000") != null);
}
