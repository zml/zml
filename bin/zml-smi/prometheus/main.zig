const std = @import("std");
const OoM = std.mem.Allocator.Error;

const tui = @import("zml-smi/tui");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

const log = std.log.scoped(.prometheus);

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .prometheus, .level = .info },
    },
};

pub fn main(init: std.process.Init) !void {
    var model: Model = try .init(init.gpa, init.io, "http://gh200:8001/metrics");
    defer model.deinit();

    try model.updateMetrics();

    var app = try vaxis.vxfw.App.init(init.gpa, init.io);
    defer app.deinit();

    try app.run(tui.ui.widget(&model), .{});
}

pub const Metric = struct {
    name: []const u8,
    value: Data,
    chart_height: u16 = 3,

    pub const Kind = enum { counter, gauge, histogram };

    pub const Data = union(Kind) {
        counter: Counter,
        gauge: Gauge,
        histogram: Histogram,
    };

    pub const Histogram = struct {
        bucket_counts: []u64,
        bucket_upper_bounds: []i64,
        total_count: u64,
        total_sum: i64,

        pub const empty: Histogram = .{ .bucket_counts = &.{}, .bucket_upper_bounds = &.{}, .total_count = 0, .total_sum = 0 };

        pub fn deinit(hist: *Histogram, allocator: std.mem.Allocator) void {
            allocator.free(hist.bucket_counts);
            allocator.free(hist.bucket_upper_bounds);
            hist.* = .empty;
        }
    };

    pub const Gauge = struct {
        min: i64,
        max: i64,
        current: i64,

        pub fn startingAtZero(current: i64) Gauge {
            return .{ .min = @min(current - 1, 0), .max = current, .current = current };
        }

        pub fn update(g: *Gauge, current: i64) void {
            g.min = @min(g.min, current);
            g.max = @max(g.max, current);
            g.current = current;
        }

        pub fn asPercentage(g: Gauge) u8 {
            return @intCast(@divFloor(100 * (g.current - g.min), g.max - g.min));
        }
    };

    pub const Counter = struct {
        current: i64,
    };
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    metrics: Metrics,
    url: []const u8,
    client: std.http.Client,
    content: std.ArrayList(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, url: []const u8) !Model {
        return .{
            .allocator = allocator,
            .url = url,
            .metrics = try .init(allocator),
            .client = .{ .allocator = allocator, .io = io },
        };
    }

    pub fn deinit(self: *Model) void {
        self.client.deinit();
        self.content.deinit(self.allocator);
        self.metrics.deinit();
    }

    pub fn handleEvent(self: *Model, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        switch (event) {
            .init => {
                try self.updateMetrics();
                try ctx.tick(1000, tui.ui.widget(self));
            },
            .tick => {
                self.updateMetrics() catch {};
                ctx.redraw = true;
                try ctx.tick(1000, tui.ui.widget(self));
            },
            .key_press => |key| {
                if (key.matches('q', .{})) {
                    ctx.quit = true;
                }
            },
            else => {},
        }
    }

    pub fn updateMetrics(self: *Model) !void {
        const content = try self.fetchMetrics(self.url);
        var reader: std.Io.Reader = .fixed(content);
        try self.metrics.parsePrometheusMetrics(&reader);
    }

    pub fn fetchMetrics(
        self: *Model,
        url: []const u8,
    ) ![]const u8 {
        var body: std.Io.Writer.Allocating = .fromArrayList(self.allocator, &self.content);
        defer body.deinit();

        _ = try self.client.fetch(.{
            .location = .{ .url = url },
            .method = .GET,
            .extra_headers = &.{},
            .response_writer = &body.writer,
        });

        self.content = body.toArrayList();
        return self.content.items;
    }

    pub fn draw(self: *Model, ctx: vxfw.DrawContext) !vxfw.Surface {
        var sb = tui.compose.surfaceBuilder(ctx.arena);
        var row: i17 = 0;

        for (self.metrics.metrics.values()) |m| {
            switch (m.value) {
                .gauge => |g| {
                    if (g.max - g.min == 1) continue;
                    const gauge: tui.Gauge = .{
                        .value = g.asPercentage(),
                        .label = m.name,
                        .suffix = "",
                        .label_reserve = 0,
                        .suffix_reserve = 0,
                    };
                    const gauge_surf = try gauge.draw(tui.ui.fixedSize(ctx, ctx.max.width orelse 40, 1));
                    try sb.add(row, 0, gauge_surf);
                    row += 1;
                    if (row < 100) row += 1;
                },
                .histogram => |hist| {
                    // Create bar chart data from histogram buckets
                    const bars = try ctx.arena.alloc(tui.BarChart.BucketData, hist.bucket_counts.len);
                    const total_count = hist.total_count;
                    var prev: u64 = 0;
                    for (bars, hist.bucket_counts, hist.bucket_upper_bounds) |*bar, count, upper_bound| {
                        bar.* = .{
                            .percentage = @intCast(@divFloor(100 * (count - prev), total_count)),
                            .upper_bound = upper_bound,
                        };
                        prev = count;
                    }

                    const bar_chart: tui.BarChart = .{
                        .buckets = bars,
                        .bar_height = m.chart_height,
                        .show_values = true,
                        .show_bounds = true,
                        .label = m.name,
                    };
                    const bar_chart_height = bar_chart.height();
                    const bar_surf = try bar_chart.draw(tui.ui.fixedSize(ctx, (ctx.max.width orelse 40), bar_chart_height));
                    try sb.add(row, 0, bar_surf);
                    row += @intCast(bar_chart_height);
                },
                .counter => |counter| {
                    // TODO: display counter
                    _ = counter;
                },
            }
        }

        return sb.finish(
            .{ .width = ctx.max.width orelse 40, .height = @max(row, 1) },
            tui.ui.drawWidget(self, drawContent),
        );
    }

    fn drawContent(_: *Model, ctx: vxfw.DrawContext) !vxfw.Surface {
        return try tui.ui.drawRichLine(
            ctx,
            &.{
                .{ .text = "Prometheus Metrics", .style = tui.theme.label_style },
            },
            ctx.max.width orelse 40,
        );
    }
};

pub const Metrics = struct {
    metrics: std.StringArrayHashMapUnmanaged(Metric) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) OoM!Metrics {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Metrics) void {
        for (self.metrics.values()) |*m| {
            self.allocator.free(m.name);
            switch (m.value) {
                .histogram => |*h| h.deinit(self.allocator),
                .counter, .gauge => {},
            }
        }
        self.metrics.deinit(self.allocator);
    }

    pub fn parsePrometheusMetrics(self: *Metrics, reader: *std.Io.Reader) !void {
        while (try reader.takeDelimiter('\n')) |line| {
            // Skip empty lines
            if (line.len == 0) continue;

            // Handle comments and TYPE declarations
            if (line[0] == '#') {
                if (std.mem.find(u8, line, "TYPE ")) |t_pos| {
                    const rest = line[t_pos + "TYPE ".len ..];
                    const name = if (std.mem.findScalarLast(u8, rest, ' ')) |r| rest[0..r] else rest;
                    const type_str = if (std.mem.findScalarLast(u8, rest, ' ')) |r| rest[r + 1 ..] else "unknown";

                    // String comparison for switch
                    if (std.mem.eql(u8, type_str, "counter")) {
                        try self.parseCounters(reader, name);
                    } else if (std.mem.eql(u8, type_str, "histogram")) {
                        try self.parseHistograms(reader, name);
                    } else if (std.mem.eql(u8, type_str, "gauge")) {
                        try self.parseGauges(reader, name);
                    } else {
                        log.warn("Unknown type: {s}", .{type_str});
                    }
                }
            }
        }
    }

    pub fn parseGauges(self: *Metrics, reader: *std.Io.Reader, name: []const u8) !void {
        lines: while (true) {
            const first = reader.peek(1) catch return;
            if (first[0] == '#') return;

            const line = (try reader.takeDelimiter('\n')) orelse return;
            std.debug.assert(std.mem.eql(u8, name, line[0..name.len]));

            const last_space = std.mem.findScalarLast(u8, line, ' ') orelse continue :lines;
            const name_with_labels = std.mem.trimEnd(u8, line[0..last_space], " ");
            const value_str = std.mem.trim(u8, line[last_space + 1 ..], " \n\r\t");
            const value = std.fmt.parseInt(i64, value_str, 10) catch continue :lines;

            const entry = try self.metrics.getOrPut(self.allocator, name_with_labels);
            found: switch (entry.found_existing) {
                true => {
                    if (entry.value_ptr.value != .gauge) continue :found false;
                    entry.value_ptr.value.gauge.update(value);
                },
                false => {
                    const owned_name = try self.allocator.dupe(u8, name_with_labels);
                    entry.key_ptr.* = owned_name;
                    entry.value_ptr.* = .{
                        .value = .{ .gauge = .startingAtZero(value) },
                        .name = owned_name,
                    };
                },
            }
        }
    }

    pub fn parseCounters(self: *Metrics, reader: *std.Io.Reader, name: []const u8) !void {
        lines: while (true) {
            const first = reader.peek(1) catch return;
            if (first[0] == '#') return;

            const line = (try reader.takeDelimiter('\n')) orelse return;
            std.debug.assert(std.mem.eql(u8, name, line[0..name.len]));

            const last_space = std.mem.findScalarLast(u8, line, ' ') orelse continue :lines;
            const name_with_labels = std.mem.trimEnd(u8, line[0..last_space], " ");
            const value_str = std.mem.trim(u8, line[last_space + 1 ..], " \n\r\t");
            const value = std.fmt.parseInt(i64, value_str, 10) catch continue :lines;

            const entry = try self.metrics.getOrPut(self.allocator, name_with_labels);
            found: switch (entry.found_existing) {
                true => {
                    if (entry.value_ptr.value != .counter) continue :found false;
                    entry.value_ptr.value.counter.current = value;
                },
                false => {
                    const owned_name = try self.allocator.dupe(u8, name_with_labels);
                    entry.key_ptr.* = owned_name;
                    entry.value_ptr.* = .{
                        .value = .{ .counter = .{ .current = value } },
                        .name = owned_name,
                    };
                },
            }
        }
    }

    pub fn parseHistograms(self: *Metrics, reader: *std.Io.Reader, name: []const u8) !void {
        const State = enum { first, bucket, count, sum };
        var state: State = .first;
        var bucket_counts: std.ArrayList(u64) = undefined;
        var bucket_upper_bounds: std.ArrayList(i64) = undefined;
        var histogram_count: u64 = 0;
        var histogram: *Metric.Histogram = undefined;

        lines: while (true) {
            const first = reader.peek(1) catch return;
            if (first[0] == '#') return;

            const line = (try reader.takeDelimiter('\n')) orelse return;
            if (line.len == 0) continue :lines;

            // log.warn("state: {}, line: {s}", .{ state, line });
            std.debug.assert(std.mem.eql(u8, name, line[0..name.len]));

            const first_label = std.mem.findScalar(u8, line, '{') orelse return error.InvalidInput;
            const histogram_metric_stream = line[name.len + 1 .. first_label];
            state: switch (state) {
                .first => {
                    std.debug.assert(std.mem.eql(u8, histogram_metric_stream, "bucket"));
                    const name_with_labels = parseBucketName(self.allocator, line) catch |err| switch (err) {
                        error.InvalidInput => continue :lines,
                        error.OutOfMemory => |e| return e,
                    };

                    const entry = try self.metrics.getOrPut(self.allocator, name_with_labels);
                    log.debug("Starting histogram: {s}", .{name_with_labels});
                    found: switch (entry.found_existing) {
                        true => {
                            if (entry.value_ptr.value != .histogram) continue :found false;
                            self.allocator.free(name_with_labels);
                            histogram = &entry.value_ptr.value.histogram;
                            bucket_counts = .initBuffer(histogram.bucket_counts);
                            bucket_upper_bounds = .initBuffer(histogram.bucket_upper_bounds);
                            histogram_count = 0;
                        },
                        false => {
                            entry.value_ptr.* = .{
                                .value = .{ .histogram = .empty },
                                .name = name_with_labels,
                            };
                            histogram = &entry.value_ptr.value.histogram;
                            entry.key_ptr.* = name_with_labels;
                            bucket_counts = .empty;
                            bucket_upper_bounds = .empty;
                            histogram_count = 0;
                        },
                    }
                    continue :state .bucket;
                },
                .bucket => {
                    if (std.mem.eql(u8, histogram_metric_stream, "count")) continue :state .count;

                    std.debug.assert(std.mem.eql(u8, histogram_metric_stream, "bucket"));
                    const upper, const cnt = parseBucketData(line) catch {
                        log.warn("Invalid line: {s}", .{line});
                        continue :lines;
                    };
                    try bucket_upper_bounds.append(self.allocator, upper);
                    try bucket_counts.append(self.allocator, cnt);

                    state = .bucket;
                },
                .count => {
                    std.debug.assert(std.mem.eql(u8, histogram_metric_stream, "count"));
                    const last_space = std.mem.findScalarLast(u8, line, ' ') orelse continue :lines;
                    const value_str = line[last_space + 1 ..];
                    histogram_count = std.fmt.parseInt(u64, value_str, 10) catch continue :lines;

                    state = .sum;
                },
                .sum => {
                    std.debug.assert(std.mem.eql(u8, histogram_metric_stream, "sum"));
                    histogram.* = .{
                        .bucket_counts = try bucket_counts.toOwnedSlice(self.allocator),
                        .bucket_upper_bounds = try bucket_upper_bounds.toOwnedSlice(self.allocator),
                        .total_count = histogram_count,
                        .total_sum = 0,
                    };

                    log.debug("Finished histogram: {}", .{histogram});
                    state = .first;
                },
            }
        }
    }

    fn parseBucketName(allocator: std.mem.Allocator, line: []const u8) error{ InvalidInput, OutOfMemory }![]const u8 {
        const bucket = std.mem.find(u8, line, "_bucket{") orelse return error.InvalidInput;
        const last_label = std.mem.findLast(u8, line, ",le=\"") orelse return error.InvalidInput;

        return try std.mem.join(allocator, "", &.{ line[0..bucket], line[bucket + "_bucket".len .. last_label], "}" });
    }

    fn parseBucketData(line: []const u8) error{InvalidInput}!struct { i64, u64 } {
        const le_start = std.mem.findLast(u8, line, ",le=\"") orelse return error.InvalidInput;
        const rest = line[le_start + ",le=\"".len ..];
        const le_end = std.mem.findScalar(u8, rest, '"') orelse return error.InvalidInput;
        const le_str = rest[0..le_end];

        const upper_bound = if (std.mem.eql(u8, le_str, "+Inf"))
            std.math.maxInt(i64)
        else
            std.fmt.parseInt(i64, le_str, 10) catch return error.InvalidInput;

        const last_space = std.mem.findScalarLast(u8, line, ' ') orelse return error.InvalidInput;
        const count_str = std.mem.trim(u8, line[last_space + 1 ..], " \n\r\t");
        const count = std.fmt.parseInt(u64, count_str, 10) catch return error.InvalidInput;

        return .{ upper_bound, count };
    }
};

test "parse counter metrics" {
    const content =
        \\# TYPE http_requests_count counter
        \\http_requests_count{path="/v1/chat/completions", method="POST", code="ok"} 242
        \\http_requests_count{path="/v1/chat/completions", method="OPTIONS", code="not_found"} 20
        \\http_requests_count{path="/", method="GET", code="not_found"} 1
        \\
    ;

    var data: Metrics = try .init(std.testing.allocator);
    defer data.deinit();

    var content_reader: std.Io.Reader = .fixed(content);
    try data.parsePrometheusMetrics(&content_reader);

    const metrics = data.metrics;
    try std.testing.expectEqual(3, metrics.count());
    const completions_post = metrics.get(
        \\http_requests_count{path="/v1/chat/completions", method="POST", code="ok"}
    ) orelse return error.NotFound;
    try std.testing.expectEqual(242, completions_post.value.counter.current);

    const completion_options = metrics.get(
        \\http_requests_count{path="/v1/chat/completions", method="OPTIONS", code="not_found"}
    ) orelse return error.NotFound;
    try std.testing.expectEqual(20, completion_options.value.counter.current);

    const root_get = metrics.get(
        \\http_requests_count{path="/", method="GET", code="not_found"}
    ) orelse return error.NotFound;
    try std.testing.expectEqual(1, root_get.value.counter.current);
}

test "parse histogram" {
    const content =
        \\# HELP token_itl token_itl (inter token latency) in MILLISECONDS.
        \\# TYPE token_itl histogram
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="1"} 0
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="2"} 0
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="3"} 0
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="4"} 0
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="5"} 0
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="6"} 0
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="7"} 0
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="8"} 0
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="11"} 12121
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="16"} 50558
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="22"} 86744
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="30"} 94566
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="41"} 99835
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="56"} 108817
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="76"} 121436
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="104"} 136017
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="143"} 153887
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="195"} 158625
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="265"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="362"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="494"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="674"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="919"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="1254"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="1710"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="2332"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="3180"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="4337"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="5914"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="8066"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="10999"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="15000"} 158650
        \\token_itl_bucket{model="Qwen3.5-35B-A3B/",le="+Inf"} 158650
        \\token_itl_count{model="Qwen3.5-35B-A3B/"} 158650
        \\token_itl_sum{model="Qwen3.5-35B-A3B/"} 7240206
    ;

    var data: Metrics = try .init(std.testing.allocator);
    defer data.deinit();
    var content_reader: std.Io.Reader = .fixed(content);
    try data.parsePrometheusMetrics(&content_reader);

    const metrics = data.metrics;
    try std.testing.expectEqual(1, metrics.count());
    const token_itl = metrics.get(
        \\token_itl{model="Qwen3.5-35B-A3B/"}
    ) orelse return error.NotFound;
    try std.testing.expectEqual(33, token_itl.value.histogram.bucket_counts.len);
    try std.testing.expectEqual(12121, token_itl.value.histogram.bucket_counts[8]);
}

test "parse several histograms" {
    const content =
        \\# HELP http_requests_latency http_requests latency handled by the server.
        \\# TYPE http_requests_latency histogram
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="1"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="2"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="3"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="4"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="5"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="6"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="7"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="8"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="11"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="16"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="22"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="30"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="41"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="56"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="76"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="104"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="143"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="195"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="265"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="362"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="494"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="674"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="919"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="1254"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="1710"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="2332"} 0
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="3180"} 1
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="4337"} 2
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="5914"} 5
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="8066"} 19
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="10999"} 46
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="15000"} 89
        \\http_requests_latency_bucket{path="/v1/chat/completions",method="POST",le="+Inf"} 334
        \\http_requests_latency_count{path="/v1/chat/completions",method="POST"} 334
        \\http_requests_latency_sum{path="/v1/chat/completions",method="POST"} 10710957
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="1"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="2"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="3"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="4"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="5"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="6"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="7"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="8"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="11"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="16"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="22"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="30"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="41"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="56"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="76"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="104"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="143"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="195"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="265"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="362"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="494"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="674"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="919"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="1254"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="1710"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="2332"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="3180"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="4337"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="5914"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="8066"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="10999"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="15000"} 115
        \\http_requests_latency_bucket{path="/metrics",method="GET",le="+Inf"} 115
        \\http_requests_latency_count{path="/metrics",method="GET"} 115
        \\http_requests_latency_sum{path="/metrics",method="GET"} 0
        \\http_requests_latency_bucket{path="/health",method="GET",le="1"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="2"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="3"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="4"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="5"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="6"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="7"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="8"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="11"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="16"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="22"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="30"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="41"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="56"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="76"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="104"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="143"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="195"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="265"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="362"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="494"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="674"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="919"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="1254"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="1710"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="2332"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="3180"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="4337"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="5914"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="8066"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="10999"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="15000"} 1
        \\http_requests_latency_bucket{path="/health",method="GET",le="+Inf"} 1
        \\http_requests_latency_count{path="/health",method="GET"} 1
        \\http_requests_latency_sum{path="/health",method="GET"} 0
    ;

    var data: Metrics = try .init(std.testing.allocator);
    defer data.deinit();
    var content_reader: std.Io.Reader = .fixed(content);
    try data.parsePrometheusMetrics(&content_reader);

    const metrics = data.metrics;
    try std.testing.expectEqual(3, metrics.count());
    const health = metrics.get(
        \\http_requests_latency{path="/health",method="GET"}
    ) orelse return error.NotFound;
    try std.testing.expectEqual(33, health.value.histogram.bucket_counts.len);
    try std.testing.expectEqual(1, health.value.histogram.total_count);
}
