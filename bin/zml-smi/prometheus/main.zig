const std = @import("std");
const OoM = std.mem.Allocator.Error;

const tui = @import("zml-smi/tui");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

pub fn main(init: std.process.Init) !void {
    var app = try vaxis.vxfw.App.init(init.gpa, init.io);
    defer app.deinit();

    var model: Model = try .init(init.gpa, init.io);
    defer model.deinit();

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
        bucket_upper_bound: []u64,
        bucket_counts_u8: []u8,
        bucket_total_count: u64,
        bucket_total_sum: i64,
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
    client: std.http.Client,
    content: std.ArrayList(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Model {
        return .{
            .allocator = allocator,
            .metrics = try .init(allocator),
            .client = .{ .allocator = allocator, .io = io },
        };
    }

    pub fn deinit(self: *Model) void {
        self.client.deinit();
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
        const content = try self.fetchMetrics("http://gh200:8001/metrics");
        var reader: std.Io.Reader = .fixed(content);
        try self.metrics.parsePrometheusMetrics(&reader);
    }

    pub fn fetchMetrics(
        self: *Model,
        url: []const u8,
    ) ![]const u8 {
        var req = try std.http.Client.request(&self.client, .GET, try std.Uri.parse(url), .{ .keep_alive = false });
        defer req.deinit();

        try req.sendBodiless();
        var response = try req.receiveHead(&.{});

        var body: std.Io.Writer.Allocating = .fromArrayList(self.allocator, &self.content);
        defer body.deinit();

        const reader = response.reader(&.{});
        _ = try reader.stream(&body.writer, .limited(64 * 1024));
        self.content = body.toArrayList();
        return self.content.items;
    }

    pub fn draw(self: *Model, ctx: vxfw.DrawContext) !vxfw.Surface {
        var sb = tui.compose.surfaceBuilder(ctx.arena);
        var row: i17 = 0;

        for (self.metrics.metrics.values()) |m| {
            switch (m.value) {
                .gauge => |g| {
                    const gauge: tui.Gauge = .{
                        .value = g.asPercentage(),
                        .label = m.label,
                        .suffix = m.suffix,
                        .label_reserve = 0,
                        .suffix_reserve = 0,
                    };
                    const gauge_surf = try gauge.draw(tui.ui.fixedSize(ctx, ctx.max.width orelse 40, 1));
                    try sb.add(row, 0, gauge_surf);
                    row += 1;
                    if (row < 100) row += 1;
                },
                .histogram => |hist| {
                    const braille: tui.BrailleChart = .{
                        .values = hist.bucket_counts_u8,
                        .height = m.chart_height,
                    };
                    const braille_surf = try braille.draw(tui.ui.fixedSize(ctx, (ctx.max.width orelse 40) - 20, m.chart_height));
                    try sb.add(row, 0, braille_surf);
                    row += @as(i17, @intCast(m.chart_height));
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
        for (self.metrics.values()) |m| {
            self.allocator.free(m.name);
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
                        std.log.warn("Unknown type: {s}", .{type_str});
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
        _ = self;
        lines: while (true) {
            const first = reader.peek(1) catch return;
            if (first[0] == '#') return;

            const line = (try reader.takeDelimiter('\n')) orelse return;
            std.debug.assert(std.mem.eql(u8, name, line[0..name.len]));

            continue :lines;
        }
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
