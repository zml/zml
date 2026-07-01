const std = @import("std");

const tui = @import("zml-smi/tui");
const compose = tui.compose;
const ui = tui.ui;
const theme = tui.theme;
const BrailleChart = tui.BrailleChart;
const Gauge = tui.Gauge;
const TitledBorder = tui.TitledBorder;
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

pub const MetricType = enum {
    counter,
    gauge,
    histogram,
};

pub const Metric = struct {
    name: []const u8,
    m_type: MetricType,
    value: f64,
    label: []const u8,
    suffix: ?[]const u8 = null,
    history: []u8 = &.{},
    chart_height: u16 = 3,
};

pub const Visualizer = struct {
    metrics: std.ArrayList(Metric),
    allocator: std.mem.Allocator,
    client: std.http.Client,
    content: std.ArrayList(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) Visualizer {
        return .{
            .metrics = .empty,
            .allocator = allocator,
            .client = std.http.Client{ .allocator = allocator, .io = io },
        };
    }

    pub fn deinit(self: *Visualizer) void {
        self.metrics.deinit(self.allocator);
        self.client.deinit();
    }

    pub fn fetchMetrics(
        self: *Visualizer,
        url: []const u8,
    ) ![]const u8 {
        var req = try std.http.Client.request(&self.client, .GET, try std.Uri.parse(url), .{ .keep_alive = false });
        defer req.deinit();

        try req.sendBodiless();
        var response = try req.receiveHead(&.{});

        // Collect response body
        var body: std.Io.Writer.Allocating = .fromArrayList(self.allocator, &self.content);
        defer body.deinit();

        const reader = response.reader(&.{});
        _ = try reader.stream(&body.writer, .limited(64 * 1024));
        self.content = body.toArrayList();
        return self.content.items;
    }

    pub fn parsePrometheusMetrics(self: *Visualizer, content: []const u8) !void {
        var lines = std.mem.splitScalar(u8, content, '\n');

        var current_type: ?MetricType = null;

        while (lines.next()) |line| {
            // Skip empty lines
            if (line.len == 0) continue;

            // Handle comments and TYPE declarations
            if (line[0] == '#') {
                if (std.mem.indexOfScalar(u8, line, 'T')) |t_pos| {
                    if (std.mem.eql(u8, line[t_pos..], "TYPE ")) {
                        const rest = line[t_pos + 5 ..];
                        var parts = std.mem.splitScalar(u8, rest, ' ');
                        _ = parts.next(); // skip metric name
                        const type_str = parts.next() orelse "unknown";

                        // String comparison for switch
                        if (std.mem.eql(u8, type_str, "counter")) {
                            current_type = .counter;
                        } else if (std.mem.eql(u8, type_str, "histogram")) {
                            current_type = .histogram;
                        } else if (std.mem.eql(u8, type_str, "gauge")) {
                            current_type = .gauge;
                        } else {
                            std.log.warn("Unknown type: {s}", .{type_str});
                            current_type = null;
                        }
                    }
                }
                continue;
            }

            if (current_type == null) continue;

            // Parse metric line: name [labels] value
            const last_space = std.mem.findScalarLast(u8, line, ' ') orelse continue;

            const value_str = line[last_space + 1 ..];
            const trimmed_value = std.mem.trim(u8, value_str, " \n\r\t");

            const name_with_labels = line[0..last_space];
            const trimmed_name = std.mem.trimEnd(u8, name_with_labels, " ");
            // std.log.warn("Line {s}: {s}", .{ name_with_labels, trimmed_value });

            const value = std.fmt.parseFloat(f64, trimmed_value) catch continue;
            self.metrics.append(self.allocator, .{
                .name = trimmed_name,
                .m_type = current_type.?,
                .value = value,
                .label = trimmed_name,
            }) catch continue;
        }
    }

    pub fn draw(self: *Visualizer, ctx: vxfw.DrawContext) !vxfw.Surface {
        var sb = compose.surfaceBuilder(ctx.arena);
        var row: i17 = 0;

        for (self.metrics.items) |m| {
            switch (m.m_type) {
                .gauge => {
                    const gauge_val = @as(u8, @intCast(@as(u32, @floor(m.value)) % 101));
                    const gauge: Gauge = .{
                        .value = gauge_val,
                        .label = m.label,
                        .suffix = m.suffix,
                        .label_reserve = 0,
                        .suffix_reserve = 0,
                    };
                    const gauge_surf = try gauge.draw(ui.fixedSize(ctx, ctx.max.width orelse 40, 1));
                    try sb.add(row, 0, gauge_surf);
                    row += 1;
                    if (row < 100) row += 1;
                },
                .histogram => {
                    const braille: BrailleChart = .{
                        .values = m.history,
                        .height = m.chart_height,
                    };
                    const braille_surf = try braille.draw(ui.fixedSize(ctx, (ctx.max.width orelse 40) - 20, m.chart_height));
                    try sb.add(row, 0, braille_surf);
                    row += @as(i17, @intCast(m.chart_height));
                },
                .counter => {
                    // TODO
                },
            }
        }

        return sb.finish(.{ .width = ctx.max.width orelse 40, .height = @max(row, 1) }, ui.drawWidget(self, drawContent));
    }

    fn drawContent(self: *Visualizer, ctx: vxfw.DrawContext) !vxfw.Surface {
        _ = self;
        return try ui.drawRichLine(ctx, &.{
            .{ .text = "Prometheus Metrics", .style = theme.label_style },
        }, ctx.max.width orelse 40);
    }
};

const Model = struct {
    visualizer: Visualizer,

    pub fn handleEvent(self: *Model, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        switch (event) {
            .init => {
                try self.updateMetrics();
                try ctx.tick(1000, ui.widget(self));
            },
            .tick => {
                self.updateMetrics() catch {};
                ctx.redraw = true;
                try ctx.tick(1000, ui.widget(self));
            },
            .key_press => |key| {
                if (key.matches('q', .{})) {
                    ctx.quit = true;
                }
            },
            else => {},
        }
    }

    pub fn draw(self: *Model, ctx: vxfw.DrawContext) !vxfw.Surface {
        return try self.visualizer.draw(ctx);
    }

    pub fn updateMetrics(self: *Model) !void {
        const content = try self.visualizer.fetchMetrics("http://gh200:8001/metrics");
        try self.visualizer.parsePrometheusMetrics(content);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var app = try vxfw.App.init(allocator, init.io);
    defer app.deinit();

    var visualizer = Visualizer.init(allocator, init.io);
    defer visualizer.deinit();

    var model = Model{ .visualizer = visualizer };

    try app.run(ui.widget(&model), .{});
}
