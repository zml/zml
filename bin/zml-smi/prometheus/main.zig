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
    url: []const u8,

    pub fn init(allocator: std.mem.Allocator, url: []const u8) Visualizer {
        return .{
            .metrics = .empty,
            .allocator = allocator,
            .url = url,
        };
    }

    pub fn deinit(self: *Visualizer, allocator: std.mem.Allocator) void {
        self.metrics.deinit(allocator);
    }

    pub fn fetch_metrics(self: *Visualizer, url: []const u8) ![]u8 {
        var host = try std.ArrayList(u8).initCapacity(self.allocator, 64);
        var port: u16 = 80;
        var path = try std.ArrayList(u8).initCapacity(self.allocator, 64);

        if (std.mem.indexOf(u8, url, "://")) |start| {
            const start_idx = start + 3;
            var parts = std.mem.splitSequence(u8, url[start_idx..], "/");
            const domain_part = parts.next() orelse return error.InvalidUrl;

            if (std.mem.indexOf(u8, domain_part, ":")) |_| {
                var d_parts = std.mem.splitSequence(u8, domain_part, ":");
                try host.appendSlice(self.allocator, std.mem.trim(u8, " ", d_parts.next().?));
                if (d_parts.next()) |p| {
                    port = try std.fmt.parseInt(u16, std.mem.trim(u8, " ", p), 10);
                }
            } else {
                try host.appendSlice(self.allocator, domain_part);
            }

            if (parts.next()) |p| {
                try path.appendSlice(self.allocator, p);
            }
        } else {
            try host.appendSlice(self.allocator, url);
        }

        const address = try std.net.Address.parseIp(host.items);

        var stream = try std.net.tcp.Stream.init(address, port);
        try stream.connect();
        defer stream.deinit();

        const request = std.fmt.allocPrint(self.allocator, "GET {s}{s} HTTP/1.1\r\nHost: {s}\r\nConnection: close\r\n\r\n", .{ host.items, path.items, host.items });
        defer self.allocator.free(request);

        try stream.writeAll(request);

        var buffer: [4096]u8 = undefined;
        var content = std.ArrayList(u8).initCapacity(self.allocator, 4096);
        defer content.deinit(self.allocator) catch {};

        while (true) {
            const bytes_read = try stream.read(&buffer);
            if (bytes_read == 0) break;
            try content.appendSlice(&buffer[0..bytes_read]);
        }

        // Very basic HTTP response parsing (split by \r\n\r\n)
        const parts = std.mem.splitSequence(u8, content.items, "\r\n\r\n");
        if (parts.next()) |header| {
            const body_parts = std.mem.splitSequence(u8, header, "\r\n\r\n");
            _ = body_parts; // autofix
            if (parts.next()) |body| {
                _ = content.deinit(self.allocator) catch {};
                _ = path.deinit(self.allocator) catch {};
                _ = host.deinit(self.allocator) catch {};
                return body;
            }

            _ = content.deinit(self.allocator) catch {};
            _ = path.deinit(self.allocator) catch {};
            _ = host.deinit(self.allocator) catch {};
            return content.items;
        }
    }

    pub fn parse_prometheus_metrics(self: *Visualizer, content: []const u8) !void {
        var lines = std.mem.splitSequence(u8, content, "\n");

        var current_type: MetricType = .gauge;

        while (lines.next()) |line| {
            if (std.mem.ltrie(u8, line)) |char| {
                if (char == '#') {
                    if (std.mem.contains(u8, line, "TYPE")) {
                        var parts = std.mem.splitSequence(u8, line, "|");
                        const type_str = parts.next() orelse continue;
                        const trimmed_type = std.mem.trim(u8, " TYPE ", type_str);
                        current_type = switch (trimmed_type.len) {
                            "counter".len => .counter,
                            "histogram".len => .histogram,
                            else => .gauge,
                        };
                    }
                    continue;
                }
            }

            const space_pos = std.mem.lastIndexOf(u8, line, " ") orelse continue;
            const name = line[0..space_pos];
            const value_str = line[space_pos + 1 ..];

            const mut_name = std.mem.trim(u8, " ", name);
            const trimmed_value = std.mem.trim(u8, " ", value_str);

            const value = try std.fmt.parseFloat(f64, trimmed_value);

            try self.metrics.append(.{
                .name = mut_name,
                .m_type = current_type,
                .value = value,
                .label = mut_name,
            });
        }
    }

    pub fn draw(self: *Visualizer, ctx: vxfw.DrawContext) !vxfw.Surface {
        var sb = compose.surfaceBuilder(ctx.arena);
        var row: i17 = 0;

        for (self.metrics.items) |m| {
            if (m.m_type == .gauge or m.m_type == .counter) {
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
            } else if (m.m_type == .histogram) {
                const braille: BrailleChart = .{
                    .values = m.history,
                    .height = m.chart_height,
                };
                const braille_surf = try braille.draw(ui.fixedSize(ctx, (ctx.max.width orelse 40) - 20, m.chart_height));
                try sb.add(row, 0, braille_surf);
                row += @as(i17, @intCast(m.chart_height));
            }
        }

        return sb.finish(.{ .width = ctx.max.width orelse 40, .height = @max(row, 1) }, ui.drawWidget(self, drawContent));
    }

    fn drawContent(self: *Visualizer, ctx: vxfw.DrawContext) !vxfw.Surface {
        _ = self; // autofix
        return try ui.drawRichLine(ctx, &.{
            .{ .text = "Prometheus Metrics", .style = theme.label_style },
        }, ctx.max.width orelse 40);
    }
};

const Model = struct {
    visualizer: Visualizer,
    ctx: *vxfw.EventContext,

    pub fn handleEvent(self: *Model, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        _ = ctx; // autofix
        switch (event) {
            .init => {
                try self.updateMetrics();
                try self.ctx.tick(1000, ui.widget(self));
            },
            .tick => {
                self.updateMetrics() catch {};
                self.ctx.redraw = true;
                try self.ctx.tick(1000, ui.widget(self));
            },
            .key_press => |key| {
                if (key.matches('q', .{})) {
                    self.ctx.quit = true;
                }
            },
            else => {},
        }
    }

    pub fn draw(self: *Model, ctx: vxfw.DrawContext) !vxfw.Surface {
        return try self.visualizer.draw(ctx);
    }

    pub fn updateMetrics(self: *Model) !void {
        const content = try self.visualizer.fetch_metrics("http://gh200:8001/metrics");
        try self.visualizer.parse_prometheus_metrics(content);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var app = try vxfw.App.init(allocator, init.io);
    defer app.deinit();

    var visualizer = Visualizer.init(allocator, "http://gh200:8001/metrics");
    defer visualizer.deinit(allocator);

    var model = Model{
        .visualizer = visualizer,
        .ctx = undefined, // set by app.run
    };

    // Custom run loop to pass context
    try app.run(ui.widget(&model), .{});
}
