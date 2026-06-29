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

    pub fn init(allocator: std.mem.Allocator) Visualizer {
        return .{
            .metrics = std.ArrayList(Metric).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Visualizer) void {
        self.metrics.deinit();
    }

    pub fn fetch_metrics(self: *Visualizer, url: []const u8) ![]u8 {
        var host = std.ArrayList(u8).init(self.allocator);
        defer host.deinit();

        var port = 80;
        var path = std.ArrayList(u8).init(self.allocator);
        defer path.deinit();

        if (std.mem.indexOf(u8, url, "://") != null) {
            const start = std.mem.indexOf(u8, url, "://") + 3;
            var parts = std.mem.splitSequence(u8, url[start..], "/");
            const domain_part = parts.next() orelse return error.InvalidUrl;

            if (std.mem.indexOf(u8, domain_part, ":")) |_| {
                var d_parts = std.mem.splitSequence(u8, domain_part, ":");
                host.appendSlice(d_parts.next().?.trim(u8, " "));
                if (d_parts.next()) |p| {
                    port = try std.fmt.parseInt(u16, p.trim(u8, " "));
                }
            } else {
                host.appendSlice(domain_part);
            }

            if (parts.next()) |p| {
                path.appendSlice(p);
            }
        } else {
            host.appendSlice(url);
        }

        const address = std.net.Address.parseIp(host.items) catch |e| {
            _ = e; // autofix
            return error.InvalidAddress;
        };

        var stream = try std.net.tcp.Stream.init(address, port);
        try stream.connect();
        defer stream.deinit();

        const request = std.fmt.allocPrint(self.allocator, "GET {s}{s} HTTP/1.1\r\nHost: {s}\r\nConnection: close\r\n\r\n", .{ host.items, path.items, host.items });
        defer self.allocator.free(request);

        try stream.writeAll(request);

        var buffer: [4096]u8 = undefined;
        var content = std.ArrayList(u8).init(self.allocator);
        defer content.deinit();

        while (true) {
            const bytes_read = try stream.read(&buffer);
            if (bytes_read == 0) break;
            try content.appendSlice(&buffer[0..bytes_read]);
        }

        // Very basic HTTP response parsing (split by \r\n\r\n)
        const parts = std.mem.splitSequence(u8, content.items, "\r\n\r\n");
        if (parts.next()) |header| {
            const body_parts = std.mem.splitSequence(u8, header, "\r\n\r\n");
            if (body_parts.next()) |body| {
                return body;
            }
        }

        return content.items;
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

            var parts = std.mem.splitSequence(u8, line, "|");
            const name = parts.next() orelse continue;
            const value_str = parts.next() orelse continue;

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

        for (self.metrics) |m| {
            if (m.m_type == .gauge or m.m_type == .counter) {
                const gauge_val = @as(u8, @intCast(@as(u32, m.value) % 101));
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
