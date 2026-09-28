const std = @import("std");
const vaxis = @import("vaxis");
const bench = @import("bench");
const TextInput = vaxis.widgets.TextInput;
const OutputView = @import("output_view.zig");

const Event = union(enum) {
    key_press: vaxis.Key,
    winsize: vaxis.Winsize,
    mouse: vaxis.Mouse,
};

const Field = enum(usize) { endpoint, prompt, batch, maxTokens, temperature, model, reasoning, systemPrompt, maxCompletionTokens, stop, tools, documents, send };
const field_count = @intFromEnum(Field.send);
const dashboard_fields = [_]Field{ .endpoint, .prompt, .batch, .maxTokens, .temperature, .model, .reasoning, .send };
const option_fields = [_]Field{ .reasoning, .temperature, .maxCompletionTokens, .systemPrompt, .stop, .tools, .documents };
const labels = [_][]const u8{ "Endpoint", "Prompt", "Batch", "Max tokens", "Temperature", "Model", "Thinking", "System", "Completion limit", "Stop", "Tools JSON", "Documents JSON" };
const muted: vaxis.Style = .{ .fg = .{ .index = 8 } };
const accent: vaxis.Style = .{ .fg = .{ .index = 6 }, .bold = true };
const green: vaxis.Style = .{ .fg = .{ .index = 2 }, .bold = true };
const amber: vaxis.Style = .{ .fg = .{ .index = 3 } };
const red: vaxis.Style = .{ .fg = .{ .index = 1 } };

const Hit = struct {
    x: i17,
    y: i17,
    width: u16,
    height: u16,
    target: union(enum) { field: Field, pane: usize, back, follow, stop, options, report },
    firstGrapheme: usize = 0,

    fn contains(self: Hit, mouse: vaxis.Mouse) bool {
        return mouse.col >= self.x and mouse.row >= self.y and
            mouse.col < self.x + self.width and mouse.row < self.y + self.height;
    }
};

const App = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    apiKey: ?[]const u8,
    inputs: [field_count]TextInput,
    focus: Field = .send,
    batch: ?*bench.Batch = null,
    scroll: usize = 0,
    wasRunning: bool = false,
    viewing: ?usize = null,
    showOptions: bool = false,
    showReport: bool = false,
    reportPath: ?[]const u8 = null,
    reportWritten: bool = false,
    outputView: OutputView = .{},
    hits: [field_count + bench.max_batch + 4]Hit = undefined,
    hitCount: usize = 0,
    message: []const u8 = "Ready. Edit the settings, then send a batch.",

    fn init(allocator: std.mem.Allocator, io: std.Io, config: bench.Config, apiKey: ?[]const u8) !App {
        var app: App = .{ .allocator = allocator, .io = io, .apiKey = apiKey, .inputs = undefined };
        for (&app.inputs) |*input| input.* = .init(allocator);
        errdefer app.deinit();
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const values = [_][]const u8{
            config.endpoint,
            config.prompt,
            try std.fmt.allocPrint(a, "{d}", .{config.batch}),
            if (config.maxTokens) |n| try std.fmt.allocPrint(a, "{d}", .{n}) else "",
            if (config.temperature) |t| try std.fmt.allocPrint(a, "{d}", .{t}) else "",
            config.model,
            config.reasoningEffort,
            config.systemPrompt,
            if (config.maxCompletionTokens) |n| try std.fmt.allocPrint(a, "{d}", .{n}) else "",
            config.stop,
            config.tools,
            config.documents,
        };
        for (&app.inputs, values) |*input, initial| try input.insertSliceAtCursor(initial);
        return app;
    }

    fn deinit(self: *App) void {
        if (self.batch) |batch| batch.destroy();
        for (&self.inputs) |*input| input.deinit();
        self.outputView.deinit(self.allocator);
    }

    fn value(self: *App, allocator: std.mem.Allocator, field: Field) ![]const u8 {
        const input = &self.inputs[@intFromEnum(field)];
        return std.mem.concat(allocator, u8, &.{ input.buf.firstHalf(), input.buf.secondHalf() });
    }

    fn start(self: *App) !void {
        if (self.batch) |batch| if (batch.summary().active > 0) {
            self.message = "A batch is running. Esc stops it.";
            return;
        };
        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const maxTokens = try self.value(a, .maxTokens);
        const temperature = try self.value(a, .temperature);
        const maxCompletion = try self.value(a, .maxCompletionTokens);
        const config: bench.Config = .{
            .endpoint = try self.value(a, .endpoint),
            .prompt = try self.value(a, .prompt),
            .model = try self.value(a, .model),
            .batch = std.fmt.parseInt(usize, try self.value(a, .batch), 10) catch return error.InvalidBatch,
            .maxTokens = if (maxTokens.len == 0) null else std.fmt.parseInt(u32, maxTokens, 10) catch return error.InvalidMaxTokens,
            .temperature = if (temperature.len == 0) null else std.fmt.parseFloat(f64, temperature) catch return error.InvalidTemperature,
            .maxCompletionTokens = if (maxCompletion.len == 0) null else std.fmt.parseInt(u32, maxCompletion, 10) catch return error.InvalidMaxTokens,
            .reasoningEffort = try self.value(a, .reasoning),
            .systemPrompt = try self.value(a, .systemPrompt),
            .stop = try self.value(a, .stop),
            .tools = try self.value(a, .tools),
            .documents = try self.value(a, .documents),
        };
        try config.validate();
        const next = try bench.Batch.create(self.allocator, self.io, config, self.apiKey);
        if (self.batch) |old| old.destroy();
        self.batch = next;
        self.showOptions = false;
        self.showReport = false;
        self.reportWritten = false;
        self.viewing = null;
        self.outputView.reset();
        self.hitCount = 0;
        self.wasRunning = true;
        self.scroll = 0;
        self.message = "Settings are locked while streaming. Esc stops all requests.";
    }

    fn key(self: *App, k: vaxis.Key) !bool {
        if (k.matches('c', .{ .ctrl = true })) return true;
        if (k.matches('r', .{ .ctrl = true })) {
            self.showReport = !self.showReport;
            self.showOptions = false;
            self.viewing = null;
            self.hitCount = 0;
            return false;
        }
        if (k.matches('o', .{ .ctrl = true })) {
            self.showOptions = !self.showOptions;
            self.showReport = false;
            self.viewing = null;
            self.focus = if (self.showOptions) .reasoning else .send;
            self.hitCount = 0;
            return false;
        }
        if (self.showReport) {
            if (k.matches(vaxis.Key.escape, .{})) self.showReport = false;
            return false;
        }
        if (self.showOptions and k.matches(vaxis.Key.escape, .{})) {
            self.showOptions = false;
            self.focus = .send;
            return false;
        }
        if (self.viewing != null) {
            if (k.matches(vaxis.Key.escape, .{}) or k.matches('q', .{})) {
                self.viewing = null;
                self.hitCount = 0;
            } else if (k.matches(vaxis.Key.up, .{})) {
                self.outputView.scrollBy(-1);
            } else if (k.matches(vaxis.Key.down, .{})) {
                self.outputView.scrollBy(1);
            } else if (k.matches(vaxis.Key.page_up, .{})) {
                self.outputView.scrollBy(-@as(i32, self.outputView.height));
            } else if (k.matches(vaxis.Key.page_down, .{})) {
                self.outputView.scrollBy(self.outputView.height);
            } else if (k.matches(vaxis.Key.home, .{})) {
                self.outputView.home();
            } else if (k.matches(vaxis.Key.end, .{}) or k.matches('f', .{})) {
                self.outputView.following = true;
            }
            return false;
        }
        if (k.matches('s', .{ .ctrl = true }) or (self.focus == .send and k.matches(vaxis.Key.enter, .{}))) {
            self.start() catch |err| {
                self.message = @errorName(err);
            };
        } else if (k.matches(vaxis.Key.escape, .{})) {
            if (self.batch) |batch| batch.stop();
            self.message = "Stopped. Edit settings or send another batch.";
        } else if (k.matches(vaxis.Key.tab, .{})) {
            self.moveFocus(false);
        } else if (k.matches(vaxis.Key.tab, .{ .shift = true })) {
            self.moveFocus(true);
        } else if (k.matches(vaxis.Key.page_down, .{})) {
            self.scroll +|= 2;
        } else if (k.matches(vaxis.Key.page_up, .{})) {
            self.scroll -|= 2;
        } else if (self.focus != .send) {
            if (self.batch) |batch| if (batch.summary().active > 0) return false;
            const input = &self.inputs[@intFromEnum(self.focus)];
            if (input.buf.firstHalf().len + input.buf.secondHalf().len < 16 * 1024 or k.text == null)
                try input.update(.{ .key_press = k });
        }
        return false;
    }

    fn moveFocus(self: *App, backwards: bool) void {
        const fields: []const Field = if (self.showOptions) &option_fields else &dashboard_fields;
        const index = std.mem.indexOfScalar(Field, fields, self.focus) orelse 0;
        self.focus = fields[(index + if (backwards) fields.len - 1 else @as(usize, 1)) % fields.len];
    }

    fn saveReport(self: *App) !void {
        if (self.reportWritten) return;
        if (self.reportPath) |path| if (self.batch) |batch| {
            try batch.saveReport(path);
            self.reportWritten = true;
        };
    }

    fn hit(self: *App, win: vaxis.Window, target: @FieldType(Hit, "target")) void {
        std.debug.assert(self.hitCount < self.hits.len);
        self.hits[self.hitCount] = .{ .x = win.x_off, .y = win.y_off, .width = win.width, .height = win.height, .target = target };
        self.hitCount += 1;
    }

    fn mouse(self: *App, event: vaxis.Mouse, win: vaxis.Window) void {
        if (event.type != .press) return;
        if (event.button == .wheel_up or event.button == .wheel_down) {
            if (self.viewing != null) {
                self.outputView.scrollBy(if (event.button == .wheel_up) -3 else 3);
            } else {
                self.scroll = if (event.button == .wheel_up) self.scroll -| 1 else self.scroll +| 1;
            }
            return;
        }
        if (event.button != .left) return;
        for (self.hits[0..self.hitCount]) |region| {
            if (!region.contains(event)) continue;
            switch (region.target) {
                .field => |field| {
                    self.focus = field;
                    if (field == .send) {
                        self.start() catch |err| {
                            self.message = @errorName(err);
                        };
                    } else {
                        const input = &self.inputs[@intFromEnum(field)];
                        const inputX = region.x + 1 + @as(i17, @intCast(labels[@intFromEnum(field)].len + 2));
                        if (event.col >= inputX) {
                            input.buf.moveGapLeft(input.buf.firstHalf().len);
                            var iter = vaxis.unicode.graphemeIterator(input.buf.secondHalf());
                            var bytes: usize = 0;
                            var col: usize = 0;
                            var graphemeIndex: usize = 0;
                            while (iter.next()) |grapheme| : (graphemeIndex += 1) {
                                const text = grapheme.bytes(input.buf.secondHalf());
                                if (graphemeIndex >= region.firstGrapheme) {
                                    const width = win.gwidth(text);
                                    if (col + width > event.col - inputX) break;
                                    col += width;
                                }
                                bytes += text.len;
                            }
                            input.buf.moveGapRight(bytes);
                        }
                    }
                },
                .pane => |index| {
                    self.viewing = index;
                    self.outputView.reset();
                    self.hitCount = 0;
                },
                .back => {
                    self.showOptions = false;
                    self.showReport = false;
                    self.focus = .send;
                    self.viewing = null;
                    self.hitCount = 0;
                },
                .options => {
                    self.showOptions = true;
                    self.focus = .reasoning;
                    self.hitCount = 0;
                },
                .report => {
                    self.showReport = true;
                    self.hitCount = 0;
                },
                .follow => self.outputView.following = true,
                .stop => if (self.batch) |batch| batch.stop(),
            }
            return;
        }
    }

    fn drawOptions(self: *App, win: vaxis.Window) void {
        if (win.width < 68 or win.height < 24) {
            put(win, 1, 1, "Resize to 68 x 24. Esc returns.", accent);
            return;
        }
        const back = win.child(.{ .x_off = 2, .width = 12, .height = 1 });
        put(back, 0, 0, "< Back", accent);
        self.hit(back, .back);
        put(win, 16, 0, "REQUEST OPTIONS", accent);
        put(win, 2, 1, "Thinking: auto/off/on, effort name, or 0-100. Model dependent.", muted);
        for (option_fields, 0..) |field, i| {
            self.drawField(win, field, 1, @intCast(3 + i * (if (win.height < 34) @as(usize, 2) else 4)), win.width - 2);
        }
        put(win, 2, win.height - 2, "Stop: endpoint dependent (currently ignored by llmd). JSON tools/docs.", muted);
        put(win, 2, win.height - 1, "Tab fields  Ctrl-S send  Esc back  Ctrl-C quit", accent);
    }

    fn drawReport(self: *App, win: vaxis.Window, a: std.mem.Allocator) !void {
        if (win.width < 68 or win.height < 24) {
            put(win, 1, 1, "Resize to 68 x 24. Esc returns.", accent);
            return;
        }
        const back = win.child(.{ .x_off = 2, .width = 12, .height = 1 });
        put(back, 0, 0, "< Back", accent);
        self.hit(back, .back);
        const batch = self.batch orelse {
            put(win, 2, 3, "Send a batch to see its report.", muted);
            return;
        };
        const summary = batch.summary();
        put(win, 16, 0, if (summary.active > 0) "LIVE REPORT" else "BATCH REPORT", accent);
        put(win, 2, 2, try std.fmt.allocPrint(a, "{d} completed  {d} active  {d} failed  {d} canceled", .{ summary.completed, summary.active, summary.failed, summary.stopped }), .{});
        put(win, 2, 3, try std.fmt.allocPrint(a, "{d:.2}s  {s}{d:.1} tok/s  {d:.2} req/s  {s}{d} output tokens", .{
            @as(f64, @floatFromInt(summary.elapsedMs)) / 1000, if (summary.estimated) "~" else "",
            summary.aggregateRate,                             summary.requestsPerSecond,
            if (summary.estimated) "~" else "",                summary.tokens,
        }), green);
        put(win, 2, 4, try std.fmt.allocPrint(a, "{d} prompt tokens ({d}/{d} reported)  {d} output chunks", .{
            summary.promptTokens, summary.promptUsageRequests, batch.results.len, summary.chunks,
        }), muted);
        put(win, 2, 6, "TIMING (ms)      COUNT     MEAN      P50      P95      P99      MAX", accent);
        const names = [_][]const u8{ "TTFT", "ITL/chunk ~", if (summary.estimated) "TPOT ~" else "TPOT", "Request latency", "First answer" };
        const stats = [_]bench.metrics.Stats{ summary.ttft, summary.itl, summary.tpot, summary.latency, summary.firstAnswer };
        for (names, stats, 0..) |name, stat, i| {
            put(win, 2, @intCast(8 + i), try std.fmt.allocPrint(a, "{s:<16}{d:>5} {s:>8} {s:>8} {s:>8} {s:>8} {s:>8}", .{
                name,                        stat.count,                  try formatMs(a, stat.meanMs), try formatMs(a, stat.p50Ms),
                try formatMs(a, stat.p95Ms), try formatMs(a, stat.p99Ms), try formatMs(a, stat.maxMs),
            }), .{});
        }
        put(win, 2, 15, "ITL = gaps between nonempty SSE chunks, not individual tokens.", muted);
        put(win, 2, 16, "ITL percentiles (~) use a bounded histogram. TPOT uses usage.", muted);
        put(win, 2, 17, "TPOT/token rates marked ~ until server token counts arrive.", muted);
        put(win, 2, 18, "Request latency includes only completed requests.", muted);
        if (self.reportPath) |path| put(win, 2, 20, try std.fmt.allocPrint(a, "JSON {s}: {s}", .{ if (self.reportWritten) "saved" else "pending", path }), accent);
        put(win, 2, win.height - 1, "Esc / Back: dashboard   Ctrl-R: toggle report   Ctrl-C: quit", accent);
    }

    fn drawDetail(self: *App, win: vaxis.Window, a: std.mem.Allocator, index: usize) !void {
        if (win.width < 48 or win.height < 12) {
            put(win, 1, 1, "Resize to at least 48 x 12. Esc returns.", accent);
            return;
        }
        const back = win.child(.{ .x_off = 2, .width = 12, .height = 1 });
        put(back, 0, 0, "< Back", accent);
        self.hit(back, .back);
        const follow = win.child(.{ .x_off = win.width - 28, .width = 12, .height = 1 });
        put(follow, 0, 0, "[ Follow ]", accent);
        self.hit(follow, .follow);
        const stop = win.child(.{ .x_off = win.width - 14, .width = 12, .height = 1 });
        put(stop, 0, 0, "[ Stop ]", red);
        self.hit(stop, .stop);

        const batch = self.batch.?;
        batch.mutex.lockUncancelable(self.io);
        defer batch.mutex.unlock(self.io);
        const result = &batch.results[index];
        put(win, 2, 1, try std.fmt.allocPrint(a, "REQUEST {d:0>2}  {s}  {s}{d:.1} tok/s", .{
            index + 1, @tagName(result.status), if (result.completionTokens == null) "~" else "", result.rate(batch.elapsed()),
        }), accent);
        const output = panel(win, 1, 4, win.width - 2, win.height - 6, muted);
        try self.outputView.draw(self.allocator, a, output, result.output.items);
        if (result.output.items.len == 0)
            put(output, 0, 0, if (result.active()) "Waiting for first token..." else "No output received.", muted);
        put(win, 2, 2, try std.fmt.allocPrint(a, "{s}  |  rows {d}-{d}/{d}  |  {s}{d} tokens", .{
            if (self.outputView.following) "Following live output" else "Scrollback",
            self.outputView.top + 1,
            @min(self.outputView.rows.items.len, self.outputView.top + output.height),
            self.outputView.rows.items.len,
            if (result.completionTokens == null) "~" else "",
            result.tokens(),
        }), muted);
        put(win, 2, 3, try std.fmt.allocPrint(a, "TTFT {s}ms  ITL {s}ms  TPOT {s}{s}ms  decode {s}{s} tok/s", .{
            try formatMs(a, result.ttftMs()),                 try formatMs(a, result.itl.stats().meanMs),
            if (result.completionTokens == null) "~" else "", try formatMs(a, result.tpotMs()),
            if (result.completionTokens == null) "~" else "", try formatMs(a, result.decodeRate()),
        }), muted);
        if (result.messageLen > 0) put(win, 2, 3, try a.dupe(u8, result.message[0..result.messageLen]), red);
        put(win, 2, win.height - 1, "Wheel/arrows scroll  PgUp/PgDn  Home/End  Esc back  Ctrl-C quit", accent);
    }

    fn drawField(self: *App, win: vaxis.Window, field: Field, x: u16, y: u16, width: u16) void {
        const selected = self.focus == field;
        self.hit(win.child(.{ .x_off = x, .y_off = y, .width = width, .height = if (win.height < 34) 1 else 3 }), .{ .field = field });
        const child = if (win.height < 34) win.child(.{ .x_off = x + 1, .y_off = y, .width = width -| 2, .height = 1 }) else panel(win, x, y, width, 3, if (selected) accent else muted);
        const label = labels[@intFromEnum(field)];
        put(child, 0, 0, label, if (selected) accent else muted);
        const offset: u16 = @intCast(label.len + 2);
        const inputWin = child.child(.{ .x_off = offset, .height = 1 });
        const input = &self.inputs[@intFromEnum(field)];
        if (selected) {
            input.draw(inputWin);
        } else {
            _ = inputWin.print(&.{ .{ .text = input.buf.firstHalf() }, .{ .text = input.buf.secondHalf() } }, .{ .wrap = .none });
        }
        self.hits[self.hitCount - 1].firstGrapheme = if (selected) input.draw_offset else 0;
        if (input.buf.firstHalf().len + input.buf.secondHalf().len == 0)
            put(inputWin, 0, 0, switch (field) {
                .maxTokens, .maxCompletionTokens => "server limit",
                .reasoning, .temperature => "auto",
                else => "not set",
            }, muted);
    }

    fn draw(self: *App, win: vaxis.Window, a: std.mem.Allocator) !void {
        win.clear();
        win.hideCursor();
        self.hitCount = 0;
        if (self.showOptions) return self.drawOptions(win);
        if (self.showReport) return self.drawReport(win, a);
        if (self.viewing) |index| return self.drawDetail(win, a, index);
        if (win.width < 68 or win.height < 24) {
            put(win, 1, 1, "zml-bench | Resize to at least 68 x 24", accent);
            put(win, 1, 3, "Ctrl-S send  /  Esc stop  /  Ctrl-C quit", muted);
            return;
        }
        const width = win.width - 2;
        put(win, 2, 0, "BENCHMARK", .{ .bold = true });
        put(win, 15, 0, "OpenAI benchmark", muted);
        const options = win.child(.{ .x_off = win.width - 25, .width = 12, .height = 1 });
        put(options, 0, 0, "[ Options ]", accent);
        self.hit(options, .options);
        const report = win.child(.{ .x_off = win.width - 13, .width = 12, .height = 1 });
        put(report, 0, 0, "[ Report ]", accent);
        self.hit(report, .report);
        put(win, 2, 1, "Concurrent requests. Live output. Time to first token.", muted);
        const top: u16 = if (win.height < 34) 2 else 3;
        const step: u16 = if (win.height < 34) 2 else 3;
        self.drawField(win, .endpoint, 1, top, width);
        self.drawField(win, .prompt, 1, top + step, width);
        const third = width / 3;
        self.drawField(win, .batch, 1, top + step * 2, third);
        self.drawField(win, .maxTokens, 1 + third, top + step * 2, third);
        self.drawField(win, .temperature, 1 + third * 2, top + step * 2, width - third * 2);
        const modelWidth = width - 42;
        self.drawField(win, .model, 1, top + step * 3, modelWidth);
        self.drawField(win, .reasoning, 1 + modelWidth, top + step * 3, 23);
        const button = if (win.height < 34) win.child(.{ .x_off = width - 17, .y_off = top + step * 3, .width = 19, .height = 1 }) else panel(win, width - 17, top + step * 3, 19, 3, if (self.focus == .send) accent else muted);
        self.hit(win.child(.{ .x_off = width - 17, .y_off = top + step * 3, .width = 19, .height = if (win.height < 34) 1 else 3 }), .{ .field = .send });
        put(button, 1, 0, "> Send batch", if (self.focus == .send) accent else .{});

        const summary = if (self.batch) |batch| batch.summary() else bench.Summary{};
        const metricsY = top + step * 4;
        const metrics = panel(win, 1, metricsY, width, 4, muted);
        const cols = [_][]const u8{ "AGGREGATE", "AVG / REQ", "COMPLETED", "MEDIAN TTFT", "ELAPSED" };
        const estimate = if (summary.estimated) "~" else "";
        const values = [_][]const u8{
            try std.fmt.allocPrint(a, "{s}{d:.1} tok/s", .{ estimate, summary.aggregateRate }),
            try std.fmt.allocPrint(a, "{s}{d:.1} tok/s", .{ estimate, summary.averageRate }),
            try std.fmt.allocPrint(a, "{d}/{d}", .{ summary.completed, if (self.batch) |batch| batch.results.len else @as(usize, 0) }),
            if (summary.medianTtftMs) |ms| try std.fmt.allocPrint(a, "{d:.0} ms", .{ms}) else "--",
            try std.fmt.allocPrint(a, "{d:.1} s", .{@as(f64, @floatFromInt(summary.elapsedMs)) / 1000}),
        };
        for (cols, values, 0..) |label, text, i| {
            const x: u16 = @intCast(i * (metrics.width / 5));
            put(metrics, x, 0, label, muted);
            put(metrics, x, 1, text, if (i == 0) green else .{ .bold = true });
        }
        put(win, 2, metricsY + 5, try std.fmt.allocPrint(a, "ITL/chunk {s}ms  p95 ~{s}ms  TPOT {s}{s}ms  req/s {d:.2}", .{
            try formatMs(a, summary.itl.meanMs), try formatMs(a, summary.itl.p95Ms),
            if (summary.estimated) "~" else "",  try formatMs(a, summary.tpot.meanMs),
            summary.requestsPerSecond,
        }), muted);
        const status = if (self.batch == null) "READY" else if (summary.active > 0) "STREAMING" else if (summary.failed + summary.stopped > 0) "FINISHED WITH ERRORS / STOPPED" else "COMPLETE";
        put(win, 2, metricsY + 4, try std.fmt.allocPrint(a, "{s} | {d} failed {d} stopped | ~ estimates: UTF-8 bytes / 4", .{ status, summary.failed, summary.stopped }), if (summary.active > 0) amber else muted);
        put(win, 2, win.height - 2, self.message, muted);
        put(win, 2, win.height - 1, "Click fields/panes  ^S send  ^O options  ^R report  Esc stop  ^C quit", accent);

        const cards = win.child(.{ .x_off = 1, .y_off = metricsY + 6, .width = width, .height = win.height - metricsY - 9 });
        if (self.batch) |batch| {
            batch.mutex.lockUncancelable(self.io);
            defer batch.mutex.unlock(self.io);
            const columns: usize = if (width >= 100) 2 else 1;
            const cardHeight: u16 = @min(8, cards.height);
            const visibleRows = @max(1, cards.height / cardHeight);
            const totalRows = (batch.results.len + columns - 1) / columns;
            self.scroll = @min(self.scroll, totalRows -| visibleRows);
            const cardWidth: u16 = @intCast(width / columns);
            const elapsed = batch.elapsed();
            for (0..visibleRows) |row| {
                for (0..columns) |col| {
                    const index = (row + self.scroll) * columns + col;
                    if (index >= batch.results.len) break;
                    const result = &batch.results[index];
                    self.hit(cards.child(.{ .x_off = @intCast(col * cardWidth), .y_off = @intCast(row * cardHeight), .width = cardWidth, .height = cardHeight }), .{ .pane = index });
                    const card = panel(cards, @intCast(col * cardWidth), @intCast(row * cardHeight), cardWidth, cardHeight, muted);
                    const color: vaxis.Style = switch (result.status) {
                        .completed => green,
                        .failed => red,
                        .connecting, .streaming, .canceled => amber,
                    };
                    const title = try std.fmt.allocPrint(a, "{d:0>2}  {s}    {s}{d:.1} tok/s", .{ index + 1, @tagName(result.status), if (result.completionTokens == null) "~" else "", result.rate(elapsed) });
                    put(card, 1, 0, title, color);
                    const output = card.child(.{ .x_off = 1, .y_off = 1, .width = card.width -| 2, .height = card.height -| 2 });
                    const text = result.output.items;
                    var tailStart = text.len -| (@as(usize, output.width) * output.height);
                    while (tailStart < text.len and text[tailStart] & 0xc0 == 0x80) : (tailStart += 1) {}
                    // Vaxis retains text slices through render, after this mutex is released.
                    const visible = if (result.messageLen > 0) result.message[0..result.messageLen] else if (text.len > 0) text[tailStart..] else "Waiting for first token...";
                    const snapshot = try a.dupe(u8, visible);
                    _ = output.print(&.{.{ .text = snapshot, .style = if (result.messageLen > 0) red else .{} }}, .{});
                    const ttft = if (result.ttftMs()) |ms| try std.fmt.allocPrint(a, "{d:.1}ms", .{ms}) else "--";
                    const itl = try formatMs(a, result.itl.stats().meanMs);
                    const latency = if (result.latencyMs()) |ms| try std.fmt.allocPrint(a, "{d:.1}s", .{ms / 1000}) else "--";
                    put(card, 1, card.height -| 1, try std.fmt.allocPrint(a, "ttft {s} itl {s} tok {s}{d} lat {s}", .{ ttft, itl, if (result.completionTokens == null) "~" else "", result.tokens(), latency }), muted);
                }
            }
        } else {
            put(cards, 2, 1, "Your request streams will appear here.", muted);
            put(cards, 2, 3, "Blank max tokens and temperature use the server defaults.", muted);
        }
    }
};

fn formatMs(allocator: std.mem.Allocator, value: ?f64) ![]const u8 {
    return if (value) |ms| std.fmt.allocPrint(allocator, "{d:.2}", .{ms}) else "--";
}

fn panel(win: vaxis.Window, x: u16, y: u16, width: u16, height: u16, style: vaxis.Style) vaxis.Window {
    return win.child(.{ .x_off = x, .y_off = y, .width = width, .height = height, .border = .{ .where = .all, .style = style } });
}

fn put(win: vaxis.Window, x: u16, y: u16, text: []const u8, style: vaxis.Style) void {
    _ = win.print(&.{.{ .text = text, .style = style }}, .{ .col_offset = x, .row_offset = y, .wrap = .none });
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, config: bench.Config, apiKey: ?[]const u8, reportPath: ?[]const u8) !void {
    var app = try App.init(allocator, io, config, apiKey);
    defer app.deinit();
    app.reportPath = reportPath;
    defer {
        if (app.batch) |batch| if (batch.summary().active > 0) {
            batch.stop();
        };
        app.saveReport() catch |err| std.log.err("Could not save benchmark report: {s}", .{@errorName(err)});
    }
    var tty = try vaxis.Tty.init(io);
    defer tty.deinit();
    var vx = try vaxis.init(allocator, .{});
    defer vx.deinit(allocator, tty.writer());
    var loop: vaxis.Loop(Event) = .{ .tty = &tty, .vaxis = &vx, .io = io, .queue = .{ .io = io } };
    try loop.init();
    try loop.start();
    defer loop.stop();
    try vx.enterAltScreen(tty.writer());
    try vx.queryTerminal(tty.writer(), io, std.time.ns_per_s);
    try vx.resize(allocator, tty.writer(), try vaxis.Tty.getWinsize(tty.fd));
    try vx.setMouseMode(tty.writer(), true);
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    while (true) {
        vaxis.Tty.pollWinch();
        while (loop.tryEvent()) |event| switch (event) {
            .key_press => |key| if (try app.key(key)) return,
            .winsize => |size| {
                app.hitCount = 0;
                try vx.resize(allocator, tty.writer(), size);
            },
            .mouse => |mouse| app.mouse(mouse, vx.window()),
        };
        if (app.batch) |batch| {
            const running = batch.summary().active > 0;
            if (app.wasRunning and !running) {
                app.message = "Batch finished. Click Report for metrics or send another batch.";
                app.saveReport() catch |err| {
                    app.message = @errorName(err);
                };
            }
            app.wasRunning = running;
        }
        _ = arena.reset(.retain_capacity);
        try app.draw(vx.window(), arena.allocator());
        try vx.render(tty.writer());
        try tty.writer().flush();
        try io.sleep(.fromMilliseconds(50), .awake);
    }
}

test "mouse focuses fields and positions the cursor by terminal columns" {
    const allocator = std.testing.allocator;
    var app = try App.init(allocator, std.testing.io, .{ .prompt = "ab界cd" }, null);
    defer app.deinit();
    var vx = try vaxis.init(allocator, .{});
    var discard: std.Io.Writer.Discarding = .init(&.{});
    defer vx.deinit(allocator, &discard.writer);
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    for ([_]vaxis.Winsize{ .{ .cols = 120, .rows = 42, .x_pixel = 0, .y_pixel = 0 }, .{ .cols = 80, .rows = 24, .x_pixel = 0, .y_pixel = 0 } }) |size| {
        try vx.resize(allocator, &discard.writer, size);
        vx.screen.width_method = .unicode;
        try app.draw(vx.window(), arena.allocator());
        const row: i16 = if (size.rows < 34) 4 else 7;
        app.mouse(.{ .col = 14, .row = row, .button = .left, .mods = .{}, .type = .press }, vx.window());
        try std.testing.expectEqual(Field.prompt, app.focus);
        try std.testing.expectEqualStrings("ab界", app.inputs[@intFromEnum(Field.prompt)].buf.firstHalf());
        app.mouse(.{ .col = 2, .row = 3, .button = .left, .mods = .{}, .type = .release }, vx.window());
        try std.testing.expectEqual(Field.prompt, app.focus);
    }
}

test "clicking a pane opens the correct request and Escape returns without canceling" {
    const allocator = std.testing.allocator;
    var app = try App.init(allocator, std.testing.io, .{}, null);
    defer app.deinit();
    var results: [2]bench.Result = .{ .{ .status = .streaming }, .{ .status = .streaming } };
    var batch: bench.Batch = .{
        .allocator = allocator,
        .io = std.testing.io,
        .results = &results,
        .url = &.{},
        .body = &.{},
        .authorization = &.{},
        .started = .now(std.testing.io, .awake),
    };
    app.batch = &batch;
    defer app.batch = null;
    var vx = try vaxis.init(allocator, .{});
    var discard: std.Io.Writer.Discarding = .init(&.{});
    defer vx.deinit(allocator, &discard.writer);
    try vx.resize(allocator, &discard.writer, .{ .cols = 120, .rows = 42, .x_pixel = 0, .y_pixel = 0 });
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    try app.draw(vx.window(), arena.allocator());
    app.mouse(.{ .col = 70, .row = 23, .button = .left, .mods = .{}, .type = .press }, vx.window());
    try std.testing.expectEqual(@as(?usize, 1), app.viewing);
    try app.draw(vx.window(), arena.allocator());
    try std.testing.expect(!try app.key(.{ .codepoint = vaxis.Key.escape }));
    try std.testing.expectEqual(null, app.viewing);
    try std.testing.expectEqual(bench.Status.streaming, results[1].status);
}

test "dashboard and detail frames survive streaming output replacement before render" {
    const allocator = std.testing.allocator;
    // Unmap old output on release so a stale renderer reference cannot pass by chance.
    const outputAllocator = std.heap.page_allocator;
    for ([_]?usize{ null, 0 }) |viewing| {
        var app = try App.init(allocator, std.testing.io, .{}, null);
        defer app.deinit();
        var results: [1]bench.Result = .{.{ .status = .streaming }};
        defer results[0].deinit(outputAllocator);
        try results[0].output.resize(outputAllocator, 64 * 1024);
        @memset(results[0].output.items, 'a');
        var batch: bench.Batch = .{
            .allocator = allocator,
            .io = std.testing.io,
            .results = &results,
            .url = &.{},
            .body = &.{},
            .authorization = &.{},
            .started = .now(std.testing.io, .awake),
        };
        app.batch = &batch;
        defer app.batch = null;
        app.viewing = viewing;
        var vx = try vaxis.init(allocator, .{});
        var discardBuffer: [4096]u8 = undefined;
        var discard: std.Io.Writer.Discarding = .init(&discardBuffer);
        defer vx.deinit(allocator, &discard.writer);
        try vx.resize(allocator, &discard.writer, .{ .cols = 120, .rows = 42, .x_pixel = 0, .y_pixel = 0 });
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();

        try app.draw(vx.window(), arena.allocator());
        try vx.render(&discard.writer);
        try app.draw(vx.window(), arena.allocator());
        const col: u16 = if (viewing == null) 3 else 2;
        const row: u16 = if (viewing == null) 23 else 5;
        try std.testing.expectEqualStrings("a", vx.window().readCell(col, row).?.char.grapheme);

        @memset(results[0].output.items, 'b');
        try std.testing.expectEqualStrings("a", vx.window().readCell(col, row).?.char.grapheme);
        var oldOutput = results[0].output;
        results[0].output = .empty;
        defer oldOutput.deinit(outputAllocator);
        try results[0].output.resize(outputAllocator, 128 * 1024);
        @memset(results[0].output.items, 'b');
        oldOutput.clearAndFree(outputAllocator);
        try vx.render(&discard.writer);
        try discard.writer.flush();

        try app.draw(vx.window(), arena.allocator());
        try std.testing.expectEqualStrings("b", vx.window().readCell(col, row).?.char.grapheme);
        try vx.render(&discard.writer);
    }
}

test "options and report navigation preserve a running batch and expose only visible fields" {
    const allocator = std.testing.allocator;
    var app = try App.init(allocator, std.testing.io, .{ .reasoningEffort = "75", .systemPrompt = "Be brief" }, null);
    defer app.deinit();
    var results: [1]bench.Result = .{.{ .status = .streaming }};
    var batch: bench.Batch = .{
        .allocator = allocator,
        .io = std.testing.io,
        .results = &results,
        .url = &.{},
        .body = &.{},
        .authorization = &.{},
        .started = .now(std.testing.io, .awake),
    };
    app.batch = &batch;
    defer app.batch = null;
    var vx = try vaxis.init(allocator, .{});
    var discardBuffer: [4096]u8 = undefined;
    var discard: std.Io.Writer.Discarding = .init(&discardBuffer);
    defer vx.deinit(allocator, &discard.writer);
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    for ([_]vaxis.Winsize{ .{ .cols = 120, .rows = 42, .x_pixel = 0, .y_pixel = 0 }, .{ .cols = 68, .rows = 24, .x_pixel = 0, .y_pixel = 0 } }) |size| {
        try vx.resize(allocator, &discard.writer, size);
        try app.draw(vx.window(), arena.allocator());
        app.mouse(.{ .col = @intCast(size.cols - 24), .row = 0, .button = .left, .mods = .{}, .type = .press }, vx.window());
        try std.testing.expect(app.showOptions);
        try app.draw(vx.window(), arena.allocator());
        try vx.render(&discard.writer);
        const row: i16 = if (size.rows < 34) 9 else 16;
        app.mouse(.{ .col = 20, .row = row, .button = .left, .mods = .{}, .type = .press }, vx.window());
        try std.testing.expectEqual(Field.systemPrompt, app.focus);
        _ = try app.key(.{ .codepoint = 'x', .text = "x" });
        try std.testing.expectEqualStrings("Be brief", try app.value(arena.allocator(), .systemPrompt));
        _ = try app.key(.{ .codepoint = vaxis.Key.tab });
        try std.testing.expectEqual(Field.stop, app.focus);
        _ = try app.key(.{ .codepoint = vaxis.Key.escape });
        try std.testing.expect(!app.showOptions);
        try std.testing.expect(results[0].active());
        _ = try app.key(.{ .codepoint = 'r', .mods = .{ .ctrl = true } });
        try std.testing.expect(app.showReport);
        try app.draw(vx.window(), arena.allocator());
        try vx.render(&discard.writer);
        app.mouse(.{ .col = 4, .row = 0, .button = .left, .mods = .{}, .type = .press }, vx.window());
        try std.testing.expect(!app.showReport);
        try std.testing.expect(results[0].active());
    }
}
