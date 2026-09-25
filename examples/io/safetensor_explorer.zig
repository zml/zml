const std = @import("std");
const zml = @import("zml");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const tensor_tree = @import("tensor_tree.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    model: []const u8,

    pub const help =
        \\Usage: safetensor-explorer --model=<MODEL_URI>
        \\
        \\Explore a model repository, safetensors file, or shard index.
        \\Supports local paths, file://, hf://, s3://, gs://, HTTP, and HTTPS.
        \\
        \\Up/Down: select   Enter: details   Left/Right: collapse/expand
        \\Space: toggle parent   /: search   Esc: clear search   q: quit
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |path| {
        var dir = try std.Io.Dir.openDirAbsolute(init.io, path, .{});
        defer dir.close(init.io);
        try std.process.setCurrentDir(init.io, dir);
    }

    var client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer client.deinit();
    try client.initDefaultProxies(allocator, init.environ_map);

    var file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer file.deinit();
    var https: zml.io.VFS.HTTP = try .init(allocator, init.io, &client, .https);
    defer https.deinit();
    var http: zml.io.VFS.HTTP = try .init(allocator, init.io, &client, .http);
    defer http.deinit();
    var hf: zml.io.VFS.HF = try .auto(allocator, init.io, &client, init.environ_map);
    defer hf.deinit();
    var s3: zml.io.VFS.S3 = try .auto(allocator, init.io, &client, init.environ_map);
    defer s3.deinit();
    var gcs: zml.io.VFS.GCS = try .auto(allocator, init.io, &client, init.environ_map);
    defer gcs.deinit();
    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();
    try vfs.register("file", file.io());
    try vfs.register("https", https.io());
    try vfs.register("http", http.io());
    try vfs.register("hf", hf.io());
    try vfs.register("s3", s3.io());
    try vfs.register("gs", gcs.io());

    std.log.info("Reading tensor metadata from {s}...", .{args.model});
    var registry = try loadRegistry(allocator, vfs.io(), args.model);
    defer registry.deinit();
    var model: Model = .{
        .allocator = allocator,
        .uri = args.model,
        .tree = try tensor_tree.Tree.init(allocator, &registry),
        .tensorCount = registry.tensors.count(),
        .totalBytes = registry.totalBytes(),
    };
    defer model.tree.deinit();
    defer model.query.deinit(allocator);

    // Terminal operations use the host IO; only model reads go through the VFS.
    var app = try vxfw.App.init(allocator, init.io);
    defer app.deinit();
    try app.run(model.widget(), .{});
}

fn loadRegistry(allocator: std.mem.Allocator, io: std.Io, uri: []const u8) !zml.safetensors.TensorRegistry {
    if (std.mem.endsWith(u8, uri, ".safetensors") or std.mem.endsWith(u8, uri, ".safetensors.index.json")) {
        const parent = std.fs.path.dirname(uri) orelse ".";
        const repo = try std.Io.Dir.openDir(.cwd(), io, parent, .{});
        defer repo.close(io);
        const entrypoint = try repo.openFile(io, std.fs.path.basename(uri), .{});
        defer entrypoint.close(io);
        return zml.safetensors.fetchRegistry(allocator, io, repo, entrypoint);
    }
    const repo = try zml.safetensors.resolveModelRepo(io, uri);
    defer repo.close(io);
    return .fromRepo(allocator, io, repo);
}

const accent: vaxis.Style = .{ .bold = true, .fg = .{ .index = 6 } };
const dim: vaxis.Style = .{ .fg = .{ .index = 8 } };

const Model = struct {
    allocator: std.mem.Allocator,
    uri: []const u8,
    tree: tensor_tree.Tree,
    tensorCount: usize,
    totalBytes: u64,
    query: std.ArrayList(u8) = .empty,
    editingSearch: bool = false,
    detail: ?zml.safetensors.Tensor = null,
    top: usize = 0,
    detailScroll: u16 = 0,

    fn widget(self: *Model) vxfw.Widget {
        return .{ .userdata = self, .eventHandler = handleEvent, .drawFn = draw };
    }

    fn handleEvent(ptr: *anyopaque, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        const self: *Model = @ptrCast(@alignCast(ptr));
        switch (event) {
            .key_press => |key| {
                if (key.matches('c', .{ .ctrl = true }) or (!self.editingSearch and key.matches('q', .{}))) {
                    ctx.quit = true;
                    return;
                }
                if (key.matches(vaxis.Key.up, .{})) {
                    self.tree.move(false);
                } else if (key.matches(vaxis.Key.down, .{})) {
                    self.tree.move(true);
                } else if (key.matches(vaxis.Key.enter, .{})) {
                    self.editingSearch = false;
                    if (self.tree.current()) |node| {
                        if (node.tensor) |tensor| {
                            self.detail = tensor;
                            self.detailScroll = 0;
                        } else try self.tree.toggle();
                    }
                } else if (key.matches(vaxis.Key.escape, .{})) {
                    self.editingSearch = false;
                    self.query.clearRetainingCapacity();
                    try self.updateSearch();
                } else if (self.editingSearch) {
                    if (key.matches(vaxis.Key.backspace, .{})) {
                        if (self.query.items.len > 0) {
                            var end = self.query.items.len - 1;
                            while (end > 0 and self.query.items[end] & 0xc0 == 0x80) end -= 1;
                            self.query.shrinkRetainingCapacity(end);
                        }
                    } else if (key.matches('u', .{ .ctrl = true })) {
                        self.query.clearRetainingCapacity();
                    } else if (key.text) |text| {
                        if (key.mods.ctrl or key.mods.alt or key.mods.super) return;
                        try self.query.appendSlice(self.allocator, text);
                    } else return;
                    try self.updateSearch();
                } else if (key.matches('/', .{})) {
                    self.editingSearch = true;
                } else if (key.matches(vaxis.Key.left, .{})) {
                    try self.tree.left();
                } else if (key.matches(vaxis.Key.right, .{})) {
                    try self.tree.right();
                } else if (key.matches(' ', .{})) {
                    try self.tree.toggle();
                } else if (key.matches(vaxis.Key.page_down, .{})) {
                    self.detailScroll +|= 5;
                } else if (key.matches(vaxis.Key.page_up, .{})) {
                    self.detailScroll -|= 5;
                } else return;
                ctx.consumeAndRedraw();
            },
            else => {},
        }
    }

    fn updateSearch(self: *Model) !void {
        try self.tree.filter(self.query.items);
        self.top = 0;
    }

    fn draw(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const self: *Model = @ptrCast(@alignCast(ptr));
        const size = ctx.max.size();
        var children: std.ArrayList(vxfw.SubSurface) = .empty;
        if (size.width < 40 or size.height < 8) {
            try addText(ctx, &children, 0, 0, size.width, size.height, "Resize terminal to at least 40 x 8", accent, true);
            return .{ .size = size, .widget = self.widget(), .buffer = &.{}, .children = children.items };
        }
        const leftWidth = size.width / 2;
        const rightCol = leftWidth + 2;
        const rightWidth = size.width - rightCol;
        const height = size.height - 5;
        try addText(ctx, &children, 0, 0, size.width, 1, try std.fmt.allocPrint(ctx.arena, "Safetensor explorer  |  {d} tensors  |  {B:.2}  |  {s}", .{ self.tensorCount, self.totalBytes, self.uri }), accent, false);
        try addText(ctx, &children, 1, 0, size.width, 1, try std.fmt.allocPrint(ctx.arena, "Search{s}: {s}{s}", .{ if (self.editingSearch) " (editing)" else "", self.query.items, if (self.editingSearch) "_" else "" }), .{}, false);
        try addText(ctx, &children, 2, 0, leftWidth, 1, try std.fmt.allocPrint(ctx.arena, "Tensors  ({d} visible rows)", .{self.tree.visible.items.len}), accent, false);
        try addText(ctx, &children, 2, rightCol, rightWidth, 1, "Tensor details", accent, false);

        if (self.tree.selected < self.top) self.top = self.tree.selected;
        if (self.tree.selected >= self.top + height) self.top = self.tree.selected - height + 1;
        self.top = @min(self.top, self.tree.visible.items.len -| height);
        if (self.tree.visible.items.len == 0) {
            try addText(ctx, &children, 3, 0, leftWidth, 1, "No matching tensors", dim, false);
        }
        const end = @min(self.tree.visible.items.len, self.top + height);
        for (self.tree.visible.items[self.top..end], self.top..) |node, i| {
            const indent = try ctx.arena.alloc(u8, @min((node.depth - 1) * 2, leftWidth / 2));
            @memset(indent, ' ');
            const marker = if (node.children.count() == 0) " " else if (self.tree.isExpanded(node)) "▾" else "▸";
            const text = if (node.tensor) |tensor|
                try std.fmt.allocPrint(ctx.arena, "{s}{s} {s}  {f}  {B:.2}", .{ indent, marker, node.name, tensor.shape, tensor.byteSize() })
            else
                try std.fmt.allocPrint(ctx.arena, "{s}{s} {s}", .{ indent, marker, node.name });
            const style: vaxis.Style = if (i == self.tree.selected) .{ .reverse = true } else if (node.tensor == null) accent else .{};
            try addText(ctx, &children, @intCast(3 + i - self.top), 0, leftWidth, 1, text, style, false);
        }
        for (3..3 + height) |row| try addText(ctx, &children, @intCast(row), leftWidth, 1, 1, "│", dim, false);

        const details = if (self.detail) |tensor|
            try std.fmt.allocPrint(
                ctx.arena,
                "Name\n{s}\n\nShape / dtype\n{f}\n\nSize\n{d} bytes ({B:.2})\n\nFile offset (absolute bytes)\n{d} (0x{x})\n\nEnd offset (exclusive)\n{d}\n\nFile URI\n{s}",
                .{ tensor.name, tensor.shape, tensor.byteSize(), tensor.byteSize(), tensor.offset, tensor.offset, tensor.offset + tensor.byteSize(), tensor.file_uri },
            )
        else
            "Select a tensor and press Enter.\n\nLeft/Right or Space folds parents.\n\nSearch matches full tensor names.\n\nPgUp/PgDn scrolls these details.";
        const detailText = try ctx.arena.create(vxfw.Text);
        detailText.* = .{ .text = details };
        const detailSurface = try detailText.draw(ctx.withConstraints(.{}, .{ .width = rightWidth, .height = null }));
        self.detailScroll = @min(self.detailScroll, detailSurface.size.height -| height);
        const viewportChildren = try ctx.arena.dupe(vxfw.SubSurface, &.{.{ .origin = .{ .row = -@as(i17, self.detailScroll), .col = 0 }, .surface = detailSurface }});
        try children.append(ctx.arena, .{ .origin = .{ .row = 3, .col = rightCol }, .surface = .{
            .size = .{ .width = rightWidth, .height = height },
            .widget = self.widget(),
            .buffer = &.{},
            .children = viewportChildren,
        } });
        try addText(ctx, &children, size.height - 2, 0, size.width, 1, "↑/↓ select  Enter details  ←/→ fold  Space toggle", dim, false);
        try addText(ctx, &children, size.height - 1, 0, size.width, 1, "/ search  Esc clear  PgUp/PgDn details  q quit", dim, false);
        return .{ .size = size, .widget = self.widget(), .buffer = &.{}, .children = children.items };
    }
};

fn addText(ctx: vxfw.DrawContext, children: *std.ArrayList(vxfw.SubSurface), row: u16, col: u16, width: u16, height: u16, text: []const u8, style: vaxis.Style, wrap: bool) !void {
    const label = try ctx.arena.create(vxfw.Text);
    label.* = .{ .text = text, .style = style, .softwrap = wrap };
    const surface = try label.draw(ctx.withConstraints(.{}, .{ .width = width, .height = height }));
    try children.append(ctx.arena, .{ .origin = .{ .row = @intCast(row), .col = @intCast(col) }, .surface = surface });
}

test "load a repository, relative file, absolute file, file URI, and shard index" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"layer.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    var bytes: [8 + header.len + 8]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8 .. 8 + header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"metadata\":{\"total_size\":8},\"weight_map\":{\"layer.weight\":\"model.safetensors\"}}",
    });
    var pathBuffer: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(io, &pathBuffer);
    const path = pathBuffer[0..len];
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var file: zml.io.VFS.File = .init(allocator, io, .{});
    defer file.deinit();
    var vfs: zml.io.VFS = try .init(allocator, io);
    defer vfs.deinit();
    try vfs.register("file", file.io());
    const paths = [_][]const u8{
        path,
        try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/model.safetensors", .{tmp.sub_path}),
        try std.fmt.allocPrint(a, "{s}/model.safetensors", .{path}),
        try std.fmt.allocPrint(a, "file://{s}/model.safetensors", .{path}),
        try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{path}),
    };
    for (paths) |uri| {
        var registry = try loadRegistry(allocator, vfs.io(), uri);
        defer registry.deinit();
        const tensor = registry.tensors.get("layer.weight").?;
        try std.testing.expectEqual(@as(u64, 8 + header.len), tensor.offset);
        try std.testing.expectEqual(@as(u64, 8), tensor.byteSize());
        try std.testing.expect(std.mem.endsWith(u8, tensor.file_uri, "/model.safetensors"));
    }
}

test "keyboard search, Enter details, pinned selection, and small terminal drawing" {
    const allocator = std.testing.allocator;
    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try registry.registerTensor(.{ .name = "layer.weight", .file_uri = "shard.safetensors", .shape = .init(.{2}, .f32), .offset = 128 });
    var model: Model = .{ .allocator = allocator, .uri = "test", .tree = try .init(allocator, &registry), .tensorCount = 1, .totalBytes = 8 };
    defer model.tree.deinit();
    defer model.query.deinit(allocator);
    var ctx: vxfw.EventContext = .{ .alloc = allocator, .cmds = .empty, .io = std.testing.io };
    defer ctx.cmds.deinit(allocator);
    try Model.handleEvent(&model, &ctx, .{ .key_press = .{ .codepoint = '/' } });
    try Model.handleEvent(&model, &ctx, .{ .key_press = .{ .codepoint = 'w', .text = "weight" } });
    try std.testing.expect(model.editingSearch);
    try Model.handleEvent(&model, &ctx, .{ .key_press = .{ .codepoint = vaxis.Key.down } });
    try Model.handleEvent(&model, &ctx, .{ .key_press = .{ .codepoint = vaxis.Key.enter } });
    try std.testing.expectEqual(@as(u64, 128), model.detail.?.offset);
    try Model.handleEvent(&model, &ctx, .{ .key_press = .{ .codepoint = vaxis.Key.up } });
    try std.testing.expectEqualStrings("layer.weight", model.detail.?.name);
    try Model.handleEvent(&model, &ctx, .{ .key_press = .{ .codepoint = vaxis.Key.escape } });
    try std.testing.expectEqual(@as(usize, 0), model.query.items.len);
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    for ([_]vxfw.Size{ .{ .width = 0, .height = 0 }, .{ .width = 20, .height = 4 }, .{ .width = 40, .height = 8 }, .{ .width = 80, .height = 24 } }) |size| {
        const surface = try Model.draw(&model, .{ .arena = arena.allocator(), .min = .{}, .max = .{ .width = size.width, .height = size.height }, .cell_size = .{ .width = 8, .height = 16 } });
        try std.testing.expectEqual(size, surface.size);
    }
    try Model.handleEvent(&model, &ctx, .{ .key_press = .{ .codepoint = 'q' } });
    try std.testing.expect(ctx.quit);
}
