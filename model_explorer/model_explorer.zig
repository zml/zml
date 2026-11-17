const std = @import("std");
const log = std.log.scoped(.pretty_xla);

const hlo2json = @import("hlo2json.zig");

comptime {
    // export
    _ = hlo2json.convertHloToJson;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var args = std.process.args();
    _ = args.next();

    const input_dot_path = args.next() orelse "/Users/guw/Documents/monorepo/pretty_xla/after_opt.dot";
    const output_json_path = args.next() orelse try std.mem.joinZ(arena.allocator(), "", &.{ input_dot_path, ".json" });

    const output_file = try std.fs.cwd().createFileZ(output_json_path, .{});
    defer output_file.close();
    var writer = std.io.bufferedWriter(output_file.writer());
    defer writer.flush() catch {};
    try convertFile(arena.allocator(), input_dot_path, writer.writer());
}

pub fn convertFile(allocator: std.mem.Allocator, input_dot_path: []const u8, writer: anytype) !void {
    const input_file = try std.fs.cwd().openFile(input_dot_path, .{});
    var reader = std.io.bufferedReader(input_file.reader());
    const file_size = (try input_file.stat()).size;

    const file_content = try std.heap.page_allocator.alloc(u8, file_size);
    _ = try reader.reader().readAll(file_content);
    defer std.heap.page_allocator.free(file_content);

    var converter: Converter = .{ .allocator = allocator };

    var file_lines = file_content[0..];
    while (std.mem.indexOfScalar(u8, file_lines, '\n')) |line_len| {
        const line, file_lines = .{ file_lines[0..line_len], file_lines[line_len + 1 ..] };

        converter.readLine(line) catch |err| {
            log.err("Failed to parse line {}: '{s}' while in state .{s}: {}", .{ converter.line_count, line, @tagName(converter.state), err });
            return err;
        };
    }
    log.info("read {} lines", .{converter.line_count});

    try converter.emit(writer);
}

threadlocal var py_writer = std.ArrayList(u8).init(std.heap.page_allocator);

pub const Slice = extern struct { ptr: u64 = 0, len: u64 = 0 };

pub export fn convertFileToJson(input_dot_path_ptr: [*]const u8, len: u32) Slice {
    const input_dot_path = input_dot_path_ptr[0..len];
    if (py_writer.capacity == 0) py_writer.ensureTotalCapacity(std.heap.page_size_min * 4) catch return .{};
    py_writer.shrinkRetainingCapacity(0);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    convertFile(arena.allocator(), input_dot_path, py_writer.writer()) catch return .{};
    return .{ .ptr = @intFromPtr(py_writer.items.ptr), .len = py_writer.items.len };
}

const State = enum {
    header,
    stylesheet,
    nodes,
    edges,
};

pub const Converter = struct {
    line_count: u32 = 0,
    state: State = .header,
    out_buf: std.BoundedArray(u8, 1024) = .{},
    namespaces: std.BoundedArray([]const u8, 128) = .{},

    allocator: std.mem.Allocator,
    nodes: std.AutoArrayHashMapUnmanaged(u64, Node) = .{},
    incoming_edges: std.AutoArrayHashMapUnmanaged(u64, std.ArrayListUnmanaged(Edge)) = .{},

    const Node = struct { data: []const u8, namespace: []const u8 };
    const Edge = struct { src: u64, data: []const u8 };

    pub fn open(self: Converter) []const u8 {
        _ = self;
        return
        \\{ "label": "module.mlir", "graphs": [{ "id": "main", "nodes": [
        \\
        ;
    }

    pub fn close(self: Converter) []const u8 {
        _ = self;
        return
        \\]}]}
        ;
    }

    pub fn readLine(self: *Converter, line: []const u8) !void {
        self.out_buf.len = 0;
        self.line_count += 1;
        return switch (self.state) {
            .header => {
                if (std.mem.indexOf(u8, line, " [label=")) |_| {
                    self.state = .nodes;
                    self.line_count -= 1;
                    return self.readLine(line);
                }
                if (std.mem.indexOfScalar(u8, line, '=')) |eql_idx| {
                    const key = line[0..eql_idx];
                    if (std.mem.eql(u8, key, "stylesheet")) {
                        self.state = .stylesheet;
                    } else if (std.mem.eql(u8, key, "label ")) {
                        if (self.namespaces.len > 0) {
                            _ = parseLabel(line, &self.out_buf);
                            const new_label = try self.allocator.dupe(u8, self.out_buf.constSlice());
                            // log.debug("renaming {s} with {s} ({s}", .{self.namespace(), new_label, line});
                            self.namespaces.set(self.namespaces.len - 1, new_label);
                        }
                    } else {
                        // log.debug("skipping key: {s}", .{key});
                    }
                    return;
                }
            },
            .stylesheet => {
                if (std.mem.eql(u8, line, ">")) {
                    self.state = .nodes;
                }
            },
            .nodes => {
                if (std.mem.startsWith(u8, line, "subgraph cluster_")) {
                    try self.pushCluster(line);
                    self.state = .header;
                    return;
                }
                if (std.mem.startsWith(u8, line, "}  // cluster_")) {
                    _ = self.popCluster();
                    return;
                }
                if (std.mem.indexOf(u8, line, " -> ")) |_| {
                    self.state = .edges;
                    self.line_count -= 1;
                    return self.readLine(line);
                }
                if (std.mem.indexOfScalar(u8, line, ' ')) |space_idx| {
                    const first_token = line[0..space_idx];
                    const id = parseNodeId(first_token) catch {
                        log.warn("Ignoring node id: {s} at line {}: '{s}'", .{ first_token, self.line_count, line });
                        return;
                    };
                    try self.nodes.put(self.allocator, id, .{ .data = line[space_idx + 1 ..], .namespace = self.namespace() });
                }
            },
            .edges => {
                const split = std.mem.indexOf(u8, line, " -> ");
                if (split == null) {
                    self.state = .nodes;
                    self.line_count -= 1;
                    return self.readLine(line);
                }
                const src_id = parseNodeId(line[0..split.?]) catch {
                    log.warn("Ignoring edge source id: {s}", .{line[0..split.?]});
                    return;
                };
                const tgt_start = split.? + " -> ".len;
                if (std.mem.indexOfScalar(u8, line[tgt_start..], ' ')) |tgt_len| {
                    const target = line[tgt_start..][0..tgt_len];
                    const tgt_id = parseNodeId(target) catch {
                        log.warn("Ignoring edge target id: {s}", .{line[0..split.?]});
                        return;
                    };
                    const edge: Edge = .{ .src = src_id, .data = line[tgt_start + tgt_len + 1 ..] };
                    if (self.incoming_edges.getPtr(tgt_id)) |edges| {
                        try edges.append(self.allocator, edge);
                    } else {
                        var new_edges: std.ArrayListUnmanaged(Edge) = .{};
                        try new_edges.ensureTotalCapacity(self.allocator, 8);
                        new_edges.appendAssumeCapacity(edge);
                        try self.incoming_edges.put(self.allocator, tgt_id, new_edges);
                    }
                    return;
                }
                log.err("Couldn't parse edge: {s}", .{line});
            },
        };
    }

    pub fn namespace(self: *const Converter) []const u8 {
        if (self.namespaces.len == 0) return "";
        return self.namespaces.get(self.namespaces.len - 1);
    }

    fn pushCluster(self: *Converter, line: []const u8) !void {
        var name = line["subgraph cluster_".len..];
        const end = std.mem.indexOfScalar(u8, name, ' ').?;
        name = name[0..end];
        // const new_name = try std.mem.join(self.allocator, "/", &.{ self.namespace(), name });
        // Default to the cluster id, but try to replace by label after.
        try self.namespaces.append("unk");
    }

    fn popCluster(self: *Converter) []const u8 {
        return self.namespaces.pop() orelse "";
    }

    pub fn emit(self: *const Converter, writer: anytype) !void {
        try writer.writeAll(self.open());
        log.info("found {} nodes and {} edges ", .{ self.nodes.count(), self.incoming_edges.count() });

        var node_need_comma: bool = false;
        for (self.nodes.keys(), self.nodes.values()) |key, node| {
            if (node_need_comma) try writer.writeAll(",\n");
            node_need_comma = true;

            // node name + attrs
            try std.fmt.format(writer,
                \\{{"id": "{d}","namespace":"{s}"
            , .{ key, node.namespace });
            try writeNodeAttrs(writer, node.data);

            // node edges
            try writer.writeAll(",\"incomingEdges\":[");
            if (self.incoming_edges.get(key)) |edges| {
                var edge_need_comma: bool = false;
                for (edges.items) |edge| {
                    if (edge_need_comma) try writer.writeAll(",");
                    edge_need_comma = true;
                    const json_edge: JsonEdge = .{
                        .sourceNodeId = formatId(edge.src),
                        .targetNodeInputId = parseAttr("headlabel", edge.data) orelse "0",
                    };
                    try std.fmt.format(writer, "{}", .{std.json.fmt(json_edge, .{})});
                }
            }

            try writer.writeAll("]}");
        }
        try writer.writeAll(self.close());
    }
};

fn parseNodeId(token: []const u8) !u64 {
    return std.fmt.parseInt(u64, token, 10) catch {
        if (std.mem.indexOfScalar(u8, token, '_')) |underscore_idx| {
            return std.fmt.parseInt(u64, token[underscore_idx + 1 ..], 10) catch error.FailedToParseId;
        }
        return error.FailedToParseId;
    };
}

pub fn writeNodeAttrs(writer: anytype, data: []const u8) !void {
    const json_attr_fmt = "{{\"key\":\"{s}\",\"value\":\"{s}\"}}";
    var label_buf: std.BoundedArray(u8, 1024) = .{};
    var shape_buf: std.BoundedArray(u8, 64) = .{};
    var label, const node_attrs = parseLabel(data, &label_buf);

    if (std.mem.lastIndexOfScalar(u8, label, ' ')) |space_idx| {
        // log.warn("param: {s} -> {s} -> {s}", .{ data, label, label[space_idx + 1..] });
        if (convertShape(label[space_idx + 1 ..], &shape_buf)) |param_shape| {
            try std.fmt.format(writer,
                \\,"outputsMetadata":[{{"attrs":[{{"key":"tensor_shape","value":"{s}"}}],"id": "0"}}]
            , .{param_shape});
            label = label[0..space_idx];
        }
    }

    try std.fmt.format(writer,
        \\,"label":"{s}","attrs":[
    , .{label});
    // if (verbose) log.warn("tooltip: {s}", .{node_attrs});
    if (parseAttr("tooltip", node_attrs)) |_tooltip| {
        // TODO: handle newline in tooltip
        // if (verbose) log.warn("--> tooltip={s}", .{tooltip});
        var tooltip = _tooltip;
        if (std.mem.indexOf(u8, tooltip, "source: ")) |source_idx| {
            try std.fmt.format(writer, json_attr_fmt, .{ "source", tooltip[source_idx + "source: ".len ..] });
            tooltip = tooltip[0..source_idx];
        }
        if (tooltip.len > 0) try std.fmt.format(writer, json_attr_fmt, .{ "tooltip", tooltip });
        try writer.writeAll("]");
    }
}

pub inline fn formatId(id: u64) []const u8 {
    var x: [16]u8 = undefined;
    return std.fmt.bufPrint(&x, "{d}", .{id}) catch unreachable;
}

fn convertShape(hlo_shape: []const u8, mlir_shape: *std.BoundedArray(u8, 64)) ?[]const u8 {
    mlir_shape.len = 0;
    mlir_shape.appendSliceAssumeCapacity("tensor<");

    var hlo = hlo_shape;
    const dtype = hlo[0 .. std.mem.indexOfScalar(u8, hlo, '[') orelse return null];
    hlo = hlo[dtype.len + 1 ..];
    hlo = hlo[0 .. std.mem.indexOfScalar(u8, hlo, ']') orelse return null];

    mlir_shape.appendSliceAssumeCapacity(hlo);
    mlir_shape.appendAssumeCapacity('x');
    mlir_shape.appendSliceAssumeCapacity(dtype);
    for (mlir_shape.slice()) |*c| {
        if (c.* == ',') c.* = 'x';
    }
    mlir_shape.appendAssumeCapacity('>');
    return mlir_shape.constSlice();
}

test convertShape {
    var buf: std.BoundedArray(u8, 64) = .{};
    try std.testing.expectEqualStrings("tensor<1x5x4xf32>", convertShape("f32[1,5,4]{2,1,0}", &buf));
}

const JsonAttr = struct { key: []const u8, value: []const u8 };

const JsonNode = struct {
    id: []const u8,
    label: []const u8,
    attrs: []const JsonAttr = &.{},
    incomingEdges: []const JsonEdge,
    // "inputsMetadata": [],
    namespace: []const u8,
    // outputsMetadata: [{attrs: []const Attr}],
    // subgraphIds
};

const JsonEdge = struct {
    // "edgeMetadata": [],
    sourceNodeId: []const u8,
    sourceNodeOutputId: []const u8 = "0",
    targetNodeInputId: []const u8 = "0",
};

pub fn parseLabel(data: []const u8, out: *std.BoundedArray(u8, 1024)) [2][]const u8 {
    const verbose = false;
    defer {
        // const verbose = std.mem.indexOf(u8, data, "Subcomputation: <b>max</b>") != null;
        if (verbose) log.debug("--> {s}", .{out.constSlice()});
        std.debug.assert(out.len <= 2 or out.buffer[out.len - 2] != '<');
    }
    out.len = 0;
    var tail = data;
    tail = leftStrip(tail, "[label=<");
    // tail = leftStrip(tail, "label = <Fused expression for <b>");
    // tail = leftStrip(tail, "label = <Subcomputation for <b>");
    tail = leftStrip(tail, "label = <");
    var i: u32 = 0;
    while (tail.len > 0) {
        if (verbose) log.debug("{s} ... {s}", .{ out.constSlice(), tail });
        tail = leftStrip(tail, "</b><br/>loop fusion<br/>kind=kLoop>");
        tail = leftStrip(tail, "</b>");
        tail = leftStrip(tail, "<b>");
        tail = leftStrip(tail, "sharding={replicated}");
        tail = leftStrip(tail, ";");
        if (std.mem.startsWith(u8, tail, "<br/>")) {
            tail = tail["<br/>".len..];
            out.append(' ') catch {};
            continue;
        }
        if (tail.len == 0) break;
        const c, tail = .{ tail[0], tail[1..] };
        if (c == '>') return .{ out.constSlice(), tail };
        out.append(c) catch {}; // continue parsing even if we can't write anymore
        i += 1;
    }
    return .{ out.constSlice(), tail };
}

pub fn parseAttr(attr: []const u8, data: []const u8) ?[]const u8 {
    var value = data;
    const AttrState = enum { outside, keys, skip_value, res };
    var state: AttrState = .outside;
    const verbose = false;
    if (verbose) log.debug("parseAttr({s}, {s}", .{ attr, data });

    while (value.len > 0) {
        if (verbose) log.debug("state={}, value={s}", .{ state, value });
        const c = value[0];
        switch (state) {
            .outside => if (c == '[') {
                state = .keys;
                value = value[1..];
            } else {
                // We didn't find an opening bracket. so we must have beend dropped inside it already.
                state = .keys;
            },
            .keys => {
                // end of attribute, but attribute wasn't found !
                if (c == ']') return null;

                // between attr
                if (c == ' ' or c == ',') {
                    value = value[1..];
                    continue;
                }

                // found new attribute
                if (std.mem.indexOfScalar(u8, value, '=')) |key_end| {
                    const key = value[0..key_end];
                    value = value[key_end + 1 ..];
                    state = if (std.mem.eql(u8, attr, key)) .res else .skip_value;
                    continue;
                } else {
                    return null;
                }
            },
            .skip_value => {
                if (c == '"') {
                    // TODO: this can break if we find escaped " inside the attribute.
                    const end = std.mem.indexOfScalar(u8, value[1..], '"') orelse value.len - 2;
                    value = value[end + 2 ..];
                } else {
                    const end = std.mem.indexOfAny(u8, value, " ,]") orelse value.len - 1;
                    value = value[end + 1 ..];
                }
                state = .keys;
            },
            .res => return if (c == '"') {
                // TODO: this can break if we find escaped " inside the attribute.
                const end = std.mem.indexOfScalar(u8, value[1..], '"') orelse value.len - 1;
                return value[1..][0..end];
            } else {
                const end = std.mem.indexOfAny(u8, value, " ,]") orelse value.len - 1;
                return value[0..end];
            },
        }
    }
    return null;
}

fn leftStrip(haystack: []const u8, prefix: []const u8) []const u8 {
    if (std.mem.startsWith(u8, haystack, prefix)) {
        return haystack[prefix.len..];
    }
    return haystack;
}
