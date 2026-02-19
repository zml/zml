const std = @import("std");
const expr = @import("expr.zig");
const value_mod = @import("value.zig");
const Value = value_mod.Value;

pub const ParseError = error{
    OutOfMemory,
    UnexpectedEnd,
    UnterminatedTag,
    InvalidStatement,
    MissingEndTag,
} || expr.ExprError;

pub const Node = union(enum) {
    text: []const u8,
    output: *expr.Expr,
    if_block: IfBlock,
    for_block: ForBlock,
    set_var: SetVar,
    set_attr: SetAttr,
    macro_def: MacroDef,

    pub const IfBlock = struct {
        condition: *expr.Expr,
        then_nodes: []const Node,
        else_nodes: []const Node,
    };

    pub const ForBlock = struct {
        var_names: []const []const u8,
        iterable: *expr.Expr,
        body: []const Node,
    };

    pub const SetVar = struct {
        name: []const u8,
        value: *expr.Expr,
    };

    pub const SetAttr = struct {
        object_name: []const u8,
        attr_name: []const u8,
        value: *expr.Expr,
    };

    pub const MacroParam = struct {
        name: []const u8,
        default_expr: ?*expr.Expr,
    };

    pub const MacroDef = struct {
        name: []const u8,
        params: []const MacroParam,
        body: []const Node,
    };
};

pub const Template = struct {
    name: []const u8,
    source: []const u8,
    nodes: []const Node,
    slot_indices: std.StringHashMap(u32),

    pub fn parse(allocator: std.mem.Allocator, name: []const u8, source: []const u8) ParseError!Template {
        var parser = Parser{
            .allocator = allocator,
            .source = source,
            .index = 0,
            .slot_indices = std.StringHashMap(u32).init(allocator),
        };
        const parsed = try parser.parseNodes(&[_][]const u8{});
        if (parsed.matched_end != null) return ParseError.InvalidStatement;
        return .{
            .name = name,
            .source = source,
            .nodes = parsed.nodes,
            .slot_indices = parser.slot_indices,
        };
    }

    pub fn render(self: *const Template, allocator: std.mem.Allocator, ctx_entries: []const Value.Entry) ![]u8 {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const temp_allocator = arena.allocator();

        var scope = expr.Scope.init(temp_allocator);
        defer scope.deinit();

        for (ctx_entries) |entry| try scope.put(entry.key, entry.value);

        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(temp_allocator);
        var writer = ArrayListWriter{ .list = &out, .allocator = temp_allocator };

        var state = RenderState{
            .allocator = temp_allocator,
            .scope = &scope,
            .macros = std.StringHashMap(Node.MacroDef).init(temp_allocator),
            .slot_indices = &self.slot_indices,
            .slot_values = try temp_allocator.alloc(Value, self.slot_indices.count()),
        };
        defer state.macros.deinit();
        for (state.slot_values) |*v| v.* = .undefined;

        for (ctx_entries) |entry| {
            if (self.slot_indices.get(entry.key)) |slot| {
                state.slot_values[slot] = entry.value;
            }
        }

        const runtime = expr.Runtime{ .context = &state, .callFunction = callFunction, .getIdentBySlot = getIdentBySlot };
        try renderNodes(&state, &runtime, self.nodes, &writer);
        // Match MiniJinja default: strip a single trailing newline from rendered output.
        if (out.items.len > 0 and out.items[out.items.len - 1] == '\n') {
            out.items.len -= 1;
        }
        const temp_output = try out.toOwnedSlice(temp_allocator);
        return allocator.dupe(u8, temp_output);
    }
};

const ArrayListWriter = struct {
    list: *std.ArrayList(u8),
    allocator: std.mem.Allocator,

    pub fn writeAll(self: *ArrayListWriter, bytes: []const u8) !void {
        try self.list.appendSlice(self.allocator, bytes);
    }

    pub fn writeByte(self: *ArrayListWriter, b: u8) !void {
        try self.list.append(self.allocator, b);
    }
};

const ParseResult = struct {
    nodes: []const Node,
    matched_end: ?[]const u8,
};

const TagKind = enum { output_tag, stmt_tag, comment_tag };
const NextTag = struct { pos: usize, kind: TagKind };

const Parser = struct {
    allocator: std.mem.Allocator,
    source: []const u8,
    index: usize,
    slot_indices: std.StringHashMap(u32),

    fn slotForName(self: *Parser, name: []const u8) ParseError!u32 {
        const gop = try self.slot_indices.getOrPut(name);
        if (!gop.found_existing) {
            gop.value_ptr.* = @intCast(self.slot_indices.count() - 1);
        }
        return gop.value_ptr.*;
    }

    fn bindExprSlots(self: *Parser, ex: *expr.Expr) ParseError!void {
        switch (ex.*) {
            .ident => |*id| id.slot = try self.slotForName(id.name),
            .list_lit => |items| for (items) |item| try self.bindExprSlots(item),
            .getattr => |g| try self.bindExprSlots(g.target),
            .index => |i| {
                try self.bindExprSlots(i.target);
                try self.bindExprSlots(i.key);
            },
            .slice => |s| {
                try self.bindExprSlots(s.target);
                if (s.start) |p| try self.bindExprSlots(p);
                if (s.end) |p| try self.bindExprSlots(p);
                if (s.step) |p| try self.bindExprSlots(p);
            },
            .call => |c| {
                try self.bindExprSlots(c.callee);
                for (c.args) |arg| switch (arg) {
                    .positional => |p| try self.bindExprSlots(p),
                    .keyword => |kv| try self.bindExprSlots(kv.value),
                };
            },
            .filter => |f| {
                try self.bindExprSlots(f.target);
                for (f.args) |arg| switch (arg) {
                    .positional => |p| try self.bindExprSlots(p),
                    .keyword => |kv| try self.bindExprSlots(kv.value),
                };
            },
            .test_expr => |t| {
                try self.bindExprSlots(t.target);
                for (t.args) |arg| switch (arg) {
                    .positional => |p| try self.bindExprSlots(p),
                    .keyword => |kv| try self.bindExprSlots(kv.value),
                };
            },
            .cond_expr => |c| {
                try self.bindExprSlots(c.then_expr);
                try self.bindExprSlots(c.cond);
                try self.bindExprSlots(c.else_expr);
            },
            .unary_not => |p| try self.bindExprSlots(p),
            .unary_neg => |p| try self.bindExprSlots(p),
            .binary => |b| {
                try self.bindExprSlots(b.left);
                try self.bindExprSlots(b.right);
            },
            .int_lit, .float_lit, .str_lit, .bool_lit, .null_lit => {},
        }
    }

    fn isWs(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\n' or c == '\r';
    }

    fn skipWsAfterTag(self: *Parser) void {
        while (self.index < self.source.len and isWs(self.source[self.index])) : (self.index += 1) {}
    }

    fn findNextTag(self: *const Parser) ?NextTag {
        if (self.source.len < 2 or self.index + 1 >= self.source.len) return null;
        var i = self.index;
        while (i + 1 < self.source.len) : (i += 1) {
            if (self.source[i] != '{') continue;
            switch (self.source[i + 1]) {
                '{' => return .{ .pos = i, .kind = .output_tag },
                '%' => return .{ .pos = i, .kind = .stmt_tag },
                '#' => return .{ .pos = i, .kind = .comment_tag },
                else => continue,
            }
        }
        return null;
    }

    fn parseMacroParams(self: *Parser, source: []const u8) ParseError![]const Node.MacroParam {
        var parts = std.ArrayList(Node.MacroParam).empty;
        errdefer parts.deinit(self.allocator);

        const trimmed = std.mem.trim(u8, source, " \t\n\r");
        if (trimmed.len == 0) return parts.toOwnedSlice(self.allocator);

        var it = std.mem.splitScalar(u8, trimmed, ',');
        while (it.next()) |raw_part| {
            const part = std.mem.trim(u8, raw_part, " \t\n\r");
            if (part.len == 0) continue;

            if (std.mem.indexOfScalar(u8, part, '=')) |eq_pos| {
                const name = std.mem.trim(u8, part[0..eq_pos], " \t\n\r");
                const expr_src = std.mem.trim(u8, part[eq_pos + 1 ..], " \t\n\r");
                if (name.len == 0) return ParseError.InvalidStatement;
                const ex = try expr.parseExpr(self.allocator, expr_src);
                try self.bindExprSlots(ex);
                try parts.append(self.allocator, .{ .name = name, .default_expr = ex });
            } else {
                try parts.append(self.allocator, .{ .name = part, .default_expr = null });
            }
        }

        return parts.toOwnedSlice(self.allocator);
    }

    fn parseIfBlock(self: *Parser, condition: *expr.Expr) ParseError!Node.IfBlock {
        const then_part = try self.parseNodes(&[_][]const u8{ "else", "elif", "endif" });

        var else_nodes: []const Node = &[_]Node{};
        if (then_part.matched_end) |tag| {
            if (std.mem.eql(u8, tag, "else")) {
                const else_part = try self.parseNodes(&[_][]const u8{"endif"});
                if (else_part.matched_end == null or !std.mem.eql(u8, else_part.matched_end.?, "endif")) {
                    return ParseError.MissingEndTag;
                }
                else_nodes = else_part.nodes;
            } else if (std.mem.startsWith(u8, tag, "elif ")) {
                const elif_cond_src = std.mem.trim(u8, tag[5..], " \t\n\r");
                const elif_cond = try expr.parseExpr(self.allocator, elif_cond_src);
                try self.bindExprSlots(elif_cond);
                const elif_block = try self.parseIfBlock(elif_cond);
                const one = try self.allocator.alloc(Node, 1);
                one[0] = .{ .if_block = elif_block };
                else_nodes = one;
            } else if (!std.mem.eql(u8, tag, "endif")) {
                return ParseError.InvalidStatement;
            }
        } else return ParseError.MissingEndTag;

        return .{
            .condition = condition,
            .then_nodes = then_part.nodes,
            .else_nodes = else_nodes,
        };
    }

    fn parseNodes(self: *Parser, end_tags: []const []const u8) ParseError!ParseResult {
        var nodes = std.ArrayList(Node).empty;
        errdefer nodes.deinit(self.allocator);

        while (self.index < self.source.len) {
            const next = self.findNextTag();
            if (next == null) {
                if (self.index < self.source.len) {
                    try nodes.append(self.allocator, .{ .text = self.source[self.index..] });
                }
                self.index = self.source.len;
                break;
            }

            const marker_pos = next.?.pos;
            if (marker_pos > self.index) {
                var text_end = marker_pos;
                switch (next.?.kind) {
                    .output_tag => {
                        if (marker_pos + 2 < self.source.len and self.source[marker_pos + 2] == '-') {
                            while (text_end > self.index and isWs(self.source[text_end - 1])) : (text_end -= 1) {}
                        }
                    },
                    .stmt_tag => {
                        if (marker_pos + 2 < self.source.len and self.source[marker_pos + 2] == '-') {
                            while (text_end > self.index and isWs(self.source[text_end - 1])) : (text_end -= 1) {}
                        }
                    },
                    .comment_tag => {
                        if (marker_pos + 2 < self.source.len and self.source[marker_pos + 2] == '-') {
                            while (text_end > self.index and isWs(self.source[text_end - 1])) : (text_end -= 1) {}
                        }
                    },
                }
                if (text_end > self.index) {
                    try nodes.append(self.allocator, .{ .text = self.source[self.index..text_end] });
                }
            }

            switch (next.?.kind) {
                .comment_tag => {
                    const end_pos = std.mem.indexOfPos(u8, self.source, marker_pos + 2, "#}") orelse return ParseError.UnterminatedTag;
                    const right_trim = end_pos > marker_pos + 2 and self.source[end_pos - 1] == '-';
                    self.index = end_pos + 2;
                    if (right_trim) self.skipWsAfterTag();
                    continue;
                },
                .output_tag => {
                    const end_pos = std.mem.indexOfPos(u8, self.source, marker_pos + 2, "}}") orelse return ParseError.UnterminatedTag;
                    var start_pos = marker_pos + 2;
                    if (start_pos < end_pos and self.source[start_pos] == '-') start_pos += 1;
                    var content_end = end_pos;
                    const right_trim = content_end > start_pos and self.source[content_end - 1] == '-';
                    if (right_trim) content_end -= 1;
                    const raw = self.source[start_pos..content_end];
                    const trimmed = std.mem.trim(u8, raw, " \t\n\r");
                    const ex = try expr.parseExpr(self.allocator, trimmed);
                    try self.bindExprSlots(ex);
                    try nodes.append(self.allocator, .{ .output = ex });
                    self.index = end_pos + 2;
                    if (right_trim) self.skipWsAfterTag();
                    continue;
                },
                .stmt_tag => {
                    const end_pos = std.mem.indexOfPos(u8, self.source, marker_pos + 2, "%}") orelse return ParseError.UnterminatedTag;
                    var start_pos = marker_pos + 2;
                    if (start_pos < end_pos and self.source[start_pos] == '-') start_pos += 1;
                    var content_end = end_pos;
                    const right_trim = content_end > start_pos and self.source[content_end - 1] == '-';
                    if (right_trim) content_end -= 1;
                    const raw_stmt = self.source[start_pos..content_end];
                    const stmt = std.mem.trim(u8, raw_stmt, " \t\n\r");
                    self.index = end_pos + 2;
                    if (right_trim) self.skipWsAfterTag();

                    for (end_tags) |tag| {
                        if (std.mem.eql(u8, stmt, tag) or
                            (std.mem.eql(u8, tag, "elif") and std.mem.startsWith(u8, stmt, "elif ")))
                        {
                            return .{ .nodes = try nodes.toOwnedSlice(self.allocator), .matched_end = stmt };
                        }
                    }

                    if (std.mem.startsWith(u8, stmt, "if ")) {
                        const cond_src = std.mem.trim(u8, stmt[3..], " \t\n\r");
                        const condition = try expr.parseExpr(self.allocator, cond_src);
                        try self.bindExprSlots(condition);
                        const if_block = try self.parseIfBlock(condition);
                        try nodes.append(self.allocator, .{ .if_block = if_block });
                        continue;
                    }

                    if (std.mem.startsWith(u8, stmt, "for ")) {
                        const rhs = std.mem.trim(u8, stmt[4..], " \t\n\r");
                        const in_pos = std.mem.indexOf(u8, rhs, " in ") orelse return ParseError.InvalidStatement;
                        const lhs = std.mem.trim(u8, rhs[0..in_pos], " \t\n\r");
                        const iter_src = std.mem.trim(u8, rhs[in_pos + 4 ..], " \t\n\r");

                        var names = std.ArrayList([]const u8).empty;
                        errdefer names.deinit(self.allocator);
                        var name_it = std.mem.splitScalar(u8, lhs, ',');
                        while (name_it.next()) |raw_name| {
                            const n = std.mem.trim(u8, raw_name, " \t\n\r");
                            if (n.len > 0) try names.append(self.allocator, n);
                        }
                        if (names.items.len == 0) return ParseError.InvalidStatement;

                        const iterable = try expr.parseExpr(self.allocator, iter_src);
                        try self.bindExprSlots(iterable);
                        const body = try self.parseNodes(&[_][]const u8{"endfor"});
                        if (body.matched_end == null or !std.mem.eql(u8, body.matched_end.?, "endfor")) return ParseError.MissingEndTag;
                        try nodes.append(self.allocator, .{ .for_block = .{
                            .var_names = try names.toOwnedSlice(self.allocator),
                            .iterable = iterable,
                            .body = body.nodes,
                        } });
                        continue;
                    }

                    if (std.mem.startsWith(u8, stmt, "set ")) {
                        const rhs = std.mem.trim(u8, stmt[4..], " \t\n\r");
                        const eq_pos = std.mem.indexOfScalar(u8, rhs, '=') orelse return ParseError.InvalidStatement;
                        const lhs = std.mem.trim(u8, rhs[0..eq_pos], " \t\n\r");
                        const value_src = std.mem.trim(u8, rhs[eq_pos + 1 ..], " \t\n\r");
                        const value_expr = try expr.parseExpr(self.allocator, value_src);
                        try self.bindExprSlots(value_expr);

                        if (std.mem.indexOfScalar(u8, lhs, '.')) |dot| {
                            const object_name = std.mem.trim(u8, lhs[0..dot], " \t\n\r");
                            const attr_name = std.mem.trim(u8, lhs[dot + 1 ..], " \t\n\r");
                            if (object_name.len == 0 or attr_name.len == 0) return ParseError.InvalidStatement;
                            try nodes.append(self.allocator, .{ .set_attr = .{ .object_name = object_name, .attr_name = attr_name, .value = value_expr } });
                        } else {
                            if (lhs.len == 0) return ParseError.InvalidStatement;
                            try nodes.append(self.allocator, .{ .set_var = .{ .name = lhs, .value = value_expr } });
                        }
                        continue;
                    }

                    if (std.mem.startsWith(u8, stmt, "macro ")) {
                        const rest = std.mem.trim(u8, stmt[6..], " \t\n\r");
                        const open_idx = std.mem.indexOfScalar(u8, rest, '(') orelse return ParseError.InvalidStatement;
                        const close_idx = std.mem.lastIndexOfScalar(u8, rest, ')') orelse return ParseError.InvalidStatement;
                        if (close_idx <= open_idx) return ParseError.InvalidStatement;
                        const name = std.mem.trim(u8, rest[0..open_idx], " \t\n\r");
                        const params_src = rest[open_idx + 1 .. close_idx];
                        const params = try self.parseMacroParams(params_src);
                        const body = try self.parseNodes(&[_][]const u8{"endmacro"});
                        if (body.matched_end == null) return ParseError.MissingEndTag;
                        try nodes.append(self.allocator, .{ .macro_def = .{ .name = name, .params = params, .body = body.nodes } });
                        continue;
                    }

                    return ParseError.InvalidStatement;
                },
            }
        }

        if (end_tags.len > 0) return ParseError.MissingEndTag;
        return .{ .nodes = try nodes.toOwnedSlice(self.allocator), .matched_end = null };
    }
};

const RenderState = struct {
    allocator: std.mem.Allocator,
    scope: *expr.Scope,
    macros: std.StringHashMap(Node.MacroDef),
    slot_indices: *const std.StringHashMap(u32),
    slot_values: []Value,

    fn getVar(self: *const RenderState, name: []const u8) ?Value {
        if (self.slot_indices.get(name)) |slot| {
            const v = self.slot_values[slot];
            if (v == .undefined) return null;
            return v;
        }
        return self.scope.get(name);
    }

    fn setVar(self: *RenderState, name: []const u8, value: Value) !void {
        if (self.slot_indices.get(name)) |slot| {
            self.slot_values[slot] = value;
            return;
        }
        try self.scope.put(name, value);
    }

    fn removeVar(self: *RenderState, name: []const u8) bool {
        if (self.slot_indices.get(name)) |slot| {
            self.slot_values[slot] = .undefined;
            return true;
        }
        return self.scope.remove(name);
    }
};

fn writeLoopMeta(state: *RenderState, loop_meta_ptr: *value_mod.LoopMeta, idx: usize, len: usize, prev: ?Value, next: ?Value) !void {
    loop_meta_ptr.* = .{
        .index0 = @intCast(idx),
        .index = @intCast(idx + 1),
        .first = idx == 0,
        .last = idx + 1 == len,
        .previtem = prev orelse .undefined,
        .nextitem = next orelse .undefined,
    };
    try state.setVar("loop", .{ .loopmeta = loop_meta_ptr });
}

fn getIdentBySlot(context: ?*anyopaque, slot: u32, _: []const u8, _: *expr.Scope) ?Value {
    const state: *RenderState = @ptrCast(@alignCast(context));
    if (slot >= state.slot_values.len) return null;
    return state.slot_values[slot];
}

fn callFunction(context: ?*anyopaque, allocator: std.mem.Allocator, name: []const u8, args: []const expr.CallArgValue, scope: *expr.Scope) expr.ExprError!Value {
    const state: *RenderState = @ptrCast(@alignCast(context));
    _ = scope;
    const macro_def = state.macros.get(name) orelse return .undefined;

    var old_present_stack: [8]bool = undefined;
    var old_vals_stack: [8]Value = undefined;
    var old_present_heap: ?[]bool = null;
    var old_vals_heap: ?[]Value = null;

    const old_present: []bool = if (macro_def.params.len <= old_present_stack.len)
        old_present_stack[0..macro_def.params.len]
    else blk: {
        const buf = try allocator.alloc(bool, macro_def.params.len);
        old_present_heap = buf;
        break :blk buf;
    };
    defer if (old_present_heap) |buf| allocator.free(buf);

    const old_vals: []Value = if (macro_def.params.len <= old_vals_stack.len)
        old_vals_stack[0..macro_def.params.len]
    else blk: {
        const buf = try allocator.alloc(Value, macro_def.params.len);
        old_vals_heap = buf;
        break :blk buf;
    };
    defer if (old_vals_heap) |buf| allocator.free(buf);

    for (macro_def.params, 0..) |param, idx| {
        if (state.getVar(param.name)) |v| {
            old_present[idx] = true;
            old_vals[idx] = v;
        } else {
            old_present[idx] = false;
            old_vals[idx] = .undefined;
        }
    }

    var has_keywords = false;
    for (args) |arg| {
        if (arg == .keyword) {
            has_keywords = true;
            break;
        }
    }

    var positional_index: usize = 0;
    for (macro_def.params) |param| {
        var bound: ?Value = null;

        if (has_keywords) {
            for (args) |arg| {
                switch (arg) {
                    .keyword => |kv| {
                        if (std.mem.eql(u8, kv.name, param.name)) {
                            bound = kv.value;
                            break;
                        }
                    },
                    else => {},
                }
            }
        }

        if (bound == null) {
            while (positional_index < args.len) : (positional_index += 1) {
                switch (args[positional_index]) {
                    .positional => |pv| {
                        bound = pv;
                        positional_index += 1;
                        break;
                    },
                    else => {},
                }
            }
        }

        if (bound == null) {
            if (param.default_expr) |def_ex| {
                const runtime = expr.Runtime{
                    .context = state,
                    .callFunction = callFunction,
                    .getIdentBySlot = getIdentBySlot,
                };
                bound = try expr.evalExpr(allocator, def_ex, state.scope, &runtime);
            }
        }

        try state.setVar(param.name, bound orelse .undefined);
    }

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    var writer = ArrayListWriter{ .list = &out, .allocator = allocator };
    const runtime = expr.Runtime{
        .context = state,
        .callFunction = callFunction,
        .getIdentBySlot = getIdentBySlot,
    };
    try renderNodes(state, &runtime, macro_def.body, &writer);

    for (macro_def.params, 0..) |param, idx| {
        if (old_present[idx]) {
            try state.setVar(param.name, old_vals[idx]);
        } else {
            _ = state.removeVar(param.name);
        }
    }

    return .{ .str = try out.toOwnedSlice(allocator) };
}

const NodeFrame = struct {
    nodes: []const Node,
    index: usize,
};

const LoopFrame = struct {
    var_names: []const []const u8,
    items: []const Value,
    idx: usize,
    body: []const Node,
    old_loop: ?Value,
    old_vals: []Value,
    old_present: []bool,
    loop_meta_ptr: *value_mod.LoopMeta,
};

const ExecFrame = union(enum) {
    nodes: NodeFrame,
    loop: LoopFrame,
};

fn freeLoopFrame(state: *RenderState, loop: *const LoopFrame) void {
    state.allocator.free(loop.old_vals);
    state.allocator.free(loop.old_present);
    state.allocator.destroy(loop.loop_meta_ptr);
}

fn renderNodes(state: *RenderState, runtime: *const expr.Runtime, nodes: []const Node, writer: anytype) !void {
    var stack = std.ArrayList(ExecFrame).empty;
    defer {
        while (stack.pop()) |frame| {
            switch (frame) {
                .loop => |loop| freeLoopFrame(state, &loop),
                .nodes => {},
            }
        }
        stack.deinit(state.allocator);
    }

    try stack.append(state.allocator, .{ .nodes = .{ .nodes = nodes, .index = 0 } });

    exec: while (stack.items.len > 0) {
        const frame = &stack.items[stack.items.len - 1];
        switch (frame.*) {
            .nodes => |*nf| {
                if (nf.index >= nf.nodes.len) {
                    _ = stack.pop();
                    continue :exec;
                }

                const node = nf.nodes[nf.index];
                nf.index += 1;

                switch (node) {
                    .text => |txt| try writer.writeAll(txt),
                    .output => |ex| {
                        try expr.writeExprOutput(state.allocator, ex, state.scope, runtime, writer);
                    },
                    .macro_def => |m| {
                        try state.macros.put(m.name, m);
                    },
                    .set_var => |sv| {
                        const v = try expr.evalExpr(state.allocator, sv.value, state.scope, runtime);
                        try state.setVar(sv.name, v);
                    },
                    .set_attr => |sa| {
                        const base = state.getVar(sa.object_name) orelse return error.TypeError;
                        const v = try expr.evalExpr(state.allocator, sa.value, state.scope, runtime);
                        try base.setAttr(sa.attr_name, v);
                    },
                    .if_block => |ifb| {
                        const cond = try expr.evalExpr(state.allocator, ifb.condition, state.scope, runtime);
                        const branch = if (cond.isTruthy()) ifb.then_nodes else ifb.else_nodes;
                        if (branch.len > 0) {
                            try stack.append(state.allocator, .{ .nodes = .{ .nodes = branch, .index = 0 } });
                        }
                    },
                    .for_block => |loop| {
                        const iter_val = try expr.evalExpr(state.allocator, loop.iterable, state.scope, runtime);
                        switch (iter_val) {
                            .list => |items| {
                                const old_vals = try state.allocator.alloc(Value, loop.var_names.len);
                                errdefer state.allocator.free(old_vals);
                                const old_present = try state.allocator.alloc(bool, loop.var_names.len);
                                errdefer state.allocator.free(old_present);

                                for (loop.var_names, 0..) |vn, idx| {
                            if (state.getVar(vn)) |v| {
                                old_present[idx] = true;
                                old_vals[idx] = v;
                                    } else {
                                        old_present[idx] = false;
                                        old_vals[idx] = .undefined;
                                    }
                                }

                                const loop_meta_ptr = try state.allocator.create(value_mod.LoopMeta);
                                errdefer state.allocator.destroy(loop_meta_ptr);

                                try stack.append(state.allocator, .{ .loop = .{
                                    .var_names = loop.var_names,
                                    .items = items,
                                    .idx = 0,
                                    .body = loop.body,
                                    .old_loop = state.getVar("loop"),
                                    .old_vals = old_vals,
                                    .old_present = old_present,
                                    .loop_meta_ptr = loop_meta_ptr,
                                } });
                            },
                            else => {},
                        }
                    },
                }
                continue :exec;
            },
            .loop => |*lf| {
                if (lf.idx >= lf.items.len) {
                    if (lf.old_loop) |v| {
                        try state.setVar("loop", v);
                    } else {
                        _ = state.removeVar("loop");
                    }

                    for (lf.var_names, 0..) |vn, idx| {
                        if (lf.old_present[idx]) {
                            try state.setVar(vn, lf.old_vals[idx]);
                        } else {
                            _ = state.removeVar(vn);
                        }
                    }

                    const done = stack.pop().?;
                    switch (done) {
                        .loop => |loop_done| freeLoopFrame(state, &loop_done),
                        .nodes => unreachable,
                    }
                    continue :exec;
                }

                const idx = lf.idx;
                const item = lf.items[idx];
                lf.idx += 1;

                if (lf.var_names.len == 1) {
                    try state.setVar(lf.var_names[0], item);
                } else switch (item) {
                    .list => |pair| {
                        var n: usize = 0;
                        while (n < lf.var_names.len and n < pair.len) : (n += 1) {
                            try state.setVar(lf.var_names[n], pair[n]);
                        }
                    },
                    else => {
                        try state.setVar(lf.var_names[0], item);
                    },
                }

                const prev = if (idx > 0) lf.items[idx - 1] else null;
                const next = if (idx + 1 < lf.items.len) lf.items[idx + 1] else null;
                try writeLoopMeta(state, lf.loop_meta_ptr, idx, lf.items.len, prev, next);
                if (lf.body.len > 0) {
                    try stack.append(state.allocator, .{ .nodes = .{ .nodes = lf.body, .index = 0 } });
                }
                continue :exec;
            },
        }
    }
}
