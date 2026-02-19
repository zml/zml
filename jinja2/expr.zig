const std = @import("std");
const value_mod = @import("value.zig");
const Value = value_mod.Value;

pub const ExprError = error{
    UnexpectedToken,
    UnterminatedString,
    InvalidNumber,
    OutOfMemory,
    TypeError,
    RaisedException,
    NoSpaceLeft,
};

const TokenTag = enum {
    ident,
    int,
    float,
    string,
    bool_true,
    bool_false,
    none_kw,
    and_kw,
    or_kw,
    not_kw,
    is_kw,
    if_kw,
    else_kw,
    in_kw,
    plus,
    minus,
    star,
    slash,
    percent,
    lparen,
    rparen,
    lbracket,
    rbracket,
    comma,
    colon,
    dot,
    pipe,
    tilde,
    assign,
    eqeq,
    noteq,
    lt,
    lte,
    gt,
    gte,
    eof,
};

const Token = struct {
    tag: TokenTag,
    lexeme: []const u8,
};

const LexState = enum { start, ident, number, string, string_escape };

fn isAlpha(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_';
}

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn isAlphaNum(c: u8) bool {
    return isAlpha(c) or isDigit(c);
}

pub fn lex(allocator: std.mem.Allocator, input: []const u8) ExprError![]Token {
    var tokens = std.ArrayList(Token).empty;
    errdefer tokens.deinit(allocator);

    var i: usize = 0;
    var state: LexState = .start;
    var start: usize = 0;
    var quote_char: u8 = 0;

    scan: while (true) {
        switch (state) {
            .start => {
                if (i >= input.len) {
                    try tokens.append(allocator, .{ .tag = .eof, .lexeme = "" });
                    break :scan;
                }
                const c = input[i];
                if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
                    i += 1;
                    continue :scan;
                }
                if (isAlpha(c)) {
                    start = i;
                    i += 1;
                    state = .ident;
                    continue :scan;
                }
                if (isDigit(c)) {
                    start = i;
                    i += 1;
                    state = .number;
                    continue :scan;
                }

                switch (c) {
                    '\'', '"' => {
                        quote_char = c;
                        start = i + 1;
                        i += 1;
                        state = .string;
                    },
                    '+' => {
                        try tokens.append(allocator, .{ .tag = .plus, .lexeme = "+" });
                        i += 1;
                    },
                    '-' => {
                        try tokens.append(allocator, .{ .tag = .minus, .lexeme = "-" });
                        i += 1;
                    },
                    '*' => {
                        try tokens.append(allocator, .{ .tag = .star, .lexeme = "*" });
                        i += 1;
                    },
                    '/' => {
                        try tokens.append(allocator, .{ .tag = .slash, .lexeme = "/" });
                        i += 1;
                    },
                    '%' => {
                        try tokens.append(allocator, .{ .tag = .percent, .lexeme = "%" });
                        i += 1;
                    },
                    '(' => {
                        try tokens.append(allocator, .{ .tag = .lparen, .lexeme = "(" });
                        i += 1;
                    },
                    ')' => {
                        try tokens.append(allocator, .{ .tag = .rparen, .lexeme = ")" });
                        i += 1;
                    },
                    '[' => {
                        try tokens.append(allocator, .{ .tag = .lbracket, .lexeme = "[" });
                        i += 1;
                    },
                    ']' => {
                        try tokens.append(allocator, .{ .tag = .rbracket, .lexeme = "]" });
                        i += 1;
                    },
                    ',' => {
                        try tokens.append(allocator, .{ .tag = .comma, .lexeme = "," });
                        i += 1;
                    },
                    ':' => {
                        try tokens.append(allocator, .{ .tag = .colon, .lexeme = ":" });
                        i += 1;
                    },
                    '.' => {
                        try tokens.append(allocator, .{ .tag = .dot, .lexeme = "." });
                        i += 1;
                    },
                    '|' => {
                        try tokens.append(allocator, .{ .tag = .pipe, .lexeme = "|" });
                        i += 1;
                    },
                    '~' => {
                        try tokens.append(allocator, .{ .tag = .tilde, .lexeme = "~" });
                        i += 1;
                    },
                    '=' => {
                        if (i + 1 < input.len and input[i + 1] == '=') {
                            try tokens.append(allocator, .{ .tag = .eqeq, .lexeme = "==" });
                            i += 2;
                        } else {
                            try tokens.append(allocator, .{ .tag = .assign, .lexeme = "=" });
                            i += 1;
                        }
                    },
                    '!' => {
                        if (i + 1 < input.len and input[i + 1] == '=') {
                            try tokens.append(allocator, .{ .tag = .noteq, .lexeme = "!=" });
                            i += 2;
                        } else return ExprError.UnexpectedToken;
                    },
                    '<' => {
                        if (i + 1 < input.len and input[i + 1] == '=') {
                            try tokens.append(allocator, .{ .tag = .lte, .lexeme = "<=" });
                            i += 2;
                        } else {
                            try tokens.append(allocator, .{ .tag = .lt, .lexeme = "<" });
                            i += 1;
                        }
                    },
                    '>' => {
                        if (i + 1 < input.len and input[i + 1] == '=') {
                            try tokens.append(allocator, .{ .tag = .gte, .lexeme = ">=" });
                            i += 2;
                        } else {
                            try tokens.append(allocator, .{ .tag = .gt, .lexeme = ">" });
                            i += 1;
                        }
                    },
                    else => return ExprError.UnexpectedToken,
                }
            },
            .ident => {
                while (i < input.len and isAlphaNum(input[i])) : (i += 1) {}
                const lexeme = input[start..i];
                const tag: TokenTag = blk: {
                    if (std.mem.eql(u8, lexeme, "true")) break :blk .bool_true;
                    if (std.mem.eql(u8, lexeme, "false")) break :blk .bool_false;
                    if (std.mem.eql(u8, lexeme, "none") or std.mem.eql(u8, lexeme, "null")) break :blk .none_kw;
                    if (std.mem.eql(u8, lexeme, "and")) break :blk .and_kw;
                    if (std.mem.eql(u8, lexeme, "or")) break :blk .or_kw;
                    if (std.mem.eql(u8, lexeme, "not")) break :blk .not_kw;
                    if (std.mem.eql(u8, lexeme, "is")) break :blk .is_kw;
                    if (std.mem.eql(u8, lexeme, "if")) break :blk .if_kw;
                    if (std.mem.eql(u8, lexeme, "else")) break :blk .else_kw;
                    if (std.mem.eql(u8, lexeme, "in")) break :blk .in_kw;
                    break :blk .ident;
                };
                try tokens.append(allocator, .{ .tag = tag, .lexeme = lexeme });
                state = .start;
            },
            .number => {
                var is_float = false;
                while (i < input.len and isDigit(input[i])) : (i += 1) {}
                if (i < input.len and input[i] == '.') {
                    if (i + 1 < input.len and isDigit(input[i + 1])) {
                        is_float = true;
                        i += 1;
                        while (i < input.len and isDigit(input[i])) : (i += 1) {}
                    }
                }
                try tokens.append(allocator, .{ .tag = if (is_float) .float else .int, .lexeme = input[start..i] });
                state = .start;
            },
            .string => {
                if (i >= input.len) return ExprError.UnterminatedString;
                const c = input[i];
                if (c == '\\') {
                    i += 1;
                    state = .string_escape;
                    continue :scan;
                }
                if (c == quote_char) {
                    try tokens.append(allocator, .{ .tag = .string, .lexeme = input[start..i] });
                    i += 1;
                    state = .start;
                    continue :scan;
                }
                i += 1;
            },
            .string_escape => {
                if (i >= input.len) return ExprError.UnterminatedString;
                i += 1;
                state = .string;
            },
        }
    }

    return tokens.toOwnedSlice(allocator);
}

pub const BinaryOp = enum {
    add,
    sub,
    mul,
    div,
    mod,
    eq,
    neq,
    lt,
    lte,
    gt,
    gte,
    in,
    not_in,
    concat,
    @"and",
    @"or",
};

pub const CallArg = union(enum) {
    positional: *Expr,
    keyword: struct { name: []const u8, value: *Expr },
};

const StringMethodId = enum {
    unknown,
    startswith,
    endswith,
    split,
    strip,
    rstrip,
    lstrip,
};

const FilterId = enum {
    unknown,
    safe,
    length,
    string,
    tojson,
    trim,
    items,
    sort,
};

const TestId = enum {
    unknown,
    defined,
    none,
    string,
    iterable,
    mapping,
    sequence,
    undefined,
};

fn detectStringMethodId(name: []const u8) StringMethodId {
    if (name.len == 0) return .unknown;
    return switch (name[0]) {
        's' => if (std.mem.eql(u8, name, "startswith")) .startswith else if (std.mem.eql(u8, name, "split")) .split else if (std.mem.eql(u8, name, "strip")) .strip else .unknown,
        'e' => if (std.mem.eql(u8, name, "endswith")) .endswith else .unknown,
        'r' => if (std.mem.eql(u8, name, "rstrip")) .rstrip else .unknown,
        'l' => if (std.mem.eql(u8, name, "lstrip")) .lstrip else .unknown,
        else => .unknown,
    };
}

fn detectFilterId(name: []const u8) FilterId {
    if (name.len == 0) return .unknown;
    return switch (name[0]) {
        's' => if (std.mem.eql(u8, name, "safe")) .safe else if (std.mem.eql(u8, name, "string")) .string else if (std.mem.eql(u8, name, "sort")) .sort else .unknown,
        'l' => if (std.mem.eql(u8, name, "length")) .length else .unknown,
        't' => if (std.mem.eql(u8, name, "tojson")) .tojson else if (std.mem.eql(u8, name, "trim")) .trim else .unknown,
        'i' => if (std.mem.eql(u8, name, "items")) .items else .unknown,
        else => .unknown,
    };
}

fn detectTestId(name: []const u8) TestId {
    if (name.len == 0) return .unknown;
    return switch (name[0]) {
        'd' => if (std.mem.eql(u8, name, "defined")) .defined else .unknown,
        'n' => if (std.mem.eql(u8, name, "none")) .none else .unknown,
        's' => if (std.mem.eql(u8, name, "string")) .string else if (std.mem.eql(u8, name, "sequence")) .sequence else .unknown,
        'i' => if (std.mem.eql(u8, name, "iterable")) .iterable else .unknown,
        'm' => if (std.mem.eql(u8, name, "mapping")) .mapping else .unknown,
        'u' => if (std.mem.eql(u8, name, "undefined")) .undefined else .unknown,
        else => .unknown,
    };
}

pub const Expr = union(enum) {
    ident: struct { name: []const u8, slot: u32 },
    int_lit: i64,
    float_lit: f64,
    str_lit: []const u8,
    bool_lit: bool,
    null_lit,
    list_lit: []const *Expr,
    getattr: struct { target: *Expr, attr: []const u8, method_id: StringMethodId },
    index: struct { target: *Expr, key: *Expr },
    slice: struct { target: *Expr, start: ?*Expr, end: ?*Expr, step: ?*Expr },
    call: struct { callee: *Expr, args: []const CallArg },
    filter: struct { target: *Expr, name: []const u8, filter_id: FilterId, args: []const CallArg },
    test_expr: struct { target: *Expr, name: []const u8, test_id: TestId, negated: bool, args: []const CallArg },
    cond_expr: struct { then_expr: *Expr, cond: *Expr, else_expr: *Expr },
    unary_not: *Expr,
    unary_neg: *Expr,
    binary: struct { op: BinaryOp, left: *Expr, right: *Expr },
};

pub const unknown_slot: u32 = std.math.maxInt(u32);

const Parser = struct {
    allocator: std.mem.Allocator,
    tokens: []const Token,
    pos: usize,

    fn peek(self: *const Parser) Token {
        return self.tokens[self.pos];
    }

    fn peekN(self: *const Parser, n: usize) Token {
        const idx = @min(self.pos + n, self.tokens.len - 1);
        return self.tokens[idx];
    }

    fn advance(self: *Parser) Token {
        const t = self.tokens[self.pos];
        self.pos += 1;
        return t;
    }

    fn eat(self: *Parser, tag: TokenTag) bool {
        if (self.peek().tag == tag) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    fn expect(self: *Parser, tag: TokenTag) ExprError!Token {
        if (self.peek().tag != tag) return ExprError.UnexpectedToken;
        return self.advance();
    }

    fn allocExpr(self: *Parser, value: Expr) ExprError!*Expr {
        const p = try self.allocator.create(Expr);
        p.* = value;
        return p;
    }

    fn unescapeString(self: *Parser, raw: []const u8) ExprError![]const u8 {
        if (std.mem.indexOfScalar(u8, raw, '\\') == null) return raw;

        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(self.allocator);

        var i: usize = 0;
        while (i < raw.len) : (i += 1) {
            const c = raw[i];
            if (c != '\\' or i + 1 >= raw.len) {
                try out.append(self.allocator, c);
                continue;
            }

            i += 1;
            const esc = raw[i];
            const decoded: u8 = switch (esc) {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '\\' => '\\',
                '"' => '"',
                '\'' => '\'',
                else => esc,
            };
            try out.append(self.allocator, decoded);
        }

        return out.toOwnedSlice(self.allocator);
    }

    fn parse(self: *Parser) ExprError!*Expr {
        const rv = try self.parseConditional();
        _ = try self.expect(.eof);
        return rv;
    }

    fn parseConditional(self: *Parser) ExprError!*Expr {
        var then_expr = try self.parseOr();
        if (self.eat(.if_kw)) {
            const cond = try self.parseOr();
            _ = try self.expect(.else_kw);
            const else_expr = try self.parseConditional();
            then_expr = try self.allocExpr(.{ .cond_expr = .{ .then_expr = then_expr, .cond = cond, .else_expr = else_expr } });
        }
        return then_expr;
    }

    fn parseOr(self: *Parser) ExprError!*Expr {
        var left = try self.parseAnd();
        while (self.eat(.or_kw)) {
            const right = try self.parseAnd();
            left = try self.allocExpr(.{ .binary = .{ .op = .@"or", .left = left, .right = right } });
        }
        return left;
    }

    fn parseAnd(self: *Parser) ExprError!*Expr {
        var left = try self.parseCompare();
        while (self.eat(.and_kw)) {
            const right = try self.parseCompare();
            left = try self.allocExpr(.{ .binary = .{ .op = .@"and", .left = left, .right = right } });
        }
        return left;
    }

    fn parseCompare(self: *Parser) ExprError!*Expr {
        var left = try self.parseAdd();
        while (true) {
            if (self.eat(.is_kw)) {
                const negated = self.eat(.not_kw);
                const name_tok = self.peek();
                if (!isNameToken(name_tok.tag)) return ExprError.UnexpectedToken;
                _ = self.advance();
                var args: []const CallArg = &[_]CallArg{};
                if (self.eat(.lparen)) {
                    args = try self.parseArgs();
                }
                left = try self.allocExpr(.{ .test_expr = .{
                    .target = left,
                    .name = name_tok.lexeme,
                    .test_id = detectTestId(name_tok.lexeme),
                    .negated = negated,
                    .args = args,
                } });
                continue;
            }

            var maybe_op: ?BinaryOp = null;
            if (self.eat(.eqeq)) {
                maybe_op = .eq;
            } else if (self.eat(.noteq)) {
                maybe_op = .neq;
            } else if (self.eat(.lt)) {
                maybe_op = .lt;
            } else if (self.eat(.lte)) {
                maybe_op = .lte;
            } else if (self.eat(.gt)) {
                maybe_op = .gt;
            } else if (self.eat(.gte)) {
                maybe_op = .gte;
            } else if (self.eat(.in_kw)) {
                maybe_op = .in;
            } else if (self.eat(.not_kw)) {
                if (self.eat(.in_kw)) {
                    maybe_op = .not_in;
                } else return ExprError.UnexpectedToken;
            }
            const op = maybe_op orelse break;

            const right = try self.parseAdd();
            left = try self.allocExpr(.{ .binary = .{ .op = op, .left = left, .right = right } });
        }
        return left;
    }

    fn isNameToken(tag: TokenTag) bool {
        return switch (tag) {
            .ident, .none_kw, .bool_true, .bool_false, .and_kw, .or_kw, .not_kw, .is_kw, .if_kw, .else_kw, .in_kw => true,
            else => false,
        };
    }

    fn parseAdd(self: *Parser) ExprError!*Expr {
        var left = try self.parseMul();
        while (true) {
            const op: BinaryOp = switch (self.peek().tag) {
                .plus => .add,
                .minus => .sub,
                .tilde => .concat,
                else => break,
            };
            _ = self.advance();
            const right = try self.parseMul();
            left = try self.allocExpr(.{ .binary = .{ .op = op, .left = left, .right = right } });
        }
        return left;
    }

    fn parseMul(self: *Parser) ExprError!*Expr {
        var left = try self.parseUnary();
        while (true) {
            const op: BinaryOp = switch (self.peek().tag) {
                .star => .mul,
                .slash => .div,
                .percent => .mod,
                else => break,
            };
            _ = self.advance();
            const right = try self.parseUnary();
            left = try self.allocExpr(.{ .binary = .{ .op = op, .left = left, .right = right } });
        }
        return left;
    }

    fn parseUnary(self: *Parser) ExprError!*Expr {
        if (self.eat(.not_kw)) {
            const inner = try self.parseUnary();
            return self.allocExpr(.{ .unary_not = inner });
        }
        if (self.eat(.minus)) {
            const inner = try self.parseUnary();
            return self.allocExpr(.{ .unary_neg = inner });
        }
        return self.parsePostfix();
    }

    fn parsePostfix(self: *Parser) ExprError!*Expr {
        var base = try self.parsePrimary();
        while (true) {
            if (self.eat(.dot)) {
                const ident = try self.expect(.ident);
                base = try self.allocExpr(.{ .getattr = .{
                    .target = base,
                    .attr = ident.lexeme,
                    .method_id = detectStringMethodId(ident.lexeme),
                } });
                continue;
            }
            if (self.eat(.lbracket)) {
                if (self.eat(.colon)) {
                    var end_expr: ?*Expr = null;
                    var step_expr: ?*Expr = null;
                    if (self.peek().tag != .rbracket and self.peek().tag != .colon) end_expr = try self.parseConditional();
                    if (self.eat(.colon)) {
                        if (self.peek().tag != .rbracket) step_expr = try self.parseConditional();
                    }
                    _ = try self.expect(.rbracket);
                    base = try self.allocExpr(.{ .slice = .{ .target = base, .start = null, .end = end_expr, .step = step_expr } });
                    continue;
                }

                const first = try self.parseConditional();
                if (self.eat(.colon)) {
                    var end_expr: ?*Expr = null;
                    var step_expr: ?*Expr = null;
                    if (self.peek().tag != .rbracket and self.peek().tag != .colon) end_expr = try self.parseConditional();
                    if (self.eat(.colon)) {
                        if (self.peek().tag != .rbracket) step_expr = try self.parseConditional();
                    }
                    _ = try self.expect(.rbracket);
                    base = try self.allocExpr(.{ .slice = .{ .target = base, .start = first, .end = end_expr, .step = step_expr } });
                } else {
                    _ = try self.expect(.rbracket);
                    base = try self.allocExpr(.{ .index = .{ .target = base, .key = first } });
                }
                continue;
            }
            if (self.eat(.lparen)) {
                const args = try self.parseArgs();
                base = try self.allocExpr(.{ .call = .{ .callee = base, .args = args } });
                continue;
            }
            if (self.eat(.pipe)) {
                const name = try self.expect(.ident);
                var args: []const CallArg = &[_]CallArg{};
                if (self.eat(.lparen)) args = try self.parseArgs();
                base = try self.allocExpr(.{ .filter = .{
                    .target = base,
                    .name = name.lexeme,
                    .filter_id = detectFilterId(name.lexeme),
                    .args = args,
                } });
                continue;
            }
            break;
        }
        return base;
    }

    fn parseArgs(self: *Parser) ExprError![]const CallArg {
        var args = std.ArrayList(CallArg).empty;
        errdefer args.deinit(self.allocator);

        if (self.eat(.rparen)) return args.toOwnedSlice(self.allocator);

        while (true) {
            if (self.peek().tag == .ident and self.peekN(1).tag == .assign) {
                const key = self.advance();
                _ = self.advance(); // '='
                const value = try self.parseConditional();
                try args.append(self.allocator, .{ .keyword = .{ .name = key.lexeme, .value = value } });
            } else {
                const value = try self.parseConditional();
                try args.append(self.allocator, .{ .positional = value });
            }
            if (self.eat(.comma)) continue;
            _ = try self.expect(.rparen);
            break;
        }

        return args.toOwnedSlice(self.allocator);
    }

    fn parsePrimary(self: *Parser) ExprError!*Expr {
        const tok = self.peek();
        switch (tok.tag) {
            .ident => {
                _ = self.advance();
                return self.allocExpr(.{ .ident = .{ .name = tok.lexeme, .slot = unknown_slot } });
            },
            .int => {
                _ = self.advance();
                const v = std.fmt.parseInt(i64, tok.lexeme, 10) catch return ExprError.InvalidNumber;
                return self.allocExpr(.{ .int_lit = v });
            },
            .float => {
                _ = self.advance();
                const v = std.fmt.parseFloat(f64, tok.lexeme) catch return ExprError.InvalidNumber;
                return self.allocExpr(.{ .float_lit = v });
            },
            .string => {
                _ = self.advance();
                const s = try self.unescapeString(tok.lexeme);
                return self.allocExpr(.{ .str_lit = s });
            },
            .bool_true => {
                _ = self.advance();
                return self.allocExpr(.{ .bool_lit = true });
            },
            .bool_false => {
                _ = self.advance();
                return self.allocExpr(.{ .bool_lit = false });
            },
            .none_kw => {
                _ = self.advance();
                return self.allocExpr(.null_lit);
            },
            .lparen => {
                _ = self.advance();
                const inner = try self.parseConditional();
                _ = try self.expect(.rparen);
                return inner;
            },
            .lbracket => {
                _ = self.advance();
                var items = std.ArrayList(*Expr).empty;
                errdefer items.deinit(self.allocator);
                if (!self.eat(.rbracket)) {
                    while (true) {
                        try items.append(self.allocator, try self.parseConditional());
                        if (self.eat(.comma)) continue;
                        _ = try self.expect(.rbracket);
                        break;
                    }
                }
                return self.allocExpr(.{ .list_lit = try items.toOwnedSlice(self.allocator) });
            },
            else => return ExprError.UnexpectedToken,
        }
    }
};

pub fn parseExpr(allocator: std.mem.Allocator, source: []const u8) ExprError!*Expr {
    const tokens = try lex(allocator, source);
    defer allocator.free(tokens);
    var parser = Parser{ .allocator = allocator, .tokens = tokens, .pos = 0 };
    return parser.parse();
}

pub const Scope = std.StringHashMap(Value);

pub const CallArgValue = union(enum) {
    positional: Value,
    keyword: struct { name: []const u8, value: Value },
};

pub const Runtime = struct {
    context: ?*anyopaque = null,
    callFunction: ?*const fn (context: ?*anyopaque, allocator: std.mem.Allocator, name: []const u8, args: []const CallArgValue, scope: *Scope) ExprError!Value = null,
    getIdentBySlot: ?*const fn (context: ?*anyopaque, slot: u32, name: []const u8, scope: *Scope) ?Value = null,
};

const StrWriter = struct {
    list: *std.ArrayList(u8),
    allocator: std.mem.Allocator,

    pub fn writeAll(self: *StrWriter, bytes: []const u8) !void {
        try self.list.appendSlice(self.allocator, bytes);
    }

    pub fn writeByte(self: *StrWriter, b: u8) !void {
        try self.list.append(self.allocator, b);
    }
};

fn valueToStringAlloc(allocator: std.mem.Allocator, value: Value) ![]u8 {
    switch (value) {
        .str => |s| return allocator.dupe(u8, s),
        .undefined, .null => return allocator.dupe(u8, ""),
        .bool => |b| return allocator.dupe(u8, if (b) "true" else "false"),
        .int => |v| {
            var buf: [32]u8 = undefined;
            const rendered = try std.fmt.bufPrint(&buf, "{}", .{v});
            return allocator.dupe(u8, rendered);
        },
        .float => |v| {
            var buf: [64]u8 = undefined;
            const rendered = try std.fmt.bufPrint(&buf, "{}", .{v});
            return allocator.dupe(u8, rendered);
        },
        else => {
            var list = std.ArrayList(u8).empty;
            errdefer list.deinit(allocator);
            var w = StrWriter{ .list = &list, .allocator = allocator };
            try value_mod.writeValue(&w, value);
            return list.toOwnedSlice(allocator);
        },
    }
}

const StringView = struct {
    bytes: []const u8,
    owned: ?[]u8 = null,
};

fn valueToStringView(allocator: std.mem.Allocator, value: Value) !StringView {
    return switch (value) {
        .str => |s| .{ .bytes = s, .owned = null },
        else => blk: {
            const owned = try valueToStringAlloc(allocator, value);
            break :blk .{ .bytes = owned, .owned = owned };
        },
    };
}

fn freeStringView(allocator: std.mem.Allocator, view: StringView) void {
    if (view.owned) |buf| allocator.free(buf);
}

fn concatValues(allocator: std.mem.Allocator, left: Value, right: Value) !Value {
    const left_view = try valueToStringView(allocator, left);
    defer freeStringView(allocator, left_view);
    const right_view = try valueToStringView(allocator, right);
    defer freeStringView(allocator, right_view);

    const out = try allocator.alloc(u8, left_view.bytes.len + right_view.bytes.len);
    @memcpy(out[0..left_view.bytes.len], left_view.bytes);
    @memcpy(out[left_view.bytes.len..], right_view.bytes);
    return .{ .str = out };
}

fn writeValueAsString(writer: anytype, value: Value) !void {
    switch (value) {
        .str => |s| try writer.writeAll(s),
        else => try value_mod.writeValue(writer, value),
    }
}

fn writeExprAsString(allocator: std.mem.Allocator, ex: *const Expr, scope: *Scope, runtime: *const Runtime, writer: anytype) ExprError!void {
    switch (ex.*) {
        .binary => |b| {
            switch (b.op) {
                .concat => {
                    try writeExprAsString(allocator, b.left, scope, runtime, writer);
                    try writeExprAsString(allocator, b.right, scope, runtime, writer);
                    return;
                },
                .add => {
                    const left = try evalExpr(allocator, b.left, scope, runtime);
                    const right = try evalExpr(allocator, b.right, scope, runtime);
                    if (left == .str or right == .str) {
                        try writeValueAsString(writer, left);
                        try writeValueAsString(writer, right);
                        return;
                    }
                    // Numeric add path.
                    const added = switch (left) {
                        .int => |lv| switch (right) {
                            .int => |rv| Value{ .int = lv + rv },
                            .float => |rv| Value{ .float = @as(f64, @floatFromInt(lv)) + rv },
                            else => return ExprError.TypeError,
                        },
                        .float => |lv| switch (right) {
                            .int => |rv| Value{ .float = lv + @as(f64, @floatFromInt(rv)) },
                            .float => |rv| Value{ .float = lv + rv },
                            else => return ExprError.TypeError,
                        },
                        else => return ExprError.TypeError,
                    };
                    try value_mod.writeValue(writer, added);
                    return;
                },
                else => {},
            }
        },
        else => {},
    }

    const v = try evalExpr(allocator, ex, scope, runtime);
    try value_mod.writeValue(writer, v);
}

pub fn writeExprOutput(allocator: std.mem.Allocator, ex: *const Expr, scope: *Scope, runtime: *const Runtime, writer: anytype) ExprError!void {
    return writeExprAsString(allocator, ex, scope, runtime, writer);
}

fn asF64(v: Value) ExprError!f64 {
    return switch (v) {
        .int => |x| @floatFromInt(x),
        .float => |x| x,
        else => ExprError.TypeError,
    };
}

fn normalizeIndex(len: usize, idx: i64) usize {
    const l: i64 = @intCast(len);
    var v = idx;
    if (v < 0) v = l + v;
    if (v < 0) return 0;
    if (v > l) return len;
    return @intCast(v);
}

fn toIntIndex(v: Value) ?i64 {
    return switch (v) {
        .int => |i| i,
        else => null,
    };
}

fn writeJsonString(writer: *StrWriter, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(c),
        }
    }
    try writer.writeByte('"');
}

fn writeJsonValue(writer: *StrWriter, value: Value) !void {
    switch (value) {
        .undefined, .null => try writer.writeAll("null"),
        .bool => |b| if (b) try writer.writeAll("true") else try writer.writeAll("false"),
        .int => |v| {
            var buf: [32]u8 = undefined;
            const s = try std.fmt.bufPrint(&buf, "{}", .{v});
            try writer.writeAll(s);
        },
        .float => |v| {
            var buf: [64]u8 = undefined;
            const s = try std.fmt.bufPrint(&buf, "{}", .{v});
            try writer.writeAll(s);
        },
        .str => |s| try writeJsonString(writer, s),
        .list => |items| {
            try writer.writeByte('[');
            for (items, 0..) |item, idx| {
                if (idx > 0) try writer.writeByte(',');
                try writeJsonValue(writer, item);
            }
            try writer.writeByte(']');
        },
        .map => |entries| {
            try writer.writeByte('{');
            for (entries, 0..) |entry, idx| {
                if (idx > 0) try writer.writeByte(',');
                try writeJsonString(writer, entry.key);
                try writer.writeByte(':');
                try writeJsonValue(writer, entry.value);
            }
            try writer.writeByte('}');
        },
        .namespace => |ns| {
            try writer.writeByte('{');
            var it = ns.map.iterator();
            var idx: usize = 0;
            while (it.next()) |entry| {
                if (idx > 0) try writer.writeByte(',');
                idx += 1;
                try writeJsonString(writer, entry.key_ptr.*);
                try writer.writeByte(':');
                try writeJsonValue(writer, entry.value_ptr.*);
            }
            try writer.writeByte('}');
        },
        .loopmeta => |lp| {
            try writer.writeByte('{');
            try writeJsonString(writer, "index0");
            try writer.writeByte(':');
            try writeJsonValue(writer, .{ .int = lp.index0 });
            try writer.writeByte(',');
            try writeJsonString(writer, "index");
            try writer.writeByte(':');
            try writeJsonValue(writer, .{ .int = lp.index });
            try writer.writeByte(',');
            try writeJsonString(writer, "first");
            try writer.writeByte(':');
            try writeJsonValue(writer, .{ .bool = lp.first });
            try writer.writeByte(',');
            try writeJsonString(writer, "last");
            try writer.writeByte(':');
            try writeJsonValue(writer, .{ .bool = lp.last });
            try writer.writeByte('}');
        },
    }
}

fn argKeywordValue(args: []const CallArg, name: []const u8) ?*Expr {
    for (args) |arg| {
        switch (arg) {
            .keyword => |kv| if (std.mem.eql(u8, kv.name, name)) return kv.value,
            else => {},
        }
    }
    return null;
}

fn valueInContainer(needle: Value, haystack: Value) bool {
    return switch (haystack) {
        .list => |items| for (items) |item| {
            if (Value.eql(needle, item)) break true;
        } else false,
        .str => |rs| switch (needle) {
            .str => |ls| std.mem.indexOf(u8, rs, ls) != null,
            else => false,
        },
        .map => |entries| switch (needle) {
            .str => |ls| for (entries) |entry| {
                if (std.mem.eql(u8, ls, entry.key)) break true;
            } else false,
            else => false,
        },
        .namespace => |ns| switch (needle) {
            .str => |ls| ns.map.contains(ls),
            else => false,
        },
        else => false,
    };
}

fn evalStringMethod(allocator: std.mem.Allocator, method_id: StringMethodId, s: []const u8, arg_values: []const CallArgValue) ExprError!Value {
    switch (method_id) {
        .endswith => {
            if (arg_values.len > 0 and arg_values[0] == .positional and arg_values[0].positional == .str) {
                return .{ .bool = std.mem.endsWith(u8, s, arg_values[0].positional.str) };
            }
            return .{ .bool = false };
        },
        .startswith => {
            if (arg_values.len > 0 and arg_values[0] == .positional and arg_values[0].positional == .str) {
                return .{ .bool = std.mem.startsWith(u8, s, arg_values[0].positional.str) };
            }
            return .{ .bool = false };
        },
        .split => {
            const delim: []const u8 = if (arg_values.len > 0 and arg_values[0] == .positional and arg_values[0].positional == .str) arg_values[0].positional.str else " ";
            var it = std.mem.splitSequence(u8, s, delim);
            var vals = std.ArrayList(Value).empty;
            errdefer vals.deinit(allocator);
            while (it.next()) |part| {
                try vals.append(allocator, .{ .str = try allocator.dupe(u8, part) });
            }
            return .{ .list = try vals.toOwnedSlice(allocator) };
        },
        .strip => {
            const cut = if (arg_values.len > 0 and arg_values[0] == .positional and arg_values[0].positional == .str) arg_values[0].positional.str else " \t\n\r";
            return .{ .str = try allocator.dupe(u8, std.mem.trim(u8, s, cut)) };
        },
        .lstrip => {
            const cut = if (arg_values.len > 0 and arg_values[0] == .positional and arg_values[0].positional == .str) arg_values[0].positional.str else " \t\n\r";
            return .{ .str = try allocator.dupe(u8, std.mem.trimStart(u8, s, cut)) };
        },
        .rstrip => {
            const cut = if (arg_values.len > 0 and arg_values[0] == .positional and arg_values[0].positional == .str) arg_values[0].positional.str else " \t\n\r";
            return .{ .str = try allocator.dupe(u8, std.mem.trimEnd(u8, s, cut)) };
        },
        .unknown => {},
    }
    return .undefined;
}

fn applyFilter(allocator: std.mem.Allocator, filter_id: FilterId, target: Value, args: []const CallArg, scope: *Scope, runtime: *const Runtime) ExprError!Value {
    switch (filter_id) {
        .items => {
                switch (target) {
                    .map => |entries| {
                        const out = try allocator.alloc(Value, entries.len);
                        for (entries, 0..) |entry, idx| {
                            const pair_vals = try allocator.alloc(Value, 2);
                            pair_vals[0] = .{ .str = entry.key };
                            pair_vals[1] = entry.value;
                            out[idx] = .{ .list = pair_vals };
                        }
                        return .{ .list = out };
                    },
                    .namespace => |ns| {
                        const count = ns.map.count();
                        const out = try allocator.alloc(Value, count);
                        var it = ns.map.iterator();
                        var idx: usize = 0;
                        while (it.next()) |entry| {
                            const pair_vals = try allocator.alloc(Value, 2);
                            pair_vals[0] = .{ .str = entry.key_ptr.* };
                            pair_vals[1] = entry.value_ptr.*;
                            out[idx] = .{ .list = pair_vals };
                            idx += 1;
                        }
                        return .{ .list = out };
                    },
                    else => return .{ .list = &[_]Value{} },
                }
        },
        .length => {
            const n: i64 = switch (target) {
                .undefined, .null => 0,
                .str => |s| @intCast(s.len),
                .list => |items| @intCast(items.len),
                .map => |entries| @intCast(entries.len),
                .namespace => |ns| @intCast(ns.map.count()),
                else => 0,
            };
            return .{ .int = n };
        },
        .safe => return target,
        .string => return .{ .str = try valueToStringAlloc(allocator, target) },
        .sort => {
            if (target != .list) return target;
            const items = target.list;
            const out = try allocator.alloc(Value, items.len);
            @memcpy(out, items);

            var attr_name: ?[]const u8 = null;
            if (argKeywordValue(args, "attribute")) |expr_ptr| {
                const v = try evalExpr(allocator, expr_ptr, scope, runtime);
                if (v == .str) attr_name = v.str;
            }

            const Ctx = struct {
                attr: ?[]const u8,
                fn less(ctx: @This(), a: Value, b: Value) bool {
                    const av = if (ctx.attr) |attr| (a.getAttr(attr) orelse .undefined) else a;
                    const bv = if (ctx.attr) |attr| (b.getAttr(attr) orelse .undefined) else b;
                    return switch (av) {
                        .str => |as| switch (bv) {
                            .str => |bs| std.mem.order(u8, as, bs) == .lt,
                            else => false,
                        },
                        .int => |ai| switch (bv) {
                            .int => |bi| ai < bi,
                            else => false,
                        },
                        else => false,
                    };
                }
            };
            std.mem.sort(Value, out, Ctx{ .attr = attr_name }, Ctx.less);
            return .{ .list = out };
        },
        .trim => {
            if (target != .str) return target;
            return .{ .str = try allocator.dupe(u8, std.mem.trim(u8, target.str, " \t\n\r")) };
        },
        .tojson => {
            var list = std.ArrayList(u8).empty;
            errdefer list.deinit(allocator);
            var w = StrWriter{ .list = &list, .allocator = allocator };
            try writeJsonValue(&w, target);
            return .{ .str = try list.toOwnedSlice(allocator) };
        },
        .unknown => {},
    }
    return target;
}

pub fn evalExpr(allocator: std.mem.Allocator, ex: *const Expr, scope: *Scope, runtime: *const Runtime) ExprError!Value {
    return switch (ex.*) {
        .ident => |ident| blk: {
            if (ident.slot != unknown_slot) {
                if (runtime.getIdentBySlot) |cb| {
                    if (cb(runtime.context, ident.slot, ident.name, scope)) |v| break :blk v;
                }
            }
            break :blk scope.get(ident.name) orelse .undefined;
        },
        .int_lit => |v| .{ .int = v },
        .float_lit => |v| .{ .float = v },
        .str_lit => |v| .{ .str = v },
        .bool_lit => |v| .{ .bool = v },
        .null_lit => .null,
        .list_lit => |items| {
            const out = try allocator.alloc(Value, items.len);
            for (items, 0..) |item, idx| out[idx] = try evalExpr(allocator, item, scope, runtime);
            return .{ .list = out };
        },
        .getattr => |item| {
            const target = try evalExpr(allocator, item.target, scope, runtime);
            return target.getAttr(item.attr) orelse .undefined;
        },
        .index => |item| {
            const target = try evalExpr(allocator, item.target, scope, runtime);
            const key = try evalExpr(allocator, item.key, scope, runtime);
            switch (target) {
                .list => |items| {
                    if (toIntIndex(key)) |idx| {
                        const pos = normalizeIndex(items.len, idx);
                        if (pos < items.len) return items[pos];
                    }
                    return .undefined;
                },
                .str => |s| {
                    if (toIntIndex(key)) |idx| {
                        const pos = normalizeIndex(s.len, idx);
                        if (pos < s.len) return .{ .str = try allocator.dupe(u8, s[pos .. pos + 1]) };
                    }
                    return .undefined;
                },
                .map => |entries| {
                    if (key == .str) {
                        for (entries) |entry| {
                            if (entry.key.ptr == key.str.ptr and entry.key.len == key.str.len) return entry.value;
                            if (entry.key.len != key.str.len) continue;
                            if (entry.key.len > 0 and entry.key[0] != key.str[0]) continue;
                            if (std.mem.eql(u8, entry.key, key.str)) return entry.value;
                        }
                    }
                    return .undefined;
                },
                .namespace => |ns| {
                    if (key == .str) return ns.map.get(key.str) orelse .undefined;
                    return .undefined;
                },
                else => return .undefined,
            }
        },
        .slice => |s| {
            const target = try evalExpr(allocator, s.target, scope, runtime);
            const start_idx: i64 = if (s.start) |sp| (toIntIndex(try evalExpr(allocator, sp, scope, runtime)) orelse 0) else 0;
            const end_idx: ?i64 = if (s.end) |ep| (toIntIndex(try evalExpr(allocator, ep, scope, runtime)) orelse 0) else null;
            const step_idx: i64 = if (s.step) |ep| (toIntIndex(try evalExpr(allocator, ep, scope, runtime)) orelse 1) else 1;
            if (step_idx == 0) return .undefined;
            switch (target) {
                .list => |items| {
                    const default_start: i64 = if (step_idx > 0) 0 else @intCast(items.len -| 1);
                    const default_end: i64 = if (step_idx > 0) @intCast(items.len) else -1;
                    var i: i64 = if (s.start != null) start_idx else default_start;
                    const end_bound: i64 = if (end_idx) |e| e else default_end;
                    var tmp = std.ArrayList(Value).empty;
                    errdefer tmp.deinit(allocator);
                    while ((step_idx > 0 and i < end_bound) or (step_idx < 0 and i > end_bound)) : (i += step_idx) {
                        const pos = normalizeIndex(items.len, i);
                        if (pos < items.len) try tmp.append(allocator, items[pos]);
                    }
                    const out = try tmp.toOwnedSlice(allocator);
                    return .{ .list = out };
                },
                .str => |str| {
                    const default_start: i64 = if (step_idx > 0) 0 else @intCast(str.len -| 1);
                    const default_end: i64 = if (step_idx > 0) @intCast(str.len) else -1;
                    var i: i64 = if (s.start != null) start_idx else default_start;
                    const end_bound: i64 = if (end_idx) |e| e else default_end;
                    var out = std.ArrayList(u8).empty;
                    errdefer out.deinit(allocator);
                    while ((step_idx > 0 and i < end_bound) or (step_idx < 0 and i > end_bound)) : (i += step_idx) {
                        const pos = normalizeIndex(str.len, i);
                        if (pos < str.len) try out.append(allocator, str[pos]);
                    }
                    return .{ .str = try out.toOwnedSlice(allocator) };
                },
                else => return .undefined,
            }
        },
        .call => |call| {
            var small_args: [8]CallArgValue = undefined;
            var heap_args: ?[]CallArgValue = null;
            const arg_values: []CallArgValue = if (call.args.len <= small_args.len)
                small_args[0..call.args.len]
            else blk: {
                const buf = try allocator.alloc(CallArgValue, call.args.len);
                heap_args = buf;
                break :blk buf;
            };
            defer if (heap_args) |buf| allocator.free(buf);

            for (call.args, 0..) |arg, idx| {
                arg_values[idx] = switch (arg) {
                    .positional => |p| .{ .positional = try evalExpr(allocator, p, scope, runtime) },
                    .keyword => |kv| .{ .keyword = .{ .name = kv.name, .value = try evalExpr(allocator, kv.value, scope, runtime) } },
                };
            }

            if (call.callee.* == .ident) {
                const callee_name = call.callee.ident.name;
                if (std.mem.eql(u8, callee_name, "raise_exception")) return ExprError.RaisedException;
                if (std.mem.eql(u8, callee_name, "namespace")) {
                    const ns_ptr = try allocator.create(value_mod.Namespace);
                    ns_ptr.* = value_mod.Namespace.init(allocator);
                    for (arg_values) |arg| {
                        switch (arg) {
                            .keyword => |kv| try ns_ptr.map.put(kv.name, kv.value),
                            else => {},
                        }
                    }
                    return .{ .namespace = ns_ptr };
                }
                if (runtime.callFunction) |cb| return cb(runtime.context, allocator, callee_name, arg_values, scope);
                return .undefined;
            }
            if (call.callee.* == .getattr) {
                const getter = call.callee.getattr;
                const target = try evalExpr(allocator, getter.target, scope, runtime);
                if (target == .str) {
                    return evalStringMethod(allocator, getter.method_id, target.str, arg_values);
                }
                if (std.mem.eql(u8, getter.attr, "items")) {
                    return applyFilter(allocator, .items, target, &[_]CallArg{}, scope, runtime);
                }
            }
            return .undefined;
        },
        .filter => |filter| {
            const target = try evalExpr(allocator, filter.target, scope, runtime);
            return applyFilter(allocator, filter.filter_id, target, filter.args, scope, runtime);
        },
        .test_expr => |test_ex| {
            const target = try evalExpr(allocator, test_ex.target, scope, runtime);
            var ok = switch (test_ex.test_id) {
                .defined => target != .undefined,
                .none => target == .null,
                .string => target == .str,
                .iterable => switch (target) { .str, .list, .map, .namespace, .loopmeta => true, else => false },
                .mapping => switch (target) { .map, .namespace => true, else => false },
                .sequence => switch (target) { .str, .list => true, else => false },
                .undefined => target == .undefined,
                .unknown => false,
            };
            if (test_ex.negated) ok = !ok;
            return .{ .bool = ok };
        },
        .cond_expr => |ce| {
            const cond = try evalExpr(allocator, ce.cond, scope, runtime);
            if (cond.isTruthy()) return evalExpr(allocator, ce.then_expr, scope, runtime);
            return evalExpr(allocator, ce.else_expr, scope, runtime);
        },
        .unary_not => |inner| {
            const v = try evalExpr(allocator, inner, scope, runtime);
            return .{ .bool = !v.isTruthy() };
        },
        .unary_neg => |inner| {
            const v = try evalExpr(allocator, inner, scope, runtime);
            return switch (v) {
                .int => |i| .{ .int = -i },
                .float => |f| .{ .float = -f },
                else => ExprError.TypeError,
            };
        },
        .binary => |b| {
            if (b.op == .@"and") {
                const left = try evalExpr(allocator, b.left, scope, runtime);
                if (!left.isTruthy()) return .{ .bool = false };
                const right = try evalExpr(allocator, b.right, scope, runtime);
                return .{ .bool = right.isTruthy() };
            }
            if (b.op == .@"or") {
                const left = try evalExpr(allocator, b.left, scope, runtime);
                if (left.isTruthy()) return .{ .bool = true };
                const right = try evalExpr(allocator, b.right, scope, runtime);
                return .{ .bool = right.isTruthy() };
            }

            const left = try evalExpr(allocator, b.left, scope, runtime);
            const right = try evalExpr(allocator, b.right, scope, runtime);

            return switch (b.op) {
                .add => {
                    if (left == .str or right == .str) {
                        return concatValues(allocator, left, right);
                    }
                    return switch (left) {
                        .int => |lv| switch (right) {
                            .int => |rv| .{ .int = lv + rv },
                            .float => |rv| .{ .float = @as(f64, @floatFromInt(lv)) + rv },
                            else => ExprError.TypeError,
                        },
                        .float => |lv| switch (right) {
                            .int => |rv| .{ .float = lv + @as(f64, @floatFromInt(rv)) },
                            .float => |rv| .{ .float = lv + rv },
                            else => ExprError.TypeError,
                        },
                        else => ExprError.TypeError,
                    };
                },
                .concat => {
                    return concatValues(allocator, left, right);
                },
                .sub => switch (left) {
                    .int => |lv| switch (right) {
                        .int => |rv| .{ .int = lv - rv },
                        .float => |rv| .{ .float = @as(f64, @floatFromInt(lv)) - rv },
                        else => ExprError.TypeError,
                    },
                    .float => |lv| switch (right) {
                        .int => |rv| .{ .float = lv - @as(f64, @floatFromInt(rv)) },
                        .float => |rv| .{ .float = lv - rv },
                        else => ExprError.TypeError,
                    },
                    else => ExprError.TypeError,
                },
                .mul => switch (left) {
                    .int => |lv| switch (right) {
                        .int => |rv| .{ .int = lv * rv },
                        .float => |rv| .{ .float = @as(f64, @floatFromInt(lv)) * rv },
                        else => ExprError.TypeError,
                    },
                    .float => |lv| switch (right) {
                        .int => |rv| .{ .float = lv * @as(f64, @floatFromInt(rv)) },
                        .float => |rv| .{ .float = lv * rv },
                        else => ExprError.TypeError,
                    },
                    else => ExprError.TypeError,
                },
                .div => .{ .float = (try asF64(left)) / (try asF64(right)) },
                .mod => switch (left) {
                    .int => |lv| switch (right) {
                        .int => |rv| .{ .int = @mod(lv, rv) },
                        else => ExprError.TypeError,
                    },
                    else => ExprError.TypeError,
                },
                .eq => .{ .bool = Value.eql(left, right) },
                .neq => .{ .bool = !Value.eql(left, right) },
                .lt => if (left == .str and right == .str) .{ .bool = std.mem.order(u8, left.str, right.str) == .lt } else .{ .bool = (try asF64(left)) < (try asF64(right)) },
                .lte => if (left == .str and right == .str) .{ .bool = std.mem.order(u8, left.str, right.str) != .gt } else .{ .bool = (try asF64(left)) <= (try asF64(right)) },
                .gt => if (left == .str and right == .str) .{ .bool = std.mem.order(u8, left.str, right.str) == .gt } else .{ .bool = (try asF64(left)) > (try asF64(right)) },
                .gte => if (left == .str and right == .str) .{ .bool = std.mem.order(u8, left.str, right.str) != .lt } else .{ .bool = (try asF64(left)) >= (try asF64(right)) },
                .in => .{ .bool = valueInContainer(left, right) },
                .not_in => .{ .bool = !valueInContainer(left, right) },
                .@"and", .@"or" => unreachable,
            };
        },
    };
}
