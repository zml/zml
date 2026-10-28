const std = @import("std");

const air = @import("mlir/dialects/air");
const arith = @import("mlir/dialects/arith");
const cf = @import("mlir/dialects/cf");
const common = @import("dsl/common/builder");
const func = @import("mlir/dialects/func");
const llvm = @import("mlir/dialects/llvm");
const mlir = @import("mlir");
const vector = @import("mlir/dialects/vector");

pub const types = @import("types.zig");

pub const Scalar = types.Scalar;
pub const Type = types.Type;
pub const Struct = types.Struct;
pub const StructField = types.StructField;
pub const Builtin = air.Builtin;

pub fn registerDialects(registry: *mlir.DialectRegistry) void {
    inline for (.{ air, arith, cf, func, llvm, vector }) |dialect| dialect.registerDialect(registry);
}

pub const FinishError = error{InvalidMlir} || std.mem.Allocator.Error || std.Io.Writer.Error;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Builder);
    std.testing.refAllDecls(Value);
}

pub const AddressSpace = air.AddressSpace;

/// `mem_flags`: `.mem_threadgroup`, or `.{ .device = true, .threadgroup = true }`
/// for `mem_device | mem_threadgroup`.
pub const MemFlags = packed struct(u32) {
    device: bool = false,
    threadgroup: bool = false,
    texture: bool = false,
    _padding: u29 = 0,

    pub const mem_none: MemFlags = .{};
    pub const mem_device: MemFlags = .{ .device = true };
    pub const mem_threadgroup: MemFlags = .{ .threadgroup = true };
    pub const mem_texture: MemFlags = .{ .texture = true };
};

/// A kernel argument, as `Builder.declareArgs` takes it. Device and constant
/// arguments take the next `[[buffer(n)]]`, threadgroup arguments the next
/// `[[threadgroup(n)]]`, in declaration order.
pub const ArgSpec = union(enum) {
    /// `device T *p`, or `const device T *p` with `.const_`.
    buffer: Buffer,
    /// `constant T &x`; the argument is its value.
    constant: Type,
    /// `constant S &s`; `Value.field` reads a field.
    constant_struct: Struct,
    /// `threadgroup T *p`.
    threadgroup: Type,
    thread_position_in_grid: Type,
    threadgroup_position_in_grid: Type,
    thread_position_in_threadgroup: Type,
    threads_per_threadgroup: Type,
    threadgroups_per_grid: Type,
    threads_per_grid: Type,
    thread_index_in_simdgroup: Type,
    simdgroup_index_in_threadgroup: Type,
    threads_per_simdgroup: Type,
    simdgroups_per_threadgroup: Type,
    thread_index_in_threadgroup: Type,

    pub const Buffer = struct {
        type: Type,
        const_: bool = false,
        /// `__restrict__`.
        restrict: bool = false,
    };
};

/// `raw` as an `ArgSpec`: an `ArgSpec`, or a literal of one of its variants
/// such as `.{ .buffer = .{ .type = .float } }`.
fn argSpec(raw: anytype) ArgSpec {
    if (@TypeOf(raw) == ArgSpec) return raw;
    const variant = comptime literalVariant(@TypeOf(raw), "ArgSpec");
    const inner = @field(raw, variant);
    return switch (@field(std.meta.Tag(ArgSpec), variant)) {
        .buffer => blk: {
            comptime knownFields(@TypeOf(inner), ArgSpec.Buffer, "ArgSpec.buffer");
            break :blk .{ .buffer = .{
                .type = typeOf(inner.type),
                .const_ = if (@hasField(@TypeOf(inner), "const_")) inner.const_ else false,
                .restrict = if (@hasField(@TypeOf(inner), "restrict")) inner.restrict else false,
            } };
        },
        .constant_struct => .{ .constant_struct = inner },
        inline else => |tag| @unionInit(ArgSpec, @tagName(tag), typeOf(inner)),
    };
}

/// The variant a spec literal names: one, as a union literal does.
fn literalVariant(Literal: type, what: []const u8) []const u8 {
    const names = @typeInfo(Literal).@"struct".field_names;
    if (names.len != 1) @compileError("dsl/metal: " ++ what ++ " literals name one variant, not " ++ std.fmt.comptimePrint("{d}", .{names.len}));
    return names[0];
}

/// Rejects a field of a spec literal that `Options` does not have.
fn knownFields(Literal: type, Options: type, what: []const u8) void {
    for (@typeInfo(Literal).@"struct".field_names) |name| {
        if (!@hasField(Options, name)) @compileError("dsl/metal: " ++ what ++ " has no field " ++ name);
    }
}

/// A `Type`, or the name of one (`.float`).
fn typeOf(t: anytype) Type {
    return if (@TypeOf(t) == @EnumLiteral()) @field(Type, @tagName(t)) else t;
}

/// A declared device, constant or threadgroup argument, for binding.
pub const Argument = struct {
    name: []const u8,
    kind: enum { buffer, constant, threadgroup },
    /// The `[[buffer(n)]]` or `[[threadgroup(n)]]` index.
    index: u32,
    /// The size of a constant argument.
    size: u64 = 0,
};

/// A pointer: its address space and pointee type.
pub const Pointer = struct {
    space: AddressSpace,
    pointee: Type,
};

/// An SSA value with its MSL type. Operators follow C++: operands take the
/// usual arithmetic conversions, integers narrower than `int` are promoted,
/// and signedness picks the operation (`uint / int` divides unsigned).
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,
    kind: Kind = undefined,

    pub const Kind = union(enum) {
        value: Type,
        pointer: Pointer,
        /// A `constant S &` argument.
        constant_struct: Struct,
        /// A `constant T &` argument or a local: an lvalue, read where it is
        /// used as C++ reads it.
        reference: Pointer,
        /// A `simdgroup_matrix<T, 8, 8>`: its 64 elements, of which a lane
        /// holds the first two.
        matrix: Scalar,
        /// A pointer local.
        pointer_reference: Pointer,
    };

    /// The value itself, an lvalue read.
    pub fn rvalue(self: Value) Value {
        return switch (self.kind) {
            .reference => |p| (Value{ .inner = self.inner, .kernel = self.kernel, .kind = .{ .pointer = p } }).load(0),
            .pointer_reference => |p| blk: {
                const k = self.kern();
                break :blk k.emit(llvm.load(k.ctx, self.inner, llvm.pointerType(k.ctx, p.space), .{ .alignment = 8 }, k.loc()), .{ .pointer = p });
            },
            else => self,
        };
    }

    /// `&x` of a local.
    pub fn address(reference: Value) Value {
        return switch (reference.kind) {
            .reference => |p| .{ .inner = reference.inner, .kernel = reference.kernel, .kind = .{ .pointer = p } },
            else => std.debug.panic("dsl/metal: the address of a {t}", .{reference.kind}),
        };
    }

    /// `x = value` of a local, converting `value` to its type.
    pub fn assign(self: Value, value: anytype) void {
        const k = self.kern();
        switch (self.kind) {
            .reference => |p| {
                if (p.space != .thread) std.debug.panic("dsl/metal: assigning to a {t} reference", .{p.space});
                (Value{ .inner = self.inner, .kernel = k, .kind = .{ .pointer = p } }).store(0, value);
            },
            .pointer_reference => |p| {
                const v = k.lift(value);
                if (v.kind != .pointer or v.kind.pointer.space != p.space) std.debug.panic("dsl/metal: assigning a {t} to a {t} pointer", .{ v.kind, p.space });
                _ = llvm.store(k.ctx, v.inner, self.inner, .{ .alignment = 8 }, k.loc()).appendTo(k.currentBlock());
            },
            else => std.debug.panic("dsl/metal: assigning to a {t}", .{self.kind}),
        }
    }

    fn kern(self: Value) *Builder {
        return self.kernel orelse @panic("dsl/metal: value has no builder");
    }

    /// The MSL type of a scalar or vector value.
    pub fn type_(self: Value) Type {
        return switch (self.kind) {
            .value => |t| t,
            .reference => |p| p.pointee,
            else => std.debug.panic("dsl/metal: expected a scalar or vector, got a {t}", .{self.kind}),
        };
    }

    pub fn pointer(self: Value) Pointer {
        return switch (self.kind) {
            .pointer => |p| p,
            else => std.debug.panic("dsl/metal: expected a pointer, got a {t}", .{self.kind}),
        };
    }

    pub fn add(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        if (self.kind == .pointer) return self.kern().offset(self, self.kern().promoted(rhs), false);
        return self.kern().arithmetic(.add, self, rhs);
    }
    pub fn sub(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        if (self.kind == .pointer) return self.kern().offset(self, self.kern().promoted(rhs), true);
        return self.kern().arithmetic(.sub, self, rhs);
    }
    pub fn mul(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().arithmetic(.mul, self, rhs);
    }
    pub fn div(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().arithmetic(.div, self, rhs);
    }
    pub fn rem(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().arithmetic(.rem, self, rhs);
    }
    pub fn bitAnd(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().arithmetic(.bit_and, self, rhs);
    }
    pub fn bitOr(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().arithmetic(.bit_or, self, rhs);
    }
    pub fn bitXor(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().arithmetic(.bit_xor, self, rhs);
    }
    pub fn shl(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().shift(.left, self, rhs);
    }
    pub fn shr(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().shift(.right, self, rhs);
    }
    pub fn neg(reference: Value) Value {
        const self = reference.rvalue();
        return self.kern().negate(self);
    }
    /// `~x`.
    pub fn bitNot(reference: Value) Value {
        const self = reference.rvalue();
        const k = self.kern();
        const v = k.convert(self, self.type_().withScalar(self.type_().scalar.promoted()));
        return v.bitXor(k.constant(v.type_(), -1));
    }

    pub fn lt(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().compare(.lt, self, rhs);
    }
    pub fn le(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().compare(.le, self, rhs);
    }
    pub fn gt(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().compare(.gt, self, rhs);
    }
    pub fn ge(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().compare(.ge, self, rhs);
    }
    pub fn eq(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().compare(.eq, self, rhs);
    }
    pub fn ne(reference: Value, rhs: anytype) Value {
        const self = reference.rvalue();
        return self.kern().compare(.ne, self, rhs);
    }

    /// `T(x)`, `(T) x` or `static_cast<T>(x)` on a scalar or vector; on a
    /// pointer, `(space T *) p`.
    pub fn to(reference: Value, t: Type) Value {
        const self = reference.rvalue();
        const k = self.kern();
        return switch (self.kind) {
            .pointer => |p| .{ .inner = self.inner, .kernel = k, .kind = .{ .pointer = .{ .space = p.space, .pointee = t } } },
            else => k.convert(self, t),
        };
    }

    /// `as_type<T>(x)`: the same bits as another type of the same size.
    pub fn asType(reference: Value, t: Type) Value {
        const self = reference.rvalue();
        const k = self.kern();
        const from = self.type_();
        if (from.size() != t.size() or from.len == 3 or t.len == 3)
            std.debug.panic("dsl/metal: as_type<{s}> of a {s}", .{ t.name(), from.name() });
        if (from.eql(t)) return self;
        return k.emit(llvm.bitcast(k.ctx, self.inner, t.mlirType(k.ctx), k.loc()), .{ .value = t });
    }

    /// `v.x`.
    pub fn x(reference: Value) Value {
        const self = reference.rvalue();
        return self.at(0);
    }
    pub fn y(reference: Value) Value {
        const self = reference.rvalue();
        return self.at(1);
    }
    pub fn z(reference: Value) Value {
        const self = reference.rvalue();
        return self.at(2);
    }
    pub fn w(reference: Value) Value {
        const self = reference.rvalue();
        return self.at(3);
    }

    /// `v[i]` of a vector; the index keeps its type, a literal is an `int`.
    pub fn at(reference: Value, i: anytype) Value {
        const self = reference.rvalue();
        const k = self.kern();
        const t = self.type_();
        if (!t.isVector()) std.debug.panic("dsl/metal: indexing a {s}", .{t.name()});
        const element = t.element();
        return k.emit(llvm.extractelement(k.ctx, self.inner, k.subscript(i).inner, element.mlirType(k.ctx), k.loc()), .{ .value = element });
    }

    /// The vector with `v[i] = element`, `i` typed as `at` types it; `v.y =
    /// element` is `insert(@as(i64, 1), element)`.
    pub fn insert(reference: Value, i: anytype, element: anytype) Value {
        const self = reference.rvalue();
        const k = self.kern();
        const t = self.type_();
        const e = k.convert(k.lift(element), t.element());
        return k.emit(llvm.insertelement(k.ctx, self.inner, e.inner, k.subscript(i).inner, k.loc()), .{ .value = t });
    }

    /// `p[i]`.
    pub fn load(reference: Value, i: anytype) Value {
        const self = reference.rvalue();
        const k = self.kern();
        const p = self.pointer();
        const at_ = k.offset(self, i, false);
        return k.loadMemory(at_.inner, p.pointee);
    }

    /// `p[i] = value`, converting `value` to the pointee type.
    pub fn store(reference: Value, i: anytype, value: anytype) void {
        const self = reference.rvalue();
        const k = self.kern();
        const p = self.pointer();
        var v = k.convert(k.lift(value), p.pointee);
        if (p.pointee.scalar == .bool) v = k.emit(arith.extui(k.ctx, v.inner, p.pointee.memory().mlirType(k.ctx), k.loc()), .{ .value = p.pointee.memory() });
        const at_ = k.offset(self, i, false);
        _ = llvm.store(k.ctx, v.inner, at_.inner, .{ .alignment = v.type_().alignment() }, k.loc()).appendTo(k.currentBlock());
    }

    /// `s.name` of a constant struct argument.
    pub fn field(reference: Value, comptime name: []const u8) Value {
        const self = reference.rvalue();
        const k = self.kern();
        const s = switch (self.kind) {
            .constant_struct => |s| s,
            else => std.debug.panic("dsl/metal: .{s} of a {t}", .{ name, self.kind }),
        };
        const index = s.fieldIndex(name);
        const f = s.fields[index];
        if (f.count != 0) std.debug.panic("dsl/metal: struct array field {s}", .{name});
        const gep = llvm.getelementptr(k.ctx, self.inner, &.{ .{ .constant = 0 }, .{ .constant = @intCast(index) } }, k.structType(s), k.loc());
        _ = gep.appendTo(k.currentBlock());
        return k.loadMemory(gep.result(0), f.type);
    }

    /// `m.thread_elements()` of a `simdgroup_matrix`: the lane's two
    /// elements.
    pub fn threadElements(m: Value) Value {
        const k = m.kern();
        const t: Type = .vec(Builder.matrixScalar(m), 2);
        return k.emit(llvm.shufflevector(k.ctx, m.inner, m.inner, t.mlirType(k.ctx), &.{ 0, 1 }, k.loc()), .{ .value = t });
    }
};

/// A loop: `while (c)` and `for (init; c; inc)` branch to the body with
/// `body(c)`, `for (;;)` has its body right away; `next()` ends the body
/// and starts the increment, and `close()` ends the loop.
///
/// A `break` and the end of the body meet in a latch that reads whether to
/// leave from a local, as clang's cleanup does: Apple's compiler then keeps
/// the loop's test at the top instead of rotating the loop, which is faster
/// (paged attention's decode loop by 2%).
pub const LoopScope = struct {
    kernel: *Builder,
    cond_block: *mlir.Block,
    end_block: *mlir.Block,
    loop_annotation: *const llvm.LoopAnnotationAttr,
    /// Where a `break` goes after setting `leave`; null until the first one.
    latch: ?*mlir.Block = null,
    leave: Value = undefined,
    exits: bool = false,

    /// Leaves the loop when `cond` is false.
    pub fn body(self: *LoopScope, cond: Value) void {
        const k = self.kernel;
        const body_block = mlir.Block.init(&.{}, &.{});
        k.condBranch(cond, body_block, self.end_block);
        self.exits = true;
        k.startBlock(body_block);
    }

    /// `break;`.
    pub fn breakLoop(self: *LoopScope) void {
        if (self.kernel.insert == null) return;
        self.toLatch(true);
        self.exits = true;
    }

    fn toLatch(self: *LoopScope, leave: bool) void {
        const k = self.kernel;
        if (self.latch == null) {
            self.latch = mlir.Block.init(&.{}, &.{});
            self.leave = k.local(.bool, null);
        }
        self.leave.assign(leave);
        _ = k.branch(self.latch.?);
    }

    /// Ends the body: the increment of `for (...; ...; inc)` follows.
    pub fn next(self: *LoopScope) void {
        const k = self.kernel;
        const inc = mlir.Block.init(&.{}, &.{});
        if (self.latch) |latch| {
            if (k.insert != null) self.toLatch(false);
            k.startBlock(latch);
            k.condBranch(self.leave, self.end_block, inc);
            self.latch = null;
        } else _ = k.branch(inc);
        k.startBlock(inc);
    }

    pub fn close(self: *LoopScope) void {
        const k = self.kernel;
        if (self.latch != null) self.next();
        k.branchLoop(self.cond_block, self.loop_annotation);
        if (self.exits) k.startBlock(self.end_block) else self.end_block.deinit();
    }
};

/// `for (T i = lower; i < upper; i += step)`: `i` is the local `iv`, the
/// condition and the increment follow C++'s conversions, and a step of 1 is
/// `++i`.
pub const ForScope = struct {
    loop: LoopScope,
    iv: Value,
    step: Step,

    const Step = union(enum) { one, value: Value, int: i64 };

    pub fn breakLoop(self: *ForScope) void {
        self.loop.breakLoop();
    }

    pub fn close(self: *ForScope) void {
        const k = self.loop.kernel;
        self.loop.next();
        if (k.insert != null) switch (self.step) {
            .one => self.iv.assign(k.increment(self.iv)),
            .value => |v| {
                const s = v.rvalue();
                self.iv.assign(self.iv.add(s));
            },
            .int => |n| self.iv.assign(self.iv.add(k.constant(if (std.math.cast(i32, n) != null) .int else .long, n))),
        };
        self.loop.close();
    }
};

/// `if (c) { ... }`, or `if (c) { ... } else { ... }` with `elseBranch()`
/// between the two.
pub const IfScope = struct {
    kernel: *Builder,
    else_block: ?*mlir.Block,
    end_block: *mlir.Block,
    joins: bool = false,

    pub fn elseBranch(self: *IfScope) void {
        const k = self.kernel;
        const else_block = self.else_block orelse @panic("dsl/metal: elseBranch of an if without else (open it with openIfElse(c, .{})), or a second one");
        if (k.branch(self.end_block)) self.joins = true;
        k.startBlock(else_block);
        self.else_block = null;
    }

    pub fn close(self: *IfScope) void {
        const k = self.kernel;
        if (k.branch(self.end_block)) self.joins = true;
        if (self.joins) k.startBlock(self.end_block) else self.end_block.deinit();
    }
};

/// A condition of `&&`, `||` and `!` terms, short-circuited: a term's code
/// runs only when the terms before it leave the outcome open. `and_(x)` is
/// `x && ...`, `or_(x)` is `x || ...`; the last term opens what the condition
/// guards. A group of `&&` terms ends a `||` chain and the
/// other way around, as in `a || (b && c)`.
pub const Condition = struct {
    kernel: *Builder,
    true_block: *mlir.Block,
    false_block: *mlir.Block,

    pub fn and_(self: *Condition, x: Value) void {
        self.term(x, false, .@"and");
    }
    pub fn andNot(self: *Condition, x: Value) void {
        self.term(x, true, .@"and");
    }
    pub fn or_(self: *Condition, x: Value) void {
        self.term(x, false, .@"or");
    }
    pub fn orNot(self: *Condition, x: Value) void {
        self.term(x, true, .@"or");
    }

    fn term(self: *Condition, x: Value, negate: bool, comptime op: enum { @"and", @"or" }) void {
        const k = self.kernel;
        const next = mlir.Block.init(&.{}, &.{});
        const on_true, const on_false = switch (op) {
            .@"and" => .{ next, self.false_block },
            .@"or" => .{ self.true_block, next },
        };
        if (negate) k.condBranch(x, on_false, on_true) else k.condBranch(x, on_true, on_false);
        k.startBlock(next);
    }

    fn last(self: *Condition, x: Value, negate: bool) void {
        if (negate) self.kernel.condBranch(x, self.false_block, self.true_block) else self.kernel.condBranch(x, self.true_block, self.false_block);
    }

    /// `if (... x) { ... }`.
    pub fn openIf(self: *Condition, x: Value) IfScope {
        return self.opened(x, false, false);
    }
    /// `if (... !x) { ... }`.
    pub fn openIfNot(self: *Condition, x: Value) IfScope {
        return self.opened(x, true, false);
    }
    /// `if (... x) { ... } else { ... }`; a chain has no result types.
    pub fn openIfElse(self: *Condition, x: Value, result_types: anytype) IfScope {
        noResults(result_types);
        return self.opened(x, false, true);
    }
    /// `if (... !x) { ... } else { ... }`.
    pub fn openIfElseNot(self: *Condition, x: Value, result_types: anytype) IfScope {
        noResults(result_types);
        return self.opened(x, true, true);
    }

    fn noResults(result_types: anytype) void {
        if (common.arity(@TypeOf(result_types), "Condition.openIfElse: result_types") != 0)
            @compileError("dsl/metal: a condition chain opens no `c ? a : b`");
    }

    fn opened(self: *Condition, x: Value, negate: bool, with_else: bool) IfScope {
        const k = self.kernel;
        self.last(x, negate);
        k.startBlock(self.true_block);
        if (!with_else) return .{ .kernel = k, .else_block = null, .end_block = self.false_block, .joins = true };
        return .{ .kernel = k, .else_block = self.false_block, .end_block = mlir.Block.init(&.{}, &.{}) };
    }
};

/// `a && b && ...` or `a || b || ...` as a bool, short-circuited: a
/// term's code runs only when the terms before it leave the result open.
/// `term(x)` adds a term followed by more, `close(x)` the last one.
pub const Logical = struct {
    kernel: *Builder,
    op: enum { @"and", @"or" },
    end_block: *mlir.Block,

    pub fn term(self: *Logical, x: Value) void {
        const k = self.kernel;
        const next = mlir.Block.init(&.{}, &.{});
        const c = k.convert(x, .bool);
        const decided: []const *const mlir.Value = &.{k.constant(.bool, self.op == .@"or").inner};
        _ = switch (self.op) {
            .@"and" => cf.cond_br(k.ctx, c.inner, next, &.{}, self.end_block, decided, .{}, k.loc()),
            .@"or" => cf.cond_br(k.ctx, c.inner, self.end_block, decided, next, &.{}, .{}, k.loc()),
        }.appendTo(k.currentBlock());
        k.insert = null;
        k.startBlock(next);
    }

    pub fn close(self: *Logical, x: Value) Value {
        const k = self.kernel;
        const c = k.convert(x, .bool);
        _ = cf.br(k.ctx, self.end_block, &.{c.inner}, k.loc()).appendTo(k.currentBlock());
        k.insert = null;
        k.startBlock(self.end_block);
        return .{ .inner = self.end_block.argument(0), .kernel = k, .kind = .{ .value = .bool } };
    }
};

/// `c ? a : b` evaluating one operand: the true operand's code, then
/// `yieldThen(.{a})`, the false operand's code, then `yieldElse(.{b})`;
/// `results[0]` is the value.
pub const TernaryScope = struct {
    kernel: *Builder,
    type: Type,
    else_block: *mlir.Block,
    end_block: *mlir.Block,
    results: [1]Value = undefined,

    pub fn yieldThen(self: *TernaryScope, values: anytype) void {
        const k = self.kernel;
        self.branchEnd(values, "yieldThen");
        k.startBlock(self.else_block);
    }

    pub fn yieldElse(self: *TernaryScope, values: anytype) void {
        const k = self.kernel;
        self.branchEnd(values, "yieldElse");
        k.startBlock(self.end_block);
        self.results = .{.{ .inner = self.end_block.argument(0), .kernel = k, .kind = .{ .value = self.type } }};
    }

    fn branchEnd(self: *TernaryScope, values: anytype, comptime what: []const u8) void {
        if (common.arity(@TypeOf(values), "TernaryScope." ++ what) != 1) @compileError("TernaryScope." ++ what ++ " takes one value");
        const k = self.kernel;
        const v = k.convert(k.lift(values[0]), self.type);
        _ = cf.br(k.ctx, self.end_block, &.{v.inner}, k.loc()).appendTo(k.currentBlock());
        k.insert = null;
    }
};

/// What `Builder.openIfElse` opens for `n` result types.
fn IfElseScope(comptime n: usize) type {
    return switch (n) {
        0 => IfScope,
        1 => TernaryScope,
        else => @compileError("dsl/metal: openIfElse has one result at most, as `c ? a : b`"),
    };
}

/// A loop pragma: `full` is `#pragma clang loop unroll(full)`, `enable`
/// `#pragma unroll`.
pub const Unroll = enum { full, enable };

pub const LoopOpts = struct {
    unroll: ?Unroll = null,
};

/// MSL's `metal::fast` and `metal::precise` functions.
pub const MathMode = enum { default, fast, precise };

/// `metal::fast::` or `metal::precise::`.
pub const Namespace = struct {
    b: *Builder,
    mode: MathMode,

    pub fn exp(self: Namespace, x: Value) Value {
        return self.b.floatMath("exp", self.mode, .{x});
    }
    pub fn exp2(self: Namespace, x: Value) Value {
        return self.b.floatMath("exp2", self.mode, .{x});
    }
    pub fn log(self: Namespace, x: Value) Value {
        return self.b.floatMath("log", self.mode, .{x});
    }
    pub fn log2(self: Namespace, x: Value) Value {
        return self.b.floatMath("log2", self.mode, .{x});
    }
    pub fn tanh(self: Namespace, x: Value) Value {
        return self.b.floatMath("tanh", self.mode, .{x});
    }
    pub fn sqrt(self: Namespace, x: Value) Value {
        return self.b.floatMath("sqrt", self.mode, .{x});
    }
    pub fn rsqrt(self: Namespace, x: Value) Value {
        return self.b.floatMath("rsqrt", self.mode, .{x});
    }
    pub fn sin(self: Namespace, x: Value) Value {
        return self.b.floatMath("sin", self.mode, .{x});
    }
    pub fn cos(self: Namespace, x: Value) Value {
        return self.b.floatMath("cos", self.mode, .{x});
    }
    pub fn max(self: Namespace, a: Value, b: anytype) Value {
        return self.b.floatMath("fmax", self.mode, .{ a, b });
    }
    pub fn min(self: Namespace, a: Value, b: anytype) Value {
        return self.b.floatMath("fmin", self.mode, .{ a, b });
    }
};

pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    name: []const u8,
    fast_math: bool,
    func_op: ?*mlir.Operation = null,
    entry_block: ?*mlir.Block = null,
    /// The kernel's body; blocks join it in the order they start.
    body: ?*mlir.Region = null,
    /// The block code goes to; null after a `break` or a `return`.
    insert: ?*mlir.Block = null,
    arguments: std.ArrayList(Argument) = .empty,

    pub const Opts = struct {
        /// MSL's default; `-fno-fast-math` otherwise.
        fast_math: bool = true,
    };

    pub fn open(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8, opts: Opts) Builder {
        const module: *mlir.Module = .init(.unknown(ctx));
        return .{
            .allocator = allocator,
            .arena = .init(allocator),
            .ctx = ctx,
            .module = module,
            .name = name,
            .fast_math = opts.fast_math,
        };
    }

    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    /// Declares the kernel's arguments, in order, from a struct literal of
    /// `ArgSpec`s, and returns them by name.
    pub fn declareArgs(self: *Builder, spec: anytype) FinishError!common.NamedArgs(@TypeOf(spec), Value) {
        const ctx = self.ctx;
        const location = self.loc();
        const fields = @typeInfo(@TypeOf(spec)).@"struct".field_names;
        const arena = self.arena.allocator();
        var arg_types: [fields.len]*const mlir.Type = undefined;
        var arg_attrs: [fields.len]*const mlir.Attribute = undefined;
        var kinds: [fields.len]Value.Kind = undefined;
        var buffers: u32 = 0;
        var threadgroups: u32 = 0;
        inline for (fields, 0..) |name, i| {
            const arg = argSpec(@field(spec, name));
            const opts: air.ArgumentOpts = .{ .name = name };
            switch (arg) {
                .buffer => |buffer| {
                    arg_types[i] = llvm.pointerType(ctx, AddressSpace.device);
                    arg_attrs[i] = air.bufferArgumentAttr(ctx, .{
                        .locationIndex = buffers,
                        .addressSpace = .device,
                        .access = if (buffer.const_) .read else .read_write,
                        .elementType = buffer.type.memory().mlirType(ctx),
                        .typeName = buffer.type.name(),
                        .typeSize = buffer.type.size(),
                        .typeAlignment = buffer.type.alignment(),
                        .noalias_ = buffer.restrict,
                    }, opts) catch return error.InvalidMlir;
                    kinds[i] = .{ .pointer = .{ .space = .device, .pointee = buffer.type } };
                    try self.arguments.append(arena, .{ .name = name, .kind = .buffer, .index = buffers });
                    buffers += 1;
                },
                .constant => |t| {
                    arg_types[i] = llvm.pointerType(ctx, AddressSpace.constant);
                    arg_attrs[i] = air.bufferArgumentAttr(ctx, .{
                        .locationIndex = buffers,
                        .addressSpace = .constant,
                        .access = .read,
                        .elementType = t.memory().mlirType(ctx),
                        .typeName = t.name(),
                        .typeSize = t.size(),
                        .typeAlignment = t.alignment(),
                        .bufferSize = t.size(),
                    }, opts) catch return error.InvalidMlir;
                    kinds[i] = .{ .reference = .{ .space = .constant, .pointee = t } };
                    try self.arguments.append(arena, .{ .name = name, .kind = .constant, .index = buffers, .size = t.size() });
                    buffers += 1;
                },
                .constant_struct => |s| {
                    var struct_fields = try arena.alloc(*const air.StructFieldAttr, s.fields.len);
                    for (s.fields, 0..) |f, j| struct_fields[j] = air.StructFieldAttr.get(ctx, .{
                        .offset = s.offset(j),
                        .size = f.type.size(),
                        .count = f.count,
                        .typeName = f.type.name(),
                        .name = f.name,
                    }) catch return error.InvalidMlir;
                    arg_types[i] = llvm.pointerType(ctx, AddressSpace.constant);
                    arg_attrs[i] = air.bufferArgumentAttr(ctx, .{
                        .locationIndex = buffers,
                        .addressSpace = .constant,
                        .access = .read,
                        .elementType = self.structType(s),
                        .typeName = s.name,
                        .typeSize = s.size(),
                        .typeAlignment = s.alignment(),
                        .bufferSize = s.size(),
                        .fields = struct_fields,
                    }, opts) catch return error.InvalidMlir;
                    kinds[i] = .{ .constant_struct = s };
                    try self.arguments.append(arena, .{ .name = name, .kind = .constant, .index = buffers, .size = s.size() });
                    buffers += 1;
                },
                .threadgroup => |t| {
                    arg_types[i] = llvm.pointerType(ctx, AddressSpace.threadgroup);
                    arg_attrs[i] = air.bufferArgumentAttr(ctx, .{
                        .locationIndex = threadgroups,
                        .addressSpace = .threadgroup,
                        .access = .read_write,
                        .elementType = t.memory().mlirType(ctx),
                        .typeName = t.name(),
                        .typeSize = t.size(),
                        .typeAlignment = t.alignment(),
                    }, opts) catch return error.InvalidMlir;
                    kinds[i] = .{ .pointer = .{ .space = .threadgroup, .pointee = t } };
                    try self.arguments.append(arena, .{ .name = name, .kind = .threadgroup, .index = threadgroups });
                    threadgroups += 1;
                },
                inline else => |t, builtin| {
                    arg_types[i] = t.mlirType(ctx);
                    arg_attrs[i] = air.builtinArgumentAttr(ctx, @field(air.Builtin, @tagName(builtin)), opts) catch return error.InvalidMlir;
                    kinds[i] = .{ .value = t };
                },
            }
        }
        var locs: [fields.len]*const mlir.Location = @splat(location);
        const block = mlir.Block.init(&arg_types, &locs);
        self.entry_block = block;
        self.func_op = air.kernel(ctx, .{
            .name = self.name,
            .block = block,
            .args_attributes = &arg_attrs,
            .location = location,
            .verify = false,
        }).appendTo(self.module.body());
        self.body = block.parentRegion();
        self.insert = block;

        var out: common.NamedArgs(@TypeOf(spec), Value) = undefined;
        inline for (fields, 0..) |name, i| @field(out, name) = .{ .inner = block.argument(i), .kernel = self, .kind = kinds[i] };
        return out;
    }

    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    /// The block code goes to: a fresh, unreachable one after a `break` or
    /// a `return`.
    pub fn currentBlock(self: *Builder) *mlir.Block {
        if (self.insert == null) self.startBlock(mlir.Block.init(&.{}, &.{}));
        return self.insert.?;
    }

    fn startBlock(self: *Builder, block: *mlir.Block) void {
        self.body.?.appendOwnedBlock(block);
        self.insert = block;
    }

    /// Branches to `dest` unless the code here is unreachable; true if it did.
    fn branch(self: *Builder, dest: *mlir.Block) bool {
        const block = self.insert orelse return false;
        _ = cf.br(self.ctx, dest, &.{}, self.loc()).appendTo(block);
        self.insert = null;
        return true;
    }

    fn condBranch(self: *Builder, cond: Value, on_true: *mlir.Block, on_false: *mlir.Block) void {
        const c = self.convert(cond, .bool);
        _ = cf.cond_br(self.ctx, c.inner, on_true, &.{}, on_false, &.{}, .{}, self.loc()).appendTo(self.currentBlock());
        self.insert = null;
    }

    /// The back edge of a loop, which carries its `llvm.loop` metadata.
    fn branchLoop(self: *Builder, dest: *mlir.Block, annotation: *const llvm.LoopAnnotationAttr) void {
        const block = self.insert orelse return;
        _ = llvm.br(self.ctx, dest, &.{}, .{ .loop_annotation = annotation }, self.loc()).appendTo(block);
        self.insert = null;
    }

    /// A thread-memory slot of `t`, at the top of the entry block.
    fn allocate(self: *Builder, t: *const mlir.Type, alignment: u64) *const mlir.Value {
        const ctx = self.ctx;
        const entry = self.entry_block.?;
        const size = arith.constant_int(ctx, 1, .int(ctx, .i32), self.loc());
        const alloca = llvm.alloca(ctx, size.result(0), llvm.pointerType(ctx, AddressSpace.thread), t, .{ .alignment = alignment }, self.loc());
        if (entry.firstOperation()) |first| entry.insertOwnedOperationBefore(first, alloca) else _ = alloca.appendTo(entry);
        entry.insertOwnedOperationBefore(alloca, size);
        return alloca.result(0);
    }

    /// `T x;`, or `T x = init;`: a local, read where it is used and written by
    /// `assign`.
    pub fn local(self: *Builder, t: Type, init: anytype) Value {
        const memory = t.memory();
        const x: Value = .{ .inner = self.allocate(memory.mlirType(self.ctx), memory.alignment()), .kernel = self, .kind = .{ .reference = .{ .space = .thread, .pointee = t } } };
        if (@TypeOf(init) != @TypeOf(null)) x.assign(init);
        return x;
    }

    /// `T *p = init;`: a pointer local.
    pub fn localPointer(self: *Builder, init: Value) Value {
        const p: Pointer = switch (init.kind) {
            .pointer, .pointer_reference => |q| q,
            else => std.debug.panic("dsl/metal: a pointer local of a {t}", .{init.kind}),
        };
        const x: Value = .{ .inner = self.allocate(llvm.pointerType(self.ctx, p.space), 8), .kernel = self, .kind = .{ .pointer_reference = p } };
        x.assign(init);
        return x;
    }

    /// A `simdgroup_matrix<T, 8, 8>` whose lane holds `elements`, two values
    /// converted to `T`: what assigning `m.thread_elements()` makes.
    pub fn matrix(self: *Builder, scalar: Scalar, elements: anytype) Value {
        const v = self.convert(self.lift(elements), .vec(scalar, 2));
        var mask: [64]i32 = @splat(-1);
        mask[0..2].* = .{ 0, 1 };
        return self.emit(llvm.shufflevector(self.ctx, v.inner, v.inner, self.matrixType(scalar), &mask, self.loc()), .{ .matrix = scalar });
    }

    /// `++i`: a narrow integer steps in its own type, as C++ steps it.
    fn increment(self: *Builder, i: Value) Value {
        const v = i.rvalue();
        const t = v.type_();
        if (t.scalar.isFloat() or t.scalar.bits() >= 32) return v.add(1);
        return self.emit(arith.addi(self.ctx, v.inner, self.constant(t, 1).inner, self.loc()), .{ .value = t });
    }

    fn emit(self: *Builder, op: *mlir.Operation, kind: Value.Kind) Value {
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self, .kind = kind };
    }

    fn fastmath(self: *const Builder) air.FloatOpts {
        return .{ .fastmath = if (self.fast_math) .fast else .none };
    }

    fn setFastmath(self: *Builder, op: *mlir.Operation) void {
        if (!self.fast_math) return;
        op.setAttribute(arith.fastmath(self.ctx, .fast) catch unreachable);
    }

    fn structType(self: *Builder, s: Struct) *const mlir.Type {
        var fields: [32]*const mlir.Type = undefined;
        for (s.fields, 0..) |f, i| {
            const t = f.type.memory().mlirType(self.ctx);
            fields[i] = if (f.count != 0) llvm.arrayType(t, f.count) else t;
        }
        var name: std.ArrayList(u8) = .empty;
        name.print(self.arena.allocator(), "struct.{s}", .{s.name}) catch @panic("dsl/metal: out of memory");
        return llvm.identifiedStructType(self.ctx, name.items, fields[0..s.fields.len]) orelse
            std.debug.panic("dsl/metal: two structs named {s}", .{s.name});
    }

    /// The MSL type of a Zig number, as `lift` gives it.
    fn literalType(value: anytype) Type {
        return switch (@typeInfo(@TypeOf(value))) {
            .comptime_int => if (value >= std.math.minInt(i32) and value <= std.math.maxInt(i32)) .int else .long,
            .comptime_float => .float,
            .int => |info| switch (info.bits) {
                8 => if (info.signedness == .signed) .char else .uchar,
                16 => if (info.signedness == .signed) .short else .ushort,
                32 => if (info.signedness == .signed) .int else .uint,
                64 => if (info.signedness == .signed) .long else .ulong,
                else => @compileError("dsl/metal: no MSL type for " ++ @typeName(@TypeOf(value))),
            },
            else => @compileError("dsl/metal: expected a literal, got " ++ @typeName(@TypeOf(value))),
        };
    }

    /// A Zig value as an MSL literal: integers are `int` (`long` when they do
    /// not fit), floats are `float`.
    pub fn lift(self: *Builder, value: anytype) Value {
        const T = @TypeOf(value);
        if (T == Value) return value.rvalue();
        return switch (@typeInfo(T)) {
            .comptime_int, .comptime_float, .int => self.constant(literalType(value), value),
            .bool => self.constant(.bool, value),
            .float => |info| self.constant(switch (info.bits) {
                16 => Type.half,
                32 => Type.float,
                else => @compileError("dsl/metal: no MSL type for " ++ @typeName(T)),
            }, value),
            else => @compileError("dsl/metal: expected a Value or a number, got " ++ @typeName(T)),
        };
    }

    /// A constant of type `t`, splat across a vector.
    pub fn constant(self: *Builder, t: Type, value: anytype) Value {
        const ctx = self.ctx;
        const ty = t.mlirType(ctx);
        const op = if (t.scalar.isFloat()) blk: {
            const v: f64 = switch (@typeInfo(@TypeOf(value))) {
                .int, .comptime_int => @floatFromInt(value),
                .bool => @floatFromInt(@intFromBool(value)),
                else => @floatCast(value),
            };
            if (t.isVector()) break :blk arith.constant_dense_splat_f(ctx, v, ty, self.loc());
            break :blk switch (t.scalar) {
                .half => arith.constant_float(ctx, v, .f16, self.loc()),
                .bfloat => arith.constant_float(ctx, v, .bf16, self.loc()),
                else => arith.constant_float(ctx, v, .f32, self.loc()),
            };
        } else blk: {
            const v: i64 = switch (@typeInfo(@TypeOf(value))) {
                .bool => @intFromBool(value),
                .int, .comptime_int => if (@TypeOf(value) == u64) @bitCast(value) else value,
                else => @intFromFloat(value),
            };
            if (t.isVector()) break :blk arith.constant_dense_splat_i(ctx, v, ty, self.loc());
            break :blk arith.constant_int(ctx, v, ty, self.loc());
        };
        return self.emit(op, .{ .value = t });
    }

    /// `T(a, b, ...)` from scalars, or `T(x)` splat from one scalar.
    pub fn vec(self: *Builder, t: Type, elements: anytype) Value {
        const fields = @typeInfo(@TypeOf(elements)).@"struct".field_names;
        if (fields.len == 1) return self.convert(self.lift(@field(elements, fields[0])), t);
        if (fields.len != t.len) std.debug.panic("dsl/metal: {d} elements of a {s}", .{ fields.len, t.name() });
        var inner: [fields.len]*const mlir.Value = undefined;
        inline for (fields, 0..) |name, i| inner[i] = self.convert(self.lift(@field(elements, name)), t.element()).inner;
        return self.emit(vector.from_elements(self.ctx, &inner, t.mlirType(self.ctx), self.loc()), .{ .value = t });
    }

    fn splat(self: *Builder, scalar: Value, t: Type) Value {
        return self.emit(vector.broadcast(self.ctx, scalar.inner, t.mlirType(self.ctx), self.loc()), .{ .value = t });
    }

    /// The conversion of `v` to `to`, as C++ converts: integer widths extend
    /// by the source's signedness or truncate, integers and floats convert
    /// through `air.convert`, and a scalar converts to a vector by splatting.
    pub fn convert(self: *Builder, reference: Value, to: Type) Value {
        const v = reference.rvalue();
        const ctx = self.ctx;
        const from = v.type_();
        if (from.eql(to)) return v;
        if (from.len != to.len) {
            if (!from.isVector()) return self.splat(self.convert(v, to.element()), to);
            std.debug.panic("dsl/metal: cannot convert a {s} to a {s}", .{ from.name(), to.name() });
        }
        const fs = from.scalar;
        const ts = to.scalar;
        const ty = to.mlirType(ctx);
        const kind: Value.Kind = .{ .value = to };
        if (ts == .bool) return v.ne(self.constant(from, 0));
        if (fs == .bool) {
            if (ts.isFloat()) return self.convert(self.convert(v, to.withScalar(.int)), to);
            return self.emit(arith.extui(ctx, v.inner, ty, self.loc()), kind);
        }
        if (fs.isFloat() and ts.isFloat()) {
            if (!from.isVector() and (fs == .float or ts == .float)) {
                const op = if (ts == .float) arith.extf(ctx, v.inner, ty, self.loc()) else arith.truncf(ctx, v.inner, ty, self.loc());
                return self.emit(op, kind);
            }
            return self.emit(air.convert(ctx, v.inner, ty, self.fastmath(), self.loc()), kind);
        }
        if (fs.isFloat()) {
            const op = if (ts.isUnsigned()) air.convert_u(ctx, v.inner, ty, .{}, self.loc()) else air.convert_s(ctx, v.inner, ty, .{}, self.loc());
            return self.emit(op, kind);
        }
        if (ts.isFloat()) {
            const op = if (fs.isUnsigned()) air.convert_u(ctx, v.inner, ty, self.fastmath(), self.loc()) else air.convert_s(ctx, v.inner, ty, self.fastmath(), self.loc());
            return self.emit(op, kind);
        }
        if (fs.bits() == ts.bits()) return .{ .inner = v.inner, .kernel = self, .kind = kind };
        if (fs.bits() > ts.bits()) return self.emit(arith.trunci(ctx, v.inner, ty, self.loc()), kind);
        const op = if (fs.isUnsigned()) arith.extui(ctx, v.inner, ty, self.loc()) else arith.extsi(ctx, v.inner, ty, self.loc());
        return self.emit(op, kind);
    }

    /// The operands of a binary operator after the usual arithmetic
    /// conversions; a scalar operand of a vector takes its element type.
    fn usualConversions(self: *Builder, a: Value, b_: anytype) [2]Value {
        const b = self.lift(b_);
        const ta = a.type_();
        const tb = b.type_();
        const converted: Type = if (ta.isVector() and tb.isVector()) blk: {
            if (!ta.eql(tb)) std.debug.panic("dsl/metal: {s} and {s} operands", .{ ta.name(), tb.name() });
            break :blk ta;
        } else if (ta.isVector()) ta else if (tb.isVector()) tb else .{ .scalar = types.commonScalar(ta.scalar, tb.scalar) };
        return .{ self.convert(a, converted), self.convert(b, converted) };
    }

    const ArithmeticOp = enum { add, sub, mul, div, rem, bit_and, bit_or, bit_xor };

    fn arithmetic(self: *Builder, comptime op: ArithmeticOp, a: Value, b: anytype) Value {
        const ctx = self.ctx;
        const l, const r = self.usualConversions(a, b);
        const t = l.type_();
        const loc_ = self.loc();
        if (t.scalar.isFloat()) {
            const result = switch (op) {
                .add => arith.addf(ctx, l.inner, r.inner, loc_),
                .sub => arith.subf(ctx, l.inner, r.inner, loc_),
                .mul => arith.mulf(ctx, l.inner, r.inner, loc_),
                .div => arith.divf(ctx, l.inner, r.inner, loc_),
                else => std.debug.panic("dsl/metal: {t} of floats", .{op}),
            };
            self.setFastmath(result);
            return self.emit(result, .{ .value = t });
        }
        const unsigned = t.scalar.isUnsigned();
        const result = switch (op) {
            .add => arith.addi(ctx, l.inner, r.inner, loc_),
            .sub => arith.subi(ctx, l.inner, r.inner, loc_),
            .mul => arith.muli(ctx, l.inner, r.inner, loc_),
            .div => if (unsigned) arith.divui(ctx, l.inner, r.inner, loc_) else arith.divsi(ctx, l.inner, r.inner, loc_),
            .rem => if (unsigned) arith.remui(ctx, l.inner, r.inner, loc_) else arith.remsi(ctx, l.inner, r.inner, loc_),
            .bit_and => arith.andi(ctx, l.inner, r.inner, loc_),
            .bit_or => arith.ori(ctx, l.inner, r.inner, loc_),
            .bit_xor => arith.xori(ctx, l.inner, r.inner, loc_),
        };
        // Signed overflow is undefined in C++.
        if (!unsigned and (op == .add or op == .sub or op == .mul)) {
            result.setAttribute(arith.overflowFlags(ctx, .{ .nsw = true }) catch unreachable);
        }
        return self.emit(result, .{ .value = t });
    }

    /// `a << b` or `a >> b`. MSL shifts by the amount modulo the width:
    /// clang casts the promoted amount to `a`'s promoted type as unsigned,
    /// then masks it, which folds away for a number within the width.
    fn shift(self: *Builder, comptime direction: enum { left, right }, a: Value, b: anytype) Value {
        const t = a.type_().withScalar(a.type_().scalar.promoted());
        const l = self.convert(a, t);
        const mask = t.scalar.bits() - 1;
        const r = if (@TypeOf(b) == Value) blk: {
            const rb = b.rvalue();
            const amount = self.shiftAmount(self.convert(rb, rb.type_().withScalar(rb.type_().scalar.promoted())), t);
            break :blk self.emit(arith.andi(self.ctx, amount.inner, self.constant(t, mask).inner, self.loc()), .{ .value = t });
        } else blk: {
            if (b < 0 or b > mask) std.debug.panic("dsl/metal: a shift of a {s} by {d}, past its width", .{ t.name(), b });
            const rb = self.lift(b);
            break :blk self.convert(self.convert(rb, rb.type_().withScalar(rb.type_().scalar.promoted())), t);
        };
        const loc_ = self.loc();
        const op = switch (direction) {
            .left => arith.shli(self.ctx, l.inner, r.inner, loc_),
            .right => if (t.scalar.isUnsigned()) arith.shrui(self.ctx, l.inner, r.inner, loc_) else arith.shrsi(self.ctx, l.inner, r.inner, loc_),
        };
        return self.emit(op, .{ .value = t });
    }

    /// A promoted shift amount `v` as the shifted type `t`: a scalar of a
    /// vector converts and splats, as C++ converts it; otherwise the cast is
    /// unsigned, as clang casts it.
    fn shiftAmount(self: *Builder, v: Value, t: Type) Value {
        const from = v.type_();
        if (from.len != t.len) return self.convert(v, t);
        if (from.scalar.bits() == t.scalar.bits()) return .{ .inner = v.inner, .kernel = self, .kind = .{ .value = t } };
        const ty = t.mlirType(self.ctx);
        const op = if (from.scalar.bits() > t.scalar.bits()) arith.trunci(self.ctx, v.inner, ty, self.loc()) else arith.extui(self.ctx, v.inner, ty, self.loc());
        return self.emit(op, .{ .value = t });
    }

    fn negate(self: *Builder, a: Value) Value {
        const t = a.type_();
        if (t.scalar.isFloat()) {
            const op = arith.negf(self.ctx, a.inner, self.loc());
            self.setFastmath(op);
            return self.emit(op, .{ .value = t });
        }
        const x = self.convert(a, t.withScalar(t.scalar.promoted()));
        return self.constant(x.type_(), 0).sub(x);
    }

    const Comparison = enum { lt, le, gt, ge, eq, ne };

    fn compare(self: *Builder, comptime op: Comparison, a: Value, b: anytype) Value {
        const l, const r = self.usualConversions(a, b);
        const t = l.type_();
        const result_type = t.withScalar(.bool);
        if (t.scalar.isFloat()) {
            const predicate: arith.CmpFPredicate = switch (op) {
                .lt => .olt,
                .le => .ole,
                .gt => .ogt,
                .ge => .oge,
                .eq => .oeq,
                .ne => .une,
            };
            const result = arith.cmpf(self.ctx, predicate, l.inner, r.inner, self.loc());
            self.setFastmath(result);
            return self.emit(result, .{ .value = result_type });
        }
        const unsigned = t.scalar.isUnsigned();
        const predicate: arith.CmpIPredicate = switch (op) {
            .lt => if (unsigned) .ult else .slt,
            .le => if (unsigned) .ule else .sle,
            .gt => if (unsigned) .ugt else .sgt,
            .ge => if (unsigned) .uge else .sge,
            .eq => .eq,
            .ne => .ne,
        };
        return self.emit(arith.cmpi(self.ctx, predicate, l.inner, r.inner, self.loc()), .{ .value = result_type });
    }

    /// `i` of `p + i` and `p - i`, promoted as C++ promotes an integer
    /// operand; `p[i]` and a vector subscript take `i` as it is.
    fn promoted(self: *Builder, i: anytype) Value {
        const v = self.subscript(i);
        const t = v.type_();
        return self.convert(v, t.withScalar(t.scalar.promoted()));
    }

    /// A vector subscript, of its own type as clang keeps it: a literal is an
    /// `int`, a `ushort` an `i16`, a `bool` an `i1`.
    fn subscript(self: *Builder, i: anytype) Value {
        const v = self.lift(i);
        const t = v.type_();
        if (t.isVector() or t.scalar.isFloat()) std.debug.panic("dsl/metal: an index of a {s}", .{t.name()});
        return v;
    }

    /// A `t` read from `p`, where it is a `t.memory()`.
    fn loadMemory(self: *Builder, p: *const mlir.Value, t: Type) Value {
        const memory = t.memory();
        const x = self.emit(llvm.load(self.ctx, p, memory.mlirType(self.ctx), .{ .alignment = memory.alignment() }, self.loc()), .{ .value = memory });
        if (t.scalar != .bool) return x;
        return self.emit(arith.trunci(self.ctx, x.inner, t.mlirType(self.ctx), self.loc()), .{ .value = t });
    }

    /// `p + i` (`p - i` when `negative`) in elements of the pointee type.
    fn offset(self: *Builder, p: Value, i: anytype, negative: bool) Value {
        if (@TypeOf(i) == comptime_int and i == 0) return p;
        const pointee = switch (p.kind) {
            .pointer => |ptr| ptr.pointee,
            else => std.debug.panic("dsl/metal: pointer arithmetic on a {t}", .{p.kind}),
        };
        var index = self.lift(i);
        if (index.type_().scalar.isFloat() or index.type_().isVector()) std.debug.panic("dsl/metal: pointer offset by a {s}", .{index.type_().name()});
        index = self.convert(index, if (index.type_().scalar.isUnsigned()) .ulong else .long);
        if (negative) index = self.constant(.long, 0).sub(index);
        const gep = llvm.getelementptr(self.ctx, p.inner, &.{.{ .value = index.inner }}, pointee.memory().mlirType(self.ctx), self.loc());
        return self.emit(gep, p.kind);
    }

    /// `cond ? a : b` of evaluated operands, a select;
    /// `openIfElse(cond, .{T})` evaluates one operand.
    pub fn ternary(self: *Builder, cond: Value, a: anytype, b: anytype) Value {
        const l, const r = self.usualConversions(self.lift(a), b);
        const c = self.convert(cond, if (l.type_().isVector() and cond.type_().isVector()) l.type_().withScalar(.bool) else .bool);
        return self.emit(arith.select(self.ctx, c.inner, l.inner, r.inner, self.loc()), l.kind);
    }

    fn openTernary(self: *Builder, cond: Value, t: Type) TernaryScope {
        const then_block = mlir.Block.init(&.{}, &.{});
        const else_block = mlir.Block.init(&.{}, &.{});
        var loc_ = [1]*const mlir.Location{self.loc()};
        const end_block = mlir.Block.init(&.{t.mlirType(self.ctx)}, &loc_);
        self.condBranch(cond, then_block, else_block);
        self.startBlock(then_block);
        return .{ .kernel = self, .type = t, .else_block = else_block, .end_block = end_block };
    }

    /// `T a[n] = {};`, zeroed as clang zeroes it.
    pub fn zeroedArray(self: *Builder, t: Type, n: u32) Value {
        const a = self.array(t, n);
        const len = self.constant(.long, @as(i64, n) * @as(i64, @intCast(t.size())));
        const operands = [_]*const mlir.Value{ a.inner, self.constant(.char, 0).inner, len.inner, self.constant(.bool, false).inner };
        _ = llvm.call_intrinsic(self.ctx, "llvm.memset.p0.i64", &operands, .{}, self.loc()).appendTo(self.currentBlock());
        return a;
    }

    /// `T a[n];` in thread memory, as a pointer to its first element.
    pub fn array(self: *Builder, t: Type, n: u32) Value {
        return .{ .inner = self.allocate(llvm.arrayType(t.memory().mlirType(self.ctx), n), t.alignment()), .kernel = self, .kind = .{ .pointer = .{ .space = .thread, .pointee = t } } };
    }

    fn floatMath(self: *Builder, comptime name: []const u8, mode: MathMode, args: anytype) Value {
        const x = self.lift(args[0]);
        const t = x.type_();
        if (!t.scalar.isFloat() or t.scalar == .bfloat)
            std.debug.panic("dsl/metal: no {s} of a {s}", .{ name, t.name() });
        const use_fast = switch (mode) {
            .default => self.fast_math and t.scalar == .float,
            .fast => true,
            .precise => false,
        };
        if (use_fast and t.scalar != .float) std.debug.panic("dsl/metal: fast::{s} of a {s}", .{ name, t.name() });
        const opts = self.fastmath();
        const loc_ = self.loc();
        const op = switch (args.len) {
            1 => if (use_fast) @field(air, "fast_" ++ name)(self.ctx, x.inner, opts, loc_) else @field(air, name)(self.ctx, x.inner, opts, loc_),
            2 => blk: {
                const y = self.convert(self.lift(args[1]), t);
                break :blk if (use_fast) @field(air, "fast_" ++ name)(self.ctx, x.inner, y.inner, opts, loc_) else @field(air, name)(self.ctx, x.inner, y.inner, opts, loc_);
            },
            3 => blk: {
                const y = self.convert(self.lift(args[1]), t);
                const z = self.convert(self.lift(args[2]), t);
                break :blk if (use_fast) @field(air, "fast_" ++ name)(self.ctx, x.inner, y.inner, z.inner, opts, loc_) else @field(air, name)(self.ctx, x.inner, y.inner, z.inner, opts, loc_);
            },
            else => unreachable,
        };
        return self.emit(op, .{ .value = t });
    }

    /// `metal::fast::f`: `b.fast().exp2(x)`.
    pub fn fast(self: *Builder) Namespace {
        return .{ .b = self, .mode = .fast };
    }

    /// `metal::precise::f`: `b.precise().tanh(x)`.
    pub fn precise(self: *Builder) Namespace {
        return .{ .b = self, .mode = .precise };
    }

    pub fn exp(self: *Builder, x: Value) Value {
        return self.floatMath("exp", .default, .{x});
    }
    pub fn exp2(self: *Builder, x: Value) Value {
        return self.floatMath("exp2", .default, .{x});
    }
    pub fn log(self: *Builder, x: Value) Value {
        return self.floatMath("log", .default, .{x});
    }
    pub fn log2(self: *Builder, x: Value) Value {
        return self.floatMath("log2", .default, .{x});
    }
    pub fn tanh(self: *Builder, x: Value) Value {
        return self.floatMath("tanh", .default, .{x});
    }
    pub fn sqrt(self: *Builder, x: Value) Value {
        return self.floatMath("sqrt", .default, .{x});
    }
    pub fn rsqrt(self: *Builder, x: Value) Value {
        return self.floatMath("rsqrt", .default, .{x});
    }
    pub fn ceil(self: *Builder, x: Value) Value {
        return self.floatMath("ceil", .default, .{x});
    }
    pub fn floor(self: *Builder, x: Value) Value {
        return self.floatMath("floor", .default, .{x});
    }
    pub fn round(self: *Builder, x: Value) Value {
        return self.floatMath("round", .default, .{x});
    }
    pub fn trunc(self: *Builder, x: Value) Value {
        return self.floatMath("trunc", .default, .{x});
    }
    pub fn sin(self: *Builder, x: Value) Value {
        return self.floatMath("sin", .default, .{x});
    }
    pub fn cos(self: *Builder, x: Value) Value {
        return self.floatMath("cos", .default, .{x});
    }
    pub fn pow(self: *Builder, x: Value, y: anytype) Value {
        return self.floatMath("pow", .default, .{ x, y });
    }
    pub fn fmod(self: *Builder, x: Value, y: anytype) Value {
        return self.floatMath("fmod", .default, .{ x, y });
    }

    /// `fma(a, b, c)` of floats.
    pub fn fma(self: *Builder, a_: Value, b: anytype, c: anytype) Value {
        const a = a_.rvalue();
        const t = a.type_();
        if (!t.scalar.isFloat() or t.scalar == .bfloat) std.debug.panic("dsl/metal: fma of a {s}", .{t.name()});
        const y = self.convert(self.lift(b), t);
        const z = self.convert(self.lift(c), t);
        return self.emit(air.fma(self.ctx, a.inner, y.inner, z.inner, self.fastmath(), self.loc()), .{ .value = t });
    }

    /// `dot(a, b)` of float vectors.
    pub fn dot(self: *Builder, a_: Value, b_: Value) Value {
        const a = a_.rvalue();
        const b = b_.rvalue();
        const t = a.type_();
        if (!t.eql(b.type_()) or !t.isVector()) std.debug.panic("dsl/metal: dot of {s} and {s}", .{ t.name(), b.type_().name() });
        return self.emit(air.dot(self.ctx, a.inner, b.inner, self.fastmath(), self.loc()), .{ .value = t.element() });
    }

    /// `abs(x)`.
    pub fn abs(self: *Builder, x_: Value) Value {
        const x = x_.rvalue();
        const t = x.type_();
        if (t.scalar.isFloat()) return self.floatMath("fabs", .default, .{x});
        if (t.scalar.isUnsigned()) return x;
        return self.emit(air.abs_s(self.ctx, x.inner, self.loc()), .{ .value = t });
    }

    /// The operands of an MSL function taking several values of one type,
    /// read, with literals converted to the other operand's type.
    fn sameType(self: *Builder, a: anytype, b: anytype) [2]Value {
        if (@TypeOf(a) != Value) {
            const r = self.lift(b);
            return .{ self.convert(self.lift(a), r.type_()), r };
        }
        const l = a.rvalue();
        const r = if (@TypeOf(b) == Value) b.rvalue() else self.convert(self.lift(b), l.type_());
        if (!l.type_().eql(r.type_())) std.debug.panic("dsl/metal: {s} and {s} arguments", .{ l.type_().name(), r.type_().name() });
        return .{ l, r };
    }

    /// `max(a, b)`.
    pub fn max(self: *Builder, a: anytype, b: anytype) Value {
        const l, const r = self.sameType(a, b);
        const t = l.type_();
        if (t.scalar.isFloat()) return self.floatMath("fmax", .default, .{ l, r });
        const op = if (t.scalar.isUnsigned()) air.max_u(self.ctx, l.inner, r.inner, self.loc()) else air.max_s(self.ctx, l.inner, r.inner, self.loc());
        return self.emit(op, .{ .value = t });
    }

    /// `min(a, b)`.
    pub fn min(self: *Builder, a: anytype, b: anytype) Value {
        const l, const r = self.sameType(a, b);
        const t = l.type_();
        if (t.scalar.isFloat()) return self.floatMath("fmin", .default, .{ l, r });
        const op = if (t.scalar.isUnsigned()) air.min_u(self.ctx, l.inner, r.inner, self.loc()) else air.min_s(self.ctx, l.inner, r.inner, self.loc());
        return self.emit(op, .{ .value = t });
    }

    /// `clamp(x, lo, hi)`.
    pub fn clamp(self: *Builder, x_: Value, lo: anytype, hi: anytype) Value {
        const x = x_.rvalue();
        const t = x.type_();
        const l = self.convert(self.lift(lo), t);
        const h = self.convert(self.lift(hi), t);
        if (t.scalar.isFloat()) return self.floatMath("clamp", .default, .{ x, l, h });
        const op = if (t.scalar.isUnsigned()) air.clamp_u(self.ctx, x.inner, l.inner, h.inner, self.loc()) else air.clamp_s(self.ctx, x.inner, l.inner, h.inner, self.loc());
        return self.emit(op, .{ .value = t });
    }

    fn simdType(x: Value, comptime name: []const u8) Type {
        const t = x.type_();
        switch (t.scalar) {
            .half, .float, .short, .ushort, .int, .uint => {},
            else => std.debug.panic("dsl/metal: no {s} of a {s}", .{ name, t.name() }),
        }
        return t;
    }

    fn simdReduce(self: *Builder, comptime name: []const u8, x_: Value) Value {
        const x = x_.rvalue();
        const t = simdType(x, name);
        const op = if (t.scalar.isFloat())
            @field(air, name)(self.ctx, x.inner, self.fastmath(), self.loc())
        else if (t.scalar.isUnsigned())
            @field(air, name ++ "_u")(self.ctx, x.inner, self.loc())
        else
            @field(air, name ++ "_s")(self.ctx, x.inner, self.loc());
        return self.emit(op, .{ .value = t });
    }

    /// A function moving `x` across lanes; `x` is frozen, so that a poison
    /// lane does not poison the others.
    fn shuffle(self: *Builder, comptime name: []const u8, x_: Value, lane: anytype) Value {
        const read = x_.rvalue();
        const t = simdType(read, name);
        const x = self.emit(llvm.freeze(self.ctx, read.inner, self.loc()), .{ .value = t });
        const l = self.convert(self.lift(lane), .ushort);
        const op = if (t.scalar.isFloat())
            @field(air, name)(self.ctx, x.inner, l.inner, self.fastmath(), self.loc())
        else if (t.scalar.isUnsigned())
            @field(air, name ++ "_u")(self.ctx, x.inner, l.inner, self.loc())
        else
            @field(air, name ++ "_s")(self.ctx, x.inner, l.inner, self.loc());
        return self.emit(op, .{ .value = t });
    }

    pub fn simdSum(self: *Builder, x: Value) Value {
        return self.simdReduce("simd_sum", x);
    }
    pub fn simdMax(self: *Builder, x: Value) Value {
        return self.simdReduce("simd_max", x);
    }
    pub fn simdMin(self: *Builder, x: Value) Value {
        return self.simdReduce("simd_min", x);
    }
    pub fn simdPrefixExclusiveSum(self: *Builder, x_: Value) Value {
        const x = x_.rvalue();
        const t = x.type_();
        if (!t.scalar.isFloat() or t.scalar == .bfloat) std.debug.panic("dsl/metal: simd_prefix_exclusive_sum of a {s}", .{t.name()});
        return self.emit(air.simd_prefix_exclusive_sum(self.ctx, x.inner, self.fastmath(), self.loc()), .{ .value = t });
    }
    /// `simd_shuffle(x, lane)`.
    pub fn simdShuffle(self: *Builder, x: Value, lane: anytype) Value {
        return self.shuffle("simd_shuffle", x, lane);
    }
    /// `simd_shuffle_xor(x, mask)`.
    pub fn simdShuffleXor(self: *Builder, x: Value, mask: anytype) Value {
        return self.shuffle("simd_shuffle_xor", x, mask);
    }
    /// `simd_shuffle_down(x, delta)`.
    pub fn simdShuffleDown(self: *Builder, x: Value, delta: anytype) Value {
        return self.shuffle("simd_shuffle_down", x, delta);
    }
    /// `simd_broadcast(x, lane)`.
    pub fn simdBroadcast(self: *Builder, x: Value, lane: anytype) Value {
        return self.shuffle("simd_broadcast", x, lane);
    }
    pub fn simdAny(self: *Builder, x: Value) Value {
        return self.emit(air.simd_any(self.ctx, self.convert(x, .bool).inner, self.loc()), .{ .value = .bool });
    }
    pub fn simdAll(self: *Builder, x: Value) Value {
        return self.emit(air.simd_all(self.ctx, self.convert(x, .bool).inner, self.loc()), .{ .value = .bool });
    }

    /// `threadgroup_barrier(flags)`.
    pub fn threadgroupBarrier(self: *Builder, flags: MemFlags) void {
        _ = air.wg_barrier(self.ctx, @bitCast(flags), self.loc()).appendTo(self.currentBlock());
    }

    /// `simdgroup_barrier(flags)`.
    pub fn simdgroupBarrier(self: *Builder, flags: MemFlags) void {
        _ = air.simdgroup_barrier(self.ctx, @bitCast(flags), self.loc()).appendTo(self.currentBlock());
    }

    pub const MatrixOpts = struct {
        /// `matrix_origin`.
        origin: [2]i64 = .{ 0, 0 },
        /// `transpose_matrix`.
        transpose: bool = false,
    };

    fn matrixType(self: *Builder, scalar: Scalar) *const mlir.Type {
        return .vector(&.{64}, scalar.mlirType(self.ctx));
    }

    fn matrixScalar(m: Value) Scalar {
        return switch (m.kind) {
            .matrix => |s| s,
            else => std.debug.panic("dsl/metal: expected a simdgroup_matrix, got a {t}", .{m.kind}),
        };
    }

    /// The extents, strides and origin a matrix load or store passes the
    /// builtin, from its `elements_per_row`, `matrix_origin` and
    /// `transpose_matrix`.
    fn accessLayout(self: *Builder, elements_per_row: anytype, opts: MatrixOpts) [3]*const mlir.Value {
        const ulong2: Type = .vec(.ulong, 2);
        const epr = self.convert(self.lift(elements_per_row), .ulong);
        const eight = self.constant(.ulong, 8);
        const one = self.constant(.ulong, 1);
        const o0 = self.constant(.ulong, opts.origin[0]);
        const o1 = self.constant(.ulong, opts.origin[1]);
        const layout: [3]Value = if (opts.transpose)
            .{ self.vec(ulong2, .{ eight, epr }), self.vec(ulong2, .{ epr, one }), self.vec(ulong2, .{ o1, o0 }) }
        else
            .{ self.vec(ulong2, .{ epr, eight }), self.vec(ulong2, .{ one, epr }), self.vec(ulong2, .{ o0, o1 }) };
        return .{ layout[0].inner, layout[1].inner, layout[2].inner };
    }

    /// `simdgroup_load(m, src, elements_per_row, matrix_origin,
    /// transpose_matrix)`: the matrix `m` it loads.
    pub fn simdgroupLoad(self: *Builder, src_: Value, elements_per_row: anytype, opts: MatrixOpts) Value {
        const src = src_.rvalue();
        const t = src.pointer().pointee.scalar;
        const layout = self.accessLayout(elements_per_row, opts);
        return self.emit(air.simdgroup_matrix_8x8_load(self.ctx, src.inner, layout[0], layout[1], layout[2], self.matrixType(t), self.fastmath(), self.loc()), .{ .matrix = t });
    }

    /// `simdgroup_store(m, dst, elements_per_row, matrix_origin,
    /// transpose_matrix)`.
    pub fn simdgroupStore(self: *Builder, m: Value, dst_: Value, elements_per_row: anytype, opts: MatrixOpts) void {
        const t = matrixScalar(m);
        const dst = dst_.rvalue();
        const p = dst.pointer();
        if (p.pointee.scalar != t) std.debug.panic("dsl/metal: simdgroup_store to a {s} pointer", .{p.pointee.name()});
        const layout = self.accessLayout(elements_per_row, opts);
        _ = air.simdgroup_matrix_8x8_store(self.ctx, m.inner, dst.inner, layout[0], layout[1], layout[2], self.loc()).appendTo(self.currentBlock());
    }

    /// `simdgroup_multiply_accumulate(d, a, b, c)`: the matrix `d = a * b + c`,
    /// of `c`'s type.
    pub fn simdgroupMultiplyAccumulate(self: *Builder, a: Value, b: Value, c: Value) Value {
        _ = .{ matrixScalar(a), matrixScalar(b) };
        const t = matrixScalar(c);
        return self.emit(air.simdgroup_matrix_8x8_multiply_accumulate(self.ctx, a.inner, b.inner, c.inner, self.matrixType(t), self.fastmath(), self.loc()), .{ .matrix = t });
    }

    /// The `llvm.loop` metadata of an MSL loop: C++ loops must make progress.
    fn loopAnnotation(self: *Builder, opts: LoopOpts) *const llvm.LoopAnnotationAttr {
        const unroll: ?*const llvm.LoopUnrollAttr = if (opts.unroll) |u| switch (u) {
            .full => llvm.LoopUnrollAttr.get(self.ctx, .{ .full = true }),
            .enable => llvm.LoopUnrollAttr.get(self.ctx, .{ .disable = false }),
        } else null;
        return llvm.LoopAnnotationAttr.get(self.ctx, .{ .unroll = unroll, .mustProgress = true });
    }

    /// The type `i < upper` compares in, for a loop variable `i` of type `t`: the
    /// loop runs in `t`, which a short or char variable may be narrower than.
    fn loopComparison(t: Type, upper: Type) Scalar {
        const compared = types.commonScalar(t.scalar, upper.scalar);
        const narrow = t.scalar.bits() < 32 and compared == .int;
        if (t.scalar.isFloat() or t.isVector() or (compared.bits() != t.scalar.bits() and !narrow))
            std.debug.panic("dsl/metal: a {s} loop variable compared to a {s}", .{ t.name(), upper.name() });
        return compared;
    }

    /// `for (T i = lower; i < upper; i += step)`, with `T` the type of
    /// `lower`; `close` ends the body.
    pub fn openFor(self: *Builder, lower: anytype, upper: anytype, step: anytype) ForScope {
        return self.openForOpts(lower, upper, step, .{});
    }

    /// `openFor` under a loop pragma.
    pub fn openForOpts(self: *Builder, lower: anytype, upper: anytype, step: anytype, opts: LoopOpts) ForScope {
        const i = self.local(if (@TypeOf(lower) == Value) lower.type_() else literalType(lower), lower);
        var loop = self.openLoop(opts);
        _ = loopComparison(i.type_(), if (@TypeOf(upper) == Value) upper.type_() else literalType(upper));
        loop.body(i.lt(upper));
        const step_: ForScope.Step = switch (@typeInfo(@TypeOf(step))) {
            .comptime_int, .int => if (step == 1) .one else .{ .int = step },
            else => .{ .value = step },
        };
        return .{ .loop = loop, .iv = i, .step = step_ };
    }

    /// A loop: `while (c)`, `for (init; c; inc)` or `for (;;)`.
    pub fn openLoop(self: *Builder, opts: LoopOpts) LoopScope {
        const cond_block = mlir.Block.init(&.{}, &.{});
        _ = self.branch(cond_block);
        self.startBlock(cond_block);
        return .{ .kernel = self, .cond_block = cond_block, .end_block = mlir.Block.init(&.{}, &.{}), .loop_annotation = self.loopAnnotation(opts) };
    }

    /// `if (cond) { ... }`.
    pub fn openIf(self: *Builder, cond: Value) IfScope {
        var c = self.condition();
        return c.openIf(cond);
    }

    /// `if (cond) { ... } else { ... }` with no result types; with one, `T`,
    /// `cond ? a : b` of type `T` evaluating one operand.
    pub fn openIfElse(self: *Builder, cond: Value, result_types: anytype) IfElseScope(common.arity(@TypeOf(result_types), "openIfElse: result_types")) {
        if (comptime common.arity(@TypeOf(result_types), "openIfElse: result_types") == 0) {
            var c = self.condition();
            return c.openIfElse(cond, .{});
        }
        return self.openTernary(cond, typeOf(result_types[0]));
    }

    /// A condition of `&&`, `||` and `!` terms.
    pub fn condition(self: *Builder) Condition {
        return .{ .kernel = self, .true_block = mlir.Block.init(&.{}, &.{}), .false_block = mlir.Block.init(&.{}, &.{}) };
    }

    /// `a && b && ...` as a bool.
    pub fn openAnd(self: *Builder) Logical {
        return self.logicalChain(.@"and");
    }

    /// `a || b || ...` as a bool.
    pub fn openOr(self: *Builder) Logical {
        return self.logicalChain(.@"or");
    }

    fn logicalChain(self: *Builder, op: @FieldType(Logical, "op")) Logical {
        var loc_ = [1]*const mlir.Location{self.loc()};
        return .{ .kernel = self, .op = op, .end_block = mlir.Block.init(&.{Type.bool.mlirType(self.ctx)}, &loc_) };
    }

    /// `return;`.
    pub fn return_(self: *Builder) void {
        const block = self.insert orelse return;
        _ = func.returns(self.ctx, &.{}, self.loc()).appendTo(block);
        self.insert = null;
    }

    /// `if (cond) return;`.
    pub fn returnIf(self: *Builder, cond: Value) void {
        var scope = self.openIf(cond);
        {
            self.return_();
            scope.close();
        }
    }

    /// Terminates the kernel and returns its module, verified.
    pub fn finish(self: *Builder) FinishError![:0]const u8 {
        self.return_();
        if (self.fast_math) air.enableFastMath(self.ctx, self.module.operation());
        try common.verify(self.module, "dsl/metal");
        var writer: std.Io.Writer.Allocating = .init(self.allocator);
        defer writer.deinit();
        try writer.writer.print("{f}", .{self.module.operation()});
        return try self.allocator.dupeSentinel(u8, writer.written(), 0);
    }
};

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registerDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectContains(ir: []const u8, needles: []const []const u8) !void {
    for (needles) |needle| {
        if (std.mem.find(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestExpectedEqual;
        }
    }
}

test "arguments and C++ arithmetic" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .buffer = .{ .type = .bfloat, .const_ = true, .restrict = true } },
        .y = .{ .buffer = .{ .type = .float4 } },
        .n = .{ .constant = .int },
        .shmem = .{ .threadgroup = .half },
        .gid = .{ .threadgroup_position_in_grid = .uint3 },
        .lane = .{ .thread_index_in_simdgroup = .ushort },
    });
    // `gid.z / n` divides unsigned, `lane * 2` promotes to int.
    const q = a.gid.z().div(a.n);
    try std.testing.expect(q.type_().eql(.uint));
    const twice = a.lane.mul(2);
    try std.testing.expect(twice.type_().eql(.int));
    const v = a.x.load(q).to(.float).mul(0.5);
    try std.testing.expect(v.type_().eql(.float));
    a.y.store(twice, b.vec(.float4, .{v}));
    a.shmem.store(a.lane, v);
    try std.testing.expectEqual(4, b.arguments.items.len);
    try std.testing.expectEqual(1, b.arguments.items[1].index);
    try std.testing.expectEqual(4, b.arguments.items[2].size);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{ "air.fast_math_enable", "air.kernel", "noalias = true" });
    try expectContains(ir, &.{ "arith.divui", "overflow<nsw>", "arith.extf", "arith.mulf", "fastmath<fast>", "arith.truncf", "llvm.getelementptr", "#air.builtin<threadgroup_position_in_grid>", "air.arg_name = \"lane\"" });
}

test "conversions follow Apple's compiler" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .h = .{ .buffer = .{ .type = .half4, .const_ = true } },
        .o = .{ .buffer = .{ .type = .bfloat4 } },
        .i = .{ .buffer = .{ .type = .int } },
        .tid = .{ .thread_position_in_grid = .uint },
    });
    const f4 = a.h.load(a.tid).to(.float4);
    a.o.store(a.tid, f4.to(.bfloat4));
    const n = a.i.load(a.tid);
    a.i.store(0, n.to(.float).add(n.to(.half).to(.float)).to(.uint).to(.short));
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{ "air.convert", ": vector<4xf16> to vector<4xf32>", ": vector<4xf32> to vector<4xbf16>", "air.convert.s", ": i32 to f32", ": i32 to f16", "arith.extf", "air.convert.u", "arith.trunci" });
}

test "math, SIMD-group and barrier functions" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .buffer = .{ .type = .float } },
        .v = .{ .buffer = .{ .type = .float4 } },
        .lane = .{ .thread_index_in_simdgroup = .ushort },
    });
    const x = a.x.load(a.lane);
    const v = a.v.load(a.lane);
    const m = b.simdMax(b.max(x, b.fast().exp2(x)));
    const s = b.simdSum(b.exp(x.sub(m)));
    const t = b.precise().tanh(b.fma(s, 2.0, b.dot(v, v)));
    const r = b.simdShuffleXor(t, 8);
    b.simdgroupBarrier(.mem_threadgroup);
    b.threadgroupBarrier(.{ .device = true, .threadgroup = true });
    a.x.store(a.lane, b.ternary(r.gt(0), b.min(a.lane.to(.int), 3).to(.float), r));
    a.x.store(1, b.ternary(r.gt(1), 1.0, 2.0));
    a.v.store(a.lane, b.simdSum(v));
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{ "air.fast_exp2", "air.fast_fmax", "air.simd_max", "air.fast_exp", "air.simd_sum", "air.dot", "air.fma", "air.tanh", "air.simd_shuffle_xor", "air.simdgroup.barrier <threadgroup>", "air.wg.barrier <device|threadgroup>", "air.min.s", "arith.select", ": vector<4xf32>" });
    try std.testing.expect(std.mem.find(u8, ir, "cf.cond_br") == null);
}

test "loops, ifs and early returns" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .buffer = .{ .type = .float } },
        .n = .{ .constant = .int },
        .tid = .{ .thread_position_in_grid = .uint },
    });
    var ret = b.openIf(a.tid.ge(1024));
    {
        a.x.store(0, 0.0);
        b.return_();
        ret.close();
    }
    b.returnIf(a.n.le(0));

    // float acc = 0; for (int i = 0; i < n; ++i) { if (x[i] < 0) break; acc += x[i]; }
    const acc = b.local(.float, 0.0);
    var loop = b.openFor(0, a.n, 1);
    {
        var neg = b.openIf(a.x.load(loop.iv).lt(0.0));
        {
            loop.breakLoop();
            neg.close();
        }
        acc.assign(acc.add(a.x.load(loop.iv)));
        loop.close();
    }

    // int lo = 0, hi = n; while (lo < hi) { int mid = (lo + hi + 1) / 2; ... }
    const lo = b.local(.int, 0);
    const hi = b.local(.int, a.n);
    var search = b.openLoop(.{});
    {
        search.body(lo.lt(hi));
    }
    {
        const mid = b.local(.int, lo.add(hi).add(1).div(2));
        var up = b.openIfElse(a.x.load(mid).le(acc), .{});
        {
            lo.assign(mid);
            up.elseBranch();
        }
        {
            hi.assign(mid.sub(1));
            up.close();
        }
        search.close();
    }

    // bool m = lo > 0 || (n > 3 && acc < 1);
    var m = b.openOr();
    m.term(lo.gt(0));
    var c = b.condition();
    c.and_(a.n.gt(3));
    var only = c.openIfNot(m.close(acc.lt(1.0)));
    {
        a.x.store(1, b.ternary(lo.gt(0), acc, 2.0));
        only.close();
    }

    const scratch = b.array(.float, 8);
    scratch.store(a.tid.rem(8), acc);
    a.x.store(2, scratch.load(3));
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{ "cf.cond_br", "llvm.br", "loop_annotation", "mustProgress = true", "llvm.alloca", "!llvm.array<8 x f32>", "arith.remui" });
    try std.testing.expect(std.mem.find(u8, ir, "scf.") == null);
    // The two early returns and the last one, each where it is.
    try std.testing.expectEqual(3, std.mem.count(u8, ir, "\n    return\n"));
    // Every local is allocated in the entry block, before any branch.
    try std.testing.expect(std.mem.findLast(u8, ir, "llvm.alloca").? < std.mem.find(u8, ir, "cf.cond_br").?);
}

test "a return inside a loop and in both branches of an if" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .buffer = .{ .type = .float } },
        .n = .{ .constant = .int },
    });
    // for (;;) { if (n <= 0) break; if (x[0] < 0) return; x[0] -= 1; }
    var forever = b.openLoop(.{});
    {
        var stop = b.openIf(a.n.le(0));
        {
            forever.breakLoop();
            stop.close();
        }
        b.returnIf(a.x.load(0).lt(0.0));
        a.x.store(0, a.x.load(0).sub(1.0));
        forever.close();
    }
    // if (n > 1) { x[1] = 1; return; } else { x[1] = 2; return; }
    var both = b.openIfElse(a.n.gt(1), .{});
    {
        a.x.store(1, 1.0);
        b.return_();
        both.elseBranch();
    }
    {
        a.x.store(1, 2.0);
        b.return_();
        both.close();
    }
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    // The loop's return and one per branch; `finish` adds none after them.
    try std.testing.expectEqual(3, std.mem.count(u8, ir, "\n    return\n"));
    try expectContains(ir, &.{"loop_annotation"});
}

test "a break and the end of the body meet in the latch" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .buffer = .{ .type = .float } },
        .n = .{ .constant = .int },
    });
    // for (int i = 0; i < n; ++i) { if (x[i] < 0) break; x[i] = 1; }
    var loop = b.openFor(0, a.n, 1);
    {
        var neg = b.openIf(a.x.load(loop.iv).lt(0.0));
        {
            loop.breakLoop();
            neg.close();
        }
        a.x.store(loop.iv, 1.0);
        loop.close();
    }
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    // `i`, and the flag the break and the end of the body set for the latch.
    try std.testing.expectEqual(2, std.mem.count(u8, ir, "llvm.alloca"));
    // `i < n`, `x[i] < 0`, and the latch's choice between leaving and `++i`.
    try std.testing.expectEqual(3, std.mem.count(u8, ir, "cf.cond_br"));
}

test "simdgroup matrices" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .o = .{ .buffer = .{ .type = .float2 } },
        .s = .{ .threadgroup = .bfloat },
        .f = .{ .constant = .int },
        .lane = .{ .thread_index_in_simdgroup = .uint },
    });
    const k = b.simdgroupLoad(a.s.add(8), 40, .{ .transpose = true });
    const q = b.simdgroupLoad(a.s, a.f, .{});
    const acc = b.simdgroupMultiplyAccumulate(q, k, b.matrix(.float, a.o.load(a.lane)));
    a.o.store(a.lane, acc.threadElements());
    b.simdgroupStore(acc, a.s.to(.float), 8, .{});
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{ "air.simdgroup_matrix_8x8_load", "-> vector<64xbf16>", "air.simdgroup_matrix_8x8_multiply_accumulate", "vector<64xbf16>, vector<64xbf16>, vector<64xf32> -> vector<64xf32>", "air.simdgroup_matrix_8x8_store" });
    // `b.matrix` widens two elements to a matrix, `threadElements` takes them back.
    try std.testing.expectEqual(2, std.mem.count(u8, ir, "llvm.shufflevector"));
    try expectContains(ir, &.{"[0, 1] : vector<64xf32>"});
    try std.testing.expect(std.mem.find(u8, ir, "llvm.alloca") == null);
}

test "constant struct arguments" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{ .fast_math = false });
    defer b.deinit();
    const Args: Struct = .{ .name = "args_t", .fields = &.{
        .{ .name = "num_heads", .type = .int },
        .{ .name = "scale", .type = .float },
    } };
    const a = try b.declareArgs(.{
        .args = .{ .constant_struct = Args },
        .o = .{ .buffer = .{ .type = .float } },
    });
    a.o.store(0, a.args.field("scale").mul(a.args.field("num_heads")));
    try std.testing.expectEqual(8, b.arguments.items[0].size);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{ "<offset = 4, size = 4, typeName = \"float\", name = \"scale\">", "typeName = \"args_t\"", "!llvm.struct<\"struct.args_t\", (i32, f32)>" });
    try std.testing.expect(std.mem.find(u8, ir, "fastmath") == null);
}

fn expectCount(ir: []const u8, needle: []const u8, expected: usize) !void {
    const n = std.mem.count(u8, ir, needle);
    if (n != expected) {
        std.debug.print("`{s}` {d} times, not {d}, in:\n{s}\n", .{ needle, n, expected, ir });
        return error.TestExpectedEqual;
    }
}

test "subscripts index with the index's type, swizzles with an int" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .v = .{ .buffer = .{ .type = .float4 } },
        .o = .{ .buffer = .{ .type = .float } },
        .n = .{ .constant = .long },
        .gid = .{ .threadgroup_position_in_grid = .uint3 },
        .lane = .{ .thread_index_in_simdgroup = .ushort },
    });
    // float4 x = v[0];
    // o[gid.z] = x.x + x.y + x.z + x.w + x[2] + x[lane] + x[(char) lane] + x[n];
    const x = b.local(.float4, a.v.load(0));
    const sum = x.x().add(x.y()).add(x.z()).add(x.w()).add(x.at(2)).add(x.at(a.lane)).add(x.at(a.lane.to(.char))).add(x.at(a.n));
    a.o.store(a.gid.z(), sum);
    // x[lane] = 1.0f; x[2] = 2.0f; x.y = 3.0f; v[0] = x;
    x.assign(x.insert(a.lane, 1.0));
    x.assign(x.insert(2, 2.0));
    x.assign(x.insert(1, 3.0));
    a.v.store(0, x);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    // `i32` for the swizzles and literals, `i16 %lane`, `i8`, `i64 %n`.
    try expectCount(ir, ": i32] : vector<4xf32>", 7);
    try expectCount(ir, ": i16] : vector<4xf32>", 2);
    try expectCount(ir, ": i8] : vector<4xf32>", 1);
    try expectCount(ir, ": i64] : vector<4xf32>", 1);
    try expectCount(ir, ": i32] : vector<3xi32>", 1);
}

test "a pointer local" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .o = .{ .buffer = .{ .type = .float } },
        .lane = .{ .thread_index_in_simdgroup = .ushort },
    });
    // device float *p = o + 4; p += lane; *p = 1.0f;
    const p = b.localPointer(a.o.add(4));
    p.assign(p.add(a.lane));
    p.store(0, 1.0);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectCount(ir, "llvm.alloca", 1);
    try expectCount(ir, "llvm.getelementptr", 2);
}

test "integer max, min, abs and clamp read locals and constant arguments" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .o = .{ .buffer = .{ .type = .int } },
        .w = .{ .buffer = .{ .type = .uint } },
        .f = .{ .buffer = .{ .type = .float } },
        .n = .{ .constant = .int },
        .u = .{ .constant = .uint },
    });
    // int m = n; o[0] = max(m, n) + min(n, m) + min(m, 7) + abs(m) + clamp(m, 0, 7);
    const m = b.local(.int, a.n);
    a.o.store(0, b.max(m, a.n).add(b.min(a.n, m)).add(b.min(m, 7)).add(b.abs(m)).add(b.clamp(m, 0, 7)));
    // w[0] = max(u, 3u) + min(3u, u) + abs(u) + clamp(u, 1u, 2u);
    a.w.store(0, b.max(a.u, 3).add(b.min(3, a.u)).add(b.abs(a.u)).add(b.clamp(a.u, 1, 2)));
    // const float x = f[0]; f[1] = clamp(x, 0.0f, 1.0f) + fabs(x) + max(x, x);
    const x = a.f.load(0);
    a.f.store(1, b.clamp(x, 0.0, 1.0).add(b.abs(x)).add(b.max(x, x)));
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{ "air.max.s", "air.min.s", "air.abs.s", "air.clamp.s", "air.max.u", "air.min.u", "air.clamp.u", "air.fast_clamp", "air.fast_fabs", "air.fast_fmax" });
    // abs of a uint is the value itself.
    try std.testing.expect(std.mem.find(u8, ir, "air.abs.u") == null);
}

test "bitwise operators and shifts" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .o = .{ .buffer = .{ .type = .int } },
        .w = .{ .buffer = .{ .type = .uint } },
        .l = .{ .buffer = .{ .type = .long } },
        .n = .{ .constant = .int },
        .lane = .{ .thread_index_in_simdgroup = .ushort },
    });
    // int s = n; o[0] = (n << s) ^ (~(n | 3) & -s); w[0] = w[1] >> s; o[1] = lane >> 1;
    // l[0] = l[1] << n;
    const s = b.local(.int, a.n);
    a.o.store(0, a.n.shl(s).bitXor(a.n.bitOr(3).bitNot().bitAnd(s.neg())));
    a.w.store(0, a.w.load(1).shr(s));
    a.o.store(1, a.lane.shr(1));
    a.l.store(0, a.l.load(1).shl(a.n));
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{ "arith.shli", "arith.shrui", "arith.shrsi", "arith.xori", "arith.ori", "arith.subi" });
    // clang masks an amount to the width (`and i32 %s, 31`), after casting it
    // as unsigned (`zext i32 %n to i64`, `and i64 ..., 63`); a number's mask
    // folds away.
    try expectCount(ir, "arith.andi", 4);
    try expectContains(ir, &.{ "arith.constant 31 : i32", "arith.constant 63 : i64" });
    var lines = std.mem.splitScalar(u8, ir, '\n');
    const unsigned_cast = while (lines.next()) |line| {
        if (std.mem.find(u8, line, "arith.extui") != null and std.mem.find(u8, line, "i32 to i64") != null) break true;
    } else false;
    try std.testing.expect(unsigned_cast);
}

test "a shift reads its amount once" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .o = .{ .buffer = .{ .type = .int } },
        .n = .{ .constant = .int },
    });
    // int s = n; *o = s << s;
    const s = b.local(.int, a.n);
    a.o.store(0, s.shl(s));
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    // n, and s twice.
    try expectCount(ir, "llvm.load", 3);
}

test "math functions" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .buffer = .{ .type = .float } },
        .h = .{ .buffer = .{ .type = .half } },
    });
    const v = a.x.load(0);
    var sum = b.local(.float, 0.0);
    for ([_]Value{
        b.exp2(v),  b.log(v),  b.log2(v),     b.tanh(v),      b.sqrt(v),
        b.rsqrt(v), b.ceil(v), b.floor(v),    b.round(v),     b.trunc(v),
        b.sin(v),   b.cos(v),  b.pow(v, 2.0), b.fmod(v, 3.0),
    }) |term| sum.assign(sum.add(term));
    const fast_ = b.fast();
    for ([_]Value{
        fast_.exp(v), fast_.log(v), fast_.log2(v), fast_.sqrt(v),     fast_.rsqrt(v),
        fast_.sin(v), fast_.cos(v), fast_.tanh(v), fast_.max(v, 1.0), fast_.min(v, 0.5),
    }) |term| sum.assign(sum.add(term));
    const precise_ = b.precise();
    for ([_]Value{
        precise_.exp(v),   precise_.exp2(v), precise_.log(v), precise_.log2(v),     precise_.sqrt(v),
        precise_.rsqrt(v), precise_.sin(v),  precise_.cos(v), precise_.max(v, 1.0), precise_.min(v, 0.5),
    }) |term| sum.assign(sum.add(term));
    a.x.store(1, sum);
    // half has no fast:: functions: exp of a half is precise.
    a.h.store(0, b.exp(a.h.load(1)));
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{
        "air.fast_exp2", "air.fast_log",   "air.fast_log2",  "air.fast_tanh",  "air.fast_sqrt", "air.fast_rsqrt",
        "air.fast_ceil", "air.fast_floor", "air.fast_round", "air.fast_trunc", "air.fast_sin",  "air.fast_cos",
        "air.fast_pow",  "air.fast_fmod",  "air.fast_exp",   "air.fast_fmax",  "air.fast_fmin",
    });
    try expectContains(ir, &.{ "air.exp ", "air.exp2 ", "air.log ", "air.log2 ", "air.sqrt ", "air.rsqrt ", "air.sin ", "air.cos ", "air.fmax ", "air.fmin " });
    try expectContains(ir, &.{"f16"});
}

test "SIMD-group functions" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .buffer = .{ .type = .float } },
        .i = .{ .buffer = .{ .type = .int } },
        .u = .{ .buffer = .{ .type = .uint } },
        .lane = .{ .thread_index_in_simdgroup = .ushort },
    });
    const f = a.x.load(a.lane);
    const n = a.i.load(a.lane);
    const m = a.u.load(a.lane);
    a.x.store(0, b.simdMin(f).add(b.simdPrefixExclusiveSum(f)).add(b.simdShuffle(f, a.lane)).add(b.simdShuffleDown(f, 1)).add(b.simdBroadcast(f, 0)));
    a.i.store(0, b.simdMin(n).add(b.simdShuffle(n, 3)).add(b.simdShuffleDown(n, a.lane)).add(b.simdBroadcast(n, 0)));
    a.u.store(0, b.simdMin(m).add(b.simdShuffleXor(m, 1)));
    a.i.store(1, b.ternary(b.simdAny(f.gt(0.0)), 1, 0).add(b.ternary(b.simdAll(n.ne(0)), 2, 0)));
    b.threadgroupBarrier(.mem_device);
    b.simdgroupBarrier(.mem_texture);
    b.threadgroupBarrier(.mem_none);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{
        "air.simd_min ",       "air.simd_min.s",       "air.simd_min.u",         "air.simd_prefix_exclusive_sum",
        "air.simd_shuffle ",   "air.simd_shuffle.s",   "air.simd_shuffle_down ", "air.simd_shuffle_down.s",
        "air.simd_broadcast ", "air.simd_broadcast.s", "air.simd_shuffle_xor.u", "air.simd_any",
        "air.simd_all",        "llvm.freeze",
    });
    try expectContains(ir, &.{ "air.wg.barrier <device>", "air.simdgroup.barrier <texture>", "air.wg.barrier <none>" });
}

test "pragmas, conditions and casts" {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = Builder.open(std.testing.allocator, ctx, "k", .{});
    defer b.deinit();
    const a = try b.declareArgs(.{
        .o = .{ .buffer = .{ .type = .float } },
        .n = .{ .constant = .int },
        .lane = .{ .thread_index_in_simdgroup = .ushort },
    });

    const r = a.o.load(a.lane).mul(2.0);
    // float acc[4] = {};
    const acc = b.zeroedArray(.float, 4);
    // #pragma clang loop unroll(full)
    var full = b.openForOpts(0, 4, 1, .{ .unroll = .full });
    {
        acc.store(full.iv, r);
        full.close();
    }
    // #pragma unroll
    var enabled = b.openForOpts(0, a.n, 2, .{ .unroll = .enable });
    {
        acc.store(0, acc.load(0).add(enabled.iv.to(.float)));
        enabled.close();
    }
    // for (int i = 0; i < 4; i += 1) {}
    const i = b.local(.int, 0);
    var counted = b.openLoop(.{});
    {
        counted.body(i.lt(4));
    }
    {
        counted.next();
        i.assign(i.add(1));
        counted.close();
    }
    // const uint bits = as_type<uint>(acc[0]);
    const bits = acc.load(0).asType(.uint);
    // float t = r; *&t = 2.0f;
    const t = b.local(.float, r);
    t.address().store(0, 2.0);
    // bool ok = n > 0 && bits != 0;
    var both = b.openAnd();
    both.term(a.n.gt(0));
    const ok = both.close(bits.ne(0));
    // if (!(n > 1) || ok) { ... } else { ... }
    var c1 = b.condition();
    c1.orNot(a.n.gt(1));
    var s1 = c1.openIfElse(ok, .{});
    {
        a.o.store(1, 1.0);
        s1.elseBranch();
    }
    {
        a.o.store(1, 2.0);
        s1.close();
    }
    // if (!ok && !(n == 3)) { ... } else { ... }
    var c2 = b.condition();
    c2.andNot(ok);
    var s2 = c2.openIfElseNot(a.n.eq(3), .{});
    {
        a.o.store(2, 1.0);
        s2.elseBranch();
    }
    {
        s2.close();
    }
    // o[lane] = n > 2 ? fast::exp(t) : t;
    var choice = b.openIfElse(a.n.gt(2), .{.float});
    {
        choice.yieldThen(.{b.fast().exp(t)});
    }
    {
        choice.yieldElse(.{t});
    }
    a.o.store(a.lane, choice.results[0]);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, &.{ "llvm.memset.p0.i64", "full = true", "disable = false", "llvm.bitcast" });
}
