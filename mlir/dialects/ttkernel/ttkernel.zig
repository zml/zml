//! tt-lang's `ttkernel` and `ttcore` dialects: the Tenstorrent kernel IR
//! (circular buffers, NoC transfers, tile math on the DST registers) and the
//! tile types it moves, plus their translation to tt-metal C++.

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const ops = @import("ops.zig");
pub const ttcore = @import("ttcore.zig");

/// Upstream dialects a TTKernel module uses besides `ttcore`/`ttkernel`
/// (`emitc` once converted).
pub const dialects_needed = [_][]const u8{ "func", "arith", "scf", "memref", "emitc", "ttcore", "ttkernel" };

pub fn insertDialects(registry: *mlir.DialectRegistry) void {
    c.mlirTTLangRegisterUpstreamDialects(registry.ptr());
    inline for (.{ "ttcore", "ttkernel", "ttl" }) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }
}

/// Registers tt-lang's passes and pipelines (`ttl-*`, `ttkernel-*`,
/// `convert-ttkernel-to-emitc`); idempotent.
pub fn registerPasses() void {
    c.mlirTTLangRegisterPasses();
}

/// `Operation.make` that names the op and its operand types on failure.
/// Operations are not verified on creation: TTKernel operations check their
/// enclosing kernel function, which a detached operation lacks; verify the
/// finished module instead.
pub fn make(ctx: *mlir.Context, name: []const u8, args: mlir.Operation.MakeArgs) *mlir.Operation {
    var unverified = args;
    unverified.verify = false;
    return mlir.Operation.try_make(ctx, name, unverified) catch |err| {
        var buf: [4096]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        if (args.operands) |operands| switch (operands) {
            .flat => |vs| for (vs, 0..) |v, i| {
                w.print("\n  operand {d}: {f}", .{ i, v.type_() }) catch break;
            },
            .variadic => |segments| for (segments) |seg| for (seg) |v| {
                w.print("\n  operand: {f}", .{v.type_()}) catch break;
            },
        };
        if (args.results) |results| switch (results) {
            .flat => |ts| for (ts, 0..) |t, i| {
                w.print("\n  result {d}: {f}", .{ i, t }) catch break;
            },
            .variadic => {},
        };
        std.debug.panic("{s}: cannot create operation ({}){s}", .{ name, err, w.buffered() });
    };
}

/// A type from its textual form, for the types without a C API (the
/// operation builders' fixed result types).
pub fn parseType(ctx: *mlir.Context, comptime text: []const u8) *const mlir.Type {
    return mlir.Type.parse(ctx, text) catch std.debug.panic("cannot parse type `{s}`", .{text});
}

// =============================================================================
// Enums
// =============================================================================

/// `ttkernel::ThreadType`.
pub const ThreadType = enum(u32) {
    noc = 0,
    compute = 1,
    ethernet = 2,
};

/// `ttkernel::ReduceType`.
pub const ReduceType = enum(u32) {
    sum = 0,
    max = 1,
    avg = 2,
};

/// `ttkernel::ReduceDim`.
pub const ReduceDim = enum(u32) {
    row = 0,
    col = 1,
    scalar = 2,
};

/// `ttkernel::BcastType`.
pub const BcastType = enum(u32) {
    none = 0,
    col = 1,
    row = 2,
    scalar = 3,
};

/// `ttkernel::ArgType`.
pub const ArgType = enum(u32) {
    cb_port = 0,
    buffer_address = 1,
    local_semaphore = 2,
    named_argument = 3,
    global_semaphore = 4,
    scalar = 5,
};

// =============================================================================
// Attributes
// =============================================================================

/// `#ttkernel.thread<...>`, the `ttkernel.thread` attribute of a kernel function.
pub fn threadAttr(ctx: *mlir.Context, thread: ThreadType) *const mlir.Attribute {
    return @ptrCast(c.ttlangTTKernelThreadTypeAttrGet(ctx.ptr(), @intFromEnum(thread)).ptr);
}

pub fn reduceTypeAttr(ctx: *mlir.Context, value: ReduceType) *const mlir.Attribute {
    return @ptrCast(c.ttlangTTKernelReduceTypeAttrGet(ctx.ptr(), @intFromEnum(value)).ptr);
}

pub fn reduceDimAttr(ctx: *mlir.Context, value: ReduceDim) *const mlir.Attribute {
    return @ptrCast(c.ttlangTTKernelReduceDimAttrGet(ctx.ptr(), @intFromEnum(value)).ptr);
}

pub fn bcastTypeAttr(ctx: *mlir.Context, value: BcastType) *const mlir.Attribute {
    return @ptrCast(c.mlirTTKernelBcastTypeAttrGet(ctx.ptr(), @intFromEnum(value)).ptr);
}

/// `#ttkernel.arg<...>`, one entry of an `ArgSpec`.
pub fn argAttr(ctx: *mlir.Context, arg_type: ArgType, operand_index: usize, is_uniform: bool) *const mlir.Attribute {
    return @ptrCast(c.ttlangTTKernelArgAttrGet(ctx.ptr(), @intFromEnum(arg_type), operand_index, is_uniform).ptr);
}

/// `#ttkernel.arg_spec<rt_args = [...] ct_args = [...]>`, the
/// `ttkernel.arg_spec` attribute of a kernel function.
pub fn argSpecAttr(ctx: *mlir.Context, rt_args: []const *const mlir.Attribute, ct_args: []const *const mlir.Attribute) *const mlir.Attribute {
    return @ptrCast(c.ttlangTTKernelArgSpecAttrGet(
        ctx.ptr(),
        @ptrCast(@constCast(rt_args.ptr)),
        rt_args.len,
        @ptrCast(@constCast(ct_args.ptr)),
        ct_args.len,
    ).ptr);
}

// =============================================================================
// Types
// =============================================================================

/// `!ttkernel.cb<num_elements, element_type>`: a circular buffer of
/// `num_elements` pages of `element_type` (a `!ttcore.tile`).
pub const CBType = opaque {
    const M = mlir.Methods(CBType, c.MlirType);

    pub const isAFn = c.mlirTypeIsATTKernelCB;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, num_elements: i64, element_type: *const mlir.Type) *const CBType {
        return @ptrCast(c.mlirTTKernelCBTypeGet(ctx.ptr(), num_elements, element_type.ptr()).ptr);
    }

    pub fn type_(self: *const CBType) *const mlir.Type {
        return @ptrCast(self);
    }

    pub fn numElements(self: *const CBType) i64 {
        return c.mlirTTKernelCBTypeGetNumElements(self.ptr());
    }

    pub fn elementType(self: *const CBType) *const mlir.Type {
        return @ptrCast(c.mlirTTKernelCBTypeGetElementType(self.ptr()).ptr);
    }
};

/// The parameterless `ttkernel` types.
pub const Type = enum {
    /// `!ttkernel.noc_addr`, a 64-bit NoC address.
    noc_addr,
    /// `!ttkernel.l1_addr`.
    l1_addr,
    /// `!ttkernel.l1_addr_ptr`.
    l1_addr_ptr,
    /// `!ttkernel.local_semaphore`.
    local_semaphore,
    /// `!ttkernel.DataFormat`.
    data_format,
    /// `!ttkernel.TensorAccessorArgs`.
    tensor_accessor_args,
    /// `!ttkernel.TensorAccessor`.
    tensor_accessor,
    /// `!ttkernel.LocalTensorAccessor`.
    local_tensor_accessor,
    /// `!ttkernel.TensorAccessorPageMapping`.
    tensor_accessor_page_mapping,

    pub fn get(self: Type, ctx: *mlir.Context) *const mlir.Type {
        const t = switch (self) {
            .noc_addr => c.ttlangTTKernelNocAddrTypeGet(ctx.ptr()),
            .l1_addr => c.ttlangTTKernelL1AddrTypeGet(ctx.ptr()),
            .l1_addr_ptr => c.ttlangTTKernelL1AddrPtrTypeGet(ctx.ptr()),
            .local_semaphore => c.ttlangTTKernelLocalSemaphoreTypeGet(ctx.ptr()),
            .data_format => c.ttlangTTKernelDataFormatTypeGet(ctx.ptr()),
            .tensor_accessor_args => c.ttlangTTKernelTensorAccessorArgsTypeGet(ctx.ptr()),
            .tensor_accessor => c.ttlangTTKernelTensorAccessorTypeGet(ctx.ptr()),
            .local_tensor_accessor => c.ttlangTTKernelLocalTensorAccessorTypeGet(ctx.ptr()),
            .tensor_accessor_page_mapping => c.ttlangTTKernelTensorAccessorPageMappingTypeGet(ctx.ptr()),
        };
        return @ptrCast(t.ptr);
    }
};

// =============================================================================
// Translation to C++
// =============================================================================

/// Runs `convert-ttkernel-to-emitc` on `module`: every TTKernel kernel
/// function becomes EmitC.
pub fn convertToEmitC(module: *mlir.Module) error{InvalidMlir}!void {
    if (c.mlirLogicalResultIsFailure(c.mlirTTKernelConvertToEmitC(module.ptr()))) return error.InvalidMlir;
}

/// The tt-metal C++ source of the EmitC kernel function `name` of `module`,
/// with its `#include`s. Caller owns the result.
pub fn translateToCpp(allocator: std.mem.Allocator, module: *mlir.Module, name: []const u8) error{ InvalidMlir, OutOfMemory }![]u8 {
    const Sink = struct {
        out: std.Io.Writer.Allocating,
        failed: bool = false,

        fn write(str: c.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
            const self: *@This() = @ptrCast(@alignCast(user_data.?));
            self.out.writer.writeAll(str.data[0..str.length]) catch {
                self.failed = true;
            };
        }
    };
    var sink: Sink = .{ .out = .init(allocator) };
    errdefer sink.out.deinit();
    const result = c.mlirTTKernelTranslateToCpp(module.ptr(), .{ .data = name.ptr, .length = name.len }, Sink.write, &sink);
    if (sink.failed) return error.OutOfMemory;
    if (c.mlirLogicalResultIsFailure(result)) return error.InvalidMlir;
    return sink.out.toOwnedSlice();
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(ops);
    std.testing.refAllDecls(ttcore.ops);
}

test "every ttkernel operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (ops.names) |name| {
        if (!c.mlirContextIsRegisteredOperation(ctx.ptr(), .{ .data = name.ptr, .length = name.len })) {
            std.debug.print("not registered: {s}\n", .{name});
            return error.TestUnexpectedResult;
        }
    }
    for (ttcore.ops.names) |name| {
        if (!c.mlirContextIsRegisteredOperation(ctx.ptr(), .{ .data = name.ptr, .length = name.len })) {
            std.debug.print("not registered: {s}\n", .{name});
            return error.TestUnexpectedResult;
        }
    }
}

test "translate a kernel to C++" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const module = try mlir.Module.parse(ctx,
        \\func.func @copy() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
        \\  %cb_in = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
        \\  %cb_out = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
        \\  %c0 = arith.constant 0 : i32
        \\  %c1 = arith.constant 1 : i32
        \\  ttkernel.init_sfpu(%cb_in, %cb_out) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>) -> ()
        \\  ttkernel.cb_wait_front(%cb_in, %c1) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
        \\  ttkernel.tile_regs_acquire() : () -> ()
        \\  ttkernel.copy_tile_init(%cb_in) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>) -> ()
        \\  ttkernel.copy_tile(%cb_in, %c0, %c0) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32, i32) -> ()
        \\  ttkernel.tile_regs_commit() : () -> ()
        \\  ttkernel.tile_regs_wait() : () -> ()
        \\  ttkernel.cb_reserve_back(%cb_out, %c1) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
        \\  ttkernel.pack_tile(%c0, %cb_out, %c0, false) : (i32, !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
        \\  ttkernel.tile_regs_release() : () -> ()
        \\  ttkernel.cb_push_back(%cb_out, %c1) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
        \\  ttkernel.cb_pop_front(%cb_in, %c1) : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
        \\  func.return
        \\}
    );
    defer module.deinit();
    try convertToEmitC(module);
    const cpp = try translateToCpp(std.testing.allocator, module, "copy");
    defer std.testing.allocator.free(cpp);
    try std.testing.expect(std.mem.indexOf(u8, cpp, "void kernel_main()") != null);
    try std.testing.expect(std.mem.indexOf(u8, cpp, "copy_tile(") != null);
}
