//! tt-lang's `ttl` dialect: tensor-level Tenstorrent kernels (dataflow
//! buffers, block math, tensor slices copied over the NoC) that tt-lang's
//! pipeline lowers to `ttkernel`.

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");
const ttkernel = @import("mlir/dialects/ttkernel");

pub const ops = @import("ops.zig");

pub const make = ttkernel.make;
pub const parseType = ttkernel.parseType;

/// Dialects a TTL module uses before lowering.
pub const dialects_needed = [_][]const u8{ "func", "arith", "scf", "tensor", "ttcore", "ttkernel", "ttl" };

/// `ttl::TransferKind`.
pub const TransferKind = enum(i32) {
    read = 0,
    write = 1,
};

/// `ttl::BufferType`.
pub const BufferType = enum(u32) {
    dram = 0,
    l1 = 1,
    system_memory = 2,
    l1_small = 3,
};

/// `ttl::TensorMemoryLayout`.
pub const TensorMemoryLayout = enum(u32) {
    interleaved = 0,
    single_bank = 1,
    height_sharded = 2,
    width_sharded = 3,
    block_sharded = 4,
    nd_sharded = 5,
};

/// `ttl::ReduceType`, printed as an integer inside `ttl.reduce`.
pub const ReduceType = enum(i32) {
    sum = 0,
    max = 1,
};

/// `!ttl.cb<[shape], element_type, block_count>`: a dataflow buffer of
/// `block_count` blocks of `shape` tiles.
pub const CircularBufferType = opaque {
    const M = mlir.Methods(CircularBufferType, c.MlirType);

    pub const isAFn = c.mlirTypeIsATTLCircularBuffer;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, shape: []const i64, element_type: *const mlir.Type, block_count: i64) *const CircularBufferType {
        return @ptrCast(c.mlirTTLCircularBufferTypeGet(ctx.ptr(), @intCast(shape.len), shape.ptr, element_type.ptr(), block_count).ptr);
    }

    pub fn type_(self: *const CircularBufferType) *const mlir.Type {
        return @ptrCast(self);
    }

    pub fn rank(self: *const CircularBufferType) usize {
        return @intCast(c.mlirTTLCircularBufferTypeGetRank(self.ptr()));
    }

    pub fn dim(self: *const CircularBufferType, pos: usize) i64 {
        return c.mlirTTLCircularBufferTypeGetDim(self.ptr(), @intCast(pos));
    }

    pub fn elementType(self: *const CircularBufferType) *const mlir.Type {
        return @ptrCast(c.mlirTTLCircularBufferTypeGetElementType(self.ptr()).ptr);
    }

    pub fn blockCount(self: *const CircularBufferType) i64 {
        return c.mlirTTLCircularBufferTypeGetBlockCount(self.ptr());
    }
};

/// `!ttl.transfer_handle<kind>`, the result of `ttl.copy`.
pub fn transferHandleType(ctx: *mlir.Context, kind: ?TransferKind) *const mlir.Type {
    return @ptrCast(c.mlirTTLTransferHandleTypeGet(ctx.ptr(), if (kind) |k| @intFromEnum(k) else -1).ptr);
}

pub const LayoutArgs = struct {
    /// Logical (element) shape of the tensor.
    shape: []const i64,
    /// `!ttcore.tile` for tiled tensors, the scalar type for row-major ones.
    element_type: *const mlir.Type,
    buffer: BufferType = .dram,
    /// `[rows, cols]` of the grid the tensor is distributed over.
    grid: []const i64 = &.{ 1, 1 },
    memory: TensorMemoryLayout = .interleaved,
};

/// `#ttl.layout<...>`, the encoding of a tensor a TTL kernel reads or writes.
pub fn layoutAttr(ctx: *mlir.Context, args: LayoutArgs) *const mlir.Attribute {
    return @ptrCast(c.mlirTTLLayoutAttrGet(
        ctx.ptr(),
        @intCast(args.shape.len),
        args.shape.ptr,
        args.element_type.ptr(),
        @intFromEnum(args.buffer),
        @intCast(args.grid.len),
        args.grid.ptr,
        @intFromEnum(args.memory),
    ).ptr);
}

/// Runs the textual pass pipeline `pipeline` (`builtin.module(...)`) on
/// `module`, registering tt-lang's passes first.
pub fn runPassPipeline(module: *mlir.Module, pipeline: []const u8) error{InvalidMlir}!void {
    ttkernel.registerPasses();
    if (c.mlirLogicalResultIsFailure(c.mlirTTLRunPassPipeline(module.ptr(), .{ .data = pipeline.ptr, .length = pipeline.len }))) return error.InvalidMlir;
}

/// What the host program of a module lowered by tt-lang's pipeline is built
/// from (`mlirTTLDescribeKernels`).
pub const Description = struct {
    pub const Dfb = struct {
        dfb_index: u32,
        storage_index: u32,
        num_tiles: u32,
        block_count: u32,
        page_size: u32,
        /// `ttcore::DataType`.
        data_type: u32,
        tile_height: u32,
        tile_width: u32,
        storage_segments: bool,
        remote: bool,
    };

    pub const Kernel = struct {
        name: []const u8,
        /// "compute" or "noc".
        thread: []const u8,
        noc_index: ?u32,
        base_cta_index: ?u32,
        crta_indices: []const u32,
        fp32_dest_acc_en: ?bool,
        dst_full_sync_en: ?bool,
        unpack_to_dest_fp32: []const u32,
        pipe_computed_address_dfb_indices: []const u32,
        specialized: bool,
        local_tensors: bool,
        fabric: bool,
    };

    dfbs: []const Dfb,
    pipe_sync_semaphores: ?u32,
    pipe_global_semaphores: ?u32,
    pipe_sram_scratch_bytes: ?u32,
    dfb_resets: ?u32,
    dfb_reconfiguration: bool,
    kernels: []const Kernel,
};

/// Describes `module`, lowered by tt-lang's TTL pipeline but not yet
/// translated to C++. The result lives in `arena`.
pub fn describeKernels(arena: std.mem.Allocator, module: *mlir.Module) error{ InvalidMlir, OutOfMemory }!Description {
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
    var sink: Sink = .{ .out = .init(arena) };
    if (c.mlirLogicalResultIsFailure(c.mlirTTLDescribeKernels(module.ptr(), Sink.write, &sink))) return error.InvalidMlir;
    if (sink.failed) return error.OutOfMemory;
    return std.json.parseFromSliceLeaky(Description, arena, sink.out.written(), .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidMlir,
    };
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(ops);
}

test "every ttl operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    ttkernel.insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (ops.names) |name| {
        if (!c.mlirContextIsRegisteredOperation(ctx.ptr(), .{ .data = name.ptr, .length = name.len })) {
            std.debug.print("not registered: {s}\n", .{name});
            return error.TestUnexpectedResult;
        }
    }
}
