//! Host-side description of a Tenstorrent program: the kernels (one per
//! thread: reader, writer, compute), their circular buffers, semaphores and
//! arguments. Mirrors tt-metal's `ProgramDescriptor` and pykernel's
//! `create_cb` / `create_kernel` / `create_program`, and prints itself as the
//! `tt.tt_lang_op` kernel artifact (format version 3) understood by tt-mlir's
//! `--ttnn-lower-tt-lang-to-generic`.
//!
//! Tensor operands are referred to by their index in the op's io-tensors:
//! the inputs, then the outputs.

const std = @import("std");

pub const CoreCoord = struct {
    x: u32,
    y: u32,
};

/// Inclusive rectangle of cores.
pub const CoreRange = struct {
    start: CoreCoord,
    end: CoreCoord,

    pub fn single(core: CoreCoord) CoreRange {
        return .{ .start = core, .end = core };
    }

    pub fn grid(columns: u32, rows: u32) CoreRange {
        return .{ .start = .{ .x = 0, .y = 0 }, .end = .{ .x = columns - 1, .y = rows - 1 } };
    }

    pub fn numCores(self: CoreRange) u32 {
        return (self.end.x - self.start.x + 1) * (self.end.y - self.start.y + 1);
    }
};

/// The first `count` cores of a grid `cols` wide in row-major order: the
/// full rows, then the partial last row, if any.
pub fn firstCores(count: u32, cols: u32) struct { CoreRange, ?CoreRange } {
    std.debug.assert(count > 0);
    const full_rows = count / cols;
    const rest = count % cols;
    if (full_rows == 0) return .{ .{ .start = .{ .x = 0, .y = 0 }, .end = .{ .x = rest - 1, .y = 0 } }, null };
    const full: CoreRange = .{ .start = .{ .x = 0, .y = 0 }, .end = .{ .x = cols - 1, .y = full_rows - 1 } };
    if (rest == 0) return .{ full, null };
    return .{ full, .{ .start = .{ .x = 0, .y = full_rows }, .end = .{ .x = rest - 1, .y = full_rows } } };
}

/// Data formats of circular buffer pages (`ttcore::DataType`).
pub const DataType = enum {
    f32,
    f16,
    bf16,
    bfp_f8,
    bfp_bf8,
    bfp_f4,
    bfp_bf4,
    u32,
    u16,
    u8,
    si32,

    /// Size in bytes of one 32x32 tile (bfp formats include their exponents).
    pub fn tileSize(self: DataType) u32 {
        return switch (self) {
            .f32, .u32, .si32 => 4096,
            .f16, .bf16, .u16 => 2048,
            .u8 => 1024,
            .bfp_f8, .bfp_bf8 => 1088,
            .bfp_f4, .bfp_bf4 => 576,
        };
    }
};

pub const MathFidelity = enum { lofi, hifi2, hifi3, hifi4 };

/// One uint32 slot (or, for tensor-accessor args, a splice of slots) of a
/// kernel's compile-time or runtime arguments, resolved at launch.
pub const Arg = union(enum) {
    /// Index of a circular buffer (`tt::CBIndex`).
    cb: u32,
    /// Base address of the io-tensor at this index.
    tensor_address: u32,
    /// `TensorAccessorArgs(buffer).get_compile_time_args()` of the io-tensor
    /// at this index; compile-time args only.
    tensor_accessor_args: u32,
    /// Id of the program semaphore at this index.
    semaphore: u32,
    /// A constant.
    value: u32,
    /// The NoC x (`dim` 0) or y (`dim` 1) coordinate of a logical worker
    /// core, resolved at launch; runtime arguments only.
    core_noc: struct { core: CoreCoord, dim: u1 },
};

pub const CoreArgs = struct {
    core: CoreCoord,
    args: []const Arg,
};

pub const ComputeConfig = struct {
    math_fidelity: MathFidelity = .hifi4,
    fp32_dest_acc_en: bool = false,
    dst_full_sync_en: bool = false,
    bfp8_pack_precise: bool = false,
    math_approx_mode: bool = false,
    /// Circular buffers unpacked to the DST registers as fp32.
    unpack_to_dest_fp32: []const u32 = &.{},
};

/// Circular buffers per core, the length of tt-metal's unpack-to-dest modes.
const max_circular_buffers = 64;

pub const Thread = union(enum) {
    /// Data movement kernel on RISCV_0 (NOC 0 by default).
    reader,
    /// Data movement kernel on RISCV_1 (NOC 1 by default).
    writer,
    compute: ComputeConfig,
};

/// A preprocessor definition of a kernel (tt-metal's kernel `defines`),
/// prepended to its source.
pub const Define = struct {
    name: []const u8,
    value: []const u8 = "1",
};

pub const Kernel = struct {
    thread: Thread,
    /// Kernel C++ source (`void kernel_main()`), JIT-compiled by tt-metal.
    source: []const u8,
    defines: []const Define = &.{},
    cores: CoreRange,
    /// Further cores, disjoint from `cores`.
    more_cores: []const CoreRange = &.{},
    compile_time_args: []const Arg = &.{},
    /// Runtime arguments shared by every core.
    common_runtime_args: []const Arg = &.{},
    /// Per-core runtime arguments.
    runtime_args: []const CoreArgs = &.{},
};

pub const CircularBuffer = struct {
    index: u32,
    dtype: DataType,
    page_size: u32,
    num_pages: u32,
    cores: CoreRange,
    more_cores: []const CoreRange = &.{},
    /// Further circular buffers sharing this one's storage.
    aliases: []const Alias = &.{},
    /// The io-tensor (an L1 shard) backing the buffer instead of its own
    /// storage.
    tensor: ?u32 = null,

    pub const Alias = struct {
        index: u32,
        dtype: DataType,
        page_size: u32,
    };
};

pub const Semaphore = struct {
    cores: CoreRange,
    more_cores: []const CoreRange = &.{},
    initial_value: u32 = 0,
};

/// The layout an io-tensor must have for the program's kernels; tt-mlir
/// converts it before the kernels run (and outputs back after). Unset
/// fields keep the tensor's layout (tiled, interleaved in DRAM).
pub const Operand = struct {
    index: u32,
    layout: ?enum { tile, row_major } = null,
    buffer: ?enum { dram, l1 } = null,
    memory_layout: ?enum { interleaved, height_sharded, width_sharded, block_sharded } = null,
    /// Virtual grid of the shards, e.g. `[num_shards, 1]` when height
    /// sharded.
    grid: ?[2]u32 = null,
    /// Cores holding the shards, in order; placed canonically when empty.
    cores: []const CoreRange = &.{},
};

pub const Program = struct {
    kernels: []const Kernel,
    circular_buffers: []const CircularBuffer = &.{},
    semaphores: []const Semaphore = &.{},
    operands: []const Operand = &.{},

    /// Writes the program as a `#ttnn.program` attribute.
    pub fn format(self: Program, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.writeAll("#ttnn.program<kernels = [");
        for (self.kernels, 0..) |kernel, i| {
            if (i > 0) try writer.writeAll(", ");
            try writeKernel(writer, kernel);
        }
        try writer.writeAll("], cbs = [");
        for (self.circular_buffers, 0..) |cb, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("#ttnn.kernel_cb<total_size = {d}, ", .{cb.page_size * cb.num_pages});
            try writeCores(writer, cb.cores, cb.more_cores);
            try writer.print(", formats = [#ttnn.kernel_cb_format<buffer_index = {d}, dtype = {t}, page_size = {d}>", .{ cb.index, cb.dtype, cb.page_size });
            for (cb.aliases) |alias| {
                try writer.print(", #ttnn.kernel_cb_format<buffer_index = {d}, dtype = {t}, page_size = {d}>", .{ alias.index, alias.dtype, alias.page_size });
            }
            try writer.writeAll("]");
            if (cb.tensor) |t| try writer.print(", buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<{d}>", .{t});
            try writer.writeAll(">");
        }
        try writer.writeAll("], semaphores = [");
        for (self.semaphores, 0..) |sem, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("#ttnn.kernel_semaphore<id = {d}, core_type = worker, ", .{i});
            try writeCores(writer, sem.cores, sem.more_cores);
            try writer.print(", initial_value = {d}>", .{sem.initial_value});
        }
        try writer.writeAll("]>");
    }

    /// Returns the `tt.tt_lang_op` kernel artifact (format version 3) for this
    /// program. Caller owns the memory.
    pub fn artifact(self: Program, allocator: std.mem.Allocator) ![]u8 {
        var module: std.Io.Writer.Allocating = .init(allocator);
        defer module.deinit();
        try module.writer.print("module attributes {{ttnn.program = {f}}} {{\n}}\n", .{self});

        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();
        try out.writer.writeAll("{\"format_version\": 3, \"module\": ");
        try std.json.Stringify.encodeJsonString(module.written(), .{}, &out.writer);
        if (self.operands.len > 0) {
            try out.writer.writeAll(", \"operands\": [");
            for (self.operands, 0..) |operand, i| {
                if (i > 0) try out.writer.writeAll(", ");
                try out.writer.print("{{\"index\": {d}", .{operand.index});
                if (operand.layout) |l| try out.writer.print(", \"layout\": \"{t}\"", .{l});
                if (operand.buffer) |b| try out.writer.print(", \"buffer\": \"{t}\"", .{b});
                if (operand.memory_layout) |m| try out.writer.print(", \"memory_layout\": \"{t}\"", .{m});
                if (operand.grid) |g| try out.writer.print(", \"grid\": [{d}, {d}]", .{ g[0], g[1] });
                if (operand.cores.len > 0) {
                    try out.writer.writeAll(", \"cores\": [");
                    for (operand.cores, 0..) |r, j| {
                        if (j > 0) try out.writer.writeAll(", ");
                        try out.writer.print("[{d}, {d}, {d}, {d}]", .{ r.start.x, r.start.y, r.end.x, r.end.y });
                    }
                    try out.writer.writeAll("]");
                }
                try out.writer.writeAll("}");
            }
            try out.writer.writeAll("]");
        }
        try out.writer.writeAll("}");
        return out.toOwnedSlice();
    }
};

fn writeCores(writer: *std.Io.Writer, range: CoreRange, more: []const CoreRange) !void {
    try writer.print("core_ranges = #ttnn.core_range_set<[#ttnn.core_range<({d},{d}), ({d},{d})>", .{ range.start.x, range.start.y, range.end.x, range.end.y });
    for (more) |r| try writer.print(", #ttnn.core_range<({d},{d}), ({d},{d})>", .{ r.start.x, r.start.y, r.end.x, r.end.y });
    try writer.writeAll("]>");
}

fn writeArg(writer: *std.Io.Writer, arg: Arg) !void {
    switch (arg) {
        .cb => |i| try writer.print("#ttnn.kernel_arg_cb_buffer_index<{d}>", .{i}),
        .tensor_address => |i| try writer.print("#ttnn.kernel_arg_address_of_tensor<{d}>", .{i}),
        .tensor_accessor_args => |i| try writer.print("#ttnn.kernel_arg_tensor_accessor_args<{d}>", .{i}),
        .semaphore => |i| try writer.print("#ttnn.kernel_arg_semaphore_at<{d}>", .{i}),
        .value => |v| try writer.print("#ttnn.kernel_named_arg<name = \"value\", value = {d}>", .{v}),
        .core_noc => |c| try writer.print("#ttnn.kernel_arg_core_noc_coord<{d}, {d}, {d}>", .{ c.core.x, c.core.y, c.dim }),
    }
}

fn writeArgs(writer: *std.Io.Writer, name: []const u8, args: []const Arg) !void {
    try writer.print("{s} = [", .{name});
    for (args, 0..) |arg, i| {
        if (i > 0) try writer.writeAll(", ");
        try writeArg(writer, arg);
    }
    try writer.writeAll("]");
}

/// Writes `s` escaped for an MLIR string literal.
fn writeMlirStringContent(writer: *std.Io.Writer, s: []const u8) !void {
    for (s) |ch| switch (ch) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\0A"),
        '\t' => try writer.writeAll("\\09"),
        else => try writer.writeByte(ch),
    };
}

fn writeKernel(writer: *std.Io.Writer, kernel: Kernel) !void {
    const mnemonic = switch (kernel.thread) {
        .reader => "source_read_kernel",
        .writer => "source_write_kernel",
        .compute => "source_compute_kernel",
    };
    try writer.print("#ttnn.{s}<source = ", .{mnemonic});
    try writer.writeByte('"');
    for (kernel.defines) |d| {
        try writeMlirStringContent(writer, "#define ");
        try writeMlirStringContent(writer, d.name);
        try writeMlirStringContent(writer, " ");
        try writeMlirStringContent(writer, d.value);
        try writeMlirStringContent(writer, "\n");
    }
    try writeMlirStringContent(writer, kernel.source);
    try writer.writeByte('"');
    try writer.writeAll(", ");
    try writeCores(writer, kernel.cores, kernel.more_cores);
    switch (kernel.thread) {
        .reader, .writer => {},
        .compute => |cfg| {
            try writer.print(", math_fidelity = {t}, fp32_dest_acc_en = {}, dst_full_sync_en = {}, unpack_to_dest_modes = [", .{ cfg.math_fidelity, cfg.fp32_dest_acc_en, cfg.dst_full_sync_en });
            if (cfg.unpack_to_dest_fp32.len > 0) {
                for (0..max_circular_buffers) |i| {
                    if (i > 0) try writer.writeAll(", ");
                    const fp32 = std.mem.indexOfScalar(u32, cfg.unpack_to_dest_fp32, @intCast(i)) != null;
                    try writer.writeAll(if (fp32) "fp32" else "default");
                }
            }
            try writer.print("], bfp8_pack_precise = {}, math_approx_mode = {}", .{ cfg.bfp8_pack_precise, cfg.math_approx_mode });
        },
    }
    try writer.writeAll(", ");
    try writeArgs(writer, "ct_args", kernel.compile_time_args);
    try writer.writeAll(", ");
    try writeArgs(writer, "common_rt_args", kernel.common_runtime_args);
    try writer.writeAll(", rt_args = [");
    for (kernel.runtime_args, 0..) |core_args, i| {
        if (i > 0) try writer.writeAll(", ");
        try writer.print("#ttnn.core_runtime_args<core_coord = #ttnn.core_coord<{d}, {d}>, ", .{ core_args.core.x, core_args.core.y });
        try writeArgs(writer, "args", core_args.args);
        try writer.writeAll(">");
    }
    try writer.writeAll("]>");
}

test "program prints a #ttnn.program attribute" {
    const program: Program = .{
        .kernels = &.{
            .{
                .thread = .reader,
                .source = "void kernel_main() {\n  \"x\";\n}",
                .cores = .single(.{ .x = 0, .y = 0 }),
                .compile_time_args = &.{.{ .tensor_accessor_args = 0 }},
                .runtime_args = &.{.{ .core = .{ .x = 0, .y = 0 }, .args = &.{ .{ .tensor_address = 0 }, .{ .value = 4 } } }},
            },
            .{
                .thread = .{ .compute = .{} },
                .source = "",
                .cores = .single(.{ .x = 0, .y = 0 }),
                .compile_time_args = &.{ .{ .cb = 0 }, .{ .cb = 16 } },
            },
        },
        .circular_buffers = &.{.{ .index = 0, .dtype = .bf16, .page_size = 2048, .num_pages = 2, .cores = .single(.{ .x = 0, .y = 0 }) }},
    };
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try out.writer.print("{f}", .{program});
    try std.testing.expectEqualStrings(
        "#ttnn.program<kernels = [" ++
            "#ttnn.source_read_kernel<source = \"void kernel_main() {\\0A  \\\"x\\\";\\0A}\", " ++
            "core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (0,0)>]>, " ++
            "ct_args = [#ttnn.kernel_arg_tensor_accessor_args<0>], common_rt_args = [], " ++
            "rt_args = [#ttnn.core_runtime_args<core_coord = #ttnn.core_coord<0, 0>, args = [#ttnn.kernel_arg_address_of_tensor<0>, #ttnn.kernel_named_arg<name = \"value\", value = 4>]>]>, " ++
            "#ttnn.source_compute_kernel<source = \"\", core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (0,0)>]>, " ++
            "math_fidelity = hifi4, fp32_dest_acc_en = false, dst_full_sync_en = false, unpack_to_dest_modes = [], bfp8_pack_precise = false, math_approx_mode = false, " ++
            "ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<16>], common_rt_args = [], rt_args = []>" ++
            "], cbs = [#ttnn.kernel_cb<total_size = 4096, core_ranges = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (0,0)>]>, formats = [#ttnn.kernel_cb_format<buffer_index = 0, dtype = bf16, page_size = 2048>]>], semaphores = []>",
        out.written(),
    );
}

test "first cores in row-major order" {
    const a = firstCores(3, 8);
    try std.testing.expectEqual(CoreRange{ .start = .{ .x = 0, .y = 0 }, .end = .{ .x = 2, .y = 0 } }, a[0]);
    try std.testing.expectEqual(null, a[1]);
    const b = firstCores(19, 8);
    try std.testing.expectEqual(CoreRange{ .start = .{ .x = 0, .y = 0 }, .end = .{ .x = 7, .y = 1 } }, b[0]);
    try std.testing.expectEqual(CoreRange{ .start = .{ .x = 0, .y = 2 }, .end = .{ .x = 2, .y = 2 } }, b[1].?);
}

test "artifact wraps the program in a format version 3 JSON payload" {
    const program: Program = .{ .kernels = &.{.{ .thread = .reader, .source = "", .cores = .single(.{ .x = 0, .y = 0 }) }} };
    const art = try program.artifact(std.testing.allocator);
    defer std.testing.allocator.free(art);
    try std.testing.expect(std.mem.startsWith(u8, art, "{\"format_version\": 3, \"module\": \"module attributes {ttnn.program = #ttnn.program<"));
}
