const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const platforms = @import("platforms");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/nki_simulator");

pub const custom_call_name = "zml$nki_simulate";

pub fn requested() bool {
    if (comptime !platforms.isEnabled(.neuron)) return false;
    const raw = std.c.getenv("NKI_SIMULATOR") orelse return false;
    const value = std.mem.span(raw);
    return value.len != 0 and !std.mem.eql(u8, value, "0");
}

pub fn initialize(io: std.Io) !void {
    if (comptime !platforms.isEnabled(.neuron)) return error.NeuronUnavailable;

    const r = try bazel.runfiles(bazel_builtin.current_repository);
    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_neuron/sandbox", &sandbox_path_buf) orelse
        return error.FileNotFound;

    var bridge_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const bridge_path = try r.rlocation("zml/platforms/neuron/nki_simulator_bridge.py", &bridge_path_buf) orelse
        return error.FileNotFound;
    const bridge_directory = std.fs.path.dirname(bridge_path) orelse return error.InvalidBridgePath;

    var python_home_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const python_home = try stdx.Io.Dir.path.bufJoinZ(&python_home_buf, &.{ sandbox_path, "lib", "python3.12" });
    var site_packages_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const site_packages = try stdx.Io.Dir.path.bufJoinZ(&site_packages_buf, &.{ sandbox_path, "site-packages" });
    var bridge_directory_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const bridge_directory_z = try std.fmt.bufPrintZ(&bridge_directory_buf, "{s}", .{bridge_directory});

    _ = io;
    if (c.zml_nki_simulator_initialize(python_home.ptr, site_packages.ptr, bridge_directory_z.ptr)) |message| {
        log.err("Failed to initialize embedded NKI simulator: {s}", .{std.mem.span(message)});
        return error.NkiSimulatorInitializationFailed;
    }
}

pub fn readKernelSource(allocator: std.mem.Allocator, io: std.Io, source_path: []const u8) ![]u8 {
    const r = try bazel.runfiles(bazel_builtin.current_repository);
    var resolved_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const resolved_path = try r.rlocation(source_path, &resolved_path_buf) orelse return error.FileNotFound;
    return std.Io.Dir.cwd().readFileAlloc(io, resolved_path, allocator, .unlimited);
}

fn dtypeName(dtype: pjrt.ffi.DataType) ?[]const u8 {
    return switch (dtype) {
        .bool => "bool",
        .i8 => "i8",
        .i16 => "i16",
        .i32 => "i32",
        .i64 => "i64",
        .u8 => "u8",
        .u16 => "u16",
        .u32 => "u32",
        .u64 => "u64",
        .f16 => "f16",
        .f32 => "f32",
        .f64 => "f64",
        .bf16 => "bf16",
        else => null,
    };
}

fn dtypeSize(dtype: pjrt.ffi.DataType) ?usize {
    return switch (dtype) {
        .bool, .i8, .u8 => 1,
        .i16, .u16, .f16, .bf16 => 2,
        .i32, .u32, .f32 => 4,
        .i64, .u64, .f64 => 8,
        else => null,
    };
}

fn bufferByteSize(buffer: *const pjrt.ffi.Buffer) !usize {
    var element_count: usize = 1;
    for (buffer.dims()) |dim| {
        if (dim < 0) return error.InvalidBufferShape;
        element_count = try std.math.mul(usize, element_count, @intCast(dim));
    }
    return std.math.mul(usize, element_count, dtypeSize(buffer.dtype) orelse return error.UnsupportedDtype);
}

fn descriptor(buffer: *const pjrt.ffi.Buffer) !c.zml_nki_simulator_buffer {
    const dtype = dtypeName(buffer.dtype) orelse return error.UnsupportedDtype;
    return .{
        .data = buffer.data,
        .byte_size = try bufferByteSize(buffer),
        .dims = buffer.dims().ptr,
        .rank = buffer.dims().len,
        .dtype = dtype.ptr,
        .dtype_len = dtype.len,
    };
}

pub const handler: *const pjrt.ffi.Handler = callback;

fn callback(call_frame: *pjrt.ffi.CallFrame) callconv(.c) ?*pjrt.ffi.Error {
    if (call_frame.registeringHook()) return null;
    return callbackInner(call_frame) catch |err| {
        log.err("NKI simulator callback failed: {}", .{err});
        return pjrt.ffi.Error.create(call_frame.api, .internal, "NKI simulator callback failed");
    };
}

fn callbackInner(call_frame: *pjrt.ffi.CallFrame) !?*pjrt.ffi.Error {
    const source = call_frame.attrs.getByName(.string, "source") orelse return error.MissingSource;
    const entrypoint = call_frame.attrs.getByName(.string, "entrypoint") orelse return error.MissingEntrypoint;
    const compiler_target = call_frame.attrs.getByName(.string, "compiler_target") orelse return error.MissingCompilerTarget;
    const grid_attr = call_frame.attrs.getByName(.scalar, "grid") orelse return error.MissingGrid;
    const grid = grid_attr.get(i64);

    const input_buffers = call_frame.args.buffers();
    const output_buffers = call_frame.results.buffers();
    if (input_buffers.len > 32 or output_buffers.len > 32) return error.TooManyBuffers;

    var input_descriptors: [32]c.zml_nki_simulator_buffer = undefined;
    for (input_buffers, 0..) |buffer, index| {
        input_descriptors[index] = try descriptor(buffer);
    }
    var output_descriptors: [32]c.zml_nki_simulator_buffer = undefined;
    for (output_buffers, 0..) |buffer, index| {
        output_descriptors[index] = try descriptor(buffer);
    }

    const source_slice = source.slice();
    const entrypoint_slice = entrypoint.slice();
    const compiler_target_slice = compiler_target.slice();
    if (c.zml_nki_simulator_execute(
        source_slice.ptr,
        source_slice.len,
        entrypoint_slice.ptr,
        entrypoint_slice.len,
        compiler_target_slice.ptr,
        compiler_target_slice.len,
        grid,
        &input_descriptors,
        input_buffers.len,
        &output_descriptors,
        output_buffers.len,
    )) |message| {
        const error_message = std.mem.span(message);
        log.err("{s}", .{error_message});
        return pjrt.ffi.Error.create(call_frame.api, .internal, error_message);
    }
    return null;
}
