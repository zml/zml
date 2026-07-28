const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const runfiles = @import("runfiles");

const log = std.log.scoped(.@"zml/platforms/cudaz/compiler");
const zig_runtime_config = "zml/platforms/cudaz/zig_runtime.txt";

pub const Error = error{
    RunfilesUnavailable,
    ToolchainConfigUnavailable,
    ZigExecutableUnavailable,
    ZigLibUnavailable,
    KernelSourceUnavailable,
    ZigCompilationFailed,
    InvalidKernelIr,
    InvalidMlirProgram,
    MissingMainFunction,
    UnsupportedOutputType,
};

pub const OutputSpec = struct {
    element_type: c.PJRT_Buffer_Type,
    dims: []i64,

    fn clone(self: OutputSpec, allocator: std.mem.Allocator) !OutputSpec {
        return .{
            .element_type = self.element_type,
            .dims = try allocator.dupe(i64, self.dims),
        };
    }

    fn deinit(self: OutputSpec, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
    }
};

pub fn cloneOutputSpecs(allocator: std.mem.Allocator, specs: []const OutputSpec) ![]OutputSpec {
    const result = try allocator.alloc(OutputSpec, specs.len);
    errdefer allocator.free(result);

    var initialized: usize = 0;
    errdefer for (result[0..initialized]) |spec| spec.deinit(allocator);
    for (specs, result) |spec, *cloned| {
        cloned.* = try spec.clone(allocator);
        initialized += 1;
    }
    return result;
}

pub fn deinitOutputSpecs(allocator: std.mem.Allocator, specs: []OutputSpec) void {
    for (specs) |spec| spec.deinit(allocator);
    allocator.free(specs);
}

pub fn parseOutputSpecs(allocator: std.mem.Allocator, program: []const u8) ![]OutputSpec {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ "func", "stablehlo", "sdy" }) |dialect| {
        mlir.DialectHandle.fromString(dialect).insertDialect(registry);
    }

    const context = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer context.deinit();
    context.loadAllAvailableDialects();

    const module = dialects.stablehlo.deserializePortableArtifact(context, program) catch
        return error.InvalidMlirProgram;
    defer module.deinit();
    return outputSpecsFromModule(allocator, module);
}

fn outputSpecsFromModule(allocator: std.mem.Allocator, module: *const mlir.Module) ![]OutputSpec {
    const main = findMainFunction(module) orelse return error.MissingMainFunction;
    const function_type_attribute = main.attributeByName("function_type") orelse
        return error.MissingMainFunction;
    const type_attribute = function_type_attribute.isA(mlir.TypeAttribute) orelse
        return error.MissingMainFunction;
    const function_type = type_attribute.value().isA(mlir.FunctionType) orelse
        return error.MissingMainFunction;

    const output_count: usize = @intCast(c.mlirFunctionTypeGetNumResults(function_type.ptr()));
    const outputs = try allocator.alloc(OutputSpec, output_count);
    errdefer allocator.free(outputs);

    var initialized: usize = 0;
    errdefer for (outputs[0..initialized]) |spec| spec.deinit(allocator);
    for (outputs, 0..) |*output, index| {
        const result_type: *const mlir.Type = @ptrCast(
            c.mlirFunctionTypeGetResult(function_type.ptr(), @intCast(index)).ptr,
        );
        const tensor_type = result_type.isA(mlir.RankedTensorType) orelse
            return error.UnsupportedOutputType;

        const dims = try allocator.alloc(i64, tensor_type.rank());
        errdefer allocator.free(dims);
        for (dims, 0..) |*dim, dimension_index| {
            dim.* = tensor_type.dimension(dimension_index);
            if (dim.* < 0) return error.UnsupportedOutputType;
        }
        output.* = .{
            .element_type = try pjrtElementType(module.context(), tensor_type.elementType()),
            .dims = dims,
        };
        initialized += 1;
    }
    return outputs;
}

fn findMainFunction(module: *const mlir.Module) ?*const mlir.Operation {
    var maybe_operation: ?*const mlir.Operation = module.body().firstOperation();
    while (maybe_operation) |operation| {
        if (std.mem.eql(u8, operation.name(), "func.func")) {
            if (operation.attributeByName("sym_name")) |name_attribute| {
                if (name_attribute.isA(mlir.StringAttribute)) |name| {
                    if (std.mem.eql(u8, name.value(), "main")) return operation;
                }
            }
        }

        const next = c.mlirOperationGetNextInBlock(operation.ptr());
        maybe_operation = if (next.ptr) |ptr| @ptrCast(ptr) else null;
    }
    return null;
}

fn pjrtElementType(context: *mlir.Context, element_type: *const mlir.Type) !c.PJRT_Buffer_Type {
    const mapping = .{
        .{ c.PJRT_Buffer_Type_PRED, mlir.Type.int(context, .i1) },
        .{ c.PJRT_Buffer_Type_S2, mlir.Type.int(context, .i2) },
        .{ c.PJRT_Buffer_Type_S4, mlir.Type.int(context, .i4) },
        .{ c.PJRT_Buffer_Type_S8, mlir.Type.int(context, .i8) },
        .{ c.PJRT_Buffer_Type_S16, mlir.Type.int(context, .i16) },
        .{ c.PJRT_Buffer_Type_S32, mlir.Type.int(context, .i32) },
        .{ c.PJRT_Buffer_Type_S64, mlir.Type.int(context, .i64) },
        .{ c.PJRT_Buffer_Type_U2, mlir.Type.int(context, .u2) },
        .{ c.PJRT_Buffer_Type_U4, mlir.Type.int(context, .u4) },
        .{ c.PJRT_Buffer_Type_U8, mlir.Type.int(context, .u8) },
        .{ c.PJRT_Buffer_Type_U16, mlir.Type.int(context, .u16) },
        .{ c.PJRT_Buffer_Type_U32, mlir.Type.int(context, .u32) },
        .{ c.PJRT_Buffer_Type_U64, mlir.Type.int(context, .u64) },
        .{ c.PJRT_Buffer_Type_F4E2M1FN, mlir.Type.float(context, .f4e2m1fn) },
        .{ c.PJRT_Buffer_Type_F8E3M4, mlir.Type.float(context, .f8e3m4) },
        .{ c.PJRT_Buffer_Type_F8E4M3, mlir.Type.float(context, .f8e4m3) },
        .{ c.PJRT_Buffer_Type_F8E4M3B11FNUZ, mlir.Type.float(context, .f8e4m3b11fnuz) },
        .{ c.PJRT_Buffer_Type_F8E4M3FN, mlir.Type.float(context, .f8e4m3fn) },
        .{ c.PJRT_Buffer_Type_F8E4M3FNUZ, mlir.Type.float(context, .f8e4m3fnuz) },
        .{ c.PJRT_Buffer_Type_F8E5M2, mlir.Type.float(context, .f8e5m2) },
        .{ c.PJRT_Buffer_Type_F8E5M2FNUZ, mlir.Type.float(context, .f8e5m2fnuz) },
        .{ c.PJRT_Buffer_Type_F8E8M0FNU, mlir.Type.float(context, .f8e8m0fnu) },
        .{ c.PJRT_Buffer_Type_BF16, mlir.Type.float(context, .bf16) },
        .{ c.PJRT_Buffer_Type_F16, mlir.Type.float(context, .f16) },
        .{ c.PJRT_Buffer_Type_F32, mlir.Type.float(context, .f32) },
        .{ c.PJRT_Buffer_Type_F64, mlir.Type.float(context, .f64) },
        .{ c.PJRT_Buffer_Type_C64, mlir.Type.complex(context, .c64) },
        .{ c.PJRT_Buffer_Type_C128, mlir.Type.complex(context, .c128) },
    };
    inline for (mapping) |entry| {
        if (element_type.eql(entry[1])) return entry[0];
    }
    return error.UnsupportedOutputType;
}

test "read output specifications from main function" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    mlir.DialectHandle.fromString("func").insertDialect(registry);

    const context = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer context.deinit();
    context.loadAllAvailableDialects();

    const module = try mlir.Module.parse(
        context,
        "module { func.func private @main() -> (tensor<2x3xf32>, tensor<4xui32>) }",
    );
    defer module.deinit();

    const outputs = try outputSpecsFromModule(std.testing.allocator, module);
    defer deinitOutputSpecs(std.testing.allocator, outputs);
    try std.testing.expectEqual(@as(usize, 2), outputs.len);
    try std.testing.expectEqual(
        @as(c.PJRT_Buffer_Type, c.PJRT_Buffer_Type_F32),
        outputs[0].element_type,
    );
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, outputs[0].dims);
    try std.testing.expectEqual(
        @as(c.PJRT_Buffer_Type, c.PJRT_Buffer_Type_U32),
        outputs[1].element_type,
    );
    try std.testing.expectEqualSlices(i64, &.{4}, outputs[1].dims);

    var portable_artifact: std.Io.Writer.Allocating = try .initCapacity(
        std.testing.allocator,
        4096,
    );
    defer portable_artifact.deinit();
    try dialects.stablehlo.serializePortableArtifact2(
        module,
        dialects.stablehlo.minimumVersion(),
        &portable_artifact.writer,
    );

    const portable_outputs = try parseOutputSpecs(
        std.testing.allocator,
        portable_artifact.written(),
    );
    defer deinitOutputSpecs(std.testing.allocator, portable_outputs);
    try std.testing.expectEqual(@as(usize, 2), portable_outputs.len);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, portable_outputs[0].dims);
    try std.testing.expectEqualSlices(i64, &.{4}, portable_outputs[1].dims);
}

pub fn compile(allocator: std.mem.Allocator) ![:0]u8 {
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const executable_path = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(executable_path);

    var runfiles_instance = try runfiles.Runfiles.create(.{
        .allocator = allocator,
        .io = io,
        .argv0 = executable_path,
        .directory = if (std.c.getenv("RUNFILES_DIR")) |value| std.mem.span(value) else null,
        .manifest = if (std.c.getenv("RUNFILES_MANIFEST_FILE")) |value| std.mem.span(value) else null,
    }) orelse return error.RunfilesUnavailable;
    defer runfiles_instance.deinit(allocator);
    const r = runfiles_instance.withSourceRepo(bazel_builtin.current_repository);

    const config_path = try r.rlocationAlloc(allocator, zig_runtime_config) orelse
        return error.ToolchainConfigUnavailable;
    defer allocator.free(config_path);
    const config = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(16 * 1024));
    defer allocator.free(config);
    var config_lines = std.mem.tokenizeScalar(u8, config, '\n');
    const zig_runfile = config_lines.next() orelse return error.ToolchainConfigUnavailable;
    const zig_lib_runfile = config_lines.next() orelse return error.ToolchainConfigUnavailable;
    const kernel_runfile = config_lines.next() orelse return error.ToolchainConfigUnavailable;

    const zig_path = try r.rlocationAlloc(allocator, zig_runfile) orelse
        return error.ZigExecutableUnavailable;
    defer allocator.free(zig_path);
    const zig_lib_path = try r.rlocationAlloc(allocator, zig_lib_runfile) orelse
        return error.ZigLibUnavailable;
    defer allocator.free(zig_lib_path);
    const kernel_path = try r.rlocationAlloc(allocator, kernel_runfile) orelse
        return error.KernelSourceUnavailable;
    defer allocator.free(kernel_path);

    const tmp_root_path = if (std.c.getenv("TMPDIR")) |value| std.mem.span(value) else "/tmp";
    var tmp_root = try std.Io.Dir.openDir(.cwd(), io, tmp_root_path, .{});
    defer tmp_root.close(io);

    var dir_name_buffer: [128]u8 = undefined;
    const dir_name = try std.fmt.bufPrint(
        &dir_name_buffer,
        "zml-cudaz-{d}",
        .{std.Io.Timestamp.now(io, .real).nanoseconds},
    );
    var tmp_dir = try tmp_root.createDirPathOpen(io, dir_name, .{ .permissions = .fromMode(0o700) });
    defer tmp_root.deleteTree(io, dir_name) catch {};
    defer tmp_dir.close(io);

    var tmp_path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const tmp_path_len = try tmp_dir.realPath(io, &tmp_path_buffer);
    const tmp_path = tmp_path_buffer[0..tmp_path_len];

    try runZig(allocator, io, tmp_path, &.{
        zig_path,
        "build-obj",
        kernel_path,
        "-target",
        "nvptx64-cuda",
        "-mcpu",
        "sm_50+ptx63",
        "-O",
        "ReleaseFast",
        "-fno-incremental",
        "-fno-entry",
        "-fno-emit-bin",
        "-femit-llvm-ir=kernel.ll",
        "--zig-lib-dir",
        zig_lib_path,
        "--cache-dir",
        "cache",
        "--global-cache-dir",
        "global-cache",
    });

    const llvm_ir = try tmp_dir.readFileAlloc(io, "kernel.ll", allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(llvm_ir);
    const fixed_llvm_ir = try rewriteKernelIr(allocator, llvm_ir);
    defer allocator.free(fixed_llvm_ir);

    const fixed_ir_file = try tmp_dir.createFile(io, "kernel.fixed.ll", .{});
    defer fixed_ir_file.close(io);
    try fixed_ir_file.writePositionalAll(io, fixed_llvm_ir, 0);

    try runZig(allocator, io, tmp_path, &.{
        zig_path,
        "cc",
        "-target",
        "nvptx64-cuda",
        "-Xclang",
        "-target-cpu",
        "-Xclang",
        "sm_50",
        "-S",
        "kernel.fixed.ll",
        "-o",
        "kernel.ptx",
    });

    return try tmp_dir.readFileAllocOptions(
        io,
        "kernel.ptx",
        allocator,
        .limited(4 * 1024 * 1024),
        .of(u8),
        0,
    );
}

fn runZig(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: []const u8,
    argv: []const []const u8,
) !void {
    const local_cache = try std.Io.Dir.path.join(allocator, &.{ cwd, "cache" });
    defer allocator.free(local_cache);
    const global_cache = try std.Io.Dir.path.join(allocator, &.{ cwd, "global-cache" });
    defer allocator.free(global_cache);
    var environ = std.process.Environ.Map.init(allocator);
    defer environ.deinit();
    try environ.put("ZIG_LOCAL_CACHE_DIR", local_cache);
    try environ.put("ZIG_GLOBAL_CACHE_DIR", global_cache);

    const result = std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = &environ,
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch return error.ZigCompilationFailed;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code == 0) return,
        else => {},
    }
    log.err("Zig PTX compilation failed:\n{s}", .{result.stderr});
    return error.ZigCompilationFailed;
}

fn rewriteKernelIr(allocator: std.mem.Allocator, llvm_ir: []const u8) ![]u8 {
    const alias_start = std.mem.indexOf(u8, llvm_ir, "@main = alias") orelse
        return error.InvalidKernelIr;
    const alias_end = std.mem.indexOfPos(u8, llvm_ir, alias_start, "\n") orelse
        return error.InvalidKernelIr;
    const kernel_declaration = "define private ptx_kernel void @kernel.main";
    if (std.mem.indexOf(u8, llvm_ir, kernel_declaration) == null) return error.InvalidKernelIr;

    const without_alias = try std.mem.concat(allocator, u8, &.{
        llvm_ir[0..alias_start],
        llvm_ir[alias_end + 1 ..],
    });
    defer allocator.free(without_alias);
    const renamed = try std.mem.replaceOwned(
        u8,
        allocator,
        without_alias,
        kernel_declaration,
        "define ptx_kernel void @main",
    );
    defer allocator.free(renamed);
    return try std.mem.replaceOwned(u8, allocator, renamed, " unnamed_addr", "");
}

test "rewrite exported Zig kernel IR for the NVPTX backend" {
    const input =
        \\@main = alias void (ptr, i64), ptr @kernel.main
        \\
        \\define private ptx_kernel void @kernel.main(ptr %0, i64 %1) unnamed_addr {
        \\  ret void
        \\}
    ;
    const actual = try rewriteKernelIr(std.testing.allocator, input);
    defer std.testing.allocator.free(actual);
    try std.testing.expectEqualStrings(
        \\
        \\define ptx_kernel void @main(ptr %0, i64 %1) {
        \\  ret void
        \\}
    ,
        actual,
    );
}

test "Bazel Zig toolchain compiles the cudaz kernel to PTX" {
    const ptx = try compile(std.testing.allocator);
    defer std.testing.allocator.free(ptx);
    try std.testing.expect(std.mem.indexOf(u8, ptx, ".visible .entry main(") != null);
}
