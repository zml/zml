const std = @import("std");

const c_interface = @import("c");
const zml = @import("zml");
const Tensor = zml.Tensor;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const cfg = struct {
    const M = 256;
    const N = 256;
    const K = 256;
    const BLOCK_M = 128;
    const BLOCK_N = 128;
    const BLOCK_K = 32;
};

var python_initialized: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);

fn pyStatusCheck(status: c_interface.PyStatus) void {
    if (c_interface.PyStatus_Exception(status) != 0) {
        if (c_interface.PyStatus_IsExit(status) != 0) {
            std.process.exit(@intCast(status.exitcode));
        }
        c_interface.Py_ExitStatusException(status);
    }
}

fn checkPythonError() void {
    if (c_interface.PyErr_Occurred() != null) {
        c_interface.PyErr_Print();
    }
}

pub fn toPosixPathW(file_path: []const u8) error{NameTooLong}![std.posix.PATH_MAX - 1:0]c_interface.wchar_t {
    if (file_path.len >= std.posix.PATH_MAX) return error.NameTooLong;

    var path_with_null: [std.posix.PATH_MAX - 1:0]c_interface.wchar_t = undefined;
    const len = c_interface.mbstowcs(&path_with_null, file_path.ptr, file_path.len);
    path_with_null[len] = 0;
    return path_with_null;
}

fn findPythonPrefixFromManifest(io: anytype) ?[]u8 {
    const manifest_path_c = std.c.getenv("RUNFILES_MANIFEST_FILE") orelse return null;
    const manifest_path = std.mem.span(manifest_path_c);
    const manifest = std.Io.Dir.cwd().readFileAlloc(io, manifest_path, std.heap.c_allocator, .unlimited) catch return null;
    defer std.heap.c_allocator.free(manifest);

    var lines = std.mem.splitScalar(u8, manifest, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, "\r");
        if (line.len == 0) continue;

        const sep_idx = std.mem.indexOfScalar(u8, line, ' ') orelse continue;
        const value = line[sep_idx + 1 ..];

        if (std.mem.indexOf(u8, value, "/lib/libpython3.12.so") == null) continue;
        const lib_dir_idx = std.mem.lastIndexOf(u8, value, "/lib/") orelse continue;

        return std.heap.c_allocator.dupe(u8, value[0..lib_dir_idx]) catch null;
    }

    return null;
}

fn findPythonPrefixFromKnownPaths(io: anytype) ?[]u8 {
    const base_paths = [_][]const u8{
        "bazel-bin",
        "bazel-out/linux_amd64-dbg/bin",
        "../bazel-out/linux_amd64-dbg/bin",
        "/mnt/workspace/zml/bazel-bin",
    };
    const libpython_paths = [_][]const u8{
        "_solib_k8/_U_A_Arules_Upython++python+python_U3_U12_Ux86_U64-unknown-linux-gnu_S_S_Clibpython___Ulib/libpython3.12.so",
        "_solib_k8/_U_A_Arules_Upython++python+python_U3_U12_Uaarch64-unknown-linux-gnu_S_S_Clibpython___Ulib/libpython3.12.so",
    };

    for (base_paths) |base| {
        for (libpython_paths) |libpython_rel| {
            var candidate_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const candidate = std.fmt.bufPrint(&candidate_buf, "{s}/{s}", .{ base, libpython_rel }) catch continue;

            var file = std.Io.Dir.openFile(.cwd(), io, candidate, .{}) catch continue;
            defer file.close(io);

            var real_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const real_path_len = file.realPath(io, &real_path_buf) catch continue;
            const real_path = real_path_buf[0..real_path_len];
            const lib_dir_idx = std.mem.lastIndexOf(u8, real_path, "/lib/") orelse continue;
            return std.heap.c_allocator.dupe(u8, real_path[0..lib_dir_idx]) catch null;
        }
    }

    return null;
}

fn bootstrapPythonPath() !void {
    const code: [:0]const u8 =
        \\import glob
        \\import os
        \\import sys
        \\
        \\roots = []
        \\for key in ("RUNFILES_DIR", "RUNFILES_DIRECTORY"):
        \\    root = os.environ.get(key)
        \\    if root:
        \\        roots.append(root)
        \\
        \\manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
        \\if manifest and os.path.isfile(manifest):
        \\    with open(manifest, "r", encoding="utf-8", errors="ignore") as f:
        \\        for line in f:
        \\            parts = line.rstrip("\\n").split(" ", 1)
        \\            if len(parts) != 2:
        \\                continue
        \\            path = parts[1]
        \\            idx = path.find("/site-packages/")
        \\            if idx >= 0:
        \\                roots.append(path[: idx + len("/site-packages")])
        \\                continue
        \\            if path.endswith("/site-packages"):
        \\                roots.append(path)
        \\
        \\exe_path = os.path.realpath(sys.argv[0]) if sys.argv else ""
        \\exe_dir = os.path.dirname(exe_path) if exe_path else ""
        \\if exe_dir:
        \\    roots.append(exe_dir + ".runfiles")
        \\    roots.append(os.path.realpath(os.path.join(exe_dir, "..", "hello_world_sandbox.runfiles")))
        \\    roots.append(os.path.realpath(os.path.join(exe_dir, "..")))
        \\
        \\cands = []
        \\for root in roots:
        \\    if not root:
        \\        continue
        \\    cands.extend(
        \\        [
        \\            root,
        \\            os.path.join(root, "_main"),
        \\            os.path.join(root, "_main", "examples", "ttir_sandbox"),
        \\            os.path.join(root, "examples", "ttir_sandbox"),
        \\            os.path.join(root, "site-packages"),
        \\        ]
        \\    )
        \\    for p in glob.glob(os.path.join(root, "**", "site-packages"), recursive=True):
        \\        cands.append(p)
        \\
        \\seen = set()
        \\for p in cands:
        \\    rp = os.path.realpath(p)
        \\    if rp in seen:
        \\        continue
        \\    seen.add(rp)
        \\    if os.path.isdir(rp) and rp not in sys.path:
        \\        sys.path.insert(0, rp)
        \\
    ++ "\x00";

    if (c_interface.PyRun_SimpleStringFlags(code.ptr, null) != 0) {
        checkPythonError();
        return error.PythonPathBootstrapFailed;
    }
}

fn initializePython(io: anytype) !void {
    {
        var preconfig: c_interface.PyPreConfig = undefined;
        c_interface.PyPreConfig_InitIsolatedConfig(&preconfig);
        preconfig.utf8_mode = 1;
        pyStatusCheck(c_interface.Py_PreInitialize(&preconfig));
    }

    var config: c_interface.PyConfig = undefined;
    c_interface.PyConfig_InitPythonConfig(&config);
    defer c_interface.PyConfig_Clear(&config);

    config.optimization_level = 2;
    config.write_bytecode = 0;

    if (findPythonPrefixFromManifest(io) orelse findPythonPrefixFromKnownPaths(io)) |python_prefix| {
        defer std.heap.c_allocator.free(python_prefix);

        var home_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const home = std.fmt.bufPrintZ(&home_buf, "{s}", .{python_prefix}) catch null;
        if (home) |h| {
            pyStatusCheck(c_interface.PyConfig_SetBytesString(&config, &config.home, h));
        }

        var stdlib_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const stdlib = std.fmt.bufPrintZ(&stdlib_buf, "{s}/lib/python{d}.{d}", .{
            python_prefix,
            c_interface.PY_MAJOR_VERSION,
            c_interface.PY_MINOR_VERSION,
        }) catch null;
        if (stdlib) |p| {
            const stdlib_wide = toPosixPathW(p) catch null;
            if (stdlib_wide) |w| {
                pyStatusCheck(c_interface.PyWideStringList_Append(&config.module_search_paths, &w));
            }
        }
    }

    pyStatusCheck(c_interface.Py_InitializeFromConfig(&config));
    try bootstrapPythonPath();
    python_initialized.store(true, .release);
}

fn ensurePythonInitialized(io: anytype) !void {
    if (!python_initialized.load(.acquire)) {
        try initializePython(io);
    }
}

fn callCompileHelloWorldTtir(allocator: std.mem.Allocator, io: anytype, kernel_params: []const u8) ![:0]u8 {
    try ensurePythonInitialized(io);

    const module = c_interface.PyImport_ImportModule("ttir_compile");
    if (module == null) {
        checkPythonError();
        return error.PythonModuleImportFailed;
    }
    defer c_interface.Py_DecRef(module);

    const compile_func = c_interface.PyObject_GetAttrString(module, "compile_hello_world_ttir");
    if (compile_func == null) {
        checkPythonError();
        return error.PythonAttributeNotFound;
    }
    defer c_interface.Py_DecRef(compile_func);

    const py_params = c_interface.PyUnicode_FromStringAndSize(kernel_params.ptr, @intCast(kernel_params.len));
    if (py_params == null) {
        checkPythonError();
        return error.PythonObjectCreationFailed;
    }
    defer c_interface.Py_DecRef(py_params);

    const result_obj = c_interface.PyObject_CallFunctionObjArgs(compile_func, py_params, @as(?*c_interface.PyObject, null));
    defer {
        if (result_obj != null) {
            c_interface.Py_DecRef(result_obj);
        }
    }

    if (result_obj == null or c_interface.Py_IsNone(result_obj) != 0) {
        checkPythonError();
        return error.PythonFunctionCallFailed;
    }

    const result_cstr = c_interface.PyUnicode_AsUTF8(result_obj);
    if (result_cstr == null) {
        checkPythonError();
        return error.PythonFunctionCallFailed;
    }

    const result_slice = std.mem.span(@as([*:0]const u8, @ptrCast(result_cstr)));
    return allocator.dupeZ(u8, result_slice);
}

pub fn wrappedHelloWorld(kernel_ttir: [:0]const u8, a: Tensor, b: Tensor, out: Tensor) Tensor {
    const grid: [3]i32 = .{
        @intCast(@divExact(cfg.M, cfg.BLOCK_M)),
        @intCast(@divExact(cfg.N, cfg.BLOCK_N)),
        1,
    };

    const num_warps: i32 = 8;

    return zml.ops.triton(.{ a, b }, .{out.shape()}, .{
        .name = "matmul_fixed_kernel",
        .ir = kernel_ttir,
        .grid = grid,
        .num_stages = 1,
        .num_warps = num_warps,
        .debug = true,
        .output_operand_aliases = &.{},
    })[0];
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    std.log.info("\n{f}", .{platform.fmtVerbose()});

    if (platform.target != .cuda and platform.target != .rocm) {
        std.log.err("ttir_sandbox requires CUDA/ROCm target, got {s}", .{@tagName(platform.target)});
        return;
    }

    const kernel_params =
        "{" ++
        "\"M\":256," ++
        "\"N\":256," ++
        "\"K\":256," ++
        "\"BLOCK_M\":128," ++
        "\"BLOCK_N\":128," ++
        "\"BLOCK_K\":32," ++
        "\"num_warps\":8" ++
        "}";

    const ttir = try callCompileHelloWorldTtir(allocator, io, kernel_params);
    defer allocator.free(ttir);

    const a_shape: zml.Tensor = .init(.{ cfg.M, cfg.K }, .f32);
    const b_shape: zml.Tensor = .init(.{ cfg.K, cfg.N }, .f32);
    const c_shape: zml.Tensor = .init(.{ cfg.M, cfg.N }, .f32);

    var exe = try platform.compileFn(allocator, io, wrappedHelloWorld, .{
        ttir,
        a_shape,
        b_shape,
        c_shape,
    });
    defer exe.deinit();

    var a = try zeroBuffer(allocator, io, platform, a_shape.shape());
    defer a.deinit();
    var b = try zeroBuffer(allocator, io, platform, b_shape.shape());
    defer b.deinit();
    var c = try zeroBuffer(allocator, io, platform, c_shape.shape());
    defer c.deinit();

    std.log.info("a shape: {f}, device: {s}", .{ a.shape(), @tagName(platform.target) });
    std.log.info("b shape: {f}, device: {s}", .{ b.shape(), @tagName(platform.target) });
    std.log.info("c shape: {f}, device: {s}", .{ c.shape(), @tagName(platform.target) });

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ a, b, c });
    exe.call(exe_args, &exe_results);

    var result: zml.Buffer = exe_results.get(zml.Buffer);
    defer result.deinit();

    var output = try result.toSliceAlloc(allocator, io);
    defer output.free(allocator);

    const output_items = output.constItems(f32);
    if (output_items.len == 0) {
        std.debug.print("c[0:10,0:10] after matmul: <empty>\n", .{});
        return;
    }

    const rows: usize = @intCast(result.shape().dim(0));
    const cols: usize = @intCast(result.shape().dim(1));
    const rmax = @min(rows, 10);
    const cmax = @min(cols, 10);

    std.debug.print("c[0:10,0:10] after matmul:\n", .{});
    for (0..rmax) |r| {
        const row_start = r * cols;
        for (0..cmax) |col| {
            const v = output_items[row_start + col];
            if (col == 0) {
                std.debug.print("{d:.5}", .{v});
            } else {
                std.debug.print(" {d:.5}", .{v});
            }
        }
        std.debug.print("\n", .{});
    }

    return;
}

fn zeroBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    @memset(slice.data(), 0);
    return zml.Buffer.fromSlice(io, platform, slice);
}
