const std = @import("std");

const c_interface = @import("c");

pub const std_options: std.Options = .{
    .log_level = .info,
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

fn toPosixPathW(file_path: []const u8) error{NameTooLong}![std.posix.PATH_MAX - 1:0]c_interface.wchar_t {
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
    const env_base_keys = [_][:0]const u8{
        "RUNFILES_DIR",
        "RUNFILES_DIRECTORY",
    };
    const libpython_paths = [_][]const u8{
        "_solib_k8/_U_A_Arules_Upython++python+python_U3_U12_Ux86_U64-unknown-linux-gnu_S_S_Clibpython___Ulib/libpython3.12.so",
        "_solib_k8/_U_A_Arules_Upython++python+python_U3_U12_Uaarch64-unknown-linux-gnu_S_S_Clibpython___Ulib/libpython3.12.so",
    };

    const SearchState = struct {
        fn probe(io_inner: anytype, base: []const u8, libpython_paths_inner: []const []const u8) ?[]u8 {
            for (libpython_paths_inner) |libpython_rel| {
                var candidate_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
                const candidate = std.fmt.bufPrint(&candidate_buf, "{s}/{s}", .{ base, libpython_rel }) catch continue;

                var file = std.Io.Dir.openFile(.cwd(), io_inner, candidate, .{}) catch continue;
                defer file.close(io_inner);

                var real_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
                const real_path_len = file.realPath(io_inner, &real_path_buf) catch continue;
                const real_path = real_path_buf[0..real_path_len];
                const lib_dir_idx = std.mem.lastIndexOf(u8, real_path, "/lib/") orelse continue;
                return std.heap.c_allocator.dupe(u8, real_path[0..lib_dir_idx]) catch null;
            }
            return null;
        }
    };

    for (env_base_keys) |key| {
        const env_base_c = std.c.getenv(key) orelse continue;
        const env_base = std.mem.span(env_base_c);
        if (SearchState.probe(io, env_base, &libpython_paths)) |p| return p;

        var base_buf1: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const base1 = std.fmt.bufPrint(&base_buf1, "{s}/_main", .{env_base}) catch continue;
        if (SearchState.probe(io, base1, &libpython_paths)) |p| return p;

        var base_buf2: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const base2 = std.fmt.bufPrint(&base_buf2, "{s}/_main/bazel-out/linux_amd64-dbg/bin", .{env_base}) catch continue;
        if (SearchState.probe(io, base2, &libpython_paths)) |p| return p;

        var base_buf3: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const base3 = std.fmt.bufPrint(&base_buf3, "{s}/bazel-out/linux_amd64-dbg/bin", .{env_base}) catch continue;
        if (SearchState.probe(io, base3, &libpython_paths)) |p| return p;
    }

    for (base_paths) |base| {
        if (SearchState.probe(io, base, &libpython_paths)) |p| return p;
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
        \\    roots.append(os.path.realpath(os.path.join(exe_dir, "..", "ttir_compile_sandbox.runfiles")))
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
        \\            os.path.join(root, "_main", "examples", "ttir_sandbox_prefill"),
        \\            os.path.join(root, "examples", "ttir_sandbox_prefill"),
        \\            os.path.join(root, "site-packages"),
        \\        ]
        \\    )
        \\    for p in glob.glob(os.path.join(root, "**", "site-packages"), recursive=True):
        \\        cands.append(p)
        \\    for p in glob.glob(os.path.join(root, "**", "ttir_compile.py"), recursive=True):
        \\        cands.append(os.path.dirname(p))
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
    c_interface.PyConfig_InitIsolatedConfig(&config);
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

fn callCompileFunction(allocator: std.mem.Allocator, io: anytype, function_name: []const u8, args_json: []const u8) ![:0]u8 {
    try ensurePythonInitialized(io);

    const module = c_interface.PyImport_ImportModule("ttir_compile");
    if (module == null) {
        checkPythonError();
        return error.PythonModuleImportFailed;
    }
    defer c_interface.Py_DecRef(module);

    const compile_func = c_interface.PyObject_GetAttrString(module, function_name.ptr);
    if (compile_func == null) {
        checkPythonError();
        return error.PythonAttributeNotFound;
    }
    defer c_interface.Py_DecRef(compile_func);

    const py_params = c_interface.PyUnicode_FromStringAndSize(args_json.ptr, @intCast(args_json.len));
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

pub fn getPrefillAttentionTtir(allocator: std.mem.Allocator, io: anytype, args_json: []const u8) ![:0]u8 {
    return callCompileFunction(allocator, io, "compile_prefill_attention_ttir", args_json);
}
