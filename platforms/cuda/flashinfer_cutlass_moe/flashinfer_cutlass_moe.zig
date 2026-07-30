const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const builtin = @import("builtin");
const c = @import("c");

pub const C = c;
pub const Status = c.zml_fi_cutlass_moe_status;
pub const DType = c.zml_fi_cutlass_moe_dtype;
pub const Activation = c.zml_fi_cutlass_moe_activation;
pub const RunnerOptions = c.zml_fi_cutlass_moe_runner_options;
pub const Context = c.zml_fi_cutlass_moe_context;
pub const Io = c.zml_fi_cutlass_moe_io;
pub const Workspace = c.zml_fi_cutlass_moe_workspace;
pub const WorkspaceRequirements = c.zml_fi_cutlass_moe_workspace_requirements;
pub const Runner = c.zml_fi_cutlass_moe_runner;

pub const abiVersionExpected = c.ZML_FI_CUTLASS_MOE_ABI_VERSION;
pub const autoTactic = c.ZML_FI_CUTLASS_MOE_AUTO_TACTIC;

const AbiVersionFn = @TypeOf(&c.zml_fi_cutlass_moe_abi_version);
const CompiledSmFn = @TypeOf(&c.zml_fi_cutlass_moe_compiled_sm);
const LastErrorFn = @TypeOf(&c.zml_fi_cutlass_moe_last_error);
const DeviceIsSupportedFn = @TypeOf(&c.zml_fi_cutlass_moe_device_is_supported);
const RunnerCreateFn = @TypeOf(&c.zml_fi_cutlass_moe_runner_create);
const RunnerDestroyFn = @TypeOf(&c.zml_fi_cutlass_moe_runner_destroy);
const GetTacticCountsFn = @TypeOf(&c.zml_fi_cutlass_moe_get_tactic_counts);
const GetTacticOccupancyFn = @TypeOf(&c.zml_fi_cutlass_moe_get_tactic_occupancy);
const GetWorkspaceRequirementsFn = @TypeOf(&c.zml_fi_cutlass_moe_get_workspace_requirements);
const RunFn = @TypeOf(&c.zml_fi_cutlass_moe_run);

pub const Api = struct {
    library: std.DynLib,
    abiVersion: AbiVersionFn,
    compiledSm: CompiledSmFn,
    lastError: LastErrorFn,
    deviceIsSupported: DeviceIsSupportedFn,
    runnerCreate: RunnerCreateFn,
    runnerDestroy: RunnerDestroyFn,
    getTacticCounts: GetTacticCountsFn,
    getTacticOccupancy: GetTacticOccupancyFn,
    getWorkspaceRequirements: GetWorkspaceRequirementsFn,
    run: RunFn,
};

const architectures = [_]u16{ 90, 100, 120 };
var apis: [architectures.len]?Api = @splat(null);
var isLoaded = false;

fn libraryRunfile(sm: u16, buffer: []u8) ![]const u8 {
    return try std.fmt.bufPrint(
        buffer,
        "flashinfer_cutlass_moe_linux_amd64/lib/libflashinfer_cutlass_moe_sm{d}.so",
        .{sm},
    );
}

fn loadApi(path: []const u8) !Api {
    var library = std.DynLib.open(path) catch |err| {
        std.log.err("Failed to open FlashInfer CUTLASS MoE library {s}: {any}", .{ path, err });
        return err;
    };
    errdefer library.close();

    const api: Api = .{
        .library = library,
        .abiVersion = library.lookup(AbiVersionFn, "zml_fi_cutlass_moe_abi_version") orelse
            return error.SymbolNotFound,
        .compiledSm = library.lookup(CompiledSmFn, "zml_fi_cutlass_moe_compiled_sm") orelse
            return error.SymbolNotFound,
        .lastError = library.lookup(LastErrorFn, "zml_fi_cutlass_moe_last_error") orelse
            return error.SymbolNotFound,
        .deviceIsSupported = library.lookup(DeviceIsSupportedFn, "zml_fi_cutlass_moe_device_is_supported") orelse
            return error.SymbolNotFound,
        .runnerCreate = library.lookup(RunnerCreateFn, "zml_fi_cutlass_moe_runner_create") orelse
            return error.SymbolNotFound,
        .runnerDestroy = library.lookup(RunnerDestroyFn, "zml_fi_cutlass_moe_runner_destroy") orelse
            return error.SymbolNotFound,
        .getTacticCounts = library.lookup(GetTacticCountsFn, "zml_fi_cutlass_moe_get_tactic_counts") orelse
            return error.SymbolNotFound,
        .getTacticOccupancy = library.lookup(GetTacticOccupancyFn, "zml_fi_cutlass_moe_get_tactic_occupancy") orelse
            return error.SymbolNotFound,
        .getWorkspaceRequirements = library.lookup(
            GetWorkspaceRequirementsFn,
            "zml_fi_cutlass_moe_get_workspace_requirements",
        ) orelse return error.SymbolNotFound,
        .run = library.lookup(RunFn, "zml_fi_cutlass_moe_run") orelse
            return error.SymbolNotFound,
    };
    if (api.abiVersion() != abiVersionExpected) return error.AbiVersionMismatch;
    return api;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !void {
    _ = allocator;
    if (isLoaded) return;
    if (builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) {
        return error.UnsupportedPlatform;
    }

    const runfiles = try bazel.runfiles(bazel_builtin.current_repository);
    for (architectures, 0..) |sm, index| {
        var runfileNameBuffer: [160]u8 = undefined;
        const runfileName = try libraryRunfile(sm, &runfileNameBuffer);
        var runfilePathBuffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const runfilePath = (try runfiles.rlocation(runfileName, &runfilePathBuffer)) orelse
            return error.NotFound;

        // Preserve the producer library's $ORIGIN-relative dependency lookup
        // when the development repository exposes it through Bazel symlinks.
        var canonicalBuffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const canonicalLength = if (std.fs.path.isAbsolute(runfilePath))
            try std.Io.Dir.realPathFileAbsolute(io, runfilePath, &canonicalBuffer)
        else
            try std.Io.Dir.cwd().realPathFile(io, runfilePath, &canonicalBuffer);
        apis[index] = try loadApi(canonicalBuffer[0..canonicalLength]);
    }
    isLoaded = true;
}

pub fn apiForDevice(device: i32) !*Api {
    if (!isLoaded) return error.BackendNotLoaded;
    for (&apis) |*maybeApi| {
        if (maybeApi.*) |*api| {
            var supported: u8 = 0;
            const status = api.deviceIsSupported(device, &supported);
            if (status != c.ZML_FI_CUTLASS_MOE_STATUS_SUCCESS) {
                if (api.lastError()) |message| {
                    std.log.err("FlashInfer CUTLASS MoE device probe failed: {s}", .{std.mem.span(message)});
                }
                return error.DeviceProbeFailed;
            }
            if (supported != 0) return api;
        }
    }
    return error.UnsupportedArchitecture;
}

test "load architecture-specific FlashInfer CUTLASS MoE libraries" {
    if (builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return;
    try load(std.testing.allocator, std.testing.io);
    for (apis, architectures) |maybeApi, sm| {
        const api = maybeApi orelse return error.NotLoaded;
        try std.testing.expectEqual(@as(i32, sm), api.compiledSm());
    }
}
