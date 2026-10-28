const std = @import("std");
const topology = @import("dump_context_proto").proto;
const hlo = @import("@xla//xla/service:hlo_proto").proto;

pub const Attribute = struct { key: []const u8, value: []const u8 };

pub const Context = struct {
    device_count: ?u64 = null,
    attrs: std.ArrayList(Attribute) = .empty,
    optimized: ?hlo.HloModuleProto = null,
    optimized_path: []const u8 = "",

    fn add(self: *Context, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        try self.attrs.append(allocator, .{ .key = key, .value = value });
    }

    // Optional diagnostics must not prevent viewing an otherwise valid HLO.
    pub fn load(self: *Context, allocator: std.mem.Allocator, io: std.Io, input: []const u8, module_name: []const u8) !void {
        const stem = dumpStem(input) orelse return;
        const topo_path = try std.fmt.allocPrint(allocator, "{s}/topology.pb", .{stem});
        self.loadTopology(allocator, io, topo_path) catch |err| {
            if (err != error.FileNotFound) try self.add(allocator, "topology diagnostic", try std.fmt.allocPrint(allocator, "{s}: {s}", .{ topo_path, @errorName(err) }));
        };
        const dir_path = std.Io.Dir.path.dirname(stem) orelse ".";
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        const prefix = try std.fmt.allocPrint(allocator, "{s}.", .{std.Io.Dir.path.basename(stem)});
        var candidate: ?[]const u8 = null;
        var ambiguous = false;
        while (try it.next(io)) |entry| {
            if (!std.mem.startsWith(u8, entry.name, prefix) or !std.mem.endsWith(u8, entry.name, "gpu_after_optimizations.hlo.pb")) continue;
            if (candidate != null) ambiguous = true;
            candidate = try std.Io.Dir.path.join(allocator, &.{ dir_path, entry.name });
        }
        if (ambiguous) {
            try self.add(allocator, "optimized HLO diagnostic", "Multiple optimized GPU siblings; no automatic match chosen");
        } else if (candidate) |path| {
            self.loadOptimized(allocator, io, path, module_name) catch |err| {
                try self.add(allocator, "optimized HLO diagnostic", try std.fmt.allocPrint(allocator, "{s}: {s}", .{ path, @errorName(err) }));
            };
        }
        const thunk_path = try std.fmt.allocPrint(allocator, "{s}.thunk_sequence_after_thunk_passes.txt", .{stem});
        self.loadThunks(allocator, io, thunk_path) catch |err| {
            if (err != error.FileNotFound) try self.add(allocator, "thunk diagnostic", @errorName(err));
        };
    }

    fn loadTopology(self: *Context, allocator: std.mem.Allocator, io: std.Io, path: []const u8) !void {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var buffer: [8192]u8 = undefined;
        var reader = file.reader(io, &buffer);
        const proto = try topology.PjRtTopologyDescriptionProto.decode(&reader.interface, allocator);
        const specific = proto.platform_specific_topology orelse return error.MissingPlatformTopology;
        if (!std.mem.eql(u8, specific.type_url, "type.googleapis.com/xla.GpuTopologyProto")) return error.UnsupportedTopology;
        var inner: std.Io.Reader = .fixed(specific.value);
        const gpu = try topology.GpuTopologyProto.decode(&inner, allocator);
        if (gpu.num_partitions > 0 and gpu.num_hosts_per_partition > 0 and gpu.num_devices_per_host > 0) {
            const hosts = try std.math.mul(u64, @intCast(gpu.num_partitions), @intCast(gpu.num_hosts_per_partition));
            self.device_count = try std.math.mul(u64, hosts, @intCast(gpu.num_devices_per_host));
        }
        try self.add(allocator, "topology source", path);
        try self.add(allocator, "GPU topology (PJRT)", try std.fmt.allocPrint(allocator, "{s}; {d} topology partitions, {d} hosts/partition, {d} devices/host. PJRT partitions are not HLO SPMD partitions; this is not a physical link map.", .{ gpu.platform_version, gpu.num_partitions, gpu.num_hosts_per_partition, gpu.num_devices_per_host }));
        if (gpu.gpu_target_config) |target| {
            if (target.gpu_device_info) |device| {
                try self.add(allocator, "GPU target", try std.fmt.allocPrint(allocator, "{s}; runtime {s}; driver {s}", .{ device.name, device.runtime_version, device.driver_version }));
                try self.add(allocator, "target PCIe bandwidth bytes/s", try std.fmt.allocPrint(allocator, "{d} (device capability; not measured collective/link bandwidth)", .{device.pcie_bandwidth}));
            }
        }
    }

    fn loadOptimized(self: *Context, allocator: std.mem.Allocator, io: std.Io, path: []const u8, module_name: []const u8) !void {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var buffer: [8192]u8 = undefined;
        var reader = file.reader(io, &buffer);
        const proto = try hlo.HloProto.decode(&reader.interface, allocator);
        const module = proto.hlo_module orelse return error.MissingModule;
        if (!std.mem.eql(u8, module.name, module_name)) return error.ModuleNameMismatch;
        self.optimized = module;
        self.optimized_path = path;
        try self.add(allocator, "optimized HLO source", path);
    }

    fn loadThunks(self: *Context, allocator: std.mem.Allocator, io: std.Io, path: []const u8) !void {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var buffer: [8192]u8 = undefined;
        var reader = file.reader(io, &buffer);
        const content = try reader.interface.allocRemaining(allocator, .limited(1024 * 1024));
        var selected: std.Io.Writer.Allocating = .init(allocator);
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            inline for (.{ "kAllReduce", "kAllGather", "kReduceScatter", "kAllToAll", "kCollective" }) |kind| {
                if (std.mem.indexOf(u8, line, kind) != null) {
                    try selected.writer.print("{s}\n", .{line});
                    break;
                }
            }
        }
        try self.add(allocator, "thunk sequence source", path);
        try self.add(allocator, "compiled collective thunks (module-wide)", if (selected.written().len > 0) try selected.toOwnedSlice() else "No recognized collective thunks recorded");
    }

    pub fn writeAttrs(self: *const Context, allocator: std.mem.Allocator, list: anytype, instr: hlo.HloInstructionProto) !void {
        for (self.attrs.items) |attr| try list.write(attr);
        if (self.optimized) |module| {
            const match = matchCollective(module, instr);
            switch (match) {
                .unique => |other| {
                    try list.write(Attribute{ .key = "optimized collective", .value = try std.fmt.allocPrint(allocator, "{s} ({s}), matched by opcode and channel {d}", .{ other.name, other.opcode, instr.channel_id }) });
                    if (other.backend_config.len > 0) try list.write(Attribute{ .key = "optimized backend config", .value = other.backend_config });
                },
                .absent => try list.write(Attribute{ .key = "optimized collective", .value = "No same-opcode/channel collective in final GPU HLO. It may have been eliminated, combined or rewritten; this stage's traffic estimate must not be counted as executable traffic." }),
                .ambiguous => try list.write(Attribute{ .key = "optimized collective", .value = "Ambiguous or unavailable channel match; no implementation attributed" }),
            }
        }
        try list.write(Attribute{ .key = "runtime algorithm / protocol", .value = "Not recorded by these compiler dumps. For NCCL-backed operations, NCCL chooses at runtime; capture NCCL TUNING/COLL diagnostics or a profiler trace for actual selection." });
        try list.write(Attribute{ .key = "physical transport topology", .value = "PJRT device grouping and target bandwidth do not identify PCIe/NVLink/NIC routes. Capture NCCL GRAPH diagnostics and NCCL_TOPO_DUMP_FILE on the execution host." });
    }
};

const Match = union(enum) { unique: *const hlo.HloInstructionProto, absent, ambiguous };
fn matchCollective(module: hlo.HloModuleProto, instr: hlo.HloInstructionProto) Match {
    if (instr.channel_id <= 0) return .ambiguous;
    var found: ?*const hlo.HloInstructionProto = null;
    for (module.computations.items) |comp| {
        for (comp.instructions.items) |*candidate| {
            if (candidate.channel_id != instr.channel_id or !std.mem.eql(u8, candidate.opcode, instr.opcode)) continue;
            if (found != null) return .ambiguous;
            found = candidate;
        }
    }
    return if (found) |value| .{ .unique = value } else .absent;
}

fn dumpStem(path: []const u8) ?[]const u8 {
    inline for (.{ ".after_spmd_partitioner.hlo.pb", ".before_optimizations.hlo.pb", ".before_config_assignment.hlo.pb" }) |suffix| {
        if (std.mem.endsWith(u8, path, suffix)) return path[0 .. path.len - suffix.len];
    }
    if (std.mem.endsWith(u8, path, "gpu_after_optimizations.hlo.pb")) {
        const name = std.Io.Dir.path.basename(path);
        if (std.mem.lastIndexOf(u8, name, ".sm_")) |i| return path[0 .. path.len - name.len + i];
    }
    return null;
}

test "dump stem keeps module identity" {
    try std.testing.expectEqualStrings("/tmp/module_2797.llama_layer", dumpStem("/tmp/module_2797.llama_layer.sm_8.9_gpu_after_optimizations.hlo.pb").?);
    try std.testing.expect(dumpStem("/tmp/arbitrary.hlo.pb") == null);
}

test "collective matches reject missing and reused channels" {
    var instructions = [_]hlo.HloInstructionProto{ .{ .opcode = "all-reduce", .channel_id = 3 }, .{ .opcode = "all-reduce", .channel_id = 4 } };
    var computations = [_]hlo.HloComputationProto{.{ .instructions = .{ .items = &instructions, .capacity = 2 } }};
    const module: hlo.HloModuleProto = .{ .computations = .{ .items = &computations, .capacity = 1 } };
    try std.testing.expect(matchCollective(module, .{ .opcode = "all-reduce", .channel_id = 3 }) == .unique);
    try std.testing.expect(matchCollective(module, .{ .opcode = "all-gather", .channel_id = 1 }) == .absent);
    instructions[1].channel_id = 3;
    try std.testing.expect(matchCollective(module, .{ .opcode = "all-reduce", .channel_id = 3 }) == .ambiguous);
}
