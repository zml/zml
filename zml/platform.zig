const std = @import("std");

const runtimes = @import("runtimes");
pub const Target = runtimes.Platform;
const stdx = @import("stdx");
const Mesh = @import("partitioning.zig").Mesh;
const Shape = @import("shape.zig").Shape;

const pjrt = @import("pjrtx.zig");

const log = std.log.scoped(.zml);

pub const available_targets = std.enums.values(Target);

/// Represents the physical arrangement of hardware devices.
/// It contains the devices ordered according to their physical topology and the shape of that topology.
pub const PhysicalMesh = struct {
    allocator: std.mem.Allocator,
    devices: []const *const pjrt.Device,
    shape: Shape,

    pub fn deinit(self: *PhysicalMesh) void {
        self.allocator.free(self.devices);
        self.* = undefined;
    }
};

// Optimal orderings for TPU collectives, from JAX.
const tpu_v5_lite_2x2_ring_order = [_]u8{ 0, 1, 3, 2 };
const tpu_v5_lite_2x2x2_iota_order = [_]u8{ 0, 4, 2, 6, 1, 5, 3, 7 };
// Same ring order for v5e tray, v6e tray, and v7x 2x4 tray
const tpu_tray_ring_order_8 = [_]u8{ 0, 1, 2, 3, 7, 6, 5, 4 };

/// Sorts devices primarily by their coordinates (lexicographically, reversed) and then by core_on_chip.
/// This matches JAX's z,y,x,core sorting for creating a physical mesh.
/// This function is allocation-free.
fn sortDevicesForPhysicalMesh(context: *const pjrt.Api, lhs_dev: *const pjrt.Device, rhs_dev: *const pjrt.Device) bool {
    const api: *const pjrt.Api = context;

    const lhs_desc = lhs_dev.getDescription(api);
    const rhs_desc = rhs_dev.getDescription(api);

    var lhs_coords: ?[]const i64 = null;
    var lhs_core: i64 = 0;
    for (lhs_desc.getAttributes(api)) |attr| {
        if (std.mem.eql(u8, attr.name(), "coords")) {
            lhs_coords = attr.inner.unnamed_0.int64_array_value[0..attr.inner.value_size];
        } else if (std.mem.eql(u8, attr.name(), "core_on_chip")) {
            lhs_core = attr.inner.unnamed_0.int64_value;
        }
    }

    var rhs_coords: ?[]const i64 = null;
    var rhs_core: i64 = 0;
    for (rhs_desc.getAttributes(api)) |attr| {
        if (std.mem.eql(u8, attr.name(), "coords")) {
            rhs_coords = attr.inner.unnamed_0.int64_array_value[0..attr.inner.value_size];
        } else if (std.mem.eql(u8, attr.name(), "core_on_chip")) {
            rhs_core = attr.inner.unnamed_0.int64_value;
        }
    }

    // TPUs are expected to have coordinates.
    // If not, we can't sort them topologically, so we treat them as equal and let the original order prevail.
    if (lhs_coords == null or rhs_coords == null) {
        return false;
    }

    const lhs_coords_slice = lhs_coords.?;
    const rhs_coords_slice = rhs_coords.?;

    stdx.debug.assert(lhs_coords_slice.len == rhs_coords_slice.len, "Devices have different coordinate ranks", .{});
    var i = lhs_coords_slice.len;
    while (i > 0) {
        i -= 1;
        if (lhs_coords_slice[i] != rhs_coords_slice[i]) {
            return lhs_coords_slice[i] < rhs_coords_slice[i];
        }
    }

    return lhs_core < rhs_core;
}

/// Sorts devices topologically for consistent ordering across platforms.
fn getTopologicallySortedDevices(
    allocator: std.mem.Allocator,
    devices: []const *const pjrt.Device,
    api: *const pjrt.Api,
) ![]*const pjrt.Device {
    const sorted_devices = try allocator.dupe(*const pjrt.Device, devices);
    std.sort.pdq(*const pjrt.Device, sorted_devices, api, sortDevicesForPhysicalMesh);
    return sorted_devices;
}

/// Creates a device mesh that arranges devices for optimal collective performance,
/// similar to `jax.experimental.mesh_utils.create_device_mesh`.
pub fn createDeviceMesh(
    allocator: std.mem.Allocator,
    logical_mesh: Mesh,
    platform: Platform,
) ![]const *const pjrt.Device {
    const api = platform.pjrt_api;
    const devices = platform.getDevices();

    const num_logical_devices: u64 = logical_mesh.topology.count();
    stdx.debug.assert(num_logical_devices <= devices.len, "Number of devices {} must be >= the product of logical_mesh_shape {f}", .{ devices.len, logical_mesh.topology });
    const active_devices = devices[0..@intCast(num_logical_devices)];

    if (active_devices.len == 0) return &.{};

    // Always sort devices topologically first to ensure a consistent base order
    // for mapping logical devices to physical devices across all platforms.
    const sequential_devices = try getTopologicallySortedDevices(allocator, active_devices, api);

    // For TPUs, we might apply an additional reordering for performance.
    // For other platforms, we will use the topologically sorted list directly.
    switch (platform.target) {
        .tpu => {
            // Keep the TPU-specific reordering logic, but operate on the sorted list.
            const first_device_kind = active_devices[0].getDescription(api).getKind(api);

            if (std.mem.eql(u8, first_device_kind, "TPU v5 lite") or std.mem.eql(u8, first_device_kind, "TPU v6 lite")) {
                if (try createTpuV5LiteMesh(allocator, logical_mesh, sequential_devices, api)) |ordered_devices| {
                    allocator.free(sequential_devices); // Free the intermediate sorted list
                    return ordered_devices;
                }
            } else if (std.mem.eql(u8, first_device_kind, "TPU7x")) {
                if (try createTpu7xMesh(allocator, logical_mesh, sequential_devices, api)) |ordered_devices| {
                    allocator.free(sequential_devices); // Free the intermediate sorted list
                    return ordered_devices;
                }
            }

            // Fallback for other TPU kinds (e.g., v2, v3, v4, v5p).
            log.warn("Using generic TPU physical mesh creation for {s}. For optimal performance, a specific handler may be needed.", .{first_device_kind});
            var physical_mesh = try getPhysicalTpuMesh(allocator, sequential_devices, api);
            defer physical_mesh.deinit();
            return try createDeviceMeshForNdTorus(allocator, physical_mesh, logical_mesh);
        },
        .cuda, .rocm, .cpu, .neuron => {
            // For other platforms, the topologically sorted list is the final list.
            return sequential_devices;
        },
    }
}

/// Creates a device mesh for TPU v5 lite and v6 lite, applying optimal orderings.
fn createTpuV5LiteMesh(
    allocator: std.mem.Allocator,
    logical_mesh: Mesh,
    devices: []const *const pjrt.Device,
    api: *const pjrt.Api,
) !?[]const *const pjrt.Device {
    const sequential_devices = try allocator.dupe(*const pjrt.Device, devices);
    defer allocator.free(sequential_devices);
    std.sort.pdq(*const pjrt.Device, sequential_devices, api, sortDevicesForPhysicalMesh);

    const reordered_devices = try allocator.alloc(*const pjrt.Device, devices.len);

    const order: ?[]const u8 = switch (devices.len) {
        4 => &tpu_v5_lite_2x2_ring_order,
        8 => &tpu_v5_lite_2x2x2_iota_order, // JAX uses iota order for v5e 2x2x2 cubes
        else => null,
    };

    if (order) |o| {
        for (o, 0..) |old_idx, new_idx| {
            reordered_devices[new_idx] = sequential_devices[old_idx];
        }
        // Reshaping is conceptual; the flat ordered array is what matters for XLA's device_assignment.
        _ = logical_mesh;
        return reordered_devices;
    }

    allocator.free(reordered_devices);
    return null;
}

/// Creates a device mesh for TPU v7x.
fn createTpu7xMesh(
    allocator: std.mem.Allocator,
    logical_mesh: Mesh,
    devices: []const *const pjrt.Device,
    api: *const pjrt.Api,
) !?[]const *const pjrt.Device {
    const sequential_devices = try allocator.dupe(*const pjrt.Device, devices);
    defer allocator.free(sequential_devices);
    std.sort.pdq(*const pjrt.Device, sequential_devices, api, sortDevicesForPhysicalMesh);

    if (devices.len == 8) {
        const reordered_devices = try allocator.alloc(*const pjrt.Device, devices.len);
        for (tpu_tray_ring_order_8, 0..) |old_idx, new_idx| {
            reordered_devices[new_idx] = sequential_devices[old_idx];
        }
        _ = logical_mesh;
        return reordered_devices;
    }

    return null;
}

/// Constructs a PhysicalMesh representing the N-D torus of a TPU slice.
fn getPhysicalTpuMesh(allocator: std.mem.Allocator, devices: []const *const pjrt.Device, api: *const pjrt.Api) !PhysicalMesh {
    stdx.debug.assert(devices.len > 0, "Cannot create physical mesh from zero devices", .{});
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    _ = arena; // autofix

    var min_coords: [3]i64 = .{ std.math.maxInt(i64), std.math.maxInt(i64), std.math.maxInt(i64) };
    var max_coords: [3]i64 = .{ std.math.minInt(i64), std.math.minInt(i64), std.math.minInt(i64) };
    var min_core: i64 = std.math.maxInt(i64);
    var max_core: i64 = std.math.minInt(i64);

    for (devices) |device| {
        const desc = device.getDescription(api);
        const attrs = desc.getAttributes(api);
        for (attrs) |attr| {
            if (std.mem.eql(u8, attr.name(), "coords")) {
                const coords = attr.inner.unnamed_0.int64_array_value[0..attr.inner.value_size];
                stdx.debug.assert(coords.len == 3, "Expected 3D coordinates for TPU device, got {any}", .{coords});
                for (0..3) |i| {
                    min_coords[i] = @min(min_coords[i], coords[i]);
                    max_coords[i] = @max(max_coords[i], coords[i]);
                }
            } else if (std.mem.eql(u8, attr.name(), "core_on_chip")) {
                min_core = @min(min_core, attr.inner.unnamed_0.int64_value);
                max_core = @max(max_core, attr.inner.unnamed_0.int64_value);
            }
        }
    }

    const dims: [3]u64 = .{
        @intCast(max_coords[0] - min_coords[0] + 1),
        @intCast(max_coords[1] - min_coords[1] + 1),
        @intCast(max_coords[2] - min_coords[2] + 1),
    };
    const cores_per_chip: u64 = @intCast(max_core - min_core + 1);
    stdx.debug.assert(devices.len == dims[0] * dims[1] * dims[2] * cores_per_chip, "Device list does not form a contiguous cuboid.", .{});

    var physical_shape: Shape = undefined;
    var physical_dims: Shape.DimsArray = .{};
    if (cores_per_chip > 1) {
        try physical_dims.appendSlice(&.{ @intCast(dims[0]), @intCast(dims[1]), @intCast(cores_per_chip) });
        physical_shape = Shape.init(physical_dims.constSlice(), .u8).withTags(.{ .x, .y, .c });
    } else {
        try physical_dims.appendSlice(&.{ @intCast(dims[0]), @intCast(dims[1]), @intCast(dims[2]) });
        physical_shape = Shape.init(physical_dims.constSlice(), .u8).withTags(.{ .x, .y, .z });
    }

    const ordered_devices = try allocator.alloc(*const pjrt.Device, devices.len);
    // @memset(ordered_devices, null);

    for (devices) |device| {
        const desc = device.getDescription(api);
        const attrs = desc.getAttributes(api);
        var coords: [3]i64 = undefined;
        var core_on_chip: i64 = 0;
        for (attrs) |attr| {
            if (std.mem.eql(u8, attr.name(), "coords")) {
                const coords_slice = attr.inner.unnamed_0.int64_array_value[0..attr.inner.value_size];
                @memcpy(coords[0..coords_slice.len], coords_slice);
            } else if (std.mem.eql(u8, attr.name(), "core_on_chip")) {
                core_on_chip = attr.inner.unnamed_0.int64_value;
            }
        }
        const x = @as(usize, @intCast(coords[0] - min_coords[0]));
        const y = @as(usize, @intCast(coords[1] - min_coords[1]));
        const z = @as(usize, @intCast(coords[2] - min_coords[2]));
        const c = @as(usize, @intCast(core_on_chip - min_core));

        const index = if (cores_per_chip > 1) (x * @as(usize, @intCast(dims[1])) * @as(usize, @intCast(cores_per_chip))) + (y * @as(usize, @intCast(cores_per_chip))) + c else (x * @as(usize, @intCast(dims[1])) * @as(usize, @intCast(dims[2]))) + (y * @as(usize, @intCast(dims[2]))) + z;
        stdx.debug.assert(index < ordered_devices.len, "Index out of bounds", .{});
        // stdx.debug.assert(ordered_devices[index] == null, "Duplicate device position in physical mesh", .{});
        ordered_devices[index] = device;
    }

    return PhysicalMesh{
        .allocator = allocator,
        .devices = ordered_devices,
        .shape = physical_shape,
    };
}

/// Maps a logical mesh onto a physical N-D torus mesh.
fn createDeviceMeshForNdTorus(
    allocator: std.mem.Allocator,
    physical_mesh: PhysicalMesh,
    logical_mesh: Mesh,
) ![]const *const pjrt.Device {
    // This is a complex combinatorial problem. For now, we implement a simplified greedy approach
    // that assumes logical axes map to single physical axes. A full implementation would explore
    // combinations and splitting physical axes.
    // TODO: Implement the full JAX axis assignment algorithm.
    log.warn("Using simplified logical to physical mesh mapping. Performance may not be optimal.", .{});

    var physical_axes = physical_mesh.shape._dims;
    var logical_axes = logical_mesh.topology._dims;
    std.sort.pdq(i64, physical_axes.slice(), {}, std.sort.desc(i64));
    std.sort.pdq(i64, logical_axes.slice(), {}, std.sort.desc(i64));

    if (!std.mem.eql(i64, physical_axes.constSlice(), logical_axes.constSlice())) {
        log.err("Physical mesh shape {any} cannot be mapped to logical mesh shape {any} with the current simplified algorithm.", .{
            physical_mesh.shape,
            logical_mesh.topology,
        });
        return error.MeshMappingNotSupported;
    }

    // Since shapes are identical after sorting, a simple copy is sufficient for this simplified version.
    return allocator.dupe(*const pjrt.Device, physical_mesh.devices);
}

pub const CompilationOptions = struct {
    xla_dump_to: ?[]const u8 = null,
    xla_dump_fusion_visualization: bool = false,
    xla_dump_hlo_pass_re: ?[]const u8 = null,
    sharding_enabled: bool = false,
    sharding_axes: stdx.BoundedArray([*:0]const u8, 8) = .{},
};

pub const Platform = struct {
    target: Target,
    pjrt_api: *const pjrt.Api,
    pjrt_client: *pjrt.Client,
    compilation_options: CompilationOptions = .{},

    pub const MAX_NUM_DEVICES: u8 = 32;
    pub const CreateOptions = _CreateOptions;

    pub fn init(target: Target, api: *const pjrt.Api, options: CreateOptions) !Platform {
        var named_values_buf: [16]pjrt.NamedValue = undefined;
        const pjrt_client = try pjrt.Client.init(api, options.toNamedValues(target, &named_values_buf));
        const true_num_devices = pjrt_client.getAddressableDevices(api).len;
        if (true_num_devices > MAX_NUM_DEVICES) {
            log.warn("platform {} got {} devices, but ZML only support up to {} devices. Some devices won't be used.", .{ target, true_num_devices, MAX_NUM_DEVICES });
        }
        return .{
            .target = target,
            .pjrt_api = api,
            .pjrt_client = pjrt_client,
            .compilation_options = .{},
        };
    }

    pub fn getDevices(self: Platform) []const *const pjrt.Device {
        const all_devices = self.pjrt_client.getAddressableDevices(self.pjrt_api);
        if (all_devices.len > MAX_NUM_DEVICES) {
            return all_devices[0..MAX_NUM_DEVICES];
        }
        return all_devices;
    }

    pub const Sharding = struct { num_replicas: u8, num_partitions: u8 };

    pub fn sharding(self: Platform) Sharding {
        // replicas run the same function but with different inputs,
        // while partitions contribute to one evaluation over a shared input.
        // Inside an inference process, we generally don't want replicas,
        // as it's best to fully isolate replicas on different processes.
        // For now we hardcode num_replicas = 1.
        const num_devices: u8 = @intCast(self.getDevices().len);
        return if (self.compilation_options.sharding_enabled)
            .{ .num_replicas = 1, .num_partitions = num_devices }
        else
            .{ .num_replicas = 1, .num_partitions = 1 };
    }

    pub fn withCompilationOptions(self: Platform, opts: CompilationOptions) Platform {
        var res = self;
        res.compilation_options = opts;
        return res;
    }

    pub fn registerFFIType(self: Platform, comptime T: type) !void {
        if (self.pjrt_api.ffi()) |ffi| {
            if (!@hasDecl(T, "type_id")) {
                stdx.debug.panic("registerFFIType requires type {s} to have a `type_id` i64 field ", .{@typeName(T)});
            }
            try ffi.registerTypeId(self.pjrt_api, T);
        } else {
            stdx.debug.panic("registerFFIType is not available for target {s}", .{@tagName(self.target)});
        }
    }

    pub fn deinit(self: *Platform) void {
        self.pjrt_client.deinit(self.pjrt_api);
    }
};

const _CreateOptions = struct {
    // XLA CPU client doesn't read options
    // https://github.com/openxla/xla/blob/42496a28c374bd35f493cc5dbde74805407245dc/xla/pjrt/c/pjrt_c_api_cpu_internal.cc#L33-L46
    cpu: struct {} = .{},

    // bump memory fraction from XLA defaults of 75% to 90%.
    // Even on a 8GB GPU it should leave enough space for the Cuda driver
    // https://github.com/openxla/xla/blob/3e87afa11a865cf91137522492918ad18bfe5b7c/xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h#L25-L60
    cuda: Cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
    rocm: struct {} = .{},
    tpu: struct {} = .{},
    neuron: struct {} = .{},

    pub const Cuda = struct {
        allocator: Allocator = .{ .bfc = .{} },
        // TODO support all of https://github.com/openxla/xla/blob/3d31c48c719d331d432132b3e0c2c5ce52650675/xla/pjrt/c/pjrt_c_api_gpu_internal.cc#L76-L86
        // visible_devices: []const i64 = &.{},
        // node_id
        // num_nodes
        // enable_mock_nccl
        // mock_gpu_topology

        pub const Allocator = union(enum) {
            /// "Best-Fit with Coalescing" algorithm
            bfc: Options,
            /// use cudaMallocAsync
            async: Options,
            /// use raw cuMalloc
            platform,

            pub const Options = struct {
                preallocate: bool = true,
                memory_fraction: f32 = 0.90,
                collective_memory_size_mb: u32 = 0,
            };
        };

        pub fn writeNamedValues(self: Cuda, values: *std.ArrayListUnmanaged(pjrt.NamedValue)) void {
            switch (self.allocator) {
                .platform => {
                    values.appendAssumeCapacity(pjrt.NamedValue.fromString("allocator", "platform"));
                },
                .bfc, .async => |opt| {
                    values.appendAssumeCapacity(pjrt.NamedValue.from("allocator", self.allocator));
                    values.appendAssumeCapacity(pjrt.NamedValue.from("preallocate", opt.preallocate));
                    if (opt.memory_fraction > 0) {
                        values.appendAssumeCapacity(pjrt.NamedValue.from("memory_fraction", opt.memory_fraction));
                    }
                    if (opt.collective_memory_size_mb > 0) {
                        const collective = @as(i64, opt.collective_memory_size_mb) * 1024 * 1024;
                        values.appendAssumeCapacity(pjrt.NamedValue.from("collective_memory_size", collective));
                    }
                },
            }
        }
    };

    pub fn toNamedValues(self: _CreateOptions, target: Target, out: []pjrt.NamedValue) []pjrt.NamedValue {
        var values = std.ArrayListUnmanaged(pjrt.NamedValue).fromOwnedSlice(out);
        values.shrinkRetainingCapacity(0);
        switch (target) {
            .cuda => self.cuda.writeNamedValues(&values),
            inline else => |t| {
                stdx.debug.assertComptime(@hasField(_CreateOptions, @tagName(t)), "zml.platform.CreateOptions doesn't list target {s}", .{@tagName(t)});
                const options = @field(self, @tagName(t));
                stdx.debug.assertComptime(@sizeOf(@TypeOf(options)) == 0, "zml.platform.CreateOptions.{s} is discarded", .{@tagName(t)});
            },
        }
        return values.items;
    }
};
