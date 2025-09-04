const std = @import("std");

const runtimes = @import("runtimes");
pub const Target = runtimes.Platform;
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const custom_call = @import("custom_call.zig");
const pjrt = @import("pjrtx.zig");
const Shape = @import("shape.zig").Shape;

const log = std.log.scoped(.zml);

pub const available_targets = std.enums.values(Target);

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

    // This make the pjrt struct quite fat, but is only used during compilation.
    // TODO: Reconsider having it here, and maybe pass explicitly to compile,
    // or create an intermediary struct:
    // `const comp = platform.compiler(compile_opts); const exe = comp.compile(...);`
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

    pub fn registerCustomCall(self: Platform, comptime CustomOp: type) pjrt.ApiError!void {
        stdx.debug.assertComptime(@hasDecl(CustomOp, "call"), "{} must have a call method", .{CustomOp});
        stdx.debug.assertComptime(@hasDecl(CustomOp, "type_id") and @TypeOf(CustomOp.type_id) == pjrt.ffi.TypeId, "{} must have a field `pub var type_id: pjrt.ffi.TypeId`", .{CustomOp});
        stdx.debug.assertComptime(@hasDecl(CustomOp, "custom_call_options") and @TypeOf(CustomOp.custom_call_options) == custom_call.CustomCallOptions, "{} must have a field `pub const custom_call_options: zml.CustomCallOptions`", .{CustomOp});

        const ffi = self.pjrt_api.ffi() orelse return error.Unavailable;
        const target_name = "zml$" ++ @typeName(CustomOp);

        const proxy = custom_call.proxy(CustomOp);
        CustomOp.type_id = try ffi.registerTypeId(self.pjrt_api, @typeName(CustomOp));
        try ffi.register(self.pjrt_api, target_name, @tagName(self.target), &proxy, CustomOp.custom_call_options.handler_traits);
        log.info("Registered custom call {} with target name \"{s}\"", .{ CustomOp, target_name });
    }

    pub fn deinit(self: *Platform) void {
        self.pjrt_client.deinit(self.pjrt_api);
    }

    pub fn batchedTransfer(platform: Platform, allocator: std.mem.Allocator, shapes: []const Shape, memory: pjrt.Memory.Kind) !TransferManager {
        return try TransferManager.init(allocator, platform, shapes, memory);
    }
};

const _CreateOptions = struct {
    cpu: Cpu = .{ .device_count = 4 },

    // bump memory fraction from XLA defaults of 75% to 90%.
    // Even on a 8GB GPU it should leave enough space for the Cuda driver
    // https://github.com/openxla/xla/blob/3e87afa11a865cf91137522492918ad18bfe5b7c/xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h#L25-L60
    cuda: Cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
    rocm: struct {} = .{},
    tpu: struct {} = .{},
    neuron: struct {} = .{},

    pub const Cpu = struct {
        device_count: u32,

        fn writeNamedValues(self: Cpu, values: *std.ArrayListUnmanaged(pjrt.NamedValue)) void {
            values.appendAssumeCapacity(pjrt.NamedValue.from("cpu_device_count", @as(i64, self.device_count)));
        }
    };

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

        fn writeNamedValues(self: Cuda, values: *std.ArrayListUnmanaged(pjrt.NamedValue)) void {
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
            .cpu => self.cpu.writeNamedValues(&values),
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

/// Allows to create Buffer and incrementally populate them.
pub const TransferManager = struct {
    // TODO: Consider providing a shape oriented API.
    // TODO: Handle replication. If target buffer is replicated transferData should handle calling the different devices.

    num_devices: u16,
    pjrt_api: *const pjrt.Api,

    buffers: []Buffer,
    managers: [*]*pjrt.AsyncHostToDeviceTransferManager,

    const Layout = struct { []Buffer, []*pjrt.AsyncHostToDeviceTransferManager };

    pub fn init(allocator: std.mem.Allocator, platform: Platform, shapes: []const Shape, memory: pjrt.Memory.Kind) !TransferManager {
        const devices = platform.getDevices();
        const num_devices = devices.len;
        const num_partitions = platform.sharding().num_partitions;

        const buffers, const managers = try stdx.mem.groupedAlloc(Layout, allocator, .{ shapes.len, num_devices });

        const tmp_allocator = pjrtArgsTmpAllocator();
        const specs = try tmp_allocator.alloc(pjrt.ShapeSpec, shapes.len);

        for (specs, buffers, shapes) |*s, *b, shape| {
            const sharding_ax: ?u3 = std.simd.firstTrue(shape._sharding_info);
            if (sharding_ax) |ax| {
                stdx.debug.assert(@rem(shape.dim(ax), num_partitions) == 0, "Buffer.from({f}) expects the sharding axis {d} to have a dimension divisble by the number of devices ({d}).", .{ shape, ax, num_partitions });
            }

            const device_shape = if (sharding_ax) |ax|
                shape.setDim(ax, @divExact(shape.dim(ax), num_partitions))
            else
                shape;
            s.* = .init(device_shape.dims(), pjrt.bufferTypeFromDtype(shape.dtype()));
            b.* = .{ ._shape = shape, ._api = platform.pjrt_api, ._shards = .{ .len = num_devices } };
        }

        for (0..num_devices, devices, managers) |device_id, dev, *manager| {
            const memories = try dev.addressableMemories(platform.pjrt_api);
            var mem = memories[0];
            for (memories) |m| {
                if (m.kind(platform.pjrt_api) == memory) {
                    mem = m;
                    break;
                }
            }
            const mgr = try platform.pjrt_client.createBuffersForAsyncHostToDevice(
                platform.pjrt_api,
                .{ .shape_specs = specs, .memory = mem, .device_layouts = null },
            );
            manager.* = mgr;

            for (0.., buffers) |buffer_id, *b| {
                b._shards.buffer[device_id] = mgr.retrieveBuffer(platform.pjrt_api, buffer_id) catch @panic("PJRT plugin internal error");
            }
        }

        return .{
            .num_devices = @intCast(num_devices),
            .pjrt_api = platform.pjrt_api,
            .managers = managers.ptr,
            .buffers = buffers,
        };
    }

    pub fn deinit(self: TransferManager, allocator: std.mem.Allocator) void {
        const managers = self.managers[0..self.num_devices];
        for (managers) |manager| {
            manager.deinit(self.pjrt_api);
        }
        stdx.mem.groupedFree(Layout, allocator, .{ self.buffers, managers });
    }

    pub const Dest = struct { buffer_id: u32, device_id: u16, offset: u64 };

    pub fn transferData(self: TransferManager, dst: Dest, data: []const u8, is_last_transfer: bool) *pjrt.Event {
        std.debug.assert(dst.buffer_id < self.buffers.len);
        std.debug.assert(dst.device_id < self.num_devices);
        return self.managers[dst.device_id].transferData(self.pjrt_api, dst.buffer_id, data, @bitCast(dst.offset), is_last_transfer) catch @panic("PJRT plugin internal error");
    }
};

threadlocal var _pjrt_args_allocator: std.heap.StackFallbackAllocator(4096) = .{
    .buffer = undefined,
    .fallback_allocator = std.testing.failing_allocator,
    .fixed_buffer_allocator = undefined,
};

/// Returns a scratch allocator where memory allocated is only guaranteed to leave until the next PJRT api call.
/// This allows to have a scratch pad to convert arguments types between ZML/PJRT without allocating.
fn pjrtArgsTmpAllocator() std.mem.Allocator {
    // TODO: consider storing _pjrt_args_allocator inside the Platform struct itself.
    if (std.debug.runtime_safety) _pjrt_args_allocator.get_called = false;
    return _pjrt_args_allocator.get();
}

test TransferManager {
    const zml = @import("zml.zig");
    const platform = zml.testing.env();

    const transfer = try platform.batchedTransfer(std.testing.allocator, &.{.init(.{ 2, 5 }, .u32)}, .device);
    defer transfer.deinit(std.testing.allocator);
    const ev0 = transfer.transferData(.{ .buffer_id = 0, .device_id = 0, .offset = 0 }, std.mem.asBytes(&[_]u32{ 0, 1, 2, 3, 4 }), false);
    const ev1 = transfer.transferData(.{ .buffer_id = 0, .device_id = 0, .offset = 5 * @sizeOf(u32) }, std.mem.asBytes(&[_]u32{ 5, 6, 7, 8, 9 }), true);
    _ = ev0;
    _ = ev1;
    const x = transfer.buffers[0];

    try std.testing.expectEqual([2][5]u32{
        .{ 0, 1, 2, 3, 4 },
        .{ 5, 6, 7, 8, 9 },
    }, x.getValue([2][5]u32));
}
