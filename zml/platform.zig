const asynk = @import("async");
const builtin = @import("builtin");
const runtimes = @import("runtimes");
const std = @import("std");
const stdx = @import("stdx");

const meta = @import("meta.zig");
const module = @import("module.zig");
const pjrt = @import("pjrtx.zig");

const Buffer = @import("buffer.zig").Buffer;
const Shape = @import("shape.zig").Shape;

const Api = pjrt.Api;
const AsyncHostToDeviceTransferManager = pjrt.AsyncHostToDeviceTransferManager;
const Client = pjrt.Client;
const Device = pjrt.Device;
const Event = pjrt.Event;
const Memory = pjrt.Memory;
const NamedValue = pjrt.NamedValue;
const Profiler = pjrt.Profiler;
const ShapeSpec = pjrt.ShapeSpec;

const log = std.log.scoped(.zml);

pub const Target = runtimes.Platform;

pub const available_targets = std.enums.values(Target);

pub const CompilationOptions = struct {
    xla_dump_to: ?[]const u8 = null,
    xla_dump_fusion_visualization: bool = false,
    xla_dump_hlo_pass_re: ?[]const u8 = null,
    sharding_enabled: bool = false,
    sharding_axes: std.BoundedArray([*:0]const u8, 8) = .{},
};

pub const Platform = struct {
    target: Target,
    pjrt_api: *const Api,
    pjrt_client: *Client,
    compilation_options: CompilationOptions = .{},

    pub const MAX_NUM_DEVICES: u8 = 32;
    pub const CreateOptions = _CreateOptions;

    pub fn init(target: Target, api: *const Api, options: CreateOptions) !Platform {
        var named_values_buf: [16]NamedValue = undefined;
        const pjrt_client = try Client.init(api, options.toNamedValues(target, &named_values_buf));
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

    pub fn getDevices(self: Platform) []const *const Device {
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

    pub fn deinit(self: *Platform) void {
        self.pjrt_client.deinit(self.pjrt_api);
    }

    /// Returns the Profiler for this API.
    /// Not all platform have a profiling api, for those the profiler object will do nothing.
    /// Platforms with known profiler extensions: cuda, xpu
    pub fn getProfiler(self: Platform, options: ?Profiler.Options) Profiler {
        return self.pjrt_client.getProfiler(self.pjrt_api, options orelse Profiler.default_options);
    }
};

// struct PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {
//   size_t struct_size;
//   PJRT_Extension_Base* extension_start;
//   PJRT_AsyncHostToDeviceTransferManager* transfer_manager;
//   int buffer_index;
//   const void* data;
//   int64_t offset;
//   int64_t transfer_size;
//   bool is_last_transfer;
//   PJRT_Event* done_with_h2d_transfer;  // out
// };

// struct PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args {
//   size_t struct_size;
//   PJRT_Extension_Base* extension_start;
//   PJRT_AsyncHostToDeviceTransferManager* transfer_manager;
//   int buffer_index;
//   PJRT_Buffer* buffer_out;  // out
// };

// struct PJRT_AsyncHostToDeviceTransferManager_Device_Args {
//   size_t struct_size;
//   PJRT_Extension_Base* extension_start;
//   PJRT_AsyncHostToDeviceTransferManager* transfer_manager;
//   PJRT_Device* device_out;  // out
// };

// struct PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args {
//   size_t struct_size;
//   PJRT_Extension_Base* extension_start;
//   PJRT_AsyncHostToDeviceTransferManager* transfer_manager;
//   size_t buffer_count;  // out
// };

// struct PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args {
//   size_t struct_size;
//   PJRT_Extension_Base* extension_start;
//   PJRT_AsyncHostToDeviceTransferManager* transfer_manager;
//   int buffer_index;
//   size_t buffer_size;  // out
// };

// struct PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args {
//   size_t struct_size;
//   PJRT_Extension_Base* extension_start;
//   PJRT_AsyncHostToDeviceTransferManager* transfer_manager;
//   const PJRT_NamedValue* transfer_metadata;
//   size_t num_metadata;
// };

// struct PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {
//   size_t struct_size;
//   PJRT_Extension_Base* extension_start;
//   PJRT_Client* client;
//   PJRT_ShapeSpec* shape_specs;
//   size_t num_shape_specs;
//   PJRT_Buffer_MemoryLayout** device_layouts;  // optional
//   size_t num_device_layouts;
//   PJRT_Memory* memory;
//   PJRT_AsyncHostToDeviceTransferManager* transfer_manager;  // out
// };

pub const TransferManager = struct {
    platform: Platform,
    pjrt_client: *Client,
    pjrt_api: *const Api,
    // pjrt_transfer_manager: []*AsyncHostToDeviceTransferManager,
    pjrt_transfer_manager: *AsyncHostToDeviceTransferManager,

    device: *const Device,
    memory: *const Memory,
    shapes: []const Shape,
    shape_specs: std.ArrayList(ShapeSpec),
    events: std.ArrayList(*Event),
    buffers_alist: std.ArrayList(Buffer),
    seen_last_buffer: bool,

    pub fn init(
        alloc: std.mem.Allocator,
        platform: Platform,
        memory_kind: Memory.Kind,
        shapes: []const Shape,
    ) !TransferManager {
        const device = platform.getDevices()[0]; // TODO: when sharding
        const memory = device.getMemoryByKind(platform.pjrt_api, memory_kind);
        if (memory == null) {
            stdx.debug.panic("Device {s} doesn't have memory of kind {s}", .{ device.getDescription(platform.pjrt_api).getKind(platform.pjrt_api), @tagName(memory_kind) });
        }
        const num_buffers = shapes.len;
        var shape_specs = try std.ArrayList(ShapeSpec).initCapacity(alloc, num_buffers);
        for (shapes) |shape| {
            shape_specs.appendAssumeCapacity(
                ShapeSpec.init(
                    shape.dims(),
                    Buffer.bufferTypeFromDtype(shape.dtype()),
                ),
            );
        }

        var self: TransferManager = .{
            .buffers_alist = try std.ArrayList(Buffer).initCapacity(alloc, num_buffers),
            .device = device,
            .events = try std.ArrayList(*Event).initCapacity(alloc, num_buffers),
            .memory = memory.?,
            .platform = platform,
            .pjrt_client = platform.pjrt_client,
            .pjrt_api = platform.pjrt_api,
            .pjrt_transfer_manager = try Client.createBuffersForAsyncHostToDevice(
                platform.pjrt_client,
                platform.pjrt_api,
                .{
                    .shape_specs = shape_specs.items,
                    .memory = memory.?,
                    .device_layouts = null,
                },
            ),
            .seen_last_buffer = false,
            .shapes = shapes,
            .shape_specs = shape_specs,
        };
        try self.toZmlBuffers();
        return self;
    }

    pub fn deinit(self: *TransferManager) void {
        self.pjrt_transfer_manager.deinit(self.pjrt_api);
        self.shape_specs.deinit();
        self.events.deinit();
        self.buffers_alist.deinit();
    }

    pub fn buffers(self: *const TransferManager) []Buffer {
        return self.buffers_alist.items;
    }

    pub fn buffer(self: *const TransferManager, index: usize) ?Buffer {
        return self.buffers_alist.items[index];
    }

    pub fn transferDataSingle(self: *TransferManager, buffer_index: usize, data: []const u8, offset: i64, is_last_transfer: bool) !*Event {
        if (self.seen_last_buffer) {
            stdx.debug.panic("Attempting to transferData after a transferData with is_last_transfer=true on device {s} with memory of kind {s}", .{
                self.device.getDescription(self.pjrt_api).getKind(self.pjrt_api),
                @tagName(self.memory.kind(self.pjrt_api)),
            });
        }

        if (is_last_transfer) {
            self.seen_last_buffer = true;
        }

        log.debug("transfer size: {d} - last transfer: {}", .{ data.len, is_last_transfer });
        const event = try self.pjrt_transfer_manager.transferData(
            self.pjrt_api,
            buffer_index,
            data,
            offset,
            is_last_transfer,
        );
        log.debug("event: {}", .{event});
        // TODO: might cause crashes if used improperly:
        self.events.appendAssumeCapacity(event);

        if (is_last_transfer and buffer_index != try self.pjrt_transfer_manager.bufferCount(self.pjrt_api) - 1) {
            stdx.debug.panic(
                "transferData: is_last_transfer = true for buffer_index {d} which is not the last buffer (expected {d} buffers)!!!",
                .{ buffer_index, try self.pjrt_transfer_manager.bufferCount(self.pjrt_api) },
            );
        }
        return event;
    }

    pub const TransferDataMultiOpts = struct {
        start_buffer_index: usize = 0,
        last_data_is_last_transfer: bool = true,
    };
    pub fn transferDataMany(self: *TransferManager, data_slices: []const []const u8, opts: TransferDataMultiOpts) ![]*Event {
        for (data_slices, @intCast(opts.start_buffer_index)..) |data, buffer_index| {
            const is_last_transfer = blk: {
                if (opts.last_data_is_last_transfer) {
                    break :blk buffer_index == data_slices.len - 1;
                } else {
                    break :blk false;
                }
            };
            log.debug("TransferManager initiating transfer {d}", .{buffer_index});
            _ = try self.transferDataSingle(buffer_index, data, 0, is_last_transfer);
        }
        return self.events.items;
    }

    pub const TransferDataSlicesSpec = struct {
        offset: usize,
        len: usize,
    };
    pub fn transferDataSlices(self: *TransferManager, input_buffer: []const u8, slice_specs: []const TransferDataSlicesSpec) ![]*Event {
        for (slice_specs, 0..) |spec, buffer_index| {
            const is_last_transfer = buffer_index == slice_specs.len - 1;
            const data_slice = input_buffer[spec.offset .. spec.offset + spec.len];
            _ = try self.transferDataSingle(buffer_index, data_slice, 0, is_last_transfer);
        }
        return self.events.items;
    }

    pub const Progress = struct {
        transferred_buffers: usize,
        total_buffers: usize,
    };
    pub fn progress(self: *const TransferManager) !Progress {
        var buffers_ready: usize = 0;
        for (self.events.items) |event| {
            if (event.isReady(self.pjrt_api)) {
                buffers_ready += 1;
            }
        }
        return .{
            .transferred_buffers = buffers_ready,
            .total_buffers = try self.pjrt_transfer_manager.bufferCount(self.pjrt_api),
        };
    }

    /// called internally to retrieve PJRT buffers that can be accessed via .buffers()
    fn toZmlBuffers(self: *TransferManager) !void {
        if (self.buffers_alist.items.len == 0) {
            for (0..try self.pjrt_transfer_manager.bufferCount(self.pjrt_api)) |buffer_index| {
                const pjrt_buffer = try self.pjrt_transfer_manager.retrieveBuffer(self.pjrt_api, buffer_index);
                const shape = self.shapes[buffer_index];
                const zml_buffer = Buffer.fromPjrtBuffers(self.platform, shape, &.{pjrt_buffer});
                self.buffers_alist.appendAssumeCapacity(zml_buffer);
            }
        }
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
        should_stage_host_to_device_transfers: bool = false,

        pub const Allocator = union(enum) {
            /// "Best-Fit with Coalescing" algorithm
            bfc: Options,
            /// use cudaMallocAsync
            @"async": Options,
            /// use raw cuMalloc
            platform,

            pub const Options = struct {
                preallocate: bool = true,
                memory_fraction: f32 = 0.90,
                collective_memory_size_mb: u32 = 0,
            };
        };

        pub fn writeNamedValues(self: Cuda, values: *std.ArrayListUnmanaged(NamedValue)) void {
            values.appendAssumeCapacity(NamedValue.fromBool("should_stage_host_to_device_transfers", self.should_stage_host_to_device_transfers));
            switch (self.allocator) {
                .platform => {
                    values.appendAssumeCapacity(NamedValue.fromString("allocator", "platform"));
                },
                .bfc, .@"async" => |opt| {
                    values.appendAssumeCapacity(NamedValue.from("allocator", self.allocator));
                    values.appendAssumeCapacity(NamedValue.from("preallocate", opt.preallocate));
                    if (opt.memory_fraction > 0) {
                        values.appendAssumeCapacity(NamedValue.from("memory_fraction", opt.memory_fraction));
                    }
                    if (opt.collective_memory_size_mb > 0) {
                        const collective = @as(i64, opt.collective_memory_size_mb) * 1024 * 1024;
                        values.appendAssumeCapacity(NamedValue.from("collective_memory_size", collective));
                    }
                },
            }
        }
    };

    pub fn toNamedValues(self: _CreateOptions, target: Target, out: []NamedValue) []NamedValue {
        var values = std.ArrayListUnmanaged(NamedValue).fromOwnedSlice(out);
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
