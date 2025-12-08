const std = @import("std");

const stdx = @import("stdx");

const pjrt = @import("pjrtx.zig");

const Shape = @import("shape.zig").Shape;
const Platform = @import("platform.zig").Platform;
const Target = @import("platform.zig").Target;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Buffer);
}

const log = std.log.scoped(.zml);

/// Buffer is a multi-dimension array, whose memory is allocated on an accelerator.
///
/// * contains a handle that the ZML runtime can use to convert into a physical address, but there is no guarantee this address is visible from the CPU.
/// * loading weights from disk directly to the `device zml.aio.loadBuffers`
/// * can be created by calling `HostBuffer.toDevice(platform)`.
pub const Buffer = struct {
    _shape: Shape,
    _api: *const pjrt.Api,
    _target: Target,
    _shards: Shards,

    pub const MAX_NUM_SHARDS: u8 = Platform.MAX_NUM_DEVICES;
    pub const Shards = stdx.BoundedArray(*pjrt.Buffer, MAX_NUM_SHARDS);

    pub const Memory = pjrt.Memory.Kind;
    pub const FromOptions = struct { wait: bool = true, memory: Memory = .device };

    /// Frees the accelerator memory.
    /// Depending on the platform, the memory is typically not released to the OS
    /// but just marked as available in the memory pool.
    pub fn deinit(self: *const Buffer) void {
        // log.warn("Unloading {f} {d} bytes", .{ self._shape, self._shape.byteSize() });
        for (self._shards.constSlice()) |buffer| {
            buffer.deinit(self._api);
        }
    }

    /// Copies the content of the given buffer from host memory to the accelerator memory.
    pub fn from(platform: Platform, shape_: Shape, data_: []const u8, io: std.Io, opts: FromOptions) !Buffer {
        var res: Buffer = .{
            ._api = platform.pjrt_api,
            ._target = platform.target,
            ._shape = shape_,
            ._shards = .{},
        };

        // We shard only on the first axis so that the chunks are still contiguous.
        // TODO: support more advanced sharding specs
        stdx.debug.assert(platform.sharding().num_replicas == 1, "ZML doesn't support num_replicas > 1 for now, got: {}", .{platform.sharding()});
        const sharding_ax: ?u3 = std.simd.firstTrue(shape_._sharding_info);
        const n_partitions = platform.sharding().num_partitions;
        const chunk_size = if (sharding_ax) |ax| cs: {
            // This kind of sharding error should be detected earlier on.
            stdx.debug.assert(@rem(shape_.dim(ax), n_partitions) == 0, "Buffer.from({f}) expects the sharding axis {} to have a dimension divisble by the number of devices ({}).", .{ shape_, ax, n_partitions });
            break :cs @divExact(shape_.dim(ax), n_partitions);
        } else 0;
        _ = chunk_size; // autofix

        const buffer_type = pjrt.bufferTypeFromDtype(shape_.dtype());
        const byte_strides = shape_.computeStrides();

        const devices = platform.getDevices();
        for (0..n_partitions) |i| {
            // If no sharding if found, the given buffer is replicated on all devices.
            //const buf = if (sharding_ax) |ax| buf: {
            //    @panic("sharding not implemented");
            //    //const start: i64 = @as(i64, @intCast(i)) * chunk_size;
            //    //break :buf host_buffer.slice1d(ax, .{ .start = start, .end = start + chunk_size });
            //} else host_buffer;

            const args = pjrt.Client.BufferFromHostBufferArgs{
                .data = data_.ptr,
                .buffer_type = buffer_type,
                .dims = shape_.dims(),
                .byte_strides = byte_strides.slice(),
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
                // CPU has no distinctions between memories.
                .dst = if (platform.target == .cpu or opts.memory == .device)
                    .{ .device = devices[i] }
                else
                    .{ .memory = platform.memoryForDevice(opts.memory, devices[i]) },
            };

            const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);

            if (event) |ev| ev.deinit(platform.pjrt_api);
            res._shards.appendAssumeCapacity(pjrt_buffer);
        }

        if (opts.wait) {
            res = try res.await(io);
        }

        return res;
    }

    /// Copies the given Zig slice to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromBytes(platform: Platform, sh: Shape, data: []const u8, io: std.Io) !Buffer {
        return from(platform, sh, data, io, .{});
    }

    pub fn await(self: Buffer, io: std.Io) !Buffer {
        for (self._shards.constSlice()) |buffer| {
            if (buffer.getReadyEvent(self._api)) |ev| {
                try ev.await(self._api, io);
            }
        }

        return self;
    }
};
