const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
pub const VFS = @import("io").VFS;

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("zml.zig").Bufferized;
const CreateOptions = @import("platform.zig").CreateOptions;
const mem = @import("mem.zig");
const Memory = @import("platform.zig").Memory;
const meta = @import("meta.zig");
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const tracer = @import("profiling/tracer.zig");
const safetensors = @import("safetensors.zig");
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Placement = Sharding.Placement;
const Slice = @import("slice.zig").Slice;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/io");
const load_log = std.log.scoped(.@"zml/io/load");

pub const TensorStore = struct {
    registry: *safetensors.TensorRegistry,
    id_map: std.AutoHashMapUnmanaged(usize, *safetensors.Tensor),
    allocator: std.mem.Allocator,

    pub fn fromRegistry(allocator: std.mem.Allocator, registry: *safetensors.TensorRegistry) TensorStore {
        return .{
            .registry = registry,
            .id_map = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TensorStore) void {
        self.id_map.deinit(self.allocator);
    }

    fn bindIdToKey(self: *TensorStore, key: []const u8, id: usize) !void {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key).?;

        const gop = try self.id_map.getOrPut(self.allocator, id);
        if (gop.found_existing) {
            stdx.debug.panic("Key {s} already has an associated tensor (id: {})", .{ key, gop.key_ptr.* });
        }
        errdefer self.id_map.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = tensor_desc_ptr;
    }

    fn getPtrFromKey(self: *const TensorStore, key: []const u8) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
        return tensor_desc_ptr;
    }

    fn getPtrFromId(self: *const TensorStore, id: usize) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.id_map.get(id) orelse return null;
        return tensor_desc_ptr;
    }

    pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        return self.registry.reader(io, key, buffer);
    }

    pub fn getReaderById(self: *const TensorStore, id: usize, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        const tensor_desc = self.id_map.get(id) orelse return error.NotFound;

        return safetensors.TensorReader.init(io, tensor_desc.*, buffer, .{});
    }

    pub fn view(self: *TensorStore) View {
        return .{ .store = self };
    }

    pub const View = struct {
        store: *TensorStore,

        prefix_buffer: [256]u8 = undefined,
        prefix_length: usize = 0,

        pub fn root(self: *const View) View {
            return .{
                .store = self.store,
            };
        }

        pub fn parent(self: *const View) View {
            const slice = self.prefix() orelse unreachable;
            const index = std.mem.lastIndexOfScalar(u8, slice[0 .. slice.len - 1], '.') orelse return self.root();
            var buffer: [256]u8 = undefined;
            @memcpy(buffer[0 .. index + 1], slice[0 .. index + 1]);
            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = index + 1,
            };
        }

        pub fn withPrefix(self: *const View, prefix_: []const u8) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        pub fn withLayer(self: *const View, index: usize) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{d}.", .{ self.prefix() orelse "", index }) catch unreachable;

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        fn prefix(self: *const View) ?[]const u8 {
            return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
        }

        pub fn hasKey(self: *const View, subkey: []const u8) bool {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            return for (self.store.registry.tensors.keys()) |k| {
                if (std.mem.startsWith(u8, k, key)) break true;
            } else false;
        }

        pub fn maybeCreateTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;
            if (@TypeOf(tagz) != @TypeOf(null)) {
                switch (@typeInfo(@TypeOf(tagz))) {
                    .optional => if (tagz) |t| {
                        ptr.shape = ptr.shape.withTags(t);
                    },
                    else => ptr.shape = ptr.shape.withTags(tagz),
                }
            }

            if (@TypeOf(partitioning) == @TypeOf(null)) {
                @compileError("TensorStore.View.createTensor partitioning cannot be null; pass .replicated or an explicit partitioning");
            }

            switch (@typeInfo(@TypeOf(partitioning))) {
                .optional => @compileError("TensorStore.View.createTensor partitioning cannot be optional; pass .replicated or an explicit partitioning"),
                .enum_literal => switch (partitioning) {
                    .replicated => ptr.shape = ptr.shape.withReplicatedPartitioning(),
                    else => @compileError("Only .replicated is supported as a standalone partitioning enum literal"),
                },
                else => ptr.shape = ptr.shape.withPartitioning(partitioning),
            }

            const tensor: Tensor = .fromShape(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) Tensor {
            return self.maybeCreateTensor(subkey, tagz, partitioning).?;
        }

        pub fn getShape(self: View, subkey: []const u8) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getShapeOpts(self: View, subkey: []const u8, opts: struct { no_prefix: bool = false }) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = if (opts.no_prefix)
                subkey
            else b: {
                break :b std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            };
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getReader(self: View, subkey: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
            var key_buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&key_buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            return self.store.getReader(key, io, buffer);
        }

        pub fn count(self: View) usize {
            var count_: usize = 0;
            const prefix_ = self.prefix() orelse "";
            var it = self.store.registry.tensors.iterator();
            while (it.next()) |item| {
                const key = item.key_ptr.*;
                if (std.mem.startsWith(u8, key, prefix_)) {
                    count_ += 1;
                }
            }
            return count_;
        }
    };
};

pub const ProgressWriter = struct {
    inner: *std.Io.Writer,
    progress: *std.Progress.Node,
    interface: std.Io.Writer,
    total: usize = 0,
    scale: usize,

    pub const InitOpts = struct {
        scale: usize = 1,
    };

    pub fn init(inner_: *std.Io.Writer, progress_: *std.Progress.Node, opts: InitOpts) ProgressWriter {
        return .{
            .inner = inner_,
            .progress = progress_,
            .scale = opts.scale,
            .interface = .{
                .buffer = inner_.buffer,
                .end = inner_.end,
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                    .sendFile = sendFile,
                },
            },
        };
    }

    pub fn pre(self: *ProgressWriter) usize {
        self.inner.buffer = self.interface.buffer;
        self.inner.end = self.interface.end;
        return self.inner.end;
    }

    pub fn post(self: *ProgressWriter, len_pre: usize, total: usize) void {
        self.interface.buffer = self.inner.buffer;
        self.interface.end = self.inner.end;
        const drained_pre = len_pre -| self.interface.end;
        self.total += drained_pre + total;
        self.progress.setCompletedItems(self.total / self.scale);
    }

    pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        errdefer self.post(len_pre, 0);
        const total = try self.inner.vtable.drain(self.inner, data, splat);
        self.post(len_pre, total);
        return total;
    }

    pub fn sendFile(w: *std.Io.Writer, file_reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.Writer.FileError!usize {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        errdefer self.post(len_pre, 0);
        const total = try self.inner.vtable.sendFile(self.inner, file_reader, limit);
        self.post(len_pre, total);
        return total;
    }

    pub fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        defer self.post(len_pre, 0);
        try self.inner.vtable.flush(self.inner);
    }

    pub fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        defer self.post(len_pre, 0);
        try self.inner.vtable.rebase(self.inner, preserve, capacity);
    }
};

const LoadMetrics = struct {
    storage_bytes: std.atomic.Value(u64) = .init(0),
    direct_read_bytes: std.atomic.Value(u64) = .init(0),
    staged_read_bytes: std.atomic.Value(u64) = .init(0),
    staged_copy_bytes: std.atomic.Value(u64) = .init(0),
    ordered_bytes: std.atomic.Value(u64) = .init(0),
    logical_submitted_bytes: std.atomic.Value(u64) = .init(0),
    read_operations: std.atomic.Value(u64) = .init(0),
    dma_submissions: std.atomic.Value(u64) = .init(0),
    submitted_bytes: std.atomic.Value(u64) = .init(0),
    committed_bytes: std.atomic.Value(u64) = .init(0),
    last_dma_commit_ns: std.atomic.Value(u64) = .init(0),
    weighted_read_latency_us: std.atomic.Value(u64) = .init(0),
    weighted_transfer_latency_us: std.atomic.Value(u64) = .init(0),
    pinned_buffer_wait_ns: std.atomic.Value(u64) = .init(0),
    dma_completion_wait_ns: std.atomic.Value(u64) = .init(0),
    read_admission_wait_ns: std.atomic.Value(u64) = .init(0),
    staging_wait_ns: std.atomic.Value(u64) = .init(0),
    dma_starved_ns: std.atomic.Value(u64) = .init(0),
    dma_starvation_covered_until_ns: std.atomic.Value(u64) = .init(0),
    dma_work_ns: std.atomic.Value(u64) = .init(0),
    staged_copy_ns: std.atomic.Value(u64) = .init(0),
    weighted_ready_age_us: std.atomic.Value(u64) = .init(0),
    ready_bytes: std.atomic.Value(u64) = .init(0),
    active_reads: std.atomic.Value(usize) = .init(0),
    active_reads_high_water: std.atomic.Value(usize) = .init(0),
    active_transfers: std.atomic.Value(usize) = .init(0),
    completed_transfers: std.atomic.Value(usize) = .init(0),
    probe_mutex: std.Io.Mutex = .init,
    config_epoch: std.atomic.Value(u64) = .init(0),
    probe_epoch: std.atomic.Value(u64) = .init(0),
    probe_committed_bytes: std.atomic.Value(u64) = .init(0),
    probe_first_ns: std.atomic.Value(u64) = .init(0),
    staging_limit: std.atomic.Value(usize) = .init(0),

    const Snapshot = struct {
        storage_bytes: u64,
        direct_read_bytes: u64,
        staged_read_bytes: u64,
        staged_copy_bytes: u64,
        ordered_bytes: u64,
        logical_submitted_bytes: u64,
        read_operations: u64,
        dma_submissions: u64,
        submitted_bytes: u64,
        committed_bytes: u64,
        weighted_read_latency_us: u64,
        weighted_transfer_latency_us: u64,
        pinned_buffer_wait_ns: u64,
        dma_completion_wait_ns: u64,
        read_admission_wait_ns: u64,
        staging_wait_ns: u64,
        dma_starved_ns: u64,
        dma_work_ns: u64,
        staged_copy_ns: u64,
        weighted_ready_age_us: u64,
        ready_bytes: u64,
        active_reads: usize,
        active_transfers: usize,
        completed_transfers: usize,
        config_epoch: u64,
        probe_epoch: u64,
        probe_committed_bytes: u64,
        probe_first_ns: u64,

        fn sub(self: Snapshot, previous: Snapshot) Snapshot {
            return .{
                .storage_bytes = self.storage_bytes -| previous.storage_bytes,
                .direct_read_bytes = self.direct_read_bytes -| previous.direct_read_bytes,
                .staged_read_bytes = self.staged_read_bytes -| previous.staged_read_bytes,
                .staged_copy_bytes = self.staged_copy_bytes -| previous.staged_copy_bytes,
                .ordered_bytes = self.ordered_bytes -| previous.ordered_bytes,
                .logical_submitted_bytes = self.logical_submitted_bytes -| previous.logical_submitted_bytes,
                .read_operations = self.read_operations -| previous.read_operations,
                .dma_submissions = self.dma_submissions -| previous.dma_submissions,
                .submitted_bytes = self.submitted_bytes -| previous.submitted_bytes,
                .committed_bytes = self.committed_bytes -| previous.committed_bytes,
                .weighted_read_latency_us = self.weighted_read_latency_us -| previous.weighted_read_latency_us,
                .weighted_transfer_latency_us = self.weighted_transfer_latency_us -| previous.weighted_transfer_latency_us,
                .pinned_buffer_wait_ns = self.pinned_buffer_wait_ns -| previous.pinned_buffer_wait_ns,
                .dma_completion_wait_ns = self.dma_completion_wait_ns -| previous.dma_completion_wait_ns,
                .read_admission_wait_ns = self.read_admission_wait_ns -| previous.read_admission_wait_ns,
                .staging_wait_ns = self.staging_wait_ns -| previous.staging_wait_ns,
                .dma_starved_ns = self.dma_starved_ns -| previous.dma_starved_ns,
                .dma_work_ns = self.dma_work_ns -| previous.dma_work_ns,
                .staged_copy_ns = self.staged_copy_ns -| previous.staged_copy_ns,
                .weighted_ready_age_us = self.weighted_ready_age_us -| previous.weighted_ready_age_us,
                .ready_bytes = self.ready_bytes,
                .active_reads = self.active_reads,
                .active_transfers = self.active_transfers,
                .completed_transfers = self.completed_transfers,
                .config_epoch = self.config_epoch,
                .probe_epoch = self.probe_epoch,
                .probe_committed_bytes = self.probe_committed_bytes,
                .probe_first_ns = self.probe_first_ns,
            };
        }
    };

    fn snapshot(self: *const LoadMetrics) Snapshot {
        return .{
            .storage_bytes = self.storage_bytes.load(.acquire),
            .direct_read_bytes = self.direct_read_bytes.load(.acquire),
            .staged_read_bytes = self.staged_read_bytes.load(.acquire),
            .staged_copy_bytes = self.staged_copy_bytes.load(.acquire),
            .ordered_bytes = self.ordered_bytes.load(.acquire),
            .logical_submitted_bytes = self.logical_submitted_bytes.load(.acquire),
            .read_operations = self.read_operations.load(.acquire),
            .dma_submissions = self.dma_submissions.load(.acquire),
            .submitted_bytes = self.submitted_bytes.load(.acquire),
            .committed_bytes = self.committed_bytes.load(.acquire),
            .weighted_read_latency_us = self.weighted_read_latency_us.load(.acquire),
            .weighted_transfer_latency_us = self.weighted_transfer_latency_us.load(.acquire),
            .pinned_buffer_wait_ns = self.pinned_buffer_wait_ns.load(.acquire),
            .dma_completion_wait_ns = self.dma_completion_wait_ns.load(.acquire),
            .read_admission_wait_ns = self.read_admission_wait_ns.load(.acquire),
            .staging_wait_ns = self.staging_wait_ns.load(.acquire),
            .dma_starved_ns = self.dma_starved_ns.load(.acquire),
            .dma_work_ns = self.dma_work_ns.load(.acquire),
            .staged_copy_ns = self.staged_copy_ns.load(.acquire),
            .weighted_ready_age_us = self.weighted_ready_age_us.load(.acquire),
            .ready_bytes = self.ready_bytes.load(.acquire),
            .active_reads = self.active_reads.load(.acquire),
            .active_transfers = self.active_transfers.load(.acquire),
            .completed_transfers = self.completed_transfers.load(.acquire),
            .config_epoch = self.config_epoch.load(.acquire),
            .probe_epoch = self.probe_epoch.load(.acquire),
            .probe_committed_bytes = self.probe_committed_bytes.load(.acquire),
            .probe_first_ns = self.probe_first_ns.load(.acquire),
        };
    }

    fn addPinnedBufferWait(self: *LoadMetrics, wait_ns: u64) void {
        _ = self.pinned_buffer_wait_ns.fetchAdd(wait_ns, .monotonic);
    }

    fn addDmaCompletionWait(self: *LoadMetrics, duration: std.Io.Duration) void {
        const ns: u64 = @intCast(@max(duration.nanoseconds, 0));
        _ = self.dma_completion_wait_ns.fetchAdd(ns, .monotonic);
    }

    fn addDmaStarvationInterval(self: *LoadMetrics, started_ns: u64, ended_ns: u64) void {
        if (ended_ns <= started_ns) return;
        var covered_until = self.dma_starvation_covered_until_ns.load(.acquire);
        while (ended_ns > covered_until) {
            const uncovered_started = @max(started_ns, covered_until);
            if (self.dma_starvation_covered_until_ns.cmpxchgWeak(covered_until, ended_ns, .acq_rel, .acquire)) |actual| {
                covered_until = actual;
                continue;
            }
            _ = self.dma_starved_ns.fetchAdd(ended_ns - uncovered_started, .monotonic);
            return;
        }
    }

    fn recordReadStart(self: *LoadMetrics) void {
        const active = self.active_reads.fetchAdd(1, .acq_rel) + 1;
        var high_water = self.active_reads_high_water.load(.acquire);
        while (active > high_water) {
            if (self.active_reads_high_water.cmpxchgWeak(high_water, active, .release, .acquire)) |actual| {
                high_water = actual;
                continue;
            }
            break;
        }
    }

    fn resetReadHighWater(self: *LoadMetrics) void {
        self.active_reads_high_water.store(self.active_reads.load(.acquire), .release);
    }

    fn recordProbeCommit(self: *LoadMetrics, io: std.Io, epoch: u64, bytes: usize) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch == self.probe_epoch.load(.acquire)) {
            _ = self.probe_committed_bytes.fetchAdd(@intCast(bytes), .monotonic);
        }
    }

    fn markProbeStart(self: *LoadMetrics, io: std.Io, epoch: u64) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch.load(.acquire)) return;
        const now_ns: u64 = @intCast(@max(std.Io.Timestamp.now(io, .awake).nanoseconds, 1));
        _ = self.probe_first_ns.cmpxchgStrong(0, now_ns, .release, .monotonic);
    }

    fn publishProbeEpoch(self: *LoadMetrics, io: std.Io, epoch: u64) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        self.probe_epoch.store(std.math.maxInt(u64), .release);
        self.probe_committed_bytes.store(0, .release);
        self.probe_first_ns.store(0, .release);
        self.probe_epoch.store(epoch, .release);
        self.config_epoch.store(epoch, .release);
    }
};

fn getDmaBuffer(
    pool: *mem.DynamicBufferPool,
    allocator: std.mem.Allocator,
    io: std.Io,
    metrics: ?*LoadMetrics,
) ![]u8 {
    const acquisition = try pool.getWithWait(allocator, io);
    if (metrics) |m| m.addPinnedBufferWait(acquisition.wait_ns);
    return acquisition.buffer;
}

pub const MemoryWriter = union(enum) {
    direct: DirectMemoryWriter,
    buffered: BufferedMemoryWriter,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: []mem.DynamicBufferPool,
        dma_allocators: []const mem.DmaAllocator,
        dma_chunk_size: usize,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
    ) !MemoryWriter {
        return initWithMetrics(allocator, io, platform, pools, dma_allocators, dma_chunk_size, shape, sharding, buffer, null);
    }

    fn initWithMetrics(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: []mem.DynamicBufferPool,
        dma_allocators: []const mem.DmaAllocator,
        dma_chunk_size: usize,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
        metrics: ?*LoadMetrics,
    ) !MemoryWriter {
        return switch (platform.target) {
            .cuda, .oneapi => .{ .direct = try DirectMemoryWriter.initWithMetrics(allocator, io, platform, pools, dma_allocators, dma_chunk_size, shape, sharding, buffer, metrics) },
            .rocm, .tpu, .neuron, .cpu, .metal => .{ .buffered = try BufferedMemoryWriter.init(allocator, io, platform, shape, sharding, buffer) },
        };
    }

    pub fn interface(self: *MemoryWriter) *std.Io.Writer {
        return switch (self.*) {
            .direct => &self.direct.interface,
            .buffered => &self.buffered.interface,
        };
    }

    pub fn deinit(self: *MemoryWriter, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .direct => self.direct.deinit(),
            .buffered => self.buffered.deinit(allocator),
        }
    }

    pub fn setProgress(self: *MemoryWriter, progress: ?*std.Progress.Node) void {
        switch (self.*) {
            .direct => self.direct.setProgress(progress),
            .buffered => {},
        }
    }

    fn setEpoch(self: *MemoryWriter, epoch: u64) std.Io.Writer.Error!void {
        switch (self.*) {
            .direct => try self.direct.setEpoch(epoch),
            .buffered => {},
        }
    }

    fn directWritable(self: *MemoryWriter) []u8 {
        return switch (self.*) {
            .direct => self.direct.directWritableSlice(),
            .buffered => unreachable,
        };
    }

    fn commitDirectRead(self: *MemoryWriter, len: usize) std.Io.Writer.Error!void {
        switch (self.*) {
            .direct => try self.direct.commitDirectRead(len),
            .buffered => unreachable,
        }
    }

    fn commitStagedWrite(self: *MemoryWriter) std.Io.Writer.Error!void {
        switch (self.*) {
            .direct => try self.direct.commitStagedWrite(),
            .buffered => {},
        }
    }

    fn rootError(self: *const MemoryWriter) ?anyerror {
        return switch (self.*) {
            .direct => blk: {
                for (self.direct.shard_writers) |*writer| {
                    if (writer.rootError()) |err| break :blk err;
                }
                break :blk null;
            },
            .buffered => null,
        };
    }

    fn park(self: *MemoryWriter) std.Io.Writer.Error!void {
        switch (self.*) {
            .direct => try self.direct.park(),
            .buffered => {},
        }
    }

    fn parkAndWait(self: *MemoryWriter) std.Io.Writer.Error!void {
        switch (self.*) {
            .direct => try self.direct.parkAndWait(),
            .buffered => {},
        }
    }

    fn waitForPendingDma(self: *MemoryWriter) std.Io.Writer.Error!void {
        switch (self.*) {
            .direct => try self.direct.waitForPendingDma(),
            .buffered => {},
        }
    }

    fn submissionCount(self: *const MemoryWriter) u64 {
        return switch (self.*) {
            .direct => self.direct.submissionCount(),
            .buffered => 0,
        };
    }

    fn unpark(self: *MemoryWriter) std.Io.Writer.Error!void {
        switch (self.*) {
            .direct => try self.direct.unpark(),
            .buffered => {},
        }
    }
};

pub const BufferedMemoryWriter = struct {
    io: std.Io,
    platform: *const Platform,
    shape: Shape,
    sharding: Sharding,
    buffer: *Buffer,
    interface: std.Io.Writer,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, shape: Shape, sharding: Sharding, buffer: *Buffer) !BufferedMemoryWriter {
        return .{
            .io = io,
            .platform = platform,
            .shape = shape,
            .sharding = sharding,
            .buffer = buffer,
            .interface = .{
                .buffer = try allocator.alloc(u8, shape.byteSize()),
                .vtable = &.{
                    .drain = std.Io.Writer.fixedDrain,
                    .flush = flush,
                    .rebase = std.Io.Writer.failingRebase,
                },
            },
        };
    }

    pub fn deinit(self: *BufferedMemoryWriter, allocator: std.mem.Allocator) void {
        if (self.interface.buffer.len > 0) {
            allocator.free(self.interface.buffer);
        }
    }

    pub fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *BufferedMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        self.buffer.* = Buffer.from(
            self.io,
            self.platform,
            self.shape,
            self.sharding,
            @ptrCast(self.interface.buffer),
            .{ .wait = true },
        ) catch return std.Io.Writer.Error.WriteFailed;
    }
};

const DirectShardWriter = struct {
    const EventContext = struct {
        self: *DirectShardWriter,
        err: ?*pjrt.Error = null,
        pjrt_event: *pjrt.Event,
        event: std.Io.Event = .unset,
        buffer: []u8,
        submitted_at: std.Io.Timestamp,
        bytes: usize,
        epoch: u64,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    memory: *const Memory,
    pool: *mem.DynamicBufferPool,
    total: usize,
    pjrt_buffer: *pjrt.Buffer,

    transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,
    offset: usize = 0,

    interface: std.Io.Writer,
    active_buffer: ?[]u8,
    flip_flop: u1 = 0,
    events_contexts: [2]?EventContext = @splat(null),
    metrics: ?*LoadMetrics,
    epoch: u64 = 0,
    submission_count: u64 = 0,
    root_error: std.atomic.Value(u16) = .init(0),

    fn recordError(self: *DirectShardWriter, err: anyerror) void {
        _ = self.root_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    fn rootError(self: *const DirectShardWriter) ?anyerror {
        const err = self.root_error.load(.acquire);
        return if (err == 0) null else @errorFromInt(err);
    }

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        memory: *const Memory,
        pool: *mem.DynamicBufferPool,
        shape: Shape,
        metrics: ?*LoadMetrics,
    ) !DirectShardWriter {
        const shape_spec: pjrt.ShapeSpec = .init(shape.dims(), pjrtx.bufferTypeFromDtype(shape.dtype()));
        const transfer_manager = try memory.platform.pjrt_client.createBuffersForAsyncHostToDevice(
            memory.platform.pjrt_api,
            .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            },
        );
        errdefer transfer_manager.deinit(memory.platform.pjrt_api);

        const pjrt_buffer = transfer_manager.retrieveBuffer(memory.platform.pjrt_api, 0) catch unreachable;

        const buf = try getDmaBuffer(pool, allocator, io, metrics);

        return .{
            .allocator = allocator,
            .io = io,
            .memory = memory,
            .pool = pool,
            .total = shape.byteSize(),
            .pjrt_buffer = pjrt_buffer,
            .transfer_manager = transfer_manager,
            .metrics = metrics,
            .active_buffer = buf,
            .interface = .{
                .buffer = buf[0..@min(buf.len, shape.byteSize())],
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                },
            },
        };
    }

    pub fn deinit(self: *DirectShardWriter) void {
        const pjrt_api = self.memory.platform.pjrt_api;
        for (&self.events_contexts) |*maybe_ctx| {
            if (maybe_ctx.*) |*ctx| {
                ctx.event.waitUncancelable(self.io);
                ctx.pjrt_event.deinit(pjrt_api);
                if (ctx.err) |err| err.deinit(pjrt_api);
                maybe_ctx.* = null;
            }
        }
        if (self.active_buffer) |buffer| {
            self.pool.put(self.io, buffer);
            self.active_buffer = null;
        }
        self.transfer_manager.deinit(self.memory.platform.pjrt_api);
    }

    fn park(self: *DirectShardWriter) std.Io.Writer.Error!void {
        if (self.interface.end > 0) try self.flushBuffered();
        if (self.active_buffer) |buffer| {
            self.pool.put(self.io, buffer);
            self.active_buffer = null;
        }
        if (self.offset < self.total) {
            self.interface.buffer = &.{};
            self.interface.end = 0;
        }
    }

    fn waitForPendingTransfers(self: *DirectShardWriter) std.Io.Writer.Error!void {
        const pjrt_api = self.memory.platform.pjrt_api;
        for (&self.events_contexts) |*maybe_ctx| {
            const ctx = if (maybe_ctx.*) |*context| context else continue;
            const wait_started: std.Io.Timestamp = .now(self.io, .awake);
            ctx.event.waitUncancelable(self.io);
            if (self.metrics) |metrics| metrics.addDmaCompletionWait(wait_started.untilNow(self.io, .awake));
            ctx.pjrt_event.deinit(pjrt_api);
            if (ctx.err) |err| {
                self.recordError(err.getCode(pjrt_api).toApiError());
                err.deinit(pjrt_api);
                maybe_ctx.* = null;
                return std.Io.Writer.Error.WriteFailed;
            }
            maybe_ctx.* = null;
        }
    }

    fn unpark(self: *DirectShardWriter) std.Io.Writer.Error!void {
        if (self.offset >= self.total or self.active_buffer != null) return;
        const buffer = getDmaBuffer(self.pool, self.allocator, self.io, self.metrics) catch |err| {
            self.recordError(err);
            return std.Io.Writer.Error.WriteFailed;
        };
        self.active_buffer = buffer;
        self.interface.buffer = buffer[0..@min(buffer.len, self.total - self.offset)];
        self.interface.end = 0;
    }

    fn setEpoch(self: *DirectShardWriter, epoch: u64) std.Io.Writer.Error!void {
        if (self.epoch == epoch) return;
        if (self.interface.end > 0) try self.flushBuffered();
        self.epoch = epoch;
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, _: usize) std.Io.Writer.Error!usize {
        const self: *DirectShardWriter = @alignCast(@fieldParentPtr("interface", w));

        const chunk = data[0];
        if (chunk.len > self.interface.buffer.len) return std.Io.Writer.Error.WriteFailed;
        if (chunk.len > self.total - (self.offset + self.interface.end)) return std.Io.Writer.Error.WriteFailed;

        const needs_fresh_buffer = chunk.len > self.interface.buffer.len - self.interface.end;
        if (needs_fresh_buffer) {
            try self.interface.flush();
        }

        @memcpy(self.interface.buffer[self.interface.end..][0..chunk.len], chunk);
        self.interface.end += chunk.len;

        const buffer_full = self.interface.end == self.interface.buffer.len;
        if (buffer_full) {
            try self.interface.flush();
        }

        return chunk.len;
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *DirectShardWriter = @alignCast(@fieldParentPtr("interface", w));
        try self.flushBuffered();
    }

    fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *DirectShardWriter = @alignCast(@fieldParentPtr("interface", w));
        if (self.interface.buffer.len - self.interface.end >= capacity) return;
        if (preserve != 0) return std.Io.Writer.Error.WriteFailed;
        try self.interface.flush();
    }

    fn flushBuffered(self: *DirectShardWriter) std.Io.Writer.Error!void {
        if (self.offset >= self.total) return;

        const pjrt_api = self.memory.platform.pjrt_api;

        const current_buffer = self.interface.buffer;
        const buffered = self.interface.buffered();

        const slice = buffered[0..@min(buffered.len, self.total - self.offset)];
        if (slice.len == 0) return;
        const is_last = (self.offset + slice.len) >= self.total;

        const submitted_at: std.Io.Timestamp = .now(self.io, .awake);
        const transfer_event = self.transfer_manager.transferData(pjrt_api, 0, slice, @intCast(self.offset), is_last) catch |err| {
            self.recordError(err);
            log.err("error when transferring data to device: {any}", .{err});
            return std.Io.Writer.Error.WriteFailed;
        };
        self.submission_count += 1;

        const ctx = &self.events_contexts[@intCast(self.flip_flop)];
        ctx.* = .{
            .self = self,
            .buffer = current_buffer,
            .pjrt_event = transfer_event,
            .submitted_at = submitted_at,
            .bytes = slice.len,
            .epoch = self.epoch,
        };
        if (self.metrics) |metrics| {
            _ = metrics.dma_submissions.fetchAdd(1, .monotonic);
            _ = metrics.submitted_bytes.fetchAdd(@intCast(slice.len), .monotonic);
        }

        transfer_event.onReady(pjrt_api, EventContext, struct {
            fn call(err: ?*pjrt.Error, ctx_: *EventContext) void {
                if (err == null) {
                    if (ctx_.self.metrics) |metrics| {
                        const elapsed = ctx_.submitted_at.untilNow(ctx_.self.io, .awake);
                        const elapsed_us: u64 = @intCast(@max(@divTrunc(elapsed.nanoseconds, std.time.ns_per_us), 0));
                        _ = metrics.committed_bytes.fetchAdd(@intCast(ctx_.bytes), .monotonic);
                        metrics.last_dma_commit_ns.store(@intCast(@max(std.Io.Timestamp.now(ctx_.self.io, .awake).nanoseconds, 1)), .release);
                        _ = metrics.weighted_transfer_latency_us.fetchAdd(@as(u64, @intCast(ctx_.bytes)) *| elapsed_us, .monotonic);
                        metrics.recordProbeCommit(ctx_.self.io, ctx_.epoch, ctx_.bytes);
                    }
                } else if (err) |e| {
                    ctx_.self.recordError(e.getCode(ctx_.self.memory.platform.pjrt_api).toApiError());
                }
                ctx_.self.pool.put(ctx_.self.io, ctx_.buffer);
                ctx_.err = err;
                ctx_.event.set(ctx_.self.io);
            }
        }.call, &(ctx.*.?)) catch |err| {
            self.recordError(err);
            log.err("error when setting up transfer completion callback: {any}", .{err});
            transfer_event.awaitRaw(pjrt_api) catch {};
            transfer_event.deinit(pjrt_api);
            ctx.* = null;
            return std.Io.Writer.Error.WriteFailed;
        };
        self.active_buffer = null;

        if (self.events_contexts[@intCast(self.flip_flop ^ 1)]) |*ctx_previous| {
            defer self.events_contexts[@intCast(self.flip_flop ^ 1)] = null;
            const wait_started: std.Io.Timestamp = .now(self.io, .awake);
            ctx_previous.event.waitUncancelable(self.io);
            if (self.metrics) |metrics| metrics.addDmaCompletionWait(wait_started.untilNow(self.io, .awake));
            defer ctx_previous.pjrt_event.deinit(pjrt_api);
            if (ctx_previous.err) |e| {
                self.recordError(e.getCode(pjrt_api).toApiError());
                defer e.deinit(pjrt_api);
                log.err("error while awaiting: {s}: {s}", .{
                    @tagName(e.getCode(pjrt_api)),
                    e.getMessage(pjrt_api),
                });
                return std.Io.Writer.Error.WriteFailed;
            }
        }

        if (is_last) {
            defer ctx.* = null;
            defer self.interface = .failing;
            const ctx_ = &ctx.*.?;
            ctx_.event.waitUncancelable(self.io);
            defer ctx_.pjrt_event.deinit(pjrt_api);
            if (ctx_.err) |e| {
                self.recordError(e.getCode(pjrt_api).toApiError());
                defer e.deinit(pjrt_api);
                log.err("error while awaiting: {s}: {s}", .{
                    @tagName(e.getCode(pjrt_api)),
                    e.getMessage(pjrt_api),
                });
                return std.Io.Writer.Error.WriteFailed;
            }
        } else {
            self.interface.end = 0;
            const buf = getDmaBuffer(self.pool, self.allocator, self.io, self.metrics) catch |err| {
                self.recordError(err);
                log.err("unable to get a new buffer from the pool: {any}", .{err});
                return std.Io.Writer.Error.WriteFailed;
            };
            self.active_buffer = buf;
            self.interface.buffer = buf[0..@min(buf.len, self.total - (self.offset + slice.len))];
        }
        self.flip_flop ^= 1;
        self.offset += slice.len;
    }
};

const ShardProgress = struct {
    const scale: usize = 1024;

    node: std.Progress.Node,
    label: [32]u8 = undefined,
    completed: usize = 0,

    fn set(self: *ShardProgress, completed: usize) void {
        self.completed = completed;
        self.node.setCompletedItems(std.math.divCeil(usize, self.completed, scale) catch unreachable);
    }
};

// Dispatch planning bridges two different orders:
//
// 1. Placement traversal emits byte ranges per shard writer, in device order.
//    That order is convenient for asking "what bytes belong to this shard?",
//    but it does not match how a tensor file is read.
//
// 2. Readers stream tensor bytes in global row-major order. DirectMemoryWriter
//    must therefore consume dispatch spans in increasing global byte offset.
//
// We first collect placement spans, sort them by global byte range, then fold
// identical ranges into one primary dispatch span plus mirror writers. Identical
// ranges represent replicated shard data: the reader provides those bytes once,
// the primary writer receives them zero-copy, and mirrors receive a copy.
const DispatchSpans = struct {
    const DispatchSpan = struct {
        start: usize,
        end: usize,
        primary_writer: usize,
        mirror_writer_start: usize,
        mirror_writer_len: usize,
    };

    const PlacementSpan = struct {
        writer_index: usize,
        start: usize,
        len: usize,
        order: usize,
    };

    spans: []DispatchSpan,
    mirror_writers: []usize,

    fn init(allocator: std.mem.Allocator, shape: Shape, sharding: Sharding) !DispatchSpans {
        const placement = try sharding.placement(shape);
        const ordered_devices = sharding.devicesInCanonicalOrder();

        var placement_span_count: usize = 0;
        for (ordered_devices) |device| {
            placement_span_count += placementSpanCount(shape, placement.slices(device.coords).constSlice());
        }

        var placement_spans: std.ArrayList(PlacementSpan) = try .initCapacity(allocator, placement_span_count);
        defer placement_spans.deinit(allocator);

        const byte_strides = shape.computeByteStrides();

        for (ordered_devices, 0..) |device, writer_index| {
            appendShardPlacementSpans(&placement_spans, shape, placement.slices(device.coords).constSlice(), byte_strides.constSlice(), writer_index);
        }

        std.debug.assert(placement_spans.items.len == placement_span_count);

        var spans: std.ArrayList(DispatchSpan) = try .initCapacity(allocator, placement_spans.items.len);
        errdefer spans.deinit(allocator);

        var mirror_writers: std.ArrayList(usize) = try .initCapacity(allocator, placement_spans.items.len);
        errdefer mirror_writers.deinit(allocator);

        try deduplicateByRange(allocator, placement_spans.items, shape.byteSize(), &spans, &mirror_writers);

        const spans_ = try spans.toOwnedSlice(allocator);
        errdefer allocator.free(spans_);

        const mirror_writers_ = try mirror_writers.toOwnedSlice(allocator);
        errdefer allocator.free(mirror_writers_);

        return .{
            .spans = spans_,
            .mirror_writers = mirror_writers_,
        };
    }

    fn deinit(self: DispatchSpans, allocator: std.mem.Allocator) void {
        allocator.free(self.spans);
        allocator.free(self.mirror_writers);
    }

    fn deduplicateByRange(
        allocator: std.mem.Allocator,
        placement_spans: []PlacementSpan,
        total_bytes: usize,
        spans: *std.ArrayList(DispatchSpan),
        mirror_writers: *std.ArrayList(usize),
    ) !void {
        const SortContext = struct {
            fn lessThan(_: void, lhs: PlacementSpan, rhs: PlacementSpan) bool {
                if (lhs.start != rhs.start) return lhs.start < rhs.start;
                if (lhs.len != rhs.len) return lhs.len < rhs.len;
                return lhs.order < rhs.order;
            }
        };

        std.mem.sort(PlacementSpan, placement_spans, {}, SortContext.lessThan);

        var i: usize = 0;
        var cursor: usize = 0;
        while (i < placement_spans.len) {
            const span = placement_spans[i];
            if (span.start != cursor) return error.NonContiguousShardPlacement;

            const mirror_writer_start = mirror_writers.items.len;
            var j = i + 1;
            while (j < placement_spans.len) : (j += 1) {
                const mirror = placement_spans[j];
                if (mirror.start != span.start or mirror.len != span.len) break;
                try mirror_writers.append(allocator, mirror.writer_index);
            }

            try spans.append(allocator, .{
                .start = span.start,
                .end = span.start + span.len,
                .primary_writer = span.writer_index,
                .mirror_writer_start = mirror_writer_start,
                .mirror_writer_len = j - i - 1,
            });
            cursor += span.len;
            i = j;
        }

        if (cursor != total_bytes) return error.NonContiguousShardPlacement;
    }

    fn appendPlacementSpan(placement_spans: *std.ArrayList(PlacementSpan), writer_index: usize, start: usize, len: usize) void {
        placement_spans.appendAssumeCapacity(.{
            .writer_index = writer_index,
            .start = start,
            .len = len,
            .order = placement_spans.items.len,
        });
    }

    fn appendShardPlacementSpans(
        placement_spans: *std.ArrayList(PlacementSpan),
        shape: Shape,
        slices: []const Placement.Slice1d,
        byte_strides: []const i64,
        writer_index: usize,
    ) void {
        if (shape.rank() == 0) {
            appendPlacementSpan(placement_spans, writer_index, 0, shape.byteSize());
            return;
        }

        appendShardAxisPlacementSpans(placement_spans, slices, byte_strides, writer_index, 0, contiguousSliceAxis(shape, slices), 0);
    }

    fn appendShardAxisPlacementSpans(
        placement_spans: *std.ArrayList(PlacementSpan),
        slices: []const Placement.Slice1d,
        byte_strides: []const i64,
        writer_index: usize,
        axis: usize,
        contiguous_axis: usize,
        base_start: i64,
    ) void {
        const slice = slices[axis];
        if (slice.size == 0) return;

        if (axis == contiguous_axis) {
            const span_start: usize = @intCast(base_start + slice.start * byte_strides[axis]);
            const span_len: usize = @intCast(slice.size * byte_strides[axis]);
            appendPlacementSpan(placement_spans, writer_index, span_start, span_len);
            return;
        }

        var i: i64 = 0;
        while (i < slice.size) : (i += 1) {
            const child_start = base_start + (slice.start + i) * byte_strides[axis];
            appendShardAxisPlacementSpans(placement_spans, slices, byte_strides, writer_index, axis + 1, contiguous_axis, child_start);
        }
    }

    fn placementSpanCount(shape: Shape, slices: []const Placement.Slice1d) usize {
        if (shape.rank() == 0) return 1;

        const contiguous_axis = contiguousSliceAxis(shape, slices);
        var count: usize = 1;
        for (slices[0..contiguous_axis]) |slice| {
            count *= @intCast(slice.size);
        }
        return count;
    }

    fn contiguousSliceAxis(shape: Shape, slices: []const Placement.Slice1d) usize {
        var axis = shape.rank() - 1;
        while (axis > 0) {
            const slice = slices[axis];
            if (slice.start != 0 or slice.size != shape.dim(axis)) break;
            axis -= 1;
        }
        return axis;
    }
};

// Direct load writer state machine.
//
//     std.Io.Reader
//          |
//          v
//     DirectMemoryWriter.interface.buffer
//          | aliases active primary DirectShardWriter DMA buffer
//          v
//     commitWindow()
//          |-- primary: advance DirectShardWriter.interface.end
//          |-- mirrors: copy committed bytes with writeAll()
//          |-- boundaries: flush full shard buffers / chunk fence
//          v
//     DirectShardWriter.flushBuffered()
//          |
//          v
//     PJRT AsyncHostToDeviceTransferManager
//
// The public writer never owns a host staging buffer: it exposes a visible
// prefix of a real shard DMA buffer. The visible prefix may extend beyond the
// current dispatch span to coalesce reader requests, but `commitWindow` must
// scatter those bytes before submitting or rotating the active buffer. The
// first active-primary segment is zero-copy; crossed spans and mirrors treat
// the active buffer as scratch and copy into their own shard writers.
//
// `byte_cursor` is the global tensor position. Shard writer `interface.end` is
// local to the current shard DMA buffer.
pub const DirectMemoryWriter = struct {
    // Bound reader windows independently of DMA submission chunks so startup
    // can observe network progress before a large DMA chunk is complete.
    const observation_chunk_size = 32 * 1024 * 1024;

    allocator: std.mem.Allocator,
    shard_writers: []DirectShardWriter,
    // Global stream-order spans produced from placement; this is the routing table for committed bytes.
    dispatch_spans: DispatchSpans,
    // Current entry in `dispatch_spans.spans`; advances whenever `byte_cursor` reaches that span end.
    span_index: usize = 0,
    // Global tensor byte offset already scattered into shard writers.
    byte_cursor: usize = 0,
    // Global bytes known to have crossed a shard-writer submission boundary.
    logical_submitted_cursor: usize = 0,
    // Shard writer whose DMA buffer is currently exposed as `interface.buffer`.
    active_writer_index: usize,
    // Start offset of new reader bytes inside the active shard writer buffer.
    window_start: usize,
    // Logical maximum public alias window before forcing a cross-shard flush fence.
    dma_chunk_size: usize,
    shard_progress: ?[]ShardProgress = null,
    metrics: ?*LoadMetrics,
    epoch: u64 = 0,
    interface: std.Io.Writer,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: []mem.DynamicBufferPool,
        dma_allocators: []const mem.DmaAllocator,
        dma_chunk_size: usize,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
    ) !DirectMemoryWriter {
        return initWithMetrics(allocator, io, platform, pools, dma_allocators, dma_chunk_size, shape, sharding, buffer, null);
    }

    fn initWithMetrics(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: []mem.DynamicBufferPool,
        dma_allocators: []const mem.DmaAllocator,
        dma_chunk_size: usize,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
        metrics: ?*LoadMetrics,
    ) !DirectMemoryWriter {
        const ordered_devices = sharding.devicesInCanonicalOrder();
        var shard_writers = try allocator.alloc(DirectShardWriter, ordered_devices.len);
        errdefer allocator.free(shard_writers);

        var initialized: usize = 0;
        errdefer for (shard_writers[0..initialized]) |*writer| {
            writer.deinit();
        };

        var pjrt_buffers: Buffer.Shards = .empty;
        const placement = try sharding.placement(shape);
        for (ordered_devices, 0..) |device, i| {
            defer initialized += 1;

            const pool = &pools[device.id];
            const shard_dma_allocator = dma_allocators[device.id].allocator();
            const pjrt_mem = platform.devices[device.id].memory(.default).?;

            shard_writers[i] = try .init(shard_dma_allocator, io, pjrt_mem, pool, placement.shape, metrics);

            pjrt_buffers.appendAssumeCapacity(shard_writers[i].pjrt_buffer);
        }

        buffer.* = .fromPjrtBuffers(platform, shape, sharding, pjrt_buffers.constSlice());

        const dispatch_spans: DispatchSpans = try .init(allocator, shape, sharding);
        errdefer dispatch_spans.deinit(allocator);

        const first_span = dispatch_spans.spans[0];
        const first_writer = &shard_writers[first_span.primary_writer];
        const first_window = @min(observation_chunk_size, @min(dma_chunk_size, first_writer.interface.buffer.len));

        return .{
            .allocator = allocator,
            .shard_writers = shard_writers,
            .dispatch_spans = dispatch_spans,
            .active_writer_index = first_span.primary_writer,
            .window_start = first_writer.interface.end,
            .dma_chunk_size = dma_chunk_size,
            .metrics = metrics,
            .interface = .{
                .buffer = first_writer.interface.buffer[0..first_window],
                .end = 0,
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                },
            },
        };
    }

    pub fn deinit(self: *DirectMemoryWriter) void {
        self.setProgress(null);
        for (self.shard_writers) |*writer| {
            writer.deinit();
        }
        self.allocator.free(self.shard_writers);
        self.dispatch_spans.deinit(self.allocator);
    }

    fn park(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        try self.commitWindow();
        for (self.shard_writers) |*writer| try writer.park();
        self.recordLogicalSubmission();
        self.interface.buffer = &.{};
        self.interface.end = 0;
        self.window_start = 0;
    }

    fn parkAndWait(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        try self.park();
        for (self.shard_writers) |*writer| try writer.waitForPendingTransfers();
    }

    fn waitForPendingDma(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        for (self.shard_writers) |*writer| try writer.waitForPendingTransfers();
    }

    fn submissionCount(self: *const DirectMemoryWriter) u64 {
        var total: u64 = 0;
        for (self.shard_writers) |writer| total +|= writer.submission_count;
        return total;
    }

    fn unpark(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        for (self.shard_writers) |*writer| try writer.unpark();
        try self.publishWindow();
    }

    pub fn setProgress(self: *DirectMemoryWriter, progress: ?*std.Progress.Node) void {
        const parent = progress orelse {
            const states = self.shard_progress orelse return;
            for (states) |*s| {
                s.node.end();
            }
            self.allocator.free(states);
            self.shard_progress = null;
            return;
        };

        std.debug.assert(self.shard_progress == null);
        const states = self.allocator.alloc(ShardProgress, self.shard_writers.len) catch return;
        for (states, self.shard_writers, 0..) |*state, writer, i| {
            state.completed = 0;
            const label = std.fmt.bufPrint(&state.label, "shard[{d}]", .{i}) catch unreachable;
            const total_items = std.math.divCeil(usize, writer.total, ShardProgress.scale) catch unreachable;
            state.node = parent.start(label, total_items);
            state.node.setCompletedItems(0);
        }
        self.shard_progress = states;
    }

    fn setEpoch(self: *DirectMemoryWriter, epoch: u64) std.Io.Writer.Error!void {
        if (self.epoch == epoch) return;
        try self.commitWindow();
        for (self.shard_writers) |*writer| try writer.setEpoch(epoch);
        self.recordLogicalSubmission();
        self.epoch = epoch;
        try self.publishWindow();
    }

    fn directWritableSlice(self: *DirectMemoryWriter) []u8 {
        return self.interface.buffer[self.interface.end..];
    }

    fn commitDirectRead(self: *DirectMemoryWriter, len: usize) std.Io.Writer.Error!void {
        if (len > self.interface.buffer.len - self.interface.end) return std.Io.Writer.Error.WriteFailed;
        self.interface.end += len;
        try self.commitWindow();
    }

    fn commitStagedWrite(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        try self.commitWindow();
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, _: usize) std.Io.Writer.Error!usize {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        const chunk = data[0];
        try self.commitWindow();

        const writable = self.interface.buffer.len - self.interface.end;
        if (writable == 0) return std.Io.Writer.Error.WriteFailed;

        const n = @min(writable, chunk.len);
        @memcpy(self.interface.buffer[self.interface.end..][0..n], chunk[0..n]);
        self.interface.end += n;
        try self.commitWindow();

        return n;
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        // Commit the active public window if one remains
        if (self.span_index < self.dispatch_spans.spans.len) {
            try self.commitWindow();
        }

        for (self.shard_writers, 0..) |*shard_writer, i| {
            try shard_writer.interface.flush();
            if (self.shard_progress) |states| {
                states[i].set(shard_writer.total);
            }
        }
        self.recordLogicalSubmission();

        self.interface = .failing;
    }

    fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        if (self.interface.buffer.len - self.interface.end >= capacity) return;
        if (preserve != 0) return std.Io.Writer.Error.WriteFailed;

        try self.commitWindow();
        if (self.interface.buffer.len - self.interface.end < capacity) return std.Io.Writer.Error.WriteFailed;
    }

    // `interface.buffer` is a clipped alias of the active primary shard DMA
    // buffer. Reader writes advance only `DirectMemoryWriter.interface.end`;
    // this function scatters that visible range through dispatch spans.
    //
    // The leading segment for `active_writer_index` is already in the right
    // DMA buffer and is committed by advancing that shard writer. Later spans
    // use the same bytes as scratch and are copied before any full buffers are
    // submitted or the public alias is rotated.
    fn commitWindow(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        // All dispatch spans are consumed; no shard DMA buffer remains valid to
        // expose as the public alias until flush marks the writer as failing.
        if (self.span_index >= self.dispatch_spans.spans.len) {
            self.interface.buffer = &.{};
            self.interface.end = 0;
            return;
        }

        // Only bytes after `window_start` are new reader writes. Bytes before
        // it may already be pending in the active shard writer.
        const source = self.interface.buffer[self.window_start..self.interface.end];
        if (source.len == 0) return;

        // The visible alias window may cross many dispatch spans. Walk the
        // global cursor and route each subrange to its primary and mirrors.
        var consumed: usize = 0;
        while (consumed < source.len) {
            if (self.span_index >= self.dispatch_spans.spans.len) return std.Io.Writer.Error.WriteFailed;

            const span = self.dispatch_spans.spans[self.span_index];
            if (self.byte_cursor < span.start or self.byte_cursor >= span.end) return std.Io.Writer.Error.WriteFailed;

            const n = @min(source.len - consumed, span.end - self.byte_cursor);
            const chunk = source[consumed..][0..n];

            // The first span in the active alias is already in the primary DMA
            // buffer. Later ranges use the active buffer as scratch and must be
            // copied into their real destination writer.
            if (span.primary_writer == self.active_writer_index and consumed == 0) {
                self.shard_writers[span.primary_writer].interface.end += n;
            } else {
                const primary_writer = &self.shard_writers[span.primary_writer];
                if (span.primary_writer == self.active_writer_index) {
                    @memmove(primary_writer.interface.buffer[primary_writer.interface.end..][0..n], chunk);
                    primary_writer.interface.end += n;
                } else {
                    try primary_writer.interface.writeAll(chunk);
                }
            }
            if (self.shard_progress) |states| {
                states[span.primary_writer].set(states[span.primary_writer].completed + n);
            }

            const mirror_writer_end = span.mirror_writer_start + span.mirror_writer_len;
            for (self.dispatch_spans.mirror_writers[span.mirror_writer_start..mirror_writer_end]) |mirror_writer_index| {
                const mirror_writer = &self.shard_writers[mirror_writer_index];
                if (mirror_writer_index == self.active_writer_index) {
                    @memmove(mirror_writer.interface.buffer[mirror_writer.interface.end..][0..n], chunk);
                    mirror_writer.interface.end += n;
                } else {
                    try mirror_writer.interface.writeAll(chunk);
                }
                if (self.shard_progress) |states| {
                    states[mirror_writer_index].set(states[mirror_writer_index].completed + n);
                }
            }

            self.byte_cursor += n;
            consumed += n;
            if (self.byte_cursor == span.end) {
                self.span_index += 1;
            }
        }

        // All scratch copies are complete now, so full DMA buffers may be
        // submitted without invalidating bytes still needed for scatter.
        for (self.shard_writers) |*writer| {
            if (writer.interface.end == writer.interface.buffer.len) {
                try writer.interface.flush();
            }
        }

        if (@mod(self.byte_cursor, self.dma_chunk_size) == 0) {
            for (self.shard_writers) |*writer| {
                try writer.interface.flush();
            }
            self.recordLogicalSubmission();
        }

        try self.publishWindow();
    }

    fn recordLogicalSubmission(self: *DirectMemoryWriter) void {
        const submitted = self.byte_cursor - self.logical_submitted_cursor;
        if (submitted == 0) return;
        if (self.metrics) |metrics| {
            _ = metrics.logical_submitted_bytes.fetchAdd(@intCast(submitted), .monotonic);
        }
        self.logical_submitted_cursor = self.byte_cursor;
    }

    fn publishWindow(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        // The last committed window may finish the tensor exactly. Leave an
        // empty public window so any further writes fail through std.Io.
        if (self.span_index >= self.dispatch_spans.spans.len) {
            self.interface.buffer = &.{};
            self.interface.end = 0;
            return;
        }

        // Rotate the public alias to the primary writer for the next stream
        // span. `window_start` preserves that writer's already-buffered prefix
        // so only newly advanced bytes are scattered on the next commit.
        const next_span = self.dispatch_spans.spans[self.span_index];
        const next_writer = &self.shard_writers[next_span.primary_writer];
        self.active_writer_index = next_span.primary_writer;
        self.window_start = next_writer.interface.end;

        // Publish the next public alias. Its backing memory is the next primary
        // shard buffer, but the visible length is allowed to coalesce across
        // upcoming spans until a buffer or tensor boundary.
        const total = self.dispatch_spans.spans[self.dispatch_spans.spans.len - 1].end;
        const buffer_remaining = next_writer.interface.buffer.len - self.window_start;
        const chunk_offset = @mod(self.byte_cursor, self.dma_chunk_size);
        const chunk_remaining = if (chunk_offset == 0) self.dma_chunk_size else self.dma_chunk_size - chunk_offset;
        const observation_offset = @mod(self.byte_cursor, observation_chunk_size);
        const observation_remaining = if (observation_offset == 0) observation_chunk_size else observation_chunk_size - observation_offset;
        const tensor_remaining = total - self.byte_cursor;
        const visible_remaining = @min(buffer_remaining, @min(chunk_remaining, @min(observation_remaining, tensor_remaining)));
        if (visible_remaining == 0) return std.Io.Writer.Error.WriteFailed;

        self.interface.buffer = next_writer.interface.buffer[0 .. self.window_start + visible_remaining];
        self.interface.end = self.window_start;
    }
};

const AdaptiveLoadController = struct {
    const Mode = enum { startup, steady };
    const Dimension = enum { read, dma, pinned, staging };
    const ProbeKind = enum { increase, reduce_resource };

    const Knobs = struct {
        read_workers: usize,
        dma_workers: usize,
        dma_chunks: usize,
        staging_chunks: usize,
    };

    const Probe = struct {
        dimension: Dimension,
        kind: ProbeKind,
        baseline: Knobs,
        candidate: Knobs,
        epoch: u64,
        baseline_goodput: f64,
    };

    const Sample = struct {
        read_goodput: f64 = 0,
        committed_goodput: f64 = 0,
        probe_goodput: f64 = 0,
        probe_committed_bytes: u64 = 0,
        capacity_pending: bool = false,
        read_latency_us: f64 = 0,
        read_admission_wait_ratio: f64 = 0,
        transfer_latency_us: f64 = 0,
        dma_latency_reliable: bool = true,
        pinned_wait_ratio: f64 = 0,
        dma_completion_wait_ratio: f64 = 0,
        staging_wait_ratio: f64 = 0,
        dma_starvation_ratio: f64 = 0,
        ready_bytes: u64 = 0,
        ready_queue_ratio: f64 = 0,
        ready_age_pressure: f64 = 0,
        ready_growth_ratio: f64 = 0,
        h2d_queue_ratio: f64 = 0,
        h2d_growth_ratio: f64 = 0,
        admitted_dma_writers: usize = 0,
        dma_lane_capacity_available: bool = false,
        slow_reads: bool = false,
        source_stalled: bool = false,
        read_capacity_demand: bool = false,
        reads_saturated: bool = false,
        dma_saturated: bool = false,
        allow_probe: bool = true,
        now_ns: u64 = 0,
    };

    const Decision = struct {
        const Action = enum {
            none,
            read_ahead_bootstrap,
            read_backoff,
            staging_backoff,
            dma_backoff,
            staging_probe_start,
            staging_probe_keep,
            staging_probe_rollback,
            read_probe_start,
            read_probe_keep,
            read_probe_rollback,
            pinned_probe_start,
            pinned_probe_keep,
            pinned_probe_rollback,
            dma_probe_start,
            dma_probe_keep,
            dma_probe_rollback,
            pinned_reduce_start,
            pinned_reduce_keep,
            pinned_reduce_rollback,
            staging_reduce_start,
            staging_reduce_keep,
            staging_reduce_rollback,
        };

        const RollbackReason = enum {
            none,
            gain_below_threshold,
            dma_starvation,
            ready_queue_growth,
            read_pressure,
            h2d_pressure,
            pinned_wait,
            capacity_not_exercised,
        };

        knobs: Knobs,
        epoch: u64,
        changed: bool = false,
        trim_pinned: bool = false,
        trim_staging: bool = false,
        action: Action = .none,
        reason: RollbackReason = .none,
    };

    mode: Mode = .startup,
    max_read_workers: usize,
    max_dma_workers: usize,
    max_dma_chunks: usize,
    max_staging_chunks: usize,
    direct: bool,
    knobs: Knobs,
    peak_goodput: f64 = 0,
    latency_baseline_us: f64 = 0,
    probe: ?Probe = null,
    epoch: u64 = 0,
    last_probe_ns: u64 = 0,
    last_resource_probe_ns: u64 = 0,
    last_probe_dimension: ?Dimension = null,
    last_probe_kind: ?ProbeKind = null,
    last_probe_baseline_goodput: f64 = 0,
    h2d_growth_windows: u8 = 0,
    pressure_backoff_blocked_until_ns: u64 = 0,

    const probe_byte_floor: u64 = 64 * 1024 * 1024;

    fn init(
        initial_dma_workers: usize,
        max_dma_workers: usize,
        initial_dma_chunks: usize,
        max_dma_chunks: usize,
        initial_read_workers: usize,
        max_read_workers: usize,
        max_staging_chunks: usize,
        direct: bool,
    ) AdaptiveLoadController {
        return .{
            .max_read_workers = max_read_workers,
            .max_dma_workers = max_dma_workers,
            .max_dma_chunks = max_dma_chunks,
            .max_staging_chunks = max_staging_chunks,
            .direct = direct,
            .knobs = .{
                .read_workers = initial_read_workers,
                .dma_workers = initial_dma_workers,
                .dma_chunks = initial_dma_chunks,
                .staging_chunks = 0,
            },
        };
    }

    fn observe(self: *AdaptiveLoadController, sample: Sample) Decision {
        if (sample.capacity_pending) return self.currentDecision();
        const measured_goodput = sample.committed_goodput;
        if (self.probe == null) self.peak_goodput = @max(self.peak_goodput, measured_goodput);

        const lightly_loaded = self.knobs.dma_workers <= 2 and sample.pinned_wait_ratio <= 0.02 and
            sample.h2d_queue_ratio <= 0.10 and sample.ready_queue_ratio <= 0.10;
        if (sample.dma_latency_reliable and sample.transfer_latency_us > 0) {
            if (self.latency_baseline_us == 0) {
                self.latency_baseline_us = sample.transfer_latency_us;
            } else if (lightly_loaded) {
                self.latency_baseline_us = 0.95 * self.latency_baseline_us + 0.05 * sample.transfer_latency_us;
            }
        }

        const latency_inflation = if (sample.dma_latency_reliable and sample.transfer_latency_us > 0 and self.latency_baseline_us > 0)
            @max(0, sample.transfer_latency_us / self.latency_baseline_us - 1)
        else
            0;
        if (measured_goodput > 0 and sample.h2d_growth_ratio > 0.20) {
            self.h2d_growth_windows = @min(2, self.h2d_growth_windows +| 1);
        } else {
            self.h2d_growth_windows = 0;
        }
        const persistent_h2d_growth = if (self.h2d_growth_windows >= 2) sample.h2d_growth_ratio else 0;
        const h2d_pressure = @max(
            latency_inflation,
            @max(sample.pinned_wait_ratio, @max(sample.dma_completion_wait_ratio, @max(sample.h2d_growth_ratio, @max(0, sample.h2d_queue_ratio - 0.75)))),
        );
        const h2d_backoff_pressure = @max(
            latency_inflation,
            @max(sample.pinned_wait_ratio, @max(sample.dma_completion_wait_ratio, @max(persistent_h2d_growth, @max(0, sample.h2d_queue_ratio - 0.75)))),
        );
        const ready_pressure = @max(sample.ready_growth_ratio, @max(sample.ready_age_pressure, @max(0, sample.ready_queue_ratio - 0.75)));
        // Admission wait means the configured read capacity is in demand, and
        // request latency normally rises with useful source parallelism. Only
        // buffered data that the DMA stage cannot drain is read-side pressure.
        const read_queue_pressure = ready_pressure;
        const read_pressure_reason: Decision.RollbackReason = if (ready_pressure > 0.10) .ready_queue_growth else .read_pressure;
        const ready_per_dma: usize = if (ready_pressure > 0.20) 1 else 2;
        const pressured_staging_target = if (ready_pressure > 0.10)
            @min(self.knobs.staging_chunks, self.knobs.dma_workers *| ready_per_dma)
        else
            self.knobs.staging_chunks;

        if ((measured_goodput > 0 or self.probe != null) and h2d_backoff_pressure > 0.20) {
            const reason: Decision.RollbackReason = if (sample.pinned_wait_ratio > 0.20) .pinned_wait else .h2d_pressure;
            if (self.probe) |probe| {
                return self.restoreProbeBaseline(probe, sample.now_ns, reason);
            }
            if (sample.now_ns < self.pressure_backoff_blocked_until_ns) return self.currentDecision();
            if (self.knobs.dma_workers > 1) {
                self.mode = .steady;
                self.epoch += 1;
                self.knobs.dma_workers = @max(@as(usize, 1), @as(usize, @intFromFloat(@floor(0.70 * @as(f64, @floatFromInt(self.knobs.dma_workers))))));
                self.knobs.dma_chunks = @max(self.minimumDmaChunks(self.knobs.dma_workers), @min(self.knobs.dma_chunks, self.max_dma_chunks));
                self.last_probe_ns = sample.now_ns;
                return self.decision(.dma_backoff, true, false, false, reason);
            }
        }

        const ready_pressure_rejects_probe = if (self.probe) |probe| rejectsReadyPressure(probe) else false;
        if (read_queue_pressure > 0.10 and
            (ready_pressure_rejects_probe or
                (self.probe == null and (self.knobs.read_workers > 1 or pressured_staging_target < self.knobs.staging_chunks))))
        {
            if (self.probe) |probe| {
                return self.restoreProbeBaseline(probe, sample.now_ns, read_pressure_reason);
            }
            if (sample.now_ns < self.pressure_backoff_blocked_until_ns) return self.currentDecision();
            self.mode = .steady;
            self.epoch += 1;
            const beta: f64 = if (read_queue_pressure > 0.20) 0.70 else 0.85;
            const old_read_workers = self.knobs.read_workers;
            if (self.knobs.read_workers > 1) {
                self.knobs.read_workers = @max(@as(usize, 1), @as(usize, @intFromFloat(@floor(beta * @as(f64, @floatFromInt(self.knobs.read_workers))))));
            }
            var trim_staging = false;
            if (ready_pressure > 0.10 and self.knobs.staging_chunks > 0) {
                const staging_target = @min(self.knobs.staging_chunks, self.knobs.dma_workers *| ready_per_dma);
                trim_staging = staging_target < self.knobs.staging_chunks;
                self.knobs.staging_chunks = staging_target;
            }
            self.last_probe_ns = sample.now_ns;
            const action: Decision.Action = if (self.knobs.read_workers < old_read_workers) .read_backoff else .staging_backoff;
            return self.decision(action, true, false, trim_staging, read_pressure_reason);
        }

        if (self.probe) |probe| {
            if (sample.probe_committed_bytes < probe_byte_floor) return self.currentDecision();

            const pressure_ok = h2d_backoff_pressure < 0.10 and
                (!rejectsReadyPressure(probe) or read_queue_pressure < 0.10);
            const resource_reference = @max(probe.baseline_goodput, self.peak_goodput);
            const keep = switch (probe.kind) {
                .increase => sample.probe_goodput >= probe.baseline_goodput * 1.03 and pressure_ok,
                .reduce_resource => sample.probe_goodput >= resource_reference * 0.97 and pressure_ok and sample.dma_starvation_ratio <= 0.10,
            };

            if (keep) {
                self.probe = null;
                self.last_probe_ns = sample.now_ns;
                if (probe.kind == .reduce_resource) self.last_resource_probe_ns = sample.now_ns;
                if (probe.kind == .reduce_resource) self.mode = .steady;
                self.peak_goodput = @max(self.peak_goodput, sample.probe_goodput);
                return self.decision(probeAction(probe.dimension, probe.kind, true), false, false, false, .none);
            }

            const reason: Decision.RollbackReason = if (!pressure_ok)
                if (rejectsReadyPressure(probe) and read_queue_pressure >= 0.10) read_pressure_reason else .h2d_pressure
            else if (probe.kind == .reduce_resource and sample.dma_starvation_ratio > 0.10)
                .dma_starvation
            else
                .gain_below_threshold;
            return self.restoreProbeBaseline(probe, sample.now_ns, reason);
        }

        if (!sample.allow_probe) return self.currentDecision();
        const read_capacity_demand = sample.read_capacity_demand or sample.reads_saturated;
        if (measured_goodput <= 0) {
            if ((sample.slow_reads or sample.source_stalled) and read_capacity_demand and read_queue_pressure < 0.10) {
                const candidate = self.readAheadCandidate() orelse return self.currentDecision();
                self.epoch += 1;
                self.last_probe_dimension = if (candidate.staging_chunks > self.knobs.staging_chunks) .staging else .read;
                self.last_probe_kind = .increase;
                self.last_probe_baseline_goodput = 0;
                self.last_probe_ns = sample.now_ns;
                self.knobs = candidate;
                return self.decision(.read_ahead_bootstrap, true, false, false, .none);
            }
            return self.currentDecision();
        }

        const performance_probe_due = self.mode == .startup or sample.now_ns -| self.last_probe_ns >= 500 * std.time.ns_per_ms;
        const source_capacity_exercised = sample.reads_saturated or sample.slow_reads;
        if (performance_probe_due and sample.dma_starvation_ratio > 0.10 and read_capacity_demand and source_capacity_exercised and read_queue_pressure < 0.10) {
            if (self.readAheadCandidate()) |candidate| {
                const dimension: Dimension = if (candidate.staging_chunks > self.knobs.staging_chunks) .staging else .read;
                return self.startProbe(dimension, .increase, candidate, measured_goodput, sample.now_ns);
            }
        }

        if (self.mode == .steady and sample.dma_starvation_ratio <= 0.10 and
            self.knobs.read_workers > self.knobs.dma_workers and
            sample.now_ns -| self.last_resource_probe_ns >= 500 * std.time.ns_per_ms)
        {
            const resource_baseline = @max(self.peak_goodput, measured_goodput);
            var candidate = self.knobs;
            candidate.read_workers = @max(
                self.knobs.dma_workers,
                std.math.divCeil(usize, self.knobs.read_workers, 2) catch unreachable,
            );
            candidate.staging_chunks = if (self.knobs.staging_chunks == 0)
                self.minimumStagingChunks(candidate.read_workers, candidate.dma_workers)
            else
                @min(self.max_staging_chunks, candidate.read_workers);
            return self.startProbe(.read, .reduce_resource, candidate, resource_baseline, sample.now_ns);
        }

        if (self.mode == .steady and sample.dma_starvation_ratio <= 0.10 and sample.now_ns -| self.last_resource_probe_ns >= 2 * std.time.ns_per_s) {
            const resource_baseline = @max(self.peak_goodput, measured_goodput);
            const minimum_chunks = self.minimumDmaChunks(self.knobs.dma_workers);
            if (self.knobs.dma_chunks > minimum_chunks and self.knobs.dma_chunks - 1 >= sample.admitted_dma_writers) {
                var candidate = self.knobs;
                candidate.dma_chunks -= 1;
                return self.startProbe(.pinned, .reduce_resource, candidate, resource_baseline, sample.now_ns);
            }
            const minimum_staging = self.minimumStagingChunks(self.knobs.read_workers, self.knobs.dma_workers);
            if (self.knobs.staging_chunks > minimum_staging) {
                var candidate = self.knobs;
                candidate.staging_chunks -= 1;
                return self.startProbe(.staging, .reduce_resource, candidate, resource_baseline, sample.now_ns);
            }
            if (self.knobs.dma_workers > 1) {
                var candidate = self.knobs;
                candidate.dma_workers = @max(
                    @as(usize, 1),
                    self.knobs.dma_workers - @max(@as(usize, 1), std.math.sqrt(self.knobs.dma_workers)),
                );
                return self.startProbe(.dma, .reduce_resource, candidate, resource_baseline, sample.now_ns);
            }
        }

        const unused_direct_read_capacity = self.knobs.read_workers > self.knobs.dma_workers and
            sample.admitted_dma_writers >= self.knobs.dma_workers and
            !sample.reads_saturated and sample.dma_lane_capacity_available;
        const direct_probe_demand = self.direct and sample.read_goodput > 0 and
            (sample.dma_starvation_ratio <= 0.10 or unused_direct_read_capacity);
        if (performance_probe_due and (sample.dma_saturated or direct_probe_demand) and h2d_pressure < 0.10) {
            if (self.knobs.dma_workers < self.max_dma_workers) {
                var candidate = self.knobs;
                const step = if (self.mode == .startup) self.knobs.dma_workers else @max(@as(usize, 1), std.math.sqrt(self.knobs.dma_workers));
                candidate.dma_workers = @min(self.max_dma_workers, self.knobs.dma_workers + step);
                candidate.dma_chunks = @max(candidate.dma_chunks, self.minimumDmaChunks(candidate.dma_workers));
                return self.startProbe(.dma, .increase, candidate, measured_goodput, sample.now_ns);
            }
            if (sample.pinned_wait_ratio > 0.05 and self.knobs.dma_chunks < self.max_dma_chunks) {
                var candidate = self.knobs;
                candidate.dma_chunks += 1;
                return self.startProbe(.pinned, .increase, candidate, measured_goodput, sample.now_ns);
            }
        }

        self.mode = .steady;
        return self.currentDecision();
    }

    fn startProbe(self: *AdaptiveLoadController, dimension: Dimension, kind: ProbeKind, candidate: Knobs, baseline_goodput: f64, now_ns: u64) Decision {
        self.epoch += 1;
        self.last_probe_dimension = dimension;
        self.last_probe_kind = kind;
        self.last_probe_baseline_goodput = baseline_goodput;
        self.probe = .{
            .dimension = dimension,
            .kind = kind,
            .baseline = self.knobs,
            .candidate = candidate,
            .epoch = self.epoch,
            .baseline_goodput = baseline_goodput,
        };
        self.knobs = candidate;
        self.last_probe_ns = now_ns;
        return self.decision(probeAction(dimension, kind, null), true, false, false, .none);
    }

    fn minimumDmaChunks(self: *const AdaptiveLoadController, dma_workers: usize) usize {
        if (!self.direct) return self.max_dma_chunks;
        return @min(self.max_dma_chunks, dma_workers + 1);
    }

    fn minimumStagingChunks(self: *const AdaptiveLoadController, read_workers: usize, dma_workers: usize) usize {
        return @min(self.max_staging_chunks, read_workers -| dma_workers);
    }

    fn readAheadCandidate(self: *const AdaptiveLoadController) ?Knobs {
        const useful_read_cap = @min(self.max_read_workers, self.knobs.dma_workers +| self.max_staging_chunks);
        if (self.knobs.read_workers >= useful_read_cap) return null;

        var candidate = self.knobs;
        candidate.read_workers = @min(
            useful_read_cap,
            @max(self.knobs.read_workers + 1, self.knobs.read_workers *| 2),
        );
        const required_staging = if (self.knobs.staging_chunks == 0)
            self.minimumStagingChunks(candidate.read_workers, candidate.dma_workers)
        else
            @min(self.max_staging_chunks, candidate.read_workers);
        candidate.staging_chunks = @max(self.knobs.staging_chunks, required_staging);
        return candidate;
    }

    fn decision(
        self: *const AdaptiveLoadController,
        action: Decision.Action,
        changed: bool,
        trim_pinned: bool,
        trim_staging: bool,
        reason: Decision.RollbackReason,
    ) Decision {
        return .{
            .knobs = self.knobs,
            .epoch = self.epoch,
            .changed = changed,
            .trim_pinned = trim_pinned,
            .trim_staging = trim_staging,
            .action = action,
            .reason = reason,
        };
    }

    fn currentDecision(self: *const AdaptiveLoadController) Decision {
        return self.decision(.none, false, false, false, .none);
    }

    fn rollbackTimedOutProbe(self: *AdaptiveLoadController, now_ns: u64, published_epoch: u64) ?Decision {
        const probe = self.probe orelse return null;
        self.epoch = published_epoch;
        return self.restoreProbeBaseline(probe, now_ns, .capacity_not_exercised);
    }

    fn restoreProbeBaseline(
        self: *AdaptiveLoadController,
        probe: Probe,
        now_ns: u64,
        reason: Decision.RollbackReason,
    ) Decision {
        self.probe = null;
        self.knobs = probe.baseline;
        self.mode = .steady;
        self.last_probe_ns = now_ns;
        if (probe.kind == .reduce_resource) self.last_resource_probe_ns = now_ns;
        self.h2d_growth_windows = 0;
        self.pressure_backoff_blocked_until_ns = now_ns +| 250 * std.time.ns_per_ms;
        return self.decision(
            probeAction(probe.dimension, probe.kind, false),
            true,
            probe.kind == .increase and probe.candidate.dma_chunks > probe.baseline.dma_chunks,
            probe.kind == .increase and probe.candidate.staging_chunks > probe.baseline.staging_chunks,
            reason,
        );
    }

    fn rejectsReadyPressure(probe: Probe) bool {
        return probe.kind == .reduce_resource or probe.dimension == .read or probe.dimension == .staging;
    }

    fn probeAction(dimension: Dimension, kind: ProbeKind, kept: ?bool) Decision.Action {
        return switch (dimension) {
            .read => if (kept == null) .read_probe_start else if (kept.?) .read_probe_keep else .read_probe_rollback,
            .dma => if (kept == null) .dma_probe_start else if (kept.?) .dma_probe_keep else .dma_probe_rollback,
            .pinned => switch (kind) {
                .increase => if (kept == null) .pinned_probe_start else if (kept.?) .pinned_probe_keep else .pinned_probe_rollback,
                .reduce_resource => if (kept == null) .pinned_reduce_start else if (kept.?) .pinned_reduce_keep else .pinned_reduce_rollback,
            },
            .staging => switch (kind) {
                .increase => if (kept == null) .staging_probe_start else if (kept.?) .staging_probe_keep else .staging_probe_rollback,
                .reduce_resource => if (kept == null) .staging_reduce_start else if (kept.?) .staging_reduce_keep else .staging_reduce_rollback,
            },
        };
    }
};

const AdaptiveLoadRuntime = struct {
    const ProbeActivation = struct {
        epoch: u64,
        dimension: AdaptiveLoadController.Dimension,
        kind: AdaptiveLoadController.ProbeKind,
        baseline: AdaptiveLoadController.Knobs,
        candidate: AdaptiveLoadController.Knobs,
        installed_at: std.Io.Timestamp,
        controller_now_ns: u64,
    };

    const ProbeTransition = enum { none, activated, rolled_back };

    controller: AdaptiveLoadController,
    pipeline: ?*AdaptivePipelineContext = null,
    dma_group: *stdx.Io.LimitedGroup,
    read_group: *stdx.Io.LimitedGroup,
    pools: []mem.DynamicBufferPool,
    dma_allocators: []const mem.DmaAllocator,
    staging_pool: ?*mem.DynamicBufferPool,
    staging_allocator: std.mem.Allocator,
    metrics: *LoadMetrics,
    dma_chunk_size: usize,
    read_chunk_size: usize,
    total_logical_bytes: u64,
    total_transfers: usize,
    probe_started: std.Io.Timestamp,
    pending_probe_activation: ?ProbeActivation = null,
    done: std.atomic.Value(bool) = .init(false),

    fn run(self: *AdaptiveLoadRuntime, io: std.Io) std.Io.Cancelable!void {
        const started: std.Io.Timestamp = .now(io, .awake);
        var window_started = started;
        var previous = self.metrics.snapshot();
        var previous_queue = previous.submitted_bytes -| previous.committed_bytes;
        var last_idle_log_ns: u64 = 0;
        load_log.debug("controller started: mode={s}, reads={d}/{d}, dma={d}/{d}, dma_chunks={d}/{d}, staging={d}/{d}, transfers={d}, logical_bytes={Bi:.2}", .{
            @tagName(self.controller.mode),
            self.controller.knobs.read_workers,
            self.controller.max_read_workers,
            self.controller.knobs.dma_workers,
            self.controller.max_dma_workers,
            self.controller.knobs.dma_chunks,
            self.controller.max_dma_chunks,
            self.controller.knobs.staging_chunks,
            self.controller.max_staging_chunks,
            self.total_transfers,
            self.total_logical_bytes,
        });
        defer {
            const final = self.metrics.snapshot();
            load_log.debug("controller stopped: mode={s}, reads={d}, dma={d}, dma_chunks={d}, staging={d}, completed={d}/{d}, read={Bi:.2}, direct={Bi:.2}, staged={Bi:.2}, submitted={Bi:.2}, committed={Bi:.2}", .{
                @tagName(self.controller.mode),
                self.controller.knobs.read_workers,
                self.controller.knobs.dma_workers,
                self.controller.knobs.dma_chunks,
                self.controller.knobs.staging_chunks,
                final.completed_transfers,
                self.total_transfers,
                final.storage_bytes,
                final.direct_read_bytes,
                final.staged_read_bytes,
                final.submitted_bytes,
                final.committed_bytes,
            });
        }

        while (!self.done.load(.acquire)) {
            try io.sleep(.fromMilliseconds(25), .awake);
            if (self.done.load(.acquire)) break;
            switch (self.activateProbeIfReady(io)) {
                .none => {},
                .activated, .rolled_back => {
                    self.metrics.resetReadHighWater();
                    previous = self.metrics.snapshot();
                    previous_queue = previous.submitted_bytes -| previous.committed_bytes;
                    window_started = .now(io, .awake);
                    continue;
                },
            }
            self.trimSurplus(io);

            const elapsed = window_started.untilNow(io, .awake);
            const elapsed_ns: u64 = @intCast(@max(elapsed.nanoseconds, 0));
            const startup = self.controller.mode == .startup;
            const min_ns: u64 = if (startup) 50 * std.time.ns_per_ms else 100 * std.time.ns_per_ms;
            const max_ns: u64 = if (startup) 100 * std.time.ns_per_ms else 250 * std.time.ns_per_ms;
            if (elapsed_ns < min_ns) continue;

            const snapshot = self.metrics.snapshot();
            const delta = snapshot.sub(previous);
            const pipeline_progress_bytes = @max(
                delta.storage_bytes,
                @max(delta.ordered_bytes, @max(delta.submitted_bytes, delta.committed_bytes)),
            );
            const progress_bytes = if (startup or self.controller.probe != null)
                pipeline_progress_bytes
            else
                delta.committed_bytes;
            const byte_floor: u64 = if (startup) 32 * 1024 * 1024 else 64 * 1024 * 1024;
            if (progress_bytes == 0) {
                const now_ns: u64 = @intCast(@max(started.untilNow(io, .awake).nanoseconds, 0));
                const active_reads_high_water = self.metrics.active_reads_high_water.load(.acquire);
                const read_capacity_demand = self.hasReadCapacityDemand(snapshot);
                const reads_saturated = @max(self.read_group.inFlight(), active_reads_high_water) >= self.controller.knobs.read_workers and
                    read_capacity_demand;
                if (startup and elapsed_ns >= max_ns and self.controller.probe == null and self.pending_probe_activation == null) {
                    const old = self.controller.knobs;
                    const ready_capacity = @max(@as(u64, @intCast(self.read_chunk_size)), @as(u64, @intCast(self.controller.knobs.staging_chunks *| self.read_chunk_size)));
                    const decision = self.controller.observe(.{
                        .source_stalled = true,
                        .read_capacity_demand = read_capacity_demand,
                        .reads_saturated = reads_saturated,
                        .ready_bytes = snapshot.ready_bytes,
                        .ready_queue_ratio = @as(f64, @floatFromInt(snapshot.ready_bytes)) / @as(f64, @floatFromInt(ready_capacity)),
                        .now_ns = now_ns,
                    });
                    if (decision.changed) {
                        load_log.debug("source bootstrap: action={s}, epoch={d}, idle={d:.1}ms, reads={d}->{d}/{d} active={d} peak={d} demand={}, dma={d}/{d}, staging={d}->{d}/{d}", .{
                            @tagName(decision.action),
                            decision.epoch,
                            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                            old.read_workers,
                            decision.knobs.read_workers,
                            self.controller.max_read_workers,
                            snapshot.active_reads,
                            active_reads_high_water,
                            read_capacity_demand,
                            decision.knobs.dma_workers,
                            self.controller.max_dma_workers,
                            old.staging_chunks,
                            decision.knobs.staging_chunks,
                            self.controller.max_staging_chunks,
                        });
                        self.apply(io, decision);
                        previous = snapshot;
                        previous_queue = snapshot.submitted_bytes -| snapshot.committed_bytes;
                        window_started = .now(io, .awake);
                        continue;
                    }
                }
                if (stalledProbeTimedOut(self.pending_probe_activation != null, elapsed_ns, pipeline_progress_bytes)) {
                    if (self.controller.probe) |probe| {
                        const published_epoch = self.metrics.config_epoch.load(.acquire);
                        if (self.controller.rollbackTimedOutProbe(now_ns, published_epoch)) |decision| {
                            load_log.debug("probe progress timeout: epoch={d}, dimension={s}, kind={s}, idle={d:.1}ms, rollback_epoch={d}, reads={d}, dma={d}, dma_chunks={d}, staging={d}", .{
                                probe.epoch,
                                @tagName(probe.dimension),
                                @tagName(probe.kind),
                                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                                decision.epoch,
                                decision.knobs.read_workers,
                                decision.knobs.dma_workers,
                                decision.knobs.dma_chunks,
                                decision.knobs.staging_chunks,
                            });
                            self.apply(io, decision);
                            previous = snapshot;
                            previous_queue = snapshot.submitted_bytes -| snapshot.committed_bytes;
                            window_started = .now(io, .awake);
                            continue;
                        }
                    }
                }
                if (now_ns -| last_idle_log_ns >= 500 * std.time.ns_per_ms) {
                    const pool_stats = self.poolStats();
                    const staging_stats = self.stagingStats();
                    const idle_dma_lanes = if (self.pipeline) |pipeline| pipeline.admitted_lanes.load(.acquire) else self.dma_group.inFlight();
                    load_log.debug("waiting for progress: mode={s}, reads={d}/{d} active={d}, dma={d}/{d} lanes={d} open_writers={d}, completed={d}/{d}, dma_buffers={d}inflight/{d}allocated/{d}limit, staging={d}inflight/{d}allocated/{d}limit ready={Bi:.2}, read={Bi:.2}, submitted={Bi:.2}, committed={Bi:.2}, h2d_queued={Bi:.2}", .{
                        @tagName(self.controller.mode),
                        self.controller.knobs.read_workers,
                        self.controller.max_read_workers,
                        self.read_group.inFlight(),
                        self.controller.knobs.dma_workers,
                        self.controller.max_dma_workers,
                        idle_dma_lanes,
                        snapshot.active_transfers,
                        snapshot.completed_transfers,
                        self.total_transfers,
                        pool_stats.in_flight,
                        pool_stats.allocated,
                        self.controller.knobs.dma_chunks * self.pools.len,
                        staging_stats.in_flight,
                        staging_stats.allocated,
                        self.controller.knobs.staging_chunks,
                        snapshot.ready_bytes,
                        snapshot.storage_bytes,
                        snapshot.submitted_bytes,
                        snapshot.committed_bytes,
                        snapshot.submitted_bytes -| snapshot.committed_bytes,
                    });
                    last_idle_log_ns = now_ns;
                }
                continue;
            }
            if (progress_bytes < byte_floor and elapsed_ns < max_ns) continue;

            const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
            const read_goodput = @as(f64, @floatFromInt(delta.storage_bytes)) / seconds;
            const direct_read_goodput = @as(f64, @floatFromInt(delta.direct_read_bytes)) / seconds;
            const staged_read_goodput = @as(f64, @floatFromInt(delta.staged_read_bytes)) / seconds;
            const ordered_goodput = @as(f64, @floatFromInt(delta.ordered_bytes)) / seconds;
            const committed_goodput = @as(f64, @floatFromInt(delta.committed_bytes)) / seconds;
            const copy_goodput = @as(f64, @floatFromInt(delta.staged_copy_bytes)) / seconds;
            const copy_bandwidth = if (delta.staged_copy_ns == 0)
                0
            else
                @as(f64, @floatFromInt(delta.staged_copy_bytes)) /
                    (@as(f64, @floatFromInt(delta.staged_copy_ns)) / std.time.ns_per_s);
            const ready_queue_age_us = if (delta.staged_copy_bytes == 0)
                0
            else
                @as(f64, @floatFromInt(delta.weighted_ready_age_us)) / @as(f64, @floatFromInt(delta.staged_copy_bytes));
            const read_latency_us = if (delta.storage_bytes == 0)
                0
            else
                @as(f64, @floatFromInt(delta.weighted_read_latency_us)) / @as(f64, @floatFromInt(delta.storage_bytes));
            const average_read_bytes = if (delta.read_operations > 0)
                @as(f64, @floatFromInt(delta.storage_bytes)) / @as(f64, @floatFromInt(delta.read_operations))
            else
                0;
            const read_service_bandwidth = if (read_latency_us > 0 and average_read_bytes > 0)
                average_read_bytes / (read_latency_us / std.time.us_per_s)
            else
                0;
            const slow_reads = read_service_bandwidth > 0 and read_service_bandwidth < 1.5 * 1024 * 1024 * 1024;
            const transfer_latency_us = if (delta.committed_bytes == 0)
                0
            else
                @as(f64, @floatFromInt(delta.weighted_transfer_latency_us)) / @as(f64, @floatFromInt(delta.committed_bytes));
            const admitted_dma = if (self.pipeline) |pipeline| pipeline.admitted_lanes.load(.acquire) else self.controller.knobs.dma_workers;
            const dma_lane_capacity_available = if (self.pipeline) |pipeline|
                snapshot.active_transfers > admitted_dma or pipeline.next_tensor.load(.acquire) < pipeline.tensors.len
            else
                snapshot.completed_transfers < self.total_transfers;
            const active_dma = @max(@as(usize, 1), admitted_dma);
            const dma_capacity = @max(@as(usize, 1), self.controller.knobs.dma_workers);
            const active_reads = @max(@as(usize, 1), self.controller.knobs.read_workers);
            const pinned_wait_ratio = @as(f64, @floatFromInt(delta.pinned_buffer_wait_ns)) /
                (@as(f64, @floatFromInt(elapsed_ns)) * @as(f64, @floatFromInt(active_dma)));
            const dma_completion_wait_ratio = @as(f64, @floatFromInt(delta.dma_completion_wait_ns)) /
                (@as(f64, @floatFromInt(elapsed_ns)) * @as(f64, @floatFromInt(active_dma)));
            const staging_wait_ratio = @as(f64, @floatFromInt(delta.staging_wait_ns)) /
                (@as(f64, @floatFromInt(elapsed_ns)) * @as(f64, @floatFromInt(active_reads)));
            const read_admission_wait_ratio = @as(f64, @floatFromInt(delta.read_admission_wait_ns)) /
                (@as(f64, @floatFromInt(elapsed_ns)) * @as(f64, @floatFromInt(active_reads)));
            const dma_starvation_ratio = @min(
                1,
                @as(f64, @floatFromInt(delta.dma_starved_ns)) / @as(f64, @floatFromInt(elapsed_ns)),
            );
            const dma_utilization = @as(f64, @floatFromInt(delta.dma_work_ns)) /
                (@as(f64, @floatFromInt(elapsed_ns)) * @as(f64, @floatFromInt(dma_capacity)));
            const queue = snapshot.submitted_bytes -| snapshot.committed_bytes;
            const queue_growth = queue -| previous_queue;
            const queue_capacity = @max(
                @as(u64, 1),
                @as(u64, @intCast(self.controller.knobs.dma_chunks *| self.dma_chunk_size *| @max(@as(usize, 1), self.pools.len))),
            );
            const h2d_growth_ratio = @as(f64, @floatFromInt(queue_growth)) / @as(f64, @floatFromInt(queue_capacity));
            const h2d_queue_ratio = @as(f64, @floatFromInt(queue)) / @as(f64, @floatFromInt(queue_capacity));
            const ready_growth = snapshot.ready_bytes -| previous.ready_bytes;
            const ready_capacity = @max(@as(u64, @intCast(self.read_chunk_size)), @as(u64, @intCast(self.controller.knobs.staging_chunks * self.read_chunk_size)));
            const ready_growth_ratio = @as(f64, @floatFromInt(ready_growth)) / @as(f64, @floatFromInt(ready_capacity));
            const ready_queue_ratio = @as(f64, @floatFromInt(snapshot.ready_bytes)) / @as(f64, @floatFromInt(ready_capacity));
            const ready_age_pressure = @max(0, ready_queue_age_us / (250 * std.time.us_per_ms) - 1);

            const remaining = self.total_logical_bytes -| snapshot.ordered_bytes;
            const buffered_for_submission = snapshot.ordered_bytes -| snapshot.logical_submitted_bytes;
            const progress_goodput = if (ordered_goodput > 0) ordered_goodput else read_goodput;
            const drain_goodput = if (committed_goodput > 0) committed_goodput else self.controller.peak_goodput;
            const source_seconds = if (remaining == 0)
                0
            else if (progress_goodput > 0)
                @as(f64, @floatFromInt(remaining)) / progress_goodput
            else
                std.math.inf(f64);
            const buffered_seconds = if (buffered_for_submission == 0)
                0
            else if (drain_goodput > 0)
                @as(f64, @floatFromInt(buffered_for_submission)) / drain_goodput
            else
                std.math.inf(f64);
            const queue_seconds = if (queue == 0)
                0
            else if (drain_goodput > 0)
                @as(f64, @floatFromInt(queue)) / drain_goodput
            else
                std.math.inf(f64);
            const remaining_ns: f64 = (source_seconds + buffered_seconds + queue_seconds) * std.time.ns_per_s;
            const now_ns: u64 = @intCast(@max(started.untilNow(io, .awake).nanoseconds, 0));
            const active_reads_high_water = self.metrics.active_reads_high_water.load(.acquire);
            const read_capacity_demand = self.hasReadCapacityDemand(snapshot);
            const reads_saturated = @max(self.read_group.inFlight(), active_reads_high_water) >= self.controller.knobs.read_workers and
                read_capacity_demand;
            const ready_dma_demand = snapshot.ready_bytes > 0 and h2d_queue_ratio < 0.25 and
                self.dma_group.inFlight() >= self.controller.knobs.dma_workers;
            const dma_saturated = (dma_utilization >= 0.80 or ready_dma_demand) and
                snapshot.completed_transfers < self.total_transfers;
            const allow_probe = remaining_ns > 500 * std.time.ns_per_ms;
            const pool_stats = self.poolStats();
            const staging_stats = self.stagingStats();
            const probe_elapsed_ns: u64 = if (snapshot.probe_first_ns > 0)
                @intCast(@max(std.Io.Timestamp.fromNanoseconds(@intCast(snapshot.probe_first_ns)).untilNow(io, .awake).nanoseconds, 1))
            else
                @intCast(@max(self.probe_started.untilNow(io, .awake).nanoseconds, 1));
            const probe_seconds = @as(f64, @floatFromInt(probe_elapsed_ns)) / std.time.ns_per_s;
            const probe_active = self.controller.probe != null or self.pending_probe_activation != null;
            const probe_committed_bytes = if (probe_active and self.pending_probe_activation == null) snapshot.probe_committed_bytes else 0;
            const probe_goodput = @as(f64, @floatFromInt(probe_committed_bytes)) / probe_seconds;
            const probe_dimension = if (probe_active) if (self.controller.last_probe_dimension) |dimension| @tagName(dimension) else "none" else "none";
            const probe_kind = if (probe_active) if (self.controller.last_probe_kind) |kind| @tagName(kind) else "none" else "none";
            const probe_baseline_goodput = if (probe_active) self.controller.last_probe_baseline_goodput else 0;
            const probe_gain_pct = if (probe_baseline_goodput > 0)
                (probe_goodput / probe_baseline_goodput - 1) * 100
            else
                0;

            const old = self.controller.knobs;
            var decision = self.controller.observe(.{
                .read_goodput = read_goodput,
                .committed_goodput = committed_goodput,
                .probe_goodput = probe_goodput,
                .probe_committed_bytes = probe_committed_bytes,
                .capacity_pending = self.pending_probe_activation != null,
                .read_latency_us = read_latency_us,
                .read_admission_wait_ratio = read_admission_wait_ratio,
                .transfer_latency_us = transfer_latency_us,
                .dma_latency_reliable = delta.committed_bytes >= 32 * 1024 * 1024,
                .pinned_wait_ratio = pinned_wait_ratio,
                .dma_completion_wait_ratio = dma_completion_wait_ratio,
                .staging_wait_ratio = staging_wait_ratio,
                .dma_starvation_ratio = dma_starvation_ratio,
                .ready_bytes = snapshot.ready_bytes,
                .ready_queue_ratio = ready_queue_ratio,
                .ready_age_pressure = ready_age_pressure,
                .ready_growth_ratio = ready_growth_ratio,
                .h2d_queue_ratio = h2d_queue_ratio,
                .h2d_growth_ratio = h2d_growth_ratio,
                .admitted_dma_writers = admitted_dma,
                .dma_lane_capacity_available = dma_lane_capacity_available,
                .slow_reads = slow_reads,
                .read_capacity_demand = read_capacity_demand,
                .reads_saturated = reads_saturated,
                .dma_saturated = dma_saturated,
                .allow_probe = allow_probe,
                .now_ns = now_ns,
            });
            if (decision.changed and self.pending_probe_activation != null and self.controller.probe == null) {
                const published_epoch = self.metrics.config_epoch.load(.acquire);
                self.controller.epoch = published_epoch;
                decision.epoch = published_epoch;
            }
            load_log.debug("window control: action={s}, reason={s}, mode={s}, epoch={d}, elapsed={d:.1}ms, reads={d}->{d}/{d} active={d} peak={d} saturated={} demand={}, dma={d}->{d}/{d} lanes={d} open_writers={d} saturated={} lane_capacity={} utilization={d:.1}%, dma_chunks={d}->{d}/{d} buffers={d}inflight/{d}allocated, staging={d}->{d}/{d} blocks={d}inflight/{d}allocated", .{
                @tagName(decision.action),
                @tagName(decision.reason),
                @tagName(self.controller.mode),
                decision.epoch,
                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                old.read_workers,
                decision.knobs.read_workers,
                self.controller.max_read_workers,
                snapshot.active_reads,
                active_reads_high_water,
                reads_saturated,
                read_capacity_demand,
                old.dma_workers,
                decision.knobs.dma_workers,
                self.controller.max_dma_workers,
                admitted_dma,
                snapshot.active_transfers,
                dma_saturated,
                dma_lane_capacity_available,
                dma_utilization * 100,
                old.dma_chunks,
                decision.knobs.dma_chunks,
                self.controller.max_dma_chunks,
                pool_stats.in_flight,
                pool_stats.allocated,
                old.staging_chunks,
                decision.knobs.staging_chunks,
                self.controller.max_staging_chunks,
                staging_stats.in_flight,
                staging_stats.allocated,
            });
            load_log.debug("window throughput: ready={Bi:.2} occupancy={d:.1}% age={d:.1}us, read={d:.2}MiB/s direct_read={d:.2}MiB/s staged_read={d:.2}MiB/s ordered={d:.2}MiB/s, staged_copy={d:.2}MiB/s copy_bandwidth={d:.2}MiB/s, dma_submissions={d} committed={d:.2}MiB/s, read_latency={d:.1}us service={d:.2}MiB/s slow={} dma_latency={d:.1}us", .{
                snapshot.ready_bytes,
                ready_queue_ratio * 100,
                ready_queue_age_us,
                read_goodput / (1024 * 1024),
                direct_read_goodput / (1024 * 1024),
                staged_read_goodput / (1024 * 1024),
                ordered_goodput / (1024 * 1024),
                copy_goodput / (1024 * 1024),
                copy_bandwidth / (1024 * 1024),
                delta.dma_submissions,
                committed_goodput / (1024 * 1024),
                read_latency_us,
                read_service_bandwidth / (1024 * 1024),
                slow_reads,
                transfer_latency_us,
            });
            load_log.debug("window pressure: read_wait={d:.1}% staging_wait={d:.1}% pinned_wait={d:.1}% dma_completion_wait={d:.1}% dma_starved={d:.1}%, h2d_queued={Bi:.2} occupancy={d:.1}% h2d_growth={d:.1}% ready_growth={d:.1}%, probe={s}/{s} epoch_bytes={Bi:.2} total_committed={Bi:.2} baseline={d:.2}MiB/s goodput={d:.2}MiB/s gain={d:.1}%, remaining={Bi:.2} buffered={Bi:.2} remaining_est={d:.2}s probe_allowed={}", .{
                read_admission_wait_ratio * 100,
                staging_wait_ratio * 100,
                pinned_wait_ratio * 100,
                dma_completion_wait_ratio * 100,
                dma_starvation_ratio * 100,
                queue,
                h2d_queue_ratio * 100,
                h2d_growth_ratio * 100,
                ready_growth_ratio * 100,
                probe_dimension,
                probe_kind,
                probe_committed_bytes,
                snapshot.committed_bytes,
                probe_baseline_goodput / (1024 * 1024),
                probe_goodput / (1024 * 1024),
                probe_gain_pct,
                remaining,
                buffered_for_submission,
                remaining_ns / std.time.ns_per_s,
                allow_probe,
            });
            if (self.pipeline) |pipeline| {
                const scheduling = pipeline.schedulingStats();
                load_log.debug("pipeline concurrency: reads={d}/{d} direct_pending={d}/{d} direct_waiters={d} detached={d} prefetched={d}/{d} dma_ready={d} dma_lanes={d}/{d} probe_lanes={d} tensors_claimed={d}/{d}", .{
                    self.read_group.inFlight(),
                    self.controller.knobs.read_workers,
                    scheduling.direct_reads_pending,
                    self.controller.knobs.dma_workers,
                    scheduling.direct_capacity_waiters,
                    scheduling.detached_direct_sources,
                    scheduling.prefetched_sources,
                    self.controller.knobs.staging_chunks,
                    scheduling.dma_ready_sources,
                    admitted_dma,
                    self.controller.knobs.dma_workers,
                    scheduling.dma_probe_lanes,
                    scheduling.claimed_tensors,
                    pipeline.tensors.len,
                });
            }
            if (decision.changed) {
                self.apply(io, decision);
                load_log.debug("limits updated: action={s}, reason={s}, epoch={d}, reads={d}->{d}, dma={d}->{d}, dma_chunks={d}->{d}, staging={d}->{d}, trim_pinned={}, trim_staging={}", .{
                    @tagName(decision.action),
                    @tagName(decision.reason),
                    decision.epoch,
                    old.read_workers,
                    decision.knobs.read_workers,
                    old.dma_workers,
                    decision.knobs.dma_workers,
                    old.dma_chunks,
                    decision.knobs.dma_chunks,
                    old.staging_chunks,
                    decision.knobs.staging_chunks,
                    decision.trim_pinned,
                    decision.trim_staging,
                });
            }

            self.metrics.resetReadHighWater();
            previous = snapshot;
            previous_queue = queue;
            window_started = .now(io, .awake);
        }
    }

    fn stalledProbeTimedOut(pending_activation: bool, elapsed_ns: u64, progress_bytes: u64) bool {
        return !pending_activation and progress_bytes == 0 and elapsed_ns >= 500 * std.time.ns_per_ms;
    }

    fn hasReadCapacityDemand(self: *const AdaptiveLoadRuntime, snapshot: LoadMetrics.Snapshot) bool {
        if (snapshot.completed_transfers >= self.total_transfers) return false;
        if (self.pipeline) |pipeline| {
            if (pipeline.hasUnclaimedTensors()) return true;
        }
        const ready_target: u64 = @intCast(self.controller.knobs.dma_workers *| self.read_chunk_size);
        return snapshot.ready_bytes < @max(@as(u64, 1), ready_target);
    }

    fn apply(self: *AdaptiveLoadRuntime, io: std.Io, decision: AdaptiveLoadController.Decision) void {
        const epoch_changed = decision.epoch != self.metrics.config_epoch.load(.acquire);
        const read_limit_changed = decision.knobs.read_workers != self.read_group.currentLimit();
        self.pending_probe_activation = if (self.controller.probe) |probe|
            if (probe.epoch == decision.epoch) .{
                .epoch = probe.epoch,
                .dimension = probe.dimension,
                .kind = probe.kind,
                .baseline = probe.baseline,
                .candidate = probe.candidate,
                .installed_at = .now(io, .awake),
                .controller_now_ns = self.controller.last_probe_ns,
            } else null
        else
            null;

        if (read_limit_changed) self.metrics.resetReadHighWater();
        if (self.pending_probe_activation) |activation| {
            if (activation.dimension == .dma) {
                if (self.pipeline) |pipeline| {
                    for (pipeline.lanes) |*lane| lane.last_dma_submission_epoch.store(0, .release);
                    pipeline.dma_probe_capacity_epoch.store(activation.epoch, .release);
                }
            }
        } else if (self.controller.probe == null) {
            if (self.pipeline) |pipeline| pipeline.dma_probe_capacity_epoch.store(0, .release);
        }

        self.read_group.setLimit(io, decision.knobs.read_workers);
        self.metrics.staging_limit.store(decision.knobs.staging_chunks, .release);
        if (self.controller.direct) {
            for (self.pools, self.dma_allocators, 0..) |*pool, *dma_allocator, device_index| {
                self.setDmaPoolLimit(pool, io, decision.knobs.dma_chunks, device_index);
                if (decision.trim_pinned) self.trimPool(pool, dma_allocator, io, decision.knobs.dma_chunks, device_index);
            }
            if (self.staging_pool) |pool| {
                if (decision.knobs.staging_chunks > 0) pool.setLimit(io, decision.knobs.staging_chunks);
                if (decision.trim_staging) self.trimStagingPool(io, decision.knobs.staging_chunks);
            }
        }
        self.dma_group.setLimit(io, decision.knobs.dma_workers);
        if (epoch_changed and self.pending_probe_activation == null) {
            self.metrics.publishProbeEpoch(io, decision.epoch);
            self.probe_started = .now(io, .awake);
        }
        if (self.pipeline) |pipeline| {
            pipeline.setDmaLaneLimit(decision.knobs.dma_workers);
            pipeline.ensurePrefetch();
            pipeline.fillDetachedReadAhead();
        }
    }

    fn activateProbeIfReady(self: *AdaptiveLoadRuntime, io: std.Io) ProbeTransition {
        const activation = self.pending_probe_activation orelse return .none;
        if (!self.probeCapacityActive(activation)) {
            const elapsed: u64 = @intCast(@max(activation.installed_at.untilNow(io, .awake).nanoseconds, 0));
            if (elapsed < 500 * std.time.ns_per_ms) return .none;
            const published_epoch = self.metrics.config_epoch.load(.acquire);
            const decision = self.controller.rollbackTimedOutProbe(activation.controller_now_ns +| elapsed, published_epoch) orelse {
                self.pending_probe_activation = null;
                return .rolled_back;
            };
            load_log.debug("probe capacity timeout: epoch={d}, dimension={s}, kind={s}, rollback_epoch={d}, reads={d}, dma={d}, dma_chunks={d}, staging={d}", .{
                activation.epoch,
                @tagName(activation.dimension),
                @tagName(activation.kind),
                decision.epoch,
                decision.knobs.read_workers,
                decision.knobs.dma_workers,
                decision.knobs.dma_chunks,
                decision.knobs.staging_chunks,
            });
            self.apply(io, decision);
            return .rolled_back;
        }

        self.metrics.publishProbeEpoch(io, activation.epoch);
        self.probe_started = .now(io, .awake);
        self.pending_probe_activation = null;
        if (activation.dimension == .dma) {
            if (self.pipeline) |pipeline| pipeline.dma_probe_capacity_epoch.store(0, .release);
        }
        load_log.debug("probe capacity active: epoch={d}, dimension={s}, kind={s}, reads={d}, dma={d}, dma_chunks={d}, staging={d}", .{
            activation.epoch,
            @tagName(activation.dimension),
            @tagName(activation.kind),
            activation.candidate.read_workers,
            activation.candidate.dma_workers,
            activation.candidate.dma_chunks,
            activation.candidate.staging_chunks,
        });
        return .activated;
    }

    fn probeCapacityActive(self: *AdaptiveLoadRuntime, activation: ProbeActivation) bool {
        return switch (activation.dimension) {
            .read => switch (activation.kind) {
                .increase => self.metrics.active_reads_high_water.load(.acquire) >= activation.candidate.read_workers,
                .reduce_resource => self.read_group.inFlight() <= activation.candidate.read_workers and
                    (self.staging_pool == null or self.staging_pool.?.inFlight() <= activation.candidate.staging_chunks),
            },
            .dma => if (self.pipeline) |pipeline| blk: {
                var exercised: usize = 0;
                for (pipeline.lanes) |*lane| {
                    if (lane.last_dma_submission_epoch.load(.acquire) == activation.epoch) exercised += 1;
                }
                break :blk exercised >= activation.candidate.dma_workers;
            } else true,
            .pinned => switch (activation.kind) {
                .increase => blk: {
                    for (self.pools) |*pool| {
                        if (pool.allocatedBlocks() <= activation.baseline.dma_chunks) break :blk false;
                    }
                    break :blk true;
                },
                .reduce_resource => blk: {
                    for (self.pools) |*pool| {
                        if (pool.currentLimit() > activation.candidate.dma_chunks or pool.inFlight() > activation.candidate.dma_chunks) break :blk false;
                    }
                    break :blk true;
                },
            },
            .staging => blk: {
                const pool = self.staging_pool orelse break :blk false;
                break :blk switch (activation.kind) {
                    .increase => pool.allocatedBlocks() >= activation.candidate.staging_chunks and
                        (activation.candidate.read_workers <= activation.baseline.read_workers or
                            self.metrics.active_reads_high_water.load(.acquire) >= activation.candidate.read_workers),
                    .reduce_resource => pool.inFlight() <= activation.candidate.staging_chunks,
                };
            },
        };
    }

    fn trimSurplus(self: *AdaptiveLoadRuntime, io: std.Io) void {
        if (!self.controller.direct) return;
        for (self.pools, self.dma_allocators, 0..) |*pool, *dma_allocator, device_index| {
            self.setDmaPoolLimit(pool, io, self.controller.knobs.dma_chunks, device_index);
            if (pool.allocatedBlocks() > self.controller.knobs.dma_chunks) {
                self.trimPool(pool, dma_allocator, io, self.controller.knobs.dma_chunks, device_index);
            }
        }
        if (self.staging_pool) |pool| {
            const target = self.controller.knobs.staging_chunks;
            if (pool.allocatedBlocks() > target) self.trimStagingPool(io, self.controller.knobs.staging_chunks);
        }
    }

    fn setDmaPoolLimit(
        self: *AdaptiveLoadRuntime,
        pool: *mem.DynamicBufferPool,
        io: std.Io,
        requested: usize,
        device_index: usize,
    ) void {
        const admitted = if (self.pipeline) |pipeline| pipeline.admitted_lanes.load(.acquire) else 0;
        const effective = @max(requested, @max(pool.inFlight(), admitted));
        if (effective == pool.currentLimit()) return;
        pool.setLimit(io, effective);
        load_log.debug("dma pool limit updated: device={d}, requested={d}, effective={d}, in_flight={d}", .{
            device_index,
            requested,
            effective,
            pool.inFlight(),
        });
    }

    fn trimPool(
        self: *AdaptiveLoadRuntime,
        pool: *mem.DynamicBufferPool,
        dma_allocator: *const mem.DmaAllocator,
        io: std.Io,
        target: usize,
        device_index: usize,
    ) void {
        _ = self;
        const before = pool.allocatedBlocks();
        pool.trim(dma_allocator.allocator(), io, target);
        const after = pool.allocatedBlocks();
        if (after != before) {
            load_log.debug("dma pool trimmed: device={d}, allocated={d}->{d}, target={d}, in_flight={d}", .{
                device_index,
                before,
                after,
                target,
                pool.inFlight(),
            });
        }
    }

    fn poolStats(self: *const AdaptiveLoadRuntime) struct { in_flight: usize, allocated: usize } {
        var in_flight: usize = 0;
        var allocated: usize = 0;
        for (self.pools) |*pool| {
            in_flight += pool.inFlight();
            allocated += pool.allocatedBlocks();
        }
        return .{ .in_flight = in_flight, .allocated = allocated };
    }

    fn trimStagingPool(self: *AdaptiveLoadRuntime, io: std.Io, target_: usize) void {
        const pool = self.staging_pool orelse return;
        const before = pool.allocatedBlocks();
        pool.trim(self.staging_allocator, io, target_);
        const after = pool.allocatedBlocks();
        if (after != before) {
            load_log.debug("staging pool trimmed: allocated={d}->{d}, target={d}, in_flight={d}", .{ before, after, target_, pool.inFlight() });
        }
    }

    fn stagingStats(self: *const AdaptiveLoadRuntime) struct { in_flight: usize, allocated: usize } {
        const pool = self.staging_pool orelse return .{ .in_flight = 0, .allocated = 0 };
        return .{ .in_flight = pool.inFlight(), .allocated = pool.allocatedBlocks() };
    }
};

const PageableReadSlot = struct {
    event: std.Io.Event = .unset,
    buffer: ?[]u8 = null,
    offset: u64 = 0,
    len: usize = 0,
    consumed: usize = 0,
    epoch: u64 = 0,
    err: ?anyerror = null,
    ready_counted: bool = false,
    scheduled_at: std.Io.Timestamp = undefined,
    ready_at_ns: u64 = 0,

    fn reset(self: *PageableReadSlot) void {
        self.event.reset();
        self.buffer = null;
        self.offset = 0;
        self.len = 0;
        self.consumed = 0;
        self.epoch = 0;
        self.err = null;
        self.ready_counted = false;
        self.ready_at_ns = 0;
    }
};

const DirectReadSlot = struct {
    event: std.Io.Event = .unset,
    err: ?anyerror = null,
    scheduled_at: std.Io.Timestamp = undefined,

    fn reset(self: *DirectReadSlot) void {
        self.event.reset();
        self.err = null;
        self.scheduled_at = undefined;
    }
};

const AdaptivePipelineContext = struct {
    const scheduling_closed_bit = @as(usize, 1) << (@bitSizeOf(usize) - 1);
    const scheduling_count_mask = scheduling_closed_bit - 1;

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    tensors: []*const Tensor,
    buffers: []*Buffer,
    shardings: []const Sharding,
    dma_allocators: []const mem.DmaAllocator,
    pinned_buffer_pools: []mem.DynamicBufferPool,
    staging_pool: ?*mem.DynamicBufferPool,
    dma_chunk_size: usize,
    read_chunk_size: usize,
    max_read_parallelism: usize,
    dma_group: stdx.Io.LimitedGroup,
    read_group: stdx.Io.LimitedGroup,
    staging_group: std.Io.Group = .init,
    resume_group: std.Io.Group = .init,
    metrics: *LoadMetrics,
    progress: ?*std.Progress.Node,
    lanes: []AdaptivePipelineLane = &.{},
    total: std.atomic.Value(usize) = .init(0),
    first_error: std.atomic.Value(u16) = .init(0),
    next_tensor: std.atomic.Value(usize) = .init(0),
    prefetched_sources: std.atomic.Value(usize) = .init(0),
    direct_reads_pending: std.atomic.Value(usize) = .init(0),
    admitted_lanes: std.atomic.Value(usize) = .init(0),
    dma_probe_capacity_epoch: std.atomic.Value(u64) = .init(0),
    dma_lane_limit: std.atomic.Value(usize) = .init(1),
    ready_mutex: std.Io.Mutex = .init,
    ready_head: ?*AdaptiveTensorLoad = null,
    ready_tail: ?*AdaptiveTensorLoad = null,
    direct_wait_mutex: std.Io.Mutex = .init,
    direct_wait_head: ?*AdaptiveTensorLoad = null,
    direct_capacity_mutex: std.Io.Mutex = .init,
    direct_capacity_condition: std.Io.Condition = .init,
    direct_capacity_waiters: std.atomic.Value(usize) = .init(0),
    scheduling_state: std.atomic.Value(usize) = .init(0),
    scheduling_idle: std.Io.Event = .unset,
    done_event: std.Io.Event = .unset,

    fn failed(self: *const AdaptivePipelineContext) bool {
        return self.first_error.load(.acquire) != 0;
    }

    fn fail(self: *AdaptivePipelineContext, err: anyerror) void {
        if (self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic) == null) {
            self.closeScheduling();
            load_log.debug("pipeline cancellation requested: error={s}, completed={d}/{d}, reads_active={d}, dma_active={d}", .{
                @errorName(err),
                self.metrics.completed_transfers.load(.acquire),
                self.tensors.len,
                self.read_group.inFlight(),
                self.dma_group.inFlight(),
            });
            self.direct_capacity_mutex.lockUncancelable(self.io);
            self.direct_capacity_condition.broadcast(self.io);
            self.direct_capacity_mutex.unlock(self.io);
            self.done_event.set(self.io);
        }
    }

    fn errorValue(self: *const AdaptivePipelineContext) ?anyerror {
        const err = self.first_error.load(.acquire);
        return if (err == 0) null else @errorFromInt(err);
    }

    fn markProbeStart(self: *AdaptivePipelineContext, epoch: u64) void {
        self.metrics.markProbeStart(self.io, epoch);
    }

    fn completeTensor(self: *AdaptivePipelineContext, bytes: usize) void {
        _ = self.total.fetchAdd(bytes, .monotonic);
        const completed = self.metrics.completed_transfers.fetchAdd(1, .acq_rel) + 1;
        if (completed == self.tensors.len) {
            self.closeScheduling();
            self.done_event.set(self.io);
        }
    }

    fn beginScheduling(self: *AdaptivePipelineContext) bool {
        var state = self.scheduling_state.load(.acquire);
        while ((state & scheduling_closed_bit) == 0) {
            std.debug.assert((state & scheduling_count_mask) != scheduling_count_mask);
            if (self.scheduling_state.cmpxchgWeak(state, state + 1, .acq_rel, .acquire)) |actual| {
                state = actual;
                continue;
            }
            return true;
        }
        return false;
    }

    fn endScheduling(self: *AdaptivePipelineContext) void {
        const previous = self.scheduling_state.fetchSub(1, .acq_rel);
        std.debug.assert((previous & scheduling_count_mask) > 0);
        if (previous == (scheduling_closed_bit | 1)) self.scheduling_idle.set(self.io);
    }

    fn closeScheduling(self: *AdaptivePipelineContext) void {
        const previous = self.scheduling_state.fetchOr(scheduling_closed_bit, .acq_rel);
        if ((previous & scheduling_count_mask) == 0) self.scheduling_idle.set(self.io);
    }

    fn claimTensorIndex(self: *AdaptivePipelineContext) ?usize {
        var index = self.next_tensor.load(.acquire);
        while (index < self.tensors.len) {
            if (self.next_tensor.cmpxchgWeak(index, index + 1, .acq_rel, .acquire)) |actual| {
                index = actual;
                continue;
            }
            return index;
        }
        return null;
    }

    fn hasUnclaimedTensors(self: *const AdaptivePipelineContext) bool {
        return self.next_tensor.load(.acquire) < self.tensors.len;
    }

    const SchedulingStats = struct {
        direct_reads_pending: usize,
        direct_capacity_waiters: usize,
        detached_direct_sources: usize,
        prefetched_sources: usize,
        dma_ready_sources: usize,
        dma_probe_lanes: usize,
        claimed_tensors: usize,
    };

    fn schedulingStats(self: *AdaptivePipelineContext) SchedulingStats {
        self.ready_mutex.lockUncancelable(self.io);
        var dma_ready_sources: usize = 0;
        var ready = self.ready_head;
        while (ready) |tensor_load| : (ready = tensor_load.ready_next) dma_ready_sources += 1;
        self.ready_mutex.unlock(self.io);

        self.direct_wait_mutex.lockUncancelable(self.io);
        var detached_direct_sources: usize = 0;
        var waiting = self.direct_wait_head;
        while (waiting) |tensor_load| : (waiting = tensor_load.direct_wait_next) detached_direct_sources += 1;
        self.direct_wait_mutex.unlock(self.io);

        const dma_probe_epoch = self.dma_probe_capacity_epoch.load(.acquire);
        var dma_probe_lanes: usize = 0;
        if (dma_probe_epoch > 0) {
            for (self.lanes) |*lane| {
                if (lane.last_dma_submission_epoch.load(.acquire) == dma_probe_epoch) dma_probe_lanes += 1;
            }
        }

        return .{
            .direct_reads_pending = self.direct_reads_pending.load(.acquire),
            .direct_capacity_waiters = self.direct_capacity_waiters.load(.acquire),
            .detached_direct_sources = detached_direct_sources,
            .prefetched_sources = self.prefetched_sources.load(.acquire),
            .dma_ready_sources = dma_ready_sources,
            .dma_probe_lanes = dma_probe_lanes,
            .claimed_tensors = @min(self.next_tensor.load(.acquire), self.tensors.len),
        };
    }

    fn tryReserveDirectRead(self: *AdaptivePipelineContext) bool {
        const limit = @max(@as(usize, 1), self.dma_lane_limit.load(.acquire));
        var pending = self.direct_reads_pending.load(.acquire);
        while (pending < limit) {
            if (self.direct_reads_pending.cmpxchgWeak(pending, pending + 1, .acq_rel, .acquire)) |actual| {
                pending = actual;
                continue;
            }
            return true;
        }
        return false;
    }

    fn releaseDirectRead(self: *AdaptivePipelineContext) void {
        self.direct_capacity_mutex.lockUncancelable(self.io);
        defer self.direct_capacity_mutex.unlock(self.io);
        const previous = self.direct_reads_pending.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
        self.direct_capacity_condition.signal(self.io);
    }

    fn enqueueDmaReady(self: *AdaptivePipelineContext, tensor_load: *AdaptiveTensorLoad) void {
        self.ready_mutex.lockUncancelable(self.io);
        std.debug.assert(!tensor_load.queued_for_dma);
        tensor_load.queued_for_dma = true;
        tensor_load.ready_next = null;
        if (self.ready_tail) |tail| {
            tail.ready_next = tensor_load;
        } else {
            self.ready_head = tensor_load;
        }
        self.ready_tail = tensor_load;
        self.ready_mutex.unlock(self.io);

        self.ensureLanes(self.dma_lane_limit.load(.acquire));
    }

    fn popDmaReady(self: *AdaptivePipelineContext) ?*AdaptiveTensorLoad {
        self.ready_mutex.lockUncancelable(self.io);
        const tensor_load = self.ready_head orelse {
            self.ready_mutex.unlock(self.io);
            return null;
        };
        self.ready_head = tensor_load.ready_next;
        if (self.ready_head == null) self.ready_tail = null;
        tensor_load.ready_next = null;
        tensor_load.queued_for_dma = false;
        const was_prefetched = tensor_load.prefetch_counted;
        tensor_load.prefetch_counted = false;
        self.ready_mutex.unlock(self.io);

        if (was_prefetched) {
            _ = self.prefetched_sources.fetchSub(1, .acq_rel);
            self.ensurePrefetch();
        }
        return tensor_load;
    }

    fn hasDmaReady(self: *AdaptivePipelineContext) bool {
        self.ready_mutex.lockUncancelable(self.io);
        defer self.ready_mutex.unlock(self.io);
        return self.ready_head != null;
    }

    fn deinitReadyLoads(self: *AdaptivePipelineContext) void {
        while (true) {
            self.ready_mutex.lockUncancelable(self.io);
            const tensor_load = self.ready_head orelse {
                self.ready_tail = null;
                self.ready_mutex.unlock(self.io);
                return;
            };
            self.ready_head = tensor_load.ready_next;
            if (self.ready_head == null) self.ready_tail = null;
            tensor_load.ready_next = null;
            tensor_load.queued_for_dma = false;
            const was_prefetched = tensor_load.prefetch_counted;
            tensor_load.prefetch_counted = false;
            self.ready_mutex.unlock(self.io);
            if (was_prefetched) _ = self.prefetched_sources.fetchSub(1, .acq_rel);
            tensor_load.deinit();
        }
    }

    fn ensurePrefetch(self: *AdaptivePipelineContext) void {
        if (self.failed()) return;
        const staging_limit = self.metrics.staging_limit.load(.acquire);
        const target = @min(staging_limit, self.max_read_parallelism);
        while (target > 0 and !self.failed()) {
            if (!self.reservePrefetchSource(target)) break;

            const tensor_index = self.claimTensorIndex() orelse {
                _ = self.prefetched_sources.fetchSub(1, .acq_rel);
                break;
            };
            self.schedulePrefetch(tensor_index);
        }
    }

    fn reservePrefetchSource(self: *AdaptivePipelineContext, target: usize) bool {
        var reserved = self.prefetched_sources.load(.acquire);
        while (reserved < target) {
            if (self.prefetched_sources.cmpxchgWeak(reserved, reserved + 1, .acq_rel, .acquire)) |actual| {
                reserved = actual;
                continue;
            }
            return true;
        }
        return false;
    }

    fn schedulePrefetch(self: *AdaptivePipelineContext, tensor_index: usize) void {
        if (!self.beginScheduling()) {
            _ = self.prefetched_sources.fetchSub(1, .acq_rel);
            return;
        }
        defer self.endScheduling();
        self.staging_group.concurrent(self.io, AdaptivePipelineContext.prefetchTensor, .{ tensor_index, self }) catch |err| {
            _ = self.prefetched_sources.fetchSub(1, .acq_rel);
            self.fail(err);
        };
    }

    fn prefetchTensor(tensor_index: usize, self: *AdaptivePipelineContext) void {
        const tensor_load = AdaptiveTensorLoad.init(self, tensor_index) catch |err| {
            _ = self.prefetched_sources.fetchSub(1, .acq_rel);
            self.fail(err);
            return;
        };
        tensor_load.prefetch_counted = true;
        tensor_load.fillReadAheadTo(1);
        if (tensor_load.pending == 0) {
            if (self.failed()) {
                tensor_load.prefetch_counted = false;
                _ = self.prefetched_sources.fetchSub(1, .acq_rel);
                tensor_load.deinit();
                return;
            }
            load_log.debug("read source staging bypassed: index={d}, name={s}, staging_limit={d}", .{
                tensor_index,
                tensor_load.reader.tensor.name,
                self.metrics.staging_limit.load(.acquire),
            });
            self.enqueueDmaReady(tensor_load);
            return;
        }

        const slot = &tensor_load.slots[tensor_load.head];
        slot.event.waitUncancelable(self.io);
        if (slot.err) |err| {
            tensor_load.prefetch_counted = false;
            _ = self.prefetched_sources.fetchSub(1, .acq_rel);
            tensor_load.deinit();
            if (!self.failed()) self.fail(err);
            return;
        }
        if (self.failed()) {
            tensor_load.prefetch_counted = false;
            _ = self.prefetched_sources.fetchSub(1, .acq_rel);
            tensor_load.deinit();
            return;
        }

        load_log.debug("read source ready: index={d}, name={s}, ready={Bi:.2}, prefetched={d}/{d}", .{
            tensor_index,
            tensor_load.reader.tensor.name,
            slot.len,
            self.prefetched_sources.load(.acquire),
            targetPrefetchCount(self),
        });
        self.enqueueDmaReady(tensor_load);
    }

    fn targetPrefetchCount(self: *const AdaptivePipelineContext) usize {
        return @min(self.metrics.staging_limit.load(.acquire), self.max_read_parallelism);
    }

    fn addDirectWait(self: *AdaptivePipelineContext, tensor_load: *AdaptiveTensorLoad) void {
        self.direct_wait_mutex.lockUncancelable(self.io);
        defer self.direct_wait_mutex.unlock(self.io);
        std.debug.assert(!tensor_load.in_direct_wait);
        tensor_load.in_direct_wait = true;
        tensor_load.direct_wait_next = self.direct_wait_head;
        self.direct_wait_head = tensor_load;
        self.fillDetachedReadAheadLocked();
    }

    fn removeDirectWait(self: *AdaptivePipelineContext, tensor_load: *AdaptiveTensorLoad) void {
        self.direct_wait_mutex.lockUncancelable(self.io);
        defer self.direct_wait_mutex.unlock(self.io);
        var previous: ?*AdaptiveTensorLoad = null;
        var current = self.direct_wait_head;
        while (current) |candidate| {
            if (candidate == tensor_load) {
                if (previous) |prev| {
                    prev.direct_wait_next = candidate.direct_wait_next;
                } else {
                    self.direct_wait_head = candidate.direct_wait_next;
                }
                candidate.direct_wait_next = null;
                candidate.in_direct_wait = false;
                return;
            }
            previous = candidate;
            current = candidate.direct_wait_next;
        }
        std.debug.assert(false);
    }

    fn fillDetachedReadAhead(self: *AdaptivePipelineContext) void {
        self.direct_wait_mutex.lockUncancelable(self.io);
        defer self.direct_wait_mutex.unlock(self.io);
        self.fillDetachedReadAheadLocked();
    }

    fn fillDetachedReadAheadLocked(self: *AdaptivePipelineContext) void {
        const staging_limit = self.metrics.staging_limit.load(.acquire);
        if (staging_limit == 0) return;
        var count: usize = 0;
        var current = self.direct_wait_head;
        while (current) |tensor_load| : (current = tensor_load.direct_wait_next) count += 1;
        if (count == 0) return;
        const per_tensor = std.math.divCeil(usize, staging_limit, count) catch unreachable;
        current = self.direct_wait_head;
        while (current) |tensor_load| : (current = tensor_load.direct_wait_next) {
            tensor_load.fillReadAheadTo(per_tensor);
        }
    }

    fn requeueDirectWhenReady(self: *AdaptivePipelineContext, tensor_load: *AdaptiveTensorLoad, event: *std.Io.Event) void {
        if (!self.beginScheduling()) {
            self.removeDirectWait(tensor_load);
            tensor_load.deinit();
            return;
        }
        defer self.endScheduling();
        self.resume_group.concurrent(self.io, AdaptivePipelineContext.waitDirectAndEnqueue, .{ tensor_load, event, self }) catch |err| {
            self.removeDirectWait(tensor_load);
            tensor_load.deinit();
            self.fail(err);
        };
    }

    fn waitDirectAndEnqueue(tensor_load: *AdaptiveTensorLoad, event: *std.Io.Event, self: *AdaptivePipelineContext) void {
        const started: std.Io.Timestamp = .now(self.io, .awake);
        event.waitUncancelable(self.io);
        self.removeDirectWait(tensor_load);
        const submitted = self.metrics.submitted_bytes.load(.acquire);
        const committed = self.metrics.committed_bytes.load(.acquire);
        if (submitted <= committed) {
            const completed_at_ns = self.metrics.last_dma_commit_ns.load(.acquire);
            const idle_started_ns: u64 = @max(@as(u64, @intCast(@max(started.nanoseconds, 0))), completed_at_ns);
            const now_ns: u64 = @intCast(@max(std.Io.Timestamp.now(self.io, .awake).nanoseconds, 0));
            self.metrics.addDmaStarvationInterval(idle_started_ns, now_ns);
        }
        if (self.failed()) {
            tensor_load.deinit();
            return;
        }
        self.enqueueDmaReady(tensor_load);
    }

    fn requeueOnDirectCapacity(self: *AdaptivePipelineContext, tensor_load: *AdaptiveTensorLoad) void {
        if (!self.beginScheduling()) {
            tensor_load.deinit();
            return;
        }
        defer self.endScheduling();
        self.resume_group.concurrent(self.io, AdaptivePipelineContext.waitDirectCapacityAndEnqueue, .{ tensor_load, self }) catch |err| {
            tensor_load.deinit();
            self.fail(err);
        };
    }

    fn waitDirectCapacityAndEnqueue(tensor_load: *AdaptiveTensorLoad, self: *AdaptivePipelineContext) void {
        _ = self.direct_capacity_waiters.fetchAdd(1, .acq_rel);
        defer _ = self.direct_capacity_waiters.fetchSub(1, .acq_rel);

        self.direct_capacity_mutex.lockUncancelable(self.io);
        var reserved = false;
        while (!self.failed()) {
            if (self.tryReserveDirectRead()) {
                reserved = true;
                break;
            }
            self.direct_capacity_condition.waitUncancelable(self.io, &self.direct_capacity_mutex);
        }
        self.direct_capacity_mutex.unlock(self.io);

        if (!reserved) {
            tensor_load.deinit();
            return;
        }
        tensor_load.direct_read_reserved = true;
        if (self.failed()) {
            tensor_load.deinit();
            return;
        }
        self.enqueueDmaReady(tensor_load);
    }

    fn requeueWhenReady(self: *AdaptivePipelineContext, tensor_load: *AdaptiveTensorLoad, event: *std.Io.Event) void {
        if (!self.beginScheduling()) {
            tensor_load.deinit();
            return;
        }
        defer self.endScheduling();
        self.resume_group.concurrent(self.io, AdaptivePipelineContext.waitAndEnqueue, .{ tensor_load, event, self }) catch |err| {
            tensor_load.deinit();
            self.fail(err);
        };
    }

    fn waitAndEnqueue(tensor_load: *AdaptiveTensorLoad, event: *std.Io.Event, self: *AdaptivePipelineContext) void {
        const started: std.Io.Timestamp = .now(self.io, .awake);
        event.waitUncancelable(self.io);
        const submitted = self.metrics.submitted_bytes.load(.acquire);
        const committed = self.metrics.committed_bytes.load(.acquire);
        if (submitted <= committed) {
            const completed_at_ns = self.metrics.last_dma_commit_ns.load(.acquire);
            const idle_started_ns: u64 = @max(@as(u64, @intCast(@max(started.nanoseconds, 0))), completed_at_ns);
            const now_ns: u64 = @intCast(@max(std.Io.Timestamp.now(self.io, .awake).nanoseconds, 0));
            self.metrics.addDmaStarvationInterval(idle_started_ns, now_ns);
        }
        if (self.failed()) {
            tensor_load.deinit();
            return;
        }
        self.enqueueDmaReady(tensor_load);
    }

    fn ensureLanes(self: *AdaptivePipelineContext, requested: usize) void {
        const target = @min(requested, self.lanes.len);
        while (!self.failed()) {
            const admitted = self.admitted_lanes.load(.acquire);
            if (admitted >= target) break;
            if (self.admitted_lanes.cmpxchgWeak(admitted, admitted + 1, .acq_rel, .acquire)) |_| continue;

            var found = false;
            for (self.lanes) |*lane| {
                if (found) break;
                if (lane.activateReserved()) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                _ = self.admitted_lanes.fetchSub(1, .acq_rel);
                break;
            }
        }
    }

    fn acquirePageable(
        tensor_load: *AdaptiveTensorLoad,
        ticket: usize,
        slot: *PageableReadSlot,
        self: *AdaptivePipelineContext,
    ) void {
        tensor_load.acquisition_mutex.lockUncancelable(self.io);
        while (ticket != tensor_load.acquisition_turn) {
            tensor_load.acquisition_condition.waitUncancelable(self.io, &tensor_load.acquisition_mutex);
        }

        if (self.failed()) {
            tensor_load.finishAcquisition(self.io);
            slot.err = error.LoadCancelled;
            slot.event.set(self.io);
            return;
        }

        const pool = self.staging_pool orelse {
            tensor_load.finishAcquisition(self.io);
            slot.err = error.StagingDisabled;
            self.fail(slot.err.?);
            slot.event.set(self.io);
            return;
        };
        const acquisition = pool.getWithWait(self.allocator, self.io) catch |err| {
            tensor_load.finishAcquisition(self.io);
            slot.err = err;
            self.fail(err);
            slot.event.set(self.io);
            return;
        };
        const buffer = acquisition.buffer;
        slot.buffer = buffer;
        _ = self.metrics.staging_wait_ns.fetchAdd(acquisition.wait_ns, .monotonic);
        const cancelled = self.failed();
        tensor_load.finishAcquisition(self.io);
        if (cancelled) {
            pool.put(self.io, buffer);
            slot.buffer = null;
            slot.err = error.LoadCancelled;
            slot.event.set(self.io);
            return;
        }

        slot.scheduled_at = .now(self.io, .awake);
        self.schedulePageableRead(&tensor_load.reader, slot);
    }

    fn readPageable(reader: *const safetensors.TensorReader, slot: *PageableReadSlot, self: *AdaptivePipelineContext) void {
        self.markProbeStart(slot.epoch);
        const admission_wait = slot.scheduled_at.untilNow(self.io, .awake);
        _ = self.metrics.read_admission_wait_ns.fetchAdd(@intCast(@max(admission_wait.nanoseconds, 0)), .monotonic);
        self.metrics.recordReadStart();
        defer _ = self.metrics.active_reads.fetchSub(1, .monotonic);
        defer slot.event.set(self.io);

        if (self.failed()) {
            slot.err = error.LoadCancelled;
            return;
        }

        const read_started: std.Io.Timestamp = .now(self.io, .awake);
        reader.readPositionalAll(slot.buffer.?[0..slot.len], slot.offset) catch |err| {
            self.staging_pool.?.put(self.io, slot.buffer.?);
            slot.buffer = null;
            slot.err = err;
            self.fail(err);
            return;
        };
        const read_elapsed = read_started.untilNow(self.io, .awake);
        const read_us: u64 = @intCast(@max(@divTrunc(read_elapsed.nanoseconds, std.time.ns_per_us), 0));
        _ = self.metrics.storage_bytes.fetchAdd(@intCast(slot.len), .monotonic);
        _ = self.metrics.read_operations.fetchAdd(1, .monotonic);
        _ = self.metrics.staged_read_bytes.fetchAdd(@intCast(slot.len), .monotonic);
        _ = self.metrics.weighted_read_latency_us.fetchAdd(@as(u64, @intCast(slot.len)) *| read_us, .monotonic);
        slot.ready_at_ns = @intCast(@max(std.Io.Timestamp.now(self.io, .awake).nanoseconds, 1));
        _ = self.metrics.ready_bytes.fetchAdd(@intCast(slot.len), .release);
        slot.ready_counted = true;
    }

    fn readDirect(
        reader: *const safetensors.TensorReader,
        destination: []u8,
        offset: u64,
        epoch: u64,
        slot: *DirectReadSlot,
        self: *AdaptivePipelineContext,
    ) void {
        self.markProbeStart(epoch);
        const admission_wait = slot.scheduled_at.untilNow(self.io, .awake);
        _ = self.metrics.read_admission_wait_ns.fetchAdd(@intCast(@max(admission_wait.nanoseconds, 0)), .monotonic);
        self.metrics.recordReadStart();
        defer {
            _ = self.metrics.active_reads.fetchSub(1, .monotonic);
            self.releaseDirectRead();
            slot.event.set(self.io);
        }

        if (self.failed()) {
            slot.err = error.LoadCancelled;
            return;
        }

        const read_started: std.Io.Timestamp = .now(self.io, .awake);
        reader.readPositionalAll(destination, offset) catch |err| {
            slot.err = err;
            self.fail(err);
            return;
        };
        const read_elapsed = read_started.untilNow(self.io, .awake);
        const read_ns: u64 = @intCast(@max(read_elapsed.nanoseconds, 0));
        const read_us: u64 = @divTrunc(read_ns, std.time.ns_per_us);
        _ = self.metrics.storage_bytes.fetchAdd(@intCast(destination.len), .monotonic);
        _ = self.metrics.read_operations.fetchAdd(1, .monotonic);
        _ = self.metrics.direct_read_bytes.fetchAdd(@intCast(destination.len), .monotonic);
        _ = self.metrics.weighted_read_latency_us.fetchAdd(@as(u64, @intCast(destination.len)) *| read_us, .monotonic);
    }

    fn schedulePageable(
        self: *AdaptivePipelineContext,
        tensor_load: *AdaptiveTensorLoad,
        ticket: usize,
        slot: *PageableReadSlot,
        offset: u64,
        len: usize,
    ) void {
        slot.* = .{
            .offset = offset,
            .len = len,
            .epoch = self.metrics.config_epoch.load(.acquire),
        };
        if (!self.beginScheduling()) {
            slot.err = error.LoadCancelled;
            slot.event.set(self.io);
            return;
        }
        defer self.endScheduling();
        self.staging_group.concurrent(self.io, AdaptivePipelineContext.acquirePageable, .{ tensor_load, ticket, slot, self }) catch |err| {
            slot.err = err;
            slot.event.set(self.io);
            self.fail(err);
        };
    }

    fn schedulePageableRead(
        self: *AdaptivePipelineContext,
        reader: *const safetensors.TensorReader,
        slot: *PageableReadSlot,
    ) void {
        if (!self.beginScheduling()) {
            self.staging_pool.?.put(self.io, slot.buffer.?);
            slot.buffer = null;
            slot.err = error.LoadCancelled;
            slot.event.set(self.io);
            return;
        }
        defer self.endScheduling();
        self.read_group.concurrentUncancelableAdmission(self.io, AdaptivePipelineContext.readPageable, .{ reader, slot, self }) catch |err| {
            self.staging_pool.?.put(self.io, slot.buffer.?);
            slot.buffer = null;
            slot.err = err;
            slot.event.set(self.io);
            self.fail(err);
        };
    }

    fn scheduleDirect(
        self: *AdaptivePipelineContext,
        reader: *const safetensors.TensorReader,
        destination: []u8,
        offset: u64,
        epoch: u64,
        slot: *DirectReadSlot,
    ) void {
        if (!self.beginScheduling()) {
            self.releaseDirectRead();
            slot.err = error.LoadCancelled;
            slot.event.set(self.io);
            return;
        }
        defer self.endScheduling();
        self.read_group.concurrentUncancelableAdmission(self.io, AdaptivePipelineContext.readDirect, .{ reader, destination, offset, epoch, slot, self }) catch |err| {
            self.releaseDirectRead();
            slot.err = err;
            slot.event.set(self.io);
            self.fail(err);
        };
    }

    fn releaseSlot(self: *AdaptivePipelineContext, slot: *PageableReadSlot) void {
        if (slot.ready_counted) {
            _ = self.metrics.ready_bytes.fetchSub(@intCast(slot.len - slot.consumed), .release);
            slot.ready_counted = false;
        }
        if (slot.buffer) |buffer| {
            self.staging_pool.?.put(self.io, buffer);
            slot.buffer = null;
        }
        slot.reset();
    }

    fn consumeReady(self: *AdaptivePipelineContext, slot: *PageableReadSlot, len: usize) void {
        std.debug.assert(slot.ready_counted and len <= slot.len - slot.consumed);
        _ = self.metrics.ready_bytes.fetchSub(@intCast(len), .release);
        slot.consumed += len;
        if (slot.consumed == slot.len) slot.ready_counted = false;
    }

    fn setDmaLaneLimit(self: *AdaptivePipelineContext, limit: usize) void {
        self.dma_lane_limit.store(@min(limit, self.lanes.len), .release);
        self.ensureLanes(limit);
    }

    fn retirementNeeded(self: *const AdaptivePipelineContext) bool {
        return self.admitted_lanes.load(.acquire) > self.dma_lane_limit.load(.acquire);
    }
};

const AdaptiveTensorLoad = struct {
    const QuantumResult = union(enum) {
        finished,
        requeue,
        direct_capacity,
        blocked: *std.Io.Event,
    };

    ctx: *AdaptivePipelineContext,
    tensor_index: usize,
    reader: safetensors.TensorReader,
    writer: ?MemoryWriter = null,
    slots: []PageableReadSlot,
    total: usize,
    next_read: usize = 0,
    next_commit: usize = 0,
    head: usize = 0,
    pending: usize = 0,
    next_acquisition_ticket: usize = 0,
    acquisition_turn: usize = 0,
    acquisition_mutex: std.Io.Mutex = .init,
    acquisition_condition: std.Io.Condition = .init,
    direct_slot: DirectReadSlot = .{},
    direct_pending: bool = false,
    direct_read_reserved: bool = false,
    direct_offset: usize = 0,
    direct_len: usize = 0,
    quantum_progress: usize = 0,
    writer_parked: bool = false,
    writer_active_counted: bool = false,
    prefetch_counted: bool = false,
    queued_for_dma: bool = false,
    ready_next: ?*AdaptiveTensorLoad = null,
    in_direct_wait: bool = false,
    direct_wait_next: ?*AdaptiveTensorLoad = null,
    progress_node: ?std.Progress.Node = null,

    fn init(ctx: *AdaptivePipelineContext, tensor_index: usize) !*AdaptiveTensorLoad {
        const tensor = ctx.tensors[tensor_index];
        const self = try ctx.allocator.create(AdaptiveTensorLoad);
        errdefer ctx.allocator.destroy(self);

        var reader = try ctx.store.getReaderById(tensor.id, ctx.io, &.{});
        errdefer reader.deinit();

        const slots = try ctx.allocator.alloc(PageableReadSlot, ctx.max_read_parallelism);
        errdefer ctx.allocator.free(slots);
        @memset(slots, .{});

        self.* = .{
            .ctx = ctx,
            .tensor_index = tensor_index,
            .reader = reader,
            .slots = slots,
            .total = reader.tensor.shape.byteSize(),
        };
        load_log.debug("read source opened: index={d}, name={s}, file={s}, bytes={Bi:.2}, prefetched={d}", .{
            tensor_index,
            self.reader.tensor.name,
            self.reader.tensor.file_uri,
            self.total,
            ctx.prefetched_sources.load(.acquire),
        });
        return self;
    }

    fn ensureWriter(self: *AdaptiveTensorLoad) !void {
        if (self.writer != null) return;
        const shape = self.reader.tensor.shape;
        const sharding = Sharding.pickSharding(self.ctx.shardings, shape, .explicit_axis_binding) orelse blk: {
            log.debug("No sharding strategy found for tensor {s} with shape {f}, using replicated sharding", .{ self.reader.tensor.name, shape });
            break :blk self.ctx.platform.replicated_sharding;
        };
        self.writer = try MemoryWriter.initWithMetrics(
            self.ctx.allocator,
            self.ctx.io,
            self.ctx.platform,
            self.ctx.pinned_buffer_pools,
            self.ctx.dma_allocators,
            self.ctx.dma_chunk_size,
            shape,
            sharding,
            self.ctx.buffers[self.tensor_index],
            self.ctx.metrics,
        );
        if (self.ctx.progress) |progress| {
            const progress_total = std.math.divCeil(usize, self.total, 1024) catch unreachable;
            self.progress_node = progress.start(self.reader.tensor.name, progress_total);
            self.writer.?.setProgress(&self.progress_node.?);
        }
        self.writer_active_counted = true;
        _ = self.ctx.metrics.active_transfers.fetchAdd(1, .monotonic);
        load_log.debug("tensor started: index={d}, name={s}, bytes={Bi:.2}, active_dma_streams={d}", .{
            self.tensor_index,
            self.reader.tensor.name,
            self.total,
            self.ctx.metrics.active_transfers.load(.acquire),
        });
    }

    fn deinit(self: *AdaptiveTensorLoad) void {
        std.debug.assert(!self.in_direct_wait);
        if (self.direct_read_reserved) {
            self.direct_read_reserved = false;
            self.ctx.releaseDirectRead();
        }
        if (self.direct_pending) {
            self.direct_slot.event.waitUncancelable(self.ctx.io);
            self.direct_pending = false;
        }
        while (self.pending > 0) {
            const slot = &self.slots[self.head];
            slot.event.waitUncancelable(self.ctx.io);
            self.ctx.releaseSlot(slot);
            self.pending -= 1;
            self.head = (self.head + 1) % self.slots.len;
        }
        if (self.writer_active_counted) {
            _ = self.ctx.metrics.active_transfers.fetchSub(1, .monotonic);
            self.writer_active_counted = false;
        }
        if (self.writer) |*writer| writer.setProgress(null);
        if (self.progress_node) |*node| {
            node.end();
            self.progress_node = null;
        }
        if (self.writer) |*writer| writer.deinit(self.ctx.allocator);
        self.writer = null;
        self.reader.deinit();
        self.ctx.allocator.free(self.slots);
        self.ctx.allocator.destroy(self);
    }

    fn updateProgress(self: *AdaptiveTensorLoad) void {
        if (self.progress_node) |*node| node.setCompletedItems(self.next_commit / 1024);
    }

    fn promoteWriterError(self: *AdaptiveTensorLoad, fallback: anyerror) anyerror {
        return if (self.writer) |*writer| writer.rootError() orelse fallback else fallback;
    }

    fn fillReadAhead(self: *AdaptiveTensorLoad) void {
        const staging_limit = self.ctx.metrics.staging_limit.load(.acquire);
        if (staging_limit == 0) return;

        const active = @max(@as(usize, 1), self.ctx.admitted_lanes.load(.acquire));
        const per_tensor_limit = @min(
            self.slots.len,
            @max(@as(usize, 1), std.math.divCeil(usize, staging_limit, active) catch unreachable),
        );
        self.fillReadAheadTo(per_tensor_limit);
    }

    fn fillReadAheadTo(self: *AdaptiveTensorLoad, pending_limit: usize) void {
        if (self.ctx.metrics.staging_limit.load(.acquire) == 0) return;
        const limit = @min(pending_limit, self.slots.len);
        while (self.pending < limit and self.next_read < self.total and !self.ctx.failed()) {
            const slot_index = (self.head + self.pending) % self.slots.len;
            const len = @min(self.ctx.read_chunk_size, self.total - self.next_read);
            const ticket = self.next_acquisition_ticket;
            self.next_acquisition_ticket += 1;
            self.ctx.schedulePageable(self, ticket, &self.slots[slot_index], self.next_read, len);
            self.next_read += len;
            self.pending += 1;
        }
    }

    fn parkWriterAndWait(self: *AdaptiveTensorLoad) !void {
        if (self.writer == null or self.writer_parked) return;
        self.writer.?.parkAndWait() catch |err| return self.promoteWriterError(err);
        self.writer_parked = true;
    }

    fn waitForPendingDma(self: *AdaptiveTensorLoad) !void {
        if (self.writer == null) return;
        self.writer.?.waitForPendingDma() catch |err| return self.promoteWriterError(err);
    }

    fn resumeWriter(self: *AdaptiveTensorLoad) !void {
        if (self.writer == null or !self.writer_parked) return;
        self.writer.?.unpark() catch |err| return self.promoteWriterError(err);
        self.writer_parked = false;
    }

    fn finishAcquisition(self: *AdaptiveTensorLoad, io: std.Io) void {
        self.acquisition_turn += 1;
        self.acquisition_condition.broadcast(io);
        self.acquisition_mutex.unlock(io);
    }

    fn processQuantum(self: *AdaptiveTensorLoad, drain_for_park: bool) !QuantumResult {
        if (self.ctx.failed()) return error.LoadCancelled;
        std.debug.assert(self.writer != null);

        var quantum_bytes = self.quantum_progress;
        self.quantum_progress = 0;
        while (self.next_commit < self.total and quantum_bytes < self.ctx.dma_chunk_size and !self.ctx.failed()) {
            if (!drain_for_park and (self.direct_pending or self.pending > 0)) self.fillReadAhead();

            if (self.direct_pending) {
                if (!self.direct_slot.event.isSet()) {
                    self.quantum_progress = quantum_bytes;
                    return .{ .blocked = &self.direct_slot.event };
                }
                if (self.direct_slot.err) |err| {
                    self.direct_pending = false;
                    return err;
                }
                if (self.direct_offset != self.next_commit) return error.InvalidReadOrder;

                self.writer.?.commitDirectRead(self.direct_len) catch |err| return self.promoteWriterError(err);
                self.next_commit += self.direct_len;
                quantum_bytes += self.direct_len;
                _ = self.ctx.metrics.ordered_bytes.fetchAdd(@intCast(self.direct_len), .monotonic);
                self.updateProgress();
                self.direct_pending = false;
                self.direct_slot.reset();
                continue;
            }

            if (self.pending > 0) {
                const slot = &self.slots[self.head];
                if (!slot.event.isSet()) {
                    self.quantum_progress = quantum_bytes;
                    return .{ .blocked = &slot.event };
                }

                if (slot.err) |err| {
                    self.ctx.releaseSlot(slot);
                    self.pending -= 1;
                    self.head = (self.head + 1) % self.slots.len;
                    return err;
                }
                if (slot.offset + slot.consumed != self.next_commit) return error.InvalidReadOrder;

                self.ctx.markProbeStart(slot.epoch);
                self.writer.?.setEpoch(slot.epoch) catch |err| return self.promoteWriterError(err);
                const copy_len = @min(slot.len - slot.consumed, self.ctx.dma_chunk_size - quantum_bytes);
                const copy_started: std.Io.Timestamp = .now(self.ctx.io, .awake);
                self.writer.?.interface().writeAll(slot.buffer.?[slot.consumed..][0..copy_len]) catch |err| return self.promoteWriterError(err);
                self.writer.?.commitStagedWrite() catch |err| return self.promoteWriterError(err);
                const copy_elapsed = copy_started.untilNow(self.ctx.io, .awake);
                const ready_age = std.Io.Timestamp.fromNanoseconds(@intCast(slot.ready_at_ns)).untilNow(self.ctx.io, .awake);
                const ready_age_us: u64 = @intCast(@max(@divTrunc(ready_age.nanoseconds, std.time.ns_per_us), 0));
                _ = self.ctx.metrics.staged_copy_bytes.fetchAdd(@intCast(copy_len), .monotonic);
                _ = self.ctx.metrics.staged_copy_ns.fetchAdd(@intCast(@max(copy_elapsed.nanoseconds, 0)), .monotonic);
                _ = self.ctx.metrics.weighted_ready_age_us.fetchAdd(@as(u64, @intCast(copy_len)) *| ready_age_us, .monotonic);
                self.next_commit += copy_len;
                quantum_bytes += copy_len;
                _ = self.ctx.metrics.ordered_bytes.fetchAdd(@intCast(copy_len), .monotonic);
                self.updateProgress();
                self.ctx.consumeReady(slot, copy_len);
                if (slot.consumed == slot.len) {
                    self.ctx.releaseSlot(slot);
                    self.pending -= 1;
                    self.head = (self.head + 1) % self.slots.len;
                }
                continue;
            }

            if (self.next_read != self.next_commit) return error.InvalidReadOrder;
            if (drain_for_park) return .requeue;
            // This tensor owns the lane for one DMA quantum. Yielding merely
            // because another tensor is queued makes direct-only loads cycle
            // through the ready queue without scheduling their next read.
            const epoch = self.ctx.metrics.config_epoch.load(.acquire);
            self.writer.?.setEpoch(epoch) catch |err| return self.promoteWriterError(err);
            const writable = self.writer.?.directWritable();
            if (writable.len == 0) return error.NoDirectReadWindow;
            if (self.direct_read_reserved) {
                self.direct_read_reserved = false;
            } else if (self.ctx.direct_capacity_waiters.load(.acquire) > 0 or !self.ctx.tryReserveDirectRead()) {
                self.fillReadAheadTo(1);
                if (self.pending > 0) continue;
                return .direct_capacity;
            }
            const len = @min(
                self.ctx.read_chunk_size,
                @min(writable.len, @min(self.total - self.next_read, self.ctx.dma_chunk_size - quantum_bytes)),
            );
            self.direct_slot = .{ .scheduled_at = .now(self.ctx.io, .awake) };
            self.direct_pending = true;
            self.direct_offset = self.next_read;
            self.direct_len = len;
            self.ctx.scheduleDirect(&self.reader, writable[0..len], self.next_read, epoch, &self.direct_slot);
            self.next_read += len;
            self.fillReadAhead();
            if (!self.direct_slot.event.isSet()) {
                self.quantum_progress = quantum_bytes;
                return .{ .blocked = &self.direct_slot.event };
            }
        }

        if (self.ctx.failed()) return error.LoadCancelled;
        if (self.next_commit < self.total) return .requeue;

        self.writer.?.interface().flush() catch |err| return self.promoteWriterError(err);
        if (self.progress_node) |*node| node.setCompletedItems(std.math.divCeil(usize, self.total, 1024) catch unreachable);
        self.ctx.completeTensor(self.total);
        load_log.debug("tensor completed: name={s}, bytes={Bi:.2}, completed={d}/{d}", .{
            self.reader.tensor.name,
            self.total,
            self.ctx.metrics.completed_transfers.load(.acquire),
            self.ctx.tensors.len,
        });
        return .finished;
    }
};

const AdaptivePipelineLane = struct {
    const State = enum(u8) { idle, dma };

    ctx: *AdaptivePipelineContext,
    active: ?*AdaptiveTensorLoad = null,
    state: std.atomic.Value(State) = .init(.idle),
    last_dma_submission_epoch: std.atomic.Value(u64) = .init(0),

    fn deinitActive(self: *AdaptivePipelineLane) void {
        if (self.active) |active| active.deinit();
        self.active = null;
    }

    fn recordDmaSubmissionSince(self: *AdaptivePipelineLane, before: u64) void {
        const active = self.active orelse return;
        const writer = if (active.writer) |*writer_| writer_ else return;
        if (writer.submissionCount() <= before) return;
        const epoch = self.ctx.dma_probe_capacity_epoch.load(.acquire);
        if (epoch > 0) self.last_dma_submission_epoch.store(epoch, .release);
    }

    fn claimTensor(self: *AdaptivePipelineLane) !bool {
        if (self.ctx.popDmaReady()) |tensor_load| {
            self.active = tensor_load;
            try tensor_load.ensureWriter();
            return true;
        }

        self.ctx.ensurePrefetch();
        if (self.ctx.direct_capacity_waiters.load(.acquire) > 0) return false;
        if (self.ctx.direct_reads_pending.load(.acquire) >= self.ctx.dma_lane_limit.load(.acquire)) return false;
        const tensor_index = self.ctx.claimTensorIndex() orelse return false;
        self.active = try .init(self.ctx, tensor_index);
        try self.active.?.ensureWriter();
        return true;
    }

    fn detachActive(self: *AdaptivePipelineLane) *AdaptiveTensorLoad {
        const active = self.active.?;
        self.active = null;
        return active;
    }

    fn activateReserved(self: *AdaptivePipelineLane) bool {
        if (self.state.cmpxchgStrong(.idle, .dma, .acq_rel, .acquire) != null) return false;
        self.schedule();
        return true;
    }

    fn releaseAdmission(self: *AdaptivePipelineLane, replenish: bool) void {
        const previous = self.state.swap(.idle, .acq_rel);
        if (previous == .idle) return;
        _ = self.ctx.admitted_lanes.fetchSub(1, .acq_rel);
        if (replenish and self.ctx.metrics.completed_transfers.load(.acquire) < self.ctx.tensors.len) {
            self.ctx.ensureLanes(self.ctx.dma_lane_limit.load(.acquire));
        }
    }

    fn schedule(self: *AdaptivePipelineLane) void {
        if (!self.ctx.beginScheduling()) {
            self.deinitActive();
            self.releaseAdmission(false);
            return;
        }
        defer self.ctx.endScheduling();
        self.ctx.dma_group.concurrent(self.ctx.io, AdaptivePipelineLane.run, .{self}) catch |err| {
            self.ctx.fail(err);
            self.deinitActive();
            self.releaseAdmission(false);
        };
    }

    fn run(self: *AdaptivePipelineLane) void {
        const work_started: std.Io.Timestamp = .now(self.ctx.io, .awake);
        defer {
            const elapsed = work_started.untilNow(self.ctx.io, .awake);
            _ = self.ctx.metrics.dma_work_ns.fetchAdd(@intCast(@max(elapsed.nanoseconds, 0)), .monotonic);
        }
        if (self.ctx.failed()) {
            self.deinitActive();
            self.releaseAdmission(false);
            return;
        }
        if (self.active == null and !(self.claimTensor() catch |err| {
            self.ctx.fail(err);
            self.releaseAdmission(false);
            return;
        })) {
            self.releaseAdmission(false);
            if (self.ctx.hasDmaReady()) self.ctx.ensureLanes(self.ctx.dma_lane_limit.load(.acquire));
            return;
        }
        self.active.?.resumeWriter() catch |err| {
            self.ctx.fail(err);
            self.deinitActive();
            self.releaseAdmission(false);
            return;
        };
        const submissions_before = self.active.?.writer.?.submissionCount();
        const result = self.active.?.processQuantum(self.ctx.retirementNeeded()) catch |err| {
            self.recordDmaSubmissionSince(submissions_before);
            self.ctx.fail(err);
            self.deinitActive();
            self.releaseAdmission(false);
            return;
        };
        self.recordDmaSubmissionSince(submissions_before);
        switch (result) {
            .finished => {
                self.deinitActive();
                self.releaseAdmission(true);
            },
            .requeue => {
                if (!self.ctx.retirementNeeded() and !self.ctx.hasDmaReady()) {
                    self.schedule();
                    return;
                }
                self.active.?.parkWriterAndWait() catch |err| {
                    self.ctx.fail(err);
                    self.deinitActive();
                    self.releaseAdmission(false);
                    return;
                };
                const active = self.detachActive();
                self.ctx.enqueueDmaReady(active);
                self.releaseAdmission(true);
            },
            .direct_capacity => {
                self.active.?.parkWriterAndWait() catch |err| {
                    self.ctx.fail(err);
                    self.deinitActive();
                    self.releaseAdmission(false);
                    return;
                };
                const active = self.detachActive();
                self.ctx.requeueOnDirectCapacity(active);
                self.releaseAdmission(true);
            },
            .blocked => |event| {
                if (self.active.?.direct_pending) {
                    self.active.?.waitForPendingDma() catch |err| {
                        self.ctx.fail(err);
                        self.deinitActive();
                        self.releaseAdmission(false);
                        return;
                    };
                    const active = self.detachActive();
                    self.ctx.addDirectWait(active);
                    self.ctx.requeueDirectWhenReady(active, event);
                    self.releaseAdmission(true);
                    return;
                }
                self.active.?.parkWriterAndWait() catch |err| {
                    self.ctx.fail(err);
                    self.deinitActive();
                    self.releaseAdmission(false);
                    return;
                };
                const active = self.detachActive();
                self.ctx.requeueWhenReady(active, event);
                self.releaseAdmission(true);
            },
        }
    }
};

pub const LoadOpts = struct {
    pub const auto: LoadOpts = .{
        .parallelism = 1,
        .shardings = &.{},
        .dma_chunks = 2,
        .dma_chunk_size = 4096,
    };

    /// Hard maximum number of concurrent tensor transfers.
    parallelism: usize,
    /// Starting limit for adaptive loading. Clamped to the worker/chunk caps.
    initial_parallelism: usize = 2,
    /// Set false to retain the fixed `parallelism` behavior.
    adaptive_parallelism: bool = true,
    /// Hard maximum number of outstanding positional chunk reads. When null,
    /// adaptive loading uses up to twice `parallelism`, capped at 32 unless
    /// the DMA limit itself is larger.
    max_read_parallelism: ?usize = null,
    /// Logical pageable read-ahead chunk size. This is independent of the
    /// larger pinned DMA submission chunk size.
    read_chunk_size: usize = 32 * 1024 * 1024,
    /// Hard pageable read-ahead budget. Blocks are allocated lazily; zero
    /// disables pageable staging while retaining direct DMA adaptation.
    max_staging_bytes: usize = 1024 * 1024 * 1024,
    shardings: []const Sharding = &.{},
    progress: ?*std.Progress.Node = null,
    /// Hard per-device DMA chunk maximum. Adaptive loading normally uses one
    /// more chunk than active workers and probes higher only when useful.
    dma_chunks: usize,
    dma_chunk_size: usize,
    total_bytes: ?*usize = null,
};

fn adaptiveDmaWorkerCap(parallelism: usize, dma_chunks: usize, tensor_count: usize) usize {
    const chunk_cap = if (dma_chunks > 1) dma_chunks - 1 else 1;
    return @min(@max(tensor_count, 1), @min(parallelism, chunk_cap));
}

pub fn load(
    comptime ModelType: type,
    model: *const ModelType,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    opts: LoadOpts,
) !Bufferized(ModelType) {
    stdx.debug.assert(opts.parallelism > 0, "zml.io.load parallelism must be greater than zero", .{});
    stdx.debug.assert(opts.initial_parallelism > 0, "zml.io.load initial_parallelism must be greater than zero", .{});
    stdx.debug.assert(opts.dma_chunks > 0, "zml.io.load dma_chunks must be greater than zero", .{});
    stdx.debug.assert(opts.dma_chunk_size >= @sizeOf(usize), "zml.io.load dma_chunk_size must hold pool metadata", .{});
    stdx.debug.assert(opts.read_chunk_size > 0, "zml.io.load read_chunk_size must be greater than zero", .{});

    const load_started: std.Io.Timestamp = .now(io, .awake);
    const tensor_count = meta.count(Tensor, model);
    var span = tracer.span("zml.io.load", .{
        .tensor_count = tensor_count,
    });
    defer span.end();

    var bufferized = try mem.bufferize(allocator, ModelType, model);
    errdefer meta.forEachVisit(&bufferized, *Buffer, struct {
        fn call(_: usize, buffer: *Buffer) void {
            buffer.deinit();
        }
    }.call, .{});

    var total_logical_bytes: u64 = 0;
    meta.forEachVisit(model, *const Tensor, struct {
        fn call(_: usize, tensor: *const Tensor, total: *u64) void {
            total.* += tensor.byteSize();
        }
    }.call, .{&total_logical_bytes});

    const direct = platform.target == .cuda or platform.target == .oneapi;
    const default_read_parallelism = @max(opts.parallelism, @min(@as(usize, 32), opts.parallelism *| 2));
    const max_read_workers = opts.max_read_parallelism orelse default_read_parallelism;
    stdx.debug.assert(max_read_workers > 0, "zml.io.load max_read_parallelism must be greater than zero", .{});
    const max_staging_chunks = if (opts.max_staging_bytes < opts.read_chunk_size) 0 else opts.max_staging_bytes / opts.read_chunk_size;
    if (max_staging_chunks > 0) {
        stdx.debug.assert(opts.read_chunk_size >= @sizeOf(usize), "zml.io.load read_chunk_size must hold pool metadata", .{});
    }
    const adaptive_candidate = opts.adaptive_parallelism and direct and total_logical_bytes > opts.read_chunk_size and (opts.parallelism > 1 or max_read_workers > 1);
    const max_workers = if (adaptive_candidate)
        adaptiveDmaWorkerCap(opts.parallelism, opts.dma_chunks, tensor_count)
    else
        opts.parallelism;
    const adaptive = adaptive_candidate and (max_workers > 1 or max_read_workers > 1);
    const initial_workers = if (adaptive) @min(max_workers, opts.initial_parallelism) else max_workers;
    const initial_read_workers = if (adaptive) @min(max_read_workers, opts.initial_parallelism) else max_read_workers;
    const initial_chunks = if (adaptive) @min(opts.dma_chunks, initial_workers + 1) else opts.dma_chunks;
    var metrics: LoadMetrics = .{};
    load_log.debug("configured: target={s}, adaptive={}, requested_adaptive={}, tensors={d}, reads={d}/{d}, dma={d}/{d} (requested_max={d}), dma_chunks={d}/{d}, staging=0/{d} blocks ({Bi:.2} max), read_chunk_size={Bi:.2}, dma_chunk_size={Bi:.2}, logical_bytes={Bi:.2}", .{
        @tagName(platform.target),
        adaptive,
        opts.adaptive_parallelism,
        tensor_count,
        initial_read_workers,
        max_read_workers,
        initial_workers,
        max_workers,
        opts.parallelism,
        initial_chunks,
        opts.dma_chunks,
        max_staging_chunks,
        opts.max_staging_bytes,
        opts.read_chunk_size,
        opts.dma_chunk_size,
        total_logical_bytes,
    });

    const pool_count = platform.devices.len;
    const dma_allocators = try allocator.alloc(mem.DmaAllocator, pool_count);
    defer allocator.free(dma_allocators);
    for (platform.devices, 0..) |*device, i| {
        dma_allocators[i] = .init(allocator, device);
    }

    const buffer_pools = try allocator.alloc(mem.DynamicBufferPool, pool_count);
    defer allocator.free(buffer_pools);
    for (buffer_pools) |*pool_| {
        pool_.* = .init(opts.dma_chunks, opts.dma_chunk_size);
        if (adaptive) pool_.setLimit(io, initial_chunks);
    }
    defer for (buffer_pools, 0..) |*pool_, i| {
        pool_.deinit(dma_allocators[i].allocator());
    };

    if (adaptive) {
        const buffers = try allocator.alloc(*Buffer, tensor_count);
        defer allocator.free(buffers);
        const tensors = try allocator.alloc(*const Tensor, tensor_count);
        defer allocator.free(tensors);
        meta.forEachVisit(&bufferized, *Buffer, struct {
            fn call(i: usize, buffer: *Buffer, buffers_: []*Buffer) void {
                buffers_[i] = buffer;
            }
        }.call, .{buffers});
        meta.forEachVisit(model, *const Tensor, struct {
            fn call(i: usize, tensor: *const Tensor, tensors_: []*const Tensor) void {
                tensors_[i] = tensor;
            }
        }.call, .{tensors});

        var staging_pool_storage: mem.DynamicBufferPool = undefined;
        const staging_pool: ?*mem.DynamicBufferPool = if (max_staging_chunks > 0) blk: {
            staging_pool_storage = .init(max_staging_chunks, opts.read_chunk_size);
            staging_pool_storage.setLimit(io, 1);
            break :blk &staging_pool_storage;
        } else null;
        defer if (staging_pool) |pool| pool.deinit(allocator);

        var pipeline: AdaptivePipelineContext = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .store = store,
            .tensors = tensors,
            .buffers = buffers,
            .shardings = opts.shardings,
            .dma_allocators = dma_allocators,
            .pinned_buffer_pools = buffer_pools,
            .staging_pool = staging_pool,
            .dma_chunk_size = opts.dma_chunk_size,
            .read_chunk_size = opts.read_chunk_size,
            .max_read_parallelism = max_read_workers,
            .dma_group = .init(initial_workers),
            .read_group = .init(initial_read_workers),
            .metrics = &metrics,
            .progress = opts.progress,
            .dma_lane_limit = .init(initial_workers),
        };
        const lanes = try allocator.alloc(AdaptivePipelineLane, max_workers);
        defer allocator.free(lanes);
        pipeline.lanes = lanes;
        for (lanes) |*lane| lane.* = .{ .ctx = &pipeline };

        var controller_runtime: AdaptiveLoadRuntime = .{
            .controller = .init(
                initial_workers,
                max_workers,
                initial_chunks,
                opts.dma_chunks,
                initial_read_workers,
                max_read_workers,
                max_staging_chunks,
                direct,
            ),
            .pipeline = &pipeline,
            .dma_group = &pipeline.dma_group,
            .read_group = &pipeline.read_group,
            .pools = buffer_pools,
            .dma_allocators = dma_allocators,
            .staging_pool = staging_pool,
            .staging_allocator = allocator,
            .metrics = &metrics,
            .dma_chunk_size = opts.dma_chunk_size,
            .read_chunk_size = opts.read_chunk_size,
            .total_logical_bytes = total_logical_bytes,
            .total_transfers = tensor_count,
            .probe_started = .now(io, .awake),
        };
        var controller_group: std.Io.Group = .init;
        controller_group.concurrent(io, AdaptiveLoadRuntime.run, .{ &controller_runtime, io }) catch unreachable;

        pipeline.ensureLanes(initial_workers);
        pipeline.done_event.waitUncancelable(io);

        controller_runtime.done.store(true, .release);
        controller_group.await(io) catch |err| pipeline.fail(err);
        pipeline.scheduling_idle.waitUncancelable(io);
        if (pipeline.failed()) {
            pipeline.read_group.cancel(io);
            pipeline.staging_group.cancel(io);
            pipeline.deinitReadyLoads();
            pipeline.resume_group.await(io) catch |err| pipeline.fail(err);
            pipeline.dma_group.await(io) catch |err| pipeline.fail(err);
            for (lanes) |*lane| lane.deinitActive();
            pipeline.deinitReadyLoads();
        } else {
            pipeline.read_group.await(io) catch |err| pipeline.fail(err);
            pipeline.staging_group.await(io) catch |err| pipeline.fail(err);
            pipeline.resume_group.await(io) catch |err| pipeline.fail(err);
            pipeline.dma_group.await(io) catch |err| pipeline.fail(err);
            for (lanes) |*lane| lane.deinitActive();
        }
        pipeline.deinitReadyLoads();
        std.debug.assert(pipeline.admitted_lanes.load(.acquire) == 0);
        std.debug.assert(pipeline.direct_wait_head == null);
        std.debug.assert(pipeline.direct_reads_pending.load(.acquire) == 0);
        std.debug.assert(pipeline.direct_capacity_waiters.load(.acquire) == 0);

        const loaded_bytes = pipeline.total.load(.monotonic);
        if (opts.total_bytes) |total_bytes_ptr| total_bytes_ptr.* = loaded_bytes;
        if (pipeline.errorValue()) |err| return err;

        const elapsed = load_started.untilNow(io, .awake);
        const elapsed_seconds = @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_s;
        const goodput = if (elapsed_seconds > 0) @as(f64, @floatFromInt(loaded_bytes)) / elapsed_seconds else 0;
        load_log.debug("completed: adaptive=true, tensors={d}, logical_bytes={Bi:.2}, elapsed={d:.3}s, logical_goodput={d:.2}MiB/s, final_reads={d}, final_dma={d}, final_dma_chunks={d}, final_staging={d}, direct={Bi:.2}, staged={Bi:.2}", .{
            tensor_count,
            loaded_bytes,
            elapsed_seconds,
            goodput / (1024 * 1024),
            controller_runtime.controller.knobs.read_workers,
            controller_runtime.controller.knobs.dma_workers,
            controller_runtime.controller.knobs.dma_chunks,
            controller_runtime.controller.knobs.staging_chunks,
            metrics.direct_read_bytes.load(.acquire),
            metrics.staged_read_bytes.load(.acquire),
        });
        return bufferized;
    }

    const Ctx = struct {
        allocator: std.mem.Allocator,
        dma_allocators: []const mem.DmaAllocator,
        dma_chunk_size: usize,
        pinned_buffer_pools: []mem.DynamicBufferPool,
        io: std.Io,
        platform: *const Platform,
        buffers: []*Buffer,
        shardings: []const Sharding,
        store: *const TensorStore,
        group: stdx.Io.LimitedGroup,
        total: std.atomic.Value(usize) = .init(0),
        progress: ?*std.Progress.Node,
        metrics: ?*LoadMetrics,
        direct: bool,
    };
    var walk_ctx: Ctx = .{
        .platform = platform,
        .buffers = try allocator.alloc(*Buffer, tensor_count),
        .store = store,
        .allocator = allocator,
        .dma_allocators = dma_allocators,
        .dma_chunk_size = opts.dma_chunk_size,
        .pinned_buffer_pools = buffer_pools,
        .io = io,
        .shardings = opts.shardings,
        .progress = opts.progress,
        .group = .init(max_workers),
        .metrics = null,
        .direct = direct,
    };
    defer allocator.free(walk_ctx.buffers);

    defer if (opts.total_bytes) |total_bytes_ptr| {
        total_bytes_ptr.* = walk_ctx.total.load(.monotonic);
    };

    meta.forEachVisit(&bufferized, *Buffer, struct {
        fn call(i: usize, buffer: *Buffer, ctx: *Ctx) void {
            ctx.buffers[i] = buffer;
        }
    }.call, .{&walk_ctx});

    meta.forEachVisit(model, *const Tensor, struct {
        fn call(i: usize, tensor: *const Tensor, ctx: *Ctx) void {
            ctx.group.concurrent(ctx.io, struct {
                fn call(i_: usize, tensor_: *const Tensor, ctx_: *Ctx) !void {
                    const transfer_started: std.Io.Timestamp = .now(ctx_.io, .awake);
                    if (ctx_.metrics) |metrics_| {
                        _ = metrics_.active_transfers.fetchAdd(1, .monotonic);
                    }
                    defer if (ctx_.metrics) |metrics_| {
                        _ = metrics_.active_transfers.fetchSub(1, .monotonic);
                        _ = metrics_.completed_transfers.fetchAdd(1, .release);
                    };

                    var reader = ctx_.store.getReaderById(tensor_.id, ctx_.io, &.{}) catch unreachable;
                    defer reader.deinit();

                    const shape = reader.tensor.shape;
                    const sharding = Sharding.pickSharding(ctx_.shardings, shape, .explicit_axis_binding) orelse blk: {
                        log.debug("No sharding strategy found for tensor {s} with shape {f}, using replicated sharding", .{ reader.tensor.name, shape });
                        break :blk ctx_.platform.replicated_sharding;
                    };

                    var writer = MemoryWriter.initWithMetrics(
                        ctx_.allocator,
                        ctx_.io,
                        ctx_.platform,
                        ctx_.pinned_buffer_pools,
                        ctx_.dma_allocators,
                        ctx_.dma_chunk_size,
                        shape,
                        sharding,
                        ctx_.buffers[i_],
                        ctx_.metrics,
                    ) catch unreachable;
                    defer writer.deinit(ctx_.allocator);

                    const scale = 1024;

                    const total = if (ctx_.progress) |progress| blk: {
                        var node = progress.start(reader.tensor.name, reader.tensor.shape.byteSize() / scale);
                        defer node.end();
                        writer.setProgress(&node);
                        defer writer.setProgress(null);
                        var progress_writer: ProgressWriter = .init(writer.interface(), &node, .{ .scale = scale });
                        const total_ = reader.interface.streamRemaining(&progress_writer.interface) catch unreachable;
                        progress_writer.interface.flush() catch unreachable;
                        break :blk total_;
                    } else blk: {
                        const total_ = reader.interface.streamRemaining(writer.interface()) catch unreachable;
                        writer.interface().flush() catch unreachable;
                        break :blk total_;
                    };
                    _ = ctx_.total.fetchAdd(total, .monotonic);

                    if (ctx_.metrics) |metrics_| {
                        if (!ctx_.direct) {
                            const elapsed = transfer_started.untilNow(ctx_.io, .awake);
                            const elapsed_us: u64 = @intCast(@max(@divTrunc(elapsed.nanoseconds, std.time.ns_per_us), 0));
                            _ = metrics_.storage_bytes.fetchAdd(@intCast(total), .monotonic);
                            _ = metrics_.submitted_bytes.fetchAdd(@intCast(total), .monotonic);
                            _ = metrics_.committed_bytes.fetchAdd(@intCast(total), .monotonic);
                            _ = metrics_.weighted_transfer_latency_us.fetchAdd(@as(u64, @intCast(total)) *| elapsed_us, .monotonic);
                        }
                    }
                }
            }.call, .{ i, tensor, ctx }) catch unreachable;
        }
    }.call, .{&walk_ctx});
    walk_ctx.group.await(io) catch unreachable;

    const loaded_bytes = walk_ctx.total.load(.monotonic);
    const elapsed = load_started.untilNow(io, .awake);
    const elapsed_seconds = @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_s;
    const goodput = if (elapsed_seconds > 0) @as(f64, @floatFromInt(loaded_bytes)) / elapsed_seconds else 0;
    load_log.debug("completed: adaptive=false, tensors={d}, logical_bytes={Bi:.2}, elapsed={d:.3}s, logical_goodput={d:.2}MiB/s, workers={d}, dma_chunks={d}", .{
        tensor_count,
        loaded_bytes,
        elapsed_seconds,
        goodput / (1024 * 1024),
        max_workers,
        opts.dma_chunks,
    });

    return bufferized;
}

test "adaptive pipeline scheduling fence drains admitted producers" {
    const io = std.testing.io;
    var pipeline: AdaptivePipelineContext = undefined;
    pipeline.io = io;
    pipeline.scheduling_state = .init(0);
    pipeline.scheduling_idle = .unset;

    try std.testing.expect(pipeline.beginScheduling());
    try std.testing.expect(pipeline.beginScheduling());
    pipeline.closeScheduling();
    try std.testing.expect(!pipeline.beginScheduling());
    try std.testing.expect(!pipeline.scheduling_idle.isSet());

    pipeline.endScheduling();
    try std.testing.expect(!pipeline.scheduling_idle.isSet());
    pipeline.endScheduling();
    try std.testing.expect(pipeline.scheduling_idle.isSet());
}

test "adaptive load metrics count overlapping DMA starvation once" {
    var metrics: LoadMetrics = .{};
    metrics.addDmaStarvationInterval(100, 200);
    metrics.addDmaStarvationInterval(150, 250);
    metrics.addDmaStarvationInterval(120, 180);
    metrics.addDmaStarvationInterval(300, 350);

    try std.testing.expectEqual(@as(u64, 200), metrics.dma_starved_ns.load(.acquire));
}

test "adaptive pipeline prefetch reservations are independent of dma lanes" {
    var tensors: [8]*const Tensor = undefined;
    var pipeline: AdaptivePipelineContext = undefined;
    pipeline.tensors = &tensors;
    pipeline.next_tensor = .init(0);
    pipeline.prefetched_sources = .init(0);

    for (0..6) |expected| {
        try std.testing.expect(pipeline.reservePrefetchSource(6));
        try std.testing.expectEqual(expected, pipeline.claimTensorIndex().?);
    }
    try std.testing.expect(!pipeline.reservePrefetchSource(6));

    _ = pipeline.prefetched_sources.fetchSub(1, .release);
    try std.testing.expect(pipeline.reservePrefetchSource(6));
    try std.testing.expectEqual(@as(usize, 6), pipeline.claimTensorIndex().?);
}

test "adaptive pipeline dma-ready queue owns stable tensor pointers" {
    const io = std.testing.io;
    var pipeline: AdaptivePipelineContext = undefined;
    pipeline.io = io;
    pipeline.first_error = .init(0);
    pipeline.lanes = &.{};
    pipeline.dma_lane_limit = .init(0);
    pipeline.admitted_lanes = .init(0);
    pipeline.ready_mutex = .init;
    pipeline.ready_head = null;
    pipeline.ready_tail = null;

    var loads: [3]AdaptiveTensorLoad = undefined;
    for (&loads) |*load_| {
        load_.ready_next = null;
        load_.queued_for_dma = false;
        load_.prefetch_counted = false;
        pipeline.enqueueDmaReady(load_);
    }
    try std.testing.expect(pipeline.popDmaReady() == &loads[0]);
    try std.testing.expect(pipeline.popDmaReady() == &loads[1]);
    try std.testing.expect(pipeline.popDmaReady() == &loads[2]);
    try std.testing.expectEqual(@as(?*AdaptiveTensorLoad, null), pipeline.popDmaReady());
    try std.testing.expectEqual(@as(?*AdaptiveTensorLoad, null), pipeline.ready_tail);
}

test "adaptive dma worker cap does not exceed tensor lanes" {
    try std.testing.expectEqual(@as(usize, 3), adaptiveDmaWorkerCap(16, 32, 3));
    try std.testing.expectEqual(@as(usize, 4), adaptiveDmaWorkerCap(16, 5, 32));
    try std.testing.expectEqual(@as(usize, 1), adaptiveDmaWorkerCap(16, 1, 0));
}

test "adaptive load controller doubles staged read-ahead probes" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);

    var decision = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.11,
        .reads_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_start, decision.action);
    try std.testing.expectEqual(2, decision.knobs.staging_chunks);
    try std.testing.expectEqual(4, decision.knobs.read_workers);

    decision = controller.observe(.{
        .committed_goodput = 150,
        .probe_goodput = 150,
        .probe_committed_bytes = 63 * 1024 * 1024,
        .dma_starvation_ratio = 0.11,
        .reads_saturated = true,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 105,
        .probe_goodput = 105,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 300 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_keep, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 105,
        .dma_starvation_ratio = 0.11,
        .reads_saturated = true,
        .now_ns = 400 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_start, decision.action);
    try std.testing.expectEqual(8, decision.knobs.staging_chunks);
    try std.testing.expectEqual(8, decision.knobs.read_workers);

    decision = controller.observe(.{
        .committed_goodput = 110,
        .probe_goodput = 110,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 500 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_keep, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 110,
        .dma_starvation_ratio = 0.11,
        .reads_saturated = true,
        .now_ns = 600 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_start, decision.action);
    try std.testing.expectEqual(16, decision.knobs.read_workers);
    try std.testing.expectEqual(16, decision.knobs.staging_chunks);
}

test "adaptive load controller probes direct read capacity before staging" {
    var controller: AdaptiveLoadController = .init(4, 16, 5, 32, 2, 32, 32, true);
    const decision = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.11,
        .read_capacity_demand = true,
        .reads_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });

    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expectEqual(4, decision.knobs.read_workers);
    try std.testing.expectEqual(0, decision.knobs.staging_chunks);
}

test "adaptive load controller caps reads by pageable staging budget" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 3, true);
    var decision = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.11,
        .read_capacity_demand = true,
        .reads_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(4, decision.knobs.read_workers);
    try std.testing.expectEqual(2, decision.knobs.staging_chunks);

    decision = controller.observe(.{
        .committed_goodput = 104,
        .probe_goodput = 104,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_keep, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 104,
        .dma_starvation_ratio = 0.11,
        .read_capacity_demand = true,
        .reads_saturated = true,
        .now_ns = 300 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_start, decision.action);
    try std.testing.expectEqual(5, decision.knobs.read_workers);
    try std.testing.expectEqual(3, decision.knobs.staging_chunks);
}

test "adaptive load controller rolls back unattributed gain" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    _ = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.11,
        .reads_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });

    const decision = controller.observe(.{
        .committed_goodput = 150,
        .probe_goodput = 102,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_rollback, decision.action);
    try std.testing.expectEqual(0, decision.knobs.staging_chunks);
    try std.testing.expectEqual(2, decision.knobs.read_workers);
    try std.testing.expect(decision.trim_staging);

    const cooldown = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.11,
        .reads_saturated = true,
        .now_ns = 300 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, cooldown.action);
    try std.testing.expectEqual(0, cooldown.knobs.staging_chunks);
}

test "adaptive load controller does not back reads off past a probe baseline" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    var decision = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.11,
        .reads_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_start, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 100,
        .ready_growth_ratio = 0.21,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_rollback, decision.action);
    try std.testing.expectEqual(.ready_queue_growth, decision.reason);
    try std.testing.expectEqual(2, decision.knobs.read_workers);
    try std.testing.expectEqual(0, decision.knobs.staging_chunks);
    try std.testing.expect(decision.trim_staging);
}

test "adaptive load controller backs reads off on a growing ready queue" {
    var controller: AdaptiveLoadController = .init(4, 16, 5, 32, 8, 32, 32, true);
    controller.knobs.staging_chunks = 16;
    const decision = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 100,
        .ready_growth_ratio = 0.21,
        .reads_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.read_backoff, decision.action);
    try std.testing.expectEqual(5, decision.knobs.read_workers);
    try std.testing.expectEqual(4, decision.knobs.dma_workers);
    try std.testing.expectEqual(5, decision.knobs.dma_chunks);
    try std.testing.expectEqual(4, decision.knobs.staging_chunks);
    try std.testing.expect(decision.trim_staging);
}

test "adaptive load controller does not confuse saturated read latency with queue pressure" {
    var controller: AdaptiveLoadController = .init(2, 2, 3, 3, 8, 8, 6, true);
    controller.mode = .steady;
    controller.knobs.staging_chunks = 6;

    var decision = controller.observe(.{
        .committed_goodput = 100,
        .read_latency_us = 100,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 100,
        .read_latency_us = 121,
        .read_admission_wait_ratio = 1,
        .reads_saturated = true,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(@as(usize, 8), decision.knobs.read_workers);
}

test "adaptive load controller trims staging after reads reach one worker" {
    var controller: AdaptiveLoadController = .init(2, 4, 3, 8, 1, 8, 32, true);
    controller.knobs.staging_chunks = 8;
    const decision = controller.observe(.{
        .committed_goodput = 100,
        .ready_growth_ratio = 0.21,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_backoff, decision.action);
    try std.testing.expectEqual(1, decision.knobs.read_workers);
    try std.testing.expectEqual(2, decision.knobs.staging_chunks);
    try std.testing.expect(decision.trim_staging);
}

test "adaptive load controller keeps fast direct reads out of pageable memory" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    const decision = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.05,
        .reads_saturated = true,
        .now_ns = 600 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(0, decision.knobs.staging_chunks);
    try std.testing.expectEqual(.dma_probe_start, decision.action);
}

test "adaptive load controller rolls back an unhelpful dma startup probe" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    var decision = controller.observe(.{
        .committed_goodput = 100,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expectEqual(4, decision.knobs.dma_workers);
    try std.testing.expectEqual(5, decision.knobs.dma_chunks);

    decision = controller.observe(.{
        .committed_goodput = 102,
        .probe_goodput = 102,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_rollback, decision.action);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(3, decision.knobs.dma_chunks);
    try std.testing.expect(decision.trim_pinned);
}

test "adaptive load controller waits for DMA probe capacity before evaluating pressure" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    var decision = controller.observe(.{
        .committed_goodput = 100,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);

    decision = controller.observe(.{
        .capacity_pending = true,
        .committed_goodput = 1,
        .transfer_latency_us = 1000,
        .pinned_wait_ratio = 1,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expect(controller.probe != null);
    try std.testing.expectEqual(@as(usize, 4), decision.knobs.dma_workers);
}

test "adaptive load controller does not back off past a pressured probe baseline" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    var decision = controller.observe(.{
        .committed_goodput = 100,
        .transfer_latency_us = 100,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expectEqual(4, decision.knobs.dma_workers);

    decision = controller.observe(.{
        .committed_goodput = 100,
        .transfer_latency_us = 121,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_rollback, decision.action);
    try std.testing.expectEqual(.h2d_pressure, decision.reason);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(3, decision.knobs.dma_chunks);
    try std.testing.expect(decision.trim_pinned);

    decision = controller.observe(.{
        .committed_goodput = 100,
        .transfer_latency_us = 140,
        .now_ns = 300 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);

    decision = controller.observe(.{
        .committed_goodput = 100,
        .transfer_latency_us = 140,
        .now_ns = 500 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_backoff, decision.action);
    try std.testing.expectEqual(1, decision.knobs.dma_workers);
}

test "adaptive load controller rolls a stalled pressured probe back without committed bytes" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    var decision = controller.observe(.{
        .committed_goodput = 100,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 0,
        .pinned_wait_ratio = 0.21,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_rollback, decision.action);
    try std.testing.expectEqual(.pinned_wait, decision.reason);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(3, decision.knobs.dma_chunks);
    try std.testing.expect(decision.trim_pinned);
}

test "adaptive load controller keeps a dma probe when pageable data is ready" {
    var controller: AdaptiveLoadController = .init(1, 2, 2, 3, 1, 2, 4, true);
    controller.knobs.staging_chunks = 1;
    var decision = controller.observe(.{
        .committed_goodput = 100,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 104,
        .probe_goodput = 104,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .ready_growth_ratio = 0.21,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_keep, decision.action);
    try std.testing.expectEqual(1, decision.knobs.read_workers);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(3, decision.knobs.dma_chunks);
    try std.testing.expectEqual(1, decision.knobs.staging_chunks);
    try std.testing.expect(!decision.trim_pinned);
}

test "adaptive load controller timeout rolls a dma probe back and trims resources" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    const started = controller.observe(.{
        .committed_goodput = 100,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_start, started.action);

    const published_epoch = controller.epoch;
    const decision = controller.rollbackTimedOutProbe(600 * std.time.ns_per_ms, published_epoch).?;
    try std.testing.expectEqual(.dma_probe_rollback, decision.action);
    try std.testing.expectEqual(.capacity_not_exercised, decision.reason);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(3, decision.knobs.dma_chunks);
    try std.testing.expect(decision.trim_pinned);
}

test "adaptive load controller timeout restores an unactivated published epoch" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    const started = controller.observe(.{
        .committed_goodput = 100,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(@as(u64, 1), started.epoch);

    const decision = controller.rollbackTimedOutProbe(600 * std.time.ns_per_ms, 0).?;
    try std.testing.expectEqual(@as(u64, 0), decision.epoch);
    try std.testing.expectEqual(@as(u64, 0), controller.epoch);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
}

test "adaptive load runtime only times out an active idle probe" {
    const timeout_ns = 500 * std.time.ns_per_ms;
    try std.testing.expect(!AdaptiveLoadRuntime.stalledProbeTimedOut(false, timeout_ns - 1, 0));
    try std.testing.expect(AdaptiveLoadRuntime.stalledProbeTimedOut(false, timeout_ns, 0));
    try std.testing.expect(!AdaptiveLoadRuntime.stalledProbeTimedOut(true, timeout_ns, 0));
    try std.testing.expect(!AdaptiveLoadRuntime.stalledProbeTimedOut(false, timeout_ns, 1));
}

test "adaptive load controller tolerates transient H2D growth during a probe" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    var decision = controller.observe(.{
        .committed_goodput = 100,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 104,
        .probe_goodput = 104,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .h2d_growth_ratio = 0.21,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_keep, decision.action);
    try std.testing.expectEqual(4, decision.knobs.dma_workers);
}

test "adaptive load controller skips probes near the finite tail" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    const decision = controller.observe(.{
        .committed_goodput = 100,
        .dma_saturated = true,
        .allow_probe = false,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(3, decision.knobs.dma_chunks);
}

test "adaptive load controller honors a disabled staging budget" {
    var controller: AdaptiveLoadController = .init(1, 4, 2, 5, 2, 32, 0, true);
    const decision = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .reads_saturated = true,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expectEqual(0, decision.knobs.staging_chunks);
}

test "adaptive load controller waits for committed baseline before probing" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    const decision = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 0,
        .dma_starvation_ratio = 0.50,
        .reads_saturated = true,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(2, decision.knobs.read_workers);
    try std.testing.expectEqual(0, decision.knobs.staging_chunks);
    try std.testing.expectEqual(null, controller.probe);
}

test "adaptive load controller ignores cold pinned allocation pressure" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    var decision = controller.observe(.{
        .read_goodput = 3402,
        .committed_goodput = 0,
        .pinned_wait_ratio = 0.32,
        .h2d_queue_ratio = 0.33,
        .h2d_growth_ratio = 0.33,
        .admitted_dma_writers = 2,
        .now_ns = 75 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(.startup, controller.mode);
    try std.testing.expectEqual(0, controller.h2d_growth_windows);

    decision = controller.observe(.{
        .committed_goodput = 7600,
        .h2d_queue_ratio = 0.66,
        .h2d_growth_ratio = 0.33,
        .now_ns = 175 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(1, controller.h2d_growth_windows);
}

test "adaptive load controller recovers a starved direct lane" {
    var controller: AdaptiveLoadController = .init(1, 16, 2, 32, 2, 32, 32, true);
    controller.mode = .steady;
    const decision = controller.observe(.{
        .read_goodput = 9000,
        .committed_goodput = 9000,
        .dma_starvation_ratio = 0.80,
        .admitted_dma_writers = 1,
        .dma_lane_capacity_available = true,
        .now_ns = 600 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(3, decision.knobs.dma_chunks);
}

test "adaptive load controller bootstraps read ahead for a slow source" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    const decision = controller.observe(.{
        .read_goodput = 100,
        .committed_goodput = 0,
        .slow_reads = true,
        .reads_saturated = true,
        .now_ns = 75 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.read_ahead_bootstrap, decision.action);
    try std.testing.expectEqual(4, decision.knobs.read_workers);
    try std.testing.expectEqual(2, decision.knobs.staging_chunks);
    try std.testing.expectEqual(null, controller.probe);
}

test "adaptive load controller bootstraps repeatedly while the source is stalled" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    var decision = controller.observe(.{
        .source_stalled = true,
        .read_capacity_demand = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.read_ahead_bootstrap, decision.action);
    try std.testing.expectEqual(4, decision.knobs.read_workers);
    try std.testing.expectEqual(2, decision.knobs.staging_chunks);

    decision = controller.observe(.{
        .source_stalled = true,
        .read_capacity_demand = true,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.read_ahead_bootstrap, decision.action);
    try std.testing.expectEqual(8, decision.knobs.read_workers);
    try std.testing.expectEqual(8, decision.knobs.staging_chunks);
    try std.testing.expectEqual(null, controller.probe);
}

test "adaptive load controller source bootstrap stops at the read hard cap" {
    var controller: AdaptiveLoadController = .init(2, 16, 3, 32, 2, 32, 32, true);
    for (0..4) |step| {
        const decision = controller.observe(.{
            .source_stalled = true,
            .read_capacity_demand = true,
            .now_ns = (step + 1) * 100 * std.time.ns_per_ms,
        });
        try std.testing.expectEqual(.read_ahead_bootstrap, decision.action);
    }
    try std.testing.expectEqual(@as(usize, 32), controller.knobs.read_workers);
    try std.testing.expectEqual(@as(usize, 32), controller.knobs.staging_chunks);
    try std.testing.expectEqual(@as(usize, 2), controller.knobs.dma_workers);
    try std.testing.expectEqual(@as(usize, 3), controller.knobs.dma_chunks);

    const capped = controller.observe(.{
        .source_stalled = true,
        .read_capacity_demand = true,
        .now_ns = 500 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, capped.action);
    try std.testing.expect(!capped.changed);
}

test "adaptive load controller reduces reads and their required staging together" {
    var controller: AdaptiveLoadController = .init(2, 2, 3, 3, 8, 8, 6, true);
    controller.mode = .steady;
    controller.knobs.staging_chunks = 6;
    const decision = controller.observe(.{
        .committed_goodput = 100,
        .now_ns = 2 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 4), decision.knobs.read_workers);
    try std.testing.expectEqual(@as(usize, 4), decision.knobs.staging_chunks);
}

test "adaptive load controller reduces DMA streams within the best throughput band" {
    var controller: AdaptiveLoadController = .init(4, 4, 5, 5, 4, 4, 0, true);
    controller.mode = .steady;

    var decision = controller.observe(.{
        .committed_goodput = 100,
        .now_ns = 2 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expectEqual(.reduce_resource, controller.probe.?.kind);
    try std.testing.expectEqual(@as(usize, 2), decision.knobs.dma_workers);
    try std.testing.expectEqual(@as(usize, 5), decision.knobs.dma_chunks);

    decision = controller.observe(.{
        .committed_goodput = 97,
        .probe_goodput = 97,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 3 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.dma_probe_keep, decision.action);
    try std.testing.expectEqual(@as(usize, 2), decision.knobs.dma_workers);
}

test "adaptive load controller rolls a failed resource reduction back one step" {
    var controller: AdaptiveLoadController = .init(4, 4, 6, 8, 4, 4, 8, true);
    controller.mode = .steady;

    var decision = controller.observe(.{
        .committed_goodput = 100,
        .now_ns = 2 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.pinned_reduce_start, decision.action);
    try std.testing.expectEqual(5, decision.knobs.dma_chunks);

    decision = controller.observe(.{
        .committed_goodput = 100,
        .probe_goodput = 96,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 3 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.pinned_reduce_rollback, decision.action);
    try std.testing.expectEqual(6, decision.knobs.dma_chunks);
    try std.testing.expect(!decision.trim_pinned);
}

test "adaptive load controller pressure rollback cools down resource probes" {
    var controller: AdaptiveLoadController = .init(4, 4, 6, 8, 4, 4, 8, true);
    controller.mode = .steady;

    var decision = controller.observe(.{
        .committed_goodput = 100,
        .now_ns = 2 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.pinned_reduce_start, decision.action);

    const rollback_ns = 2100 * std.time.ns_per_ms;
    decision = controller.observe(.{
        .committed_goodput = 100,
        .pinned_wait_ratio = 0.21,
        .now_ns = rollback_ns,
    });
    try std.testing.expectEqual(.pinned_reduce_rollback, decision.action);
    try std.testing.expectEqual(rollback_ns, controller.last_resource_probe_ns);
    try std.testing.expectEqual(6, decision.knobs.dma_chunks);

    decision = controller.observe(.{
        .committed_goodput = 100,
        .now_ns = 3 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.none, decision.action);
}

test "adaptive load controller resource reductions stay within global best band" {
    var controller: AdaptiveLoadController = .init(4, 4, 7, 8, 4, 4, 8, true);
    controller.mode = .steady;

    var decision = controller.observe(.{
        .committed_goodput = 100,
        .now_ns = 2 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.pinned_reduce_start, decision.action);
    try std.testing.expectEqual(6, decision.knobs.dma_chunks);

    decision = controller.observe(.{
        .committed_goodput = 97,
        .probe_goodput = 97,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 3 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.pinned_reduce_keep, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 97,
        .now_ns = 5 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.pinned_reduce_start, decision.action);
    try std.testing.expectEqual(5, decision.knobs.dma_chunks);

    decision = controller.observe(.{
        .committed_goodput = 94,
        .probe_goodput = 94,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 6 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.pinned_reduce_rollback, decision.action);
    try std.testing.expectEqual(6, decision.knobs.dma_chunks);
}

test "adaptive load controller backs DMA off on pinned pressure" {
    var controller: AdaptiveLoadController = .init(4, 8, 5, 9, 4, 8, 8, true);
    const decision = controller.observe(.{
        .committed_goodput = 100,
        .pinned_wait_ratio = 0.21,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_backoff, decision.action);
    try std.testing.expectEqual(.pinned_wait, decision.reason);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
    try std.testing.expectEqual(5, decision.knobs.dma_chunks);
}

test "adaptive load controller requires persistent H2D queue growth" {
    var controller: AdaptiveLoadController = .init(4, 8, 5, 9, 4, 8, 8, true);
    var decision = controller.observe(.{
        .committed_goodput = 100,
        .h2d_growth_ratio = 0.21,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(4, decision.knobs.dma_workers);

    decision = controller.observe(.{
        .committed_goodput = 100,
        .h2d_growth_ratio = 0.21,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_backoff, decision.action);
    try std.testing.expectEqual(.h2d_pressure, decision.reason);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
}

test "adaptive load controller backs DMA off on latency inflation" {
    var controller: AdaptiveLoadController = .init(4, 8, 5, 9, 4, 8, 8, true);
    _ = controller.observe(.{
        .committed_goodput = 100,
        .transfer_latency_us = 100,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    const decision = controller.observe(.{
        .committed_goodput = 100,
        .transfer_latency_us = 121,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.dma_backoff, decision.action);
    try std.testing.expectEqual(.h2d_pressure, decision.reason);
    try std.testing.expectEqual(2, decision.knobs.dma_workers);
}

test "adaptive load controller ignores undersized DMA latency samples" {
    var controller: AdaptiveLoadController = .init(4, 8, 5, 9, 4, 8, 8, true);
    _ = controller.observe(.{
        .committed_goodput = 100,
        .transfer_latency_us = 100,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    const decision = controller.observe(.{
        .committed_goodput = 100,
        .transfer_latency_us = 1000,
        .dma_latency_reliable = false,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(@as(usize, 4), decision.knobs.dma_workers);
}

test "adaptive load controller does not baseline DMA latency from an undersized transfer" {
    var controller: AdaptiveLoadController = .init(2, 8, 3, 9, 2, 8, 8, true);
    var decision = controller.observe(.{
        .committed_goodput = 1,
        .transfer_latency_us = 100,
        .dma_latency_reliable = false,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);

    decision = controller.observe(.{
        .committed_goodput = 100,
        .transfer_latency_us = 500,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(@as(usize, 2), decision.knobs.dma_workers);
}

test "adaptive load controller gives staged read recovery executable capacity" {
    var controller: AdaptiveLoadController = .init(1, 4, 2, 5, 1, 8, 8, true);
    controller.mode = .steady;
    controller.knobs.staging_chunks = 1;

    const decision = controller.observe(.{
        .read_goodput = 10,
        .committed_goodput = 10,
        .dma_starvation_ratio = 0.50,
        .reads_saturated = true,
        .now_ns = 600 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.staging_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 2), decision.knobs.read_workers);
    try std.testing.expectEqual(@as(usize, 2), decision.knobs.staging_chunks);
}

test "adaptive load controller rolls back an unhelpful pinned probe" {
    var controller: AdaptiveLoadController = .init(2, 2, 3, 5, 2, 2, 0, true);
    var decision = controller.observe(.{
        .committed_goodput = 100,
        .pinned_wait_ratio = 0.06,
        .dma_saturated = true,
        .now_ns = 100 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.pinned_probe_start, decision.action);
    try std.testing.expectEqual(4, decision.knobs.dma_chunks);

    decision = controller.observe(.{
        .committed_goodput = 102,
        .probe_goodput = 102,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 200 * std.time.ns_per_ms,
    });
    try std.testing.expectEqual(.pinned_probe_rollback, decision.action);
    try std.testing.expectEqual(3, decision.knobs.dma_chunks);
    try std.testing.expect(decision.trim_pinned);
}

test "adaptive load controller reduces staging within the best throughput band" {
    var controller: AdaptiveLoadController = .init(2, 2, 3, 3, 2, 4, 4, true);
    controller.mode = .steady;
    controller.knobs.staging_chunks = 2;

    var decision = controller.observe(.{
        .committed_goodput = 100,
        .now_ns = 2 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.staging_reduce_start, decision.action);
    try std.testing.expectEqual(1, decision.knobs.staging_chunks);

    decision = controller.observe(.{
        .committed_goodput = 97,
        .probe_goodput = 97,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 3 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.staging_reduce_keep, decision.action);
    try std.testing.expectEqual(1, decision.knobs.staging_chunks);

    decision = controller.observe(.{
        .committed_goodput = 97,
        .now_ns = 5 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.staging_reduce_start, decision.action);
    try std.testing.expectEqual(0, decision.knobs.staging_chunks);

    decision = controller.observe(.{
        .committed_goodput = 96,
        .probe_goodput = 96,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .now_ns = 6 * std.time.ns_per_s,
    });
    try std.testing.expectEqual(.staging_reduce_rollback, decision.action);
    try std.testing.expectEqual(1, decision.knobs.staging_chunks);
}

test "adaptive load probe metrics reject stale epochs" {
    const io = std.testing.io;
    var metrics: LoadMetrics = .{};

    metrics.publishProbeEpoch(io, 1);
    metrics.recordProbeCommit(io, 1, 32 * 1024 * 1024);
    try std.testing.expectEqual(32 * 1024 * 1024, metrics.probe_committed_bytes.load(.acquire));

    metrics.publishProbeEpoch(io, 2);
    metrics.recordProbeCommit(io, 1, 32 * 1024 * 1024);
    metrics.recordProbeCommit(io, 2, 16 * 1024 * 1024);
    try std.testing.expectEqual(16 * 1024 * 1024, metrics.probe_committed_bytes.load(.acquire));
}

test "adaptive load runtime delays probe epoch until capacity is active" {
    const io = std.testing.io;
    var controller: AdaptiveLoadController = .init(2, 4, 3, 5, 2, 4, 0, false);
    const baseline = controller.knobs;
    var candidate = baseline;
    candidate.read_workers = 4;
    controller.epoch = 1;
    controller.knobs = candidate;
    controller.probe = .{
        .dimension = .read,
        .kind = .increase,
        .baseline = baseline,
        .candidate = candidate,
        .epoch = 1,
        .baseline_goodput = 100,
    };

    var metrics: LoadMetrics = .{};
    var read_group: stdx.Io.LimitedGroup = .init(2);
    var dma_group: stdx.Io.LimitedGroup = .init(2);
    var runtime: AdaptiveLoadRuntime = undefined;
    runtime.controller = controller;
    runtime.pipeline = null;
    runtime.read_group = &read_group;
    runtime.dma_group = &dma_group;
    runtime.metrics = &metrics;
    runtime.pending_probe_activation = null;

    runtime.apply(io, .{
        .knobs = candidate,
        .epoch = 1,
        .changed = true,
        .action = .read_probe_start,
    });

    try std.testing.expectEqual(0, metrics.config_epoch.load(.acquire));
    try std.testing.expectEqual(0, metrics.probe_epoch.load(.acquire));
    try std.testing.expect(runtime.pending_probe_activation != null);
}

test "adaptive load DMA probe capacity requires distinct submitting lanes" {
    const io = std.testing.io;
    const baseline: AdaptiveLoadController.Knobs = .{
        .read_workers = 2,
        .dma_workers = 2,
        .dma_chunks = 3,
        .staging_chunks = 0,
    };
    var candidate = baseline;
    candidate.dma_workers = 4;
    candidate.dma_chunks = 5;

    var pipeline: AdaptivePipelineContext = undefined;
    var lanes: [4]AdaptivePipelineLane = undefined;
    pipeline.lanes = &lanes;
    for (&lanes) |*lane| lane.* = .{ .ctx = &pipeline };

    var runtime: AdaptiveLoadRuntime = undefined;
    runtime.pipeline = &pipeline;
    const activation: AdaptiveLoadRuntime.ProbeActivation = .{
        .epoch = 7,
        .dimension = .dma,
        .kind = .increase,
        .baseline = baseline,
        .candidate = candidate,
        .installed_at = .now(io, .awake),
        .controller_now_ns = 0,
    };

    for (lanes[0..3]) |*lane| lane.last_dma_submission_epoch.store(7, .release);
    try std.testing.expect(!runtime.probeCapacityActive(activation));
    lanes[3].last_dma_submission_epoch.store(7, .release);
    try std.testing.expect(runtime.probeCapacityActive(activation));
}

fn buildMesh2x2(
    allocator: std.mem.Allocator,
    target: @import("platform.zig").Target,
    devices: []const @import("platform.zig").Device,
) !Sharding.PhysicalMesh {
    if (devices.len < 4) return error.NotEnoughDevices;
    const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[0]),
            .device(devices[1]),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[2]),
            .device(devices[3]),
        }),
    });

    return Sharding.PhysicalMesh.fromTree(allocator, target, topology);
}

fn buildMesh2x2x2(
    allocator: std.mem.Allocator,
    target: @import("platform.zig").Target,
    devices: []const @import("platform.zig").Device,
) !Sharding.PhysicalMesh {
    if (devices.len < 8) return error.NotEnoughDevices;
    const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[0]),
                .device(devices[1]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[2]),
                .device(devices[3]),
            }),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[4]),
                .device(devices[5]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[6]),
                .device(devices[7]),
            }),
        }),
    });

    return Sharding.PhysicalMesh.fromTree(allocator, target, topology);
}

const DirectMemoryWriterDeviceTest = struct {
    const WriteMode = enum {
        stream_remaining,
        writable_slice_greedy,
        park_between_windows,
        staged_then_direct,
    };

    pub const Scenario = struct {
        name: []const u8,
        create_options: CreateOptions,
        shape: Shape,
        logical_mesh: Sharding.LogicalMesh,
        strategy: Sharding.Strategy,
        write_mode: WriteMode = .stream_remaining,
        writable_slice_min_len: usize = 128,
        pool_chunks: usize = 4,
        pool_chunk_size: usize = 1 << 20,
    };

    allocator: std.mem.Allocator,
    io: std.Io,

    fn run(self: DirectMemoryWriterDeviceTest, scenario: Scenario) !void {
        var platform = Platform.auto(self.allocator, self.io, scenario.create_options) catch return error.SkipZigTest;
        defer platform.deinit(self.allocator, self.io);

        const sharding: Sharding.Data = try .init(scenario.name, &platform.physical_mesh, scenario.logical_mesh, scenario.strategy);
        try self.runDirectMemoryWriter(
            platform,
            scenario.shape,
            .{ .data = &sharding },
            scenario.write_mode,
            scenario.writable_slice_min_len,
            scenario.pool_chunks,
            scenario.pool_chunk_size,
        );
    }

    fn runDirectMemoryWriter(
        self: DirectMemoryWriterDeviceTest,
        platform: *const Platform,
        shape: Shape,
        sharding: Sharding,
        write_mode: WriteMode,
        writable_slice_min_len: usize,
        pool_chunks: usize,
        pool_chunk_size: usize,
    ) !void {
        const slice = try Slice.alloc(self.allocator, shape);
        defer slice.free(self.allocator);

        for (slice.items(f32), 0..) |*e, i| {
            e.* = @as(f32, @floatFromInt(i));
        }

        const pool_count = platform.devices.len;
        const dma_allocators = try self.allocator.alloc(mem.DmaAllocator, pool_count);
        defer self.allocator.free(dma_allocators);
        for (platform.devices, 0..) |*device, i| {
            dma_allocators[i] = .init(self.allocator, device);
        }

        const pools = try self.allocator.alloc(mem.DynamicBufferPool, pool_count);
        defer self.allocator.free(pools);
        for (pools) |*pool| {
            pool.* = .init(pool_chunks, pool_chunk_size);
        }
        defer for (pools, 0..) |*pool, i| {
            pool.deinit(dma_allocators[i].allocator());
        };

        var written_buffer: Buffer = undefined;
        var writer: DirectMemoryWriter = try .init(
            self.allocator,
            self.io,
            platform,
            pools,
            dma_allocators,
            pool_chunk_size,
            shape,
            sharding,
            &written_buffer,
        );
        defer writer.deinit();
        defer written_buffer.deinit();

        switch (write_mode) {
            .stream_remaining => {
                var reader: std.Io.Reader = .fixed(slice.constData());
                const streamed = try reader.streamRemaining(&writer.interface);
                try std.testing.expectEqual(slice.constData().len, streamed);
            },
            .writable_slice_greedy => {
                var offset: usize = 0;
                while (offset < slice.constData().len) {
                    const min_len = @max(@as(usize, 1), writable_slice_min_len);
                    const dest = try writer.interface.writableSliceGreedy(min_len);
                    const to_write = @min(dest.len, slice.constData().len - offset);
                    if (to_write == 0) return std.Io.Writer.Error.WriteFailed;
                    @memcpy(dest[0..to_write], slice.constData()[offset..][0..to_write]);
                    writer.interface.advance(to_write);
                    offset += to_write;
                }
            },
            .park_between_windows => {
                var offset: usize = 0;
                while (offset < slice.constData().len) {
                    const dest = try writer.interface.writableSliceGreedy(1);
                    const to_write = @min(@as(usize, 257), @min(dest.len, slice.constData().len - offset));
                    if (to_write == 0) return std.Io.Writer.Error.WriteFailed;
                    @memcpy(dest[0..to_write], slice.constData()[offset..][0..to_write]);
                    writer.interface.advance(to_write);
                    offset += to_write;
                    if (offset < slice.constData().len) {
                        try writer.park();
                        try writer.unpark();
                    }
                }
            },
            .staged_then_direct => {
                var offset: usize = 0;
                while (offset < slice.constData().len) {
                    const staged_len = @min(
                        writer.interface.buffer.len - writer.interface.end,
                        slice.constData().len - offset,
                    );
                    if (staged_len == 0) return std.Io.Writer.Error.WriteFailed;
                    try writer.interface.writeAll(slice.constData()[offset..][0..staged_len]);
                    offset += staged_len;
                    try writer.commitStagedWrite();
                    if (offset == slice.constData().len) break;

                    const direct = writer.directWritableSlice();
                    if (direct.len == 0) return std.Io.Writer.Error.WriteFailed;
                    const direct_len = @min(@as(usize, 257), @min(direct.len, slice.constData().len - offset));
                    @memcpy(direct[0..direct_len], slice.constData()[offset..][0..direct_len]);
                    try writer.commitDirectRead(direct_len);
                    offset += direct_len;
                }
            },
        }

        try writer.interface.flush();
        try written_buffer.await(self.io);

        var written_slice = try written_buffer.toSliceAlloc(self.allocator, self.io);
        defer written_slice.free(self.allocator);
        try std.testing.expectEqualSlices(u8, slice.constData(), written_slice.constData());
    }
};

test "DirectMemoryWriter: replicated with auto topology" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "replicated_auto",
        .create_options = .{
            .physical_mesh = .auto,
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .rows = 8, .cols = 128 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .replicated }),
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
    });
}

test "DirectMemoryWriter: 1D model split with 2x2 physical mesh" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "model_auto",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2 },
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .rows = 8, .cols = 1024 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .model }),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .model = .link_x }),
    });
}

test "DirectMemoryWriter: 2D batch/model split with 2x2 physical mesh" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "batch_model_2d_torus",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2 },
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .batch = 8, .model = 1024 }, .f32)
            .withPartitioning(.{ .batch = .batch, .model = .model }),
        .logical_mesh = .mesh(.{
            .batch = .low_bandwidth,
            .model = .high_bandwidth,
        }),
        .strategy = .parseBindings(.{ .batch = .link_x, .model = .link_y }),
    });
}

test "DirectMemoryWriter: folded model sharding with 2x2 physical mesh" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "model_folded_2d_torus",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2 },
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .model = 4096 }, .f32).withPartitioning(.{ .model = .model }),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_y });
            break :blk strategy;
        },
    });
}

test "DirectMemoryWriter: writableSliceGreedy with mirrored shards" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "model_auto_writable_slice",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2 },
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .rows = 8, .cols = 1024 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .model }),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .model = .link_x }),
        .write_mode = .writable_slice_greedy,
        .writable_slice_min_len = 64,
        .pool_chunk_size = 1024,
    });
}

test "DirectMemoryWriter: park and unpark pinned windows" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "park_unpark",
        .create_options = .{
            .physical_mesh = .auto,
            .cpu = .{ .device_count = 2 },
        },
        .shape = Shape.init(.{ .rows = 8, .cols = 1024 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .replicated }),
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
        .write_mode = .park_between_windows,
        .pool_chunk_size = 1024,
    });
}

test "DirectMemoryWriter: staged window followed by direct read" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "staged_then_direct",
        .create_options = .{
            .physical_mesh = .auto,
            .cpu = .{ .device_count = 2 },
        },
        .shape = Shape.init(.{ .rows = 8, .cols = 1024 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .replicated }),
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
        .write_mode = .staged_then_direct,
        .pool_chunk_size = 1024,
    });
}

test "DirectMemoryWriter: staged window followed by direct read across shards" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "staged_then_direct_sharded",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2 },
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .rows = 8, .cols = 1024 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .model }),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .model = .link_x }),
        .write_mode = .staged_then_direct,
        .pool_chunk_size = 1024,
    });
}

test "DirectMemoryWriter: 3D topology folded model + replicated batch" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "topology_3d_folded_model",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2x2 },
            .cpu = .{ .device_count = 8 },
        },
        .shape = Shape.init(.{ .batch = 16, .model = 4096 }, .f32)
            .withPartitioning(.{ .batch = .replicated, .model = .model }),
        .logical_mesh = .mesh(.{
            .batch = .low_bandwidth,
            .model = .high_bandwidth,
        }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_z });
            break :blk strategy;
        },
    });
}
