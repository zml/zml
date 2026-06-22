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

pub const TensorStore = struct {
    registry: *safetensors.TensorRegistry,
    id_to_binding: std.AutoHashMapUnmanaged(usize, Binding),
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,

    pub fn fromRegistry(allocator: std.mem.Allocator, registry: *safetensors.TensorRegistry) TensorStore {
        const arena: std.heap.ArenaAllocator = .init(allocator);
        return .{
            .registry = registry,
            .id_to_binding = .empty,
            .allocator = allocator,
            .arena = arena,
        };
    }

    pub fn deinit(self: *TensorStore) void {
        self.id_to_binding.deinit(self.allocator);
        self.arena.deinit();
    }

    fn putBindingNoClobber(self: *TensorStore, id: usize, binding: Binding) std.mem.Allocator.Error!void {
        const gop = try self.id_to_binding.getOrPut(self.allocator, id);
        if (gop.found_existing) {
            stdx.debug.panic("Id {} already has an associated binding", .{id});
        }
        errdefer self.id_to_binding.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = binding;
    }

    fn getPtrFromKey(self: *const TensorStore, key: []const u8) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
        return tensor_desc_ptr;
    }

    fn getPtrFromId(self: *const TensorStore, id: usize) ?*safetensors.Tensor {
        const binding = self.id_to_binding.get(id) orelse return null;
        stdx.debug.assert(binding == .direct, "Expect binding to be .direct for id {}, got {}", .{ id, @as(Binding.Type, binding) });
        return binding.direct;
    }

    pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        return self.registry.reader(io, key, buffer);
    }

    pub fn getReaderById(self: *const TensorStore, id: usize, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        const binding = self.id_to_binding.get(id) orelse return error.NotFound;
        stdx.debug.assert(binding == .direct, "Expect binding to be .direct for id {}, got {}", .{ id, @as(Binding.Type, binding) });

        return binding.direct.reader(io, buffer, .{});
    }

    pub fn getBindingById(self: *const TensorStore, id: usize) ?Binding {
        return self.id_to_binding.get(id);
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
            return self.maybeCreateBinding(.{ .direct = subkey }, tagz, partitioning);
        }

        pub fn createTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) Tensor {
            return self.maybeCreateTensor(subkey, tagz, partitioning).?;
        }

        fn applyTags(shape_: Shape, tagz: anytype) Shape {
            var shape = shape_;
            if (@TypeOf(tagz) != @TypeOf(null)) {
                switch (@typeInfo(@TypeOf(tagz))) {
                    .optional => if (tagz) |t| {
                        shape = shape.withTags(t);
                    },
                    else => shape = shape.withTags(tagz),
                }
            }
            return shape;
        }

        fn applyPartitioning(shape_: Shape, partitioning: anytype) Shape {
            var shape = shape_;

            if (@TypeOf(partitioning) == @TypeOf(null)) {
                @compileError("TensorStore.View.createTensor partitioning cannot be null; pass .replicated or an explicit partitioning");
            }

            switch (@typeInfo(@TypeOf(partitioning))) {
                .optional => @compileError("TensorStore.View.createTensor partitioning cannot be optional; pass .replicated or an explicit partitioning"),
                .enum_literal => switch (partitioning) {
                    .replicated => shape = shape.withReplicatedPartitioning(),
                    else => @compileError("Only .replicated is supported as a standalone partitioning enum literal"),
                },
                else => shape = shape.withPartitioning(partitioning),
            }

            return shape;
        }

        pub fn maybeCreateBinding(self: View, request: Binding.Request, tagz: anytype, partitioning: anytype) ?Tensor {
            return switch (request) {
                .direct => |direct| b: {
                    var buffer: [256]u8 = undefined;
                    const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", direct }) catch unreachable;

                    const ptr = self.store.getPtrFromKey(key) orelse return null;

                    var result_shape = ptr.shape;
                    result_shape = applyTags(result_shape, tagz);
                    result_shape = applyPartitioning(result_shape, partitioning);

                    const tensor: Tensor = .fromShape(result_shape);
                    self.store.putBindingNoClobber(tensor.id, .{ .direct = ptr }) catch unreachable;

                    break :b tensor;
                },
                .concatenate => |concatenate| b: {
                    var buffer: [256]u8 = undefined;
                    const arena = self.store.arena.allocator();

                    const tensors = arena.alloc(*safetensors.Tensor, concatenate.keys.len) catch unreachable;
                    errdefer arena.free(tensors);

                    for (concatenate.keys, 0..) |subkey, i| {
                        const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
                        tensors[i] = self.store.getPtrFromKey(key) orelse return null;
                    }

                    var result_shape = tensors[0].shape;
                    for (tensors[1..]) |tensor| {
                        result_shape = result_shape.setDim(concatenate.axis, result_shape.dim(concatenate.axis) + tensor.shape.dim(concatenate.axis));
                    }

                    result_shape = applyTags(result_shape, tagz);
                    result_shape = applyPartitioning(result_shape, partitioning);

                    const tensor: Tensor = .fromShape(result_shape);

                    self.store.putBindingNoClobber(tensor.id, .{ .concatenate = .{ .tensors = tensors, .axis = concatenate.axis } }) catch unreachable;

                    break :b tensor;
                },
                .custom => |custom| b: {
                    var buffer: [256]u8 = undefined;
                    const arena = self.store.arena.allocator();

                    const tensors = arena.alloc(*safetensors.Tensor, custom.keys.len) catch unreachable;
                    errdefer arena.free(tensors);

                    for (custom.keys, 0..) |subkey, i| {
                        const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
                        tensors[i] = self.store.getPtrFromKey(key) orelse return null;
                    }

                    const shapes = arena.alloc(Shape, custom.keys.len) catch unreachable;
                    errdefer arena.free(shapes);
                    for (tensors, shapes) |t, *s| {
                        s.* = t.shape;
                    }

                    var result_shape = custom.computeShapeCallback(shapes);

                    result_shape = applyTags(result_shape, tagz);
                    result_shape = applyPartitioning(result_shape, partitioning);

                    const tensor: Tensor = .fromShape(result_shape);

                    self.store.putBindingNoClobber(tensor.id, .{ .custom = .{ .tensors = tensors } }) catch unreachable;

                    break :b tensor;
                },
            };
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

/// Binding represents a mapping from a tensor to the sources that are required to construct it from the store.
pub const Binding = union(Binding.Type) {
    pub const Type = enum {
        /// Direct mapping to a tensor in the store.
        direct,
        /// Concatenation of multiple tensors along a specified axis.
        concatenate,
        /// Custom mapping that allows for arbitrary tensor construction logic.
        custom,
    };

    pub const Request = union(Type) {
        direct: []const u8,
        concatenate: struct {
            keys: []const []const u8,
            axis: i64,
        },
        custom: struct {
            keys: []const []const u8,
            computeShapeCallback: *const fn (shapes: []const Shape) Shape,
        },
    };

    direct: *const safetensors.Tensor,
    concatenate: struct {
        tensors: []const *safetensors.Tensor,
        axis: i64,
    },
    custom: struct {
        tensors: []const *safetensors.Tensor,
    },
};

pub const Loader = struct {
    allocator: std.mem.Allocator,
    platform: *const Platform,
    dma_allocators: []const mem.DmaAllocator,
    dma_chunk_size: usize,
    pinned_buffer_pools: []mem.DynamicBufferPool,
    group: stdx.Io.LimitedGroup,
    bytes_loaded: std.atomic.Value(usize) = .init(0),

    pub const Opts = struct {
        pub const default: Opts = .{
            .parallelism = 1,
            .dma_chunks = 2,
            .dma_chunk_size = 4096,
        };
        parallelism: usize,
        dma_chunks: usize,
        dma_chunk_size: usize,
    };

    pub fn init(allocator: std.mem.Allocator, platform: *const Platform, opts: Opts) !Loader {
        const pool_count = platform.devices.len;
        const dma_allocators = try allocator.alloc(mem.DmaAllocator, pool_count);
        errdefer allocator.free(dma_allocators);
        for (platform.devices, 0..) |*device, i| {
            dma_allocators[i] = .init(allocator, device);
        }

        const buffer_pools = try allocator.alloc(mem.DynamicBufferPool, pool_count);
        errdefer allocator.free(buffer_pools);
        for (buffer_pools) |*pool_| {
            pool_.* = .init(opts.dma_chunks, opts.dma_chunk_size);
        }
        errdefer for (buffer_pools, 0..) |*pool_, i| {
            pool_.deinit(dma_allocators[i].allocator());
        };

        return .{
            .allocator = allocator,
            .platform = platform,
            .dma_allocators = dma_allocators,
            .dma_chunk_size = opts.dma_chunk_size,
            .pinned_buffer_pools = buffer_pools,
            .group = .init(opts.parallelism),
        };
    }

    pub fn deinit(self: Loader) void {
        for (self.pinned_buffer_pools, 0..) |*pool, i| pool.deinit(self.dma_allocators[i].allocator());
        self.allocator.free(self.pinned_buffer_pools);
        self.allocator.free(self.dma_allocators);
    }

    pub fn await(self: *Loader, io: std.Io) std.Io.Cancelable!void {
        return self.group.await(io);
    }

    pub const AutoOpts = struct {
        progress: ?*std.Progress.Node = null,
    };

    pub fn load(self: *Loader, io: std.Io, comptime T: type, model: *const T, buffers: *Bufferized(T), store: *const TensorStore, shardings: []const Sharding, opts: AutoOpts) void {
        self.group.async(io, struct {
            fn call(self_: *Loader, io_: std.Io, model_: *const T, buffers_: *Bufferized(T), store_: *const TensorStore, shardings_: []const Sharding, opts_: AutoOpts) void {
                self_.loadInner(io_, T, model_, buffers_, store_, shardings_, opts_) catch unreachable;
            }
        }.call, .{ self, io, model, buffers, store, shardings, opts });
    }

    fn loadInner(self: *Loader, io: std.Io, comptime T: type, model: *const T, buffers: *Bufferized(T), store: *const TensorStore, shardings: []const Sharding, opts: AutoOpts) !void {
        const tensor_count = meta.count(Tensor, model);

        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        defer arena.deinit();

        const flattened_buffers = try arena.allocator().alloc(*Buffer, tensor_count);
        meta.forEachVisit(buffers, *Buffer, struct {
            fn call(i: usize, buffer: *Buffer, flattened_buffers_: []*Buffer) void {
                flattened_buffers_[i] = buffer;
            }
        }.call, .{flattened_buffers});

        const Ctx = struct {
            self: *Loader,
            io: std.Io,
            store: *const TensorStore,
            shardings: []const Sharding,
            buffers: []*Buffer,
            opts: AutoOpts,
        };

        var ctx: Ctx = .{
            .self = self,
            .io = io,
            .store = store,
            .shardings = shardings,
            .buffers = flattened_buffers,
            .opts = opts,
        };

        meta.forEachVisit(model, *const Tensor, struct {
            fn call(i: usize, tensor: *const Tensor, ctx_: *Ctx) void {
                ctx_.self.group.async(ctx_.io, defaultCallback, .{ ctx_.self, ctx_.io, tensor, ctx_.buffers[i], ctx_.store, ctx_.shardings, ctx_.opts });
            }
        }.call, .{&ctx});
    }

    fn defaultCallback(self: *Loader, io: std.Io, tensor: *const Tensor, buffer: *Buffer, store: *const TensorStore, shardings: []const Sharding, opts: AutoOpts) void {
        const binding = store.getBindingById(tensor.id) orelse {
            std.log.warn("Failed to get binding for tensor with id: {}", .{tensor.id});
            return;
        };

        switch (binding) {
            .direct => |direct| {
                var reader = direct.reader(io, &.{}, .{}) catch unreachable;
                defer reader.deinit();

                const shape = tensor.shape();
                const sharding = Sharding.pickSharding(shardings, shape, .explicit_axis_binding) orelse blk: {
                    log.debug("No sharding strategy found for tensor {s} with shape {f}, using replicated sharding", .{ reader.tensor.name, shape });
                    break :blk self.platform.replicated_sharding;
                };

                var writer = MemoryWriter.init(
                    self.allocator,
                    io,
                    self.platform,
                    self.pinned_buffer_pools,
                    self.dma_allocators,
                    self.dma_chunk_size,
                    shape,
                    sharding,
                    buffer,
                ) catch unreachable;
                defer writer.deinit(self.allocator);

                const scale = 1024;

                if (opts.progress) |progress| {
                    var node = progress.start(reader.tensor.name, reader.tensor.shape.byteSize() / scale);
                    defer node.end();
                    writer.setProgress(&node);
                    defer writer.setProgress(null);
                    var progress_writer: ProgressWriter = .init(writer.interface(), &node, .{ .scale = scale });
                    const total = reader.interface.streamRemaining(&progress_writer.interface) catch unreachable;
                    progress_writer.interface.flush() catch unreachable;
                    _ = self.bytes_loaded.fetchAdd(total, .monotonic);
                } else {
                    const total = reader.interface.streamRemaining(writer.interface()) catch unreachable;
                    writer.interface().flush() catch unreachable;
                    _ = self.bytes_loaded.fetchAdd(total, .monotonic);
                }
            },
            .concatenate => |concatenate| {
                if (concatenate.axis < 0) @panic("Negative axis are not supported yet");

                if (concatenate.axis == 0) {
                    const shape = tensor.shape();
                    const sharding = Sharding.pickSharding(shardings, shape, .explicit_axis_binding) orelse blk: {
                        log.debug("No sharding strategy found for tensor with shape {f}, using replicated sharding", .{shape});
                        break :blk self.platform.replicated_sharding;
                    };

                    var writer = MemoryWriter.init(
                        self.allocator,
                        io,
                        self.platform,
                        self.pinned_buffer_pools,
                        self.dma_allocators,
                        self.dma_chunk_size,
                        shape,
                        sharding,
                        buffer,
                    ) catch unreachable;
                    defer writer.deinit(self.allocator);

                    const scale = 1024;

                    const Formatter = struct {
                        tensors: []const *safetensors.Tensor,
                        pub fn format(
                            self_: @This(),
                            writer_: *std.Io.Writer,
                        ) std.Io.Writer.Error!void {
                            for (self_.tensors, 0..) |tensor_, i| {
                                if (i > 0) try writer_.writeAll(", ");
                                try writer_.writeAll(tensor_.name);
                            }
                        }
                    };

                    if (opts.progress) |progress| {
                        var node = progress.startFmt(tensor.shape().byteSize() / scale, "Concat {f}", .{Formatter{ .tensors = concatenate.tensors }});
                        defer node.end();
                        writer.setProgress(&node);
                        defer writer.setProgress(null);
                        var progress_writer: ProgressWriter = .init(writer.interface(), &node, .{ .scale = scale });

                        var total: usize = 0;
                        for (concatenate.tensors) |t| {
                            var reader = t.reader(io, &.{}, .{}) catch unreachable;
                            defer reader.deinit();
                            total += reader.interface.streamRemaining(writer.interface()) catch unreachable;
                        }
                        progress_writer.interface.flush() catch unreachable;
                        _ = self.bytes_loaded.fetchAdd(total, .monotonic);
                    } else {
                        var total: usize = 0;
                        for (concatenate.tensors) |t| {
                            var reader = t.reader(io, &.{}, .{}) catch unreachable;
                            defer reader.deinit();
                            total += reader.interface.streamRemaining(writer.interface()) catch unreachable;
                        }
                        writer.interface().flush() catch unreachable;
                        _ = self.bytes_loaded.fetchAdd(total, .monotonic);
                    }
                } else @panic("Non-major concatenation is not supported yet");
            },
            .custom => @panic("Custom binding can't be loaded automatically"),
        }
    }
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
        return switch (platform.target) {
            .cuda, .oneapi => .{ .direct = try DirectMemoryWriter.init(allocator, io, platform, pools, dma_allocators, dma_chunk_size, shape, sharding, buffer) },
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
    flip_flop: u1 = 0,
    events_contexts: [2]?EventContext = @splat(null),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, memory: *const Memory, pool: *mem.DynamicBufferPool, shape: Shape) !DirectShardWriter {
        const shape_spec: pjrt.ShapeSpec = .init(shape.dims(), pjrtx.bufferTypeFromDtype(shape.dtype()));
        const transfer_manager = try memory.platform.pjrt_client.createBuffersForAsyncHostToDevice(
            memory.platform.pjrt_api,
            .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            },
        );

        const pjrt_buffer = transfer_manager.retrieveBuffer(memory.platform.pjrt_api, 0) catch unreachable;

        const buf = try pool.get(allocator, io);

        return .{
            .allocator = allocator,
            .io = io,
            .memory = memory,
            .pool = pool,
            .total = shape.byteSize(),
            .pjrt_buffer = pjrt_buffer,
            .transfer_manager = transfer_manager,
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
        self.transfer_manager.deinit(self.memory.platform.pjrt_api);
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

        const transfer_event = self.transfer_manager.transferData(pjrt_api, 0, slice, @intCast(self.offset), is_last) catch |err| {
            log.err("error when transferring data to device: {any}", .{err});
            return std.Io.Writer.Error.WriteFailed;
        };

        const ctx = &self.events_contexts[@intCast(self.flip_flop)];
        ctx.* = .{
            .self = self,
            .buffer = current_buffer,
            .pjrt_event = transfer_event,
        };

        transfer_event.onReady(pjrt_api, EventContext, struct {
            fn call(err: ?*pjrt.Error, ctx_: *EventContext) void {
                ctx_.self.pool.put(ctx_.self.io, ctx_.buffer);
                ctx_.err = err;
                ctx_.event.set(ctx_.self.io);
            }
        }.call, &(ctx.*.?)) catch |err| {
            log.err("error when setting up transfer completion callback: {any}", .{err});
            return std.Io.Writer.Error.WriteFailed;
        };

        if (self.events_contexts[@intCast(self.flip_flop ^ 1)]) |*ctx_previous| {
            defer self.events_contexts[@intCast(self.flip_flop ^ 1)] = null;
            ctx_previous.event.waitUncancelable(self.io);
            defer ctx_previous.pjrt_event.deinit(pjrt_api);
            if (ctx_previous.err) |e| {
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
                defer e.deinit(pjrt_api);
                log.err("error while awaiting: {s}: {s}", .{
                    @tagName(e.getCode(pjrt_api)),
                    e.getMessage(pjrt_api),
                });
                return std.Io.Writer.Error.WriteFailed;
            }
        } else {
            self.interface.end = 0;
            const buf = self.pool.get(self.allocator, self.io) catch |err| {
                log.err("unable to get a new buffer from the pool: {any}", .{err});
                return std.Io.Writer.Error.WriteFailed;
            };
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
    allocator: std.mem.Allocator,
    shard_writers: []DirectShardWriter,
    // Global stream-order spans produced from placement; this is the routing table for committed bytes.
    dispatch_spans: DispatchSpans,
    // Current entry in `dispatch_spans.spans`; advances whenever `byte_cursor` reaches that span end.
    span_index: usize = 0,
    // Global tensor byte offset already scattered into shard writers.
    byte_cursor: usize = 0,
    // Shard writer whose DMA buffer is currently exposed as `interface.buffer`.
    active_writer_index: usize,
    // Start offset of new reader bytes inside the active shard writer buffer.
    window_start: usize,
    // Logical maximum public alias window before forcing a cross-shard flush fence.
    dma_chunk_size: usize,
    shard_progress: ?[]ShardProgress = null,
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

            shard_writers[i] = try .init(shard_dma_allocator, io, pjrt_mem, pool, placement.shape);

            pjrt_buffers.appendAssumeCapacity(shard_writers[i].pjrt_buffer);
        }

        buffer.* = .fromPjrtBuffers(platform, shape, sharding, pjrt_buffers.constSlice());

        const dispatch_spans: DispatchSpans = try .init(allocator, shape, sharding);
        errdefer dispatch_spans.deinit(allocator);

        const first_span = dispatch_spans.spans[0];
        const first_writer = &shard_writers[first_span.primary_writer];
        const first_window = @min(dma_chunk_size, first_writer.interface.buffer.len);

        return .{
            .allocator = allocator,
            .shard_writers = shard_writers,
            .dispatch_spans = dispatch_spans,
            .active_writer_index = first_span.primary_writer,
            .window_start = first_writer.interface.end,
            .dma_chunk_size = dma_chunk_size,
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
        }

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
        const tensor_remaining = total - self.byte_cursor;
        const visible_remaining = @min(buffer_remaining, @min(chunk_remaining, tensor_remaining));
        if (visible_remaining == 0) return std.Io.Writer.Error.WriteFailed;

        self.interface.buffer = next_writer.interface.buffer[0 .. self.window_start + visible_remaining];
        self.interface.end = self.window_start;
    }
};

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
