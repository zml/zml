const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const zml = @import("../zml.zig");

const log = std.log.scoped(.@"zml/attnd");

var context_type_id: ?zml.pjrt.ffi.TypeId = null;

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    // One client per device.
    clients: []Client,
};

pub const Config = struct {
    desctination: std.Io.net.IpAddress,
    client_options: Options = .{},
};

pub const Parameters = struct {
    model_id: ModelId,
    head_dim: u32,
    head_bytes: u32,
    num_kv_heads: u8,
    num_q_per_head: u8,
    num_q_per_packet: u8,

    pub const Init = struct {
        model_id: ModelId,
        head_dim: u32,
        num_attention_heads: u32,
        num_kv_heads: u8,
        mtu: u32,
    };

    pub fn init(args: Init) Parameters {
        const num_q_per_head: u8 = @intCast(@divExact(args.num_attention_heads, args.num_kv_heads));
        const head_bytes = args.head_dim * @sizeOf(u16); // TODO: Here we assume bf16 or equivalent.
        const kv_bytes = 2 * head_bytes;

        var max_q_per_packet: u8 = @intCast(std.math.divFloor(u32, args.mtu - @sizeOf(Header) - kv_bytes, head_bytes) catch unreachable);
        max_q_per_packet = @min(max_q_per_packet, num_q_per_head);

        const num_packets_per_head: u8 = @intCast(std.math.divCeil(u32, num_q_per_head, max_q_per_packet) catch unreachable);
        const num_q_per_packet = @divExact(num_q_per_head, num_packets_per_head);
        if (num_q_per_packet < num_q_per_head) {
            log.warn("Will split attnd requests in {d} packets of {d} queries", .{ num_packets_per_head, max_q_per_packet });
        }

        return .{
            .model_id = args.model_id,
            .head_dim = args.head_dim,
            .head_bytes = head_bytes,
            .num_kv_heads = args.num_kv_heads,
            .num_q_per_head = num_q_per_head,
            .num_q_per_packet = num_q_per_packet,
        };
    }
};

pub const Metadata = struct {
    conversation_id: zml.Tensor,
    layer_id: zml.Tensor,
    num_tokens: zml.Tensor,

    pub fn init() Metadata {
        return .{
            .conversation_id = .fromShape(.scalar(.u64)),
            .layer_id = .fromShape(.scalar(.u16)),
            .num_tokens = .fromShape(.scalar(.u32)),
        };
    }

    pub fn initBuffer(
        self: Metadata,
        io: std.Io,
        platform: *const zml.Platform,
        sharding: zml.sharding.Sharding,
    ) !zml.Bufferized(Metadata) {
        return .{
            .conversation_id = try zml.Buffer.scalar(io, platform, 749, .u64, sharding),
            .layer_id = try zml.Buffer.uninitialized(io, platform, self.layer_id.shape(), sharding, .{}),
            .num_tokens = try zml.Buffer.uninitialized(io, platform, self.num_tokens.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        self.conversation_id.deinit();
        self.layer_id.deinit();
        self.num_tokens.deinit();
    }
};

pub fn register(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, config: Config) !void {
    var ctx = try allocator.create(Context);
    errdefer allocator.destroy(ctx);

    ctx.allocator = allocator;
    ctx.io = io;
    ctx.clients = try allocator.alloc(Client, platform.devices.len);
    errdefer allocator.free(ctx.clients);

    var i: usize = 0;
    errdefer for (ctx.clients[0..i]) |*c| c.deinit(io);
    for (ctx.clients) |*c| {
        c.* = try Client.init(io, config.desctination, config.client_options);
        i += 1;
    }

    if (context_type_id) |_| {
        @panic("Cannot configure the attnd context twice.");
    }

    context_type_id = try platform.registerData("attnd", ctx, &.{
        .deleter = deleter,
    });

    switch (platform.target) {
        .cpu => {
            try targets.cpu.register(platform);
        },
        else => @panic("Not supported yet"),
    }
}

fn deleter(object: ?*anyopaque) callconv(.c) void {
    const ptr = object.?;
    const ctx: *Context = @ptrCast(@alignCast(ptr));
    for (ctx.clients) |*c| {
        c.deinit(ctx.io);
    }
    ctx.allocator.free(ctx.clients);
}

pub fn causalAttention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_offset: zml.Tensor, metadata: Metadata, parameters: Parameters) zml.Tensor {
    // Here is doesn't matter which target we call, they all have the same input/output/parameters
    // zml.Tensor.print(metadata.layer_id, "layer_id");
    //
    const ctx = zml.module.CompilationContext.current();
    const num_partitions = ctx.partitioning.numPartitionsForLogicalAxis(q.shape(), .model) catch unreachable;

    const out = targets.cpu.call(
        .{
            .q = q,
            .k = k,
            .v = v,
            .conversation_id = metadata.conversation_id,
            .layer_id = metadata.layer_id,
            .token_offset = token_offset,
            .num_tokens = metadata.num_tokens,
        },
        .{
            .attn = q.shape(),
        },
        .{
            .model_id = parameters.model_id,
            .head_dim = parameters.head_dim,
            .head_size_in_bytes = parameters.head_bytes,
            .num_kv_heads = parameters.num_kv_heads,
            .num_q_per_head = parameters.num_q_per_head,
            .num_q_per_packet = parameters.num_q_per_packet,
            .num_partitions = @intCast(num_partitions),
        },
    );
    return out.attn;
}

pub const targets = struct {
    pub const cpu = zml.ops.CustomCall(Input, Output, Attributes, cpuCall, .{
        .name = "attnd",
        .sharding_aware = true,
        .has_side_effect = false,
        .output_operand_aliases = .{ .attn = null },
    });
};

const Input = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    conversation_id: zml.Tensor,
    layer_id: zml.Tensor,
    token_offset: zml.Tensor,
    num_tokens: zml.Tensor,
};

const Output = struct {
    attn: zml.Shape,
};

const Attributes = struct {
    model_id: ModelId,
    head_dim: u32,
    head_size_in_bytes: u32,
    num_kv_heads: u8,
    num_q_per_head: u8,
    num_q_per_packet: u8,
    num_partitions: u32,
};

fn cpuCall(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    attrs: Attributes,
) !?*zml.pjrt.ffi.Error {
    const ctx: *Context = @ptrCast(@alignCast(try call_frame.ctx.getContext(context_type_id.?, call_frame.api)));
    const device_ordinal: u64 = @intCast(try call_frame.ctx.getDeviceOrdinal(call_frame.api));

    var buffer: []u8 = try ctx.allocator.alloc(u8, @sizeOf(Header) + (2 + attrs.num_q_per_head) * attrs.head_size_in_bytes);
    defer ctx.allocator.free(buffer);

    const client = &ctx.clients[device_ordinal];

    const req: Request = b: {
        const header: *align(1) Header = std.mem.bytesAsValue(Header, buffer[0..@sizeOf(Header)]);
        const payload: []u8 = buffer[@sizeOf(Header)..buffer.len];
        const q: []u8 = payload[0 .. attrs.num_q_per_packet * attrs.head_size_in_bytes];
        const k: []u8 = payload[q.len..(q.len + attrs.head_size_in_bytes)];
        const v: []u8 = payload[(q.len + attrs.head_size_in_bytes)..(q.len + 2 * attrs.head_size_in_bytes)];
        break :b .{
            .bytes = buffer,
            .header = header,
            .q = q,
            .k = k,
            .v = v,
        };
    };

    const q_size_in_bytes = attrs.num_q_per_head * attrs.head_size_in_bytes;
    const q_ = asConstSlice(input.q);
    const k_ = asConstSlice(input.k);
    const v_ = asConstSlice(input.v);
    const attn_ = asSlice(output.attn);
    const head_size_in_bytes = attrs.head_size_in_bytes;
    const kv_heads_per_partitions = @divExact(attrs.num_kv_heads, attrs.num_partitions);
    const kv_head_offset = kv_heads_per_partitions * device_ordinal;
    const token_offset = asScalar(u32, input.token_offset);
    const num_tokens = asScalar(u32, input.num_tokens);

    for (0..num_tokens) |t| {
        req.header.* = .{
            .type = .attn,
            .kv_head_id = undefined, // set in the for loop
            .first_q_id = undefined, // set in the inner while loop
            .num_queries = undefined, // set in the inner while loop
            .model_id = attrs.model_id,
            .conversation_id = asScalar(u64, input.conversation_id),
            .layer_id = asScalar(u16, input.layer_id),
            .token_pos = token_offset + @as(u32, @intCast(t)),
        };

        const q = q_[t * kv_heads_per_partitions * q_size_in_bytes ..][0 .. kv_heads_per_partitions * q_size_in_bytes];
        const k = k_[(token_offset + t) * kv_heads_per_partitions * head_size_in_bytes ..][0 .. kv_heads_per_partitions * head_size_in_bytes];
        const v = v_[(token_offset + t) * kv_heads_per_partitions * head_size_in_bytes ..][0 .. kv_heads_per_partitions * head_size_in_bytes];

        for (0..kv_heads_per_partitions) |kv_head_id| {
            const k_offset = kv_head_id * head_size_in_bytes;
            @memcpy(req.k, k[k_offset..][0..head_size_in_bytes]);

            const v_offset = kv_head_id * head_size_in_bytes;
            @memcpy(req.v, v[v_offset..][0..head_size_in_bytes]);

            req.header.kv_head_id = @intCast(kv_head_offset + kv_head_id);

            var q_offset = kv_head_id * head_size_in_bytes * attrs.num_q_per_head;
            var q_sent: u8 = 0;

            while (q_sent < attrs.num_q_per_head) {
                const nq = @min(attrs.num_q_per_head - q_sent, attrs.num_q_per_packet);
                @memcpy(req.q, q[q_offset..][0 .. head_size_in_bytes * nq]);
                q_offset += head_size_in_bytes * nq;

                req.header.num_queries = nq;
                req.header.first_q_id = q_sent;
                q_sent += nq;

                try client.send(ctx.io, req);
            }
        }

        //
        // ---------
        //

        var attn = attn_[t * kv_heads_per_partitions * q_size_in_bytes ..][0 .. kv_heads_per_partitions * q_size_in_bytes];
        var recv_q: usize = 0;

        while (recv_q < kv_heads_per_partitions * attrs.num_q_per_head) {
            const resp = client.receive(ctx.io, buffer) catch |err| {
                log.err("Failed to receive response: {any}", .{err});
                return err;
            };

            std.debug.assert(resp.header.num_queries <= attrs.num_q_per_head);
            std.debug.assert(resp.payload.len == resp.header.num_queries * attrs.head_size_in_bytes);
            std.debug.assert(resp.header.kv_head_id < attrs.num_kv_heads);
            std.debug.assert(resp.header.first_q_id + resp.header.num_queries <= attrs.num_q_per_head);

            const kv_head_id = resp.header.kv_head_id - kv_head_offset;
            const q_offset = (@as(usize, kv_head_id) * attrs.num_q_per_head + resp.header.first_q_id) * attrs.head_size_in_bytes;
            @memcpy(attn[q_offset .. q_offset + resp.payload.len], resp.payload);
            recv_q += resp.header.num_queries;
        }
    }

    return null;
}

fn asSlice(buf: zml.pjrtx.CustomCallBuffer) []u8 {
    const bytes: [*]u8 = @ptrCast(@alignCast(buf.ptr));
    return bytes[0..buf.shape.byteSize()];
}

fn asConstSlice(buf: zml.pjrtx.CustomCallBuffer) []const u8 {
    const bytes: [*]const u8 = @ptrCast(@alignCast(buf.ptr));
    return bytes[0..buf.shape.byteSize()];
}

fn asScalar(T: type, buf: zml.pjrtx.CustomCallBuffer) T {
    std.debug.assert(buf.shape.byteSize() == @sizeOf(T));
    return std.mem.bytesAsValue(T, asConstSlice(buf)[0..@sizeOf(T)]).*;
}

pub const Options = struct {
    buffer_size: usize = 1500 * 64,
};

const Client = struct {
    socket: std.Io.net.Socket,
    destination: std.Io.net.IpAddress,

    fn init(io: std.Io, destination: std.Io.net.IpAddress, options: Options) !Client {
        const src: std.Io.net.IpAddress = .{ .ip4 = .unspecified(0) };
        const socket = try src.bind(io, .{ .mode = .dgram, .protocol = .udp });

        try std.posix.setsockopt(socket.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, @ptrCast(&options.buffer_size));
        try std.posix.setsockopt(socket.handle, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, @ptrCast(&options.buffer_size));

        // Set the Don't Fragment bit to avoid fragmentation on IPv4
        switch (builtin.os.tag) {
            .linux => {
                // With PMTUDISC_DO the kernel will keep track of the ICMP messages to detect the real MTU.
                // Best practice [1] seems to be to use IP_PMTUDISC_PROBE and do DPLPMTUD to avoid Blind Performance-Degrading Attack [2]
                // But we assume to be within an internal network, otherwise attnd is useless anyway.
                // [1] https://seemann.io/posts/2025-02-19---ip-fragmentation/
                // [2] https://www.rfc-editor.org/rfc/rfc5927#section-7
                // TODO: Do we actually try to use the max MTU? If we just assume it to be fine, we could just as well use IP_PMTUDISC_PROBE
                const mtu_discover: u32 = std.os.linux.IP.PMTUDISC_DO;
                try std.posix.setsockopt(socket.handle, std.posix.IPPROTO.IP, std.os.linux.IP.MTU_DISCOVER, @ptrCast(&mtu_discover));
            },
            .macos => {
                // We could set IP_DONTFRAG on the socket, but Zig doesn't expose this option (yet).
            },
            else => {},
        }

        return .{
            .socket = socket,
            .destination = destination,
        };
    }

    pub fn send(self: *Client, io: std.Io, request: Request) !void {
        try self.socket.send(io, &self.destination, request.bytes);
    }

    pub fn receive(self: *Client, io: std.Io, buffer: []u8) !Response {
        const incoming = try self.socket.receive(io, buffer);
        return Response.fromBytes(incoming.data);
    }

    pub fn deinit(self: *Client, io: std.Io) void {
        self.socket.close(io);
    }
};

const Message = struct {
    bytes: []u8,
    header: *align(1) Header,
    payload: []u8,

    fn fromBytes(bytes: []u8) !Message {
        const n = @sizeOf(Header);
        if (bytes.len < n) return error.ShortMessage;

        const msg: Message = .{
            .bytes = bytes,
            .header = std.mem.bytesAsValue(Header, bytes[0..n]),
            .payload = bytes[n..],
        };

        if (!(msg.header.magic[0] == 'Z' and msg.header.magic[1] == 'M' and msg.header.magic[2] == 'L' and @intFromEnum(msg.header.model_id) < 5)) {
            return error.InvalidHeader;
        }

        return msg;
    }
};

const Request = struct {
    bytes: []u8,
    header: *align(1) Header,
    q: []u8,
    k: []u8,
    v: []u8,

    fn fromBytes(bytes: []u8, q_len: usize, k_len: usize, v_len: usize) !Request {
        const message = try Message.fromBytes(bytes);
        if (message.payload.len != q_len + k_len + v_len) return error.ShortMessage;

        return .{
            .bytes = message.bytes,
            .header = message.header,
            .q = message.payload[0..q_len],
            .k = message.payload[q_len .. q_len + k_len],
            .v = message.payload[q_len + k_len ..],
        };
    }
};

const Response = Message;

const Header = extern struct {
    const MAGIC: [3]u8 = .{ 'Z', 'M', 'L' }; // 0x5A4D4C

    const Type = enum(u8) {
        // Compute attention
        attn = 'a',
        // Starts a conversation
        start = 's',
        // Ping: the server answers immediately skipping all compute
        ping = 'p',
        _,
    };

    type: Type,
    kv_head_id: u8,

    conversation_id: u64 align(1),
    token_pos: u32 align(1),
    layer_id: u16 align(1),

    model_id: ModelId,
    first_q_id: u8,
    num_queries: u8,
    magic: [3]u8 align(1) = MAGIC,

    comptime {
        // destination MAC (6) + source MAC (6) + Ethertype (2) [+ 802.1Q / VLAN tag (4)]
        // We the optional VLAN, we can only guarantee an alignment of 4 for the payload after the prefix + header.
        const ETH_HEADER_SIZE = 6 + 6 + 2;
        const IP_HEADER_SIZE = 20;
        const UDP_HEADER_SIZE = 8;
        const PREFIX_SIZE = ETH_HEADER_SIZE + IP_HEADER_SIZE + UDP_HEADER_SIZE;
        stdx.debug.assertComptime(
            (PREFIX_SIZE + @sizeOf(Header)) % 4 == 0,
            "ETH_HEADER_SIZE ({}) + @sizeOf(AttnRequest) ({}) == {d} != k * 4",
            .{ PREFIX_SIZE, @sizeOf(Header), PREFIX_SIZE + @sizeOf(Header) },
        );
    }
};

const ModelId = enum(u8) {
    @"llama-3.1-8B" = 0,
    @"llama-3.2-1B" = 1,
    @"qwen3-14B" = 2,
    @"qwen3-32B" = 3,
    @"lfm2.5-1.2B" = 4,
};
