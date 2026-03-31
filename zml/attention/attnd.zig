const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const zml = @import("../zml.zig");

const log = std.log.scoped(.@"zml/attnd");
const BATCH_SIZE = 8;

var context: ?*Context = null;

pub const Config = struct {
    destination: std.Io.net.IpAddress,
    mtu: u16,
};

pub const Parameters = struct {
    model_id: ModelId,
    head_dim: u32,
    head_bytes: u32,
    num_kv_heads: u8,
    num_q_per_head: u8,
    num_q_per_packet: u8,
    is_prefill: bool,

    pub const Init = struct {
        model_id: ModelId,
        head_dim: u32,
        num_attention_heads: u32,
        num_kv_heads: u8,
        is_prefill: bool,
    };

    pub fn init(args: Init) Parameters {
        const mtu: u32 = if (context) |ctx| @intCast(ctx.mtu) else @panic("attnd wasn't registered yet, use register() first.");
        const num_q_per_head: u8 = @intCast(@divExact(args.num_attention_heads, args.num_kv_heads));
        const head_bytes = args.head_dim * @sizeOf(u16); // TODO: Here we assume bf16 or equivalent.
        const kv_bytes = 2 * head_bytes;

        var max_q_per_packet: u8 = @intCast(std.math.divFloor(u32, mtu - @sizeOf(Header) - kv_bytes, head_bytes) catch unreachable);
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
            .is_prefill = args.is_prefill,
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

    try ctx.initialize(io, allocator, platform, config);

    context = ctx;

    try targets.attnd.register(platform);
}

pub fn deinit() !void {
    if (context) |ctx| {
        try ctx.deinit();
        context = null;
    }
}

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    mtu: u16,
    // One client per device.
    clients: []Client,

    fn initialize(ctx: *Context, io: std.Io, allocator: std.mem.Allocator, platform: *const zml.Platform, config: Config) !void {
        ctx.allocator = allocator;
        ctx.io = io;
        ctx.mtu = config.mtu;
        ctx.clients = try allocator.alloc(Client, platform.devices.len);
        errdefer allocator.free(ctx.clients);

        var c_len: usize = 0;
        errdefer for (ctx.clients[0..c_len]) |*c| c.deinit(allocator, io);
        for (ctx.clients) |*c| {
            try c.initialize(allocator, io, config);
            c_len += 1;
        }
    }

    fn deinit(self: *Context) !void {
        for (self.clients) |*c| {
            c.deinit(self.allocator, self.io);
        }
        self.allocator.free(self.clients);
        self.allocator.destroy(self);
    }
};

pub fn causalAttention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_offset: zml.Tensor, metadata: Metadata, parameters: Parameters) zml.Tensor {
    const ctx = zml.module.CompilationContext.current();
    const num_partitions = ctx.partitioning.numPartitionsForLogicalAxis(q.shape(), .model) catch unreachable;

    const actual_k, const actual_v = if (parameters.is_prefill)
        .{ k, v }
    else
        .{
            k.dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_offset, .len = 1 } }),
            v.dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_offset, .len = 1 } }),
        };

    // Concatenate q/k/v and metadata into a single byte buffer on the GPU.
    // This reduces 7 separate D2H transfers to 1 during host-compute offload.
    const packed_input = zml.Tensor.concatenate(&.{
        q.bytes(),
        actual_k.bytes(),
        actual_v.bytes(),
        metadata.conversation_id.bytes(),
        metadata.layer_id.bytes(),
        token_offset.bytes(),
        metadata.num_tokens.bytes(),
    }, .bytes);

    const out = targets.attnd.call(
        .{ .mqkv = packed_input },
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
            .q_byte_size = @intCast(q.shape().byteSize()),
            .k_byte_size = @intCast(actual_k.shape().byteSize()),
            .v_byte_size = @intCast(actual_v.shape().byteSize()),
        },
    );

    return out.attn;
}

pub const targets = struct {
    const NAME = "attnd";

    pub const attnd = zml.ops.CustomCall(Input, Output, Attributes, attndCall, .{
        .name = NAME,
        .sharding_aware = true,
        .has_side_effect = false,
        .compute_on_host = true,
    });
};

const Input = struct {
    mqkv: zml.Tensor,
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
    /// Byte sizes for unpacking the concatenated packed buffer.
    q_byte_size: u32,
    k_byte_size: u32,
    v_byte_size: u32,
};

fn attndCall(
    call_frame: *zml.pjrt.ffi.CallFrame,
    input: zml.pjrtx.TensorToCustomCallBuffer(Input),
    output: zml.pjrtx.ShapeToCustomCallBuffer(Output),
    attrs: Attributes,
) !?*zml.pjrt.ffi.Error {
    const ctx: *Context = context.?;
    const device_ordinal: u64 = @intCast(try call_frame.ctx.getDeviceOrdinal(call_frame.api));

    const client = &ctx.clients[device_ordinal];
    // Unpack the concatenated byte buffer: [q_bytes | k_bytes | v_bytes | conversation_id(8) | layer_id(2) | token_offset(4) | num_tokens(4)]
    const out_full = output.attn.asHostSlice();
    const mqkv = input.mqkv.asHostConstSlice();
    const q_end = attrs.q_byte_size;
    const k_end = q_end + attrs.k_byte_size;
    const v_end = k_end + attrs.v_byte_size;
    const q_full = mqkv[0..q_end];
    const k_full = mqkv[q_end..k_end];
    const v_full = mqkv[k_end..v_end];
    const conversation_id = std.mem.bytesAsValue(u64, mqkv[v_end..][0..@sizeOf(u64)]).*;
    const layer_id = std.mem.bytesAsValue(u16, mqkv[v_end + @sizeOf(u64) ..][0..@sizeOf(u16)]).*;
    const token_offset = std.mem.bytesAsValue(u32, mqkv[v_end + @sizeOf(u64) + @sizeOf(u16) ..][0..@sizeOf(u32)]).*;
    const num_tokens = std.mem.bytesAsValue(u32, mqkv[v_end + @sizeOf(u64) + @sizeOf(u16) + @sizeOf(u32) ..][0..@sizeOf(u32)]).*;

    const q_size_in_bytes = attrs.num_q_per_head * attrs.head_size_in_bytes;
    const head_size_in_bytes = attrs.head_size_in_bytes;
    const kv_heads_per_partitions = @divExact(attrs.num_kv_heads, attrs.num_partitions);
    const kv_head_offset = kv_heads_per_partitions * device_ordinal;
    const payload_size = attrs.num_q_per_packet * attrs.head_size_in_bytes;

    for (0..BATCH_SIZE) |i| {
        const payload = client.payload_buffer[i * payload_size ..][0..payload_size];
        client.recv_iovecs[i][1] = .{ .base = payload.ptr, .len = payload_size };
    }

    for (0..num_tokens) |t| {
        const q = q_full[t * kv_heads_per_partitions * q_size_in_bytes ..][0 .. kv_heads_per_partitions * q_size_in_bytes];
        const k = k_full[t * kv_heads_per_partitions * head_size_in_bytes ..][0 .. kv_heads_per_partitions * head_size_in_bytes];
        const v = v_full[t * kv_heads_per_partitions * head_size_in_bytes ..][0 .. kv_heads_per_partitions * head_size_in_bytes];

        var num_send_messages: usize = 0;
        var packet_i: usize = 0;
        for (0..kv_heads_per_partitions) |kv_head_id| {
            const k_offset = kv_head_id * head_size_in_bytes;
            const v_offset = kv_head_id * head_size_in_bytes;
            var q_offset = kv_head_id * head_size_in_bytes * attrs.num_q_per_head;
            var q_sent: u8 = 0;

            while (q_sent < attrs.num_q_per_head) {
                const nq = @min(attrs.num_q_per_head - q_sent, attrs.num_q_per_packet);
                client.send_headers[packet_i] = .{
                    .type = .attn,
                    .kv_head_id = @intCast(kv_head_offset + kv_head_id),
                    .conversation_id = conversation_id,
                    .token_pos = token_offset + @as(u32, @intCast(t)),
                    .layer_id = layer_id,
                    .model_id = attrs.model_id,
                    .first_q_id = q_sent,
                    .num_queries = nq,
                };
                client.send_iovecs[packet_i][1] = .{ .base = @constCast(q[q_offset..].ptr), .len = head_size_in_bytes * nq };
                client.send_iovecs[packet_i][2] = .{ .base = @constCast(k[k_offset..].ptr), .len = head_size_in_bytes };
                client.send_iovecs[packet_i][3] = .{ .base = @constCast(v[v_offset..].ptr), .len = head_size_in_bytes };

                q_offset += head_size_in_bytes * nq;
                q_sent += nq;
                num_send_messages += 1;
                packet_i += 1;
                if (packet_i == BATCH_SIZE) {
                    try client.rawSendMany(&client.send_messages);
                    packet_i = 0;
                }
            }
        }

        if (packet_i > 0) {
            try client.rawSendMany(client.send_messages[0..packet_i]);
        }

        var out = out_full[t * kv_heads_per_partitions * q_size_in_bytes ..][0 .. kv_heads_per_partitions * q_size_in_bytes];
        var recv_q: usize = 0;
        var recv_messages_remaining = num_send_messages;
        while (recv_q < kv_heads_per_partitions * attrs.num_q_per_head) {
            const received = client.rawReceiveMany(client.recv_messages[0..@min(recv_messages_remaining, BATCH_SIZE)]) catch |err| {
                log.err("Failed to receive response: {any}", .{err});
                return err;
            };
            for (0..received) |i| {
                const header_in = &client.recv_headers[i];
                const payload = client.payload_buffer[i * payload_size ..][0..payload_size];

                std.debug.assert(header_in.num_queries <= attrs.num_q_per_head);
                std.debug.assert(header_in.kv_head_id < attrs.num_kv_heads);
                std.debug.assert(header_in.first_q_id + header_in.num_queries <= attrs.num_q_per_head);

                const n = header_in.num_queries * attrs.head_size_in_bytes;
                const kv_head_id = header_in.kv_head_id - kv_head_offset;
                const q_offset = (@as(usize, kv_head_id) * attrs.num_q_per_head + header_in.first_q_id) * attrs.head_size_in_bytes;
                @memcpy(out[q_offset .. q_offset + n], payload[0..n]);
                recv_q += header_in.num_queries;
            }
            recv_messages_remaining -= received;
        }
    }

    return null;
}

const SO_BUF_SIZE: usize = 1500 * 64;

const Client = struct {
    socket: std.Io.net.Socket,
    destination: std.Io.net.IpAddress,
    payload_buffer: []u8,
    send_addr: std.Io.Threaded.PosixAddress,
    send_headers: [BATCH_SIZE]Header,
    send_iovecs: [BATCH_SIZE][4]std.posix.iovec,
    send_messages: [BATCH_SIZE]std.os.linux.mmsghdr,
    recv_addr: std.Io.Threaded.PosixAddress,
    recv_headers: [BATCH_SIZE]Header,
    recv_iovecs: [BATCH_SIZE][2]std.posix.iovec,
    recv_messages: [BATCH_SIZE]std.os.linux.mmsghdr,

    fn initialize(self: *Client, allocator: std.mem.Allocator, io: std.Io, config: Config) !void {
        const src: std.Io.net.IpAddress = .{ .ip4 = .unspecified(0) };
        const socket = try src.bind(io, .{ .mode = .dgram, .protocol = .udp });
        errdefer socket.close(io);

        try std.posix.setsockopt(socket.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, @ptrCast(&SO_BUF_SIZE));
        try std.posix.setsockopt(socket.handle, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, @ptrCast(&SO_BUF_SIZE));

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

        const payload_buffer = try allocator.alloc(u8, BATCH_SIZE * @as(usize, @intCast(config.mtu)));
        errdefer allocator.free(payload_buffer);

        self.socket = socket;
        self.destination = config.destination;
        self.payload_buffer = payload_buffer;

        const send_name = &self.send_addr.any;
        const send_namelen = std.Io.Threaded.addressToPosix(&self.destination, &self.send_addr);
        const recv_name = &self.recv_addr.any;
        const recv_namelen = std.Io.Threaded.addressToPosix(&src, &self.recv_addr);
        for (0..BATCH_SIZE) |i| {
            self.send_iovecs[i][0] = .{ .base = @ptrCast(&self.send_headers[i]), .len = @sizeOf(Header) };
            self.send_messages[i] = .{
                .hdr = .{
                    .name = send_name,
                    .namelen = send_namelen,
                    .iov = self.send_iovecs[i][0..],
                    .iovlen = self.send_iovecs[i].len,
                    .control = null,
                    .controllen = 0,
                    .flags = 0,
                },
                .len = 0,
            };

            self.recv_iovecs[i][0] = .{ .base = @ptrCast(&self.recv_headers[i]), .len = @sizeOf(Header) };
            self.recv_messages[i] = .{
                .hdr = .{
                    .name = recv_name,
                    .namelen = recv_namelen,
                    .iov = self.recv_iovecs[i][0..],
                    .iovlen = self.recv_iovecs[i].len,
                    .control = null,
                    .controllen = 0,
                    .flags = 0,
                },
                .len = 0,
            };
        }
    }

    pub fn rawSendMany(self: *Client, messages: []std.os.linux.mmsghdr) !void {
        var sent: usize = 0;
        while (sent < messages.len) {
            const rc = std.os.linux.sendmmsg(
                self.socket.handle,
                @ptrCast(messages[sent..].ptr),
                @intCast(messages.len - sent),
                std.os.linux.MSG.DONTWAIT,
            );
            switch (std.posix.errno(rc)) {
                .SUCCESS => {
                    std.debug.assert(rc > 0);
                    sent += @intCast(rc);
                },
                else => {
                    log.err("Failed to send request batch: {any}", .{rc});
                    @panic("oups");
                },
            }
        }
    }

    pub fn rawReceiveMany(self: *Client, messages: []std.os.linux.mmsghdr) !usize {
        while (true) {
            const rc = std.os.linux.recvmmsg(
                self.socket.handle,
                @ptrCast(messages.ptr),
                @intCast(messages.len),
                0,
                null,
            );
            switch (std.posix.errno(rc)) {
                .SUCCESS => {
                    std.debug.assert(rc > 0);
                    return @intCast(rc);
                },
                else => {
                    log.err("Failed to receive response batch: {any}", .{rc});
                    @panic("oups");
                },
            }
        }
    }

    pub fn deinit(self: *Client, allocator: std.mem.Allocator, io: std.Io) void {
        allocator.free(self.payload_buffer);
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

        if (!(msg.header.magic[0] == 'Z' and msg.header.magic[1] == 'M' and msg.header.magic[2] == 'L')) {
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
    _, // Unsupported models
};
