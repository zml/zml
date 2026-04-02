const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const ModelType = enum(u8) {
    @"llama-3.1-8B" = 0,
    @"llama-3.2-1B" = 1,
    @"qwen3-14B" = 2,
    @"qwen3-32B" = 3,
};

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
    token_id: u32 align(1),
    layer_id: u16 align(1),

    model_type: u8,
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
            (PREFIX_SIZE + @sizeOf(Message)) % 4 == 0,
            "ETH_HEADER_SIZE ({}) + @sizeOf(AttnRequest) ({}) == {d} != k * 4",
            .{ PREFIX_SIZE, @sizeOf(Message), PREFIX_SIZE + @sizeOf(Message) },
        );
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

        if (msg.header.magic[0] == 'Z' and msg.header.magic[1] == 'M' and msg.header.magic[2] == 'L' and msg.header.model_type < 4) {
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

const Options = struct {
    buffer_size: usize = 1500 * 64,
};

const Client = struct {
    socket: std.Io.net.Socket,
    destination: std.Io.net.IpAddress,

    fn init(io: std.Io, destination: std.Io.net.IpAddress, options: Options) Client {
        const src: std.Io.net.IpAddress = .{ .ip4 = .unspecified(0) };
        const socket = try src.bind(io, .{ .mode = .dgram, .protocol = .udp });

        try std.posix.setsockopt(socket.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, @ptrCast(&options.buffer_size));
        try std.posix.setsockopt(socket.handle, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, @ptrCast(&options.buffer_size));

        if (builtin.os.tag == .linux) {
            // Error on packet fragmentation.
            const mtu_discover: u32 = std.os.linux.IP.PMTUDISC_DO;
            try std.posix.setsockopt(socket.handle, std.posix.IPPROTO.IP, std.os.linux.IP.MTU_DISCOVER, @ptrCast(&mtu_discover));
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
