pub const Selection = struct {
    pub const Kind = enum { none, device, process };

    kind: Kind = .none,
    device: u16 = 0,
    pid: u32 = 0,

    /// Clear the selection
    pub fn clear(self: *Selection) void {
        self.* = .{};
    }

    pub fn deviceEq(self: Selection, id: u16) bool {
        return self.kind == .device and self.device == id;
    }

    pub fn processEq(self: Selection, p: u32) bool {
        return self.kind == .process and self.pid == p;
    }

    /// Select a device. Returns true if the selection actually changed.
    pub fn setDevice(self: *Selection, id: u16) bool {
        if (self.deviceEq(id)) return false;
        self.* = .{ .kind = .device, .device = id };
        return true;
    }

    /// Select a process by PID. Returns true if the selection actually changed.
    pub fn setProcess(self: *Selection, p: u32) bool {
        if (self.processEq(p)) return false;
        self.* = .{ .kind = .process, .pid = p };
        return true;
    }
};
