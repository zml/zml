const std = @import("std");

pub const max_frequency_hz: u32 = 4000;

extern fn ProfilerStart(name: [*:0]const u8) c_int;
extern fn ProfilerFlush() void;
extern fn ProfilerStop() void;
extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

pub const Profile = struct {
    allocator: std.mem.Allocator,
    file_path: ?[:0]u8 = null,
    frequency_hz: u32 = max_frequency_hz,
    started: bool = false,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        output_dir: []const u8,
        session_id: []const u8,
        frequency_hz: u32,
    ) !@This() {
        var profile: @This() = .{
            .allocator = allocator,
            .frequency_hz = frequency_hz,
        };
        if (output_dir.len == 0) return profile;

        try std.Io.Dir.createDirPath(.cwd(), io, output_dir);

        const basename = try std.fmt.allocPrint(allocator, "{s}.prof", .{session_id});
        defer allocator.free(basename);

        profile.file_path = try std.Io.Dir.path.joinZ(allocator, &.{ output_dir, basename });
        return profile;
    }

    pub fn deinit(self: *@This()) void {
        self.stop();
        if (self.file_path) |file_path| self.allocator.free(file_path);
        self.file_path = null;
    }

    pub inline fn enabled(self: *const @This()) bool {
        return self.file_path != null;
    }

    pub fn start(self: *@This()) !void {
        const file_path = self.file_path orelse return;
        const frequency = try std.fmt.allocPrint(self.allocator, "{d}", .{@min(self.frequency_hz, max_frequency_hz)});
        defer self.allocator.free(frequency);
        const frequency_z = try self.allocator.dupeZ(u8, frequency);
        defer self.allocator.free(frequency_z);
        if (setenv("CPUPROFILE_FREQUENCY", frequency_z.ptr, 1) != 0) return error.PprofSetFrequencyFailed;
        if (ProfilerStart(file_path.ptr) == 0) return error.PprofStartFailed;
        self.started = true;
    }

    pub fn stop(self: *@This()) void {
        if (!self.started) return;
        ProfilerFlush();
        ProfilerStop();
        self.started = false;
    }

    pub fn outputPath(self: *const @This()) ?[:0]const u8 {
        return self.file_path;
    }
};
