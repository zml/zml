const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Exe = @import("exe.zig").Exe;
const Platform = @import("platform.zig").Platform;
const tracer = @import("profiling/tracer.zig");

pub const DecodeChain = struct {
    pub const OutputRef = pjrt.LoadedExecutable.ExecuteChainOutputRef;

    pub const ArgOverride = struct {
        arg_index: usize,
        output: OutputRef,
    };

    const Step = struct {
        exe: *const Exe,
        input_offset: usize,
        input_len: usize,
        num_args: usize,
        results: *Exe.Results,
        returned_outputs: []const bool,
    };

    allocator: std.mem.Allocator,
    platform: *const Platform,
    num_devices: ?usize = null,
    context: ?*pjrt.ExecuteContext = null,
    steps: std.ArrayList(Step) = .empty,
    inputs: std.ArrayList(pjrt.LoadedExecutable.ExecuteChainInput) = .empty,

    pub fn init(allocator: std.mem.Allocator, platform: *const Platform) DecodeChain {
        return .{
            .allocator = allocator,
            .platform = platform,
        };
    }

    pub fn deinit(self: *DecodeChain) void {
        self.steps.deinit(self.allocator);
        self.inputs.deinit(self.allocator);
    }

    pub fn append(
        self: *DecodeChain,
        exe: *const Exe,
        arguments: *const Exe.Arguments,
        results: *Exe.Results,
        returned_outputs: []const bool,
        overrides: []const ArgOverride,
    ) !usize {
        stdx.debug.assert(exe.platform == self.platform, "DecodeChain steps must use the same platform", .{});
        stdx.debug.assert(returned_outputs.len == results.expected_shapes.len, "DecodeChain returned output mask has wrong length", .{});
        stdx.debug.assert(arguments.expected_shapes.len == exe.input_shapes.len, "DecodeChain arguments do not match executable inputs", .{});

        if (self.num_devices) |num_devices| {
            stdx.debug.assert(num_devices == exe.num_devices, "DecodeChain steps must use the same device count", .{});
        } else {
            self.num_devices = exe.num_devices;
            self.context = exe.context;
        }
        stdx.debug.assert(self.context == exe.context, "DecodeChain steps must use the same execute context", .{});

        const step_index = self.steps.items.len;
        const num_args = arguments.expected_shapes.len;
        const input_offset = self.inputs.items.len;
        try self.inputs.ensureUnusedCapacity(self.allocator, exe.num_devices * num_args);

        for (0..exe.num_devices) |device_index| {
            for (0..num_args) |arg_index| {
                if (findOverride(overrides, arg_index)) |override| {
                    self.inputs.appendAssumeCapacity(.{ .output = override.output });
                } else {
                    self.inputs.appendAssumeCapacity(.{ .buffer = arguments.flat_buffers.buffers[device_index][arg_index] });
                }
            }
        }

        try self.steps.append(self.allocator, .{
            .exe = exe,
            .input_offset = input_offset,
            .input_len = exe.num_devices * num_args,
            .num_args = num_args,
            .results = results,
            .returned_outputs = returned_outputs,
        });
        return step_index;
    }

    pub fn output(step_index: usize, output_index: usize) OutputRef {
        return .{ .step = step_index, .output = output_index };
    }

    pub const CallOpts = struct {
        wait: bool = false,
    };

    pub fn execute(self: *DecodeChain, io: ?std.Io, opts: CallOpts) !void {
        var span = tracer.span("zml.decode_chain.execute", .{
            .wait = opts.wait,
            .step_count = self.steps.items.len,
        });
        defer span.end();

        stdx.debug.assert(opts.wait == false or io != null, "io should not be null when waiting for DecodeChain completion", .{});
        const num_devices = self.num_devices orelse return;

        const pjrt_steps = try self.allocator.alloc(pjrt.LoadedExecutable.ExecuteChainStep, self.steps.items.len);
        defer self.allocator.free(pjrt_steps);

        for (self.steps.items, pjrt_steps) |step, *pjrt_step| {
            pjrt_step.* = .{
                .executable = step.exe.exe,
                .num_args = step.num_args,
                .arguments = self.inputs.items[step.input_offset .. step.input_offset + step.input_len],
                .results = step.results.flat_buffers.buffers,
                .returned_outputs = step.returned_outputs,
            };
        }

        var events = [_]?*pjrt.Event{null} ** Platform.MAX_NUM_DEVICES;
        const partition_events = events[0..num_devices];
        const events_slice: ?[]?*pjrt.Event = switch (self.platform.target) {
            .neuron => partition_events,
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal => if (opts.wait) partition_events else null,
        };

        try pjrt.LoadedExecutable.executeChain(self.platform.pjrt_api, self.allocator, .{
            .num_devices = num_devices,
            .steps = pjrt_steps,
            .events = events_slice,
            .context = self.context,
        });

        switch (self.platform.target) {
            .neuron => {
                for (events_slice.?) |event| {
                    if (event) |ev| {
                        if (opts.wait) {
                            ev.await(self.platform.pjrt_api, io.?) catch unreachable;
                        }
                        ev.deinit(self.platform.pjrt_api);
                    }
                }
            },
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal => if (opts.wait) {
                for (events_slice.?) |event| {
                    if (event) |ev| {
                        ev.await(self.platform.pjrt_api, io.?) catch unreachable;
                    }
                }
            },
        }
    }

    fn findOverride(overrides: []const ArgOverride, arg_index: usize) ?ArgOverride {
        for (overrides) |override| {
            if (override.arg_index == arg_index) return override;
        }
        return null;
    }
};
