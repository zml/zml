const std = @import("std");
const zml = @import("zml");
const tools = @import("tools.zig");
const log = std.log.scoped(.flux2_scheduler);

fn linspace(allocator: std.mem.Allocator, start: f32, end: f32, n: usize) ![]f32 {
    const res = try allocator.alloc(f32, n);
    if (n == 0) return res;
    if (n == 1) {
        res[0] = start;
        return res;
    }
    const step = (end - start) / @as(f32, @floatFromInt(n - 1));
    for (0..n) |idx| {
        res[idx] = start + step * @as(f32, @floatFromInt(idx));
    }
    return res;
}

fn time_shift(shifttype: []const u8, mu: f32, sigma: f32, t: f32) f32 {
    if (std.mem.eql(u8, shifttype, "exponential")) {
        return @exp(mu) / (@exp(mu) + std.math.pow(f32, (1.0 / t - 1.0), sigma));
    } else {
        return mu / (mu + std.math.pow(f32, (1.0 / t - 1.0), sigma));
    }
}

pub const FlowMatchEulerDiscreteScheduler = struct {
    num_train_timesteps: usize = 1000,
    shift: f32 = 1.0,
    use_dynamic_shifting: bool = false,
    base_shift: f32 = 0.5,
    max_shift: f32 = 1.15,
    base_image_seq_len: usize = 256,
    max_image_seq_len: usize = 4096,
    invert_sigmas: bool = false,
    shift_terminal: ?f32 = null,
    use_karras_sigmas: bool = false,
    use_exponential_sigmas: bool = false,
    use_beta_sigmas: bool = false,
    time_shift_type: []const u8 = "exponential", // "exponential" or "linear"
    stochastic_sampling: bool = false,

    allocator: std.mem.Allocator,
    timesteps: []f32,
    sigmas: []f32,
    num_inference_steps: usize,
    sigma_min: f32,
    sigma_max: f32,
    step_index: ?usize,
    begin_index: ?usize,

    pub const Config = struct {
        num_train_timesteps: usize = 1000,
        shift: f32 = 1.0,
        use_dynamic_shifting: bool = false,
        base_shift: f32 = 0.5,
        max_shift: f32 = 1.15,
        base_image_seq_len: usize = 256,
        max_image_seq_len: usize = 4096,
        invert_sigmas: bool = false,
        shift_terminal: ?f32 = null,
        use_karras_sigmas: bool = false,
        use_exponential_sigmas: bool = false,
        use_beta_sigmas: bool = false,
        time_shift_type: []const u8 = "exponential",
        stochastic_sampling: bool = false,
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !*@This() {
        const self = try allocator.create(@This());
        self.* = .{
            .num_train_timesteps = config.num_train_timesteps,
            .shift = config.shift,
            .use_dynamic_shifting = config.use_dynamic_shifting,
            .base_shift = config.base_shift,
            .max_shift = config.max_shift,
            .base_image_seq_len = config.base_image_seq_len,
            .max_image_seq_len = config.max_image_seq_len,
            .invert_sigmas = config.invert_sigmas,
            .shift_terminal = config.shift_terminal,
            .use_karras_sigmas = config.use_karras_sigmas,
            .use_exponential_sigmas = config.use_exponential_sigmas,
            .use_beta_sigmas = config.use_beta_sigmas,
            .time_shift_type = try allocator.dupe(u8, config.time_shift_type),
            .stochastic_sampling = config.stochastic_sampling,
            .allocator = allocator,
            .timesteps = &[_]f32{},
            .sigmas = &[_]f32{},
            .num_inference_steps = 0,
            .sigma_min = 0,
            .sigma_max = 0,
            .step_index = null,
            .begin_index = null,
        };

        // Initialize default sigmas/timesteps
        // Based on Python __init__ logic
        var timesteps_np = try linspace(allocator, 1.0, @floatFromInt(self.num_train_timesteps), self.num_train_timesteps);
        defer allocator.free(timesteps_np);
        // reverse
        std.mem.reverse(f32, timesteps_np);

        // sigmas = timesteps / num_train_timesteps
        const sigmas = try allocator.alloc(f32, timesteps_np.len);
        for (timesteps_np, 0..) |timestep, idx| {
            sigmas[idx] = timestep / @as(f32, @floatFromInt(self.num_train_timesteps));
        }
        defer allocator.free(sigmas); // We will dupe or use it for final_sigmas

        var final_sigmas = try allocator.dupe(f32, sigmas);

        if (!self.use_dynamic_shifting) {
            for (final_sigmas) |*s| {
                s.* = self.shift * s.* / (1.0 + (self.shift - 1.0) * s.*);
            }
        }

        self.sigmas = final_sigmas;
        self.timesteps = try allocator.alloc(f32, final_sigmas.len);
        for (final_sigmas, 0..) |sigma, idx| {
            self.timesteps[idx] = sigma * @as(f32, @floatFromInt(self.num_train_timesteps));
        }

        self.sigma_min = self.sigmas[self.sigmas.len - 1];
        self.sigma_max = self.sigmas[0];

        return self;
    }

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.time_shift_type);
        if (self.timesteps.len > 0) self.allocator.free(self.timesteps);
        if (self.sigmas.len > 0) self.allocator.free(self.sigmas);
        self.allocator.destroy(self);
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, io: std.Io, repo_dir: std.Io.Dir, progress: ?*std.Progress.Node, options: struct { subfolder: []const u8 = "scheduler", json_name: []const u8 = "scheduler_config.json" }) !*@This() {
        if (progress) |p| {
            p.increaseEstimatedTotalItems(1);
            var node = p.start("Loading scheduler config...", 1);
            defer node.end();
        }
        const config_json = try tools.parseConfig(Config, allocator, io, repo_dir, .{ .subfolder = options.subfolder, .json_name = options.json_name });
        errdefer config_json.deinit();

        return try init(allocator, config_json.value);
    }

    pub fn set_timesteps(self: *@This(), num_inference_steps: ?usize, sigmas_opt: ?[]const f32, mu: ?f32, timesteps_opt: ?[]const f32) !void {
        if (self.use_dynamic_shifting and mu == null) {
            return error.MuRequired;
        }

        var num_steps = num_inference_steps orelse
            (if (sigmas_opt) |s| s.len else if (timesteps_opt) |t| t.len else 0);

        self.num_inference_steps = num_steps;

        // Clean up old
        if (self.timesteps.len > 0) self.allocator.free(self.timesteps);
        if (self.sigmas.len > 0) self.allocator.free(self.sigmas);

        var sigmas: []f32 = undefined;
        var is_timesteps_provided = false;

        if (timesteps_opt) |timestep| {
            is_timesteps_provided = true;
            // Python: if sigmas is None: if timesteps is None: ... else: sigmas = timesteps / num_train_timesteps
            if (sigmas_opt == null) {
                sigmas = try self.allocator.alloc(f32, timestep.len);
                for (timestep, 0..) |val, i| {
                    sigmas[i] = val / @as(f32, @floatFromInt(self.num_train_timesteps));
                }
            }
        }

        if (sigmas_opt) |s| {
            sigmas = try self.allocator.dupe(f32, s);
            num_steps = s.len;
        } else if (!is_timesteps_provided) {
            const start_t = self.sigma_to_t(self.sigma_max);
            const end_t = self.sigma_to_t(self.sigma_min);
            const timesteps_arr = try linspace(self.allocator, start_t, end_t, num_steps);
            defer self.allocator.free(timesteps_arr);

            sigmas = try self.allocator.alloc(f32, timesteps_arr.len);
            for (timesteps_arr, 0..) |t, i| {
                sigmas[i] = t / @as(f32, @floatFromInt(self.num_train_timesteps));
            }
        }

        // 2. Timestep shifting
        if (self.use_dynamic_shifting) {
            if (mu) |m| {
                for (sigmas) |*s| {
                    s.* = time_shift(self.time_shift_type, m, 1.0, s.*);
                }
            }
        } else {
            for (sigmas) |*s| {
                s.* = self.shift * s.* / (1.0 + (self.shift - 1.0) * s.*);
            }
        }

        // 3. Stretch shift to terminal
        if (self.shift_terminal) |term| {
            const last_sigma = sigmas[sigmas.len - 1];
            const one_minus_last_sigma = 1.0 - last_sigma;
            const scale_factor = one_minus_last_sigma / (1.0 - term);

            for (sigmas) |*s| {
                const one_minus_z = 1.0 - s.*;
                s.* = 1.0 - (one_minus_z / scale_factor);
            }
        }

        // 5. Convert sigmas to timesteps
        if (!is_timesteps_provided) {
            const new_timesteps = try self.allocator.alloc(f32, sigmas.len);
            for (sigmas, 0..) |s, i| {
                new_timesteps[i] = s * @as(f32, @floatFromInt(self.num_train_timesteps));
            }
            self.timesteps = new_timesteps;
        } else {
            if (timesteps_opt) |t| {
                self.timesteps = try self.allocator.dupe(f32, t);
            } else {
                self.timesteps = try self.allocator.alloc(f32, sigmas.len);
                for (sigmas, 0..) |s, i| {
                    self.timesteps[i] = s * @as(f32, @floatFromInt(self.num_train_timesteps));
                }
            }
        }

        // 6. Append terminal sigma
        const new_sigmas_len = sigmas.len + 1;
        var new_sigmas = try self.allocator.realloc(sigmas, new_sigmas_len);
        if (self.invert_sigmas) {
            new_sigmas[new_sigmas_len - 1] = 1.0;
        } else {
            new_sigmas[new_sigmas_len - 1] = 0.0;
        }
        self.sigmas = new_sigmas;

        self.step_index = null;
        self.begin_index = null;
    }

    pub fn step(
        self: *@This(),
        model_output: []const f32,
        sample: []const f32,
        out_sample: []f32,
    ) !void {
        if (self.step_index == null) {
            return error.StepIndexNotSet;
        }

        const step_idx = self.step_index.?;
        const sigma = self.sigmas[step_idx];
        const sigma_next = self.sigmas[step_idx + 1];

        const dt = sigma_next - sigma;

        for (sample, 0..) |s, i| {
            const m = model_output[i];
            out_sample[i] = s + dt * m;
        }

        self.step_index.? += 1;
    }

    pub fn set_begin_index(self: *@This(), begin_index: usize) void {
        self.begin_index = begin_index;
        self.step_index = begin_index;
    }

    fn index_for_timestep(self: *@This(), timestep: f32) usize {
        const epsilon = 1e-4; // slightly loose
        var found_i: ?usize = null;
        for (self.timesteps, 0..) |t, i| {
            if (@abs(t - timestep) < epsilon) {
                found_i = i;
                break;
            }
        }

        if (found_i) |i| {
            return i;
        }
        return 0;
    }

    fn sigma_to_t(self: *@This(), sigma: f32) f32 {
        return sigma * @as(f32, @floatFromInt(self.num_train_timesteps));
    }
};
