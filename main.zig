const std = @import("std");
const upb = @import("upb");
const c = @import("c");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();

    var upb_alloc: upb.Allocator = .init(arena.allocator());
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());
    defer c.upb_Arena_Free(upb_arena);

    const sharding = struct {
        num_replicas: i32 = 2,
        num_partitions: i32 = 4,
    }{};

    const options = blk: {
        const options = try upb.new(c.xla_CompileOptionsProto, upb_arena);
        c.xla_CompileOptionsProto_set_executable_build_options(options, executable_build_options_blk: {
            const exec_build_options = try upb.new(c.xla_ExecutableBuildOptionsProto, upb_arena);

            c.xla_ExecutableBuildOptionsProto_set_device_ordinal(exec_build_options, -1);
            c.xla_ExecutableBuildOptionsProto_set_num_replicas(exec_build_options, sharding.num_replicas);
            c.xla_ExecutableBuildOptionsProto_set_num_partitions(exec_build_options, sharding.num_partitions);
            c.xla_ExecutableBuildOptionsProto_set_use_spmd_partitioning(exec_build_options, sharding.num_partitions > 1 or sharding.num_replicas > 1);

            c.xla_ExecutableBuildOptionsProto_set_device_assignment(exec_build_options, device_assignment_blk: {
                const device_assignment = try upb.new(c.xla_DeviceAssignmentProto, upb_arena);

                c.xla_DeviceAssignmentProto_set_replica_count(device_assignment, sharding.num_replicas);
                c.xla_DeviceAssignmentProto_set_computation_count(device_assignment, sharding.num_partitions);

                const computation_devices = c.xla_DeviceAssignmentProto_resize_computation_devices(device_assignment, sharding.num_partitions, upb_arena);
                for (computation_devices[0..sharding.num_partitions], 0..) |*computation_device, i| {
                    computation_device.* = try upb.new(c.xla_DeviceAssignmentProto_ComputationDevice, upb_arena);
                    _ = c.xla_DeviceAssignmentProto_ComputationDevice_add_replica_device_ids(computation_device.*, @intCast(i), upb_arena);
                }
                break :device_assignment_blk device_assignment;
            });
            break :executable_build_options_blk exec_build_options;
        });

        // const overrides_map = c._xla_CompileOptionsProto_env_option_overrides_mutable_upb_map(options, upb_arena);

        break :blk options;
    };

    _ = options;

}
