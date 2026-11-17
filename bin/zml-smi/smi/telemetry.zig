/// Populate the named fields via read(args..., binding) and an optional grouped query(args...).
pub fn Bind(comptime Sample: type, comptime plan: anytype) type {
    return struct {
        pub fn poll(sample: *Sample, args: anytype) void {
            const bindings = if (@hasField(@TypeOf(plan), "fields")) plan.fields else .{};
            inline for (@typeInfo(@TypeOf(bindings)).@"struct".field_names) |name| {
                const binding = @field(bindings, name);
                @field(sample.*, name) = @call(.auto, plan.read, args ++ .{binding});
            }
            if (@hasField(@TypeOf(plan), "query")) {
                const reply = @call(.auto, plan.query, args);
                inline for (@typeInfo(@TypeOf(reply)).@"struct".field_names) |name| {
                    if (@hasField(@TypeOf(bindings), name))
                        @compileError("more than one telemetry query writes " ++ name);
                    @field(sample.*, name) = @field(reply, name);
                }
            }
        }
    };
}
