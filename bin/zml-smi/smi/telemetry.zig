/// Populate the named fields via read(args..., binding) and an optional grouped query(args...).
pub fn poll(sample: anytype, args: anytype, comptime plan: anytype) void {
    const bindings = if (@hasField(@TypeOf(plan), "fields")) plan.fields else .{};
    inline for (@typeInfo(@TypeOf(bindings)).@"struct".field_names) |name| {
        @field(sample.*, name) = @call(.auto, plan.read, args ++ .{@field(bindings, name)});
    }
    if (@hasField(@TypeOf(plan), "query")) {
        const reply = @call(.auto, plan.query, args);
        inline for (@typeInfo(@TypeOf(reply)).@"struct".field_names) |name| {
            if (@hasField(@TypeOf(bindings), name)) @compileError("more than one telemetry query writes " ++ name);
            @field(sample.*, name) = @field(reply, name);
        }
    }
}
