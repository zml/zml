pub fn NamedArgs(comptime Spec: type, comptime ValueT: type) type {
    const field_names = @typeInfo(Spec).@"struct".field_names;
    return @Struct(.auto, null, field_names, &@splat(ValueT), &@splat(.{}));
}
