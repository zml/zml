pub fn NamedArgs(comptime Spec: type, comptime ValueT: type) type {
    const in = @typeInfo(Spec).@"struct".fields;
    comptime var names: [in.len][]const u8 = undefined;
    for (in, 0..) |f, i| names[i] = f.name;
    return @Struct(.auto, null, &names, &@splat(ValueT), &@splat(.{}));
}
