pub fn toVoidSlice(data: anytype) []void {
    const info = @typeInfo(@TypeOf(data));
    if (info != .Pointer or info.Pointer.size != .Slice) {
        @compileError("toVoidSlice expects a slice");
    }
    return @as([*]void, @ptrCast(@alignCast(data.ptr)))[0..data.len];
}
