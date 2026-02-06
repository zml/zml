pub fn saveBufferToNpy(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, buf: zml.Buffer, path: []const u8) !void {
    _ = platform;
    const shape = buf.shape();
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    try buf.toSlice(io, slice);

    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const file = std.c.fopen(path_z.ptr, "wb") orelse {
        log.err("Failed to open file for writing: {s}", .{path});
        return error.CountNotOpenFile;
    };
    defer _ = std.c.fclose(file);

    // 1. Magic
    _ = std.c.fwrite("\x93NUMPY", 1, 6, file);

    // 2. Version (1.0)
    const version = [2]u8{ 1, 0 };
    _ = std.c.fwrite(&version, 1, 2, file);

    // Determine effective dtype for header (bf16 -> f32)
    var effective_dtype = shape.dtype();
    if (effective_dtype == .bf16) effective_dtype = .f32;

    // 3. Prepare Header
    // "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 3, 128, 128), }"
    var header_list = try std.ArrayList(u8).initCapacity(allocator, 128);
    defer header_list.deinit(allocator);

    try header_list.appendSlice(allocator, "{'descr': '");
    switch (effective_dtype) {
        .f32 => try header_list.appendSlice(allocator, "<f4"),
        .i64 => try header_list.appendSlice(allocator, "<i8"),
        .i32 => try header_list.appendSlice(allocator, "<i4"),
        .f16 => try header_list.appendSlice(allocator, "<f2"),
        else => return error.UnsupportedDtype,
    }
    try header_list.appendSlice(allocator, "', 'fortran_order': False, 'shape': (");

    const dims = shape.dims();
    for (dims, 0..) |d, i| {
        if (i > 0) try header_list.appendSlice(allocator, ", ");
        var dim_buf: [32]u8 = undefined;
        const s = try std.fmt.bufPrint(&dim_buf, "{d}", .{d});
        try header_list.appendSlice(allocator, s);
    }
    if (dims.len == 1) try header_list.appendSlice(allocator, ","); // Single dim tuple needs trailing comma

    try header_list.appendSlice(allocator, "), }");

    // Padding
    while ((10 + header_list.items.len + 1) % 64 != 0) {
        try header_list.append(allocator, ' ');
    }
    try header_list.append(allocator, '\n');

    // 4. Write Header Length
    const header_len: u16 = @intCast(header_list.items.len);
    const header_len_le = std.mem.nativeToLittle(u16, header_len);
    const len_bytes = std.mem.asBytes(&header_len_le);
    _ = std.c.fwrite(len_bytes.ptr, 1, 2, file);

    // 5. Write Header
    _ = std.c.fwrite(header_list.items.ptr, 1, header_list.items.len, file);

    // 6. Write Data
    if (shape.dtype() == .bf16) {
        // Convert bf16 to f32
        const src = slice.items(u16);
        const f32_data = try allocator.alloc(f32, src.len);
        defer allocator.free(f32_data);
        for (src, 0..) |s, i| {
            const u32_val = @as(u32, s) << 16;
            f32_data[i] = @bitCast(u32_val);
        }
        _ = std.c.fwrite(std.mem.sliceAsBytes(f32_data).ptr, 1, f32_data.len * 4, file);
    } else {
        const data_bytes = slice.items(u8);
        _ = std.c.fwrite(data_bytes.ptr, 1, data_bytes.len, file);
    }

    log.info("Saved buffer to {s}", .{path});
}
