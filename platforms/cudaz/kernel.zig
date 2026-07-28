export fn main(buffers: [*]*const anyopaque, buffer_len: usize) callconv(.nvptx_kernel) void {
    _ = buffers;
    _ = buffer_len;
}
