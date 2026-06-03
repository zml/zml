# `test_tensor_to_device.zig` refactor plan

Two goals:

- **A.** Remove PJRT from this binary. Replace the device buffer + `tm.transferData`
  flow with a host-side `Sink` exposing both `std.Io.Writer` (random-offset writes)
  and `std.Io.Reader` (full sequential read). No `@memcpy` in our code.
- **B.** Remove the verification mirror (`host_ref` + readback compare) from the
  hot path. Replace it with a post-streaming oracle: SHA-256 of the streamed bytes
  vs SHA-256 of the same byte range fetched via HF LFS.

---

## Plan A — Sink behind `std.Io.Writer` / `std.Io.Reader`

### Why random-offset (not append-only)

Plans are emitted in *term order* (by `dst_off`), but the hot loop processes
them in *xorb order*. Terms can alias xorbs out of order, e.g.:

| term | dst_off | xorb |
|------|---------|------|
| 0    | 0       | A    |
| 1    | 100     | B    |
| 2    | 200     | A    |

Processing xorb A first writes `dst_off=0` then `dst_off=200`, skipping the hole
`[100, 200)` that xorb B fills later. So writes are random-offset, not
append-only. A single fixed `Writer` is insufficient; we need a per-call fixed
writer keyed on `dst_off`.

### Sink type

File-scope, ~6 lines:

```zig
const Sink = struct {
    buf: []u8,
    fn writeAt(self: *Sink, off: usize, src: []const u8) !void {
        try std.Io.Writer.fixed(self.buf[off..]).writeAll(src);
    }
    fn reader(self: *Sink) std.Io.Reader { return .fixed(self.buf); }
};
```

- `writeAt` does random-offset writes via a per-call fixed writer; the copy
  happens inside `Writer.fixed(...).writeAll(...)`, not in our code.
- `reader()` returns one `std.Io.Reader` over the full buffer, used by Plan B's
  hasher.

### Allocation

In `main`, after `tensor_size` is known:

```zig
var sink: Sink = .{ .buf = try allocator.alloc(u8, tensor_size) };
defer allocator.free(sink.buf);
```

Replaces both `host_ref` and `readback` allocations.

### Hot loop

Each chunk-match collapses to:

```zig
if (!decompressed) { _ = try xet.decompressChunk(...); decompressed = true; }
try sink.writeAt(p.dst_off + p.bytes_written, chunk_buf[src_off..src_off+take]);
p.bytes_written += take;
```

The look-back `have_pending` machinery and the `@memcpy(host_ref, ...)` line go
away.

### Deletions

- Imports: `pjrt`, `pjrtx`.
- Setup: `Platform.auto`, `platform.deinit`.
- Transfer manager: `createBuffersForAsyncHostToDevice`, `retrieveBuffer`, `tm`,
  `tm.transferData`, the final `transferData(..., true)` flush block.
- Readback: `device_buffer`, `toHostBuffer`, `std.mem.eql` mismatch loop.
- Globals: `host_ref`, `readback`.
- Locals: `have_pending`, `pending_src_off`, `pending_dst_off`, `pending_len`.
- Constant: `MAX_UNCOMP` (cap is now just `try allocator.alloc(...)` failing
  naturally on OOM).

### BUILD

`examples/xet_cas/BUILD.bazel`: drop `"//pjrt"` from the
`test_tensor_to_device` zig_binary deps.

---

## Plan B — Oracle = SHA-256 vs HF LFS, post-hot-path

### Hashing the streamed bytes (symmetric with LFS side)

After the streaming loop succeeds:

```zig
var hasher = std.crypto.hash.sha2.Sha256.init(.{});
var r = sink.reader();
var stage: [64 * 1024]u8 = undefined;
while (true) {
    const n = try r.read(&stage);
    if (n == 0) break;
    hasher.update(stage[0..n]);
}
const xet_digest = hasher.finalResult();
```

### Hashing the LFS range

New helper:

```zig
fn sha256LfsRange(
    client: *std.http.Client,
    repo: RepoInfo,
    auth: []const u8,
    offset: u64,
    len: u64,
) ![32]u8 {
    // GET https://huggingface.co/{ns}/{model}/resolve/{rev}/{filepath}
    // with Range: bytes={offset}-{offset+len-1}.
    // Use default redirect handling (LFS redirects to a CDN that supports Range).
    // Stream the response Reader through Sha256 in 64 KiB chunks.
    // Never materialize the whole range.
}
```

### Compare

```zig
const lfs_digest = try sha256LfsRange(&http_client, repo, auth, tensor_offset, tensor_size);
if (std.mem.eql(u8, &xet_digest, &lfs_digest)) {
    log.info("OK: xet sha256 = lfs sha256 = {x}", .{xet_digest});
} else {
    log.err("MISMATCH: xet={x} lfs={x}", .{ xet_digest, lfs_digest });
    return error.ReadbackMismatch;
}
```

### New deps

- `std.crypto.hash.sha2.Sha256`
- One small HTTP helper (similar shape to existing `httpRangeGetIntoSlot`, but
  follows redirects and streams into a `Sha256` instead of into a fixed slot).

No PJRT, no devices, no test platform setup.

---

## Net effect

- Hot loop body: ~3 lines per chunk-match.
- File loses ~100–150 lines (all PJRT plumbing, look-back state, mismatch loop,
  two big tensor-sized buffers → one).
- Correctness is verified end-to-end against the authoritative LFS bytes,
  outside the streaming hot path.
