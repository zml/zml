# Native XET download for ZML (via xet-core C-API)

Downloads HuggingFace XET-backed files by driving the mature **xet-core** (Rust)
client through its C-API, wired into the existing `hf://` VFS provider.

## Why the C-API

XET files reconstruct client-side (fetch compressed chunks from the CAS, LZ4/BG4
decompress, reassemble). Doing that fast needs adaptive concurrency, bounded
streaming, decompress-once scheduling, connection pooling and an on-disk chunk
cache; xet-core already has all of it. On an m6i.2xlarge, cold download of
`model-00001-of-00004.safetensors` (4.98 GB):

| path | MB/s |
|------|------|
| Zig -> C-API -> xet-core (raw), `HF_XET_HIGH_PERFORMANCE=1` | ~1070 |
| hf-xet (Python, same Rust core) | ~1030 |
| ZML `parallel_read.zig` (plain parallel HTTPS) | ~1000 |
| from-scratch Zig reimpl (jedisct1/zig-xet) | ~386 |

## How it is wired

- `xet_hub.zig`: HF Hub HTTPS/JSON (CAS read-token exchange, file listing).
- `xet_capi.zig`: `extern` declarations of the `hf_xet.h` C ABI (Session +
  downloadToPath). Symbols are declared inline, so only linking `libxet_capi.a`
  is needed, no header include path.
- `xet.zig`: `openRemote` -> `RemoteFile.readRange` (lazy range reads) and
  `downloadFile` (eager whole-file).
- `hf.zig`: `performRead` detects XET-backed files on first access and, for
  those, reconstructs each requested byte range on demand through the C-API
  stream API (`readRange`), which fetches only the covering xorbs and serves
  overlapping/repeated ranges from the chunk cache. Non-XET files fall back to
  the existing `resolve` range-GET path. So `hf://` XET files go through
  xet-core transparently, keeping ZML's lazy positional-read model.

Set `HF_XET_HIGH_PERFORMANCE=1` for peak concurrency.

## Verification (built + run in ZML's Bazel build)

Built `//examples/io:playground` with Zig 0.16.0 + `libxet_capi` from xet-core's
`assaf/c-api` branch (`cargo build --release -p xet_capi`), on an m6i.2xlarge.

- `xet_capi.zig` compiles and `libxet_capi.a` links into the real ZML binary,
  and `hf.zig` reaches it.
- `playground cp hf://.../model-00001-of-00004.safetensors file://…`
  reconstructed the full 4.98 GB file (exact size) through the integrated path.
  Throughput on that (worst-case, sequential 16 MB reads) `cp` pattern:

  | mode | MB/s |
  |------|------|
  | lazy `readRange` (this integration) | ~36 |
  | eager download-to-cache | ~228 |
  | current `resolve` range-GET (baseline) | ~23 |
  | raw parallel xet download (no `cp` loop) | ~1070 |

  Lazy reconstructs a fresh range per 16 MB read, so a sequential full-file copy
  pays per-read reconstruction overhead; its win is partial/random reads (a few
  tensors out of a shard) where it avoids pulling the whole file, which is the
  point of keeping the lazy model. Eager wins for full-file copies. A real model
  load (parallel positional reads) sits between these, not on the `cp` worst
  case.

## Known issues / follow-ups

- **Pre-existing crash on exit (not from this change):** `examples/io cp` aborts
  in ZML's own `file://` provider (`file.zig` `getFileHandle` -> `fileClose`,
  from `examples/io/main.zig:96`) after the file is fully written. The stack is
  entirely in the local-file provider close path, unrelated to XET.
- **Cache location / lifecycle:** files are materialized under `/tmp/zml-xet-*`
  and never evicted. Should live under xet-core's own cache dir and be reused.
- **Eager whole-file download** vs the lazy positional-read model; a design
  decision for large shards.
- **Bazel `@xet_capi`** external repo must be declared (prebuilt per platform or
  via rules_rust). The Rust static lib is large and cross-compiled per target,
  which is the one real cost of this approach.
