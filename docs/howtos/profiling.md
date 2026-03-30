# Profiling ZML Programs

ZML has two profiling layers:

- host tracing through `zml.tracer.scope(...)`
- backend profiling through `zml.Platform.profiler(...)`

The host tracing path is always based on `TraceMe`. Depending on the target
platform, the tracing bridge also emits device-visible annotations:

- CUDA on Linux: NVTX ranges
- ROCm on Linux: ROCTx ranges
- macOS: `os_signpost` intervals for Instruments

When backend profiling is enabled, ZML writes both the raw `XSpace` protobuf
and a streamed Perfetto / trace-viewer JSON file.

## The generated artifacts

By default, a profiling session is written under:

```text
/tmp/xprof/plugins/profile/profiling/
```

The relevant files are:

- `profiling.xplane.pb`: raw `XSpace` protobuf
- `profiling.trace.json`: Perfetto / trace-viewer JSON export

These paths come from `zml/profiling/profiler.zig` and can be changed through
`zml.Platform.ProfilerOptions`.

## Add trace scopes in Zig

Use `zml.tracer.scope(name, metadata)` to bracket the code you want to inspect.

```zig
const zml = @import("zml");

var trace = try zml.tracer.scope("llm.compile", .{
    .seqlen = args.seqlen,
    .interactive = args.prompt == null,
});
defer trace.end();

try models.LoadedModel.compile(...);
```

Use `.{}` when there is no metadata:

```zig
var trace = try zml.tracer.scope("llm.load_buffers", .{});
defer trace.end();
```

Metadata is encoded into the `TraceMe` scope name, so keep it small and
high-signal.

## Record a ZML profile from code

If you want ZML to collect a backend profile and export `profiling.trace.json`,
create a profiler from the current platform:

```zig
const platform: *zml.Platform = try .auto(allocator, io, .{});

var profiler = try platform.profiler(allocator, io, .defaults);
defer profiler.deinit();

try profiler.start();
defer {
    if ((profiler.stop() catch unreachable)) |profile| {
        std.log.info("Profile dumped: {s} and {s}", .{
            profile.protobuf_path,
            profile.perfetto_path,
        });
    }
}
```

This is the normal ZML flow:

1. start the PJRT backend profiler
2. record local `TraceMe` scopes
3. merge the local host trace into the backend `XSpace`
4. write `profiling.xplane.pb`
5. stream `profiling.trace.json`

If the backend profiler is unavailable, ZML still keeps the local host trace
path and logs a warning.

## CUDA on Linux with Nsight Systems

ZML provides a Bazel run config for `nsys`:

```bash
bazel run --config=nsys //examples/benchmark:benchmark
```

Or for a real model:

```bash
bazel run --config=nsys //examples/llm -- \
    --model=/path/to/model \
    --prompt="Write a story about a cat"
```

The wrapper target is `//tools:nsys`. By default it runs:

```text
sudo -E nsys profile ...
```

with CUDA, cuBLAS, cuSPARSE, cuDNN, NVTX, syscall, and OS runtime tracing
enabled.

If you need to install Nsight Systems manually, use NVIDIA's official download
page: <https://developer.nvidia.com/nsight-systems/get-started>. For the server
or target machine where `nsys` will run, choose the `Linux CLI only` package.
For the client machine where you open and inspect reports, choose the full host
package for your OS. NVIDIA's installation guide is here:
<https://docs.nvidia.com/nsight-systems/2025.4/InstallationGuide/index.html>.

Useful overrides:

- `ZML_PROFILE_NO_SUDO=1`: run `nsys` without `sudo`
- `ZML_NSYS_ARGS='...'`: replace the default `nsys` arguments

Example:

```bash
ZML_PROFILE_NO_SUDO=1 bazel run --config=nsys //examples/benchmark:benchmark
```

### Important interaction with `zml.profiler`

The `nsys` wrapper sets:

```text
SKIP_PJRT_PROFILER=true
```

That tells `zml/profiling/profiler.zig` to skip nested PJRT backend profiling. This is
intentional. Under `nsys`, you normally want:

- the external Nsight session
- local `TraceMe` / NVTX annotations from your scopes

and not a second backend profile session layered on top.

## ROCm on Linux with `rocprofv3`

ZML also provides a Bazel run config for ROCm:

```bash
bazel run --config=rocprofv3 //examples/benchmark:benchmark
```

Or:

```bash
bazel run --config=rocprofv3 //examples/llm -- \
    --model=/path/to/model \
    --prompt="Write a story about a cat"
```

The wrapper target is `//tools:rocprofv3`.

This path is intentionally sandboxed inside the repository's ROCm packaging:

- it launches the Bazel-runfiles copy of `rocprofv3`
- it passes `--rocm-root` pointing at the sandboxed ROCm runtime

So it does not depend on a host-installed `rocprofv3` layout.

The default wrapper enables system, runtime, HIP, kernel, memory copy,
scratch-memory, HSA core, and marker tracing, and writes `pftrace` output.

Useful overrides:

- `ZML_ROCPROFV3_ARGS='...'`: replace the default `rocprofv3` arguments

### Important interaction with `zml.profiler`

Like the `nsys` wrapper, the `rocprofv3` wrapper sets:

```text
SKIP_PJRT_PROFILER=true
```

That means:

- the external `rocprofv3` session owns backend/device profiling
- ZML still records local host `TraceMe` scopes
- ROCTx ranges from `zml.tracer.scope(...)` stay visible as device annotations

## macOS Instruments

On macOS, `zml.tracer.scope(...)` emits `os_signpost` intervals. These show up
in Instruments, especially in the `Points of Interest` template.

Build the binary first:

```bash
bazel build //examples/llm:llm
```

Then either launch it from the Instruments UI or use `xctrace`:

```bash
xctrace record \
    --template 'Points of Interest' \
    --output /tmp/llm.trace \
    --launch -- \
    /home/you/zml/bazel-bin/examples/llm/llm \
    --model=/path/to/model \
    --prompt="Write a story about a cat"
```

Use full Xcode for `xctrace`. Command Line Tools alone are not enough.

## Open the exported trace

The `profiling.trace.json` file can be opened in Perfetto or in tools that still
understand the trace-event JSON format.

Typical flow:

1. run a ZML program with `zml.Platform.profiler(...)`
2. wait for `profiling.trace.json`
3. open the JSON in Perfetto

The converter is implemented in Zig and streams JSON directly from `XSpace`,
which keeps large traces much more manageable than the old in-memory path.

## Platform summary

- Plain ZML profiling: backend `XSpace` profile plus merged local host trace
- CUDA + `--config=nsys`: external Nsight session plus local `TraceMe` / NVTX
- ROCm + `--config=rocprofv3`: external `rocprofv3` session plus local `TraceMe` / ROCTx
- macOS Instruments: local `TraceMe` scopes exposed as `os_signpost`

## Troubleshooting

- If you run under `--config=nsys` or `--config=rocprofv3`, PJRT profiling is
  skipped on purpose because the wrappers set `SKIP_PJRT_PROFILER=true`.
- If you do not see device annotations, first check whether you are on a
  supported platform:
  - CUDA/Linux for NVTX
  - ROCm/Linux for ROCTx
  - macOS for Instruments signposts
- If you only need host scopes, `zml.tracer.scope(...)` works independently of
  whether backend profiling succeeds.
