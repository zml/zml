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

## Profiling architecture

ZML profiling is split into four layers:

1. `zml.tracer.scope(...)` in Zig marks lexical regions of interest
2. `zml/profiling/traceme_lib.cc` bridges those scopes into platform-specific
   tracing backends
3. `zml.Platform.profiler(...)` manages a profiling session and collects profile
   data from PJRT
4. `tools/xspace_to_perfetto` converts the final `XSpace` protobuf into a
   streamed trace-event JSON file

The important architectural split is:

- `zml.tracer.scope(...)` is for instrumentation inside your code
- `zml.Platform.profiler(...)` is for session lifecycle and artifact generation

You can use scopes without a profiler session, but you only get exported
`profiling.xplane.pb` and `profiling.trace.json` when a profiler session is
started.

### Scope instrumentation path

`zml.tracer.scope(...)` is the lowest-level API you use directly in Zig code.
Each scope creates a `TraceMe` region, and the C bridge adds extra
platform-visible annotations when supported:

- CUDA/Linux: NVTX ranges
- ROCm/Linux: ROCTx ranges
- macOS: `os_signpost` intervals

Metadata passed to `zml.tracer.scope(...)` is encoded into the `TraceMe` scope
name. That is why the API only accepts small scalar or string-like fields: it is
designed to produce compact trace labels rather than structured protobuf
payloads.

### Profiler session path

`zml.Platform.profiler(...)` builds a `tensorflow.ProfileOptions` protobuf from
`zml.Platform.ProfilerOptions` and asks the active PJRT backend for a profiler
session.

At `start()` time, ZML does two distinct things:

- it tries to start the backend PJRT profiler session
- it configures local host-side `TraceMeRecorder` capture when enabled

At `stop()` time, ZML:

1. stops the PJRT profiler session if one exists
2. collects the backend `XSpace` protobuf
3. merges locally recorded host `TraceMe` events into the host plane
4. writes the final protobuf to disk
5. streams trace JSON from that final protobuf

This separation is important because host scopes and backend profiler output are
not produced by the same subsystem. ZML joins them only at the end of the
session.

### Export path

The final export path is intentionally one-way:

- source data format: `XSpace` protobuf
- derived export format: trace-event JSON for Perfetto / trace viewer

ZML keeps `profiling.xplane.pb` as the source of truth and generates
`profiling.trace.json` from it afterwards. The converter is implemented in Zig
and streams JSON directly from the protobuf instead of materializing the whole
JSON document in memory first.

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

## Annotate producer/consumer contexts

Both `span_at` and `span_at_for` emit source locations as `_src=file:line:column`,
matching the backend source-metadata key. `_src` is reserved by these helpers.

Use `span_at_for` to combine source metadata with a typed correlation context:

```zig
const context: zml.tracer.Context = .init(.generic);

{
    var span = zml.tracer.span_at_for(@src(), "schedule", .{}, .{
        .producer = context,
    });
    defer span.end();
    // Pass a copy of context along with the scheduled work.
}

// In the worker receiving that context:
{
    var span = zml.tracer.span_at_for(@src(), "execute", .{}, .{
        .consumer = context,
    });
    defer span.end();
    // Perform the work.
}
```

`Context.init(namespace)` obtains an activity ID from the executable's TraceMe
generator. `ContextType` mirrors XLA's named namespaces, including `.generic`,
`.pjrt_library_call`, and `.threadpool_event`. Plugins have separate generators;
when crossing an API boundary, explicitly pass the same namespace and ID.
Existing backend contexts can be represented with
`Context{ .namespace = .pjrt_library_call, .id = backend_id }`.

The last argument is a `SpanContext`: its optional `producer` and `consumer`
fields can both be set for a span which receives work and schedules new work.
Use separate contexts for those two handoffs. The helper encodes the required
integer `_pt`/`_p` and `_ct`/`_c` fields in the same metadata section as `_src` and
the ordinary metadata. Those four keys are reserved in `span_at_for` metadata.
Existing `span` and `span_at` calls are unchanged. These annotations describe
relationships; the caller remains responsible for scheduling and synchronization.

`zml.exe.call` creates a `.pjrt_library_call` producer context and passes its ID
through the PJRT execution arguments' profiler extension. Supporting XLA plugins
record a matching `PJRT_LoadedExecutable_Execute` consumer, so the exported trace
connects the host call to the backend entry point. The extension is only attached
when the plugin advertises profiler support. With asynchronous execution, this
arrow still describes the handoff even if backend work outlives the host span;
waiting and buffer ownership are unaffected.

`zml.compiler.pjrt_compile` uses the same mechanism to connect to the backend's
`PJRT_Client_Compile` consumer. The correlation ID is passed by value through
`io.async`, and the profiler extension is constructed on the compilation worker
for the duration of the C API call.
JSON flow endpoints bind to the enclosing slice at their timestamp. If a backend
entry and its nested operation start at exactly the same timestamp, Perfetto can
attach the arrow to that nested slice instead of the entry slice.

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

## Neuron Profiling

Use `--config=neuron-profile` with the command you already run. The wrapper
prints the profile directory when the program exits.

```bash
NEURON_RT_VISIBLE_CORES=0,1 \
bazel run --config=neuron-profile //examples/llm -- \
    --model=/path/to/model \
    --prompt="Explain mixture-of-experts routing in one paragraph." \
    --seqlen=128 \
    --backend=nki \
    --topk=1
```

The default profile root is `/tmp/zml-neuron-profile`. Each run contains:

```text
/tmp/zml-neuron-profile/<run>/
  compile/
  execution/
```

Start the Neuron Explorer server in one terminal:

```bash
bazel run //tools/neuron:server
```

Open <http://127.0.0.1:3001>. When working through SSH, forward ports `3001`
and `3002`.

With the server running, ingest the run from another terminal:

```bash
bazel run //tools/neuron:ingest -- /tmp/zml-neuron-profile/<run>
```

Create a quick summary from one run when you want a file artifact:

```bash
bazel run //tools/neuron:summary-json -- /tmp/zml-neuron-profile/<run>
bazel run //tools/neuron:summary-txt -- /tmp/zml-neuron-profile/<run>
bazel run //tools/neuron:summary-perfetto -- /tmp/zml-neuron-profile/<run>
```

Each summary command writes beside the run and prints the output path:
`summary.json`, `summary.txt`, or `summary.pftrace`.

## CPU backend execution traces

CPU plugins without the PJRT profiler extension only produce ZML's local host
spans. In particular, model `forward` spans measure graph construction, while
`zml.exe.call` brackets execution and any requested wait. They cannot reveal
the work performed inside a separately loaded plugin.

To collect CPU backend events, build the plugin from source with the profiler
extension enabled by this repository's XLA patch:

```bash
bazel run //examples/phonon --//platforms/cpu:build_from_source=true -- \
    --no-play "Bonjour ZML"
```

The first build compiles the CPU backend and its dependencies and can take a
long time. The default remains the downloaded plugin; the flag selects the
source-built plugin through the usual runfiles sandbox.
The source build follows the selected compilation mode. For performance
measurements, add `--config=debug` to optimize the C++ backend while retaining
Zig debug builds; an unoptimized backend is useful for inspection but its
timings should not be compared directly with the prebuilt plugin.
Debug YNNPACK builds also log individual kernel calls; add
`--define=ynn_log_level=warning` when building to suppress that logging while
keeping profiler events enabled.

The profiler extension starts and collects the recorder inside the plugin.
ZML's existing profiling session merges that backend XSpace with the local host
trace and exports both to `profiling.trace.json`. Open that file in Perfetto
and look for CPU runtime events such as `ThunkExecutor::Execute`, alongside
the ZML spans. A thunk is a compiled runtime task; compiler fusion means it
need not correspond to one original tensor operation or model module.

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

Producer/consumer annotations (`_pt`/`_p` and `_ct`/`_c`) are exported as
Perfetto flow arrows between the corresponding slices. Select a connected
slice to inspect its incoming and outgoing flows. These links describe work
handoffs recorded by the runtime, such as scheduling a task on another thread;
they do not represent every tensor dependency in the model.

The converter matches both context type and ID, supports multiple consumers
per producer, and ignores endpoints whose matching producer was not captured.
It uses the most recent preceding producer if an ID is reused. Existing
profiles can be converted again without rerunning inference:

```bash
bazel run //tools/xspace_to_perfetto -- \
    /tmp/xprof/plugins/profile/profiling/profiling.xplane.pb \
    /tmp/phonon-with-flows.trace.json
```

## Platform summary

- Plain ZML profiling: backend `XSpace` profile plus merged local host trace
- CUDA + `--config=nsys`: external Nsight session plus local `TraceMe` / NVTX
- ROCm + `--config=rocprofv3`: external `rocprofv3` session plus local `TraceMe` / ROCTx
- macOS Instruments: local `TraceMe` scopes exposed as `os_signpost`

## Troubleshooting

- If you run under `--config=nsys`, `--config=rocprofv3`, or
  `--config=neuron-profile`, PJRT profiling is skipped on purpose because the
  wrappers set `SKIP_PJRT_PROFILER=true`.
- If you do not see device annotations, first check whether you are on a
  supported platform:
  - CUDA/Linux for NVTX
  - ROCm/Linux for ROCTx
  - macOS for Instruments signposts
- If you only need host scopes, `zml.tracer.scope(...)` works independently of
  whether backend profiling succeeds.
