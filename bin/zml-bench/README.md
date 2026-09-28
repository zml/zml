# zml-bench

A Zig + libvaxis terminal dashboard for benchmarking an OpenAI-compatible
streaming chat completions endpoint. Send concurrent requests, follow each
response in full-screen output, and inspect live timing distributions.

Run from the repository root:

```sh
bazel run //bin/zml-bench -- \
  --endpoint http://127.0.0.1:8000/v1 --model zml_model
```

Use `OPENAI_API_KEY` for bearer authentication when needed. HTTP and HTTPS are
supported. The endpoint may be a host URL, a base URL ending in `/v1`, or the
full URL ending in `/chat/completions`. Redirects are rejected.

The default batch has 16 concurrent requests (maximum 256). There is no
benchmark timeout: requests run until they finish, fail, or you cancel them.
Settings are locked during a run; results remain until the next batch.

## Request options

Core settings are on the dashboard. Click **Options** or press **Ctrl-O** for
additional controls. All settings also have command-line flags; use `--help`.

| Option | Behavior |
| --- | --- |
| `--temperature N` | Nonnegative temperature; blank omits it and uses the server default |
| `--thinking VALUE` | Blank/auto omits reasoning_effort; off sends "none", on sends "high"; effort names or integer budgets 0–100 are also accepted |
| `--reasoning-effort VALUE` | Alias for thinking |
| `--max-tokens N` | Output limit; blank uses the server default |
| `--max-completion-tokens N` | Supersedes max_tokens in llmd |
| `--system TEXT` | Optional system message before the user prompt |
| `--stop TEXT` | Plain stop string or JSON array of up to four strings |
| `--tools JSON` | JSON array of tool definitions |
| `--documents JSON` | JSON array of documents for llmd's chat template |

Thinking support and accepted effort names depend on the model/template.
llmd's DeepSeek template accepts none/low/high/max and integer budgets 0–100.
Tools and documents are passed to the endpoint; the benchmark does not execute
tool calls. The inspected llmd version parses stop sequences but does not
enforce them. Top-p, seed, and frequency/presence penalties are not exposed
because that llmd chat endpoint does not implement them.

## Interaction

| Key | Action |
| --- | --- |
| Click a field | Focus it and position the cursor |
| Click Send batch / Ctrl-S | Start a batch |
| Click a response pane | Open its full-screen live output |
| Click Options / Ctrl-O | Open additional request options |
| Click Report / Ctrl-R | Open live/final timing distributions |
| Tab / Shift-Tab | Move between visible settings |
| Enter on Send batch | Send a batch |
| Esc | Return from output/options/report; on the dashboard, cancel requests |
| PgUp / PgDn, mouse wheel | Scroll request cards or full-screen output |
| Up / Down in full-screen output | Scroll one line |
| Home / End in full-screen output | Beginning / follow live output |
| Click Back / Follow / Stop in full-screen output | Return / follow / cancel |
| Ctrl-A / Ctrl-E | Beginning / end of an input |
| Ctrl-U / Ctrl-K | Clear input before / after the cursor |
| Ctrl-C | Cancel outstanding work and quit |

The dashboard uses two columns at 102 or more terminal columns, one below that,
and compact settings below 34 rows. Minimum dashboard/options/report size is
68 × 24; full-screen output needs 48 × 12. Scrolling up pauses following;
End or Follow resumes it. Returning to the dashboard leaves requests running.

## Metrics

All timings use the client's monotonic clock. Request timings start when that
request's worker begins and include connection setup and server queueing.

- **TTFT:** time to the first nonempty content, reasoning, or tool delta.
  Role-only, usage-only, and finish-only events do not count.
- **First answer:** time to the first nonempty content delta, after any thinking.
- **ITL/chunk:** gaps between nonempty SSE events. llmd can combine or split
  tokens into events, so this measures observed chunk arrival latency, not exact
  per-token latency. An event containing several output parts counts once.
- **TPOT:** (last output arrival − first output arrival) / (output tokens − 1).
  Requires at least two output chunks and two tokens; excludes TTFT. Chunking
  also limits its precision. **Decode tok/s** is the reciprocal of TPOT.
- **Request latency:** duration to completion. The report distribution includes
  only successfully completed requests; each request's JSON record also retains
  its duration when failed or canceled.
- **Aggregate tok/s:** total output tokens / elapsed batch time.
- **Average / request:** mean of each request's output tokens / its elapsed
  duration, including connection setup and TTFT.
- **Requests/s:** successful completions / elapsed batch time.
- **Token counts:** server-reported completion_tokens and prompt_tokens when
  available. Otherwise output tokens use one token per four UTF-8 output bytes,
  including reasoning/tool output. A `~` marks estimates in the UI; reports
  carry `estimated`. Prompt counts include only requests that reported usage.
- Finished request rates and batch duration freeze. Partial failed/canceled
  output contributes to throughput and observed TTFT/ITL/TPOT statistics.

The Report screen and JSON include mean, minimum, p50, p95, p99, maximum, and
sample count (minimum is JSON-only). Request percentiles use linear interpolation.
ITL uses a bounded logarithmic histogram per request: mean/min/max are exact;
percentiles use bucket upper bounds, approximately within 4.5% + 1 microsecond
for gaps below roughly 68 minutes. Larger gaps share an overflow bucket, whose
percentile estimate is the observed maximum. Aggregate ITL merges all observed
gaps rather than averaging request percentiles.

Each request asks for `stream_options.include_usage`. The server must accept
this option and support streaming chat completions. Both `[DONE]` and EOF after
a finish_reason complete a request; earlier disconnects fail. HTTP errors,
malformed events, and streaming API errors appear in the affected request pane.

Full output stays in memory until the next batch or exit for scrolling back.
Cards show a recent tail. Individual SSE lines are limited to 64 KiB and
assembled events to 256 KiB. User/system prompts are limited to 16 KiB each.

## JSON reports

For scripts, run one batch without a terminal and print a JSON report:

```sh
bazel run //bin/zml-bench -- --headless \
  --endpoint http://127.0.0.1:8000/v1 --model zml_model \
  --batch 8 --max-completion-tokens 256 --temperature 0 --thinking off \
  --report /tmp/benchmark.json
```

The report contains a version, endpoint, request payload, aggregate summary,
per-request timing/token/error records, and timing notes. It excludes the
authentication header. Headless mode prints the report to stdout and exits
unsuccessfully if a request fails.

`--report PATH` also works in the TUI: the latest batch report is saved when it
finishes or is canceled, including when quitting. Each subsequent batch
overwrites that file. Without the flag, inspect results using Report/Ctrl-R.

## Validation

```sh
bazel build //bin/zml-bench
bazel test //bin/zml-bench:test //bin/zml-bench:tui_test //bin/zml-bench:integration_test
```

Tests cover payload types, timing aggregation, concurrent fragmented SSE, UTF-8,
usage, errors, redirects, delayed headers/body, JSON export, mouse navigation,
full-output scrolling, and renderer ownership while output storage changes.
