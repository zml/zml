#!/usr/bin/env python3
"""Analyze GPU kernel breakdown from a raw Perfetto/XLA trace JSON file.

Handles overlapping events that occur in XLA/CUDA traces (e.g. prepare_varlen_num_blocks
overlapping with FlashAttention kernels by ~1-2 us). On a GPU stream, only one kernel
runs at a time, so overlaps are trace artifacts. This script de-overlaps by sweeping
through events sorted by start time and clipping any portion that falls within an
already-accounted interval.

Usage:
    python3 analyze_trace.py <trace.json>
    python3 analyze_trace.py <trace.json> --details
    python3 analyze_trace.py <trace.json> --gpu-tid 7
    python3 analyze_trace.py <trace.json> --fix                  # writes <input>_fixed.trace.json
    python3 analyze_trace.py <trace.json> --fix -o fixed.json    # custom output path
"""

import json
import sys
import argparse
from collections import defaultdict


def load_trace_raw(path):
    """Load trace JSON, returning the raw top-level structure."""
    with open(path) as f:
        return json.load(f)


def get_events(data):
    """Extract the events list from the raw trace data."""
    if isinstance(data, list):
        return data
    return data.get("traceEvents", [])


def fix_overlaps(events, gpu_tid):
    """Fix overlapping X events in-place, only on GPU stream tracks.

    CPU tracks legitimately have nested/overlapping spans (e.g. a parent
    "inference" span containing child "compile" and "execute" spans), so
    we leave those untouched. GPU streams execute kernels sequentially, so
    overlaps there are trace artifacts from XLA's CUDA profiler.

    Runs multiple passes to handle nested call hierarchies where fixing one
    overlap may expose the next level.
    """
    total_fixed = 0

    for pass_num in range(20):  # safety limit
        # Collect X events on GPU tracks only, indexed by position in events list
        gpu_x = []
        for i, e in enumerate(events):
            if (e.get("ph") == "X" and "ts" in e and "dur" in e
                    and e.get("tid") == gpu_tid):
                gpu_x.append((i, e))

        # Sort by timestamp, then by duration descending (parent before child)
        gpu_x.sort(key=lambda x: (x[1]["ts"], -x[1]["dur"]))

        pass_fixed = 0
        for j in range(len(gpu_x) - 1):
            idx_a, a = gpu_x[j]
            idx_b, b = gpu_x[j + 1]

            a_end = a["ts"] + a["dur"]
            b_start = b["ts"]
            overlap = a_end - b_start

            if overlap > 0:
                if b["dur"] == 0:
                    # Zero-duration event inside a parent: move to parent end
                    events[idx_b]["ts"] = a_end
                elif a["dur"] <= b["dur"]:
                    # Shorten a so it ends at b's start
                    events[idx_a]["dur"] = max(0, b_start - a["ts"])
                else:
                    # Delay b so it starts at a's end
                    events[idx_b]["ts"] = a_end
                    events[idx_b]["dur"] = max(0, b["dur"] - overlap)
                pass_fixed += 1

        total_fixed += pass_fixed
        if pass_fixed == 0:
            break

    return total_fixed


def find_gpu_stream_tid(events):
    """Find the GPU compute stream tid.

    Strategy:
    1. Look for thread_name metadata naming a CUDA stream.
    2. Fallback: pick the tid with the most 'X' (complete) events whose names
       look like GPU kernels (fusion_, nvjet_, flash, etc.).
    """
    # Try metadata events first
    stream_tids = []
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tname = e.get("args", {}).get("name", "")
            if "stream" in tname.lower():
                stream_tids.append((e.get("tid"), tname))

    if stream_tids:
        # If multiple streams, pick the one with most X events
        tid_counts = defaultdict(int)
        for e in events:
            if e.get("ph") == "X":
                tid_counts[e.get("tid")] += 1
        best = max(stream_tids, key=lambda t: tid_counts.get(t[0], 0))
        print(f"GPU stream: tid={best[0]} ({best[1]})")
        return best[0]

    # Fallback: tid with most X events that look like GPU kernels
    tid_gpu_counts = defaultdict(int)
    gpu_patterns = ["fusion_", "nvjet_", "flash", "gemm", "memset", "memcpy",
                    "loop_", "input_", "wrapped_", "reduce_", "scatter_"]
    for e in events:
        if e.get("ph") == "X":
            name = (e.get("name") or "").lower()
            if any(p in name for p in gpu_patterns):
                tid_gpu_counts[e.get("tid")] += 1

    if tid_gpu_counts:
        best_tid = max(tid_gpu_counts, key=tid_gpu_counts.get)
        print(f"GPU stream (heuristic): tid={best_tid} ({tid_gpu_counts[best_tid]} GPU-like events)")
        return best_tid

    # Last resort: tid with most events
    tid_counts = defaultdict(int)
    for e in events:
        if e.get("ph") == "X":
            tid_counts[e.get("tid")] += 1
    if tid_counts:
        best_tid = max(tid_counts, key=tid_counts.get)
        print(f"GPU stream (fallback): tid={best_tid} ({tid_counts[best_tid]} events)")
        return best_tid

    return None


def categorize_kernel(name):
    """Classify a GPU kernel name into a high-level category.

    Order matters: more specific patterns are checked first to avoid
    'gemm_fusion' matching as Fusion instead of GEMM.
    """
    # FlashAttention kernels (check before GEMM since some use cutlass)
    if any(k in name for k in ["FlashAttn", "flash_fwd", "flash_bwd",
                                "cutlass::device_kernel<flash", "flash::",
                                "prepare_varlen_num_blocks"]):
        return "FlashAttn"
    # GEMM / matmul kernels
    if any(k in name for k in ["nvjet_", "gemm_fusion", "gemm", "cublas"]):
        return "GEMM"
    # Memory operations
    if any(k in name.lower() for k in ["memset", "memcpy", "nccl"]):
        return "Memops"
    # XLA fused kernels
    if any(name.startswith(p) for p in ["fusion_", "loop_", "input_",
                                         "wrapped_", "reduce_", "scatter_"]):
        return "Fusion"
    return "Other"


def deoverlap_and_analyze(gpu_events):
    """De-overlap GPU events and compute per-category breakdown.

    Algorithm (greedy sweep with high-water mark):
    1. Sort events by (start_time, -duration). When two events start at the
       same time, the longer one comes first — it's the "real" kernel.
    2. Sweep left-to-right tracking a high-water mark (hwm) = end of the
       latest committed interval.
    3. For each event:
       - Starts at or after hwm → no overlap, use full duration.
       - Starts before hwm but ends after → partial overlap, clip start to hwm.
       - Fully contained within hwm → skip (0 contribution).

    This correctly handles the XLA pattern where a short metadata event
    (prepare_varlen_num_blocks, ~1.5 us) starts just before a long kernel
    (FlashAttention, ~35 ms). The short event gets its tiny duration; the
    long event gets clipped by ~1.5 us — negligible error.
    """
    gpu_events.sort(key=lambda e: (e["ts"], -e["dur"]))

    intervals = []  # (effective_start, effective_end, category, original_name)
    hwm = 0
    n_clipped = 0
    n_dropped = 0

    for e in gpu_events:
        ts = e["ts"]
        dur = e["dur"]
        end = ts + dur
        cat = categorize_kernel(e.get("name", ""))

        if ts >= hwm:
            # No overlap
            intervals.append((ts, end, cat, e.get("name", "")))
            hwm = end
        elif end > hwm:
            # Partial overlap: clip the beginning
            intervals.append((hwm, end, cat, e.get("name", "")))
            n_clipped += 1
            hwm = end
        else:
            # Fully contained in a previous event: skip
            n_dropped += 1

    return intervals, n_clipped, n_dropped


def print_summary(intervals, n_clipped, n_dropped, show_details=False):
    if not intervals:
        print("No GPU events found.")
        return

    # Aggregate by category
    cat_time_us = defaultdict(float)
    cat_count = defaultdict(int)
    cat_kernels = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))  # cat -> name -> [count, us]

    for start, end, cat, name in intervals:
        dur = end - start
        cat_time_us[cat] += dur
        cat_count[cat] += 1
        cat_kernels[cat][name][0] += 1
        cat_kernels[cat][name][1] += dur

    total_busy_us = sum(cat_time_us.values())
    wall_us = intervals[-1][1] - intervals[0][0]
    idle_us = wall_us - total_busy_us

    # Print overlap resolution stats
    print(f"Overlap resolution: {n_clipped} clipped, {n_dropped} fully contained (dropped)")
    print()

    # Category table
    print(f"{'Category':<12} {'Count':>7} {'Time (ms)':>11} {'%':>7}")
    print("-" * 40)
    for cat in sorted(cat_time_us, key=cat_time_us.get, reverse=True):
        ms = cat_time_us[cat] / 1000
        pct = 100.0 * cat_time_us[cat] / wall_us
        print(f"{cat:<12} {cat_count[cat]:>7} {ms:>11.1f} {pct:>6.1f}%")

    print("-" * 40)
    print(f"{'Busy':<12} {'':>7} {total_busy_us/1000:>11.1f} {100*total_busy_us/wall_us:>6.1f}%")
    print(f"{'Idle':<12} {'':>7} {idle_us/1000:>11.1f} {100*idle_us/wall_us:>6.1f}%")
    print(f"{'Wall':<12} {'':>7} {wall_us/1000:>11.1f}")
    print()
    print(f"GPU utilization: {100*total_busy_us/wall_us:.1f}%")

    # Detailed per-kernel breakdown within each category
    if show_details:
        print()
        print("=" * 70)
        print("Per-kernel breakdown")
        print("=" * 70)
        for cat in sorted(cat_time_us, key=cat_time_us.get, reverse=True):
            print(f"\n--- {cat} ({cat_time_us[cat]/1000:.1f} ms) ---")
            kernels = cat_kernels[cat]
            # Sort by total time descending
            for name, (cnt, us) in sorted(kernels.items(), key=lambda x: -x[1][1])[:15]:
                avg_us = us / cnt if cnt else 0
                print(f"  {cnt:>5}x  {us/1000:>9.1f} ms  (avg {avg_us:.1f} us)  {name[:80]}")
            remaining = len(kernels) - 15
            if remaining > 0:
                print(f"  ... and {remaining} more kernel types")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GPU kernel breakdown from a Perfetto/XLA trace JSON file."
    )
    parser.add_argument("trace", help="Path to trace JSON file (raw or fixed)")
    parser.add_argument("--gpu-tid", type=int, default=None,
                        help="Thread ID of the GPU compute stream (auto-detected if omitted)")
    parser.add_argument("--details", action="store_true",
                        help="Show per-kernel breakdown within each category")
    parser.add_argument("--fix", action="store_true",
                        help="Fix overlapping events and write a corrected trace file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path for fixed trace (default: <input>_fixed.trace.json)")
    args = parser.parse_args()

    print(f"Loading {args.trace}...")
    data = load_trace_raw(args.trace)
    events = get_events(data)
    x_events = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    print(f"Total events: {len(events)}, complete (X) events: {len(x_events)}")

    gpu_tid = args.gpu_tid
    if gpu_tid is None:
        gpu_tid = find_gpu_stream_tid(events)
        if gpu_tid is None:
            print("ERROR: Could not detect GPU stream. Use --gpu-tid.", file=sys.stderr)
            sys.exit(1)

    if args.fix:
        n_fixed = fix_overlaps(events, gpu_tid)
        output_path = args.output
        if output_path is None:
            base = args.trace.rsplit(".", 1)[0] if "." in args.trace else args.trace
            output_path = base + "_fixed.trace.json"
        with open(output_path, "w") as f:
            json.dump(data, f)
        print(f"Overlaps fixed: {n_fixed} (GPU stream only, tid={gpu_tid})")
        print(f"Fixed trace written to: {output_path}")
        # Re-extract X events after fixing for analysis below
        x_events = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]

    gpu_events = [e for e in x_events if e.get("tid") == gpu_tid]
    print(f"GPU stream events: {len(gpu_events)}")
    print()

    intervals, n_clipped, n_dropped = deoverlap_and_analyze(gpu_events)
    print_summary(intervals, n_clipped, n_dropped, show_details=args.details)


if __name__ == "__main__":
    main()
