"""Summarize complete, warmed MoE benchmark batches from an Nsight SQLite export."""
import argparse
import collections
import json
import pathlib
import re
import sqlite3
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def union(events):
    total = 0
    end = None
    for a, b in sorted(events):
        if end is None or a > end:
            total += b - a
        elif b > end:
            total += b - end
        end = b if end is None else max(end, b)
    return total


def summarize(values):
    median = statistics.median(values)
    return {
        'mean': statistics.fmean(values),
        'median': median,
        'mad': statistics.median(abs(x - median) for x in values),
        'min': min(values),
        'max': max(values),
        'samples': values,
    }


def extract(db_path, records_path):
    rows = [json.loads(x) for x in pathlib.Path(records_path).read_text().splitlines()]
    runs = [x for x in rows if x['kind'] == 'run']
    require(len(runs) == 1, 'expected exactly one run record')
    metadata = runs[0]
    cases = [x for x in rows if x['kind'] == 'case']
    case_records = {x['id']: x for x in cases}
    require(len(case_records) == len(cases), 'duplicate workload records')
    records = {key: row for key, row in case_records.items() if row.get('statistics')}
    db = sqlite3.connect(f'file:{pathlib.Path(db_path).resolve()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    strings = dict(db.execute('select id,value from StringIds'))
    markers = [dict(x) for x in db.execute("""
        SELECT start, end,
            coalesce(text, (SELECT value FROM StringIds WHERE id=textId)) AS name,
            globalTid
        FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start
    """)]
    if metadata.get('configuration_name') is not None:
        name = 'moe_bench.configuration/' + metadata['configuration_name']
        scopes = [x for x in markers if x['name'] == name]
        require(len(scopes) == 1, ('missing or duplicate configuration range', name))
        scope = scopes[0]
        markers = [x for x in markers if x['globalTid'] == scope['globalTid']
                   and x['start'] >= scope['start'] and x['end'] <= scope['end']]
    measurements = [x for x in markers if (x['name'] or '').startswith('moe_bench.measure/')]
    batches = [x for x in markers if (x['name'] or '').startswith('moe_bench.batch#')]
    kernels = [dict(x) for x in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
    memories = []
    tables = {x[0] for x in db.execute("select name from sqlite_master where type='table'")}
    for table in ['CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET']:
        if table in tables:
            memories.extend((dict(x) for x in db.execute(f'select * from {table}')))
    db.close()
    output = []
    seen_measurements = set()
    for m in measurements:
        identity = m['name'].split('moe_bench.measure/', 1)[1]
        require(identity in case_records, ('unknown workload range', identity))
        require(identity not in seen_measurements, ('duplicate workload range', identity))
        seen_measurements.add(identity)
        if identity not in records:
            continue
        record = records[identity]
        timing = record['statistics']['timing']
        n = timing['iterations_per_sample']
        samples = len(timing['samples_us'])
        require(n > 0 and samples > 0, ('empty timing samples', identity))
        bs = [b for b in batches
              if b['globalTid'] == m['globalTid'] and b['start'] >= m['start']
              and b['end'] <= m['end']][-samples:]
        require(len(bs) == samples, ('missing samples', identity))
        require(all(int(re.search(r'iterations=(\d+)', b['name'])[1]) == n for b in bs),
                'retained iteration count does not match JSONL')
        metrics = collections.defaultdict(lambda: collections.defaultdict(list))
        kernel_metrics = collections.defaultdict(lambda: collections.defaultdict(list))
        raw_durations = collections.defaultdict(lambda: collections.defaultdict(list))
        replay_spans = collections.defaultdict(list)
        expected_nodes = {}
        launch_modes = {}
        device_set = None
        for batch in bs:
            a, b = (batch['start'], batch['end'])
            elapsed = b - a
            ks = [e for e in kernels if e['start'] < b and e['end'] > a]
            require(ks and all((e['start'] >= a and e['end'] <= b for e in ks)), 'incomplete kernel boundary')
            require(len({e['globalPid'] for e in ks}) == 1, 'ambiguous process scope')
            pid = ks[0]['globalPid']
            ms = [e for e in memories if e['globalPid'] == pid and e['start'] < b and (e['end'] > a)]
            devices = sorted({e['deviceId'] for e in ks})
            if device_set is None:
                device_set = devices
            require(devices == device_set and len(devices) == len(metadata['devices']), 'missing devices')
            for device in devices + ['all']:
                dk = ks if device == 'all' else [e for e in ks if e['deviceId'] == device]
                dm = ms if device == 'all' else [e for e in ms if e['deviceId'] == device]
                ki = [(e['start'], e['end']) for e in dk]
                mi = [(max(a, e['start']), min(b, e['end'])) for e in dm]
                work = sum((y - x for x, y in ki))
                active = union(ki)
                any_active = union(ki + mi)
                metrics[str(device)]['full_batch_us_per_call'].append(elapsed / n / 1000)
                for name, value in [
                    ('kernel_work', work), ('kernel_union', active),
                    ('overlap', work - active), ('memory_only', any_active - active),
                    ('no_recorded_gpu_activity', elapsed - any_active),
                ]:
                    metrics[str(device)][name + '_us_per_call'].append(value / n / 1000)
                require(active + (any_active - active) + (elapsed - any_active) == elapsed,
                        'elapsed-time accounting does not close')
                require(work - (work - active) == active, 'Invalid or incomplete benchmark capture')
                if device == 'all':
                    continue
                replays = collections.defaultdict(list)
                for event in dk:
                    replays[event['contextId'], event['correlationId']].append(event)
                is_graph = (len(replays) == n and
                            all(e['graphId'] and e['graphNodeId'] is not None for e in dk))
                mode = 'graph_replays' if is_graph else 'launch_counts'
                if device in launch_modes:
                    require(launch_modes[device] == mode, 'changing launch mode')
                launch_modes[device] = mode
                if is_graph:
                    for events in replays.values():
                        nodes = sorted((e['graphId'], e['graphNodeId'],
                                        strings[e['demangledName']]) for e in events)
                        require(len({(x[0], x[1]) for x in nodes}) == len(nodes),
                                'duplicate graph node')
                        if device not in expected_nodes:
                            expected_nodes[device] = nodes
                        require(nodes == expected_nodes[device],
                                ('incomplete or changing graph', identity, device))
                        span = max(e['end'] for e in events) - min(e['start'] for e in events)
                        replay_spans[str(device)].append(span / 1000)
                else:
                    # XLA may leave short workloads uncaptured, or mix graphs
                    # with ordinary launches. Validate the complete batch's
                    # per-call multiplicities without inventing replay spans.
                    counts = collections.Counter(
                        (e['contextId'], e['graphId'] or 0, e['graphNodeId'] or 0,
                         strings[e['demangledName']]) for e in dk)
                    require(all(count % n == 0 for count in counts.values()),
                            ('incomplete launch counts', identity, device))
                    nodes = sorted((*key, count // n) for key, count in counts.items())
                    if device not in expected_nodes:
                        expected_nodes[device] = nodes
                    require(nodes == expected_nodes[device],
                            ('changing launch counts', identity, device))
                by_name = collections.defaultdict(list)
                for e in dk:
                    by_name[strings[e['demangledName']]].append((e['end'] - e['start']) / 1000)
                for name, durations in by_name.items():
                    kernel_metrics[str(device)][name].append(sum(durations) / n)
                    raw_durations[str(device)][name].extend(durations)
        output.append({
            'id': identity,
            'workload': record['workload'],
            'iterations_per_sample': n,
            'retained_samples': samples,
            'devices': {
                d: {k: summarize(v) for k, v in data.items()}
                for d, data in metrics.items()
            },
            'kernels': {
                d: {k: {'work_us_per_call': summarize(v),
                        'individual_duration_us': summarize(raw_durations[d][k])}
                    for k, v in data.items()}
                for d, data in kernel_metrics.items()
            },
            'replay_span_us': {d: summarize(v) for d, v in replay_spans.items()},
            'nodes': expected_nodes,
            'launch_modes': launch_modes,
        })
    require(seen_measurements >= records.keys(), ('missing workload ranges', sorted(records.keys() - seen_measurements)))
    return {
        'metadata': metadata,
        'timing_boundary': (
            'Last retained synchronized batches; per-call batch span includes host dispatch, '
            'disposal and inter-call gaps. Kernel work is a sum; kernel union removes overlap. '
            'Replay span excludes inter-call gaps. All is a union over every participating GPU. '
            'Mean component times are additive across samples; component medians need not sum. '
            'Replay spans are omitted for ordinary or mixed launches; those use per-call '
            'launch-count validation. No dependency critical path inferred.'
        ),
        'cases': output,
        'unmeasured': [
            {key: row.get(key) for key in ('id', 'status', 'reason')}
            for row in rows if row['kind'] == 'case' and not row.get('statistics')
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlite')
    parser.add_argument('jsonl')
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    pathlib.Path(a.output).write_text(json.dumps(extract(a.sqlite, a.jsonl), indent=2))
