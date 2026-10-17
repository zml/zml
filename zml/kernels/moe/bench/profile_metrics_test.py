"""Reject incomplete captures and account for overlapped kernel execution."""
import json
import pathlib
import sqlite3
import tempfile
import unittest

import profile_metrics


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = pathlib.Path(self.temp.name)
        self.db_path = directory / "capture.sqlite"
        self.records_path = directory / "results.jsonl"
        self.records_path.write_text("\n".join(map(json.dumps, [
            {"kind": "run", "devices": ["GPU"]},
            {"kind": "case", "id": "case1", "workload": {}, "statistics": {
                "timing": {"iterations_per_sample": 2, "samples_us": [0.05, 0.05]}
            }},
        ])))
        with sqlite3.connect(self.db_path) as db:
            db.executescript("""
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                INSERT INTO StringIds VALUES(1, 'up'), (2, 'down');
                CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT,
                    textId INTEGER, globalTid INTEGER);
                INSERT INTO NVTX_EVENTS VALUES
                    (10, 400, 'moe_bench.measure/case1', NULL, 1),
                    (20, 80, 'moe_bench.batch#iterations=1#', NULL, 1),
                    (100, 200, 'moe_bench.batch#iterations=2#', NULL, 1),
                    (250, 350, 'moe_bench.batch#iterations=2#', NULL, 1);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,
                    end INTEGER, globalPid INTEGER, deviceId INTEGER,
                    contextId INTEGER, correlationId INTEGER, graphId INTEGER,
                    graphNodeId INTEGER, demangledName INTEGER);
            """)
            for sample, start in enumerate([100, 250]):
                for replay, offset in enumerate([1, 30]):
                    correlation = 10 + 2 * sample + replay
                    for node, delta in [(1, 0), (2, 5)]:
                        a = start + offset + delta
                        db.execute(
                            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?)",
                            (a, a + 10, 1, 0, 1, correlation, 1, node, node),
                        )

    def extract(self):
        return profile_metrics.extract(self.db_path, self.records_path)

    def test_overlap_and_elapsed_time_close(self):
        result = self.extract()["cases"][0]
        metrics = result["devices"]["all"]
        self.assertAlmostEqual(metrics["kernel_work_us_per_call"]["median"], 0.020)
        self.assertAlmostEqual(metrics["kernel_union_us_per_call"]["median"], 0.015)
        self.assertAlmostEqual(metrics["overlap_us_per_call"]["median"], 0.005)
        self.assertAlmostEqual(metrics["no_recorded_gpu_activity_us_per_call"]["median"], 0.035)
        self.assertEqual(result["retained_samples"], 2)

    def test_duplicate_records_and_ranges_are_rejected(self):
        original = self.records_path.read_text()
        for line, error in ((original.splitlines()[0], "exactly one run"),
                            (original.splitlines()[1], "duplicate workload records")):
            with self.subTest(error=error):
                self.records_path.write_text(original + "\n" + line)
                with self.assertRaisesRegex(ValueError, error):
                    self.extract()
        self.records_path.write_text(original)
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE text='moe_bench.measure/case1'")
        with self.assertRaisesRegex(ValueError, "duplicate workload range"):
            self.extract()

    def test_means_preserve_additive_accounting_with_variable_overlap(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=start+10, end=end+10 WHERE start>250 AND graphNodeId=2")
        metrics = self.extract()["cases"][0]["devices"]["all"]
        value = lambda key: metrics[key + "_us_per_call"]["mean"]
        self.assertAlmostEqual(value("kernel_work") - value("overlap"), value("kernel_union"))
        self.assertAlmostEqual(value("kernel_union") + value("memory_only") + value("no_recorded_gpu_activity"),
                               value("full_batch"))

    def test_failed_replay_stays_visible_without_timing_claims(self):
        records = [json.loads(line) for line in self.records_path.read_text().splitlines()]
        records[1].update(statistics=None, status="execution_failed", reason="ReplayMismatch")
        self.records_path.write_text("\n".join(map(json.dumps, records)))
        result = self.extract()
        self.assertEqual(result["cases"], [])
        self.assertEqual(result["unmeasured"], [{
            "id": "case1", "status": "execution_failed", "reason": "ReplayMismatch",
        }])

    def test_tuning_capture_selects_its_configuration(self):
        records = [json.loads(line) for line in self.records_path.read_text().splitlines()]
        records[0]['configuration_name'] = 'default'
        self.records_path.write_text("\n".join(map(json.dumps, records)))
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT INTO NVTX_EVENTS SELECT start+500,end+500,text,textId,globalTid FROM NVTX_EVENTS")
            db.execute("INSERT INTO NVTX_EVENTS VALUES(0,450,'moe_bench.configuration/default',NULL,1)")
            db.execute("INSERT INTO NVTX_EVENTS VALUES(500,950,'moe_bench.configuration/tile-64-single',NULL,1)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT start+500,end+500,globalPid,deviceId,contextId,correlationId+100,graphId+1,graphNodeId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")
        first = self.extract()
        self.assertEqual(len(first['cases']), 1)
        records[0]['configuration_name'] = 'tile-64-single'
        self.records_path.write_text("\n".join(map(json.dumps, records)))
        second = self.extract()
        self.assertEqual(first['cases'][0]['devices'], second['cases'][0]['devices'])
        records[0]['configuration_name'] = 'missing'
        self.records_path.write_text("\n".join(map(json.dumps, records)))
        with self.assertRaisesRegex(ValueError, 'configuration range'):
            self.extract()

    def test_missing_graph_node_is_rejected(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=11 AND graphNodeId=2")
        with self.assertRaisesRegex(ValueError, "incomplete or changing graph"):
            self.extract()

    def test_boundary_straddling_kernel_is_rejected(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=99 WHERE correlationId=10 AND graphNodeId=1")
        with self.assertRaisesRegex(ValueError, "incomplete kernel boundary"):
            self.extract()

    def test_missing_rank_is_rejected(self):
        records = self.records_path.read_text().splitlines()
        metadata = json.loads(records[0])
        metadata["devices"].append("missing GPU")
        records[0] = json.dumps(metadata)
        self.records_path.write_text("\n".join(records))
        with self.assertRaisesRegex(ValueError, "missing devices"):
            self.extract()

    def test_uncaptured_launches_have_counts_but_no_replay_spans(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphId=NULL, graphNodeId=NULL, correlationId=rowid")
        result = self.extract()["cases"][0]
        self.assertEqual(result["launch_modes"], {0: "launch_counts"})
        self.assertEqual(result["replay_span_us"], {})
        self.assertAlmostEqual(result["devices"]["0"]["kernel_work_us_per_call"]["median"], 0.02)
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid=1")
        with self.assertRaisesRegex(ValueError, "incomplete launch counts"):
            self.extract()


if __name__ == "__main__":
    unittest.main()
