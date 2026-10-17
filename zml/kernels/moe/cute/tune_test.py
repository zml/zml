"""Check that the driver always measures and validates its own CuTe sweep."""
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TuneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bench = self.root / "fake bench"
        implementation = self.root / "fake_bench.py"
        self.bench.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(implementation))} "$@"\n')
        implementation.write_text(
            "import json, os, sys\n"
            "with open(os.environ['FAKE_LOG'], 'a') as f:\n"
            "    f.write(json.dumps({'args': sys.argv[1:], 'env': {k: v for k, v in os.environ.items() if k.startswith('ZML_MOE_CUTE_')}}) + '\\n')\n"
            "sys.exit(int(os.environ.get('FAIL_TILE', '') == os.environ.get('ZML_MOE_CUTE_TILE_N', 'default')))\n"
        )
        self.bench.chmod(0o755)
        self.out = self.root / "results"
        self.log = self.root / "calls.jsonl"
        self.env = dict(os.environ, FAKE_LOG=str(self.log), NSYS="", TILES="64",
                        QUANTIZE_LANES="4", PERSISTENT_CTAS="128",
                        ZML_MOE_CUTE_TILE_N="8", ZML_MOE_CUTE_TWO_CTA="true")
        self.driver = Path(__file__).with_name("tune.sh")

    def run_driver(self, *args):
        return subprocess.run(["bash", str(self.driver), str(self.bench), str(self.out), *args],
                              env=self.env, capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_settings_and_saved_commands_are_reproducible(self):
        run = self.run_driver("--tokens=1,17", "--check")
        details = self.out / "default.stderr"
        self.assertEqual(run.returncode, 0, run.stderr + (details.read_text() if details.exists() else ""))
        calls = self.calls()
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[0]["env"], {})
        self.assertEqual(calls[1]["env"], {"ZML_MOE_CUTE_TILE_N": "64"})
        self.assertEqual(calls[2]["env"], {"ZML_MOE_CUTE_TILE_N": "64", "ZML_MOE_CUTE_TWO_CTA": "false"})
        self.assertEqual(calls[3]["env"], {"ZML_MOE_CUTE_QUANTIZE_LANES": "4"})
        self.assertEqual(calls[4]["env"], {"ZML_MOE_CUTE_PERSISTENT_CTAS": "128"})
        for call in calls:
            self.assertIn("--backend=cute", call["args"])
            self.assertIn("--tokens=1,17", call["args"])
        for call in calls[1:]:
            self.assertIn("--compare-exact", call["args"])
            self.assertIn(f"--compare-dir={self.out}/baseline", call["args"])
        replay = subprocess.run(["bash", str(self.out / "tile-64-single.command")], env=self.env)
        self.assertEqual(replay.returncode, 0)
        self.assertEqual(self.calls()[-1], calls[2])
        self.assertTrue((self.out / "benchmark.sha256").is_file())

    def test_conflicting_options_rejected_before_creating_results(self):
        for arg in ("--help", "-h", "--list", "--list=false", "--backend=triton", "--backend=cute,triton",
                    "--dump=x", "--dump-dir=x", "--compare=x", "--compare-dir=x",
                    "--compare-exact", "--json=x"):
            with self.subTest(arg=arg):
                self.assertEqual(self.run_driver(arg).returncode, 2)
                self.assertFalse(self.out.exists())
                self.assertFalse(self.log.exists())

    def test_candidate_failures_do_not_hide_remaining_results(self):
        self.env["FAIL_TILE"] = "64"
        self.assertEqual(self.run_driver().returncode, 1)
        self.assertEqual(len(self.calls()), 5)

    def test_failed_baseline_stops_sweep(self):
        self.env["FAIL_TILE"] = "default"
        self.assertEqual(self.run_driver().returncode, 1)
        self.assertEqual(len(self.calls()), 1)


if __name__ == "__main__":
    unittest.main()
