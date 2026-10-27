"""Exercise real compiler reuse, invalidation, failures and Bazel framing."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import worker


ZIG = Path(sys.argv.pop(1)).absolute() if len(sys.argv) > 1 else None
if ZIG and not ZIG.exists():
    # ZIG_BIN is execroot-relative, while tests execute in the runfiles tree.
    ZIG = Path(os.environ["TEST_SRCDIR"], str(ZIG).split("/external/", 1)[1])


class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "main.zig"
        self.dependency = self.root / "value.zig"
        self.embedded = self.root / "value.txt"
        self.output = self.root / "program"
        self.source.write_text('pub fn main() u8 { return @import("value.zig").value; }\n')
        self.dependency.write_text("pub const value: u8 = 42;\n")
        self.embedded.write_text("x")
        self.worker = worker.Worker(str(ZIG))
        self.addCleanup(self.worker.close)

    def request(self, extra=()):
        return {
            "arguments": ["build-exe", str(self.source),
                          "-femit-bin=" + str(self.output),
                          "--global-cache-dir", str(self.root / "global-cache"), *extra],
            "inputs": [{"path": str(path), "digest": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path in (self.source, self.dependency, self.embedded)],
        }

    def build(self, expected):
        code, diagnostics = self.worker.build(self.request())
        self.assertEqual(code, 0, diagnostics)
        self.assertEqual(subprocess.run([str(self.output)], check=False).returncode, expected)
        return self.worker.compiler.process.pid

    def test_reuse_error_recovery_and_output_recreation(self):
        pid = self.build(42)
        self.dependency.write_text("pub const value: u8 = 43;\n")
        self.output.unlink()  # Bazel removes declared outputs before execution.
        self.assertEqual(self.build(43), pid)
        self.output.unlink()
        self.assertEqual(self.build(43), pid)
        self.dependency.write_text('pub const value: u8 = @compileError("intentional failure");\n')
        self.output.unlink()
        code, diagnostics = self.worker.build(self.request())
        self.assertEqual(code, 1)
        self.assertIn("intentional failure", diagnostics)
        self.assertIn("value.zig:1:", diagnostics)
        self.assertFalse(self.output.exists())
        self.dependency.write_text("pub const value: u8 = 44;\n")
        self.assertEqual(self.build(44), pid)

    def test_input_and_argument_invalidation(self):
        pid = self.build(42)
        self.embedded.write_text("changed embedded input")
        self.assertNotEqual(self.build(42), pid)
        pid = self.worker.compiler.process.pid
        code, diagnostics = self.worker.build(self.request(["-OReleaseSafe"]))
        self.assertEqual(code, 0, diagnostics)
        self.assertNotEqual(self.worker.compiler.process.pid, pid)
        pid = self.worker.compiler.process.pid
        request = self.request(["-OReleaseSafe"])
        request["inputs"].pop()
        self.assertEqual(self.worker.build(request)[0], 0)
        self.assertNotEqual(self.worker.compiler.process.pid, pid)

    def test_same_stat_source_edit_restarts(self):
        pid = self.build(42)
        stat = self.dependency.stat()
        self.dependency.write_text("pub const value: u8 = 43;\n")
        os.utime(self.dependency, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(self.build(43), pid)

    def test_compiler_crash_does_not_poison_worker(self):
        pid = self.build(42)
        self.worker.compiler.process.kill()
        self.worker.compiler.process.wait()
        with self.assertRaises((BrokenPipeError, RuntimeError)):
            self.worker.build(self.request())
        self.assertNotEqual(self.build(42), pid)

    def test_json_protocol_and_shutdown(self):
        process = subprocess.Popen(
            [sys.executable, worker.__file__, "--zig=" + str(ZIG),
             "--action=test", "--persistent_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        try:
            pids = []
            for index in range(2):
                self.dependency.write_text(f"pub const value: u8 = {42 + index};\n")
                request = self.request()
                request.update(requestId=index + 7, verbosity=1)
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                response = json.loads(process.stdout.readline())
                self.assertEqual(response["requestId"], index + 7)
                self.assertEqual(response["exitCode"], 0, response["output"])
                pids.append(response["output"].split("pid=")[1].split()[0])
                self.assertEqual(subprocess.run([str(self.output)]).returncode, 42 + index)
            self.assertEqual(pids[0], pids[1])
            process.stdin.close()
            self.assertEqual(process.wait(timeout=10), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()


if __name__ == "__main__":
    unittest.main()
