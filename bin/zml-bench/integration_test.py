#!/usr/bin/env python3
"""Exercise the built benchmark against a local, deterministic SSE endpoint."""

import http.server
import json
import os
import subprocess
import sys
import threading
import time
import tempfile
import unittest


BINARY = os.path.abspath(sys.argv.pop(1)) if len(sys.argv) > 1 else os.path.abspath(
    "bazel-bin/bin/zml-bench/zml-bench"
)


class Endpoint(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_POST(self):
        self.close_connection = True
        try:
            self.serve()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def serve(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.requests.append((self.path, dict(self.headers), body))
            if len(self.server.requests) >= self.server.expected:
                self.server.ready.set()
        if not self.server.ready.wait(5):
            self.send_error(503, "Requests were not concurrent")
            return
        model = body["model"]
        if model == "stall-head":
            time.sleep(3)
        if model == "http-error":
            self.send_error(401, "Test unauthorized")
            return
        if model == "redirect":
            self.send_response(307)
            self.send_header("Location", "/must-not-follow")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if model == "long-event":
            content = "a" * 20000
            payload = ("data: " + json.dumps({"choices": [{"delta": {"content": content}}]})
                       + "\n\ndata: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if model == "stall-body":
            time.sleep(3)

        def write(data):
            # Fragment both SSE lines and UTF-8 characters across HTTP chunks.
            data = data.encode()
            for offset in range(0, len(data), 7):
                part = data[offset:offset + 7]
                self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            self.wfile.flush()

        def event(value):
            write("data: " + json.dumps(value, ensure_ascii=False) + "\r\n\r\n")

        if model == "malformed":
            write("data: {bad json}\n\n")
        elif model == "api-error":
            event({"error": {"message": "Test streaming error"}})
        else:
            write(": heartbeat\r\n\r\n")
            event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
            time.sleep(0.04)
            event({"choices": [{"index": 0, "delta": {"reasoning_content": "Thinking. "}}]})
            time.sleep(0.04)
            event({"choices": [{"index": 0, "delta": {"content": 'hello 🙂 "world"\n'}}]})
            if model != "truncated":
                event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                if model != "no-usage":
                    event({"choices": [], "usage": {"completion_tokens": 23, "prompt_tokens": 11}})
                if model != "finish-only":
                    write("data: [DONE]\r\n\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
        self.server.requests = []
        self.server.expected = 1
        self.server.lock = threading.Lock()
        self.server.ready = threading.Event()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def run_batch(self, model="ok", batch=1, extra=()):
        self.server.expected = batch
        endpoint = f"http://127.0.0.1:{self.server.server_port}/v1"
        process = subprocess.run(
            [BINARY, "--headless", "--endpoint", endpoint, "--model", model,
             "--batch", str(batch), "--prompt", 'quote " and\nnewline', *extra],
            capture_output=True, text=True, timeout=8,
            env={**os.environ, "OPENAI_API_KEY": "test-key"},
        )
        self.assertTrue(process.stdout, process.stderr)
        self.report = json.loads(process.stdout)
        return process, self.report["summary"]

    def test_concurrent_fragmented_streams_and_request_payload(self):
        process, summary = self.run_batch(batch=8, extra=("--max-tokens", "64", "--temperature", "0"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(summary["completed"], 8)
        self.assertEqual(summary["tokens"], 8 * 23)
        self.assertFalse(summary["estimated"])
        self.assertGreaterEqual(summary["medianTtftMs"], 35)
        self.assertGreater(summary["aggregateRate"], 0)
        for path, headers, body in self.server.requests:
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(headers["Authorization"], "Bearer test-key")
            self.assertEqual(body["messages"][0]["content"], 'quote " and\nnewline')
            self.assertEqual(body["max_tokens"], 64)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_large_event_without_chunked_transfer(self):
        process, summary = self.run_batch("long-event")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["tokens"], 5000)

    def test_missing_usage_is_explicitly_estimated(self):
        process, summary = self.run_batch("no-usage")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(summary["estimated"])
        self.assertGreater(summary["tokens"], 0)
        body = self.server.requests[0][2]
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("temperature", body)

    def test_finish_reason_without_done(self):
        process, summary = self.run_batch("finish-only")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(summary["completed"], 1)

    def test_http_error(self):
        process, summary = self.run_batch("http-error")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(summary["failed"], 1)

    def test_redirect_is_not_followed(self):
        process, summary = self.run_batch("redirect")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(self.server.requests), 1)

    def test_stream_error(self):
        process, summary = self.run_batch("api-error")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(summary["failed"], 1)

    def test_malformed_stream(self):
        process, summary = self.run_batch("malformed")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(summary["failed"], 1)

    def test_truncated_stream(self):
        process, summary = self.run_batch("truncated")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["completed"], 0)

    def test_delayed_headers_complete_without_a_deadline(self):
        process, summary = self.run_batch("stall-head")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(summary["completed"], 1)
        self.assertGreater(summary["ttft"]["meanMs"], 2900)

    def test_delayed_body_completes_without_a_deadline(self):
        process, summary = self.run_batch("stall-body")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(summary["completed"], 1)

    def test_options_and_saved_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "report.json")
            process, summary = self.run_batch(extra=(
                "--thinking", "75", "--system", "Be concise", "--max-completion-tokens", "128",
                "--stop", '["END"]', "--tools", "[]", "--documents", '[{"text":"context"}]',
                "--report", path,
            ))
            self.assertEqual(process.returncode, 0, process.stderr)
            body = self.server.requests[0][2]
            self.assertEqual(body["reasoning_effort"], 75)
            self.assertEqual(body["messages"][0], {"role": "system", "content": "Be concise"})
            self.assertEqual(body["max_completion_tokens"], 128)
            self.assertEqual(body["stop"], ["END"])
            self.assertEqual(body["tools"], [])
            self.assertEqual(summary["itl"]["count"], 1)
            self.assertGreater(summary["itl"]["meanMs"], 20)
            self.assertEqual(summary["ttft"]["count"], 1)
            self.assertEqual(summary["latency"]["count"], 1)
            self.assertEqual(summary["promptTokens"], 11)
            self.assertEqual(summary["chunks"], 2)
            self.assertGreater(summary["tpot"]["meanMs"], 0)
            self.assertGreater(summary["firstAnswer"]["meanMs"], summary["ttft"]["meanMs"])
            with open(path) as report_file:
                saved = json.load(report_file)
            self.assertEqual(saved, self.report)
            self.assertEqual(saved["requests"][0]["finishReason"], "stop")
            self.assertEqual(saved["requests"][0]["completionTokens"], 23)
            self.assertNotIn("test-key", json.dumps(saved))

    def test_timeout_option_is_removed(self):
        process = subprocess.run([BINARY, "--timeout", "1"], capture_output=True, text=True, timeout=5)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("UnknownOption", process.stderr)


if __name__ == "__main__":
    unittest.main()
