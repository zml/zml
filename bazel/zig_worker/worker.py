"""Bridge Bazel's JSON worker protocol to Zig 0.17's --listen=- protocol."""

import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading


HEADER = struct.Struct("<II")
OUTPUT_NAME = "bazel_incremental"


def expand_arguments(arguments):
    """Bazel expands these in worker mode; local execution reads them itself."""
    result = []
    for arg in arguments:
        if arg.startswith("@"):
            result.extend(Path(arg[1:]).read_text().splitlines())
        else:
            result.append(arg)
    return result


def render_errors(body):
    """Decode std.zig.ErrorBundle, including locations, notes and compile logs."""
    extra_len, strings_len = HEADER.unpack_from(body)
    if len(body) != 8 + 4 * extra_len + strings_len:
        raise ValueError("invalid Zig error bundle length")
    if extra_len == 0:
        return ""
    extra = struct.unpack_from(f"<{extra_len}I", body, 8)
    strings = body[8 + 4 * extra_len:]

    def string(index):
        return strings[index:strings.index(b"\0", index)].decode("utf-8", "replace")

    lines = []

    def message(index, kind, indent=""):
        text, count, location, notes = extra[index:index + 4]
        prefix = ""
        if location:
            path, line, column = extra[location:location + 3]
            prefix = f"{string(path)}:{line + 1}:{column + 1}: "
        lines.append(f"{indent}{prefix}{kind}: {string(text)}")
        if location and extra[location + 6]:
            lines.append(indent + string(extra[location + 6]))
            lines.append(indent + " " * column + "^")
        for note in extra[index + 4:index + 4 + notes]:
            message(note, "note", indent + "    ")

    count, start, compile_log = extra[:3]
    for index in extra[start:start + count]:
        message(index, "error")
    if compile_log:
        lines.append(string(compile_log))
    return "\n".join(lines) + ("\n" if lines else "")


def compiler_arguments(arguments, cache):
    """Server mode emits into its cache, never directly into Bazel's outputs."""
    result = []
    outputs = []
    skip = False
    for arg in arguments:
        if skip:
            skip = False
            continue
        if arg in ("--cache-dir", "--name"):
            skip = True
        elif arg.startswith("-femit-") and "=" in arg:
            flag, path = arg.split("=", 1)
            outputs.append((flag.removeprefix("-femit-"), Path(path)))
            result.append(flag)
        elif arg in ("--watch", "-fno-incremental") or arg.startswith("--listen"):
            raise ValueError(f"{arg} conflicts with the incremental worker")
        else:
            result.append(arg)
    result.extend(["--name", OUTPUT_NAME, "--cache-dir", cache,
                   "-fincremental", "--listen=-"])
    return result, outputs


def copy_outputs(directory, outputs, command):
    for kind, destination in outputs:
        if kind == "bin":
            if command == "build-lib":
                names = [f"lib{OUTPUT_NAME}.a", f"{OUTPUT_NAME}.lib",
                         f"lib{OUTPUT_NAME}.so", f"lib{OUTPUT_NAME}.dylib",
                         f"{OUTPUT_NAME}.dll"]
            elif command in ("build-obj", "test-obj"):
                names = [f"{OUTPUT_NAME}.o", f"{OUTPUT_NAME}.obj"]
            else:
                names = [OUTPUT_NAME, f"{OUTPUT_NAME}.exe"]
        else:
            suffix = {"asm": ".s", "llvm-ir": ".ll", "llvm-bc": ".bc", "h": ".h"}[kind]
            names = [OUTPUT_NAME + suffix]
        candidates = [directory / name for name in names if (directory / name).is_file()]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one {kind} output in {directory}, found {candidates}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Copies, not hardlinks: the next compiler update mutates the cache file.
        shutil.copyfile(candidates[0], destination)
        shutil.copymode(candidates[0], destination)


class Compiler:
    def __init__(self, zig, arguments):
        self.cache = tempfile.TemporaryDirectory(prefix="zml-zig-worker-")
        self.arguments = arguments
        command, self.outputs = compiler_arguments(arguments, self.cache.name)
        self.stderr = collections.deque(maxlen=128)
        self.process = subprocess.Popen(
            [zig, *command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()
        self.updates = 0

    def _read_stderr(self):
        while data := self.process.stderr.read1(8192):
            self.stderr.append(data)

    def update(self):
        self.process.stdin.write(HEADER.pack(1, 0))  # Client.Tag.update
        self.process.stdin.flush()
        output_dir = None
        while True:
            header = self.process.stdout.read(HEADER.size)
            if len(header) != HEADER.size:
                self.process.wait()
                self.stderr_thread.join()
                diagnostics = b"".join(self.stderr).decode("utf-8", "replace")
                raise RuntimeError(f"Zig exited with status {self.process.returncode}\n{diagnostics}")
            tag, length = HEADER.unpack(header)
            body = self.process.stdout.read(length)
            if len(body) != length:
                raise RuntimeError("truncated Zig protocol message")
            if tag == 0:  # Server.Tag.zig_version
                if not body.startswith(b"0.17."):
                    raise RuntimeError(f"unsupported Zig compiler protocol version: {body!r}")
            elif tag == 2:  # Server.Tag.emit_digest: flags byte + 16-byte digest
                if len(body) != 17:
                    raise RuntimeError("invalid Zig output digest")
                output_dir = Path(self.cache.name, "o", body[1:].hex())
            elif tag == 1:  # An error bundle, including an empty one, ends an update.
                diagnostics = render_errors(body)
                self.updates += 1
                if diagnostics:
                    return 1, diagnostics
                if self.outputs:
                    if output_dir is None:
                        raise RuntimeError("Zig completed without an output digest")
                    copy_outputs(output_dir, self.outputs, self.arguments[0])
                return 0, ""
            # config, file_system_inputs and timing messages are informational.

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(HEADER.pack(0, 0))  # Client.Tag.exit
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.stderr_thread.join()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except BrokenPipeError:
                pass
        self.cache.cleanup()


class Worker:
    def __init__(self, zig):
        self.zig = zig
        self.compiler = None
        self.inputs = None
        self.stats = {}

    def close(self):
        if self.compiler:
            compiler = self.compiler
            self.compiler = None
            compiler.close()

    def build(self, request):
        arguments = expand_arguments(request["arguments"])
        inputs = {item["path"]: item.get("digest", "") for item in request.get("inputs", [])}
        stats = {}
        for path in inputs:
            if path.endswith(".zig"):
                stat = os.stat(path)
                stats[path] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        restart = self.compiler is None or self.compiler.arguments != arguments
        if self.inputs is not None:
            restart |= inputs.keys() != self.inputs.keys()
            for path, digest in inputs.items():
                if digest != self.inputs.get(path):
                    # Zig detects source edits itself. Other inputs (objects,
                    # embedded data, headers) can invalidate retained state.
                    restart |= not path.endswith(".zig") or stats.get(path) == self.stats.get(path)
        self.inputs, self.stats = inputs, stats
        if restart:
            self.close()
            self.compiler = Compiler(self.zig, arguments)
        try:
            code, output = self.compiler.update()
            if request.get("verbosity", 0):
                output += (f"zig worker: pid={self.compiler.process.pid} "
                           f"update={self.compiler.updates} restarted={restart}\n")
            return code, output
        except Exception:
            # A protocol failure or compiler crash must never poison later work.
            self.close()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--persistent_worker", action="store_true")
    options, arguments = parser.parse_known_args()
    worker = Worker(options.zig)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        if not options.persistent_worker:
            code, output = worker.build({"arguments": arguments})
            sys.stderr.write(output)
            return code
        for line in sys.stdin:
            request = json.loads(line)
            try:
                code, output = worker.build(request)
            except Exception as error:
                code, output = 1, f"zig worker: {error}\n"
            response = {"exitCode": code, "output": output, "requestId": request.get("requestId", 0)}
            print(json.dumps(response), flush=True)
        return 0
    finally:
        worker.close()


if __name__ == "__main__":
    sys.exit(main())
