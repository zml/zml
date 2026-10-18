"""Process and concurrent-configuration checks through the real editor entry point."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile

from zls_navigation_test import Client, WORKSPACE, builtin_settings, external_first


def session_path(logs):
    match = re.search(r'zls: session metadata (.*)/manifest.json', logs)
    return Path(match[1]) if match else None


def assert_clean(process, logs):
    assert process.poll() is not None, 'launcher was not reaped'
    path = session_path(logs)
    if path:
        assert not path.exists(), f'leaked session directory: {path}'
        processes = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
        assert str(path) not in processes, f'orphaned session process: {path}'


def run(primary):
    command = [str(WORKSPACE / 'tools/zls.sh')]
    for args in ([], ['--not-a-zls-argument']):
        result = subprocess.run([*command, *args], input=b'', capture_output=True, cwd='/tmp', timeout=600)
        remaining = result.stdout
        while remaining:
            header, remaining = remaining.split(b'\r\n\r\n', 1)
            assert header.startswith(b'Content-Length: '), header
            length = int(header.split(b':', 1)[1])
            assert json.loads(remaining[:length])['jsonrpc'] == '2.0'
            remaining = remaining[length:]
        if args:
            assert result.returncode != 0, 'server failure was swallowed'
        path = session_path(result.stderr.decode())
        assert path and not path.exists(), result.stderr.decode()
    environment = dict(os.environ)
    environment['ZML_ZLS_BAZEL_FLAGS'] = json.dumps(['--not-a-bazel-flag'])
    result = subprocess.run(command, env=environment, input=b'', capture_output=True, timeout=60)
    assert result.returncode != 0 and result.stdout == b''
    assert b'preparation failed' in result.stderr, result.stderr

    # Different configurations run concurrently; one must not overwrite the
    # generated options used by the other server's published snapshot.
    environment = dict(os.environ)
    environment['ZML_ZLS_BAZEL_FLAGS'] = json.dumps([
        *json.loads(environment.get('ZML_ZLS_BAZEL_FLAGS', '[]')), '--@zml//platforms:cpu=false',
        '--@rules_zig//zig/settings:mode=release_fast', '--@rules_zig//zig/settings:threaded=single'])
    with tempfile.TemporaryDirectory(prefix='zml-workspace-link-') as directory:
        link = Path(directory) / 'workspace'
        link.symlink_to(WORKSPACE, target_is_directory=True)
        second = Client(command=[str(link / 'tools/zls.sh'), '--log-level', 'debug'], env=environment, cwd='/tmp')
        third = Client(cwd='/tmp')
        try:
            second.initialize()
            third.initialize()
            external_first(third)
            for client in (second, third):
                path = client.open(WORKSPACE / 'platforms/capabilities.zig')
                client.definition(path, 'platforms/cuda', WORKSPACE / 'platforms/cuda/cuda.zig', configuring=True)
                builtin_settings(client)
            manifests = [client.manifest() for client in (primary, second, third)]
            assert manifests[1]['settings']['mode'] == 'release_fast'
            assert manifests[1]['settings']['threaded'] == 'single'
            print('PASS: ReleaseFast and single-threaded builtin values match the requested configuration', flush=True)
            options = [next(Path(m['source']) for m in manifest['modules']
                            if m['source'].endswith('/platforms/options.zig')) for manifest in manifests]
            assert len(set(options)) == 3, options
            assert 'pub const @"cpu_enabled": bool = false' in options[1].read_text(), options[1].read_text()
            assert options[0].read_text() == options[2].read_text()
            primary_options = options[0].read_bytes()
            second.process.send_signal(signal.SIGTERM)
            second.process.wait(timeout=15)
            assert second.process.returncode != 0
            assert_clean(second.process, ''.join(second.logs))
            assert options[0].read_bytes() == primary_options
            third.close()
            assert_clean(third.process, ''.join(third.logs))
        finally:
            second.close()
            third.close()
    terminal_build_race()
    print('PASS: EOF, nonzero exit, preparation failure, SIGTERM, concurrent configurations and symlinked workspace', flush=True)


def terminal_build_race():
    # Interleave a real ordinary build after editor generation and before the
    # launcher reads its outputs. The terminal build must not share those paths.
    with tempfile.TemporaryDirectory(prefix='zml-terminal-race-') as directory:
        wrapper = Path(directory) / 'bazel'
        marker = Path(directory) / 'interleaved'
        environment = dict(os.environ)
        environment['ZML_TEST_REAL_BAZEL'] = shutil.which('bazel')
        environment['ZML_TEST_RACE_MARKER'] = str(marker)
        environment['ZML_ZLS_BAZEL_FLAGS'] = json.dumps([
            *json.loads(environment.get('ZML_ZLS_BAZEL_FLAGS', '[]')), '--@zml//platforms:cpu=true'])
        wrapper.write_text('#!' + sys.executable + '\n' + '''import json, os, subprocess, sys
real = os.environ['ZML_TEST_REAL_BAZEL']
args = sys.argv[1:]
if 'info' in args and 'execution_root' in args:
    startup = json.loads(os.environ.get('ZML_ZLS_BAZEL_STARTUP_FLAGS', '[]'))
    flags = json.loads(os.environ['ZML_ZLS_BAZEL_FLAGS'])
    subprocess.run([real, *startup, 'build', *flags, '--@zml//platforms:cpu=false',
                    '--symlink_prefix=/', '//:completion_metadata'], check=True, stdout=sys.stderr)
    with open(os.environ['ZML_TEST_RACE_MARKER'], 'w') as marker:
        marker.write('completed')
os.execv(real, [real, *args])
''')
        wrapper.chmod(0o755)
        environment['PATH'] = directory + os.pathsep + environment['PATH']
        client = Client(env=environment)
        try:
            client.initialize()
            assert marker.read_text() == 'completed'
            manifest = client.manifest()
            options = next(Path(module['source']) for module in manifest['modules']
                           if module['source'].endswith('/platforms/options.zig'))
            assert 'pub const @"cpu_enabled": bool = true;' in options.read_text(), options.read_text()
            source = client.open(WORKSPACE / 'platforms/capabilities.zig')
            client.definition(source, 'platforms/cuda', WORKSPACE / 'platforms/cuda/cuda.zig', configuring=True)
        finally:
            client.close()
        assert_clean(client.process, ''.join(client.logs))
    print('PASS: terminal build cannot replace the editor configuration during publication', flush=True)
