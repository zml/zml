#!/usr/bin/env python3
"""Standalone LSP integration probe. Run in devenv; never from a Bazel action."""
import argparse
import json
import os
from pathlib import Path
import queue
import re
import sys
import subprocess
import threading
import time
from urllib.parse import unquote, urlparse

WORKSPACE = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, command=None, env=None, cwd=None, workspace=WORKSPACE):
        self.workspace = workspace
        self.process = subprocess.Popen(command or [str(WORKSPACE / 'tools/zls.sh'), '--log-level', 'debug'],
                                        cwd=cwd or WORKSPACE, env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, start_new_session=True)
        self.messages = queue.Queue()
        self.logs = []
        self.serial = 0
        self.encoding = 'utf-16'
        self.documents = {}
        self.versions = {}
        threading.Thread(target=self.read, daemon=True).start()
        threading.Thread(target=self.drain, daemon=True).start()

    def drain(self):
        for line in self.process.stderr:
            self.logs.append(line.decode(errors='replace'))
            if os.environ.get('ZML_ZLS_TEST_VERBOSE'):
                print(self.logs[-1], end='', file=sys.stderr, flush=True)

    def read(self):
        try:
            while True:
                headers = {}
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        raise EOFError('server closed stdout')
                    if line == b'\r\n':
                        break
                    key, value = line.decode().split(':', 1)
                    headers[key.lower()] = value.strip()
                body = self.process.stdout.read(int(headers['content-length']))
                self.messages.put(json.loads(body))
        except Exception as error:
            self.messages.put(error)

    def send(self, message):
        data = json.dumps({'jsonrpc': '2.0', **message}).encode()
        self.process.stdin.write(f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)
        self.process.stdin.flush()

    def notify(self, method, params):
        self.send({'method': method, 'params': params})

    def request(self, method, params, timeout=15):
        self.serial += 1
        serial = self.serial
        self.send({'id': serial, 'method': method, 'params': params})
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.messages.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty:
                raise AssertionError(f'{method} timed out\n' + ''.join(self.logs[-100:]))
            if isinstance(message, Exception):
                raise AssertionError(f'{method}: {message}\n' + ''.join(self.logs[-100:]))
            if 'method' in message and 'id' in message:
                result = None
                if message['method'] == 'workspace/configuration':
                    result = [False if x.get('section', '').endswith('enable_build_on_save') else None
                              for x in message['params']['items']]
                self.send({'id': message['id'], 'result': result})
            elif message.get('method') == 'textDocument/publishDiagnostics':
                if message['params']['diagnostics']:
                    self.logs.append(json.dumps(message) + '\n')
                    if os.environ.get('ZML_ZLS_TEST_VERBOSE'):
                        print(json.dumps(message), file=sys.stderr, flush=True)
            elif message.get('id') == serial:
                if 'error' in message:
                    raise AssertionError(message)
                return message.get('result')

    def initialize(self):
        response = self.request('initialize', {
            'processId': os.getpid(), 'rootUri': self.workspace.as_uri(),
            'workspaceFolders': [{'uri': self.workspace.as_uri(), 'name': 'probe'}],
            'capabilities': {'general': {'positionEncodings': ['utf-8', 'utf-16']},
                             'workspace': {'configuration': True},
                             'textDocument': {'definition': {'linkSupport': True}}},
        }, timeout=600)
        self.encoding = response['capabilities'].get('positionEncoding', 'utf-16')
        self.notify('initialized', {})

    def open(self, path, text=None):
        path = path.resolve()
        text = text if text is not None else path.read_text()
        self.documents[path] = text
        self.versions[path] = 1
        self.notify('textDocument/didOpen', {'textDocument': {
            'uri': path.as_uri(), 'languageId': 'zig', 'version': 1, 'text': text}})
        return path

    def change(self, path, text):
        self.documents[path] = text
        self.versions[path] += 1
        self.notify('textDocument/didChange', {
            'textDocument': {'uri': path.as_uri(), 'version': self.versions[path]},
            'contentChanges': [{'text': text}],
        })

    def manifest(self):
        for line in self.logs:
            match = re.search(r'zls: session metadata (.*)', line)
            if match:
                return json.loads(Path(match[1]).read_text())
        raise AssertionError('launcher did not publish a session manifest')

    def position(self, path, snippet, offset=0):
        prefix = self.documents[path][:self.documents[path].index(snippet) + offset]
        line = prefix.rsplit('\n', 1)[-1]
        character = len(line.encode('utf-8')) if self.encoding == 'utf-8' else len(line.encode('utf-16-le')) // 2
        return {'line': prefix.count('\n'), 'character': character}

    def at(self, method, path, snippet, offset=0):
        return self.request('textDocument/' + method, {
            'textDocument': {'uri': path.as_uri()}, 'position': self.position(path, snippet, offset)})

    def definition(self, path, snippet, expected, offset=0, configuring=False):
        deadline = time.monotonic() + (90 if configuring else 0)
        while True:
            result = self.at('definition', path, snippet, offset)
            locations = result if isinstance(result, list) else [result] if result else []
            for location in locations:
                target = Path(unquote(urlparse(location.get('targetUri', location.get('uri'))).path)).resolve()
                if target == expected.resolve():
                    assert target.exists(), target
                    return location.get('targetSelectionRange', location.get('range'))
            if result or time.monotonic() >= deadline:
                raise AssertionError(f'{path}: {snippet}: expected {expected}, got {result}\n' + ''.join(self.logs[-100:]))
            time.sleep(0.2)

    def close(self):
        try:
            if self.process.poll() is None:
                self.request('shutdown', None, timeout=10)
                self.notify('exit', None)
                self.process.wait(timeout=10)
                assert self.process.returncode == 0, self.process.returncode
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()


def navigation(client, path):
    manifest = client.manifest()
    modules = {module['id']: module for module in manifest['modules']}

    def imported(source, name):
        module = next(module for module in modules.values() if Path(module['source']) == source)
        return Path(modules[module['imports'][name]]['source'])

    cuda = client.open(WORKSPACE / 'platforms/cuda/cuda.zig')
    capabilities = imported(cuda, 'platforms/cuda/capabilities')
    client.definition(cuda, 'platforms/cuda/capabilities', capabilities)
    builtin = next(json.loads(line.split("'builtin_path' to ", 1)[1])
                   for line in client.logs if "Set config option 'builtin_path' to" in line)
    client.definition(cuda, '"builtin"', Path(builtin), offset=2)
    builtin_settings(client)
    # ZLS may follow aliases directly; verify the actual enum declaration range.
    result = client.definition(path, 'cuda.Cuda', capabilities, offset=len('cuda.'))
    target_text = capabilities.read_text().splitlines()
    assert 'Cuda = enum' in target_text[result['start']['line']], result
    hover = client.at('hover', path, 'cuda.Cuda', len('cuda.'))
    assert hover and 'Cuda' in json.dumps(hover), hover
    original = client.documents[path]
    client.change(path, original + '\npub const __editor_probe = Cuda.\n')
    completion = client.at('completion', path, '__editor_probe = Cuda.', len('__editor_probe = Cuda.'))
    items = completion.get('items', []) if isinstance(completion, dict) else completion or []
    assert {'sm90', 'Capabilities'} <= {item['label'] for item in items}, completion
    client.change(path, original)
    client.definition(path, 'platforms.zig', WORKSPACE / 'platforms/platforms.zig')
    client.definition(path, '"std"', Path(manifest['tools']['zig_lib']) / 'std/std.zig', offset=1)
    platforms = client.open(WORKSPACE / 'platforms/platforms.zig')
    options = imported(platforms, 'platforms/options')
    client.definition(platforms, 'platforms/options', options)
    options_path = client.open(options)
    assert 'pub const' in client.documents[options_path]
    # Resolve an actual generated declaration through the importing document.
    option_match = re.search(r'platforms_options\.([A-Za-z_][A-Za-z_0-9]*)', client.documents[cuda])
    assert option_match, 'no options use found'
    result = client.definition(cuda, option_match[0], options, len('platforms_options.'))
    assert option_match[1] in options.read_text().splitlines()[result['start']['line']]
    print('PASS: transitive imports, CUDA declaration, hover/completion, std, relative imports and options', flush=True)

    # Find the module owning get.zig without relying on Bazel repository names.
    get = next(Path(module['source']).parent / 'get.zig' for module in modules.values()
               if 'tables' in module['imports'] and (Path(module['source']).parent / 'get.zig').exists())
    owner = next(module for module in modules.values()
                 if Path(module['source']).parent == get.parent and 'tables' in module['imports'])
    tables = Path(modules[owner['imports']['tables']]['source'])
    get = client.open(get)
    client.definition(get, '"tables"', tables, offset=1, configuring=True)
    table_doc = client.open(tables)
    config = imported(tables, 'build_config')
    client.definition(table_doc, '"build_config"', config, offset=1, configuring=True)
    client.notify('workspace/didChangeConfiguration', {'settings': {'zls': {'enable_build_on_save': False}}})
    client.notify('textDocument/didSave', {'textDocument': {'uri': path.as_uri()}})
    assert client.at('hover', path, 'cuda.Cuda', len('cuda.'))
    settings = ''.join(client.logs)
    assert "Set config option 'enable_build_on_save' to false" in settings
    assert "Set config option 'enable_build_on_save' to true" not in settings
    print('PASS: external source, generated transitive dependency, builtin and build-on-save configuration', flush=True)


def builtin_settings(client):
    manifest = client.manifest()
    config = next(json.loads(line.split("'builtin_path' to ", 1)[1])
                  for line in client.logs if "Set config option 'builtin_path' to" in line)
    builtin = Path(config).read_text()
    modes = {'debug': 'debug', 'release_safe': 'safe', 'release_fast': 'fast', 'release_small': 'small'}
    assert f'pub const optimize: std.lang.Optimize = .{modes[manifest["settings"]["mode"]]};' in builtin
    single_threaded = str(manifest['settings']['threaded'] == 'single').lower()
    assert f'pub const single_threaded = {single_threaded};' in builtin


def external_first(client):
    manifest = client.manifest()
    modules = {module['id']: module for module in manifest['modules']}
    owner = next(module for module in modules.values()
                 if 'tables' in module['imports'] and (Path(module['source']).parent / 'get.zig').exists())
    source = client.open(Path(owner['source']).parent / 'get.zig')
    tables = Path(modules[owner['imports']['tables']]['source'])
    client.definition(source, '"tables"', tables, offset=2, configuring=True)
    print('PASS: external source first, without opening a workspace document', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-regression', action='store_true')
    args = parser.parse_args()
    client = Client()
    try:
        client.initialize()
        path = client.open(WORKSPACE / 'platforms/capabilities.zig')
        try:
            client.definition(path, 'platforms/cuda', WORKSPACE / 'platforms/cuda/cuda.zig', configuring=True)
        except AssertionError:
            if not args.expect_regression:
                raise
            print('Confirmed original named-import navigation regression')
        else:
            assert not args.expect_regression, 'Expected named-import regression was not reproduced'
            print('PASS: direct non-root named import', flush=True)
            navigation(client, path)
            from zls_bridge_test import run
            run(client.manifest())
            from zls_lifecycle_test import run as lifecycle
            lifecycle(client)
    finally:
        client.close()


if __name__ == '__main__':
    main()
