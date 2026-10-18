#!/usr/bin/env python3
"""Prepare Bazel metadata, then own the ZLS process and its immutable snapshot."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile


class Launcher:
    def __init__(self):
        self.child = None
        self.signum = None

    def terminate(self, signum, _frame):
        self.signum = signum
        if self.child is not None:
            try:
                os.killpg(self.child.pid, signum)
            except ProcessLookupError:
                pass
        else:
            raise InterruptedError('editor startup terminated')

    def run(self, command, *, capture=False, env=None):
        if self.signum:
            raise InterruptedError('editor startup terminated')
        self.child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE if capture else sys.stderr,
                                      env=env, start_new_session=True)
        try:
            output, _ = self.child.communicate()
            code = self.child.returncode
        finally:
            self.child = None
        if code:
            raise subprocess.CalledProcessError(code, command)
        return output.decode().strip() if capture else None


def flags(name):
    result = json.loads(os.environ.get(name, '[]'))
    if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
        raise ValueError(f'{name} must be a JSON array of arguments')
    return result


def prepare(raw, workspace, execution_root, session, bazel_flags):
    if raw['schema_version'] != 1:
        raise ValueError('unsupported editor metadata schema')

    def resolve(path):
        path = Path(path)
        if path.is_absolute():
            return path.resolve(strict=True)
        if '..' in path.parts:
            raise ValueError(f'unsafe metadata path: {path}')
        base = execution_root if path.parts[0] in ('external', 'bazel-out') else workspace
        return (base / path).resolve(strict=True)

    # Copy generated inputs, retaining relative paths for imports and @embedFile.
    # Never hard-link: a subsequent build may replace or rewrite an output.
    copied = {}
    for name in raw['inputs']:
        source = resolve(name)
        if name.startswith('bazel-out/'):
            destination = session / 'sources' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
            copied[name] = str(destination)

    modules = {}
    sources = {}
    for module in raw['modules']:
        identity = module['id']
        if identity in modules:
            raise ValueError(f'duplicate configured module: {identity}')
        module['source'] = copied.get(module['source']) or str(resolve(module['source']))
        if module['source'] in sources:
            raise ValueError(f'ambiguous module source: {module["source"]}')
        modules[identity] = module
        sources[module['source']] = identity
    for module in modules.values():
        for dep in module['imports'].values():
            if dep not in modules:
                raise ValueError(f'dangling import: {module["id"]} -> {dep}')
    pending = [root['root'] for root in raw['compilations']]
    covered = set()
    while pending:
        identity = pending.pop()
        if identity in covered:
            continue
        covered.add(identity)
        pending.extend(modules[identity]['imports'].values())
    if covered != set(modules):
        raise ValueError(f'modules without a compilation root: {set(modules) - covered}')
    raw['tools'] = {key: str(resolve(value)) if key != 'zig_version' else value
                    for key, value in raw['tools'].items()}
    if raw['tools']['zig_version'] != '0.17.0':
        raise ValueError('this bridge requires the repository Zig 0.17.0 toolchain')
    raw.update(workspace=str(workspace), execution_root=str(execution_root), bazel_flags=bazel_flags)
    # Reject global options not represented by the public std.Build bridge.
    allowed = {'-O', 'Debug', 'ReleaseSafe', 'ReleaseFast', 'ReleaseSmall',
               '-fsingle-threaded', '-fno-single-threaded', '-fstrip'}
    unknown = set(raw['settings']['args']) - allowed
    if unknown:
        raise ValueError(f'unsupported editor Zig settings: {sorted(unknown)}')
    if raw['target_args'] != ['-target', raw['target']]:
        raise ValueError(f'unsupported editor target settings: {raw["target_args"]}')
    manifest = session / 'manifest.json'
    staging = session / 'manifest.json.tmp'
    staging.write_text(json.dumps(raw, sort_keys=True, indent=2) + '\n')
    staging.replace(manifest)
    return manifest, raw['tools']


def main():
    workspace = Path(os.environ.get('BUILD_WORKSPACE_DIRECTORY', Path(__file__).resolve().parents[1])).resolve(strict=True)
    if not (workspace / 'MODULE.bazel').is_file() or not (workspace / 'build.zig').is_file():
        raise ValueError(f'not a ZML workspace: {workspace}')
    os.chdir(workspace)
    owner = Launcher()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, owner.terminate)
    startup = flags('ZML_ZLS_BAZEL_STARTUP_FLAGS')
    build_flags = flags('ZML_ZLS_BAZEL_FLAGS')
    bazel = ['bazel', *startup]
    # Ordinary Bazel commands do not take our lock. Reserve a sibling output
    # base for editor preparation so their actions cannot overwrite its inputs.
    output_base = Path(owner.run([*bazel, 'info', 'output_base'], capture=True)).resolve(strict=True)
    editor_base = output_base.with_name(output_base.name + '-zls')
    print(f'zls: editor Bazel output base {editor_base}', file=sys.stderr, flush=True)
    bazel += [f'--output_base={editor_base}']
    # Editor builds must not repoint the checkout's convenience symlinks.
    preparation_flags = [*build_flags, '--symlink_prefix=/']
    with tempfile.TemporaryDirectory(prefix='zml-zls-') as directory:
        session = Path(directory).resolve()
        # Serialize preparation through publication, not the editor session.
        # Only launchers use this output base; serialize different configurations.
        key = hashlib.sha256(str(editor_base).encode()).hexdigest()[:20]
        with open(Path(tempfile.gettempdir()) / f'zml-zls-{os.getuid()}-{key}.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            owner.run([*bazel, 'build', *preparation_flags, '--show_result=0', '//:completion_metadata'])
            execution_root = Path(owner.run([*bazel, 'info', *preparation_flags, 'execution_root'], capture=True)).resolve(strict=True)
            manifest_path = owner.run([*bazel, 'cquery', *preparation_flags, '--output=files', '--output_groups=manifest', '//:completion_metadata'], capture=True)
            raw = json.loads((execution_root / manifest_path).read_text())
            manifest, tools = prepare(raw, workspace, execution_root, session, build_flags)
        builtin = session / 'builtin.zig'
        builtin_args = [tools['zig'], 'build-exe', '--show-builtin', '--zig-lib-dir', tools['zig_lib'],
                        *raw['target_args'], *raw['settings']['args']]
        if any('-lc' in module['zigopts'] for module in raw['modules']):
            builtin_args.append('-lc')
        builtin.write_text(owner.run(builtin_args, capture=True) + '\n')
        config = session / 'zls.json'
        config.write_text(json.dumps({
            'zig_exe_path': tools['zig'], 'zig_lib_path': tools['zig_lib'],
            'global_cache_path': str(session / 'zls-cache'),
            'builtin_path': str(builtin),
            'enable_build_on_save': False,
        }))
        environment = dict(os.environ, ZML_ZLS_BUILD_CONFIG_PATH=str(manifest))
        print(f'zls: session metadata {manifest}', file=sys.stderr, flush=True)
        owner.child = subprocess.Popen([tools['zls'], '--config-path', str(config), '--enable-stderr-logs',
                                        '--log-file', str(session / 'zls.log'), *sys.argv[1:]],
                                       env=environment, start_new_session=True)
        try:
            code = owner.child.wait()
        finally:
            # Reap/terminate any BSP process left in the server's process group.
            try:
                os.killpg(owner.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            owner.child = None
        return 128 - code if code < 0 else code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        print(f'zls: preparation failed: {error}', file=sys.stderr)
        sys.exit(128 - error.returncode if error.returncode < 0 else error.returncode)
    except (OSError, ValueError, KeyError, InterruptedError) as error:
        print(f'zls: {error}', file=sys.stderr)
        sys.exit(1)
