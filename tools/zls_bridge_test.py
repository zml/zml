"""Native configure/BSP regression fixture, using the launcher's resolved tools."""
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from zls_navigation_test import Client, WORKSPACE


def run(manifest):
    with tempfile.TemporaryDirectory(prefix='zml-bsp-fixture-') as directory:
        directory = Path(directory).resolve()
        project = directory / 'project'
        external = directory / 'external'
        project.mkdir()
        external.mkdir()
        shutil.copyfile(WORKSPACE / 'bazel/zls_bridge.zig', project / 'bridge.zig')
        (project / 'build.zig').write_text('pub const build = @import("bridge.zig").build;\n')
        for name in ('first', 'second'):
            (project / f'{name}.zig').write_text('pub const shared = @import("shared");\npub const name = "' + name + '";\n')
        (project / 'shared.zig').write_text('pub const roots = @import("root");\npub const dep = @import("external");\n')
        (external / 'root.zig').write_text('pub const child = @import("child.zig");\npub const table = @import("table");\n')
        (external / 'child.zig').write_text('pub const table = @import("table");\n')
        for name in ('old', 'new'):
            (external / f'{name}.zig').write_text(f'pub const {name}_option = true;\n')
        files = {name: project / f'{name}.zig' for name in ('first', 'second', 'shared')}
        files.update(external=external / 'root.zig', table=external / 'old.zig')
        edges = {'first': {'shared': 'shared'}, 'second': {'shared': 'shared'},
                 'shared': {'external': 'external'}, 'external': {'table': 'table'}, 'table': {}}
        snapshot = {
            'schema_version': 1, 'workspace': str(project), 'target': manifest['target'],
            'settings': manifest['settings'],
            'modules': [{'id': name, 'source': str(path), 'imports': edges[name], 'zigopts': []}
                        for name, path in files.items()],
            'compilations': [{'root': name, 'kind': 'exe', 'label': f'//:{name}'} for name in ('first', 'second')],
        }
        zig = manifest['tools']['zig']
        config = directory / 'zls.json'
        config.write_text(json.dumps({'zig_exe_path': zig, 'zig_lib_path': manifest['tools']['zig_lib'],
                                     'global_cache_path': str(directory / 'zls-cache'), 'enable_build_on_save': False}))
        environment = dict(os.environ, ZIG_GLOBAL_CACHE_DIR=str(directory / 'zig-cache'))

        def configure(value, name):
            path = directory / (name + '.json')
            path.write_text(json.dumps(value))
            env = dict(environment, ZML_ZLS_BUILD_CONFIG_PATH=str(path))
            result = subprocess.run([zig, 'build', '--print-configuration'], cwd=project, env=env,
                                    capture_output=True, text=True, timeout=90)
            return env, result

        result = subprocess.run([zig, 'build', '--print-configuration'], cwd=project, env=environment,
                                capture_output=True, text=True, timeout=90)
        assert result.returncode and 'MissingEditorMetadata' in result.stderr, result.stderr
        env, result = configure(snapshot, 'first')
        assert result.returncode == 0, result.stderr
        assert len(re.findall(r'\.tag = \.compile', result.stdout)) == 2, result.stdout[:2000]
        assert '.tag = .fail' in result.stdout
        for field, value, error in [('schema_version', 99, 'UnsupportedEditorSchema'),
                                    ('workspace', str(directory), 'EditorWorkspaceMismatch')]:
            bad = dict(snapshot, **{field: value})
            _, result = configure(bad, field)
            assert result.returncode and error in result.stderr, result.stderr
        bad = copy.deepcopy(snapshot)
        bad['modules'][0]['imports']['missing'] = 'absent'
        _, result = configure(bad, 'dangling')
        assert result.returncode and 'DanglingModuleImport' in result.stderr, result.stderr
        bad = copy.deepcopy(snapshot)
        bad['compilations'] = bad['compilations'][:1]
        _, result = configure(bad, 'uncovered')
        assert result.returncode and 'UnreachableEditorModules' in result.stderr, result.stderr
        for command in ([], ['editor-metadata']):
            result = subprocess.run([zig, 'build', *command], cwd=project, env=env,
                                    capture_output=True, text=True, timeout=90)
            assert result.returncode and 'ZML builds with Bazel' in result.stderr, result.stderr

        # No output from either compilation can exist: even BSP must not execute them.
        assert not (project / 'zig-out').exists()
        command = [manifest['tools']['zls'], '--config-path', str(config), '--enable-stderr-logs',
                   '--log-file', str(directory / 'fixture.log')]
        for version in ('old', 'new'):
            current = copy.deepcopy(snapshot)
            current['modules'][-1]['source'] = str(external / f'{version}.zig')
            env, result = configure(current, version)
            assert result.returncode == 0, result.stderr
            client = Client(command=command, cwd=project, workspace=project, env=env)
            try:
                client.initialize()
                # No ancestor build.zig or workspace document opened to warm discovery.
                child = client.open(external / 'child.zig')
                client.definition(child, '"table"', external / f'{version}.zig', offset=2, configuring=True)
                shared = client.open(project / 'shared.zig')
                result = client.at('definition', shared, '"root"', 2)
                assert {x.get('targetUri', x.get('uri')) for x in result} == {
                    (project / 'first.zig').as_uri(), (project / 'second.zig').as_uri()}, result
                assert not (project / 'zig-out').exists()
                # An unrelated file must not inherit the graph's named imports.
                unrelated = directory / 'unrelated.zig'
                unrelated.write_text('pub const table = @import("table");\n')
                other = client.open(unrelated)
                assert client.at('definition', other, '"table"', 2) is None
                nested = project / 'nested'
                nested.mkdir(exist_ok=True)
                (nested / 'root.zig').write_text('pub const leaf = @import("leaf.zig");\n')
                (nested / 'leaf.zig').write_text('pub const table = @import("table");\n')
                (nested / 'table.zig').write_text('pub const nested_option = true;\n')
                (nested / 'build.zig').write_text('''const std = @import("std");
pub fn build(b: *std.Build) void {
    const root = b.createModule(.{ .root_source_file = b.path("root.zig"), .target = b.graph.host });
    root.addImport("table", b.createModule(.{ .root_source_file = b.path("table.zig") }));
    const exe = b.addExecutable(.{ .name = "nested", .root_module = root });
    exe.step.dependOn(&b.addFail("fixture compilation must never execute").step);
    b.getInstallStep().dependOn(&exe.step);
}
''')
                leaf = client.open(nested / 'leaf.zig')
                client.definition(leaf, '"table"', nested / 'table.zig', offset=2, configuring=True)
            finally:
                client.close()
        overlapping_graphs(manifest, directory)
        print('PASS: native fixture, real root semantics, schema/graph rejection and configure cache invalidation', flush=True)


def overlapping_graphs(manifest, directory):
    # An unrelated loaded graph also uses this source with a different import.
    # Exercise both load orders, including an ancestor still configuring.
    owner = directory / 'owner'
    other = directory / 'other'
    broken = directory / 'broken'
    owner.mkdir()
    other.mkdir()
    broken.mkdir()
    (broken / 'build.zig').write_text('@compileError("fixture configuration deliberately fails");\n')
    shared = owner / 'shared.zig'
    shared.write_text('pub const dep = @import("dep");\n')
    for project in (owner, other):
        (project / 'dep.zig').write_text(f'pub const name = "{project.name}";\n')
        (project / 'root.zig').write_text('pub const shared = @import("shared");\n')
        (project / 'build.zig').write_text('''const std = @import("std");
pub fn build(b: *std.Build) void {
    const shared = b.createModule(.{ .root_source_file = .{ .cwd_relative = ''' + json.dumps(str(shared)) + ''' } });
    shared.addImport("dep", b.createModule(.{ .root_source_file = b.path("dep.zig") }));
    const root = b.createModule(.{ .root_source_file = b.path("root.zig"), .target = b.graph.host });
    root.addImport("shared", shared);
    const exe = b.addExecutable(.{ .name = "overlap", .root_module = root });
    exe.step.dependOn(&b.addFail("fixture compilation must never execute").step);
    b.getInstallStep().dependOn(&exe.step);
}
''')
    config = directory / 'overlap.json'
    config.write_text(json.dumps({'zig_exe_path': manifest['tools']['zig'],
                                 'zig_lib_path': manifest['tools']['zig_lib'],
                                 'global_cache_path': str(directory / 'overlap-cache'),
                                 'enable_build_on_save': False}))
    command = [manifest['tools']['zls'], '--config-path', str(config), '--enable-stderr-logs',
               '--log-file', str(directory / 'overlap.log')]
    for workspace, first in ((owner, other), (directory, other), (owner, owner), (broken, other)):
        client = Client(command=command, cwd=workspace, workspace=workspace)
        try:
            client.initialize()
            root = client.open(first / 'root.zig')
            client.definition(root, '"shared"', shared, offset=2, configuring=True)
            source = client.open(shared)
            client.definition(source, '"dep"', owner / 'dep.zig', offset=2, configuring=True)
        finally:
            client.close()
    print('PASS: overlapping graphs preserve source ownership in either load order', flush=True)
