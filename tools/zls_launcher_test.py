"""Unit checks for snapshot publication, isolation, and invalid metadata."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from zls_launcher import prepare


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name).resolve()
        self.workspace = root / 'workspace'
        self.execution = root / 'execroot'
        self.session = root / 'session'
        for path in (self.workspace, self.execution, self.session):
            path.mkdir()
        (self.workspace / 'root.zig').write_text('pub const options = @import("options");\n')
        generated = self.execution / 'bazel-out/config/bin'
        generated.mkdir(parents=True)
        (generated / 'options.zig').write_text('pub const enabled = false;\n')
        (self.execution / 'zig').touch()
        (self.execution / 'zls').touch()
        (self.execution / 'lib').mkdir()
        self.raw = {
            'schema_version': 1, 'target': 'aarch64-macos-none',
            'target_args': ['-target', 'aarch64-macos-none'],
            'settings': {'args': ['-O', 'Debug']},
            'inputs': ['root.zig', 'bazel-out/config/bin/options.zig'],
            'modules': [
                {'id': 'root', 'source': 'root.zig', 'imports': {'options': 'options'}, 'zigopts': []},
                {'id': 'options', 'source': 'bazel-out/config/bin/options.zig', 'imports': {}, 'zigopts': []},
            ],
            'compilations': [{'root': 'root', 'kind': 'exe', 'label': '//:root'}],
            'tools': {'zig': str(self.execution / 'zig'), 'zls': str(self.execution / 'zls'),
                      'zig_lib': str(self.execution / 'lib'), 'zig_version': '0.17.0'},
        }

    def prepare(self, raw=None):
        return prepare(copy.deepcopy(raw or self.raw), self.workspace, self.execution, self.session, [])

    def test_generated_sources_survive_later_bazel_configuration(self):
        path, _ = self.prepare()
        snapshot = json.loads(path.read_text())
        generated = Path(snapshot['modules'][1]['source'])
        (self.execution / 'bazel-out/config/bin/options.zig').write_text('pub const enabled = true;\n')
        self.assertIn('false', generated.read_text())
        self.assertEqual(snapshot['workspace'], str(self.workspace))
        self.assertEqual(snapshot['modules'][0]['imports'], {'options': 'options'})
        self.assertFalse((self.session / 'manifest.json.tmp').exists())

    def test_invalid_graph_is_not_published(self):
        cases = []
        raw = copy.deepcopy(self.raw)
        raw['modules'][0]['imports']['missing'] = 'unknown'
        cases.append(raw)
        raw = copy.deepcopy(self.raw)
        raw['modules'].append(raw['modules'][0])
        cases.append(raw)
        raw = copy.deepcopy(self.raw)
        raw['compilations'] = []
        cases.append(raw)
        raw = copy.deepcopy(self.raw)
        raw['schema_version'] = 99
        cases.append(raw)
        raw = copy.deepcopy(self.raw)
        raw['inputs'].append('__BAZEL_EXECUTION_ROOT__/missing.zig')
        cases.append(raw)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises((ValueError, OSError)):
                self.prepare(raw)
            self.assertFalse((self.session / 'manifest.json').exists())

    def test_source_roots_cannot_have_conflicting_import_tables(self):
        raw = copy.deepcopy(self.raw)
        raw['modules'].append(dict(raw['modules'][0], id='other', imports={}))
        with self.assertRaisesRegex(ValueError, 'ambiguous module source'):
            self.prepare(raw)

    def test_unsupported_global_options_are_explicit(self):
        raw = copy.deepcopy(self.raw)
        raw['settings']['args'].append('-mcpu=other')
        with self.assertRaisesRegex(ValueError, 'unsupported editor Zig settings'):
            self.prepare(raw)


if __name__ == '__main__':
    unittest.main()
