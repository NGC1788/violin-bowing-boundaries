"""Offline safety tests; no live network or large dataset downloads."""

import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/zenodo_catalog.py'
SPEC = importlib.util.spec_from_file_location('zenodo_catalog', SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
PAYLOAD = b'small observed bowed-string payload'
FILENAME = 'example.7z'
CHECKSUM = hashlib.md5(PAYLOAD).hexdigest()
DATA = {'id': 123, 'files': [{'key': FILENAME, 'size': len(PAYLOAD), 'checksum': 'md5:' + CHECKSUM,
        'links': {'self': 'https://zenodo.org/api/records/123/files/example.7z/content'}}]}


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.args = types.SimpleNamespace(data_dir=str(self.root), file=FILENAME, max_gib=0.001)
        self.target = self.root / 'raw/zenodo/123' / FILENAME
        self.part = self.target.with_name(FILENAME + '.part')
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()

    def tearDown(self):
        self.stdout.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_fresh_download_verifies_and_existing_skips_network(self):
        with mock.patch.object(m, 'open_url', return_value=Response(PAYLOAD)) as net:
            m.download(self.args, DATA)
            m.download(self.args, DATA)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertFalse(self.part.exists())
        self.assertEqual(net.call_count, 1)

    def test_budget_rejects_before_network_and_folder_creation(self):
        self.args.max_gib = 1e-12
        with mock.patch.object(m, 'open_url') as net:
            with self.assertRaisesRegex(ValueError, 'above'):
                m.download(self.args, DATA)
        net.assert_not_called()
        self.assertFalse(self.target.parent.exists())

    def test_resume_requires_exact_range_and_validates_full_checksum(self):
        offset = 9
        self.part.parent.mkdir(parents=True)
        self.part.write_bytes(PAYLOAD[:offset])
        headers = {'Content-Range': f'bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}',
                   'Content-Length': str(len(PAYLOAD) - offset)}
        with mock.patch.object(m, 'open_url', return_value=Response(PAYLOAD[offset:], 206, headers)) as net:
            m.download(self.args, DATA)
        self.assertEqual(net.call_args.args[1], {'Range': f'bytes={offset}-'})
        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_ignored_range_does_not_append_full_body(self):
        self.part.parent.mkdir(parents=True)
        self.part.write_bytes(PAYLOAD[:9])
        with mock.patch.object(m, 'open_url', return_value=Response(PAYLOAD, 200)):
            with self.assertRaisesRegex(ValueError, 'resume range'):
                m.download(self.args, DATA)
        self.assertEqual(self.part.read_bytes(), PAYLOAD[:9])
        self.assertFalse(self.target.exists())

    def test_wrong_checksum_never_publishes_final(self):
        bad = bytes([PAYLOAD[0] ^ 1]) + PAYLOAD[1:]
        with mock.patch.object(m, 'open_url', return_value=Response(bad)):
            with self.assertRaisesRegex(ValueError, 'MD5 mismatch'):
                m.download(self.args, DATA)
        self.assertFalse(self.target.exists())
        self.assertTrue(self.part.exists())

    def test_existing_wrong_file_is_never_overwritten(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'user data')
        with mock.patch.object(m, 'open_url') as net:
            with self.assertRaisesRegex(ValueError, 'not be overwritten'):
                m.download(self.args, DATA)
        net.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b'user data')

    def test_no_disk_space_prevents_network(self):
        with mock.patch.object(m.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=0)):
            with mock.patch.object(m, 'open_url') as net:
                with self.assertRaisesRegex(ValueError, 'Insufficient'):
                    m.download(self.args, DATA)
        net.assert_not_called()

    def test_path_and_url_guards(self):
        for filename in ('../example.7z', 'folder/example.7z', 'folder\\example.7z', '.', ''):
            with self.assertRaises(ValueError):
                m.file_spec(DATA, filename)
        for url in ('http://zenodo.org/a', 'https://evil.example/a', 'https://zenodo.org@evil.example/a', 'file:///tmp/a'):
            with self.assertRaises(ValueError):
                m.safe_url(url)

    def test_wrong_range_total_is_rejected(self):
        self.part.parent.mkdir(parents=True)
        self.part.write_bytes(PAYLOAD[:9])
        with mock.patch.object(m, 'open_url', return_value=Response(PAYLOAD[9:], 206,
                {'Content-Range': f'bytes 9-{len(PAYLOAD)-1}/{len(PAYLOAD)+1}'})):
            with self.assertRaisesRegex(ValueError, 'resume range'):
                m.download(self.args, DATA)
        self.assertEqual(self.part.read_bytes(), PAYLOAD[:9])


if __name__ == '__main__':
    unittest.main(verbosity=2)
