"""Offline safety tests; no live network or large dataset downloads."""

import contextlib
import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
import urllib.error
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

    def test_premature_eof_automatically_resumes_exact_saved_offset(self):
        self.args.retries = 2
        self.args.retry_delay = 10
        headers = {'Content-Range': f'bytes 9-{len(PAYLOAD)-1}/{len(PAYLOAD)}'}
        responses = [Response(PAYLOAD[:9]), Response(PAYLOAD[9:], 206, headers)]
        with mock.patch.object(m, 'open_url', side_effect=responses) as net:
            with mock.patch.object(m.time, 'sleep') as sleep:
                m.download(self.args, DATA)
        self.assertEqual(net.call_args_list[1].args[1], {'Range': 'bytes=9-'})
        sleep.assert_called_once_with(10)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_incomplete_read_preserves_exception_partial_bytes(self):
        self.args.retries = 1
        first = Response(b'')
        first.read = mock.Mock(side_effect=http.client.IncompleteRead(PAYLOAD[:9], len(PAYLOAD)-9))
        headers = {'Content-Range': f'bytes 9-{len(PAYLOAD)-1}/{len(PAYLOAD)}'}
        with mock.patch.object(m, 'open_url', side_effect=[first, Response(PAYLOAD[9:], 206, headers)]) as net:
            with mock.patch.object(m.time, 'sleep'):
                m.download(self.args, DATA)
        self.assertEqual(net.call_args_list[1].args[1], {'Range': 'bytes=9-'})
        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_transient_attempt_limit_and_exponential_cap(self):
        self.args.retries = 4
        self.args.retry_delay = 10
        with mock.patch.object(m, 'open_url', side_effect=TimeoutError('network timeout')) as net:
            with mock.patch.object(m.time, 'sleep') as sleep:
                with self.assertRaises(TimeoutError):
                    m.download(self.args, DATA)
        self.assertEqual(net.call_count, 5)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [10, 20, 40, 60])
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_name(FILENAME + '.download.lock').exists())

    def test_zero_retries_retains_partial_and_returns_immediately(self):
        self.args.retries = 0
        with mock.patch.object(m, 'open_url', return_value=Response(PAYLOAD[:9])) as net:
            with mock.patch.object(m.time, 'sleep') as sleep:
                with self.assertRaises(m.PrematureEOF):
                    m.download(self.args, DATA)
        self.assertEqual(net.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(self.part.read_bytes(), PAYLOAD[:9])

    def test_http_auth_and_missing_files_never_retry(self):
        for status in (401, 403, 404):
            error = urllib.error.HTTPError('https://zenodo.org/x', status, 'failure', {}, None)
            with mock.patch.object(m, 'open_url', side_effect=error) as net:
                with mock.patch.object(m.time, 'sleep') as sleep:
                    with self.assertRaises(urllib.error.HTTPError):
                        m.download(self.args, DATA)
            self.assertEqual(net.call_count, 1)
            sleep.assert_not_called()

    def test_integrity_failure_not_retried_even_with_eight_retries(self):
        self.args.retries = 8
        with mock.patch.object(m, 'open_url', return_value=Response(b'x' * len(PAYLOAD))) as net:
            with mock.patch.object(m.time, 'sleep') as sleep:
                with self.assertRaisesRegex(ValueError, 'MD5 mismatch'):
                    m.download(self.args, DATA)
        self.assertEqual(net.call_count, 1)
        sleep.assert_not_called()


class MetadataRetryTests(unittest.TestCase):
    def test_metadata_http_429_retries_then_parses(self):
        error = urllib.error.HTTPError('https://zenodo.org/x', 429, 'rate limit', {}, None)
        with mock.patch.object(m, 'open_url', side_effect=[error, Response(json.dumps(DATA).encode())]) as net:
            with mock.patch.object(m.time, 'sleep') as sleep:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = m.fetch_record('123', retries=1, retry_delay=5)
        self.assertEqual(result, DATA)
        self.assertEqual(net.call_count, 2)
        sleep.assert_called_once_with(5)

    def test_metadata_truncation_retries_but_invalid_json_does_not(self):
        raw = json.dumps(DATA).encode()
        first = Response(raw[:9], headers={'Content-Length': str(len(raw))})
        with mock.patch.object(m, 'open_url', side_effect=[first, Response(raw)]) as net:
            with mock.patch.object(m.time, 'sleep'):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(m.fetch_record('123', retries=1), DATA)
        self.assertEqual(net.call_count, 2)
        with mock.patch.object(m, 'open_url', return_value=Response(b'not json')) as net:
            with self.assertRaises(ValueError):
                m.fetch_record('123', retries=3)
        self.assertEqual(net.call_count, 1)

    def test_retry_policy_only_accepts_transient_transport_errors(self):
        for status in (408, 429, 500, 502, 503, 504):
            self.assertTrue(m.is_transient(urllib.error.HTTPError('https://zenodo.org/x', status, '', {}, None)))
        self.assertTrue(m.is_transient(urllib.error.URLError('temporary network failure')))
        self.assertFalse(m.is_transient(OSError(m.errno.ENOSPC, 'disk full')))
        self.assertFalse(m.is_transient(ValueError('wrong range or host')))
        self.assertFalse(m.is_transient(urllib.error.URLError(m.ssl.SSLCertVerificationError('bad certificate'))))
        for retries, delay in [(-1, 1), (21, 1), (1, -1), (1, float('inf')), (1, 61)]:
            with self.assertRaises(ValueError):
                m.validate_retry_policy(retries, delay)


if __name__ == '__main__':
    unittest.main(verbosity=2)
