"""Offline synthetic tests only; no archive, process, or network access."""
import unittest

from disk_guard import (assess_budget, exact_catalog_entry,
                        parse_slt_listing, remaining_download_bytes)


def listing(records):
    return '7-Zip\n--\nPath = /absolute/archive.7z\nPhysical Size = 99\n----------\n' + '\n\n'.join(
        f'Path = {path}\nSize = {size}\n' for path, size in records)


class InventoryTests(unittest.TestCase):
    def test_full_folders_do_not_alias(self):
        text = listing([('A/v1/whole_1.csv', 100), ('A/v1/beta_1.csv', 5),
                        ('B/v1/whole_1.csv', 200), ('B/v1/beta_1.csv', 5),
                        ('A/v1/whole_2.csv', 80), ('A/v1/beta_2.csv', 5),
                        ('notes.txt', 10000), ('A', 0)])
        self.assertEqual(parse_slt_listing(text), {'A/v1': 190, 'B/v1': 205})

    def test_windows_separators(self):
        text = listing([('A\\v1\\whole_1.csv', 100), ('A\\v1\\beta_1.csv', 2)])
        self.assertEqual(parse_slt_listing(text), {'A/v1': 102})

    def test_missing_member_separator(self):
        with self.assertRaises(ValueError):
            parse_slt_listing('Path = A/v1/whole_1.csv\nSize = 20')

    def test_unsafe_paths_anywhere_rejected(self):
        for path in ('/etc/file', '../file', 'A/../file', 'C:/file', 'C:\\file', '//server/file', 'A/./file'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                parse_slt_listing(listing([(path, 1)]))

    def test_duplicate_normalized_member_rejected(self):
        with self.assertRaises(ValueError):
            parse_slt_listing(listing([('A/v1/whole_1.csv', 1), ('A//v1/whole_1.csv', 1)]))

    def test_incomplete_pair_rejected(self):
        with self.assertRaises(ValueError):
            parse_slt_listing(listing([('A/v1/whole_1.csv', 20)]))

    def test_invalid_selected_size_or_identity(self):
        for records in ([('A/v1/whole_1.csv', 0), ('A/v1/beta_1.csv', 1)],
                        [('A/v1/whole_1.csv', 'unknown'), ('A/v1/beta_1.csv', 1)],
                        [('A/v1/whole_01.csv', 10), ('A/v1/beta_01.csv', 1)],
                        [('A/v1/whole_0.csv', 10), ('A/v1/beta_0.csv', 1)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                parse_slt_listing(listing(records))

    def test_empty_or_unrecognized_layout_rejected(self):
        with self.assertRaises(ValueError):
            parse_slt_listing(listing([('index/1/whole.csv', 100)]))

    def test_missing_or_duplicate_size_rejected(self):
        for text in ('----------\nPath = A/v1/whole_1.csv\n',
                     '----------\nPath = A/v1/whole_1.csv\nSize = 1\nSize = 1\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_slt_listing(text)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'record': {'files': [{'key': 'high.7z', 'size': 29_000_000_000}]}}

    def test_sparse_logical_size_does_not_mean_download_complete(self):
        self.assertEqual(remaining_download_bytes(self.manifest, 'high.7z',
                         allocated_bytes=26_000_000_000, logical_bytes=29_000_000_000), 3_000_000_000)

    def test_absent_partial_requires_explicit_zero(self):
        self.assertEqual(remaining_download_bytes(self.manifest, 'high.7z',
                         allocated_bytes=0, logical_bytes=0), 29_000_000_000)
        for allocated, logical in ((None, 0), (0, None)):
            with self.subTest(allocated=allocated, logical=logical), self.assertRaises(ValueError):
                remaining_download_bytes(self.manifest, 'high.7z', allocated_bytes=allocated, logical_bytes=logical)

    def test_catalog_exact_name_only_and_no_duplicates(self):
        with self.assertRaises(ValueError):
            exact_catalog_entry(self.manifest, 'high')
        with self.assertRaises(ValueError):
            exact_catalog_entry(self.manifest, 'other.7z')
        self.manifest['record']['files'] *= 2
        with self.assertRaises(ValueError):
            exact_catalog_entry(self.manifest, 'high.7z')

    def test_budget_boundary_and_retained_pin_reflected_in_free(self):
        sizes = {'A/v1': 20_000_000_000, 'A/v2': 12_000_000_000}
        ok = assess_budget(85_000_000_000, sizes, remaining_bytes=3_000_000_000)
        self.assertTrue(ok['ok'])
        self.assertEqual(ok['headroom_bytes'], 0)
        self.assertFalse(assess_budget(84_999_999_999, sizes, remaining_bytes=3_000_000_000)['ok'])
        self.assertFalse(assess_budget(70_000_000_000, sizes, remaining_bytes=3_000_000_000)['ok'])

    def test_unknown_or_invalid_values_fail_closed(self):
        for free, sizes, remain in ((None, {'A/v1': 1}, 0),
                                   (100, {}, 0), (100, {'A/v1': None}, 0),
                                   (100, {'../v1': 1}, 0), (100, {'A/v1': 1}, None),
                                   (100, {'A/v1': 1}, -1), (True, {'A/v1': 1}, 0)):
            with self.subTest(free=free, sizes=sizes, remain=remain), self.assertRaises(ValueError):
                assess_budget(free, sizes, remaining_bytes=remain)

    def test_invalid_partial_and_size(self):
        for allocated, logical in ((1, 30_000_000_000), (-1, 1), (100, 0)):
            with self.subTest(allocated=allocated, logical=logical), self.assertRaises(ValueError):
                remaining_download_bytes(self.manifest, 'high.7z', allocated_bytes=allocated, logical_bytes=logical)
        self.manifest['record']['files'][0]['size'] = None
        with self.assertRaises(ValueError):
            remaining_download_bytes(self.manifest, 'high.7z', allocated_bytes=0, logical_bytes=0)


if __name__ == '__main__':
    unittest.main()
