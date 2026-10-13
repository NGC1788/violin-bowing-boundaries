"""Lockbox identity, fixed observation rules, checkpoint integrity and proof gates."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import evaluate_lockbox as ev


def config():
    diagram="2024-04-19_TypeA_sample2"
    frozen={"lockbox":{"file":diagram+".7z","record":"17782542","md5":"x"*32},
            "observation_family":{"reference_f0_hz":98.,"flyback_fraction":[.3,.4,.5,.6,.7],"periodicity":[.75,.8,.85]}}
    settings={"flyback_frac":.5,"hysteresis_sigma":3.,"f0_search_hz":[40.,160.]}
    summary={"source_trials":f"reports/collection/{diagram}/trial_audit/fixed/trials.csv",
             "thresholds_provisional":{"flyback_frac":.5,"hysteresis_sigma":3.},"f0_search_hz":[40.,160.]}
    spec={"feature_settings":settings,"expected_raw_rows":6}
    rows=[]
    for c,speed in (("v1",.05),("v2",.1),("v3",.2)):
        for trial in (1,2):
            rows.append({"condition":c,"trial":trial,"label":"helmholtz" if trial==1 else "multiple_slip",
                         "beta":.1,"beta_level":0,"force_rank":trial,"c1_window_mean":.4*trial,
                         "c2_window_mean":speed,"amplitude_std":.5,"periodicity":.99,"f0_hz":98.,
                         "flybacks_per_period":trial,"second_flyback_ratio":0. if trial==1 else .9,
                         "window_inside_plateau":True})
    return diagram,frozen,summary,spec,rows


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class EvaluateLockboxTests(unittest.TestCase):
    def test_target_labels_pitch_summary_and_floor_are_ignored(self):
        diagram,frozen,summary,spec,rows=config()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"regimes.csv";write_csv(path,rows)
            first=ev.load_target(path,summary,frozen,spec,diagram)
            for r in rows:r["label"]="no_oscillation"
            summary.update(noise_floor={"min_amplitude":1e9},conditions={"v1":{"reference_f0_hz":10000}})
            write_csv(path,rows);second=ev.load_target(path,summary,frozen,spec,diagram)
            np.testing.assert_array_equal(first["X"],second["X"])
            np.testing.assert_array_equal(first["family"],second["family"])
            self.assertEqual(second["legacy_label_disagreements"],3)

    def test_sample1_identity_and_wrong_provenance_rejected(self):
        diagram,frozen,summary,spec,rows=config()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"regimes.csv";write_csv(path,rows)
            with self.assertRaisesRegex(ValueError,"pinned sample2"):
                ev.load_target(path,summary,frozen,spec,diagram.replace("sample2","sample1"))
            summary["source_trials"]=summary["source_trials"].replace("sample2","sample1")
            with self.assertRaisesRegex(ValueError,"provenance"):
                ev.load_target(path,summary,frozen,spec,diagram)

    def test_feature_changes_partial_panels_and_unknown_speed_rejected(self):
        diagram,frozen,summary,spec,rows=config()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"regimes.csv";write_csv(path,rows)
            bad=copy.deepcopy(summary);bad["thresholds_provisional"]["hysteresis_sigma"]=2
            with self.assertRaisesRegex(ValueError,"feature setting"):
                ev.load_target(path,bad,frozen,spec,diagram)
            write_csv(path,rows[:-1])
            with self.assertRaisesRegex(ValueError,"trial count"):
                ev.load_target(path,summary,frozen,spec,diagram)
            rows[-1]["condition"]="v4";write_csv(path,rows)
            with self.assertRaisesRegex(ValueError,"unknown verified condition"):
                ev.load_target(path,summary,frozen,spec,diagram)

    def test_ambiguous_rows_retained_and_input_exclusions_applied(self):
        diagram,frozen,summary,spec,rows=config()
        rows[0]["label"]="ambiguous"
        rows[1]["c1_window_mean"]=-.1
        rows[3]["window_inside_plateau"]=False
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"regimes.csv";write_csv(path,rows)
            got=ev.load_target(path,summary,frozen,spec,diagram)
            self.assertEqual(got["counts"]["eligible"],4)
            self.assertTrue(got["nominal"][0])
            self.assertEqual(got["family"].shape,(4,15))

    def test_metadata_expected_checksum_is_not_verification(self):
        _,frozen,_,_,_=config()
        with self.assertRaisesRegex(ValueError,"whole archive verification"):
            ev.validate_receipt({"md5":frozen["lockbox"]["md5"]},frozen["lockbox"])
        receipt={"status":"verified","whole_archive_verified":True,"archive_filename":frozen["lockbox"]["file"],
                 "actual_md5":frozen["lockbox"]["md5"],"record":"17782542","bytes":20}
        ev.validate_receipt(receipt,frozen["lockbox"])
        receipt["actual_md5"]="0"*32
        with self.assertRaisesRegex(ValueError,"corrupt archive"):
            ev.validate_receipt(receipt,frozen["lockbox"])

    def test_whole_archive_verification_checks_actual_bytes(self):
        _,frozen,_,_,_=config();content=b"small test archive, no waveform"
        frozen["lockbox"]["md5"]=hashlib.md5(content).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/frozen["lockbox"]["file"];path.write_bytes(content)
            got=ev.verify_archive(path,frozen["lockbox"])
            self.assertTrue(got["whole_archive_verified"])
            path.write_bytes(content+b"corrupt")
            with self.assertRaisesRegex(ValueError,"corrupt archive"):
                ev.verify_archive(path,frozen["lockbox"])

    def test_checkpoint_and_manifest_tampering_rejected_before_unpickle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=root/"model.joblib";model.write_bytes(b"not an actual model; only integrity is tested")
            frozen={"implementation_sha256":{},"all_declared_models":{"m":{"path":"model.joblib","sha256":ev.digest(model)}}}
            frozen["freeze_sha256"]=ev.canonical_digest(frozen,"freeze_sha256")
            path=root/"freeze.json";path.write_text(json.dumps(frozen))
            ev.check_freeze(path,root)
            model.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"model checksum mismatch"):
                ev.check_freeze(path,root)
            frozen["scope"]="tampered";path.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError,"fingerprint mismatch"):
                ev.check_freeze(path,root)


if __name__=="__main__":unittest.main()
