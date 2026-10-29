"""List a verified RAR and extract only MATLAB schema/source plus readme files.

Downloaded MATLAB source is evidence to read, never code executed by this audit.
"""
from pathlib import Path, PurePosixPath
import argparse, json, shutil, subprocess
from download_blanco import NAME, MD5, digest

HERE = Path(__file__).resolve().parent
CAP = 2*1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path.home()/'violin-rne/data/external/blanco2021')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    archive = root/NAME
    assert digest(archive,'md5') == MD5, 'Verify complete archive before extracting'
    tool = shutil.which('bsdtar')
    if tool is None:
        raise RuntimeError('libarchive bsdtar with RAR5 support is required')
    listing = subprocess.check_output([tool,'-tvf',str(archive)],text=True)
    members = subprocess.check_output([tool,'-tf',str(archive)],text=True).splitlines()
    (HERE/'archive_listing.txt').write_text(listing)
    selected = []
    for member in members:
        p = PurePosixPath(member)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe archive path')
        if p.suffix == '.m' or p.name.lower().startswith('readme') or p.name == 'participants_raw_Sound_Motion.mat':
            selected.append(member)
    assert any(PurePosixPath(p).name == 'participants_raw_Sound_Motion.mat' for p in selected)
    # Primary MAT is 469,462,149 bytes per the verified inventory; allow 10 MiB
    # for selected source/readme files. Do not extract other aggregate packages.
    expected_added = 469462149 + 10*1024**2
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    missing = [p for p in selected if not (root/'extracted'/p).exists()]
    if missing and used + expected_added > CAP:
        raise RuntimeError('2 GiB local storage budget would be exceeded')
    extracted = root/'extracted'
    extracted.mkdir(exist_ok=True)
    if missing:
        subprocess.run([tool,'-xkf',str(archive),'-C',str(extracted),*missing],check=True)
    used_after = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert used_after <= CAP
    receipt = {'extractor':subprocess.check_output([tool,'--version'],text=True).strip(),
               'archive_md5_verified':MD5,'member_count':len(members),'selected_members':selected,
               'extracted_sizes':{p:(extracted/p).stat().st_size for p in selected},
               'stored_bytes_in_data_root':used_after,'storage_cap_bytes':CAP,
               'matlab_source_execution':False}
    (HERE/'extraction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
