"""Check the extractors' archive stream-order assumption using real bundled 7-Zip."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile


def test_filtered_stream_matches_listing_and_original_members():
    root=Path.home()/'violin-rne'
    seven=Path.home()/'violin-bowing-boundaries/.local-tools/7zip-26.03/7zz'
    for script in ('ramp_meta','stroke_blocks'):
        spec=importlib.util.spec_from_file_location(script,root/'claude_extra'/f'{script}.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            contents={}
            for folder in ('sample/v2','sample/v1'):
                (p/folder).mkdir(parents=True)
                for n in (10,1,2):
                    contents[f'{folder}/whole_{n}.csv']=f'{n},0.1,2,3\n{n},0.1,4,5\n'.encode()
                    contents[f'{folder}/beta_{n}.csv']=f'{n/1000}\n'.encode()
            for name,raw in contents.items():(p/name).write_bytes(raw)
            (p/'sample/ignored.txt').write_text('not selected')
            archive=p/'fixture.7z'
            subprocess.run([str(seven),'a',str(archive),'sample','-ms=on'],cwd=p,check=True,capture_output=True)
            inventory=module.listing(archive)
            raw=subprocess.check_output([str(seven),'x','-so',str(archive),'-ir!*/whole_*.csv','-ir!*/beta_*.csv'])
            offset=0
            assert {name for name,size in inventory}==set(contents)
            for name,size in inventory:
                assert raw[offset:offset+size]==contents[name]
                offset+=size
            assert offset==len(raw)
