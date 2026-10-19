import sys, os
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
from httpzip import open_remote
OUT = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw'
z, f = open_remote()
n = 0
for i in z.infolist():
    fn = i.filename
    if fn.endswith('/') or fn.endswith('.wav') or 'slanCM' in fn or 'natsortfiles' in fn:
        continue
    if i.file_size > 5_000_000:
        print('skip big', fn, i.file_size); continue
    dst = os.path.normpath(os.path.join(OUT, fn))
    assert dst.startswith(OUT + '/'), fn
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with z.open(i) as src, open(dst, 'wb') as d:
        d.write(src.read())
    n += 1
print('extracted', n, 'requests', f.nreq, 'MB', f.nbytes / 1e6)
