"""Stream one Guettler-diagram session (Zenodo 10946413) member by member, keep compact per-stroke arrays.
Raw wav bytes stay in memory only; nothing raw is written to disk.
usage: python -I stream_session.py <session_subpath e.g. first_repetitions/2023-10-18> [threads]
"""
import sys, os, io, re, zlib, struct, time, urllib.request, threading, queue
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, decimate
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
from httpzip import open_remote, URL

SESSION = sys.argv[1]
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
OUT = f'/Users/jeongjihoon/violin-rne/novelty/attack_data/work/strokes/{SESSION.replace("/", "__")}'
os.makedirs(OUT, exist_ok=True)
CAL_FB = 10.8460
FS = 50000

z, _ = open_remote()
infos = [i for i in z.infolist() if f'raw_data/{SESSION}/' in i.filename and i.filename.endswith('.wav')]
def idx_of(i): return int(re.search(r'_(\d+)\.wav$', i.filename).group(1))
infos.sort(key=idx_of)
n = len(infos)
na = 30
# column-interleaved order: even acceleration columns first, then odd
order = [k for c in list(range(0, na, 2)) + list(range(1, na, 2)) for k in range(n) if (k % na) == c]
ORDER_FILE = os.environ.get('ORDER_FILE')
if ORDER_FILE:
    order = [int(v) for v in open(ORDER_FILE).read().split()]
print(SESSION, n, 'strokes', flush=True)

def fetch(info):
    a = info.header_offset
    for attempt in range(8):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{a + 30 + 1024 + info.compress_size}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                blob = r.read()
            sig, ver, flag, meth, mt, md, crc, cs, us, fl, el = struct.unpack('<IHHHHHIIIHH', blob[:30])
            assert sig == 0x04034b50
            start = 30 + fl + el
            comp = blob[start:start + info.compress_size]
            assert len(comp) == info.compress_size, (len(comp), info.compress_size)
            data = zlib.decompress(comp, -15) if info.compress_type == 8 else comp
            assert zlib.crc32(data) & 0xffffffff == info.CRC, 'crc'
            return data
        except Exception as e:
            print('retry', info.filename[-30:], attempt, repr(e)[:120], flush=True)
            time.sleep(3 * (attempt + 1))
    raise IOError('failed ' + info.filename)

def process(k, data):
    fs, x = wavfile.read(io.BytesIO(data))
    assert fs == FS and x.shape[1] == 11
    x = x.astype(np.float64) / 2**31
    fb = x[:, 0] - x[:, 1]; fb = (fb - fb[:fs].mean()) * CAL_FB
    ft = x[:, 0] + x[:, 1]; ft = (ft - ft[:fs].mean()) * CAL_FB
    bn = -(x[:, 4] + x[:, 5]) * 10.0
    bn_off = bn[:fs].mean()
    y = x[:, 9] - x[:fs, 9].mean()
    zz = x[:, 10]
    ten = np.abs(x[:, 7]) * 10 * 19.6931 + 3.0
    # stroke window
    moving = np.flatnonzero(y > 1e-4)
    i1 = moving[0]
    ymax_i = int(np.argmax(y))
    a0 = max(0, i1 - int(0.15 * fs)); a1 = min(len(y), ymax_i + int(0.02 * fs))
    sos = butter(4, 400, 'low', fs=fs, output='sos')
    def dec(v):  # 1 kHz
        return sosfiltfilt(sos, v)[::50].astype(np.float32)
    np.savez_compressed(f'{OUT}/{k + 1:04d}.npz', fb=fb[a0:a1].astype(np.float32), ft=ft[a0:a1].astype(np.float32),
                        y=y[a0:a1].astype(np.float32), seg_start=a0, i_move=i1, i_ymax=ymax_i,
                        fb_rest=fb[int(0.2 * fs):int(0.6 * fs)].astype(np.float32),
                        bn_1k=dec(bn), bn_off=bn_off, y_1k=dec(y), z_1k=dec(zz), ten_1k=dec(ten), n_total=len(y))

q = queue.Queue()
for k in order:
    pass
for k in order:
    if not os.path.exists(f'{OUT}/{k + 1:04d}.npz'):
        q.put(k)
lock = threading.Lock()
done = [0]; t0 = time.time(); total = q.qsize(); nbytes = [0]
def worker():
    while True:
        try: k = q.get_nowait()
        except queue.Empty: return
        info = infos[k]
        assert idx_of(info) == k + 1
        data = fetch(info)
        try:
            process(k, data)
        except Exception as e:
            print('process error', k + 1, repr(e)[:200], flush=True)
        with lock:
            done[0] += 1; nbytes[0] += info.compress_size
            if done[0] % 10 == 0:
                el = time.time() - t0
                print(f'{done[0]}/{total} {el/60:.1f} min {nbytes[0]/el/1e3:.0f} kB/s', flush=True)
ths = [threading.Thread(target=worker) for _ in range(THREADS)]
for t in ths: t.start()
for t in ths: t.join()
print('DONE', time.time() - t0, flush=True)
