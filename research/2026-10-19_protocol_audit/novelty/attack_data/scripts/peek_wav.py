import sys, io, time
sys.path.insert(0, '/Users/jeongjihoon/violin-rne/novelty/attack_data/scripts')
from httpzip import open_remote
import numpy as np
from scipy.io import wavfile
z, f = open_remote()
name = sys.argv[1]
t0 = time.time()
data = z.read(name)
t1 = time.time()
fs, x = wavfile.read(io.BytesIO(data)); x = x.astype(np.float64)
print('fetch s', round(t1 - t0, 2), 'MB', len(data) / 1e6, 'fs', fs, 'shape', x.shape, 'dur', x.shape[0] / fs)
np.save('/Users/jeongjihoon/violin-rne/novelty/attack_data/work/peek.npy', x.astype(np.float32))
for c in range(x.shape[1]):
    print(c + 1, 'mean %.4f std %.4f min %.4f max %.4f first1s %.4f' % (x[:, c].mean(), x[:, c].std(), x[:, c].min(), x[:, c].max(), x[:fs, c].mean()))
