import sys, numpy as np, scipy.io as sio
BASE = '/Users/jeongjihoon/violin-rne/novelty/attack_data/raw/zenodo/raw_data/'
s = sys.argv[1]
m = sio.loadmat(BASE + s + '/matx.mat')['matx']
a, F, tt = m.T
na = 30; nF = len(a) // na
A = a.reshape(nF, na); FF = F.reshape(nF, na); T = tt.reshape(nF, na)
# sort rows by force and columns by acceleration (reverse sessions)
ro = np.argsort(FF.mean(1)); co = np.argsort(A.mean(0))
A, FF, T = A[ro][:, co], FF[ro][:, co], T[ro][:, co]
sym = lambda x: '.' if x == 30 else ('L' if x > 30 else ('#' if x >= 20 else ('o' if x >= 10 else ('+' if x >= 5 else '0'))))
print(s, 'cols acc:', ' '.join(f'{v:.2f}' for v in A.mean(0)[::3]))
for i in range(nF):
    print(f'{FF[i].mean():5.2f} ' + ''.join(sym(x) for x in T[i]))
