import os
import numpy as np
from trace_r import *
cases = load_cases()
inv_rows = 0; rows = 0; inv_frac = []
for case in cases:
    lab = case['labels'].astype(np.int8); C, R, N = lab.shape
    for c in range(C):
        for r in range(R):
            row = lab[c, r]; rows += 1
            ones = np.flatnonzero(row == 1)
            if len(ones) and (row[ones[0]:] == 0).any():
                inv_rows += 1
            # fraction of (one at j, zero at k>j) among all (one, zero) pairs j<k
    lab2 = lab
print(os.environ.get('BUNDLE', 'violin').split('/')[-1], 'rows with a zero above some positive:', round(inv_rows / rows, 3))
