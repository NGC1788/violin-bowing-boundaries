import json, sys
a = json.load(open(f'res/{sys.argv[1]}.json'))['per_case']; b = json.load(open(f'res/{sys.argv[2]}.json'))['per_case']
for i, (x, y) in enumerate(zip(a, b)):
    if x[2] != y[2]:
        print(i, x[0].split('|')[0], x[0][-3:], x[1], x[2], '->', y[2], f'{y[2]-x[2]:+d}')
