import json, sys, os
S = os.path.dirname(os.path.abspath(__file__))
names = sys.argv[2:]
out = open(sys.argv[1], 'w')
print('name', 'violin_search', 'violin_eps0.2', 'violin_eps0.4', 'violin_compat', 'violin_max_s',
      'audit_search', 'audit_eps0.2', 'audit_eps0.4', 'audit_compat', 'audit_max_s', sep='\t', file=out)
for n in names:
    row = [n]
    for k in 'VR':
        d = json.load(open(f'{S}/res/{n}.{k}.json'))
        row += [d['total'], d['totals']['eps0.2'], d['totals']['eps0.4'], d['compatible_total'], round(d['max_case_seconds'], 2)]
    print(*row, sep='\t', file=out)
