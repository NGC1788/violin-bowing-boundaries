import sys, collections
from trace import *
m = load(sys.argv[1])
cases = load_cases()
tot = collections.Counter(); totneed = collections.Counter()
for i, case in enumerate(cases):
    m.LOG.clear()
    qs = run(case, m.choose)
    if not qs: continue
    pr = set(prune(case, qs))
    tags = list(m.LOG)
    cnt = collections.Counter(tags); need = collections.Counter(t for t, q in zip(tags, qs) if q in pr)
    tot += cnt; totneed += need
    print(i, case['group'].split('|')[0], case['group'][-3:], case['epsilon'], case['expected'], len(qs), len(pr), dict(cnt))
print('ALL', dict(tot)); print('NEEDED', dict(totneed))
