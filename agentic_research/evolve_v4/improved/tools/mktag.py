import sys
src = open(sys.argv[1]).read()
rep = [("def choose(view):\n    res = _choose(view)", "TAG=[None]\nLOG=[]\ndef choose(view):\n    TAG[0]='?'\n    res = _choose(view)\n    LOG.append(TAG[0])"),
("    if view.queries_so_far < KSEED:\n        return _fallback(view)", "    if view.queries_so_far < KSEED:\n        TAG[0]='seed'\n        return _fallback(view)"),
("    if s0.any():\n", "    if s0.any():\n        TAG[0]='s0'\n"),
("        if rd is not None:\n            return rd\n\n    refs = []", "        if rd is not None:\n            TAG[0]='ver_pair' if (cb and rd[0]==b) else ('ver_sweep' if cb else 'ver_sweepA')\n            return rd\n\n    refs = []"),
("            if len(ranks):\n                return col, int(ranks[np.abs(ranks - q).argmin()])", "            if len(ranks):\n                TAG[0]='refs_'+who\n                return col, int(ranks[np.abs(ranks - q).argmin()])"),
("        if rd is not None:\n            return rd\n    return _fallback(view)", "        if rd is not None:\n            TAG[0]='tail_'+kind\n            return rd\n    TAG[0]='fb_end'\n    return _fallback(view)")]
for a, b in rep:
    if a in src: src = src.replace(a, b)
    else: print('missing', a[:40], file=sys.stderr)
open(sys.argv[2], 'w').write(src)
