import sys
src = open(sys.argv[1]).read()
rep = [
("def choose(view):\n    res = _choose(view)", "TAG=['?']\ndef choose(view):\n    res = _choose(view)"),
("    if view.queries_so_far < KSEED:\n        return _fallback(view)", "    if view.queries_so_far < KSEED:\n        TAG[0]='seed'\n        return _fallback(view)"),
("            if rd is not None:\n                return rd\n\n    # ---- attainable", "            if rd is not None:\n                TAG[0]='pin'\n                return rd\n\n    # ---- attainable"),
("        if rd is not None:\n            return rd\n    if GATE and not noisy:", "        if rd is not None:\n            TAG[0]=('ver_pair' if (cb and rd[0]==b) else ('ver_sweepB' if cb else 'ver_sweepA'))+('N' if noisy else 'M')\n            return rd\n    if GATE and not noisy:"),
("                if len(ranks):\n                    return col", "                if len(ranks):\n                    TAG[0]='refs_'+who\n                    return col"),
("                    return col, int(ranks[np.abs(ranks - q).argmin()])\n    return _fallback(view)", "                    return col, int(ranks[np.abs(ranks - q).argmin()])\n    TAG[0]='fb'\n    return _fallback(view)"),
]
for a, b in rep:
    if a in src: src = src.replace(a, b)
    else: print('missing', a[:50], file=sys.stderr)
open(sys.argv[2], 'w').write(src)
