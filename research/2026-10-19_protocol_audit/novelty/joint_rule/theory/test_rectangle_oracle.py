"""Literal-classification check of every small synthetic arrangement cell."""
import itertools
import json
import math
import random
from pathlib import Path
from rectangle_oracle import first_run_regions, locate


def probes(scores, domain):
    low, high = domain
    endpoints = sorted({low, high, *(v for v in scores if low <= v <= high)})
    result = set(endpoints)
    for left, right in zip(endpoints, endpoints[1:]):
        if left < right:
            result.add((left + right) / 2)
            if math.nextafter(left, math.inf) <= right:
                result.add(math.nextafter(left, math.inf))
    return sorted(result)


def literal(x, y, guard, fixed, k, u, v):
    labels = [f or (g and a >= u and b < v)
              for a, b, g, f in zip(x, y, guard, fixed)]
    for s in range(len(labels) - k + 1):
        if all(labels[s:s + k]):
            return s
    return None


def run():
    fixtures = checks = rectangles = 0
    # Each state either constant, or a tied feature pair in a 2x2 arrangement.
    states = [(None, None, False, False), (None, None, False, True)] + [
        (x, y, True, False) for x in (.7, .8) for y in (.3, .5)]
    domains = [((.7, .9), (.3, .7)), ((.8, .8), (.5, .5))]
    cases = []
    for n in range(4):
        cases.extend(itertools.product(states, repeat=n))
    rng = random.Random(94021)
    for _ in range(160):
        n = rng.randint(4, 15)
        cases.append(tuple((rng.choice((.6, .7, .75, .8, .9, .95)),
                            rng.choice((.2, .3, .45, .5, .7, .9)), True, False)
                           if rng.random() < .8 else rng.choice(states[:2])
                           for _ in range(n)))
    for case in cases:
        x = [row[0] for row in case]
        y = [row[1] for row in case]
        g = [row[2] for row in case]
        f = [row[3] for row in case]
        for k in (1, 2, 3):
            for ud, vd in domains:
                regions = first_run_regions(x, y, g, f, k, ud, vd)
                for u in probes([a for a in x if a is not None], ud):
                    for v in probes([a for a in y if a is not None], vd):
                        expected = literal(x, y, g, f, k, u, v)
                        actual = locate(regions, u, v)
                        assert actual == [expected], (case, k, u, v, actual, expected)
                        checks += 1
                fixtures += 1
                rectangles += len(regions)
    result = {'status': 'PASS', 'fixtures': fixtures,
              'direct_point_checks': checks, 'rectangles': rectangles,
              'scope': 'Synthetic geometry partition only; no empirical file accessed.'}
    Path(__file__).with_name('rectangle_test_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    run()
