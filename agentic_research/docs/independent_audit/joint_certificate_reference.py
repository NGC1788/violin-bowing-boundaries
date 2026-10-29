"""Independent exact shared-query hindsight certificate for K=2.

This is an audit/reference implementation, not part of the frozen harness.
It never chooses queries online: it uses the entire hidden label matrix.
"""
from pathlib import Path
import itertools, json, time
import numpy as np

HERE = Path(__file__).resolve().parent


def first_pair(row):
    return next((i for i in range(len(row)-1) if row[i] == row[i+1] == 1), len(row))


def minimum_joint_certificate(labels):
    rows = np.atleast_2d(np.asarray(labels, dtype=int))
    if not np.isin(rows, [0, 1]).all() or rows.shape[1] < 1:
        raise ValueError('Expected nonempty binary label matrix')
    n = rows.shape[1]
    mandatory = np.zeros(n, dtype=bool)
    edge = np.zeros(max(0, n-1), dtype=bool)
    for row in rows:
        s = first_pair(row)
        if s < n:
            mandatory[s:s+2] = True
        for t in range(min(s, n-1)):
            zeros = [i for i in (t, t+1) if row[i] == 0]
            if len(zeros) == 1:
                mandatory[zeros[0]] = True
            elif len(zeros) == 2:
                edge[t] = True
            else:
                raise AssertionError('Earlier pair cannot be all ones')
    # Minimum vertex cover on path edges, with mandatory vertices.
    # State records whether the current position has been selected.
    states = {bit: ((0,) if bit else ()) for bit in (0, 1) if bit or not mandatory[0]}
    for i in range(1, n):
        next_states = {}
        for bit in (0, 1):
            if mandatory[i] and not bit:
                continue
            options = [picked + ((i,) if bit else ()) for previous, picked in states.items()
                       if not edge[i-1] or previous or bit]
            if options:
                next_states[bit] = min(options, key=lambda a: (len(a), a))
        states = next_states
    subset = min(states.values(), key=lambda a: (len(a), a))
    return len(subset), subset


def brute_minimum(rows, outcomes):
    n = len(rows[0])
    for size in range(n+1):
        for indices in itertools.combinations(range(n), size):
            chosen = set(indices)
            if all(len(outcomes[tuple(row[i] if i in chosen else -1 for i in range(n))]) == 1
                   for row in rows):
                return size
    raise AssertionError('Impossible')


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(20261031)
    cases = 0
    for n in range(1, 8):
        outcomes = {}
        for observed in itertools.product((-1, 0, 1), repeat=n):
            missing = [i for i, v in enumerate(observed) if v == -1]
            possibilities = set()
            for fill in itertools.product((0, 1), repeat=len(missing)):
                row = list(observed)
                for i, bit in zip(missing, fill):
                    row[i] = bit
                possibilities.add(first_pair(row))
            outcomes[observed] = possibilities
        vectors = list(itertools.product((0, 1), repeat=n))
        banks = list(itertools.product(vectors, repeat=2)) if n <= 6 else []
        banks.extend([rng.integers(0, 2, (9, n)).tolist() for _ in range(100)])
        for rows in banks:
            expected = brute_minimum(rows, outcomes)
            size, subset = minimum_joint_certificate(rows)
            assert size == expected, (rows, expected, size)
            assert all(len(outcomes[tuple(int(row[i]) if i in subset else -1 for i in range(n))]) == 1
                       for row in rows)
            cases += 1
    example = np.asarray([[0, 0, 1, 1], [1, 1, 0, 0]])
    individual = [minimum_joint_certificate(row)[0] for row in example]
    joint, witness = minimum_joint_certificate(example)
    assert max(individual) < joint
    report = {'status': 'PASS', 'scope': 'K=2 shared query exact hindsight certificate; no learned policy or hidden-data acquisition',
              'cases': cases, 'enumeration': 'Every two-rule binary bank n<=6, plus 100 random nine-rule banks for each n=1..7; brute-force subset/completion reference',
              'strict_example': {'labels': example.tolist(), 'per_rule_minimum': individual,
                                 'joint_minimum': joint, 'joint_witness': witness},
              'seconds': time.perf_counter()-start}
    (HERE/'joint_certificate_results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
