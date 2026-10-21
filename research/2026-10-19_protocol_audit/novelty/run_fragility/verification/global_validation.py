"""Input and numerical edge cases for the public global API."""
from pathlib import Path
import json
import math
from exhaustive import load_file

HERE = Path(__file__).resolve().parent


def main():
    glob = load_file("global_validation_module", HERE.parent / "global_profile.py")
    bad = [
        dict(lo=2., hi=1., cost=0),
        dict(lo=0., hi=1., cost=-1),
        dict(lo=float('nan'), hi=1., cost=0),
        dict(lo=0., hi=float('nan'), cost=0),
        dict(lo=0., hi=1., cost=float('nan')),
        dict(lo=float('inf'), hi=float('inf'), cost=0),
        dict(lo=-float('inf'), hi=-float('inf'), cost=0),
        dict(lo=0., hi=1., cost=-float('inf')),
    ]
    rejected = 0
    for algorithm in (glob.optimize_intercept, glob.optimize_incompatibility):
        for option in bad:
            try:
                algorithm([[dict(option, state=0)]])
            except (TypeError, ValueError):
                rejected += 1
            else:
                raise AssertionError((algorithm.__name__, option))
    extremes = []
    for upper in (-1e308, -1.7976931348623157e308):
        answer = glob.optimize_intercept([[dict(lo=-math.inf, hi=upper, cost=0., state=0)]])
        assert answer['feasible'] and answer['cost'] == 0
        assert math.isfinite(answer['intercept']), answer
        assert answer['intercept'] <= upper
        extremes.append(dict(upper=upper, intercept=answer['intercept']))
    report = dict(status="passed", invalid_cases_rejected=rejected,
                  extreme_unbounded_cases=extremes)
    (HERE / 'global_validation_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
