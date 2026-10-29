"""Black-box independent audit of shared nine-rule query accounting and API boundary."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
ROOT=Path('/Users/jeongjihoon/violin-bowing-boundaries/agentic_research')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from harness.experiment import QueryOracle, simulate, POLICIES, even_order


def target(row):
    return next((i for i in range(len(row)-1) if row[i] and row[i+1]),len(row))


def done(obs):
    return all(target([1 if x==-1 else int(x) for x in row])==
               target([0 if x==-1 else int(x) for x in row]) for row in obs)


def main():
    rng=np.random.default_rng(713)
    n=50
    banks=[np.zeros((9,n),int),np.ones((9,n),int),np.tile(np.arange(n)%2,(9,1))]
    for _ in range(9):
        # All nine labels are functions of the same P/R observation.
        per=rng.uniform(.5,1,n); ratio=rng.uniform(.1,.9,n)
        banks.append(np.asarray([(per>=p)&(ratio<r) for p in (.7,.8,.9) for r in (.3,.5,.7)],int))
    cases=0
    for h in banks:
        prior=np.full((9,n),.5)
        for policy in POLICIES:
            oracle=QueryOracle(h)
            calls=[]
            def wrapped(i):
                assert i not in calls
                calls.append(i)
                return oracle.query(i)
            out=simulate(prior,wrapped,policy,np.random.default_rng(119))
            assert len(calls)==len(oracle.queried)==out['certificate_count']==len(out['queries'])
            assert calls==out['queries']
            assert np.array_equal(out['certified_start'],[target(row) for row in h])
            obs=np.full((9,n),-1)
            for i,pos in enumerate(calls):
                assert not done(obs), 'Policy should stop at first proven shared certificate'
                obs[:,pos]=h[:,pos]
            assert done(obs)
            assert np.array_equal(obs,out['observed'])
            for b,snapshot in zip((5,10,20,35,50),out['snapshots']):
                assert np.allclose(snapshot.sum(-1),1)
                if b>=len(calls):
                    assert np.array_equal(np.argmax(snapshot,axis=-1),out['certified_start'])
            # Noninterference: alter every unqueried private value; same API outputs
            # along the old path must induce exactly the same trace and conclusion.
            altered=h.copy(); unqueried=sorted(set(range(n))-set(calls))
            altered[:,unqueried]=1-altered[:,unqueried]
            alternate=QueryOracle(altered)
            again=simulate(prior,alternate.query,policy,np.random.default_rng(119))
            assert again['queries']==calls
            assert np.array_equal(again['certified_start'],out['certified_start'])
            cases+=1
    # Response aliasing and repeated query checks.
    o=QueryOracle(np.zeros((9,50),int)); response=o.query(0); response[:]=1
    assert np.all(o.query(1)==0)
    try:o.query(0)
    except ValueError:pass
    else:raise AssertionError('Repeated query accepted')
    order=even_order(10)
    report={'status':'PASS','cases':cases,'rules':9,'shared_positions':50,
            'noninterference':'Unqueried labels flipped; all traces and exact conclusions unchanged',
            'query_accounting':'One query reveals 9-rule vector and counts once; no duplicate query',
            'even_order_n10':order,'even_order_is_parity':order==list(range(0,10,2))+list(range(1,10,2)),
            'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'harness').glob('*.py')}}
    (HERE/'query_audit_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
