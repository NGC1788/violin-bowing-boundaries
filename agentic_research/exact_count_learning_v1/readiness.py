"""Code-only preparation seals. This module cannot authorize or run fitting."""
from pathlib import Path
import argparse,hashlib,json,platform
import numpy as np
import scipy
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
LOCAL_FILES=['AGENTS.md','CONTRACT_PROPOSAL.json','model.py','preprocess.py','test_model.py','INVARIANT_OBJECTIVE.md','readiness.py']
DEPENDENCIES=['agentic_research/discovery_v3_math/decoder.py','agentic_research/violin_raw_v1/data.py']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sources():
    paths=[HERE/n for n in LOCAL_FILES]+[REPO/n for n in DEPENDENCIES]
    return {str(p.relative_to(REPO)):sha(p) for p in paths}


def versions():return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__}


def validate_scope():
    c=json.loads((HERE/'CONTRACT_PROPOSAL.json').read_text())
    d=json.loads((HERE.parent/'discovery_v2/LOOP_BUDGET.json').read_text())
    if c['new_training_allowed_now'] or c['new_math_audits_allowed'] or c['old_dev_test_truth_access_allowed']:
        raise ValueError('readiness cannot authorize fitting, standalone audits or held truth')
    for a,b in [('maximum_actual_jobs_if_later_frozen','max_jobs_per_additional_batch'),
                ('max_seconds_per_job','max_seconds_per_training_job'),('max_seconds_per_batch','max_training_seconds_per_batch')]:
        if c[a]>d[b]:raise ValueError('proposed execution exceeds existing limit')
    if c['deadline_kst']!=d['deadline_kst']:raise ValueError('deadline changed')
    parent=json.loads((HERE.parent/'discovery_v3_math/FREEZE.json').read_text())
    if sha(REPO/DEPENDENCIES[0])!=parent['sources']['decoder.py']:raise ValueError('original CDF primitive changed')
    return c


def freeze_readiness():
    target=HERE/'READINESS_FREEZE.json'
    if target.exists():raise ValueError('readiness seal exists; do not replace it')
    validate_scope()
    u=json.loads((HERE/'UNIT_TESTS.json').read_text());r=json.loads((HERE/'REVIEW.json').read_text())
    if u['failed'] or u['source_sha256']!=sources():raise ValueError('current source not tested')
    if r['verdict']!='APPROVE_PREPARATION_ONLY' or r['blockers'] or r['source_sha256']!=sources():
        raise ValueError('current preparation not approved')
    value={'status':'READY_TO_FREEZE_TRAINING_CONTRACT','execution_authorized':False,
           'sources':sources(),'versions':versions(),'review_sha256':sha(HERE/'REVIEW.json'),
           'unit_tests_sha256':sha(HERE/'UNIT_TESTS.json'),
           'actual_input_hashes':None,'actual_data_accessed':False,'actual_training_jobs':0,
           'missing_before_fitting':['reviewed runner and input allowlist','actual source-only input/row-order seals',
             'full execution contract and its freeze','atomic registration of actual attempts and failures']}
    with target.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return value


def check():
    d=json.loads((HERE/'READINESS_FREEZE.json').read_text());validate_scope()
    if d['execution_authorized'] or d['sources']!=sources() or d['versions']!=versions():raise ValueError('readiness changed')
    if d['review_sha256']!=sha(HERE/'REVIEW.json') or d['unit_tests_sha256']!=sha(HERE/'UNIT_TESTS.json'):
        raise ValueError('review or tests changed')
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze-readiness','check']);a=p.parse_args()
    result=freeze_readiness() if a.action=='freeze-readiness' else check()
    print(json.dumps({'status':result['status'],'execution_authorized':False}))
