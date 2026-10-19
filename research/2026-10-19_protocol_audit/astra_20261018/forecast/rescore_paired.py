"""Post-hoc numerical consistency correction; no fitting or new data."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import benchmark as b

def main():
    out=Path(__file__).parent/'results'
    old=json.loads((out/'results.json').read_text())
    predictions=np.load(out/'predictions.npz',allow_pickle=False)
    new=deepcopy(old)
    for item in new['paired']:
        stem=f'{item["train"]}_to_{item["test"]}_{item["target_start"]:.2f}_'
        y=predictions[f'{item["test"]}_y_{item["target_start"]:.2f}']
        p0,p1=predictions[stem+item['baseline']],predictions[stem+item['candidate']]
        result=b.paired(y,p0,p1,predictions[item['test']+'_level'])
        difference=b.metrics(y,p1)['log_loss']-b.metrics(y,p0)['log_loss']
        if not np.isclose(result['candidate_minus_baseline_log_loss'],difference,atol=1e-12):
            raise AssertionError('Headline/paired metric disagreement remains')
        item.update(result)
    new['posthoc_correction']={'reason':'쌍대 손실 계산의 확률 절단값을 본문 sklearn 로그손실과 통일했다. 포화 예측이 많은 로지스틱 이력 모델의 차이가 정정됐다.',
        'fitting':'재학습 없음. 기존 예측·본문 점수 불변.',
        'original_results_sha256':b.audit.sha(out/'results.json'),
        'corrected_code_sha256':b.audit.sha(b.__file__)}
    with (out/'results_paired_corrected.json').open('x') as f:json.dump(new,f,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=='__main__':main()
