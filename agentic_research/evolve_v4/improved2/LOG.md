# general_policy 개선 기록 2 (cloud/general-v2-20261105) — 작성 중

- 출발점: `improved/general_policy.py` (바이올린 search 2349, AI 감사 search 3078). 재채점으로 두 숫자 모두 재현했다.
- 채점(search만): 바이올린 `colab/bundle_noeps`(guard가 `view.epsilon`도 거부), AI 감사 `colab/bundle_robust`.
- 채택 기준: 두 점수 중 하나라도 나빠지면 채택하지 않는다.
- 규칙: 라벨만(`view.revealed` 금지), `view.epsilon` 사용·역추정 금지, 사례별 상수·폴더 분기 금지, 사례당 3초 이내, holdout 사용 안 함.

## 시도 기록
(작성 중. 원자료는 `results.tsv`.)
