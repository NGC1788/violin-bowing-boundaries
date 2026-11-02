# 방법론 선택 결과 — 2026-11-03

**판정: JUSTIFIED_TO_PREPARE. 후보 B 하나만 별도 코드 준비로 넘긴다.** 실제 학습 실행 승인이나 높은 novelty 판정이 아니다. 원자료/NPZ/행별/가중치 열람·hash, fitting, forward, standalone 수치 감사, 데이터/모델 다운로드는 이번 단위에서0건이다.

세 후보를 핵심 직접 선행6개와 대조했다. [원문 대조](PRIOR_REVIEW.md), [선택한 목적식과 반박 대조](METHOD.md), [독립 검토](REVIEW.json).

| 후보 | 판정 | 이유 |
|---|---|---|
| A: scale-equivariant 경로 학습 | 주 방법 후보 기각 | 고정 σ하한/정규화를 수정할 필요는 있지만 RevIN·ScaleIn을 넘는 학습 원리를 제시하지 못했다. Gaussian exact_count 실패를 같은 설정의 noise floor 수정으로 구조하지 않는다. |
| B: 제공자 삭제 학습 검색 | 제한된 코드 준비 채택 | 같은 기억에서 donor 한 명의 전체 원자를 지운 최악 공동 사건 손실로 encoder를 학습한다. 일반 retrieval/무작위 삭제/query-side robust weighting과 비교 가능한 구체적 연산이 있다. 고전 weighted retrieval/minimax/Dirichlet smoothing 자체는 novelty가 아니다. |
| C: prefix-only functional meta-adaptation | 주 방법 후보 기각 | 관측 sequence의 auxiliary inner update를 미래 예측 outer loss로 학습하는 구조는 이미 Meta-Auxiliary2023 Eq11–13에 있다. 짧은 prefix 안에 실제1초context+0.25초gap+0.5초target과 같은 내부 과제를 만들 수도 없다. 기능/손실 변경만으로 새 학습법이라 할 수 없다. |

B의 출발 가설은 ‘한 donor 사람의 기억 의존을 줄이는 학습이 새로운 사람의 예측을 개선할 수 있다’이다. 현재 결과가 이 가설을 실증한 것은 아니다. 새 학습 연산의 좁은 차이를 정의했고 시험할 대조가 생긴 것이 이번 성과다. 문헌 전체를 망라하지 않았으므로 최초/높은 신규성은 주장하지 않는다.

독립 검토는 C의 중복을 지적했고, B에는 삭제 후 event support 소실·W≈1 상쇄·9query 공동 donor 삭제·full-memory 주 성능·all-zero 수렴을 필수 반박 조건으로 요구했다. 이를 METHOD.md와 다음 준비 관문에 반영했다. Uniform10pattern prior ρ=.01은 clip dead region 방지를 위한 사전 제안이고 원자 support가 없는 곳의 학습 가능성을 보장하지 않는다.

**다음 한 작업:** [NEXT_PREPARATION_GATE.md](NEXT_PREPARATION_GATE.md)의 별도 `donor_deletion_learning_v1` 코드/기능시험/검토 준비. 실제 source를 여는 것은 소스 검토 승인 뒤 전체 실행 계약을 동결할 때만 허용한다. 원음/olddev/test는 사용하지 않는다. 이 서면 검토를 반복하지 않고 다음 미완료 단계로 이동한다. 새 실제 학습은 전체 동결 뒤 별도 실행 단위다.

남은 예산은 실제 학습2batch×12job, batch1800초/job180초, standalone 수치 감사0회다. Ledger는 변경하지 않았다. 종료일2026-11-06 00:00KST를 유지한다. 현재 실험 결과와 실패를 보존하며 모든 AI 연구의 불가능성으로 확대하지 않는다.
