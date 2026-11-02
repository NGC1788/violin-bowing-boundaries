# 직접 선행 대조 — 2026-11-03

범위는 3후보×핵심 원문2개, 총6개다. 검색 결과의 초록에서 출발해 아래의 관련 수식·방법 절을 확인했다. 원문 전체를 빠짐없이 읽거나 모든 최신 연구를 망라한 검토가 아니며 최초성을 입증하지 않는다. 자료·가중치·논문 파일을 내려받거나 실행하지 않았다. 웹 원문은 읽기만 했다. 문헌의 성능 주장을 우리 자료에 적용 가능한 성능 근거로 사용하지 않는다.

| 후보 | 핵심 원문과 확인 위치 | 이미 알려진 원리 | 우리 후보의 차이와 판단 |
|---|---|---|---|
| A: prefix scale에 맞춘 경로 학습 | [RevIN, ICLR2022](https://openreview.net/pdf?id=cGDAkQo1C0p), 원문 검색 추출의 방법 설명. 직접 열기는 OpenReview challenge로 막힘 | 인스턴스별 정규화와 복원 | 정규화와 noise floor를 고치는 데 필요하지만 새 학습 원리가 아니다. PDF 전체를 읽었다고 주장하지 않는다. |
| A | [Scale-Invariant Training, arXiv2610.07324v1, 2026-10-05](https://arxiv.org/html/2610.07324v1), §3·Theorem3.2·Table1 주석 | 정규화 공간의 homogeneous loss와 scale-equivariant gradient; 고정 stabilizer의 한계 | 우리 고정 σ하한 결함과 논문의 역정규화 loss weighting은 다른 문제다. 원문은 NLL에서 scale Jacobian 상수의 기울기 소거도 구별한다. 단순 scale fix를 새로운 발견으로 확대할 수 없다. |
| B: 제공자 삭제 학습 검색 | [RAF, AISTATS2026](https://proceedings.mlr.press/v300/tire26a.html), [arXiv2411.08249v2](https://arxiv.org/html/2411.08249v2) §4, AppendixD.4·G | 과거 context와 continuation 검색, retrieval-only/모형 결합, 검색 오염에 대한 강건성 분석 | 검색된 미래를 쓴다는 점은 기존이다. 후보의 다른 연산은 학습 중 query별 donor group 하나를 함께 삭제한 공동 확률의 최악 proper loss다. 현재 확인한 절에서 이 학습 목적은 확인하지 못했다. 부재의 증명은 아니다. |
| B | [Dr.k-NN, arXiv2006.04004v5, 2022-02-16](https://arxiv.org/html/2006.04004v5), §2.2·§4.2 | minimax weighted neighbors와 neural metric의 공동 학습, class별 Wasserstein uncertainty | 후보는 feature 공간의 class별 확률 이동 대신 한 donor group의 원자 전체를 제거·재정규화하는 유한 memory intervention을 사용한다. 불확실성 집합과 학습 loss가 다르지만 minimax·weighted retrieval 자체는 기존이다. |
| C: prefix-only functional meta-adaptation | [Meta-Auxiliary Learning for Adaptive Human Pose Prediction, AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/download/25760/25532), Eq11–13·Algorithm1 | 관측 sequence의 보조 과제 inner step을 미래 예측 outer loss로 학습한 뒤 각 test sequence에 적응 | variance/guard proxy·9query BCE로 바꾸는 것은 현재로서는 task/loss 변경이다. 새 학습 원리 후보로 기각한다. |
| C | [Towards Principled TSF-TTA, arXiv2605.17250v1, 2026-05-17](https://arxiv.org/html/2605.17250v1), §3.2–3.3 | 예측 시점까지 matured된 목표만 사용하는 시간적 적응 규약 | 미래 값이 드러난 뒤 이미 발행한 예측을 고쳐서 같은 시점 성능으로 세면 안 된다. prefix만 사용하는 조건은 필요한 누수 제약이며 novelty가 아니다. |

## 검색과 남은 불확실성

검색어에는 scale-equivariant/scaled-target training, conditional kernel/retrieval forecasting, meta-auxiliary forecasting, test-time adaptation, group deletion/leave-one-group-out retrieval, distributionally robust neighbors를 사용했다. 검색에서 현대 retrieval·kernel 추정·TTT의 여러 인접 연구가 나왔지만 여기서 상세 대조한 핵심 원문은 위6개로 제한한다. 일부 원문 접근은 실패해 공개 arXiv나 저자/학회 원문으로 확인했고 RevIN은 부분 접근 한계를 남겼다. 동명의 retrieval 논문들을 섞지 않았다: B의 RAF는 Tire et al.이고 다른 저자의 RAFT와 동일 논문이 아니다.

특히 adversarial dropout, memory editing, grouped support-set robustness 문헌 전체를 포괄하지 않았다. 따라서 **B의 특정 목적식은 시험할 차이가 있지만 높은 신규성·최초성은 미확보**다. 구현/성능이 좋아진 뒤 이 인접 선행 공백을 별도로 보완해야 한다. 작은 문헌 예산을 근거로 ‘선행이 없다’고 쓰지 않는다.
