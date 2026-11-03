# 직접 선행과 세 후보의 서면 검토 — 2026-11-03

이 검토는 공개 집계와 기존 코드, 아래 여섯 핵심 1차 문헌에 한정한다. 다른 검색 결과를 모두 정독했다는 뜻이 아니며 최초성을 인증하는 체계적 문헌고찰도 아니다. 실제 자료·가중치 열람/hash, fitting, forward, gradient, 다운로드, 추가 수치 감사는 하지 않았다. AI 학습 방법이 주제이고 바이올린은 적용 자료다.

## A. 같은 사람 안에서 미래 순서를 학습

현재 prefix로 미래 변동량의 순서를 예측하는 보조 손실을 같은 사람의 녹음 쌍에 적용한다는 후보였다. 사람 간 차이를 줄일 수 있다는 가설은 있지만, 사람 차이가 모두 nuisance라는 근거는 없다. 녹음 쌍의 조건도 일치하지 않으며 시간적 순서를 가정할 수 없다.

| 핵심 선행 | 확인한 내용과 직접 중복 | 남는 차이와 판단 |
|---|---|---|
| [Rank-N-Contrast, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/39e9c5913c970e3e49c2df629daff636-Paper-Conference.pdf) | 연속 정답의 거리 순서로 representation을 학습하고 회귀로 연결한다. 목표 순서 감독 자체는 기존 방법이다. | 같은 사람으로 pair를 제한하는 적용 차이만으로 새로운 학습 원리나 높은 novelty를 주장할 수 없다. |
| [Subject-Aware Contrastive Learning for Biosignals, 2020](https://arxiv.org/html/2007.04871v1) | subject 정보를 사용한 contrastive 학습과 subject-adversarial 학습을 제시한다. subject를 이용한 표현 학습도 이미 직접 선행이다. | 이 자료의 미래 순서와 결합하는 것만으로 마지막 실제 학습 batch를 쓸 충분한 방법론 차이가 없다. |

**A 판정: 현재 명세는 주 방법 후보로 채택하지 않는다.** 순서 학습 전체가 무의미하거나 불가능하다는 판정은 아니다.

## B. 보조 감독으로 선택하는 함수보존 재매개화

특정 보조 단계에서는 현재 사건 확률을 보존하되, 이후 주 학습의 조건을 바꿀 수 있는 선형 adapter의 이동을 학습한다. [METHOD.md](METHOD.md)의 보장은 제한된 선형 구조의 대수적 성질이다.

| 핵심 선행 | 확인한 내용과 직접 중복 | 우리 후보의 한정된 차이 |
|---|---|---|
| [Orthogonal Gradient Descent for Continual Learning, AISTATS 2020](https://proceedings.mlr.press/v108/farajtabar20a.html) | 이전 출력의 gradient 방향을 보존하도록 새 gradient를 제한한다. 출력 보호를 위한 직교 투영은 기존 원리다. | 고정 선형 readout 앞 adapter의 **실제 유한 변화량**을 제한한다. 이 구조에서만 모든 입력의 softmax 불변성을 대수적으로 보인다. OGD의 선형 특수 경우와 겹칠 수 있으며 새로운 투영 원리라고 부르지 않는다. |
| [Hessian Null Space Continuation, arXiv:2609.38081v1](https://arxiv.org/html/2609.38081v1) | §2는 probe 기반 함수 일치 손실, 곡률의 근사 null space, steering, 함수 복원과 재선형화를 설명한다. 별도 목적을 향해 함수보존 방향으로 이동한다는 발상까지 직접 선행이다. | 우리의 제한된 adapter는 Hessian/probe 복원을 사용하지 않는다. 차별 가설은 온라인 보조 감독으로 선택한 conditioning이 이후 예측 학습에 도움이 되는가다. HNC보다 일반적·빠르거나 더 정확하다는 실험 증거는 없다. |

**B 판정: 제한된 코드 준비만 정당화한다.** 새로운 정보/표현 용량/고전 null-space 정리를 만드는 후보가 아니다. 보조 단계의 보존과 이후 단계의 성능은 별개다. 높은 신규성은 유보한다.

## C. coarse 목표가 구분하지 않는 정보를 추가 감독

9개 겹치는 사건 BCE 이외의 더 세밀한 정답을 학습하는 축이다. 단독으로는 표준 auxiliary learning/label refinement와 구분되지 않는다. B의 제어 가능한 steering label로만 사용하고 독립적인 새 원리로 세지 않는다.

| 핵심 선행 | 확인한 내용과 직접 중복 | 남는 차이와 판단 |
|---|---|---|
| [Task-Relevant Null-Space Residuals for Non-Injective Neural Mappings, arXiv:2609.37272v1](https://arxiv.org/html/2609.37272v1) | §3는 선형 집계가 버리는 성분을 generalized inverse의 null-space residual로 분리하고 downstream task가 활용하게 한다. 선형 null-space에 task 정보를 남기는 원리는 선행이다. | 그 논문의 집계 규칙 보존과 모든 최종 classifier 출력 보존은 다르다. B는 입력 residual을 더하는 대신 adapter **parameter step**을 제약한다. 이 차이를 최초성으로 확대하지 않는다. |
| [Disentangled Latent Spaces Facilitate Data-Driven Auxiliary Learning, arXiv:2310.09278v1](https://arxiv.org/html/2310.09278v1) | §4.1은 주 label 관련 subspace와 그 밖의 subspace를 분리하고 후자에서 auxiliary label을 구성한다. 주 목표 밖의 정보를 auxiliary task로 학습하는 발상은 기존이다. | B의 train-only fine label과 유한 update 보존 조건은 구체화 차이다. 단순히 두 아이디어를 합쳤다는 것으로 높은 novelty를 선언할 수 없다. |

**C 판정: 독립 주 방법 후보로는 기각한다.** B의 네 arm 모두에 같은 fine label을 주므로 그 label 자체의 필요성은 이번 비교로 식별되지 않는다.

## 검토의 결론

세 축 중 B 하나만 **CODE PREP ONLY**로 남긴다. 목적은 고전 투영의 새로운 이름을 만드는 것이 아니라, 특정한 학습 효과를 반증 가능한 대조로 분리하는 것이다. 직접 선행 대비 넓은 우월성·최초·높은 novelty·실제 예측 성공은 아직 주장하지 않는다. 마지막 실제 batch의 사용은 별도 코드·누수 시험·검토·전체 계약 동결 뒤의 결정이다.
