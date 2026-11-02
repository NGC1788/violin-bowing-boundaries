# 현재 방법의 선행 범위 — 2026-11-02

실제 바이올린 목표/점수 열람 전에 직접 출처를 재확인했다. 아래는 원문 페이지/초록 수준 확인이며, 논문 전체를 검토하거나 전 세계의 관련 연구를 모두 배제했다는 뜻이 아니다. 현재12개 비교는 고정된 응용·기전 검증이고 새 알고리즘 확증으로 부르지 않는다.

- Menon et al., ICLR2021, [Long-tail learning via logit adjustment](https://arxiv.org/abs/2007.07314): 라벨 빈도에 기반한 logit 조정을 학습 중/학습 후에 적용한다. 우리 후보는 prefix 상태별 train Markov 분포와9개 주변 BCE를 사용하므로 목적과 조건화가 같지는 않다. 그래도 log-prior를 logits에 더하는 연산 자체로 신규성을 주장할 수 없다. plain joint와 같은 backbone/업데이트/입력의 제거 비교가 필요하다.
- Chilinski & Silva, UAI2020, [Neural Likelihoods via Cumulative Distribution Functions](https://proceedings.mlr.press/v124/chilinski20a.html): 조건부·다변량 CDF 및 tail probability 계산이 직접 선행이다. 미래의 여러 임계값 확률을 함께 출력한다는 설명만으로 새로운 방법이 되지 않는다.
- Utkarsh et al., ICLR2026, [End-to-End Probabilistic Framework for Learning with Hard Constraints](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17bae482066287ce54814514cf12f248-Abstract-Conference.html): 미분 가능한 확률 projection 층과 proper score를 이용해 제약을 학습에 통합한다. 우리 유한10패턴 head와 같은 구현은 아니지만, 제약을 만족하는 확률 학습이라는 넓은 주장은 이미 존재한다.
- Kato, 2026-10-15, [Sharp and Stable Label-Free Calibration Certificates from Logical Coherence](https://arxiv.org/abs/2610.21567): 논리적으로 관련된 확률 질의의 label-free calibration 하한을 다룬다. 초록만 확인했으며 우리의 개별 유한 입력 점수 비악화 검사와 동등하다고 판단하지 않았다. label-free coherence 인증 전반을 새롭다고 주장하지 않도록 직접 확인할 후속 문헌이다. 종료한 projection 분기를 다시 여는 이유로 사용하지 않는다.

**현재 신규성 판정:** 공동 표현·prior offset·CDF·일관성 구성 자체의 높은 방법론 신규성은 미입증이다. 검증할 의미는 공개 바이올린 원음에서 파일 내 미래 참조를 차단하고, 참여자를 통째로 분리해 미래 지표 예측과 구조 추가의 이득을 엄격하게 비교하는 데 있다. 이것도 최초 바이올린 예측 연구라는 증명은 아니다. 같은 고정 비교에서 추가 이득이 없으면 실패를 기록하고 자동 탐색을 늘리지 않는다.
