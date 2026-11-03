# 새 가설 검토의 결정 — 2026-11-03

**B: JUSTIFIED_TO_PREPARE_EXPLORATORY, CODE PREP ONLY.** A(같은 사람 안의 미래 순서 감독)와 C(더 세밀한 정답 감독 자체)는 현재 명세만으로 마지막 실제 batch를 쓸 방법론 차이를 정당화하지 못했다. [직접 선행](PRIOR_REVIEW.md), [수식과 대조](METHOD.md).

B의 반증 가능한 질문은 **현재 예측을 보존하는 보조 감독의 parameter 이동이, 같은 norm의 일반/무작위 이동보다 이후 미래 예측 학습에 도움이 되는가**다. 선형 readout 앞 adapter의 유한 step을 투영하면 해당 보조 단계에서 모든 입력의 softmax/사건 확률은 실수 대수상 보존된다. 이후 주 단계는 달라질 수 있다. 그러나 이는 고전 null-space와 재매개화 원리이며 새로운 정보·용량·고전 정리의 발명이 아니다. 높은 novelty·최초·예측 개선은 미입증이다.

독립 검토자는 수학적 범위와 conditioning 해석을 인정하며 제한된 코드 준비에 찬성했다. 실제 optimizer 변화량 투영, common generator의 u 분기, train-only coarse/fine 대응, arm 내부 norm matching, 누적 conditioning 진단, fine-label 자체의 필요성 미식별을 필수 조건으로 남겼다. 검토 기록은 [REVIEW.json](REVIEW.json)에 있다.

이번 단위는 서면 검토만 완료했다. 새 fitting/forward/gradient, 실제 NPZ·개인별 자료·가중치 열람/hash, 다운로드, standalone 수치 감사는0이다. 마지막 학습1batch×12job은 그대로 남긴다. 완료된 모든 이전 실험과 부정 결과·STOP은 보존한다. 사용자 재개는 이전 실패한 batch의 구조 허가가 아니다.

다음 한 단위는 [NEXT_PREPARATION_GATE.md](NEXT_PREPARATION_GATE.md)의 별도 `null_conditioning_learning_v1` 코드·작은 합성 기능시험·독립 검토다. 코드 승인이 나기 전 실제 source 파일도 열거나 hash하지 않는다. 준비 후반의 source-only binding/전체 계약 동결까지 통과해야 그 **다음 별도 실행 단위**에서 고정12fit을 검토할 수 있다. 이 결정 자체는 실제 학습 실행 허가가 아니다. 자동화는 이 준비 작업까지만 재개할 수 있다.

한 번의 source 실험 성공으로 넓은 AI 방법론 성과가 되지는 않는다. 직접 선행 대비 기여가 확인되지 않거나 준비/누수 검증이 실패하면 마지막 batch를 쓰지 않고 다시 일시중지한다. 예산·종료일(2026-11-06 00:00KST)은 늘리지 않는다.
