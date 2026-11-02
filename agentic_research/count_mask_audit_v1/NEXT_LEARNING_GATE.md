# 다음 한 단위: 정확 count decoder를 쓰는 AI 학습 방법의 준비 검토

상태: **계획만 등록, 학습 코드·계약 미완료, 실제 fitting 없음**. 별도 이름은 `exact_count_learning_v1`로 한다.

## 지킬 결론

현재 beam 수치 감사는 기준 미달·STOP이다. 수치 감사 예산2회는 모두 소진했다. 정확 count 법칙·local mask 미분은 합성 자료에서 일치했지만, 이것만으로 새로운 학습 방법이나 성능 개선이 입증되지 않는다. 느리고 부정확했던 beam을 다음 학습의 장점으로 내세우지 않는다.

AI·정보 연구를 중심에 둔다. [AI_RESEARCH_POSITION.md](../path_mask_feasibility_v1/AI_RESEARCH_POSITION.md), [LEARNING_METHOD.md](../path_mask_feasibility_v1/LEARNING_METHOD.md)를 읽고 미래 관측 누락과 비선형 통계를 예측하는 **학습 방법**을 구체화한다. 바이올린은 적용 자료다.

## 준비할 한 작업

1. 실제 프로세스·잠금·작업 큐·Git·남은 공유 예산을 확인한다. 이번 단위에서는 **원자료·NPZ·행별 정답을 읽지 않고**, 학습 명세와 필요한 코드·합성 기능 시험·별도 검토만 준비한다.
2. 같은 prefix encoder가 잠재 regime·마스크 전이·구간 평균·noise scale을 출력하고 정확 count decoder가9개 확률을 만드는 흐름을 명세한다. 실제 미래 mask/guard/path는 추론에 넣지 않는다. Prefix와 미래 사이250ms gap도 명세한다.
3. 미완료인 Gaussian mean/scale 미분과 batch 구현을 점검한다. 고전 CDF 미분을 새 방법으로 주장하지 않는다. 기존 동결된 nc-chi-square 연산을 재사용할 수 있지만 원본은 수정하지 않는다. 필요한 작고 합성적인 기능 시험은 새로운 학습 코드의 정확성 검사다. 이를 명목으로 추가 standalone 수치 감사나 속도/정확도 설정 탐색을 하지 않는다.
4. Direct joint/MDN에 동일한 미래 경로·mask 보조 감독을 주는 대조를 설계한다. Count head와 moment approximation 비교도 검토하되, 동일 생성 법칙·정보·backbone·감독·계산 예산을 명시한다. Source Markov/HGB 기준선은 유지한다.
5. 미래 경로 iid·등분산·2block mean 가정의 한계를 숨기지 않는다. 새 AI 효과는 사람 단위 예측 성능으로 입증해야 하며, 단순한 고전 구성의 적용만 남으면 높은 novelty가 아니라고 판정한다.
6. 준비 검토 판정은 `READY_TO_FREEZE_TRAINING_CONTRACT` 또는 `NO_JUSTIFIED_EXACT_COUNT_TRAINING`이다. 첫 판정도 자동 fitting 허가가 아니다. 최종 optimizer/epoch/loss weight/noise bounds와 비교·평가·예산을 동결한 별도 계약 후에만 실제 자료를 연다.

실제 비교가 정당화될 경우 원source34명1442개만 사용하고, 같은 세 참여자 fold에서 전처리·문턱·prior·선택·stopping은 innertrain만 쓴다. olddev/test 정답은hash도 열지 않는다. Invalid15개를 포함한다. 역사적 source 노출은 탐색적 한계다. 같은 노출 자료를 새로운 독립 확인시험으로 부르지 않는다.

남은 실제 학습은 최대3batch×12job, 각1,800초/개별180초,2026-11-06 00:00KST까지다. 실패/timeout도 시작 시도와 비용에 포함하고 같은 batch를 구조하지 않는다. 이 후보를 학습할 근거가 부족하면 해당 분기를 닫고 가설 은행의 다른 구체적 근거를 검토한다. 물리/음악 해석으로 주 연구를 바꾸지 않는다.
