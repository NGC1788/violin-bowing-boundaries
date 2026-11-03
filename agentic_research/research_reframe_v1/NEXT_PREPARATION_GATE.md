# 다음 한 단위: null_conditioning_learning_v1 준비

이번 서면 후보를 실제 학습 성공으로 취급하지 않는다. 새 폴더에서만 구현하고 완료된 코드/계약/입력/결과를 수정·재실행·재동결하지 않는다. 남은 학습1batch·12job·1800초/개별180초, standalone 수치 감사0회다. 예산 증가 없이 이 준비에서는 **실제 fitting0**이다.

1. 실제 process/Git/planning·worker lock/선점을 먼저 확인하고 준비를 선점한다. `research_reframe_v1/{METHOD,PRIOR_REVIEW,DECISION,REVIEW}.md/json`과 최신 큐를 읽는다.
2. 실제 NPZ·행별 자료·가중치 열람/hash 전에 새 model/label transform/source-only allowlist loader/controller와 기능시험을 작성한다. 기존 원본은 수정하지 않고 필요한 고정 함수를 명시적으로 재사용한다. CDF/count/beam은 필요 없으며 추가 standalone math audit나 설정 탐색을 하지 않는다.
3. 구조는 같은73prefix→32 encoder, identity W, coarse10head, u 분기의 같은 common auxiliary,65finehead다. 네 arm의 구조/초기값/label/minibatch/주 손실을 맞춘다. 개인 ID는 fold/가중치 용도이며 predictor 입력이 아니다. inference는 prefix와 train 통계만 받고 future guard/mask/center/target/fine ID를 받지 않는다.
4. 작은 합성 기능시험은 실제 유한 step 투영(optimizer/decay 포함 또는 추가 단계 plain SGD로 명시), A_cP residual, softmax/9query drift, unseen synthetic x, rank cutoff, zero/random norm, nonfinite, ill-conditioned W 처리, 비선형/encoder/readout 동시 변경의 negative control, G의 u 분기/동결을 확인한다. gradient 검사는 코드 기능 확인에 한정한다. 보존 오차·rank·conditioning 실패 기준은 실제 자료/fit 전에 고정한다. 합성 학습 성능 탐색이나 실제 학습 시간을 추정하는 시험으로 확대하지 않는다.
5. Label 시험은 octile의 기존 quartile 포함·정확한 <= 규칙·65→10 fibre·empty/tied bins·invalid 유지·모든9사건 대응을 확인한다. 금지된 미래 필드나 held 정답의 poisoning은 inference 예측과 train 통계에 영향을 주면 안 된다. 관측된 허용 prefix 자체의 변경은 예측에 영향을 줄 수 있다. 허용 train fine label 변경은 auxiliary loss/gradient에 영향을 주어야 정상이다. held label 변경은 prediction/선택에 영향을 주면 안 된다. 사람 분리/ID정렬/allowlist/atomic budget/timeout/출력 봉인 검사도 포함한다.
6. 별도 소스 검토가 통과하면 같은 준비 단위 후반에만 기존 원train34명1442개 source 입력과 재사용 reference를 allowlist로 확인한다. Invalid15개와 정렬을 유지한다. olddev/test 정답은 hash도 열지 않는다. 새 원음 처리/다운로드0이다. Source는 역사적으로 노출된 탐색임을 표시한다.
7. Optimizer/epoch/보조 가중치/norm 규칙/RNG/rank cutoff/안정성 한도/최종 checkpoint/비용/봉인/실패 조건의 전체 계약과 code/input/environment를 실제fit 전에 동결한다. 주 AdamW1e-3·40epoch·batch64·seed0·CPUfloat64와 추가 plain-SGD1e-3·aux0.1은 **후보**이며 바꿀 이유가 있으면 실제 input 열람/fit 전에 문서화·독립 검토한다. score를 보고 선택하지 않는다. conditioning 숫자에 대한 사후 W 정규화는 하지 않는다.
8. 네 arm×source3fold12job만 후보이다. Markov/HGB/joint_aux/MDN/기존retrieval은 봉인 예측만 reference로 재사용하고 새로운 hidden refit0으로 명시한다. 모든 예측/가중치/trace/stats를 parent가 고정한 뒤에만 held 사건을 채점한다. parameter count/같은 표현·정보/기억 없는 inference·추가 step 계산비를 공개한다. binding 조건을 못 맞추면 숨은 fit을 추가하지 않고 blocker로 남긴다.
9. 독립 검토가 source binding/전체동결까지 확인한 뒤 준비 결과와 **다음 별도 실제 실행 한 단위**를 큐에 남긴다. 그 전 실제run을 호출하지 않는다. 첫 실제 시도/실패는 budget ledger에 먼저 소비하며 같은batch를 구조하지 않는다.

주 screen은 원래 사람 macro9query log loss의 Markov/HGB 대비2%·최소2fold이고 null은 세 matched arm 각각 대비 전체/최소2fold 이득도 필요하다. 보존만 성공하거나 reference 하나보다 나아졌다는 이유로 주 성능 성공으로 바꾸지 않는다. fine-label 필요성·다른 optimizer·OGD/HNC 일반 우월성은 이 네 arm으로 식별되지 않는다. 실패/불리한 사람·fold·비용·conditioning을 남긴다. 정당한 준비/계약을 만들 수 없으면 자동화를 다시 일시중지한다.
