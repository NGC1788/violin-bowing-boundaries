# 다음 한 작업: donor deletion 학습의 제한된 코드 준비

`method_redesign_v1`은 서면 검토 완료다. 이를 반복하거나 기존 exact_count_runner_v1을 재실행하지 않는다. 실제 process/lock/선점/Git/ledger를 확인하고 별도 `donor_deletion_learning_v1/`에서 진행한다.

## 이번 다음 단위에서 완료할 것

1. METHOD.md의 네 목적식(ERM/random-donor-delete/adversarial-donor-delete/query-group-DRO)을 같은 prefix encoder·보조 감독·초기 공통 parameter·minibatch·memory·prior로 구현한다. 원본 frozen 코드는 수정하지 않는다. 모델/손실/입력과 추론 memory 접근을 명시한다. Hyperparameter는 실제 정답 열람/학습 전에 이유와 함께 고정하고 grid search하지 않는다.
2. 실제 NPZ/행별 자료를 읽거나 hash하기 전에 source-only allowlist loader/controller와 작은 **합성 기능시험**을 준비한다. 이는 추가 standalone 수치 감사/속도 sweep이 아니다. 삭제 stable sum 대 직접 재계산, 10pattern mass·9query coherence, 같은 donor 공동 삭제, 동일인 전체 memory exclusion, 추론 query의 금지된 미래/held 정답 poisoning에 대한 **예측** 불변성(허용 innertrain memory label 변조에는 예측이 달라져야 하며 train query의 감독 label 변조는 loss/gradient를 바꾸는 것이 정상), row ID/순서, invalid·positive support소실·empty memory·큰logit 경계, gradient의 유한성·공통초기/aux 감독, timeout/atomicbudget/중복실행 방지를 검증한다. 모델의 근본 성능/학습 시간은 이 시험만으로 추정하지 않는다.
3. 삭제 후 raw support 소실은 prior가 숨기지 못하게 집계한다. 제안 uniform10prior ρ=.01이면 9query확률이[.001,.999]에 있고 scoreclip1e-6에 구조적으로 걸리지 않음을 수식/기능시험으로 확인한다. 모든 retrieval 대조에 같은 prior를 둔다. 미래를 보고 smoothing/floor를 고르지 않는다.
4. 별도 코드·계약 검토를 받는다. 기존 source Markov/HGB/joint_aux/mdn_aux 봉인 예측을 재사용하고 새 참조fit0을 명시한다. 직접 head의공통입력/aux/전처리/초기 parameter 대응을 입증하고 memory/비용/구조 차이는 공개한다. 네 방법×3fold=12새job를 넘기지 않는다. 같은aux를 쓰는 조건을 못 맞추면 임의 fitting을 더하지 않고 blocker로 남긴다.
5. 코드 승인 뒤 **같은 준비 단위 후반**에만 원train34명1442개의 기존 봉인 prefix/source metadata/future labels·기존 reference예측의 허용 입력 동일성을 검증하고 source/환경/입력/전체 실행 계약을 동결한다. olddev/test정답은hash도열지 않고 원음재처리/다운로드0. 전처리/문턱/priors/선택은innertrain만; heldfuture는job payload/stopper에들어가지않고 all12의full-memory와모든donor삭제반사실예측을함께봉인한뒤평가한다. Held정답으로최악donor를선택하는것은봉인뒤보조risk집계에만허용하고실제예측/모델선택에는쓰지않는다. 이 준비에서 새 fitting/forward실험은 하지 않는다.

## 계약과 다음 실행 경계

- 실제학습2batch중최대1batch를후속실행단위에서소비한다.12job/180초job/1800초batch·첫실패STOP·atomicledger시작등록·실패/timeout소비·동일batch구조금지·종료2026-11-06 00:00KST를 유지한다.
- 주 평가는 full-memory 사람macro9querylogloss이고 Markov/HGB대비2%·최소2fold를완화하지않는다. 특별한연산의이득은같은retrievalERM과random삭제대비전체loss/최소2fold개선도필요하다. QueryDRO비교와worstperson/Brier/사건별·support·all-zero·메모리집계·비용도보존한다. 삭제시험만개선된결과를미래예측성공으로승격하지않는다.
- 승인·기능시험·전체동결이완료되면다음은실제고정실행이다. 준비를계속반복하지않고새사용자승인을묻지않는다. 그러나이준비단위에서학습하지않는다. 조건이막히면구체적blocker와다음필요조치를남긴다.

공개는코드/문서/합성·aggregate/해시뿐이다. 실제memory원자·행별·개인별특징/정답/예측/가중치·원자료·토큰/CLI로그를공개하지않는다. 추가수치감사/다운로드/유료API/계정변경/main병합/원자료삭제/외부이메일금지. 최신완료/다음한작업을EXECUTION_STATUS/WORK_QUEUE에기록한다.
