# 실제 실행 현황 — 2026-10-30

## 2026-11-03 고정12개 학습·독립 재계산까지 완료 — 추가 학습 예산 소진 STOP

`null_conditioning_learning_v1`을 한 번 실행해12/12fit 완료, ledger54.085116초/결과 기록 직전54.059636초다. Markov0.542081 대 main_only0.541589(+0.0907%),ordinary0.546120(−0.7451%),random0.538427(+0.6741%),null0.541000(+0.1994%)로 고정2% screen 승격0개다. null은 random보다0.4779% 나쁘고 이긴 fold도1개뿐이므로 제안 연산의 추가 이득도 실패했다. Null 보조 step의 query drift는 최대9.80e-16이어서 구현상 보존과 예측 성능을 구분했다. [전체 결과](../null_conditioning_learning_v1/RESULTS.md), [원 점수](../null_conditioning_learning_v1/RESULTS.json).

독립 검토가37소스/8입력/60봉인 산출물/12receipt·weights·trace와7520fine substep을 확인했다. Source targets/guard에서 scalar event·log loss·Brier·person macro·fold·coherence·reference를 독립 재계산해 최대 차이3.33e-16이었다. [검토](../null_conditioning_learning_v1/EXECUTION_REVIEW.json). Invalid15개 유지, olddev/test 정답 read/hash0, 새 reference fit/다운로드/추가 수치 감사0이다. 함수 수준 train/prefix 분리이며 source는 역사적으로 노출된 탐색이다. 조건별 macro는 조건 내 재가중이므로 전체 loss의 가산분해가 아니다.

추가 학습3/3batch·standalone 수치 감사2/2회를 사용했다. **자동화 PAUSED**, 저장 프롬프트 일치까지 확인했다. [설정 기록](../null_conditioning_learning_v1/EXECUTION_AUTOMATION_STATUS.json). 마지막 batch와 독립 검토·종료 정리까지 마쳤으므로 현재 예산의 다음 미완료 실험은 없다. [이번 AI 탐색 종료 기록](../null_conditioning_learning_v1/CLOSURE.md). 높은 novelty/독립 성능 개선은 아직 확보하지 못했다. 같은 batch의 epoch/seed/bin/보조 가중치/cutoff 변경 구조·재실행·재동결·조용한 예산 증가를 하지 않는다. 이것은 모든 AI/후속 연구의 불가능성 판정이 아니다. 아래 선점과 실행 대기는 완료 이력이다.

## 2026-11-03 함수보존 conditioning 준비·전체동결·독립 검토 완료

`null_conditioning_learning_v1`의30개 합성 기능시험과 독립 소스 검토 뒤8개 source-only 입력/37소스/전체계약을 동결했다. 원train34명1442개·invalid15개/행정렬/참가자 분리/미래 support와 SD 재구성,3fold train-only 통계·fine/coarse 라벨·가중치·73prefix 변환의 독립 재계산이 모두 일치했다. [준비 결과](../null_conditioning_learning_v1/PREPARATION_RESULT.json), [동결](../null_conditioning_learning_v1/FREEZE.json), [동결 검토](../null_conditioning_learning_v1/FREEZE_REVIEW.json). 실제 model fit/실제자료 forward/새 score/참조 refit/다운로드/추가 standalone 수치 감사0이다. Olddev/test 정답은 hash도 열지 않았다. Synthetic gradient/유한 step 기능시험은 수행했으며 실제 성능 증거로 해석하지 않는다.

**다음 미완료 한 단위는 [고정12개 실제 학습](../null_conditioning_learning_v1/NEXT_EXECUTION_GATE.md)**이다. `.venv-mac/bin/python -m agentic_research.null_conditioning_learning_v1.runner check` 뒤 full-freeze review·state·실제 process/lock을 확인해 한 번만 `...runner run`한다. 같은 모델7583parameter의 main_only/ordinary_matched/random_matched/null_conditioned×3fold,40epoch/batch64/seed0/CPUfloat64,job180초/worker1800초다. 기존 reference 예측은 재사용하고 새fit0. Parent가12개 산출물60개를 봉인한 뒤 held 사건을 채점한다. 첫 실패/timeout은 전체 STOP이며 마지막1batch를 atomic ledger에서 소비한다. 재시드/epoch/bin/보조 가중치/cutoff 변경 구조는 하지 않는다. 같은 캠페인의 state가 이미 있으면 재실행하지 않는다.

기존 Markov/HGB 대비2%·최소2fold screen과 null의 세 matched arm 각각 대비 전체/최소2fold 이득을 유지한다. 보존 성공은 예측 성공이 아니고 source는 반복 노출 탐색이다. 새 정보/용량·고전 null-space 원리·최초·높은 novelty를 주장하지 않는다. Worker는 source 전체 라벨을 검증하므로 train/prefix 분리는 함수 수준이며 OS 보안 격리가 아니다. 자동화 상태는 [설정 기록](../null_conditioning_learning_v1/AUTOMATION_STATUS.json)에 남긴다. 아래의 준비 선점/‘최신’/실행 대기는 과거 이력이다.

## 2026-11-03 새 가설 서면 검토 완료 — 다음은 코드 준비만

사용자 재개 요청 뒤 `research_reframe_v1`에서 세 축·직접 선행6개를 검토했다. **함수보존 conditioning B만 CODE PREP ONLY**다. [수식](../research_reframe_v1/METHOD.md), [직접 선행](../research_reframe_v1/PRIOR_REVIEW.md), [결정](../research_reframe_v1/DECISION.md), [독립 검토](../research_reframe_v1/REVIEW.json). 선형 adapter의 실제 추가 step을 nullspace로 제한하면 해당 단계의 coarse 확률이 보존되지만, W는 새 정보/용량을 추가하지 않으며 이후 예측 개선·높은 novelty는 미입증이다. 현재 서면 단위 fit/forward/gradient/실제 자료·가중치 접근/hash/다운로드/추가 수치 감사0, 남은 실제 학습1batch와 종료일은 그대로다. 이전 donor의 부정 결과와 NO_JUSTIFIED 판정은 그 분기의 이력으로 보존한다.

**다음 미완료 한 단위:** [NEXT_PREPARATION_GATE.md](../research_reframe_v1/NEXT_PREPARATION_GATE.md)의 별도 `null_conditioning_learning_v1` 코드·합성 기능시험·독립 검토. 코드 승인 전 실제 source 자료/hash0, 승인 뒤 같은 준비 단위 후반에만 source-only 입력/전체 계약 동결, 이 준비의 실제fit0이다. 네 matched arm×source3fold12job만 후보이며 실제 실행은 준비/전체동결 뒤 별도 단위다. 원래 Markov/HGB 대비2%·최소2fold screen과 matched-arm 추가 이득 기준을 유지한다. 자동화 설정은 [기록](../research_reframe_v1/AUTOMATION_STATUS.json)에 남긴다. 아래 선점·실행 대기·‘최신’은 과거 이력이며 재실행 지시가 아니다.

## 2026-11-03 저장결과 진단완료: 이번 학습 탐색 일시중지

`donor_postmortem_v1`에서고정사람가중치로5개비교의fold×outcome×event기여를재구성했다. 제안adversarial−ERM의손실+.00092834는음성개선−.00307849와양성악화+.00400683의합이었다. 실제양성event예측확률은평균.5021%p낮아졌다. 삭제시사건확률민감도는34명모두의평균에서줄었지만정답기반삭제risk는+.00002388악화했다. Donor entropy와loss변화상관으로인과기전을확정하지않는다. [진단](../donor_postmortem_v1/CAUSE_ANALYSIS.md), [집계](../donor_postmortem_v1/RESULTS.json), [독립검토](../donor_postmortem_v1/REVIEW.json). 독립재계산불일치0건,새fit/forward/gradient/자료·문헌탐색/다운로드/수치감사0이다.

**NO_JUSTIFIED_NEXT_LEARNING**이다. 일반검색/queryDRO의작은1.34–1.50%개선은보존하지만2%screen미달이며제안연산성공으로바꾸지않는다. 현재근거로높은novelty가있는다음학습을정당화하지못해자동화를일시중지한다. 앱설정에서도PAUSED·종료프롬프트일치를확인했다. [설정기록](../donor_postmortem_v1/AUTOMATION_STATUS.json). 남은학습1batch는미사용으로남기고기존STOP캠페인/epoch/seed/temp/prior/성공기준을바꿔구조하지않는다. 이번분기에추가등록된작업은없다. 이는모든AI/retrieval연구의불가능성을뜻하지않는다. 아래실행/진단대기와선점은완료이력이다.

## 2026-11-03 donor deletion12학습 완료·STOP — 승격0

12/12fit을82.473초에완료했다. Markov.542081 대비ERM.534838(+1.3361%),random.534964(+1.3128%),adversarial.535767(+1.1648%),queryDRO.533974(+1.4956%)다. 작은개선은관찰했지만고정2%screen은모두실패했다. 특히제안adversarial은ERM보다.1736%나쁘고두matchedcontrol보다나은fold는1개뿐이어서operator추가이득도기각이다. [전체결과](../donor_deletion_learning_v1/RESULTS.md). 같은batch를구조·재실행하지않는다. 독립재계산도27소스/7입력/72산출물·주/보조점수불일치0건이었다. [검토](../donor_deletion_learning_v1/EXECUTION_REVIEW.json). 자동화ai는저장산출물진단으로30분ACTIVE갱신·프롬프트일치를확인했다. [설정](../donor_deletion_learning_v1/EXECUTION_AUTOMATION_STATUS.json).

다음한작업은[NEXT_POSTMORTEM_GATE.md](../donor_deletion_learning_v1/NEXT_POSTMORTEM_GATE.md)의별도저장산출물진단이다. 같은사람가중치로gap기여·기억분산·저장곡선만분석하고새fit/forward/자료탐색/추가수치감사0이다. 원자료·개인별값은공개하지않는다. 마지막학습1batch는정당화된별도가설이있을때만쓰며근거없으면자동화를일시중지한다. 성능/높은novelty/독립확증의성공으로과장하지않는다. 아래실행대기/선점은완료이력이다.

## 2026-11-03 donor deletion 실행 준비·전체동결 완료

`donor_deletion_learning_v1`의28합성기능시험·별도코드검토를완료했다. 검토가요구한사건빈도/entropy/저장·inference memory비용과DRO근사표현·trace구분을보완했고,stale검토해시가실제입력열람전에차단되는것도확인후재검토했다. 승인뒤7source입력의동일성/34명1442개·invalid15개/행정렬·목표·guard를검증하고27소스/환경/전체계약을동결했다. [준비결과](../donor_deletion_learning_v1/PREPARATION_RESULT.json), [코드/규약](../donor_deletion_learning_v1/README.md). 실제새fit0·참조fit0·실제forward0·추가수치감사0·다운로드0·olddev/test정답열람/hash0이다. 동결뒤별도검토도27소스/7입력/검토·시험·영수증의불일치0건이었다. [동결검토](../donor_deletion_learning_v1/FREEZE_REVIEW.json). 자동화ai는고정실행단계로30분ACTIVE·저장프롬프트일치를확인했다. [설정기록](../donor_deletion_learning_v1/AUTOMATION_STATUS.json).

**다음 한 작업은고정12개실제학습:** `.venv-mac/bin/python -m agentic_research.donor_deletion_learning_v1.runner check`로전체동결을확인한뒤state/다른worker가없을때만 `...runner run`을한번시작한다. 네retrieval목적×3fold,40epoch/batch64/seed0/CPUfloat64,job180초/worker1800초/첫실패STOP이다. 기존Markov/HGB/joint_aux/mdn_aux봉인예측은재사용하며새참조fit0. Parent가fullmemory와모든donor반사실예측·가중치·trace·stats·memory·영수증72개산출물을고정하고all12 OUTPUT_SEAL뒤에만held사건을평가한다. 실제학습비용은미검증이며합성시험시간으로예상하지않는다.

이미state가있으면같은batch를다시시작하지않고RUNNING은실제process/lock을확인해관찰만하며완료/실패는결과검토로넘어간다. orphan/중단도소비된시도로남긴다. Epoch/seed/prior/temp/학습률/기준을바꾸어구조하지않는다. 새방법성공/높은novelty/독립확증은아직미입증이다. 아래준비선점/서면검토다음단계는완료이력이다.

## 2026-11-03 새 AI 학습 가설 선택: 제공자 삭제에 강건한 검색

`method_redesign_v1`에서3후보·직접선행6개를대조했다. A(scale/σ하한수정)와C(prefix-only meta적응)는기존학습원리를넘는차이가없어주방법후보로기각했다. B는**query별기억에서donor한명의원자를함께삭제한최악9query손실로검색encoder를학습**하는가설로좁혔다. [선택결과](../method_redesign_v1/DECISION.md), [목적식/반박조건](../method_redesign_v1/METHOD.md), [직접선행](../method_redesign_v1/PRIOR_REVIEW.md). 신규성/성능성공은아직아니며실제 fitting/forward/NPZ·행별자료열람·hash/다운로드/standalone수치감사0. 이전exact12개결과와STOP은보존했다. 독립반박검토도JUSTIFIED_TO_PREPARE·blocker0으로완료했고,금지된held정답과허용memorylabel의구분·retrieval만의삭제risk·반사실예측사전봉인을보완했다. [검토기록](../method_redesign_v1/REVIEW.json). 기존자동화ai는새준비단계로30분ACTIVE·저장프롬프트일치를확인했다. [설정기록](../method_redesign_v1/AUTOMATION_STATUS.json).

**다음 한 작업:** [NEXT_PREPARATION_GATE.md](../method_redesign_v1/NEXT_PREPARATION_GATE.md)의 별도 `donor_deletion_learning_v1` 코드/합성기능시험/별도검토, 통과뒤같은준비단위후반의source-only입력·전체계약동결. 그전실제NPZ/hash0·이번준비새학습0. 일반검색/무작위삭제/최악삭제/query-sideDRO×3fold12새job만후보이며Markov/HGB/과거joint_aux·mdn_aux는봉인예측재사용(참조fit0). 동일정보/보조감독·전처리·common초기값과배포memory/비용차이를명시한다. 삭제후support소실·prior·stable재정규화·같은donor공동삭제·미래변조불변성·heldlabel격리부터검증한다. 전체동결후별도실행단위이며서면선택/옛학습을반복하지않는다. 남은실제학습2batch·수치감사0. 아래현재선점/실행대기문구는완료이력이다.

## 2026-11-03 고정12개 실제 학습·독립 검토 완료 — 이 batch STOP

`exact_count_runner_v1`을 한 번 실행해12/12개를365.468초에 완료했다. Markov **0.542081**, HGB0.581722, joint_aux **0.541763**(+0.0586%), mdn_aux0.570751, exact_count3.029807, moment_count1.924280이다. 고정2%개선screen 승격은0개이며 높은novelty/성능개선은아직미입증이다. [전체결과](../exact_count_runner_v1/RESULTS.md), [독립검토](../exact_count_runner_v1/EXECUTION_REVIEW.json). 소스18/입력6/산출물60 및원점수 재계산 불일치0건, 34명1442개·invalid15개유지·olddev/test정답열람/hash0.

**구체적 실패 원인:** exact의정규화+고정σ하한.05가 작은pitch문턱을표현하지못해9질의중6개가항상clip아래에머무르고주BCE기울기가0이다. 사후모델용량 box-oracle loss하한도2.927144(중앙χ²)/2.700568(보수적Chernoff)로Markov보다높다. 우리의설계결함이며데이터손상이나모든count/AI방법의불가능성은아니다. moment실패까지같은이유로확정하지않는다. [집계진단](../exact_count_runner_v1/CAPACITY_DIAGNOSIS.json).

**다음 한 작업:** [NEXT_METHOD_GATE.md](../exact_count_runner_v1/NEXT_METHOD_GATE.md)의 별도 `method_redesign_v1` 서면방법론선택. 새학습/forward/NPZ·행별정답열람/다운로드/standalone수치감사0. 최대3가설/직접선행최대6개로 기존방법과의학습원리차이·표현/기울기조건·반박대조를검토하고최대1개만채택한다. 단순noisefloor수정은novelty가아니다. 정당하면다음별도코드준비,없으면자동화일시중지. 남은학습2batch×12job·수치감사0회. 기존학습/수치STOP결과는재실행·구조하지않는다. 아래실행대기/실행중/준비완료문구는과거이력이다. 기존자동화ai를30분ACTIVE로갱신하고저장프롬프트일치를확인했다. [새설정기록](../exact_count_runner_v1/EXECUTION_AUTOMATION_STATUS.json).

## 자동 루프 실행 단계로 갱신 — 2026-11-03

기존 `ai`를 완료된 준비를 반복하지 않고 고정12개 실제 학습을 실행하도록 갱신했다. 앱 응답 및 ACTIVE·30분간격·저장프롬프트 일치를 확인했다. [설정 기록](../exact_count_runner_v1/AUTOMATION_STATUS.json). 실행 중이면 기존worker만 관찰하고 재시작하지 않는다.

## 2026-11-03 전체 실행 계약 동결 완료: 다음은 실제 고정12개 학습

`exact_count_runner_v1`의 실행기·source-only loader·예산/종료 controller·결과 봉인을18합성 기능시험과 별도 검토로 완료했다. 초기 검토에서 parent출력봉인·중복Markovfit·watchdog소유/정리·checkpoint임시파일 문제를 찾아 실제입력열람 전에 모두 수정했다. [검토](../exact_count_runner_v1/REVIEW.json).

코드 승인 뒤에만6개 source 입력(학습prefix/meta/future와 기존HGB예측3개)을 열고 hash·행정렬·34명1442개·invalid15개·연속목표/guard/시간지지를 확인했다. olddev/test 정답은hash도 열지 않았고 raw음원·실제fitting·추가수치감사0건이다. [전체동결](../exact_count_runner_v1/FREEZE.json)의 SHA는 `aa7f4dc023a3885db4aab19c52e113ed9af765850621e8bb661481c3d45abfa3`, 상태 FROZEN_READY_FOR_BOUNDED_TRAINING이다. [준비결과](../exact_count_runner_v1/PREPARATION_RESULT.json).

**다음 한 작업은 준비 반복이 아니라 실제 실행:** `.venv-mac/bin/python -m agentic_research.exact_count_runner_v1.runner check` 후 다른worker/state가 없으면 `...runner run`을 한 번 실행한다. 네 head×3fold12neural jobs·40epochs/batch64/seed0/180초job/1800초worker 고정. HGB새fit0·sealed예측재사용, analyticMarkov3회는worker시간에 포함한다. 모든12예측을parent가고정한 OUTPUT_SEAL 뒤에만held-event score를계산한다. 첫실패는batchSTOP, 재실행/구조/불완전승격을하지않는다. 다음실행승인은이미사용자연구범위이며새확인을묻지않는다. 아직학습성능/novelty는미입증이다. 아래준비선점은완료이력이다.

## 자동 루프 갱신 — 2026-11-03

기존 `ai`를 새 실행기·전체계약 동결 준비로 갱신하고 ACTIVE·30분 간격·프롬프트 일치를 확인했다. [확인 기록](../exact_count_learning_v1/AUTOMATION_STATUS.json). 준비 봉인은 실행 허가가 아니며 실제 입력을 여는 시점은 실행기 검토 통과 뒤로 제한했다.

## 2026-11-03 AI 학습 코드 준비 완료: 실제 학습은 아직0건

`exact_count_learning_v1`의 batch countDP·Gaussian contrast/scale 역전파·prefix-only 입력·동일 보조감독 네 모델을 구현했다. 직접 joint/MDN은4414개, exact/moment count는4084개 파라미터이며 추가330개 free-head를 공개했다. [준비 결과](../exact_count_learning_v1/PREPARATION_RESULT.json), [코드/명세](../exact_count_learning_v1/README.md).

실제 자료 없이 학습 목표의 nuisance를 수정했다. 원시 미래 레벨 회귀 대신 정규화된 Gaussian contrast 밀도와 pooled within-record scale을 사용하며 자유도n-1을 맞춘다. 미래 mask/center는 학습 label 변환에만 있고 추론 입력이 아니다. 모든 방법에 같은 aux를 준다. 이는 고전 error-contrast 원리의 적용이며 그 자체가 novelty가 아니다. [목표 명세](../exact_count_learning_v1/INVARIANT_OBJECTIVE.md).

합성 기능시험15개가2.01초에 통과했고 별도 코드/수학검토는 APPROVE_PREPARATION_ONLY다. 실제 원음/NPZ/행별 정답/가중치 열람·hash0, optimizer step·실제 fitting·추가 standalone 수치감사0. [코드 준비 봉인](../exact_count_learning_v1/READINESS_FREEZE.json)은 execution_authorized=false다. 실제 예측 점수·큰 batch 비용·학습 수렴·높은 novelty는 미검증이다. 기존 수치 감사2/2STOP, 실제 학습3batch 미사용을 유지한다.

**다음 한 작업:** [NEXT_RUNNER_GATE.md](../exact_count_learning_v1/NEXT_RUNNER_GATE.md)의 별도 exact_count_runner_v1 실행기·누수 기능검사·전체계약 검토, 통과하면 같은 단위에서 source-only 허용 입력/소스/환경/계약 동결까지 완료한다. 그 전 실제 자료를 열거나 fitting하지 않는다. 사용자 연구 허가는 이미 있으므로 새 사용자 확인 단계가 아니다. 아래 준비 선점은 완료 이력이다.

## 자동화 갱신 확인 — count 수치 감사 후

기존 `ai`를 AI·정보 중심 학습 준비 프롬프트로 갱신했다. ACTIVE·30분 간격·저장 프롬프트 일치를 확인했다. [설정 기록](../count_mask_audit_v1/AUTOMATION_STATUS.json). 추가 standalone 수치 감사는 금지하고 다음 원자료 비열람 학습 준비 단위를 따르며, 실제 fitting에는 별도 계약 동결이 필요하다.

## 2026-11-02 AI 방법 후보 구현·반박 시험: 현재 beam 기각

사용자가 AI·정보를 주 연구로 확인했다. [AI 연구 위치](../path_mask_feasibility_v1/AI_RESEARCH_POSITION.md), [학습 방법 후보](../path_mask_feasibility_v1/LEARNING_METHOD.md), [직접 선행11개](../path_mask_feasibility_v1/PRIOR_REVIEW.md)를 정리했다. Prefix encoder가 미래 mask/regime/path 법칙을 예측하고 count decoder가 비선형 미래 통계를 주변화하는 후보이며, 아직 학습된 새 AI 방법이나 높은 novelty가 확립된 것은 아니다.

별도 `count_mask_audit_v1`을8합성시험·소스검토·새동결 후1회(0.469초) 실행했다.9/10기준은 통과했지만 concentrated64beam residual.007239가 고정.002를 초과해 `FAILED_PREDEFINED_GATE_STOP`이다. Dense13.945ms/beam14.748ms로 가속 주장도 기각했다. 정확 count/CDF·local mask 미분·고정법칙 coarsening은 일치했지만 조건부iid Gaussian 모형의 계산 근거이며 실제 미래 예측 성능이 아니다. [결과](../count_mask_audit_v1/RESULTS.md). 별도 저장근거 검토 material 불일치0건, FD원query 미저장 한계도 보존했다.

현재 practical beam 제안은 닫고 재실행·재동결·기준완화를 하지 않는다. 공유 수치 감사2/2소진, 새 학습 배치3개는 미사용이다. **다음 한 작업**은 [NEXT_LEARNING_GATE.md](../count_mask_audit_v1/NEXT_LEARNING_GATE.md)의 별도 `exact_count_learning_v1` 준비 검토다. 원자료/NPZ/행별 정답 접근·fitting 없이 학습 명세, batch코드, 필요한 작은 기능시험, 공정한 directjoint/MDN 동일감독 대조와 별도검토를 준비한다. 추가 standalone 수치 감사로 예산을 우회하지 않는다. 실제 fitting은 별도 계약 동결 뒤에만 허용하며 olddev/test 정답hash 금지를 유지한다. 아래 서면/수치 준비 선점은 완료 이력이다.

## 2026-11-02 source 미래 경로 준비 완료: 현재 유한 mask 사전은 관문 실패

별도 `path_source_gate_v1` 준비를 1회(9.155초) 완료했다. 합성 11시험·별도 코드 검토·동결 후 원 train34명1442개의 미래46frame을 준비했고 기존 연속 목표와 차이0·guard 완전 일치, 실제 미래 변조8건/역방향 시간 지지8건을 통과했다. 유효1427개·무효15개를 모두 보존했다. 입력과 미래 정답은 별도 private 파일이다. olddev/test 정답 열람/hash·새 학습·다운로드0건. [결과](../path_source_gate_v1/RESULTS.md).

마스크 사전은 각 inner train에서만 만들었다. 내부 고유 패턴67/58/35종, held-valid 미지 비율1.77%/3.68%/7.22%로 사전 고정64종·5% 기준에 두 분할이 실패했다. `PREPARATION_PASS_CATALOGUE_SCREEN_FAIL`을 보존한다. 전체78패턴을 출력 사전으로 쓰거나 기준을 완화하거나 invalid/미지 행을 삭제하지 않는다. 이는 운영 기준 실패이며 모든 마스크 모형의 불가능성이 아니다. 별도 집계 검토는 불일치0건이지만 행별 원자료 재계산은 아니다. [검토](../path_source_gate_v1/FINAL_REVIEW.json).

기존 자동화 `ai`도 이번 준비 완료와 사전 관문 실패를 반영한 [새 프롬프트](../path_source_gate_v1/LOOP_PROMPT.md)로 갱신했다. 앱 응답과 로컬 설정에서 ACTIVE·30분 간격·프롬프트 일치를 확인했다. [설정 확인](../path_source_gate_v1/AUTOMATION_STATUS.json). Mac과 앱이 동작해야 후속 실행이 가능하다.

**다음 한 작업:** [NEXT_METHOD_REVIEW.md](../path_source_gate_v1/NEXT_METHOD_REVIEW.md)의 별도 `path_mask_feasibility_v1` 서면 검토. 기존 집계·코드·직접 선행만으로 미지 마스크에 지원을 주는 조건부 모형, 표본분산 공동확률의 미분·계산비, 같은 추가 감독을 받는 기준선을 검토한다. 이 다음 단위에서는 새 경로 탐색·수치 실행·학습·다운로드를 하지 않는다. 이전 준비와 math 감사는 완료·STOP이다. 아래 자료 준비 대기/선점은 완료 이력이다.

## 2026-11-02 수치 감사 후 자동화 갱신

수치 감사와 별도 검토를 커밋 `3fb679b`로 연구 브랜치에 공유했다. 기존 자동화 `ai`를 완료된 수치 감사를 반복하지 않고 다음 `path_source_gate_v1` 자료 준비 감사를 수행하도록 갱신했다. 앱 응답과 로컬 설정에서 ACTIVE·30분 간격·저장 프롬프트 일치를 확인했다. [설정 확인](../discovery_v3_math/AUTOMATION_STATUS.json). 실제 후속 실행은 Mac과 Codex 앱이 동작해야 한다. 새 자료 준비·학습은 아직 수행하지 않았다. 아래 자동화 기록은 이전 단계의 이력이다.

## 2026-11-02 경로→공동변동량 수치 프로토타입 통과

`discovery_v3_math`의label-free합성감사를단1회완료했다. [결과](../discovery_v3_math/RESULTS.md), [가정/수식](../discovery_v3_math/METHOD.md), [봉인](../discovery_v3_math/RESULTS_SEAL.json).8단위시험·별도검토·5소스/환경/문헌/계약동결뒤10개수치기준전부통과. 알려진분포오차5.55e-16,GLorder차7.52e-8,QMC16개대조최대확률차.001983,64CDF/9pathparametergradient검사통과. 상관을무시한주변곱은.2104,미래누락을무시한fullmask모형은.2648의합성절대확률오차를보여joint/mask가필요함을확인했다. 모델성능21%/26%개선이아니다.

실제자료/이전예측·가중치/정답접근0·새학습0,감사1.325초. Ledger수치감사1/2소진,추가학습3batch미사용. CPUfixtureforward/backward중앙값4.146ms는5개원timing미저장요약이며큰모델속도주장이아니다. 별도검토는8기준을저장수치로재계산했고불변성/guard2기준은소스·요약검토만가능했으며불일치0건. 이증거한계를보존하고감사를재실행하지않았다.

수치PASS는1sharedfactor·조건부iid등분산Gaussian혁신·finite-mask모형안의구현가능성이다. MNAR관측·일반시간공분산·실제바이올린성능·novelty는미해결이다. 다음은 [NEXT_DATA_GATE.md](../discovery_v3_math/NEXT_DATA_GATE.md)의별도source-only미래경로자료준비·mask지원범위감사이며아직구현/실행하지않았다. 준비/누수검사/검토/동결후진행하고이후학습은별도계약이필요하다. 기존v1/v2및이번수치감사는완료상태를유지한다.

## 자동 루프 재개 확인 — 2026-11-02

기존heartbeat자동화 `ai`를새탐색프롬프트로갱신하고앱도구의ACTIVE응답및로컬설정의prompt일치·30분간격·ACTIVE를확인했다. [설정확인](../discovery_v2/AUTOMATION_STATUS.json), [저장프롬프트](../discovery_v2/LOOP_PROMPT.md). 다음작업은label-free `discovery_v3_math`이며아직실행하지않았다. 기존batch재실행과olddev/test정답열람을금지하고추가예산ledger·11월6일종료를유지한다. 과거PAUSED문구는이전단계이력이다. 실제후속실행은Mac과앱이동작해야한다.

## 2026-11-02 사용자 요청으로 넓은 탐색: 새24학습 완료

사용자가비정상적으로보이는조합도폭넓게시도하도록요청했다. 이전보정인증분기는STOP으로보존하고새source-only `discovery_v1/v2`를만들었다. 기존raw_v1학습결과를재실행한것이아니다. Source34명1442녹음만3사람fold로사용했고olddev/test정답은이번에hash도열지않았다. 모든fold문턱/정규화/prior는innertrain만쓴다. source자체도역사적으로노출돼있어독립확증은아니다.

v1: prefix96×4시계열추출,8실제미래변조검사·8합성시험·별도검토후12학습55.141초. Markov.542081/HGB.581722/orderedTCN.561511/shuffledTCN.560271/MDN.559653. v2: 같은정확prefix상태와unclipped변동량을모든모델에주고4합성시험·별도검토후12학습11.583초. MLPjoint.554539/MLPdelta.563987/attentionjoint.568173/attentiondelta.558815. 두batch모두HGB보다나은신경망이있지만Markov를못넘어고정screen통과후보는없다. 24학습총66.724초(준비·검토·시작시간별도)이며모든실패/점수를보존했다.

v2의delta상호작용+.018806은세fold에서양수지만delta가attention에항상도움된것은아니고MLP에서는세fold모두악화였다. bundledhead/anchoring/auxloss와context/모수수변경이므로기전/novelty를선언하지않는다. trainBCE가60까지감소해수렴미확인이고held-train격차도모두양수다. v2는가중치·train곡선을private로보존했다. 별도결과검토가두batch의hash/집계/score조건을재현했고불일치0건이었다.

다음은 `discovery_v2/NEXT_EXPLORATION.md`의별도label-free수치프로토타입이다. Gaussian경로의표본분산pushforward는직접MDN과다른실질표현후보이며선행6개검토상선형집계예측과구별된다. 그러나고전이차형식CDF·gradient/lowrankpath자체는선행이고joint/mask가미해결이다. 아직새방법이나개선이확보된것은아니다. 넓은가설은행은 `discovery_v1/IDEAS.md`,추가루프예산은 `discovery_v2/LOOP_BUDGET.json`에있다. 완료된batch만STOP이며새요청의탐색전체를닫지않는다.

## 2026-11-02 최종 가능성 감사: 조건부 보정·인증 방향 STOP

사용자가요청한“현재자료규모에서성립하는지,단순혼합을넘는novelty가있는지”검증을완료했다. 현재후속은**STOP_CORRECTION_CERTIFICATION_BRANCH**다. [최종판정](../correction_feasibility_v1/DECISION.md), [집계](../correction_feasibility_v1/RESULTS.json), [추가선행·수학](../correction_feasibility_v1/PRIOR_AND_MATH.md)을읽는다. 이전문서의가능조건미완료는이번감사로대체됐다.

기존offset3seed를공통계수로보정했다. 전역/상태별혼합의devoracle과한dev참여자제외보정,고정span cap.025/.1만검사했다.7합성시험·별도AI소스검토→코드6/입력19동결→감사1회(108scalar적합)→별도결과재계산검토완료. 새base학습0·test정답열람/재평가0,기존결과와17개데이터소스도변경없음. 실제보정계수는적합했으므로새fitting이전혀없었다고쓰지않는다.

Markov dev손실.536024,원offset.710557. 한사람제외보정에서전역혼합.521987(기준선대비2.6186%감소),상태별.527730(전역보다1.10024%악화/4of8만개선),고정cap이득.147%/.567%. 같은자료oracle에서는상태별.507229가전역.517824보다좋았지만사람을제외하면역전됐다. 모든숫자는사후개발진단이며독립성능개선이아니다. 두혼합의최대사람손실도원Markov보다높아안전성을주장할수없다.

HB/betting은작은n에서도비자명할수있음을양성대조로확인해기존Hoeffding522명예를불가능성하한으로오해하지않도록했다. 그러나재사용·겹치는보정자료의독립성/조건부평균근거는없고더강한검정도이간극을자동해결하지않는다. 기존온라인전문가혼합은다른목표인순차누적regret를보장할수있지만직접선행이며offline새사람위험의인증이아니다. 신규성·인증·독립확증·추가성능관문모두미통과로종료했다. 자동화PAUSED,추가실행없음.

## 2026-11-02 신경망 실패 진단·직접 선행 검토 완료

사용자요청에따라기존실험을보존하고별도 `violin_diagnosis_v1/`에서사후진단했다. 새학습0·test정답재열람0,저장신경망9개와Markov의dev예측복원오차0. 초기진단의clip경계미분보완은실제경계값0건으로기존수치변화0이었다. 초기snapshot/hash를보존했다.9합성시험통과,별도AI최종소스/수학검토approve이며독립인간심사나새경험적확증은아니다.

offset은train0.422824로Markov0.533097보다좋았지만dev0.710557로Markov0.536024보다나빴다. dev확신오류4.083%가seed평균손실의40.956%를차지했다. 가장영향큰한사람이net초과손실64.62%를차지했지만제외하지않았다. centeredresidualnorm은train4.656/dev8.299,dev분포는더확신적이었다. 평균적인prior전체상쇄설명은지지되지않는다. 학습곡선·개입시험은없어최적epoch나정확한인과원인은모른다.

원음gain0.5/2에서dev337개모두진단변환의조건을만족했고,각8개총16개원음대조에서준비자료일치·9규칙보존,특징작용오차4.27e-13미만을확인했다. 그러나민감도와실패의상관은방향이일치하지않아음량gate를다음방법으로채택할근거가부족하다. 미래창을본admission은진단검증에만썼으며배포gate/입력으로사용할수없다.

후속후보는**참여자불확실성에따른조건부보정예산**이다. 혼합·logitprior·DRO·LTT·HCP/GHCP에직접선행이있으므로아직새방법이아니다. 보정의비영허용조건·의존처리·비자명한통계경계가실제기술과제다. 정의에서따르는무악화no-go와logitspan손실상한을32개수학fixture로검산했고,이는학습실험이아니다. 가상의독립34명/c=.1의단순Hoeffding폭0.04198은2%이득예0.01072보다커서,느슨한인증구상만으로바로새학습을시작할근거는없다. 이는최적표본하한이나모든방법의불가능성주장이아니다.

[원인분석](../violin_diagnosis_v1/CAUSE_ANALYSIS.md), [후속명세·기각조건](../violin_diagnosis_v1/NEXT_METHOD.md), [직접선행](../violin_diagnosis_v1/LITERATURE_RESIDUAL_REVIEW.md), [집계·감사](../violin_diagnosis_v1/FINAL_AUDIT.json). 기존캠페인STOP·자동화PAUSED. 다음은수학적가능조건검토이며추가모델/seed탐색이아니다.17개동결데이터소스와기존실행/예측/결과봉인도변경없음을확인했다.

## 2026-11-02 고정12개 학습·시험 평가 완료: 개선 미입증

유한 비교를 마쳤으므로 앱 자동화 `ai`를 **PAUSED**로 전환했다. 실패를 만회하기 위한 추가 학습은 진행하지 않는다. 새로운 연구 방향은 이 종료 결과와 이미 노출된 시험 자료를 전제로 별도 검토해야 한다.

사용자가요청한실행기검토와고정12개실험까지완료했다.별도최종소스검토approve,합성시험57개후실행소스11개·입력4개·환경을동결했다. 실행manifest SHA `e23ebe938134885ae132ec57913f02ea9ef7cc0b20091154339c5aeda8c98cec`. 실제12시도모두성공,worker16.968829초. 모든예측·영수증봉인뒤개발자료에서Markov를비교군으로고정하고시험정답을한번열었다. 상태는EVALUATED,stopped=true다.

**주결과:** test 참여자macro log loss는Markov0.552413,HGB0.556620,Frequency0.586117,독립MLP0.663424,공동MLP0.666765,offset0.664192다. 신경망은모두3seed손실평균. offset은Markov보다20.2346%나쁘고,손실차이+0.111778의탐색적95%pairedparticipant구간은[+0.028140,+0.193551]다. plainjoint대비0.386%개선과독립대비0.116%악화는모두구간이0을포함한다. 사전2%screen실패. 일관성100%를성능개선으로바꾸어설명하지않는다.새방법론의높은novelty도확보되지않았다.

시험389개중388개유효/1개무효를모두보존했고,유효부분에서도같은실패방향이다.HGB의Brier가Markov보다낮은보조결과도함께공개했다.9명·한분할·고정설정·파일시작기준미래지표의범위이며교육/음질/물리법칙의검증으로확대하지않는다.

별도검토자가실행소스11개·데이터소스17개·준비파일4개·예측/영수증24개와결과/선택/test-marker봉인을확인했고불일치0건이었다. JSON참여자/seed손실로평균·상대변화·세bootstrap구간을1e-12이내로재현했다. 시험정답파일은재열람하지않았다. 이는독립적인행별재평가나외부입회가아니다.

[결과보고서](../violin_raw_v1/RESULTS.md), [점수와봉인](../violin_raw_v1/RESULTS.json), [최종감사](../violin_raw_v1/FINAL_AUDIT.json)에근거를보존한다. 원자료·준비특징/정답·행별예측·가중치·원시CLI로그는공개하지않는다. 이캠페인에서모델/seed/문턱추가탐색은하지않고종료한다. 아래의미완료실행기계획은모두이전이력이다.

## 2026-11-02 명시적 재개

**이후 완료:** 도구를 끈 별도 Claude Opus4.7 호출 두 건으로 데이터 준비 핵심과 WAV/아카이브 의존 코드를 검토했고, 모두 차단 사항 없이 data-preparation scope 승인을 받았다. 모든 관련 합성 시험 **42 passed in2.69s**. 데이터 코드·규약·검토를 동결한 manifest SHA는 `383c081891fd67fee670809c969bb06aa224f4f5375848f0715eae8d725f172b`다. 동결된 데이터 코드와 규약을 수정하거나 준비를 덮어쓰지 않는다.

실제 **2,168녹음×60개 특징**을 생성했다. 참여자34/8/9명, 녹음1442/337/389개를 유지했고, 해시로 고른 실제8녹음에서 미래를NaN으로 바꿔도 X가 bitwise 동일했다. 준비 파일4개 hash, 입력 파일의 허용 key, 참여자 분할 비중복을 확인했다. `features.npz`와 train/dev/test 정답 파일을 분리해 ignored `violin_raw_v1/local/prepared/`에 보존했다. 시험 정답은 생성됐으나 모델 학습·선택·점수 계산에 사용하지 않았다. 공개하는 것은 집계와 hash인 [PREPARED_SUMMARY.json](../violin_raw_v1/PREPARED_SUMMARY.json)뿐이다.

현재 프로토콜은 세 seed 손실 평균을 주 비교로, 예측 ensemble은 별도 보조로 명시한다. offset/CDF/일관성 자체가 신규 방법이 아니라는 [직접 선행 범위](../violin_raw_v1/LITERATURE_SCOPE_20261102.md)를 재확인했다. 원논문에서도 upstream clipping/onset 인과성은 확인되지 않아 recording-relative forecast로 제한한다. 별도AI 코드 검토는 인간 심사·독립 경험적 확증이 아니다.

**다음 미완료:** 학습/예측 봉인/단일 test 평가를 담당하는 실행기가 아직 없다. `metrics.py`의 macro 점수·paired bootstrap 합성 시험만 추가됐다. 다음 작업은 이 실행기 구현·합성 검증·별도 검토 → 준비 입력/모델/평가 코드 추가 동결 →12개 고정CPU 비교다. 모델 학습 결과는 여전히 없고 데이터 준비 승인은 학습/평가 실행의 승인으로 확대하지 않는다.

사용자의 “루프 재개” 요청으로 자동화 `ai`를ACTIVE로 변경했고 해당 연구의 중복 실행 프로세스가 없음을 확인했다.10월31일 일시중지는 해제됐다. 아래는 작업 이력이다.

## 2026-10-31 사용자 요청으로 일시중지

사용자가 “중단했다가 이따 이어하자”라고 요청했다. 자동화 `ai`를 **PAUSED**로 변경했고, 실행 중인 해당 연구 Python 작업은 없었다. 명시적인 재개 요청 전에는 새 다운로드·검증·학습을 시작하지 않는다. 아래 ACTIVE 표시는 이전 시점의 이력이다.

중단 직전 완료: 원음2,203,971,621bytes 다운로드와 MD5/SHA256 검증. 실제 헤더/PCM byte 감사로 CG18·FG18·EG15, 총51명과 바이올린2,174녹음을 확인했고 고정 미래창 길이를 만족하는2,168개를 확보했다.44.1kHz/16-bit/mono이며 전체/입력1초 파형의 참여자 간 exact duplicate는0건이다. 원음과 개별 metadata는 저장소 밖 `~/violin-rne/data/external/blanco2021_raw/`에 보존했다.

최초 감사의 미분류431개는 모두AG2 폴더였다. 사전에 정한 CG/FG/EG 허용 목록 밖의 그룹으로 명시하여 재분류했고 **선택된 녹음은 바꾸지 않았다**. 실패한 원래 `inventory_summary.json`을 보존했으며 해소 근거는 `data_partition_receipt.json`이다. AG2 의미나 MAT의72명과 대응 관계를 추정하여 합치지 않는다.

미래 음향 목표/라벨 비열람으로 참여자 분할을 고정했다: train34명/1,442녹음, dev8명/337녹음, test9명/389녹음. Manifest hash `5a5940d93092e12d7cfcb255964a640584258f3575dc2c03c214c72e1e9d43e9`. 로컬 원본은 `participant_split_v1.json`. 특징·데이터·아카이브·분할 시험 **36 passed in2.66s**. 실제 미래 목표 생성과 바이올린 학습/성능 평가는 아직 하지 않았다.

재개 후 다음 작업: 새 데이터 계약/목표 생성기와 평가 실행기의 독립 검토·녹음 시작점 provenance 확인 → 규약 동결 → 실제 준비 자료 생성 및 최대12개 고정CPU 비교. test9명의 불확실성을 숨기지 않는다. 이번 추가 코드·영수증·이 상태 기록은 아직 미커밋이며 직전 공개 커밋은 `e7514da`다. pause 요청 이후 커밋/푸시·추가 실험은 하지 않았다.

## 2026-10-31 루프 재개: 합성 미래 누수 검사 통과

사용자의 `루프 ㄱ` 요청으로 자동화 ID `ai`의30분 루프를 최신 상태로 갱신해ACTIVE를 확인했다. v3/projection 분기 종료를 명시하고 원음 바이올린 자료 감사→누수 검증→독립 검토/동결→고정 비교 순서로 진행한다.

원음 다운로드는 보존된42개 완료 조각(352,321,536bytes)에서 재개했다. 미완성 tmp와 옛 prefix는 중복일 수 있어 진척량에 더하지 않는다. 다운로드는 아직 완료/체크섬 통과로 보고하지 않는다. 새 `features.py`는 입력 `[0,1)`초와 목표 `[1.25,1.75)`초를 먼저 자른 뒤 구간 안에서만 음향 특징을 계산한다. 미래 신호를 큰 잡음·NaN·Inf로 바꾸어도 X가 bitwise 동일한 시험,5개 sample rate,PCM/stereo,무음/잡음,알려진 pitch/RMS와 기존 모델의 전처리 시험까지 **19개 통과(2.53초)**했다. 최초 import 수집 오류를 고친 뒤 통과했으며 실제 데이터의 학습/성능 시험은 아니다. [구현과 한계](../violin_raw_v1/RAW_FEATURE_AUDIT.md).

마지막 보정 감사의 동결 소스/입력/결과15개 해시도 읽기 전용으로 확인했고 불일치0건이다. solver를 다시 실행하거나 라벨 점수를 보지 않았다. 다음 작업은 다운로드 완료 후 raw의 실제 참여자·중복·길이·WAV 형식 감사와 participant split/훈련 문턱 생성기다. 독립 검토·실험 동결 전에는 모델을 학습하지 않는다.

## 최신 추가: Claude 인계 시점

최신 다음 작업은 [CLAUDE_NEXT.md](../CLAUDE_NEXT.md)를 따른다. 아래 v3 수치는 그대로 보존한다. 마지막 `projection_audit_v2`는 256건 중 gap 통과239건, clipping 밖 identity11건, 실제 실패6건으로 완료했고 보정 solver 탐색을 종료했다. [완료 결과](../projection_audit_v2/RESULTS.md).

저장 MAT의 YIN 특징에서 양방향 참조가 확인되어 그 데이터의 실제 예측 실험은 보류했다. 원음에서 구간을 먼저 자르는 `violin_raw_v1`을 준비한다. 인계 확인 시 다운로드/학습 프로세스는 없었고, 다운로드 조각 약390MiB와 이전 prefix 약130MiB를 보존했다(중복·미완성 포함 가능, 완성 아카이브 아님). 다운로드·모델 코드와 합성 모델 시험3개 통과까지이며 특징 추출기/실제 데이터 생성기/평가 실행기는 미완성이다. 아직 바이올린 성능 결과는 없다. 아래의 MAT 자료 가용성 통과는 이 새 시간 의존성 감사로 제한되며 엄격한 causal 예측 관문 통과를 뜻하지 않는다.

`projection_audit_v2/`, `violin_forecast_v1/`, `violin_raw_v1/`는 이 확인 시 로컬 미커밋 작업이다. 새 컴퓨터의 GitHub clone만으로 이어받지 말고 이 Mac의 최신 작업물을 사용한다. 인계 파일을 만들었으며 Claude 작업의 실제 시작/완료는 아직 확인하지 않았다.

**미래 예측 AI 실험 12개를 완료했다. 공동 패턴 신경망은 보정된 HGB보다 손실이 3.335% 낮았으나, 독립 신경망보다 0.185% 높았다.** 미래 판정을 예측하는 성능 개선은 관찰됐지만, 공동 구조 자체의 추가 이득과 높은 방법론 신규성은 입증하지 못했다.

공유 대상은 `NGC1788/violin-bowing-boundaries`, 브랜치 `research/agentic-forecast-20261030`다. [실행 안내](../START_HERE.md), [다른 AI에게 줄 인계문서](../HANDOFF_ANY_AI.md), [전체 결과](../published_results/forecast_20261030_v3/RESULTS.md), [독립 재계산 검토](../published_results/forecast_20261030_v3/FORECAST_REVIEW.md)를 사용한다. 업로드 완료 여부와 최종 커밋은 GitHub 및 이 대화의 완료 메시지로 확인한다.

## 고정 실험 결과

학습 54,000개(A/B), 개발 24,000개(C/D), 입력 60차원, 입력 종료 0.20초, 미래 창 0.25–0.45초다. 과거 판정 78,000개를 재구성했으며 불일치는 0건이었다. 모든 현재 자료는 과거에 노출됐고, plateau 기준점은 전체 신호에 의존한다. 따라서 실시간 예측이나 독립 확증으로 부르지 않는다.

| 방법 | 평균 log loss ↓ | 설명 |
|---|---:|---|
| Source Markov | 0.217187 | 훈련 자료에서 과거→미래 패턴 전이 학습 |
| HGB | 0.115583 | 표형 자료 기준선, 세 seed 결과 동일 |
| HGB + 기존 KL 기반 보정 | 0.115223 | 수치 검사 통과 보정만 채택, 나머지는 원예측 유지 |
| 독립 9-head 신경망 | 0.111175 | 세 seed 평균 |
| 독립 신경망 + 기존 보정 | 0.111115 | 가장 낮은 평균 손실 |
| 공동 10-pattern 신경망 | 0.111380 | 세 seed 평균, 출력 논리 일관성 100% |

사전 선별 기준은 강한 단순 기준선보다 2% 이상 개선하고 각 세션의 악화를 0.02 nats 이내로 제한하는 것이었다. 공동 head는 이를 통과했다. 하지만 독립 head와의 직접 비교에서는 개선되지 않았다. 기준선 대비 순개선의 약 66%는 D1에서 나왔다. 네 세션과 세 seed를 독립적인 모집단 표본처럼 취급해 작은 p값을 만들지 않았다.

일관성 비율은 허용 패턴의 확률분포로 표현 가능한지를 허용오차 1e-6에서 검사한 값이다. 작은 위반도 집계되며 분류 오류율이나 음질 불량률이 아니다. 고전적인 정확 KL projection과 현재 64회 반복·fallback 근사 보정도 구분한다.

## 자동 실행과 검증

- `forecast_20261030_v3`: 12 completed, 검증 오류 0, STOP. Worker 누적 77.96초이며 LLM 대기는 제외한다.
- 80개 테스트 통과 후 동결했다. 입력 NPZ·JSON 영수증·규약·후보·소스를 해시로 결합했다. Manifest: `f530cb6aeada65f58e3065e48f3ca5212b355501b40962f3f6226f16c5e1ec11`.
- 독립 검토자가 모든 원예측의 주 점수를 재계산했고 차이는 0.0이었다. 동결 소스 32개, 결과 봉인 12개, 사후 산출물 체크섬 104개도 일치했다. 사후 체크섬은 실행 당시의 봉인과 구분한다.
- Claude 제안자 + Codex 검토자로 두 실험을 완료했다. 이후 Claude HTTP429 실패 라운드를 보존하고 Codex/Codex로 전환해 나머지 고정 후보를 완료했다. 총 제안 라운드 10회 중 실패 1회다.
- 사용자 요청 후 Claude 재확인도 HTTP429였다. CLI는 03:10 Asia/Seoul 초기화를 안내했다. 추가 결제는 하지 않았다. 다음 사용 시 필요하면 짧게 재확인한다.
- `codex`, `claude`, `manual` JSON 모드를 구현했다. 수동 응답은 request_sha·후보 해시에 묶이며 같은 라운드를 재개한다. 다른 AI 제품의 별도 호출이 통계적으로 독립된 검토를 보장하지는 않는다.
- 앱 자동화 **활현 AI 연구 루프**(ID `ai`)는 30분 간격 ACTIVE다. 완료된 v3를 재개하지 않고 작업 큐의 다음 단계를 수행한다. 현재 무거운 학습은 실행 중이지 않다. GitHub 복제만으로 이 앱의 자동화까지 복제되지는 않는다.
- 새 Git 파일 복사본에서 80개 테스트와 준비 자료 로딩을 확인했다. 기존 Python 환경을 사용한 검사이며 빈 환경에 새로 설치한 검사와는 다르다. [이식성 감사](PORTABILITY_AUDIT.md).

## 바이올린 공개자료의 실제 확인

Blanco 2021의 CC-BY-4.0 자료 약 469 MB를 받아 MD5/SHA256을 검증하고 MAT 구조를 해석했다. 72개 참여자 기록과 2,117개 고유 음향 시퀀스를 확인했다. 중복 cell 1,044개, F0 단위 설명 불일치, timestamp 부재, 동작 신호의 반복·이상값이 있다.

**음향 지표만 사용하는 frame 단위 미래 예측의 자료 가용성은 통과했다.** 초 단위 실시간성과 다중모달 동기화는 미확인이다. 바이올린 모델은 아직 훈련·평가하지 않았다. [자료 감사·재현 코드·다음 프로토콜](../forecast_v3/violin_data_audit/README.md).

원자료는 Git에 올리지 않는다. 공개하는 기존 첼로 준비 자료 약 9 MB는 다섯 Zenodo 기록의 현재 CC-BY-4.0 라이선스를 재확인하고 출처를 표시했다.

## 후속 수치 감사 1회 완료

[확률 보정 수치 감사](../projection_audit_v1/RESULTS.md)를 정답 비열람 방식으로 수행했다. 해시로 선택한 동일 ID 64개 × 네 모델 = 256개 예측에서 참조 KL solver를 검사했다. 합성 테스트 17개 통과, 코드·규약·입력 해시 동결 후 한 번 실행했으며 기존 v3는 수정하지 않았다.

최적화 라이브러리는 모든 행에서 적어도 한 시작점을 성공으로 표시했지만, 사전 FW gap 조건(≤ 1e-8)은 50/256개만 통과했다. 따라서 전부 충분히 수렴한 참조 기준선으로 채택할 수 없다. 가능한 모든 출력의 clipped loss 검사까지 통과해 보정을 채택한 것은 28개였고, 나머지는 원예측으로 돌아가 최종 점수 증가 상한 검사는 0.0이었다. 이 감사는 모델 성능을 다시 평가하지 않았으므로 기존 순위나 신규성 결론을 바꾸지 않는다.

한 번의 별도 수치 보완 검증만 허용한다. 목적함수 변화보다 KKT/FW 잔차를 직접 확인하는 방법을 합성 자료에서 먼저 검증하고, 같은 수치 기준을 낮추지 않는다. 실패해도 반복 확장하지 않고 참조 solver의 한계를 남긴다.

## 다음 작업

[작업 큐](WORK_QUEUE.md)의 순서로 정확 보정 기준선의 수렴, 다음 방법의 직접 선행과 필요성, 바이올린 participant 분할·입력 누수·훈련 자료만의 문턱 결정을 검증한다. 정당화된 경우에만 별도 규약을 동결하고 제한된 CPU 비교를 수행한다. 이미 확인한 v3 점수에 맞춰 라벨·후보·좋은 seed만 선택하지 않는다.

## 보존한 이전 결과

v1 파일럿 6회와 v2 전체 480열 기준선 5회도 보존했다. 정확 질의 목표의 가능 개선 상한은 각각 7.77%, 5.83%로, 원래 10% 조건이 불가능해 종료했다. [v2 결과](FULL480_RESULTS.md). 기존 물리 연구와 10월 24일 보고서를 새 AI 성과로 덮어쓰지 않았다.
