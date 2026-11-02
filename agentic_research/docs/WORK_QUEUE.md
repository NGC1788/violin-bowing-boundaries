# 유한 연구 작업 큐 — 2026-10-30

## 2026-11-03 전체 실행 계약 동결 완료: 다음은 실제 고정12개 학습

`exact_count_runner_v1`의 실행기·source-only loader·예산/종료 controller·결과 봉인을18합성 기능시험과 별도 검토로 완료했다. 초기 검토에서 parent출력봉인·중복Markovfit·watchdog소유/정리·checkpoint임시파일 문제를 찾아 실제입력열람 전에 모두 수정했다. [검토](../exact_count_runner_v1/REVIEW.json).

코드 승인 뒤에만6개 source 입력(학습prefix/meta/future와 기존HGB예측3개)을 열고 hash·행정렬·34명1442개·invalid15개·연속목표/guard/시간지지를 확인했다. olddev/test 정답은hash도 열지 않았고 raw음원·실제fitting·추가수치감사0건이다. [전체동결](../exact_count_runner_v1/FREEZE.json)의 SHA는 `aa7f4dc023a3885db4aab19c52e113ed9af765850621e8bb661481c3d45abfa3`, 상태 FROZEN_READY_FOR_BOUNDED_TRAINING이다. [준비결과](../exact_count_runner_v1/PREPARATION_RESULT.json).

**다음 한 작업은 준비 반복이 아니라 실제 실행:** `.venv-mac/bin/python -m agentic_research.exact_count_runner_v1.runner check` 후 다른worker/state가 없으면 `...runner run`을 한 번 실행한다. 네 head×3fold12neural jobs·40epochs/batch64/seed0/180초job/1800초worker 고정. HGB새fit0·sealed예측재사용, analyticMarkov3회는worker시간에 포함한다. 모든12예측을parent가고정한 OUTPUT_SEAL 뒤에만held-event score를계산한다. 첫실패는batchSTOP, 재실행/구조/불완전승격을하지않는다. 다음실행승인은이미사용자연구범위이며새확인을묻지않는다. 아직학습성능/novelty는미입증이다. 아래준비선점은완료이력이다.

## 2026-11-03 현재 선점: exact_count_runner_v1 실행기·전체 동결

`/root`가 별도 실행기·누수 기능시험·계약을 원자료 비열람 상태에서 작성한다. 실제 planning.lock을 보유한다. 코드 검토 통과 뒤에만 같은 단위에서 source-only 입력 동일성/행 정렬을 확인하고 전체 실행 동결을 완료한다. 이 단위에서는 실제 fitting하지 않는다. 다른 작업자는 같은 실행기나 학습을 중복 시작하지 않는다.

## 2026-11-03 AI 학습 코드 준비 완료: 실제 학습은 아직0건

`exact_count_learning_v1`의 batch countDP·Gaussian contrast/scale 역전파·prefix-only 입력·동일 보조감독 네 모델을 구현했다. 직접 joint/MDN은4414개, exact/moment count는4084개 파라미터이며 추가330개 free-head를 공개했다. [준비 결과](../exact_count_learning_v1/PREPARATION_RESULT.json), [코드/명세](../exact_count_learning_v1/README.md).

실제 자료 없이 학습 목표의 nuisance를 수정했다. 원시 미래 레벨 회귀 대신 정규화된 Gaussian contrast 밀도와 pooled within-record scale을 사용하며 자유도n-1을 맞춘다. 미래 mask/center는 학습 label 변환에만 있고 추론 입력이 아니다. 모든 방법에 같은 aux를 준다. 이는 고전 error-contrast 원리의 적용이며 그 자체가 novelty가 아니다. [목표 명세](../exact_count_learning_v1/INVARIANT_OBJECTIVE.md).

합성 기능시험15개가2.01초에 통과했고 별도 코드/수학검토는 APPROVE_PREPARATION_ONLY다. 실제 원음/NPZ/행별 정답/가중치 열람·hash0, optimizer step·실제 fitting·추가 standalone 수치감사0. [코드 준비 봉인](../exact_count_learning_v1/READINESS_FREEZE.json)은 execution_authorized=false다. 실제 예측 점수·큰 batch 비용·학습 수렴·높은 novelty는 미검증이다. 기존 수치 감사2/2STOP, 실제 학습3batch 미사용을 유지한다.

**다음 한 작업:** [NEXT_RUNNER_GATE.md](../exact_count_learning_v1/NEXT_RUNNER_GATE.md)의 별도 exact_count_runner_v1 실행기·누수 기능검사·전체계약 검토, 통과하면 같은 단위에서 source-only 허용 입력/소스/환경/계약 동결까지 완료한다. 그 전 실제 자료를 열거나 fitting하지 않는다. 사용자 연구 허가는 이미 있으므로 새 사용자 확인 단계가 아니다. 아래 준비 선점은 완료 이력이다.

## 2026-11-02 현재 선점: exact_count_learning_v1 원자료 비열람 학습 준비

`/root`가 정확 count decoder의 batch·Gaussian mean/scale 미분·공통 보조감독 모델·prefix-only 전처리·합성 기능시험과 별도 검토를 담당한다. 실제 planning.lock을 보유한다. 원음/NPZ/행별 실제 정답을 읽거나 hash하지 않고 실제 fitting·다운로드·추가 standalone 수치 감사는 하지 않는다. 준비가 완료돼도 실행기/입력/학습계약 동결 전에는 학습하지 않는다.

## 2026-11-02 AI 방법 후보 구현·반박 시험: 현재 beam 기각

사용자가 AI·정보를 주 연구로 확인했다. [AI 연구 위치](../path_mask_feasibility_v1/AI_RESEARCH_POSITION.md), [학습 방법 후보](../path_mask_feasibility_v1/LEARNING_METHOD.md), [직접 선행11개](../path_mask_feasibility_v1/PRIOR_REVIEW.md)를 정리했다. Prefix encoder가 미래 mask/regime/path 법칙을 예측하고 count decoder가 비선형 미래 통계를 주변화하는 후보이며, 아직 학습된 새 AI 방법이나 높은 novelty가 확립된 것은 아니다.

별도 `count_mask_audit_v1`을8합성시험·소스검토·새동결 후1회(0.469초) 실행했다.9/10기준은 통과했지만 concentrated64beam residual.007239가 고정.002를 초과해 `FAILED_PREDEFINED_GATE_STOP`이다. Dense13.945ms/beam14.748ms로 가속 주장도 기각했다. 정확 count/CDF·local mask 미분·고정법칙 coarsening은 일치했지만 조건부iid Gaussian 모형의 계산 근거이며 실제 미래 예측 성능이 아니다. [결과](../count_mask_audit_v1/RESULTS.md). 별도 저장근거 검토 material 불일치0건, FD원query 미저장 한계도 보존했다.

현재 practical beam 제안은 닫고 재실행·재동결·기준완화를 하지 않는다. 공유 수치 감사2/2소진, 새 학습 배치3개는 미사용이다. **다음 한 작업**은 [NEXT_LEARNING_GATE.md](../count_mask_audit_v1/NEXT_LEARNING_GATE.md)의 별도 `exact_count_learning_v1` 준비 검토다. 원자료/NPZ/행별 정답 접근·fitting 없이 학습 명세, batch코드, 필요한 작은 기능시험, 공정한 directjoint/MDN 동일감독 대조와 별도검토를 준비한다. 추가 standalone 수치 감사로 예산을 우회하지 않는다. 실제 fitting은 별도 계약 동결 뒤에만 허용하며 olddev/test 정답hash 금지를 유지한다. 아래 서면/수치 준비 선점은 완료 이력이다.

## 2026-11-02 서면 방법 검토 완료·별도 count_mask_audit_v1 준비

`path_mask_feasibility_v1`에서 직접 선행11개와 별도 수학 검토를 완료했다. `JUSTIFIED_NUMERICAL_NEXT_STEP`이며 novelty/성능 승인이 아니다. 다음은 `/root`가 담당하는 별도 합성 연산기 `count_mask_audit_v1`: count DP·정확 guard·surviving beam valid residual·local mask-logit 미분 경계·mean coarsening을 반박 시험한다. 새 데이터 접근과 학습은0이며 남은 수치 감사1회/600초만 사용한다. 코드·합성 시험·별도 검토·새 동결 후 실행한다. 앞선 서면 단위는 완료됐고 여기의 수치 실행은 별도 계약이다.

## 2026-11-02 현재 선점: 미지 mask를 열거하지 않는 새 방법의 서면 검토

사용자의 방법론 발전 요청으로 `/root`가 별도 `path_mask_feasibility_v1`을 담당한다. 실제 planning.lock을 보유하고 직접 선행·수학·복잡도·공정한 대조를 검토한다. 새 자료 탐색·학습·수치 실행은 하지 않는다. 기존 유한 사전 실패와 수치 PASS는 그대로 보존한다. 다른 작업자는 같은 검토를 중복 시작하지 않는다.

## 2026-11-02 source 미래 경로 준비 완료: 현재 유한 mask 사전은 관문 실패

별도 `path_source_gate_v1` 준비를 1회(9.155초) 완료했다. 합성 11시험·별도 코드 검토·동결 후 원 train34명1442개의 미래46frame을 준비했고 기존 연속 목표와 차이0·guard 완전 일치, 실제 미래 변조8건/역방향 시간 지지8건을 통과했다. 유효1427개·무효15개를 모두 보존했다. 입력과 미래 정답은 별도 private 파일이다. olddev/test 정답 열람/hash·새 학습·다운로드0건. [결과](../path_source_gate_v1/RESULTS.md).

마스크 사전은 각 inner train에서만 만들었다. 내부 고유 패턴67/58/35종, held-valid 미지 비율1.77%/3.68%/7.22%로 사전 고정64종·5% 기준에 두 분할이 실패했다. `PREPARATION_PASS_CATALOGUE_SCREEN_FAIL`을 보존한다. 전체78패턴을 출력 사전으로 쓰거나 기준을 완화하거나 invalid/미지 행을 삭제하지 않는다. 이는 운영 기준 실패이며 모든 마스크 모형의 불가능성이 아니다. 별도 집계 검토는 불일치0건이지만 행별 원자료 재계산은 아니다. [검토](../path_source_gate_v1/FINAL_REVIEW.json).

**다음 한 작업:** [NEXT_METHOD_REVIEW.md](../path_source_gate_v1/NEXT_METHOD_REVIEW.md)의 별도 `path_mask_feasibility_v1` 서면 검토. 기존 집계·코드·직접 선행만으로 미지 마스크에 지원을 주는 조건부 모형, 표본분산 공동확률의 미분·계산비, 같은 추가 감독을 받는 기준선을 검토한다. 이 다음 단위에서는 새 경로 탐색·수치 실행·학습·다운로드를 하지 않는다. 이전 준비와 math 감사는 완료·STOP이다. 아래 자료 준비 대기/선점은 완료 이력이다.

## 2026-11-02 현재 선점: path_source_gate_v1 자료 준비 감사

`/root`가 별도 source-only 미래 경로·mask 지원범위 준비 감사를 담당한다. 실제 `path_source_gate_v1/local/planning.lock`을 보유한다. 기존 math 감사는 완료·STOP이다. 새 계획·합성 누수 시험·별도 코드 검토·입력/소스 동결 뒤 원 train34명1442개만 한 번 준비한다. olddev/test 정답은 hash도 열지 않고 새 학습은 하지 않는다. 다른 실행자는 이 준비를 중복 시작하지 않는다.

## 2026-11-02 discovery_v3_math 완료: 제한된 공동 decoder 수치 PASS

고정합성감사1회를완료했다.8단위시험·별도소스/수학검토후동결했고10기준전부통과했다. 이론특수사례오차5.55e-16,GL256/512차7.52e-8,독립경로QMC16대조최대차.001983,64scalar+9pathgradient유한차분통과. 잘못된주변CDF곱/미래fullmask고정의합성절대확률오차는.210398/.264838이다. 실제예측성능개선이아니다. source/raw/dev/test자료접근·새학습0. 별도결과검토불일치0이며8기준수치재구성/2기준요약·소스근거의한계를기록했다.

해당감사는완료·STOP으로보존하고재실행/재동결하지않는다. Ledger수치감사1/2개를1.325초로소진했고추가학습3batch는아직미사용이다. restricted1factor·조건부등분산iid혁신·선언finite-mask법칙안의계산가능성일뿐실제observer/value-dependentmask/일반공분산/높은novelty를해결한것이아니다.

**다음 한 작업:** [NEXT_DATA_GATE.md](../discovery_v3_math/NEXT_DATA_GATE.md)의별도 `path_source_gate_v1` 준비감사. 아직구현/실행안했다. 새계획·누수시험·검토·동결후원source34명의미래경로/voicing을정답파일로분리해재구성하고,기존연속목표/guard일치와innertrain-onlymask사전의held미지패턴비율을aggregate로확인한다. olddev/test정답은hash도열지않고실제futuremask를입력으로주지않는다. 준비600초/새학습0/추가다운로드0. 준비승인을모델학습으로확대하지않는다. root의마무리중planning.lock실제보유를확인하고해제후선점한다.

## 2026-11-02 현재 선점: discovery_v3_math 수치·gradient 프로토타입

사용자의“바로진행”요청으로 `/root`가새 `discovery_v3_math`를담당한다. 실제 planning.lock을보유하는guardian프로세스와private claim.json이있으며완료전다른실행자는이작업이나동일감사를시작하지않는다. source라벨/원음/olddev/test정답을모두읽지않는합성수치감사다. 좁은shared-factor+채널별whiteGaussian혁신모형에서두표본분산jointCDF와미래mask혼합을적분하고CDFparametergradient를검사한다. 일반Gaussian/임의mask선택을해결했다고주장하지않는다. 코드·오차기준·별도검토·동결후ledger에1개수치감사시도를기록하고한번만실행한다. CPU600초이내,이전동결분기는그대로보존한다.

## 2026-11-02 넓은 탐색 24학습 완료·다음 수치 가설 준비

사용자는예상밖조합을넓게시도하도록새로요청했다. 보정인증분기의STOP을연구전체종료로확대하지않는다. `discovery_v1`과별도 `discovery_v2`의12개씩24학습완료,모두결과·동결·별도검토를보존했다. olddev/test정답은이번탐색에서hash도열지않았다. v1/v2각batch는종료상태로재실행하지않는다.

v2는공통의정확prefix정보+attention/MLP×joint/deltaMDN의2x2다. Source34명3fold에서Markov.542081,MLPjoint.554539,MLPdelta.563987,attentionjoint.568173,attentiondelta.558815. Markov개선screen후보없음. delta는MLP를해치고attention을2fold에서돕는bundled상호작용(+.018806,세fold양수)이있지만성능/기전/novelty증거가아니다. train곡선은60epoch까지감소,일반화격차도있어수렴·최적epoch를확정하지않는다.

**다음 한 작업:** [NEXT_EXPLORATION.md](../discovery_v2/NEXT_EXPLORATION.md)와 [경로분포 선행6개](../discovery_v2/PATH_DISTRIBUTION_REVIEW.md)를읽고별도 `discovery_v3_math`에서source라벨도쓰지않는Gaussian경로→공동표본분산decoder의작은수치·gradient반박시험을명세/구현/검토/동결한다. Cents/dBFS/46frame/ddof1,unknownvoicingmask/guard,두채널joint의존을존중한다. 고전χ²CDFgradient자체는novelty가아니다. 실패하면그가설을버리거나수정하되모든탐색불가능성으로확대하지않는다.

자동루프재개시추가예산은 `../discovery_v2/LOOP_BUDGET.json`을따른다. 이미끝난24회뒤별도학습batch최대3개×12회,각1800초/작업180초,수치감사최대2개×600초,종료2026-11-06 00:00KST. 시작한시도/실패도ledger에먼저기록하고늘리지않는다. 준비가안된후보를자동학습하지않고각batch의가설·반박대조·누수시험·별도검토·계약동결을먼저완성한다. 원자료·가중치·개인별점수공개와olddev/test선택을금지한다. 아래수동학습중/이전전체종료문구는이력이다.

## 2026-11-02 넓은 탐색 첫batch완료·두번째검토중

`discovery_v1`의고정12학습완료(55.141초),별도결과검토통과. Source34명사람fold에서Markov.542081/HGB.581722/orderedTCN.561511/shuffledTCN.560271/MDN.559653. HGB보다좋지만Markov를못넘어유망후보screen없음. 이batch만STOP으로닫고모든숫자를보존한다. 시계열무정보나학습최적성의증명이아니다(첫pilot은곡선/가중치미보존).

사용자의넓은탐색요청범위에서 `/root`는별도 `discovery_v2`를검토·실행한다. 모든모델에동일한정확prefix상태/변동량을주고,summaryMLP/crossattention × joint/relativecontinuousMDN의2x2를같은source3fold에서12학습한다. Source입력/목표외olddev/test정답은열지않고v1결과도보존한다. train곡선과가중치를이번에는보존한다. 기존종료분기는재개하지않으며앱자동화PAUSED인상태로이검토된batch만수동실행한다. 중복학습을시작하지않는다.

## 2026-11-02 새 사용자 요청: 넓은 탐색을 별도 공간에서 시작

사용자는조건부보정종료를연구전체종료로보지않고,예상밖아이디어·조합을많이시험하는탐색을명시적으로요청했다. Codex `/root`는별도 `discovery_v1/`에서원train34명만사용하는탐색을담당한다. 기존STOP분기·원결과·노출된dev/test는보존한다. 새후보의자료구조/학습목표를바꾸는것은이번새요청의범위이며기존동결을재사용·수정하는것이아니다. 첫batch는prefix시계열/순서제거대조/연속감독결합을검증하는4방법×3사람fold의12학습으로제한한다. fold문턱/전처리/prior는매번내부train만사용하며기존dev/test정답은hash도열지않는다. 탐색은새미사용확증이아니며좋은조합이나novelty를보장하지않는다. 실행전합성누수검사·별도검토·새동결후실행한다. 같은감사·학습을중복시작하지않는다.

## 2026-11-02 조건부 보정 가능성 감사 완료·종료

사용자요청의추가감사를완료했고 `correction_feasibility_v1`도**STOP_CORRECTION_CERTIFICATION_BRANCH**로닫았다. [DECISION.md](../correction_feasibility_v1/DECISION.md), [RESULTS.json](../correction_feasibility_v1/RESULTS.json), [PRIOR_AND_MATH.md](../correction_feasibility_v1/PRIOR_AND_MATH.md)가최신이다.7합성시험·별도AI소스검토후코드6/입력19동결,실제감사1회,108개의사후보정계수최적화,새base학습0·test정답열람0. 결과별도재계산검토도통과했다.

전역혼합은한dev사람제외보정에서Markov보다2.6186%낮은손실(.521987)을보였지만사후개발진단이다. 상태별혼합은.527730으로전역보다1.10024%나빴고4/8명만나아고정screen실패. 둘다최대사람손실은Markov보다높았다. cap의평균이득도.147%/.567%에그쳤다. 더강한HB/betting및온라인aggregation선행을확인했으나,현재재사용자료의유효인증·독립확증·실질신규성이없어모든승격관문미통과다. 작은n/모든보정의보편적불가능성을주장하지않는다.

**이 방향의 다음 등록 실험은 없다.** cap/gate/seed/혼합범위를바꾸어다시돌리거나기존캠페인·이번감사를재개하지않는다. 자동화PAUSED유지. 기존연구와부정적결과를보존하고현재후속은종료한다. 아래미완료가능조건검토와선점은모두완료이력이다.

## 2026-11-02 현재 선점: 조건부 보정의 유한 가능성 감사

사용자의 “현재자료규모에서성립·단순혼합대비novelty검증,근거없으면접기” 요청에따라 Codex `/root`가 별도 `correction_feasibility_v1/`의 수학·사후 보정 감사를 담당한다. 기존STOP캠페인·동결자료는읽기전용,새신경망학습·test정답열람·raw다운로드없음. 기존offset3seed만사용하며전역/상태별혼합의dev-oracle및한dev참여자제외보정,미리명시한span cap2개만검사한다. 이는보정계수의사후적합이며새미사용확증이아니다. 기존LTT/Betting과자료의독립성조건도검토해유효성·novelty근거가없으면이방향을닫는다. 상세범위/실패조건은해당PLAN.json을따른다. 다른실행자는중복감사를시작하지않는다.

## 2026-11-02 사후 진단 완료: 새 학습 대신 방법론 가능 조건 검토

Codex `/root`의 원인 분석 단위를 완료했다. [CAUSE_ANALYSIS.md](../violin_diagnosis_v1/CAUSE_ANALYSIS.md), [NEXT_METHOD.md](../violin_diagnosis_v1/NEXT_METHOD.md), [FINAL_AUDIT.json](../violin_diagnosis_v1/FINAL_AUDIT.json)이 최신이다. 새 fitting0·test정답재열람0,9개합성시험통과,저장9모델dev복원오차0. 기존17개데이터소스·실행소스·예측·점수봉인변경없음. 별도AI검토의경계미분/원음대조/prefix상태/clipping보완을반영했고원진단수치변화0이다. 기존캠페인STOP·자동화PAUSED유지.

**근거:** offset train/dev손실0.422824/0.710557(Markov0.533097/0.536024),dev확신오류4.083%가손실40.956%를차지했다. 일반화격차와확신오류집중이지확정된인과원인은아니다. 입력이탈/보정량이큰사람도유용한보정을얻었고,gain민감도와실패의상관은일관되지않아단순거리/음량gate를다음핵심방법으로채택할근거가부족하다.

**다음 한 미완료 작업:** `NEXT_METHOD.md` §5의조건부regret예산이비자명해질수있는가정·의존처리·기존LTT/전역혼합대비차이를수학적으로검토한다. 정확무악화의no-go와작은참여자수의보수적경계부터존중한다. 새방법의높은novelty·성능개선·인증은아직없다. 근거가없으면이후속도종료하며모델을추가로돌리지않는다. 새로운학습은별도버전·규약·독립검토·동결후의별도작업이다. 기존test는이미노출됐고새확증이아니다. 아래사후진단선점은완료이력이다.

## 2026-11-02 새 요청: 사후 원인 분석·후속 방법론 검토

사용자가 기존 시험 결과 보존과 원인 기반 새 방법론 검토를 요청했다. 기존raw_v1은STOP으로 유지한다. Codex `/root`는 별도 `violin_diagnosis_v1/`에서 저장된 가중치의 train/dev forward pass, 봉인된 dev 예측, source/dev 분포·확신도·오류 집중을 분석한다. 추가 fitting/seed/epoch 탐색, 기존 test 정답 재열람, 기존시험 재평가는 하지 않는다. 기존 test 집계는 이미 노출된 참고 증거로만 쓴다. 새 방법 검토는 직접 선행·명확한 차이·반박 조건을 문서화하며 실제 학습은 이번 분석에 포함하지 않는다. 자동 학습 루프는PAUSED를 유지한다.

## 현재: raw 고정12개 완료·STOP (2026-11-02)

앱 자동화 `ai`는 유한 비교 완료로 **PAUSED**다. 이 캠페인에서 더 실행할 등록 작업은 없다.

실행기별도검토·합성57시험·추가동결·12학습·예측봉인·개발비교군선택·시험1회평가·최종감사완료. 학습실패0,worker16.969초. offset은Markov보다주손실20.23%높아개선screen실패했다. `violin_raw_v1/RESULTS.md`, `RESULTS.json`, `FINAL_AUDIT.json`이최신이다. 상태EVALUATED/stopped=true이며아래실행단계선점은종료됐다.

**미완료인등록실험은없다.** 같은캠페인을재개·재동결하거나모델/seed/문턱을더탐색하지않는다. 실제성과·한계를보존하고자동학습루프를종료상태로전환한다. 새로운방법을하려면직접선행과차이·반박조건이있는별도계약이필요하며이미평가한test는미사용확증이아니다.

## 2026-11-02 실행·평가 단계 선점

사용자가 실행기 검토와 고정12개 예측 실험까지 요청했다. Codex `/root`가 새 `violin_raw_v1/runner.py`와 그 테스트·별도 검토·추가 동결·고정 학습/평가를 담당한다. 기존 데이터 동결과 준비 파일 hash4개를 먼저 확인했다. 다른 실행자는 같은 학습/평가를 시작하지 않는다. 모델/지표/학습설정은 동결된 protocol.json을 따른다. 이 선점이 완료되거나 실패/중단 기록이 생기기 전에는 자동화도 중복 실행하지 않는다.

## 2026-11-02 사용자 요청으로 재개

**현재 데이터 준비 단위 완료:** Claude Opus4.7 별도 호출 두 건의 검토 승인,42개 합성 시험, 데이터 코드/규약 동결, 실제2168×60개 입력/분리 정답 생성과 실제8녹음 미래 변조 불변성 검증 완료. `violin_raw_v1/README.md`와 `PREPARED_SUMMARY.json`을 읽는다. 데이터 동결에 포함된 파일은 변경하지 않는다. download/inventory/partition/prepare는 이미 완료했으므로 다시 실행하거나 영수증을 덮어쓰지 않는다.

**다음 한 단위:** 고정12개 학습/예측 봉인/개발 비교군 선택/단일 test 평가의 실행기와 합성 검증을 구현한다. 새 평가 부분은 기존 데이터 준비 승인에 포함되지 않아 별도 코드 검토와 준비 입력/모델/평가 동결을 해야 한다. 모델은 `protocol.json` 설정을 그대로 사용한다. `metrics.py`의 참여자 macro/seed 손실 평균/paired bootstrap을 활용하며 test_labels.npz는 예측과 비교군 봉인 후 한 번 평가할 때만 읽는다. 현재 학습 실행기는 없으며 학습 결과도 없다. 최대12시도/누적1800초/개별600초/2026-11-06 00:00 KST를 유지한다.

사용자의 “루프 재개”가10월31일 중단 요청을 해제했다. 자동화 `ai`를ACTIVE로 변경했다. Codex `/root`가 raw 준비/평가 실행기와 규약, 별도 검토 준비를 담당한다. 기존34/8/9명 분할과 원음/metadata 영수증을 유지하며 기존 종료 캠페인은 재실행하지 않는다. 다음 단계는 source provenance와 코드 검토 → 실제 데이터 준비/실험 계약 동결 관문이다.

## 사용자 요청: PAUSED (2026-10-31)

사용자가 나중에 이어가도록 중단을 요청했다. 자동화 `ai` 일시중지, 해당 연구 Python 실행 없음. 명시적 재개 전 새 작업을 시작하지 않는다. 원음 다운로드/감사, target-blind participant split, 합성 시험36개까지 완료했다. 세부 숫자와 남은 작업은 `EXECUTION_STATUS.md` 최상단을 읽는다. raw 연구의 실제 목표/성능은 아직 미평가이며 이번 추가 파일은 미커밋 상태로 보존한다. 이전 작업 선점은 종료됐다.

## 2026-10-31 현재 작업 선점

후속 사용자 요청: Codex `/root`가 `violin_raw_v1/data.py`, `test_data.py`와 메타데이터 감사 도구를 담당한다. 기존 다운로드 PID40294는 그대로 유지한다. 실제 목표 생성/학습은 하지 않고 participant 분리·train-only 문턱·중복/guard 처리의 합성 검증과 다운로드 완료 후 메타데이터 감사를 진행한다.

Codex `/root`, 시작 15:20 KST: `violin_raw_v1/download.py` 재개 및 원음 구간별 특징 추출기·합성 누수 시험 구현을 담당했다. **특징 추출기와 합성 시험19개 완료**. 다운로드는 실제 프로세스/잠금과 종료 영수증을 확인하고 중복 실행하지 않는다. 모델 학습·실제 목표 추출은 아직 하지 않았다. 다음 미완료 단위는 다운로드 완료/체크섬 → raw 메타데이터·중복 감사 → train-only 문턱·participant split 생성기와 검증이다. 실제 학습은 독립 검토와 새 동결 뒤에만 한다. 새 Claude 데스크톱 프로세스는 관찰됐으나 이 저장소의 실행·작업 선점 기록은 없었다.

## 최신 우선 지시: 수치 보완 완료, raw WAV 예측 준비

아래의 보완 검증 대기 지시는 완료 이력으로 바뀌었다. `projection_audit_v2`를 한 번 실행해239/256 gap 통과,11 clipping 밖 identity,6 실제 실패를 확인했다. **projection 작업은 종료하며 제3 보완이나 재실행을 하지 않는다.**

MAT descriptor는 YIN 양방향 참조 문제로 실제 학습하지 않는다. 다음은 [CLAUDE_NEXT.md](../CLAUDE_NEXT.md)의 원음 다운로드 재개 → 원음/참여자/시간 지지 감사 → 합성 누수 시험 → 직접 선행·독립 검토·새 계약 동결 → 최대12개 raw 바이올린 비교다. raw runner는 아직 없다. `START_HERE.md`의 첼로 v3 실행 예시는 이 새 실험 실행 명령이 아니다.

Claude로 이어받는 안내를 준비했다. 실제 작업자 시작은 확인되지 않았다. 작업 시작자가 이 절 아래에 소유 작업·파일·시작 시각·상태를 기록하고 실제 프로세스/잠금과 대조한다. Codex 자동화와 Claude가 같은 다운로드·파일·학습을 동시에 실행하지 않는다.

## 현재 상태: v3 완료, 다음 검증을 별도로 준비

`pilot_20261030_v1`, `development480_20261030_v2`, `forecast_20261030_v3`는 모두 종료했다. v3는 고정 비교 12개 완료, 실행 검증 오류 0건이며 STOP을 유지한다. 공동 신경망은 보정 HGB보다 log loss가 3.335% 낮지만 독립 신경망보다 0.185% 높다. 주 선별 조건 통과와 방법론의 추가 이득 실패를 함께 기록한다.

먼저 실제 프로세스·상태·Git 변경을 확인한다. 같은 자료·규약·소스를 재동결하거나 완료된 캠페인을 재개하지 않는다. 현재 실행 안내는 `START_HERE.md`, 최신 결과는 `EXECUTION_STATUS.md`와 `../published_results/forecast_20261030_v3/`다. 아래 과거 계획의 미실행 후보 목록을 지금의 추가 실행 지시로 해석하지 않는다.

## 첫 후속 감사 완료

`projection_audit_v1`의 256건 정답 비열람 수치 감사를 완료했다. 17개 합성 테스트 후 동결·1회 실행했고, FW gap ≤ 1e-8은 50건, 모든 가능한 출력의 점수 비악화까지 통과한 보정은 28건이었다. SLSQP success만으로 충분한 정밀도를 보장할 수 없다는 한계를 확인했다. 기존 모델 성능은 재계산하지 않았다. 이 버전을 다시 실행하거나 기준을 완화하지 않는다.

다음 수치 작업은 **별도 v2의 한 번뿐인 보완 검증**으로 제한한다. 합성 자료에서 KKT/FW 잔차에 직접 반응하는 보완 계산을 검토하고 독립 검토·새 동결 뒤 같은 64개 ID와 같은 오차 기준으로 한 번 평가한다. 총 600초 이내, 관측 정답 비열람을 유지한다. 추가 보완으로도 기준을 못 넘으면 한계를 기록하고 수치 solver 탐색을 종료한다. 기존 12개 학습 실험은 계속 STOP이다.

## 다음 작업 순서

1. **첫 수치 감사의 한계를 반영한다.** `../projection_audit_v1/RESULTS.md`, 독립 결과 검토 `../published_results/forecast_20261030_v3/FORECAST_REVIEW.md`, `NEXT_METHOD_REVIEW.md`를 읽는다. 앞 절의 별도 v2 보완 검증 한 번만 허용한다. 결과 라벨 비열람·ID 해시 선택·같은 허용오차를 유지하며 기존 frozen 파일은 변경하지 않는다. 전부 충분히 수렴했다는 증거가 없으면 고전 정확 보정을 실제 비교 완료라고 쓰지 않는다.
2. **바이올린 예측 규약을 준비한다.** `../forecast_v3/violin_data_audit/README.md`의 실제 감사가 기존 `VIOLIN_DATA_GATE.md`의 미다운로드 상태보다 최신이다. 72개 참여자 기록과 2,117개 고유 시퀀스에서 내용 중복 제거, 참여자 전체 단위 분할, prefix-only 특징, 훈련 자료만의 문턱 결정을 검사한다. 저장된 pitch 좌표와 frame 단위를 유지한다. timestamp·YIN 단위·동작 정렬 문제를 해결했다고 가정하지 않는다. 미래 창의 원래 연속 지표를 보존해야 임계값 사이의 분포를 학습할 수 있다.
3. **직접 선행과 필수 비교군을 검토한다.** 연속 임계값에 답하는 조건부 CDF·혼합모형이나 확률 일관성 자체는 새로운 기술이 아니다. 임의 임계값 예측이 실제로 필요한 사용 사례와 기존 간단한 혼합모형이 해결하지 못하는 문제를 명시한다. 근거가 없으면 새로운 방법론 주장과 큰 모델 탐색을 중단한다.
4. **관문 통과 후에만 별도 소규모 실험을 동결한다.** 입력·누수 검사, participant split, 목표·지표·비교군·실패 기준, 독립 검토, 코드·입력 해시를 먼저 고정한다. 후속은 최대 12개 CPU 실행, worker 누적 1,800초, 단일 실행 600초, 2026-11-06 00:00 KST 이내다. 검증용 자료를 반복 선택에 쓰지 않고 test는 고정 후 한 번 평가한다. 자료 타당성이나 필수 기준선 검증이 실패하면 근거와 함께 종료한다.

Claude CLI 연동은 실제 검증했으나 이후 HTTP429가 발생했다. 사용자 요청 후 재확인도 같은 한도 오류였다. 실패 기록을 보존하고 Codex/Codex로 전환해 v3를 완료했다. 이후 Claude를 쓸 때는 새 한도 상태를 확인하되 추가 결제·권한 우회는 하지 않는다. 실행기가 제공자를 몰래 바꾸게 하지 않는다.

사용자는 코드·문서·허가된 준비 데이터를 GitHub의 `research/agentic-forecast-20261030` 브랜치에서 다른 AI와 공유하도록 요청했다. 원자료·계정정보·원시 CLI 로그는 공개하지 않고 main을 임의 병합하지 않는다.

한 번의 자동 깨우기는 위 미완료 작업 중 의미 있는 한 단위만 수행한다. 가설·반박 조건·입력/코드 해시·결과·결정·다음 한 작업을 기록한다. 완료·예산 소진·종료일·정당한 다음 실험 부재 시 자동화를 일시중지한다. 새로운 자료와 지표를 계속 바꾸어 종료를 미루지 않는다.

---

## 아래 내용은 종료된 단계의 계획 이력이다

현재 실행 지시가 아니다. 추가로 실행하지 않은 옛 후보도 포함되어 있으며 실제 실행 범위는 봉인 결과를 따른다.

## 0. 이미 닫힌 파일럿

`pilot_20261030_v1`: 6회 실행, 인증 오류 0. 강한 비학습 기준선 12.0625개, HGB12.5625개, 정확 공동 하한11.125개. 최대 가능 개선7.77%로 원래10% 목표 불가능. STOP 유지. 잠금·소스·평가 기준을 덮어쓰지 않는다.

## 1. 전체 개발 480열의 진단과 제한된 비교

현재 실행 대상: `engine_v2`, 캠페인 `development480_20261030_v2`.

```bash
cd "$HOME/violin-bowing-boundaries"
bash agentic_research/scripts/research_tick.sh
```

- 한 tick은 초기5개 기준선을 처리하고 정확 공동 하한을 봉인한 뒤, 통과하는 경우에만 제안→독립 검토→실험 한 건을 진행한다.
- headroom<10%이면 자동 STOP. 학습 후보를 더 돌리지 않고 아래의 별도 연구 준비로 넘어간다.
- 통과하면 등록된5가지 학습 비교×시드0/1/2를 동일 조건에서 처리한다. 후보 메뉴는 평가 전에 고정되어 있으며 LLM은 실행 순서만 고른다. 새 모델·유리한 시드·하위집단을 추가하지 않는다.
- 최대24시도, worker2700초, 작업600초, LLM18라운드, 2026-11-06 00:00 KST 중 먼저 도달한 제한을 따른다.
- 제안 거절·CLI 실패·중단도 이력으로 남긴다. 인증 오류·무결성 실패는 해당 캠페인을 종료하고 원인과 재현 사례를 기록한다. 자동 재동결은 금지다.
- 종료 후 전체480열, 기존16열을 제외한464열(보조 설명), 각 세션, 비용, 불리한 결과를 정리한다. 464열을 새 확인시험이라고 부르지 않는다. 시드 전체가 끝나지 않았으면 평균 비교를 완료했다고 쓰지 않는다.

## 2. 목적을 바꾸는 경우는 별도 후속 연구로 명시한다

위 정확 질의 과제가 불가능하거나 개선에 실패하면 무한 최적화를 하지 않는다. `AI_DECISION_GATES.md`의 추천 대안을 **한 번만** 구체화한다. 목표는 처음0.20초로 이후0.25–0.45초의 관측 규칙 출력을 예측하는 학습이다. 기존 목표가 성공한 것처럼 쓰지 않는다.

반복 실행의 다음 작업은 우선 데이터·규칙·시간축을 검증하고 별도 `forecast_v3/PROTOCOL.md`와 입력 manifest를 만드는 것이다. 기존 `timeres/features/` 17개 압축 CSV와 `astra_20261018/forecast/` 코드를 읽되 수정하지 않는다. 기존78,000개에 해당하는 계열과 추가34,400개의 조건 차이를 명시한다.

필수 구현 전 검토:

1. future observer와 joint-rule observer의 차이를 fixture로 확인한다. 9개 규칙에서 가능한10개 패턴이라는 명제를 완전탐색으로 확인한다. 성립하지 않으면 억지로10-class head를 쓰지 않는다.
2. prefix 특징의 계산 끝이0.20초 이내인지, plateau landmark가 미래를 이용했는지 점검한다. 미래 특징·미래 힘·전체 길이·미래 결측 패턴·target session ID는 입력 금지다. causal landmark 검증이 없으면 실시간 예측이라고 쓰지 않는다.
3. 실제 현/설치 계열 분할을 명시한다. A/B 개발학습 내부에서만 전처리·튜닝하고 C/D는 회고적 개발 평가임을 유지한다. 겹치는 창을 무작위로 섞지 않는다.
4. persistence, source-fitted Markov, logistic, controls/snapshot/history HGB, 작은 동일 backbone의 독립 head·공동 head·공동 head+경계손실을 비교하도록 계획한다. 모든 최신 대형 모델을 무조건 실행하지 않는다.
5. future log loss를 주 지표로 하고 Brier·전이 부분집합·최악 계열·경계오차·논리적 일관성·계산비를 보조 지표로 고정한다. 효과 기준은 source 내부 비교와 실제 용도에서 정하고 C/D를 본 뒤 바꾸지 않는다.
6. 모델 후보와 제거 실험, 출처와 가까운 선행, 결과를 기각할 조건을 별도 리뷰한다. 통계·데이터 리뷰가 통과한 뒤 테스트·동결한다. 새 캠페인은 최대12개 작업, worker1800초, 모델군당시드0/1/2, 전체 자동화 종료일2026-11-06을 넘지 않는다.

이 단계의 당장 허용된 산출물은 입력 감사·누수 테스트·규약·최소 baseline이다. 학습 비교를 수행하려면 위 체크리스트의 기계 검증과 독립 검토 기록을 먼저 완성한다. 미흡하면 BLOCKED_WITH_EVIDENCE를 기록하고 해당 방향을 종료한다. 실패해도 다른 데이터셋과 지표를 계속 만들어내지 않는다.

## 3. 바이올린 적용 가능성 확인

현재 주 자료는 첼로 현이다. 원래 과제의 바이올린 관련성 조건이 해결된 것처럼 쓰지 않는다. `VIOLIN_DATA_GATE.md`의 1차 출처를 확인해 공개 바이올린 자료의 센서 종류·라벨·시간 정보·라이선스가 선택한 AI 목표와 실제로 맞는지 판단한다. 힘/접촉위치가 없는 음원으로 활 힘 경계를 검증하지 않는다. 대용량 다운로드 전에 파일·용량·사용권·디스크 여유를 확인하고 최초 다운로드는 합계2GiB 이내로 제한한다. 적합한 자료가 없으면 그 공백을 보고한다.

## 4. 작업 단위와 종료

한 번의 자동 깨우기에는 위의 미완료 항목 중 의미 있는 한 단위만 수행한다. 먼저 현재 프로세스·worker/planner 잠금·status를 확인한다. 긴 학습이 이미 돌고 있으면 두 번째를 시작하지 않는다. 실제 실험은 고정 worker가, 정확성 판단은 테스트/검사기가 한다. LLM의 글은 측정 결과가 아니다.

각 작업은 가설, 반박 조건, 코드/입력/규약 해시, 실행 결과, 결정, 다음 한 작업을 남긴다. 실패·원점수·로그는 지우지 않는다. 새 작업은 위 세 단계 범위에만 둔다. 모든 단계 완료, 종료일, 예산 소진, 데이터 불일치 또는 정당한 다음 실험이 없으면 반복 자동화를 일시중지하고 최종 결과·미해결 사항을 짧게 알린다.
