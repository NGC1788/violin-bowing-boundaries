# Donor-deletion learning v1 — AI future forecasting

**준비/전체동결완료(2026-11-03):** 28합성기능시험·별도코드검토뒤7source입력을검증했고27소스/환경/계약을동결했다. 실제새학습0건이며다음은고정12fit을한번실행하는단위다. `PREPARATION_RESULT.json`, `FREEZE.json`, `FREEZE_REVIEW.json`을읽는다. 코드/준비를다시하거나이전STOP실험을재개하지않는다.

다른 제공자의 사례를 검색하는 encoder를 학습하며, 한 donor 제공자의 기억 전체를 지운 최악9query손실을 학습 목적에 넣는 가설이다. Weighted retrieval/DRO/공동pattern확률/평활화는 기존원리이며 높은novelty나성능개선은 아직입증되지않았다. 근거는 `../method_redesign_v1/`에 있다.

## 고정 비교와 경계

네retrieval학습목적 ERM/randomdonordelete/adversarialdonordelete/querygroupDRO×기존source3personfold=12새fit을후속실행한다. 이번준비단위실제fit은0이다. 원train34명1442개/invalid15개를유지하며olddev/test정답은hash도열지않는다. 원음/추가모델가중치다운로드0,추가standalone수치감사0이다.

- 같은73derivedprefix→Linear32/GELU/Linear32/GELU encoder와2regime generator를쓴다. Generator는이전과같은mask/projectedcontrast auxiliary에만쓰고주예측에Gaussian표본분산법칙을사용하지않는다. Epoch40,batch64,seed0,AdamW1e-3/decay1e-3,aux.1,CPUfloat64를유지한다. 공유encoder/generator초기값·전처리·minibatch순서를과거영수증과검증한다.
- Retrievalscore는normalizedembeddingcosine/.25이며donor별recordcount로나눈exp-score를쓴다. Query와같은사람의모든기억은trainforward에서0가중치다. 실제held추론query는group=-1로주고innertrain기억만사용한다. Uniform10patternpriorρ=.01은모든arm에같으며eventprobability가이론적으로[.001,.999]다. Raw support소실은별도집계한다.
- Donor내softmax와donor간잔여항softmax를분리해삭제를안정계산한다. 차감식q−A/1−W는실제계산에쓰지않는다. Null먼저,donorcode오름차순이다. Adversary는동일donor의9query평균loss뒤max를취한다. Random은null+허용donor에서균등추출하고별도seedstream을쓴다.
- Query-groupDRO는eta.01의expweight를사용하며shuffled-minibatch importance-scaled risk추정으로업데이트한다. 이는adaptiveiterate에서조건부unbiased한oracle이라는보장이없다. Commonaux는DRO재가중하지않고personmacro로유지한다.
- 추론에는최종innertrainprefix/memorylabel/groupcode/원행대응이필요하다. Memory는private artifact다. Parametrichead와다른inference storage/resource임을공개한다. 네retrievalarm에는같은all-deletionforward를제공하지만목적식마다backward는달라동일FLOP이라고하지않는다. 실제runtime은아직미검증이다.

## 자료와 실행 규약

먼저합성기능시험·별도소스검토를하고그승인뒤에만7개허용source입력을검증/동결한다. 기존prefix/meta/future3개,HGBsourcefold예측3개,기존exactbatch의sourceall_predictions1개다. Reference는기존Markov/HGB/joint_aux/mdn_aux를재사용하며새referencefit0이다. 후자는새실험에서재학습한matchedparametriccontrol이아니다. 공통입력/aux/전처리/초기값의대응은검증하지만추론memory와head구조/330parameter차이는남는다. 특별한학습연산의비교는새matchedretrievalarm을중심으로한다.

실제학습은전체FREEZE후**다음실행단위**에서만한다. Parent가12개job의예측/모든donor반사실예측/가중치/trace/stats/기억/영수증을고정하고OUTPUT_SEAL뒤held사건정답을평가한다. 최악helddonor는봉인뒤보조risk집계에만쓰고발행예측이나모델선택에는쓰지않는다. Same-userfilesystem/hashguard는OS보안sandbox가아니다.

주점수fullmemory사람macro9querylogloss,Markov/HGB대비2%/최소2fold조건을유지한다. Adversarialoperator판정에는ERM/random대비전체와2fold개선도필요하다. QueryDRO비교는별도보고한다. Deletionrisk는새retrieval4arm만의보조지표다. 모든fold/실패/worstperson/Brier/사건빈도·loss/invalid대validallzero/support/entropy·effectivedonors/memorybytes·inference시간을보존한다.

## 다음 실행에서 사용할 명령

```bash
.venv-mac/bin/python -m agentic_research.donor_deletion_learning_v1.runner check
# FREEZE와state/lock을확인한후다음실행단위에서한번만:
.venv-mac/bin/python -m agentic_research.donor_deletion_learning_v1.runner run
```

State가이미있으면재실행하지않고기존worker/완료/실패만검토한다. Atomicledger에시작/실패를먼저소비하고첫실패STOP,job180초/batch1800초/12job을지킨다. 같은batch를구조하거나설정/seed를바꿔재동결하지않는다. Source는이미노출된탐색으로독립확증이아니다.

합성기능시험명령:

```bash
.venv-mac/bin/python -m pytest --import-mode=importlib agentic_research/donor_deletion_learning_v1/test_runner.py -q
```

코드/집계/해시만공개하며local/의실제memory/개인별자료/예측/가중치/trace/원시CLI로그는공개하지않는다.
